from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import smplx
import torch
import trimesh
from scipy.optimize import least_squares
from yourdfpy import URDF

for _name, _value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)


FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]
MANO_FINGERTIP_INDICES = np.array([4, 8, 12, 16, 20], dtype=np.int64)

# smplx MANO returns 16 regressed joints by default. These are the MANO tip
# vertices used by smplx for its MANO extra joints, appended here to recreate
# the 21-joint order used by OmniDexGrasp/manotorch.
MANO_TIP_VERTEX_IDS = np.array([744, 320, 443, 554, 671], dtype=np.int64)

SHADOW_TIP_LINKS = [
    "robot0:thdistal",
    "robot0:ffdistal",
    "robot0:mfdistal",
    "robot0:rfdistal",
    "robot0:lfdistal",
]


@dataclass
class ValidationResult:
    action: np.ndarray
    target_tips: np.ndarray
    shadow_tips: np.ndarray
    error_m: np.ndarray
    output_html: Path


@dataclass
class ManoGeometry:
    joints21: np.ndarray
    vertices: np.ndarray
    faces: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate MANO-to-Shadow retargeting with OmniDexGrasp-compatible fingertip points."
    )
    parser.add_argument("--mano-root", default="models", help="Root containing mano/MANO_RIGHT.pkl.")
    parser.add_argument(
        "--shadow-urdf",
        default="external/OmniDexGrasp/assets/robo/shadow_hand/shadowhand.urdf",
        help="Shadow Hand URDF, preferably OmniDexGrasp's shadowhand.urdf.",
    )
    parser.add_argument("--output-dir", default="outputs/shadow_phase6_validation")
    parser.add_argument("--pose", choices=["zero", "random"], default="zero")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-nfev", type=int, default=250)
    parser.add_argument("--pose-scale", type=float, default=0.45)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument(
        "--relax-little-limits",
        type=float,
        default=0.0,
        help="Diagnostic only: expand LFJ joint bounds by this many radians during optimization.",
    )
    parser.add_argument(
        "--shadow-point",
        choices=["distal_mesh_tip", "distal_origin"],
        default="distal_mesh_tip",
        help="Shadow point to align. distal_mesh_tip uses the end of the last link; distal_origin matches OmniDexGrasp.",
    )
    return parser.parse_args()


def make_mano_21_joints(model: smplx.MANO, output: smplx.utils.MANOOutput) -> np.ndarray:
    joints16 = output.joints.detach().cpu().numpy()[0]
    vertices = output.vertices.detach().cpu().numpy()[0]
    tips = vertices[MANO_TIP_VERTEX_IDS]

    joints21 = np.empty((21, 3), dtype=np.float64)
    joints21[0] = joints16[0]
    joints21[1:4] = joints16[13:16]  # thumb chain
    joints21[4] = tips[0]
    joints21[5:8] = joints16[1:4]  # index chain
    joints21[8] = tips[1]
    joints21[9:12] = joints16[4:7]  # middle chain
    joints21[12] = tips[2]
    joints21[13:16] = joints16[10:13]  # ring chain
    joints21[16] = tips[3]
    joints21[17:20] = joints16[7:10]  # little chain
    joints21[20] = tips[4]
    return joints21


def create_mano_geometry(mano_root: str, pose: str, seed: int, pose_scale: float) -> ManoGeometry:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = smplx.create(
        model_path=mano_root,
        model_type="mano",
        is_rhand=True,
        use_pca=False,
        flat_hand_mean=True,
        batch_size=1,
    )

    betas = torch.zeros(1, 10)
    global_orient = torch.zeros(1, 3)
    if pose == "zero":
        hand_pose = torch.zeros(1, 45)
    else:
        hand_pose = torch.randn(1, 45, generator=generator) * pose_scale
        # Keep the first validation target plausible by biasing flexion positive.
        hand_pose[:, 2::3] = hand_pose[:, 2::3].abs()
    transl = torch.zeros(1, 3)

    output = model(
        betas=betas,
        global_orient=global_orient,
        hand_pose=hand_pose,
        transl=transl,
        return_verts=True,
        return_full_pose=True,
    )
    joints21 = make_mano_21_joints(model, output)
    return ManoGeometry(
        joints21=joints21,
        vertices=output.vertices.detach().cpu().numpy()[0].astype(np.float64),
        faces=np.asarray(model.faces, dtype=np.int64),
    )


def distal_mesh_tip_offsets(robot: URDF, urdf_path: str) -> dict[str, np.ndarray]:
    mesh_dir = Path(urdf_path).parent
    offsets: dict[str, np.ndarray] = {}
    for link_name in SHADOW_TIP_LINKS:
        link = robot.link_map[link_name]
        candidates = []
        for visual in link.visuals:
            mesh = visual.geometry.mesh
            if mesh is None:
                continue
            mesh_path = mesh_dir / Path(mesh.filename).name
            loaded = trimesh.load_mesh(str(mesh_path), process=False)
            vertices = np.asarray(loaded.vertices, dtype=np.float64)
            if mesh.scale is not None:
                vertices = vertices * np.asarray(mesh.scale, dtype=np.float64)
            vertices_h = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)
            vertices_link = (visual.origin @ vertices_h.T).T[:, :3]
            candidates.append(vertices_link)
        if not candidates:
            offsets[link_name] = np.zeros(3, dtype=np.float64)
            continue
        all_vertices = np.concatenate(candidates, axis=0)
        offsets[link_name] = all_vertices[np.argmax(np.linalg.norm(all_vertices, axis=1))]
    return offsets


def shadow_points(robot: URDF, q: np.ndarray | None = None, tip_offsets: dict[str, np.ndarray] | None = None) -> np.ndarray:
    names = robot.actuated_joint_names
    if q is None:
        q = np.zeros(len(names), dtype=np.float64)
    robot.update_cfg(dict(zip(names, q)))
    tips = []
    for link in SHADOW_TIP_LINKS:
        transform = robot.get_transform(link)
        if tip_offsets is None:
            tips.append(transform[:3, 3].copy())
        else:
            tips.append(transform[:3, :3] @ tip_offsets[link] + transform[:3, 3])
    return np.stack(tips, axis=0)


def joint_bounds(robot: URDF, relax_little_limits: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    lower = []
    upper = []
    for name in robot.actuated_joint_names:
        joint = robot.joint_map[name]
        limit = joint.limit
        lo = float(limit.lower) if limit is not None and limit.lower is not None else -np.pi
        hi = float(limit.upper) if limit is not None and limit.upper is not None else np.pi
        if relax_little_limits > 0.0 and "LFJ" in name:
            lo -= relax_little_limits
            hi += relax_little_limits
        lower.append(lo)
        upper.append(hi)
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def normalize(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        return value
    return value / norm


def build_hand_frame(wrist: np.ndarray, index: np.ndarray, middle: np.ndarray, little: np.ndarray) -> np.ndarray:
    forward = normalize(middle - wrist)
    lateral = normalize(index - little)
    normal = normalize(np.cross(lateral, forward))
    if np.linalg.norm(normal) < 1e-8:
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    lateral = normalize(np.cross(forward, normal))
    return np.stack([lateral, forward, normal], axis=1)


def mano_to_shadow_transform(mano_joints21: np.ndarray, shadow_open_tips: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    mano_wrist = mano_joints21[0]

    mano_frame = build_hand_frame(
        wrist=mano_wrist,
        index=mano_joints21[5],
        middle=mano_joints21[9],
        little=mano_joints21[17],
    )
    shadow_frame = build_hand_frame(
        wrist=np.zeros(3),
        index=shadow_open_tips[1],
        middle=shadow_open_tips[2],
        little=shadow_open_tips[4],
    )

    mano_middle = mano_joints21[12] - mano_wrist
    scale = np.linalg.norm(shadow_open_tips[2]) / max(np.linalg.norm(mano_middle), 1e-8)
    rotation = mano_frame @ shadow_frame.T
    return mano_wrist, rotation, scale


def map_mano_points_to_shadow_frame(points: np.ndarray, mano_wrist: np.ndarray, rotation: np.ndarray, scale: float) -> np.ndarray:
    return ((points - mano_wrist) @ rotation) * scale


def map_mano_tips_to_shadow_frame(mano_joints21: np.ndarray, shadow_open_tips: np.ndarray) -> np.ndarray:
    mano_wrist, rotation, scale = mano_to_shadow_transform(mano_joints21, shadow_open_tips)
    mano_tips = mano_joints21[MANO_FINGERTIP_INDICES]
    return map_mano_points_to_shadow_frame(mano_tips, mano_wrist, rotation, scale)


def shadow_mesh(robot: URDF, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    robot.update_cfg(dict(zip(robot.actuated_joint_names, q)))
    mesh = robot.scene.to_geometry()
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)
    scale = np.linalg.norm(shadow_open_tips[2]) / max(np.linalg.norm(mano_rel[2]), 1e-8)
    return (mano_local * scale) @ shadow_frame.T


def optimize_shadow(
    robot: URDF,
    target_tips: np.ndarray,
    max_nfev: int,
    restarts: int,
    seed: int,
    tip_offsets: dict[str, np.ndarray] | None,
    relax_little_limits: float,
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = joint_bounds(robot, relax_little_limits)
    q0 = np.clip(np.zeros(len(robot.actuated_joint_names), dtype=np.float64), lower, upper)
    rng = np.random.default_rng(seed)
    starts = [q0]
    center = 0.5 * (lower + upper)
    starts.append(np.clip(center, lower, upper))
    for _ in range(max(restarts - len(starts), 0)):
        starts.append(rng.uniform(lower, upper))

    def residual(q: np.ndarray) -> np.ndarray:
        tips = shadow_points(robot, q, tip_offsets)
        fingertip_residual = (tips - target_tips).reshape(-1) * 100.0
        regularizer = 0.002 * q
        return np.concatenate([fingertip_residual, regularizer])

    best_result = None
    for start in starts:
        result = least_squares(
            residual,
            start,
            bounds=(lower, upper),
            max_nfev=max_nfev,
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8,
            x_scale="jac",
            diff_step=1e-4,
        )
        if best_result is None or result.cost < best_result.cost:
            best_result = result
    q = best_result.x
    return q, shadow_points(robot, q, tip_offsets)


def write_visualization(
    output_path: Path,
    target_tips: np.ndarray,
    shadow_tips: np.ndarray,
    mano_joints21: np.ndarray,
    mano_vertices: np.ndarray,
    mano_faces: np.ndarray,
    shadow_vertices: np.ndarray,
    shadow_faces: np.ndarray,
) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=mano_vertices[:, 0],
            y=mano_vertices[:, 1],
            z=mano_vertices[:, 2],
            i=mano_faces[:, 0],
            j=mano_faces[:, 1],
            k=mano_faces[:, 2],
            name="MANO mesh mapped to Shadow frame",
            color="rgba(40, 180, 80, 0.28)",
            flatshading=False,
            opacity=0.28,
        )
    )
    fig.add_trace(
        go.Mesh3d(
            x=shadow_vertices[:, 0],
            y=shadow_vertices[:, 1],
            z=shadow_vertices[:, 2],
            i=shadow_faces[:, 0],
            j=shadow_faces[:, 1],
            k=shadow_faces[:, 2],
            name="Shadow mesh optimized",
            color="rgba(40, 100, 230, 0.42)",
            flatshading=False,
            opacity=0.42,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=mano_joints21[:, 0],
            y=mano_joints21[:, 1],
            z=mano_joints21[:, 2],
            mode="markers",
            name="MANO 21 joints",
            marker=dict(size=3, color="rgba(80,80,80,0.45)"),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=target_tips[:, 0],
            y=target_tips[:, 1],
            z=target_tips[:, 2],
            mode="markers+text",
            name="Target MANO joints [4,8,12,16,20]",
            text=FINGER_NAMES,
            marker=dict(size=7, color="red"),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=shadow_tips[:, 0],
            y=shadow_tips[:, 1],
            z=shadow_tips[:, 2],
            mode="markers+text",
            name="Shadow distal link origins",
            text=FINGER_NAMES,
            marker=dict(size=7, color="blue"),
        )
    )
    for i, name in enumerate(FINGER_NAMES):
        pair = np.stack([target_tips[i], shadow_tips[i]], axis=0)
        fig.add_trace(
            go.Scatter3d(
                x=pair[:, 0],
                y=pair[:, 1],
                z=pair[:, 2],
                mode="lines",
                name=f"{name} error",
                line=dict(color="rgba(30,30,30,0.45)", width=4),
                showlegend=False,
            )
        )

    fig.update_layout(
        title="MANO-to-Shadow Phase6 Validation",
        scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(output_path, include_plotlyjs="cdn")


def validate(args: argparse.Namespace) -> ValidationResult:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mano = create_mano_geometry(args.mano_root, args.pose, args.seed, args.pose_scale)
    robot = URDF.load(args.shadow_urdf)
    tip_offsets = None
    if args.shadow_point == "distal_mesh_tip":
        tip_offsets = distal_mesh_tip_offsets(robot, args.shadow_urdf)
    shadow_open = shadow_points(robot, tip_offsets=tip_offsets)

    mano_wrist, mano_rotation, mano_scale = mano_to_shadow_transform(mano.joints21, shadow_open)
    mano_joints_shadow = map_mano_points_to_shadow_frame(mano.joints21, mano_wrist, mano_rotation, mano_scale)
    mano_vertices_shadow = map_mano_points_to_shadow_frame(mano.vertices, mano_wrist, mano_rotation, mano_scale)
    target_tips = mano_joints_shadow[MANO_FINGERTIP_INDICES]

    action, shadow_tips = optimize_shadow(
        robot,
        target_tips,
        args.max_nfev,
        args.restarts,
        args.seed,
        tip_offsets,
        args.relax_little_limits,
    )
    error_m = np.linalg.norm(shadow_tips - target_tips, axis=1)
    shadow_vertices, shadow_faces = shadow_mesh(robot, action)

    output_html = output_dir / f"{args.pose}_shadow_retarget_overlay.html"
    write_visualization(
        output_html,
        target_tips,
        shadow_tips,
        mano_joints_shadow,
        mano_vertices_shadow,
        mano.faces,
        shadow_vertices,
        shadow_faces,
    )

    summary = {
        "pose": args.pose,
        "point_definition": {
            "mano": "21-joint MANO order; fingertips are joints [4,8,12,16,20], matching OmniDexGrasp.",
            "shadow": (
                "distal link frame origins for robot0:{th,ff,mf,rf,lf}distal, matching OmniDexGrasp."
                if args.shadow_point == "distal_origin"
                else "distal mesh endpoint on robot0:{th,ff,mf,rf,lf}distal, i.e. the end of the last link."
            ),
        },
        "shadow_point": args.shadow_point,
        "relax_little_limits": args.relax_little_limits,
        "finger_names": FINGER_NAMES,
        "mano_fingertip_indices": MANO_FINGERTIP_INDICES.tolist(),
        "shadow_tip_links": SHADOW_TIP_LINKS,
        "joint_names": robot.actuated_joint_names,
        "action": action.tolist(),
        "error_m": error_m.tolist(),
        "mean_error_m": float(error_m.mean()),
        "max_error_m": float(error_m.max()),
        "output_html": str(output_html),
    }
    (output_dir / f"{args.pose}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ValidationResult(action, target_tips, shadow_tips, error_m, output_html)


def main() -> None:
    args = parse_args()
    result = validate(args)
    print(f"Point definition: MANO joints [4,8,12,16,20] -> Shadow {args.shadow_point}")
    for name, err in zip(FINGER_NAMES, result.error_m):
        print(f"{name:>6}: {err * 1000.0:8.3f} mm")
    print(f"  mean: {result.error_m.mean() * 1000.0:8.3f} mm")
    print(f"   max: {result.error_m.max() * 1000.0:8.3f} mm")
    print(f"visualization: {result.output_html}")


if __name__ == "__main__":
    main()
