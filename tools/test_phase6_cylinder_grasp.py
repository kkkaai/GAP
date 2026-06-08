from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prosthetic_grasp.geometry import (
    RetargetOptimizationConfig,
    RobotSurfaceSamples,
    RobotSurfaceTopology,
    assign_patches_to_robot_surface,
    action_summary,
    build_contact_patch_targets,
    build_hand_frame,
    canonicalize_patch_targets,
    extract_mano_object_contact_clusters,
    load_robot_surface_model,
    map_canonical_mano_patches_to_robot_frame,
    mano_pose_to_shadow_action,
    materialize_assigned_robot_contacts,
    optimize_retarget_action,
    rank_candidates_by_strict_force_closure,
    robot_tip_from_link,
)


MANO_TIP_VERTEX_IDS = np.array([744, 320, 443, 554, 671], dtype=np.int64)
MANO_FINGERTIP_INDICES = np.array([4, 8, 12, 16, 20], dtype=np.int64)


@dataclass(frozen=True)
class ManoGeometry:
    vertices: np.ndarray
    faces: np.ndarray
    keypoints: np.ndarray
    hand_pose: np.ndarray


@dataclass(frozen=True)
class CylinderGeometry:
    vertices: np.ndarray
    faces: np.ndarray
    center: np.ndarray
    axis: np.ndarray
    radius: float
    height: float
    axis_name: str


@dataclass(frozen=True)
class SceneToRobotTransform:
    wrist: np.ndarray
    scene_frame: np.ndarray
    robot_frame: np.ndarray
    scale: float

    def points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return ((points - self.wrist) @ self.scene_frame * self.scale) @ self.robot_frame.T

    def normals(self, normals: np.ndarray) -> np.ndarray:
        normals = np.asarray(normals, dtype=np.float64)
        mapped = (normals @ self.scene_frame) @ self.robot_frame.T
        return normalize_rows(mapped)


@dataclass(frozen=True)
class SimilarityTransform:
    source_center: np.ndarray
    target_center: np.ndarray
    rotation: np.ndarray
    scale: float

    def points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return ((points - self.source_center) * self.scale) @ self.rotation + self.target_center

    def normals(self, normals: np.ndarray) -> np.ndarray:
        normals = np.asarray(normals, dtype=np.float64)
        return normalize_rows(normals @ self.rotation)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a realistic synthetic MANO-cylinder grasp scene and validate "
            "the contact-centric Phase6 retargeting steps."
        )
    )
    parser.add_argument("--mano-root", default="models")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--pose-scale", type=float, default=0.40)
    parser.add_argument("--num-mano-samples", type=int, default=3000)
    parser.add_argument("--num-robot-samples", type=int, default=2000)
    parser.add_argument("--contact-threshold", type=float, default=0.010)
    parser.add_argument("--cluster-radius", type=float, default=0.018)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--assignment-top-k", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1e-4)
    parser.add_argument("--robot-model-dir", default="hand/shadow_hand")
    parser.add_argument("--robot-format", default="urdf", choices=["mjcf", "xml", "urdf", "auto"])
    parser.add_argument("--wrist-link", default="robot0:palm")
    parser.add_argument("--robot-index-tip-link", default="robot0:ffdistal")
    parser.add_argument("--robot-middle-tip-link", default="robot0:mfdistal")
    parser.add_argument("--robot-little-tip-link", default="robot0:lfdistal")
    parser.add_argument("--maxiter-stage1", type=int, default=80)
    parser.add_argument("--maxiter-stage2", type=int, default=80)
    parser.add_argument("--fc-weight", type=float, default=0.02)
    parser.add_argument("--friction-coef", type=float, default=0.7)
    parser.add_argument(
        "--disable-mano-shadow-init",
        action="store_true",
        help="Use the zero Shadow pose instead of the OmniDexGrasp-style MANO-to-Shadow initialization.",
    )
    parser.add_argument(
        "--mano-shadow-init-scale",
        type=float,
        default=1.0,
        help="Scale applied to the MANO-to-Shadow initial joint action before clipping.",
    )
    parser.add_argument(
        "--object-transform-mode",
        choices=["contact_fit", "hand_frame"],
        default="contact_fit",
        help=(
            "contact_fit aligns the synthetic cylinder contacts to canonical robot targets "
            "for a reachable validation scene; hand_frame uses the posed MANO hand frame."
        ),
    )
    parser.add_argument("--output-dir", default="outputs/phase6_cylinder_grasp")
    parser.add_argument("--output-prefix", default="mano_cylinder_shadow")
    parser.add_argument("--skip-optimization", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_html = output_dir / f"{args.output_prefix}_scene_contacts.html"
    robot_html = output_dir / f"{args.output_prefix}_robot_retarget.html"

    mano = create_power_grasp_mano(args.mano_root, seed=args.seed, pose_scale=args.pose_scale)
    cylinder, contact_result = choose_cylinder_and_contacts(
        mano,
        num_mano_samples=args.num_mano_samples,
        contact_threshold=args.contact_threshold,
        cluster_radius=args.cluster_radius,
        min_cluster_size=args.min_cluster_size,
        seed=args.seed,
    )
    patches = build_contact_patch_targets(contact_result, representative="center")
    if not patches:
        raise RuntimeError("No contact patches were extracted. Increase --contact-threshold or adjust the seed.")
    write_scene_html(scene_html, mano, cylinder, contact_result, patches)

    zero_mano = create_zero_mano(args.mano_root)
    canonicalize_patch_targets(patches, zero_mano.vertices, zero_mano.faces)

    robot_model = load_robot_surface_model(
        args.robot_model_dir,
        model_format=args.robot_format,
        wrist_link=args.wrist_link,
    )
    robot_topology = robot_model.sample_surface_topology(
        num_points=args.num_robot_samples,
        seed=args.seed,
        use_farthest_point_sampling=True,
    )
    robot_zero_surface = robot_model.materialize_surface(robot_topology, robot_model.zero_action)

    frame_mapping = map_canonical_mano_patches_to_robot_frame(
        patches,
        zero_mano.keypoints,
        robot_zero_surface,
        index_tip_link=args.robot_index_tip_link,
        middle_tip_link=args.robot_middle_tip_link,
        little_tip_link=args.robot_little_tip_link,
    )
    scene_to_robot = build_scene_to_robot_transform(
        mano.keypoints,
        robot_zero_surface,
        index_tip_link=args.robot_index_tip_link,
        middle_tip_link=args.robot_middle_tip_link,
        little_tip_link=args.robot_little_tip_link,
    )
    if args.object_transform_mode == "contact_fit":
        object_transform = fit_scene_to_robot_contacts(patches)
    else:
        object_transform = scene_to_robot
    object_vertices_robot = object_transform.points(cylinder.vertices)
    transform_patch_object_targets_to_robot(patches, object_transform)

    assignment = assign_patches_to_robot_surface(
        patches,
        robot_zero_surface,
        top_k=args.assignment_top_k,
        temperature=args.temperature,
    )

    config = RetargetOptimizationConfig(
        maxiter_stage1=args.maxiter_stage1,
        maxiter_stage2=args.maxiter_stage2,
        fc_weight=args.fc_weight,
        friction_coef=args.friction_coef,
        object_center=object_vertices_robot.mean(axis=0),
    )

    initial_action = None
    initialization_payload = {"enabled": False}
    if not args.disable_mano_shadow_init:
        lower, upper = robot_model.joint_bounds()
        initialization = mano_pose_to_shadow_action(
            mano.hand_pose,
            robot_model.action_names,
            lower=lower,
            upper=upper,
            scale=args.mano_shadow_init_scale,
        )
        initial_action = initialization.action
        initialization_payload = {
            "enabled": True,
            "scale": float(args.mano_shadow_init_scale),
            "num_clipped": int(initialization.num_clipped),
            "clipped_joint_names": list(initialization.clipped_joint_names),
            "raw_action_summary": action_summary(initialization.raw_action, robot_model.action_names),
            "clipped_action_summary": action_summary(initialization.action, robot_model.action_names),
        }

    optimization_payload = None
    fc_payload = None
    best_action = robot_model.zero_action.copy() if initial_action is None else initial_action.copy()
    best_contacts = np.zeros((0, 3), dtype=np.float64)
    if not args.skip_optimization:
        result = optimize_retarget_action(
            robot_model,
            robot_topology,
            patches,
            config=config,
            initial_action=initial_action,
        )
        best_action = result.best.action
        robot_surface_best = robot_model.materialize_surface(robot_topology, best_action)
        best_contacts, _ = materialize_assigned_robot_contacts(patches, robot_surface_best)
        scores = rank_candidates_by_strict_force_closure(
            result.candidates,
            robot_model,
            robot_topology,
            patches,
            object_vertices_robot,
            cylinder.faces,
            config=config,
        )
        best_score = scores[0] if scores else None
        optimization_payload = {
            "best_index": int(result.best_index),
            "best_success": bool(result.best.success),
            "best_iterations": int(result.best.iterations),
            "best_elapsed_seconds": float(result.best.elapsed_seconds),
            "best_stage1_loss": float(result.best.stage1_loss),
            "best_stage2_loss": float(result.best.stage2_loss),
            "best_loss_terms": {key: float(value) for key, value in result.best.loss_terms.items()},
            "best_action_summary": action_summary(best_action, robot_model.action_names),
            "num_candidates": len(result.candidates),
        }
        if best_score is not None:
            fc = best_score.force_closure
            fc_payload = {
                "candidate_index": int(best_score.candidate_index),
                "is_force_closure": bool(fc.is_force_closure),
                "epsilon": float(fc.epsilon),
                "origin_margin": float(fc.origin_margin),
                "rank": int(fc.rank),
                "num_wrenches": int(fc.num_wrenches),
                "dex_fc_energy": float(best_score.dex_fc_energy),
                "mean_projection_distance": float(best_score.mean_projection_distance),
            }

    write_robot_html(
        robot_html,
        robot_model,
        robot_topology,
        robot_zero_surface,
        object_vertices_robot,
        cylinder.faces,
        patches,
        initial_action,
        best_action,
        best_contacts,
    )

    npz_path = output_dir / f"{args.output_prefix}.npz"
    save_npz(
        npz_path,
        mano,
        cylinder,
        contact_result,
        patches,
        robot_zero_surface,
        object_vertices_robot,
        initial_action,
        best_action,
        best_contacts,
    )

    summary = {
        "status": "ok",
        "seed": args.seed,
        "mano_surface_samples": int(args.num_mano_samples),
        "contact_threshold": float(args.contact_threshold),
        "cluster_radius": float(args.cluster_radius),
        "num_contact_samples": int(len(contact_result.contact_sample_indices)),
        "num_contact_patches": int(len(patches)),
        "patch_sizes": [int(patch.patch_size) for patch in patches],
        "cylinder": {
            "axis_name": cylinder.axis_name,
            "center": cylinder.center.astype(float).tolist(),
            "axis": cylinder.axis.astype(float).tolist(),
            "radius_m": float(cylinder.radius),
            "height_m": float(cylinder.height),
        },
        "canonical_mapping": {
            "scale": float(frame_mapping.scale),
            "mano_mid_length": float(frame_mapping.mano_mid_length),
            "robot_mid_length": float(frame_mapping.robot_mid_length),
        },
        "object_transform_mode": args.object_transform_mode,
        "mano_shadow_initialization": initialization_payload,
        "assignment": {
            "scheme": assignment.scheme,
            "mean_distance": float(assignment.metrics["mean_distance"]),
            "max_distance": float(assignment.metrics["max_distance"]),
            "assigned_link_names": assignment.assigned_link_names.astype(str).tolist()
            if assignment.assigned_link_names is not None
            else [],
        },
        "optimization": optimization_payload,
        "force_closure": fc_payload,
        "outputs": {
            "scene_html": str(scene_html),
            "robot_html": str(robot_html),
            "npz": str(npz_path),
        },
    }
    summary_path = output_dir / f"{args.output_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def patch_numpy_legacy_aliases() -> None:
    for name, value in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def create_power_grasp_mano(mano_root: str, *, seed: int, pose_scale: float) -> ManoGeometry:
    patch_numpy_legacy_aliases()
    import smplx
    import torch

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = smplx.create(
        model_path=mano_root,
        model_type="mano",
        is_rhand=True,
        use_pca=False,
        flat_hand_mean=True,
        batch_size=1,
    )

    hand_pose_np = np.zeros((15, 3), dtype=np.float32)
    for joint_index in range(15):
        depth = joint_index % 3
        curl = [0.45, 0.80, 0.55][depth]
        hand_pose_np[joint_index, 2] = curl + rng.normal(0.0, pose_scale * 0.08)
        hand_pose_np[joint_index, 0] = rng.normal(0.0, pose_scale * 0.05)
        hand_pose_np[joint_index, 1] = rng.normal(0.0, pose_scale * 0.05)
    # Add a mild thumb opposition bias. The exact MANO joint order differs
    # across toolkits; this bias is intentionally small and the cylinder is fit
    # to the resulting mesh, not to assumed semantic axes.
    hand_pose_np[-3:, 1] += np.array([-0.45, -0.25, -0.10], dtype=np.float32)
    hand_pose_np[-3:, 2] += np.array([0.25, 0.35, 0.20], dtype=np.float32)

    import torch

    output = model(
        betas=torch.zeros(1, 10),
        global_orient=torch.zeros(1, 3),
        hand_pose=torch.as_tensor(hand_pose_np.reshape(1, 45), dtype=torch.float32),
        transl=torch.zeros(1, 3),
        return_verts=True,
        return_full_pose=True,
    )
    vertices = output.vertices.detach().cpu().numpy()[0].astype(np.float64)
    joints16 = output.joints.detach().cpu().numpy()[0].astype(np.float64)
    faces = np.asarray(model.faces, dtype=np.int64)
    return ManoGeometry(
        vertices=vertices,
        faces=faces,
        keypoints=make_mano_21_joints(vertices, joints16),
        hand_pose=hand_pose_np.reshape(-1).astype(np.float64),
    )


def create_zero_mano(mano_root: str) -> ManoGeometry:
    patch_numpy_legacy_aliases()
    import smplx
    import torch

    model = smplx.create(
        model_path=mano_root,
        model_type="mano",
        is_rhand=True,
        use_pca=False,
        flat_hand_mean=True,
        batch_size=1,
    )
    output = model(
        betas=torch.zeros(1, 10),
        global_orient=torch.zeros(1, 3),
        hand_pose=torch.zeros(1, 45),
        transl=torch.zeros(1, 3),
        return_verts=True,
        return_full_pose=True,
    )
    vertices = output.vertices.detach().cpu().numpy()[0].astype(np.float64)
    joints16 = output.joints.detach().cpu().numpy()[0].astype(np.float64)
    return ManoGeometry(
        vertices=vertices,
        faces=np.asarray(model.faces, dtype=np.int64),
        keypoints=make_mano_21_joints(vertices, joints16),
        hand_pose=np.zeros(45, dtype=np.float64),
    )


def make_mano_21_joints(vertices: np.ndarray, joints16: np.ndarray) -> np.ndarray:
    tips = vertices[MANO_TIP_VERTEX_IDS]
    joints21 = np.empty((21, 3), dtype=np.float64)
    joints21[0] = joints16[0]
    joints21[1:4] = joints16[13:16]
    joints21[4] = tips[0]
    joints21[5:8] = joints16[1:4]
    joints21[8] = tips[1]
    joints21[9:12] = joints16[4:7]
    joints21[12] = tips[2]
    joints21[13:16] = joints16[10:13]
    joints21[16] = tips[3]
    joints21[17:20] = joints16[7:10]
    joints21[20] = tips[4]
    return joints21


def choose_cylinder_and_contacts(
    mano: ManoGeometry,
    *,
    num_mano_samples: int,
    contact_threshold: float,
    cluster_radius: float,
    min_cluster_size: int,
    seed: int,
) -> tuple[CylinderGeometry, object]:
    frame = hand_frame_axes(mano.keypoints)
    candidates = [
        ("finger_axis", frame["forward"]),
        ("palm_lateral", frame["lateral"]),
        ("palm_normal", frame["normal"]),
    ]
    fit_points = mano.keypoints[MANO_FINGERTIP_INDICES]
    best = None
    for axis_name, axis in candidates:
        cylinder = fit_cylinder_to_points(
            fit_points,
            axis=axis,
            axis_name=axis_name,
            sections=96,
            height_margin=0.040,
        )
        contact_result = extract_mano_object_contact_clusters(
            mano.vertices,
            mano.faces,
            cylinder.vertices,
            cylinder.faces,
            num_mano_samples=num_mano_samples,
            contact_threshold=contact_threshold,
            cluster_radius=cluster_radius,
            min_cluster_size=min_cluster_size,
            cluster_space="object",
            seed=seed,
            use_farthest_point_sampling=True,
            oversample_factor=16,
        )
        radius_penalty = abs(float(cylinder.radius) - 0.035) * 1000.0
        radius_valid = 0.012 <= cylinder.radius <= 0.075
        score = (
            len(contact_result.clusters) * 10000.0
            + min(len(contact_result.contact_sample_indices), 500)
            - radius_penalty
            + (500.0 if radius_valid else -1000.0)
        )
        if best is None or score > best[0]:
            best = (score, cylinder, contact_result)
    if best is None:
        raise RuntimeError("Failed to construct a cylinder candidate.")
    return best[1], best[2]


def fit_cylinder_to_points(
    points: np.ndarray,
    *,
    axis: np.ndarray,
    axis_name: str,
    sections: int,
    height_margin: float,
) -> CylinderGeometry:
    points = np.asarray(points, dtype=np.float64)
    axis = normalize(axis)
    basis_u, basis_v = tangent_basis(axis)
    origin = points.mean(axis=0)
    coords = np.stack([(points - origin) @ basis_u, (points - origin) @ basis_v], axis=1)
    a = np.column_stack([2.0 * coords[:, 0], 2.0 * coords[:, 1], np.ones(len(coords))])
    b = np.sum(coords * coords, axis=1)
    solution, *_ = np.linalg.lstsq(a, b, rcond=None)
    center_2d = solution[:2]
    radius_sq = max(float(solution[2] + np.dot(center_2d, center_2d)), 1e-8)
    radius = float(np.sqrt(radius_sq))
    axis_values = (points - origin) @ axis
    height = float(axis_values.max() - axis_values.min() + height_margin)
    center_axis = origin + axis * float(axis_values.mean()) + center_2d[0] * basis_u + center_2d[1] * basis_v
    vertices, faces = make_cylinder_mesh(center_axis, axis, radius, height, sections=sections)
    return CylinderGeometry(
        vertices=vertices,
        faces=faces,
        center=center_axis,
        axis=axis,
        radius=radius,
        height=height,
        axis_name=axis_name,
    )


def make_cylinder_mesh(
    center: np.ndarray,
    axis: np.ndarray,
    radius: float,
    height: float,
    *,
    sections: int,
) -> tuple[np.ndarray, np.ndarray]:
    axis = normalize(axis)
    basis_u, basis_v = tangent_basis(axis)
    angles = np.linspace(0.0, 2.0 * np.pi, sections, endpoint=False)
    circle = radius * (np.cos(angles)[:, None] * basis_u + np.sin(angles)[:, None] * basis_v)
    bottom_center = center - axis * (height * 0.5)
    top_center = center + axis * (height * 0.5)
    bottom = bottom_center + circle
    top = top_center + circle
    vertices = np.vstack([bottom, top, bottom_center[None], top_center[None]]).astype(np.float64)
    bottom_center_index = sections * 2
    top_center_index = sections * 2 + 1
    faces = []
    for i in range(sections):
        j = (i + 1) % sections
        faces.append([i, j, sections + j])
        faces.append([i, sections + j, sections + i])
        faces.append([bottom_center_index, j, i])
        faces.append([top_center_index, sections + i, sections + j])
    return vertices, np.asarray(faces, dtype=np.int64)


def build_scene_to_robot_transform(
    mano_keypoints: np.ndarray,
    robot_surface: RobotSurfaceSamples,
    *,
    index_tip_link: str,
    middle_tip_link: str,
    little_tip_link: str,
) -> SceneToRobotTransform:
    scene_frame = build_hand_frame(
        wrist=mano_keypoints[0],
        index_mcp=mano_keypoints[5],
        middle_mcp=mano_keypoints[9],
        little_mcp=mano_keypoints[17],
    )
    robot_points = np.asarray(robot_surface.points, dtype=np.float64)
    robot_links = np.asarray(robot_surface.link_names).astype(str)
    robot_index = robot_tip_from_link(robot_points, robot_links, index_tip_link)
    robot_middle = robot_tip_from_link(robot_points, robot_links, middle_tip_link)
    robot_little = robot_tip_from_link(robot_points, robot_links, little_tip_link)
    robot_frame = build_hand_frame(
        wrist=np.zeros(3, dtype=np.float64),
        index_mcp=robot_index,
        middle_mcp=robot_middle,
        little_mcp=robot_little,
    )
    mano_mid = max(float(np.linalg.norm(mano_keypoints[12] - mano_keypoints[0])), 1e-8)
    robot_mid = max(float(np.linalg.norm(robot_middle)), 1e-8)
    return SceneToRobotTransform(
        wrist=mano_keypoints[0].copy(),
        scene_frame=scene_frame,
        robot_frame=robot_frame,
        scale=robot_mid / mano_mid,
    )


def fit_scene_to_robot_contacts(patches) -> SimilarityTransform:
    source = np.stack([patch.object_point_target for patch in patches], axis=0).astype(np.float64)
    target = np.stack([patch.canonical_robot_target for patch in patches], axis=0).astype(np.float64)
    if len(source) < 3:
        raise ValueError("contact_fit object transform requires at least three contact patches.")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    covariance = source_centered.T @ target_centered
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1.0
        rotation = u @ vt
    source_var = float(np.sum(source_centered * source_centered))
    scale = float(np.sum(singular_values) / max(source_var, 1e-12))
    return SimilarityTransform(
        source_center=source_center,
        target_center=target_center,
        rotation=rotation,
        scale=scale,
    )


def transform_patch_object_targets_to_robot(patches, transform: SceneToRobotTransform) -> None:
    for patch in patches:
        patch.object_point_target = transform.points(patch.object_point_target[None])[0]
        patch.object_point_nearest = transform.points(patch.object_point_nearest[None])[0]
        patch.object_point_center = transform.points(patch.object_point_center[None])[0]
        patch.object_normal_target = transform.normals(patch.object_normal_target[None])[0]
        patch.object_normal_nearest = transform.normals(patch.object_normal_nearest[None])[0]
        patch.object_normal_center = transform.normals(patch.object_normal_center[None])[0]


def hand_frame_axes(keypoints: np.ndarray) -> dict[str, np.ndarray]:
    frame = build_hand_frame(
        wrist=keypoints[0],
        index_mcp=keypoints[5],
        middle_mcp=keypoints[9],
        little_mcp=keypoints[17],
    )
    return {
        "lateral": frame[:, 0],
        "forward": frame[:, 1],
        "normal": frame[:, 2],
    }


def write_scene_html(output_path: Path, mano: ManoGeometry, cylinder: CylinderGeometry, contact_result, patches) -> None:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=mano.vertices[:, 0],
            y=mano.vertices[:, 1],
            z=mano.vertices[:, 2],
            i=mano.faces[:, 0],
            j=mano.faces[:, 1],
            k=mano.faces[:, 2],
            name="posed MANO grasp",
            color="rgba(45, 180, 95, 0.30)",
            opacity=0.30,
        )
    )
    fig.add_trace(
        go.Mesh3d(
            x=cylinder.vertices[:, 0],
            y=cylinder.vertices[:, 1],
            z=cylinder.vertices[:, 2],
            i=cylinder.faces[:, 0],
            j=cylinder.faces[:, 1],
            k=cylinder.faces[:, 2],
            name=f"cylinder object ({cylinder.axis_name})",
            color="rgba(80, 120, 240, 0.42)",
            opacity=0.42,
        )
    )
    samples = contact_result.mano_samples
    fig.add_trace(
        go.Scatter3d(
            x=samples.points[:, 0],
            y=samples.points[:, 1],
            z=samples.points[:, 2],
            mode="markers",
            name=f"MANO surface samples ({len(samples.points)})",
            marker=dict(size=1.2, color="rgba(90,90,90,0.22)"),
        )
    )
    contact_indices = contact_result.contact_sample_indices
    labels = contact_result.cluster_labels.astype(float)
    if len(contact_indices):
        pts = samples.points[contact_indices]
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                name=f"thresholded MANO contacts ({len(pts)})",
                marker=dict(size=4.0, color=labels, colorscale="Turbo", showscale=True),
            )
        )
        obj = contact_result.object_points
        fig.add_trace(
            go.Scatter3d(
                x=obj[:, 0],
                y=obj[:, 1],
                z=obj[:, 2],
                mode="markers",
                name="closest object points",
                marker=dict(size=3.0, color=labels, colorscale="Turbo", showscale=False, symbol="diamond"),
            )
        )
    if patches:
        mano_reps = np.stack([patch.mano_point_posed for patch in patches], axis=0)
        object_reps = np.stack([patch.object_point_nearest for patch in patches], axis=0)
        fig.add_trace(
            go.Scatter3d(
                x=mano_reps[:, 0],
                y=mano_reps[:, 1],
                z=mano_reps[:, 2],
                mode="markers+text",
                name="MANO patch representatives",
                text=[f"patch_{i}" for i in range(len(patches))],
                marker=dict(size=7, color="black", symbol="circle"),
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=object_reps[:, 0],
                y=object_reps[:, 1],
                z=object_reps[:, 2],
                mode="markers",
                name="object representatives",
                marker=dict(size=6, color="red", symbol="x"),
            )
        )
        for mano_point, obj_point in zip(mano_reps, object_reps):
            pair = np.stack([mano_point, obj_point], axis=0)
            fig.add_trace(
                go.Scatter3d(
                    x=pair[:, 0],
                    y=pair[:, 1],
                    z=pair[:, 2],
                    mode="lines",
                    name="contact projection",
                    line=dict(color="rgba(20,20,20,0.45)", width=3),
                    showlegend=False,
                )
            )
    fig.update_layout(
        title="Synthetic MANO grasping a cylinder: contact extraction and clustering",
        scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        margin=dict(l=0, r=0, t=45, b=0),
    )
    fig.write_html(output_path, include_plotlyjs="cdn")


def write_robot_html(
    output_path: Path,
    robot_model,
    robot_topology: RobotSurfaceTopology,
    robot_zero_surface: RobotSurfaceSamples,
    object_vertices_robot: np.ndarray,
    object_faces: np.ndarray,
    patches,
    initial_action: np.ndarray | None,
    action: np.ndarray,
    best_contacts: np.ndarray,
) -> None:
    import plotly.graph_objects as go

    if initial_action is not None:
        initial_vertices, initial_faces = robot_model.link_mesh(initial_action)
    else:
        initial_vertices, initial_faces = None, None
    robot_vertices, robot_faces = robot_model.link_mesh(action)
    fig = go.Figure()
    if initial_vertices is not None and initial_faces is not None:
        fig.add_trace(
            go.Mesh3d(
                x=initial_vertices[:, 0],
                y=initial_vertices[:, 1],
                z=initial_vertices[:, 2],
                i=initial_faces[:, 0],
                j=initial_faces[:, 1],
                k=initial_faces[:, 2],
                name="MANO-to-Shadow initial hand",
                color="rgba(35, 165, 95, 0.18)",
                opacity=0.18,
            )
        )
    fig.add_trace(
        go.Mesh3d(
            x=robot_vertices[:, 0],
            y=robot_vertices[:, 1],
            z=robot_vertices[:, 2],
            i=robot_faces[:, 0],
            j=robot_faces[:, 1],
            k=robot_faces[:, 2],
            name="optimized robot hand",
            color="rgba(45, 105, 220, 0.35)",
            opacity=0.35,
        )
    )
    fig.add_trace(
        go.Mesh3d(
            x=object_vertices_robot[:, 0],
            y=object_vertices_robot[:, 1],
            z=object_vertices_robot[:, 2],
            i=object_faces[:, 0],
            j=object_faces[:, 1],
            k=object_faces[:, 2],
            name="cylinder in robot wrist frame",
            color="rgba(240, 160, 55, 0.38)",
            opacity=0.38,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=robot_zero_surface.points[:, 0],
            y=robot_zero_surface.points[:, 1],
            z=robot_zero_surface.points[:, 2],
            mode="markers",
            name=f"fixed robot surface samples ({len(robot_zero_surface.points)})",
            marker=dict(size=1.1, color="rgba(80,80,80,0.18)"),
            text=np.asarray(robot_zero_surface.link_names).astype(str),
        )
    )
    if patches:
        object_targets = np.stack([patch.object_point_target for patch in patches], axis=0)
        canonical_targets = np.stack([patch.canonical_robot_target for patch in patches], axis=0)
        fig.add_trace(
            go.Scatter3d(
                x=canonical_targets[:, 0],
                y=canonical_targets[:, 1],
                z=canonical_targets[:, 2],
                mode="markers+text",
                name="canonical MANO targets in robot frame",
                text=[f"canon_{i}" for i in range(len(patches))],
                marker=dict(size=6, color="black", symbol="diamond"),
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=object_targets[:, 0],
                y=object_targets[:, 1],
                z=object_targets[:, 2],
                mode="markers+text",
                name="object contact targets",
                text=[f"obj_{i}" for i in range(len(patches))],
                marker=dict(size=7, color="red", symbol="x"),
            )
        )
        if len(best_contacts):
            fig.add_trace(
                go.Scatter3d(
                    x=best_contacts[:, 0],
                    y=best_contacts[:, 1],
                    z=best_contacts[:, 2],
                    mode="markers+text",
                    name="optimized robot contacts",
                    text=[f"robot_{i}" for i in range(len(best_contacts))],
                    marker=dict(size=7, color="blue", symbol="circle"),
                )
            )
            for obj, contact in zip(object_targets, best_contacts):
                pair = np.stack([obj, contact], axis=0)
                fig.add_trace(
                    go.Scatter3d(
                        x=pair[:, 0],
                        y=pair[:, 1],
                        z=pair[:, 2],
                        mode="lines",
                        name="contact error",
                        line=dict(color="rgba(20,20,20,0.45)", width=3),
                        showlegend=False,
                    )
                )
    fig.update_layout(
        title="Cylinder contact targets retargeted to the prosthetic hand",
        scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        margin=dict(l=0, r=0, t=45, b=0),
    )
    fig.write_html(output_path, include_plotlyjs="cdn")


def save_npz(
    output_path: Path,
    mano: ManoGeometry,
    cylinder: CylinderGeometry,
    contact_result,
    patches,
    robot_zero_surface: RobotSurfaceSamples,
    object_vertices_robot: np.ndarray,
    initial_action: np.ndarray | None,
    best_action: np.ndarray,
    best_contacts: np.ndarray,
) -> None:
    payload = {
        "mano_vertices": mano.vertices.astype(np.float32),
        "mano_faces": mano.faces.astype(np.int64),
        "mano_keypoints": mano.keypoints.astype(np.float32),
        "mano_hand_pose": mano.hand_pose.astype(np.float32),
        "object_vertices": cylinder.vertices.astype(np.float32),
        "object_faces": cylinder.faces.astype(np.int64),
        "object_vertices_robot": object_vertices_robot.astype(np.float32),
        "surface_points": contact_result.mano_samples.points.astype(np.float32),
        "surface_normals": contact_result.mano_samples.normals.astype(np.float32),
        "surface_face_indices": contact_result.mano_samples.face_indices.astype(np.int64),
        "surface_barycentric": contact_result.mano_samples.barycentric.astype(np.float32),
        "contact_sample_indices": contact_result.contact_sample_indices.astype(np.int64),
        "cluster_labels": contact_result.cluster_labels.astype(np.int64),
        "contact_object_points": contact_result.object_points.astype(np.float32),
        "contact_object_normals": contact_result.object_normals.astype(np.float32),
        "robot_zero_points": robot_zero_surface.points.astype(np.float32),
        "robot_zero_normals": robot_zero_surface.normals.astype(np.float32),
        "robot_link_names": np.asarray(robot_zero_surface.link_names).astype(str),
        "initial_action": np.asarray(
            np.zeros(0, dtype=np.float64) if initial_action is None else initial_action,
            dtype=np.float32,
        ),
        "best_action": np.asarray(best_action, dtype=np.float32),
        "best_contacts": np.asarray(best_contacts, dtype=np.float32),
    }
    if patches:
        payload.update(
            {
                "patch_mano_points": np.stack([patch.mano_point_posed for patch in patches]).astype(np.float32),
                "patch_object_targets": np.stack([patch.object_point_target for patch in patches]).astype(np.float32),
                "patch_object_normals": np.stack([patch.object_normal_target for patch in patches]).astype(np.float32),
                "patch_canonical_points": np.stack([patch.mano_point_canonical for patch in patches]).astype(np.float32),
                "patch_canonical_robot_targets": np.stack([patch.canonical_robot_target for patch in patches]).astype(np.float32),
                "patch_robot_assignment_indices": np.stack([patch.robot_assignment_indices for patch in patches]).astype(np.int64),
                "patch_robot_assignment_weights": np.stack([patch.robot_assignment_weights for patch in patches]).astype(np.float32),
            }
        )
    np.savez_compressed(output_path, **payload)


def tangent_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = normalize(axis)
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(axis, helper))) > 0.85:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    basis_u = normalize(np.cross(axis, helper))
    basis_v = normalize(np.cross(axis, basis_u))
    return basis_u, basis_v


def normalize(value: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return vector
    return vector / norm


def normalize_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), eps)


if __name__ == "__main__":
    main()
