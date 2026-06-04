from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prosthetic_grasp.geometry import (
    mapping_result_to_json_dict,
    run_all_contact_mapping_schemes,
)


MANO_TIP_VERTEX_IDS = np.array([744, 320, 443, 554, 671], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run four contact-target to robot-surface mapping schemes and record timings."
    )
    parser.add_argument(
        "--target-npz",
        default="outputs/mano_contact_clustering/random_mano_finger_patches.npz",
        help="NPZ from tools/test_mano_contact_clustering.py.",
    )
    parser.add_argument(
        "--robot-npz",
        default="outputs/robot_surface_samples/folding_hand_right_zero_samples.npz",
        help="Robot surface samples NPZ from tools/visualize_robot_surface_samples.py.",
    )
    parser.add_argument("--representative", choices=["center", "nearest"], default="center")
    parser.add_argument(
        "--canonicalize-mano",
        action="store_true",
        help=(
            "Use representative sample topology to recover targets on zero MANO, "
            "then map them into the robot wrist frame."
        ),
    )
    parser.add_argument("--mano-root", default="models")
    parser.add_argument("--robot-index-tip-link", default="robot0:ffdistal")
    parser.add_argument("--robot-middle-tip-link", default="robot0:mfdistal")
    parser.add_argument("--robot-little-tip-link", default="robot0:lfdistal")
    parser.add_argument("--min-scale", type=float, default=0.25)
    parser.add_argument("--max-scale", type=float, default=4.0)
    parser.add_argument("--temperature", type=float, default=1e-4)
    parser.add_argument("--spread-sigma", type=float, default=0.025)
    parser.add_argument("--diversity-weight", type=float, default=1.0)
    parser.add_argument("--unique-links", action="store_true")
    parser.add_argument("--output-dir", default="outputs/contact_mapping")
    parser.add_argument("--output-prefix", default="mano_targets_to_folding_hand_right")
    parser.add_argument("--write-html", action="store_true")
    return parser.parse_args()


def load_targets(path: str | Path, representative: str) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    contact_points = np.asarray(data["contact_points"], dtype=np.float64)
    key = f"{representative}_rep_indices"
    if key not in data:
        raise KeyError(f"{path} has no {key!r}.")
    indices = np.asarray(data[key], dtype=np.int64)
    return contact_points[indices]


def load_canonical_mano_targets(
    path: str | Path,
    representative: str,
    *,
    mano_root: str,
    robot_points: np.ndarray,
    robot_link_names: np.ndarray,
    robot_index_tip_link: str,
    robot_middle_tip_link: str,
    robot_little_tip_link: str,
    min_scale: float,
    max_scale: float,
) -> tuple[np.ndarray, dict[str, object]]:
    data = np.load(path, allow_pickle=True)
    sample_key = f"{representative}_rep_sample_indices"
    if sample_key not in data:
        raise KeyError(
            f"{path} has no {sample_key!r}. Re-run tools/test_mano_contact_clustering.py "
            "so representative topology metadata is saved."
        )
    for key in ["surface_face_indices", "surface_barycentric"]:
        if key not in data:
            raise KeyError(f"{path} has no {key!r}. Re-run the clustering script.")

    rep_sample_indices = np.asarray(data[sample_key], dtype=np.int64)
    surface_face_indices = np.asarray(data["surface_face_indices"], dtype=np.int64)
    surface_barycentric = np.asarray(data["surface_barycentric"], dtype=np.float64)
    face_indices = surface_face_indices[rep_sample_indices]
    barycentric = surface_barycentric[rep_sample_indices]

    zero_vertices, zero_faces, zero_keypoints = create_mano(mano_root, pose="zero", seed=0, pose_scale=0.0)
    canonical_points = materialize_points_from_topology(zero_vertices, zero_faces, face_indices, barycentric)
    mapped_points, metadata = map_mano_points_to_robot_wrist_frame(
        canonical_points,
        zero_keypoints,
        robot_points,
        robot_link_names,
        index_tip_link=robot_index_tip_link,
        middle_tip_link=robot_middle_tip_link,
        little_tip_link=robot_little_tip_link,
        min_scale=min_scale,
        max_scale=max_scale,
    )
    metadata.update(
        {
            "rep_sample_indices": rep_sample_indices.astype(int).tolist(),
            "face_indices": face_indices.astype(int).tolist(),
            "canonical_mano_points": canonical_points.astype(float).tolist(),
        }
    )
    return mapped_points, metadata


def load_robot_samples(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    points = np.asarray(data["points"], dtype=np.float64)
    link_names = np.asarray(data["link_names"].astype(str), dtype=object)
    return points, link_names


def create_mano(
    mano_root: str,
    *,
    pose: str,
    seed: int,
    pose_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    patch_numpy_legacy_aliases()
    import smplx
    import torch

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
    if pose == "zero":
        hand_pose = torch.zeros(1, 45)
    else:
        hand_pose = torch.randn(1, 45, generator=generator) * pose_scale
        hand_pose[:, 2::3] = hand_pose[:, 2::3].abs()
    output = model(
        betas=torch.zeros(1, 10),
        global_orient=torch.zeros(1, 3),
        hand_pose=hand_pose,
        transl=torch.zeros(1, 3),
        return_verts=True,
        return_full_pose=True,
    )
    vertices = output.vertices.detach().cpu().numpy()[0].astype(np.float64)
    joints16 = output.joints.detach().cpu().numpy()[0].astype(np.float64)
    keypoints = make_mano_21_joints(vertices, joints16)
    return vertices, np.asarray(model.faces, dtype=np.int64), keypoints


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


def materialize_points_from_topology(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_indices: np.ndarray,
    barycentric: np.ndarray,
) -> np.ndarray:
    triangles = vertices[faces[face_indices]]
    return np.einsum("ni,nij->nj", barycentric, triangles)


def map_mano_points_to_robot_wrist_frame(
    mano_points: np.ndarray,
    mano_keypoints: np.ndarray,
    robot_points: np.ndarray,
    robot_link_names: np.ndarray,
    *,
    index_tip_link: str,
    middle_tip_link: str,
    little_tip_link: str,
    min_scale: float,
    max_scale: float,
) -> tuple[np.ndarray, dict[str, object]]:
    mano_frame = build_hand_frame(
        wrist=mano_keypoints[0],
        index_mcp=mano_keypoints[5],
        middle_mcp=mano_keypoints[9],
        little_mcp=mano_keypoints[17],
    )
    robot_index = robot_tip_from_link(robot_points, robot_link_names, index_tip_link)
    robot_middle = robot_tip_from_link(robot_points, robot_link_names, middle_tip_link)
    robot_little = robot_tip_from_link(robot_points, robot_link_names, little_tip_link)
    robot_frame = build_hand_frame(
        wrist=np.zeros(3, dtype=np.float64),
        index_mcp=robot_index,
        middle_mcp=robot_middle,
        little_mcp=robot_little,
    )

    mano_rel = mano_points - mano_keypoints[0]
    mano_local = mano_rel @ mano_frame
    mano_mid = max(float(np.linalg.norm(mano_keypoints[12] - mano_keypoints[0])), 1e-8)
    robot_mid = max(float(np.linalg.norm(robot_middle)), 1e-8)
    scale = float(np.clip(robot_mid / mano_mid, min_scale, max_scale))
    mapped = (mano_local * scale) @ robot_frame.T
    return mapped, {
        "mano_mid_length": mano_mid,
        "robot_mid_length": robot_mid,
        "scale": scale,
        "robot_tip_links": [index_tip_link, middle_tip_link, little_tip_link],
    }


def robot_tip_from_link(robot_points: np.ndarray, robot_link_names: np.ndarray, link_name: str) -> np.ndarray:
    links = robot_link_names.astype(str)
    mask = links == link_name
    if not np.any(mask):
        raise ValueError(f"robot_link_names does not contain {link_name!r}.")
    points = robot_points[mask]
    return points[np.argmax(np.linalg.norm(points, axis=1))]


def build_hand_frame(
    *,
    wrist: np.ndarray,
    index_mcp: np.ndarray,
    middle_mcp: np.ndarray,
    little_mcp: np.ndarray,
) -> np.ndarray:
    forward = normalize(middle_mcp - wrist)
    lateral = normalize(index_mcp - little_mcp)
    normal = normalize(np.cross(lateral, forward))
    if np.linalg.norm(normal) < 1e-8:
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    lateral = normalize(np.cross(forward, normal))
    return np.stack([lateral, forward, normal], axis=1)


def normalize(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        return value
    return value / norm


def save_npz(output_path: Path, results) -> None:
    payload = {}
    for result in results:
        prefix = result.scheme
        payload[f"{prefix}_assigned_indices"] = result.assigned_indices.astype(np.int64)
        payload[f"{prefix}_assigned_points"] = result.assigned_points.astype(np.float32)
        payload[f"{prefix}_distances"] = result.distances.astype(np.float32)
        if result.assigned_link_names is not None:
            payload[f"{prefix}_assigned_link_names"] = result.assigned_link_names.astype(str)
        if result.expected_points is not None:
            payload[f"{prefix}_expected_points"] = result.expected_points.astype(np.float32)
        if result.assigned_link_indices is not None:
            payload[f"{prefix}_assigned_link_indices"] = result.assigned_link_indices.astype(np.int64)
        if result.link_names is not None:
            payload[f"{prefix}_link_names"] = result.link_names.astype(str)
        if result.link_scores is not None:
            payload[f"{prefix}_link_scores"] = result.link_scores.astype(np.float32)
    np.savez_compressed(output_path, **payload)


def write_html(output_path: Path, target_points: np.ndarray, robot_points: np.ndarray, robot_link_names: np.ndarray, results) -> None:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=robot_points[:, 0],
            y=robot_points[:, 1],
            z=robot_points[:, 2],
            mode="markers",
            name=f"robot surface samples ({len(robot_points)})",
            marker=dict(size=1.4, color="rgba(90,90,90,0.28)"),
            text=robot_link_names.astype(str),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=target_points[:, 0],
            y=target_points[:, 1],
            z=target_points[:, 2],
            mode="markers+text",
            name="target representatives",
            text=[f"target_{i}" for i in range(len(target_points))],
            marker=dict(size=7, color="black", symbol="diamond"),
        )
    )
    colors = ["red", "blue", "green", "purple", "orange", "cyan"]
    for scheme_index, result in enumerate(results):
        color = colors[scheme_index % len(colors)]
        fig.add_trace(
            go.Scatter3d(
                x=result.assigned_points[:, 0],
                y=result.assigned_points[:, 1],
                z=result.assigned_points[:, 2],
                mode="markers+text",
                name=result.scheme,
                text=[f"{result.scheme}_{i}" for i in range(len(result.assigned_points))],
                marker=dict(size=5, color=color),
            )
        )
        for i, target in enumerate(target_points):
            pair = np.stack([target, result.assigned_points[i]], axis=0)
            fig.add_trace(
                go.Scatter3d(
                    x=pair[:, 0],
                    y=pair[:, 1],
                    z=pair[:, 2],
                    mode="lines",
                    name=f"{result.scheme} line",
                    line=dict(color=color, width=3),
                    showlegend=False,
                )
            )
    fig.update_layout(
        title="Contact Mapping Schemes",
        scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(output_path, include_plotlyjs="cdn")


def main() -> None:
    args = parse_args()
    robot_points, robot_link_names = load_robot_samples(args.robot_npz)
    canonical_metadata = None
    if args.canonicalize_mano:
        target_points, canonical_metadata = load_canonical_mano_targets(
            args.target_npz,
            args.representative,
            mano_root=args.mano_root,
            robot_points=robot_points,
            robot_link_names=robot_link_names,
            robot_index_tip_link=args.robot_index_tip_link,
            robot_middle_tip_link=args.robot_middle_tip_link,
            robot_little_tip_link=args.robot_little_tip_link,
            min_scale=args.min_scale,
            max_scale=args.max_scale,
        )
    else:
        target_points = load_targets(args.target_npz, args.representative)
    results = run_all_contact_mapping_schemes(
        target_points,
        robot_points,
        robot_link_names=robot_link_names,
        temperature=args.temperature,
        spread_sigma=args.spread_sigma,
        diversity_weight=args.diversity_weight,
        unique_links=args.unique_links,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.output_prefix}_{args.representative}.json"
    npz_path = output_dir / f"{args.output_prefix}_{args.representative}.npz"
    payload = {
        "target_npz": args.target_npz,
        "robot_npz": args.robot_npz,
        "representative": args.representative,
        "num_targets": int(len(target_points)),
        "num_robot_samples": int(len(robot_points)),
        "canonicalize_mano": bool(args.canonicalize_mano),
        "canonical_metadata": canonical_metadata,
        "results": [mapping_result_to_json_dict(result) for result in results],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_npz(npz_path, results)
    print(f"targets: {target_points.shape}")
    print(f"robot samples: {robot_points.shape}")
    for result in results:
        print(
            f"{result.scheme}: {result.elapsed_seconds:.6f}s, "
            f"mean_dist={result.metrics['mean_distance']:.6f}, "
            f"max_dist={result.metrics['max_distance']:.6f}"
        )
    print(f"saved stats: {json_path}")
    print(f"saved mapping arrays: {npz_path}")
    if args.write_html:
        html_path = output_dir / f"{args.output_prefix}_{args.representative}.html"
        write_html(html_path, target_points, robot_points, robot_link_names, results)
        print(f"visualization: {html_path}")


if __name__ == "__main__":
    main()
