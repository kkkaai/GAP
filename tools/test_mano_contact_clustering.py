from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prosthetic_grasp.geometry import radius_connected_components, sample_mano_surface


MANO_TIP_VERTEX_IDS = np.array([744, 320, 443, 554, 671], dtype=np.int64)
FINGER_CHAINS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "little": [17, 18, 19, 20],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic MANO contact-clustering test using random per-finger contact patches."
    )
    parser.add_argument("--mano-root", default="models")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--pose-scale", type=float, default=0.45)
    parser.add_argument("--num-surface-points", type=int, default=2000)
    parser.add_argument("--oversample-factor", type=int, default=20)
    parser.add_argument("--patch-radius", type=float, default=0.010)
    parser.add_argument("--cluster-radius", type=float, default=0.014)
    parser.add_argument("--min-cluster-size", type=int, default=4)
    parser.add_argument("--min-seed-distance", type=float, default=0.025)
    parser.add_argument("--output-dir", default="outputs/mano_contact_clustering")
    parser.add_argument("--output-prefix", default="random_mano_finger_patches")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


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


def create_random_mano(
    mano_root: str,
    *,
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
    faces = np.asarray(model.faces, dtype=np.int64)
    return vertices, faces, keypoints


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


def assign_samples_to_fingers(points: np.ndarray, keypoints: np.ndarray) -> dict[str, np.ndarray]:
    assignments: dict[str, np.ndarray] = {}
    finger_names = list(FINGER_CHAINS)
    finger_keypoints = np.stack([keypoints[FINGER_CHAINS[name]] for name in finger_names], axis=0)
    distances = np.linalg.norm(points[:, None, None, :] - finger_keypoints[None, :, :, :], axis=-1).min(axis=-1)
    nearest_finger = np.argmin(distances, axis=1)
    for index, name in enumerate(finger_names):
        assignments[name] = np.nonzero(nearest_finger == index)[0].astype(np.int64)
    return assignments


def build_synthetic_contact_set(
    sample_points: np.ndarray,
    finger_assignments: dict[str, np.ndarray],
    *,
    patch_radius: float,
    min_seed_distance: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    contact_indices = []
    seed_indices: dict[str, int] = {}
    chosen_seed_points = []
    for finger_name, candidates in finger_assignments.items():
        if len(candidates) == 0:
            continue
        shuffled = np.asarray(candidates, dtype=np.int64).copy()
        rng.shuffle(shuffled)
        seed_index = int(shuffled[0])
        if chosen_seed_points and min_seed_distance > 0:
            chosen = np.stack(chosen_seed_points, axis=0)
            candidate_distances = np.linalg.norm(sample_points[shuffled, None, :] - chosen[None, :, :], axis=-1)
            min_distances = candidate_distances.min(axis=1)
            valid = np.nonzero(min_distances >= min_seed_distance)[0]
            if len(valid) > 0:
                seed_index = int(shuffled[valid[0]])
            else:
                seed_index = int(shuffled[np.argmax(min_distances)])
        seed_indices[finger_name] = seed_index
        chosen_seed_points.append(sample_points[seed_index])
        distances = np.linalg.norm(sample_points - sample_points[seed_index], axis=1)
        patch = np.nonzero(distances <= patch_radius)[0]
        # Keep the patch finger-local so adjacent fingers do not create an artificial bridge.
        patch = np.intersect1d(patch, candidates, assume_unique=False)
        if len(patch) == 0:
            patch = np.asarray([seed_index], dtype=np.int64)
        contact_indices.append(patch.astype(np.int64))
    if not contact_indices:
        raise RuntimeError("No synthetic contact patches were generated.")
    merged = np.unique(np.concatenate(contact_indices))
    return merged, sample_points[merged], seed_indices


def representative_indices(
    contact_points: np.ndarray,
    labels: np.ndarray,
    seed_points: np.ndarray,
) -> tuple[list[int], list[int]]:
    nearest_reps = []
    center_reps = []
    for label in sorted(int(x) for x in np.unique(labels) if x >= 0):
        local = np.nonzero(labels == label)[0]
        center = contact_points[local].mean(axis=0)
        center_reps.append(int(local[np.argmin(np.linalg.norm(contact_points[local] - center, axis=1))]))
        seed_distances = np.linalg.norm(contact_points[local, None, :] - seed_points[None, :, :], axis=-1).min(axis=1)
        nearest_reps.append(int(local[np.argmin(seed_distances)]))
    return nearest_reps, center_reps


def make_figure(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    keypoints: np.ndarray,
    surface_points: np.ndarray,
    contact_points: np.ndarray,
    labels: np.ndarray,
    seed_points: np.ndarray,
    nearest_reps: list[int],
    center_reps: list[int],
):
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            name="random MANO mesh",
            color="rgba(40, 180, 80, 0.24)",
            opacity=0.24,
            flatshading=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=surface_points[:, 0],
            y=surface_points[:, 1],
            z=surface_points[:, 2],
            mode="markers",
            name=f"surface samples ({len(surface_points)})",
            marker=dict(size=1.4, color="rgba(120,120,120,0.30)"),
        )
    )
    color_values = labels.astype(float)
    color_values[color_values < 0] = -1.0
    fig.add_trace(
        go.Scatter3d(
            x=contact_points[:, 0],
            y=contact_points[:, 1],
            z=contact_points[:, 2],
            mode="markers",
            name=f"synthetic contact set ({len(contact_points)})",
            marker=dict(size=4.0, color=color_values, colorscale="Turbo", showscale=True),
            text=[f"cluster {label}" for label in labels],
        )
    )
    if len(seed_points):
        fig.add_trace(
            go.Scatter3d(
                x=seed_points[:, 0],
                y=seed_points[:, 1],
                z=seed_points[:, 2],
                mode="markers+text",
                name="manual per-finger seed points",
                text=list(FINGER_CHAINS),
                marker=dict(size=8, color="black", symbol="diamond"),
            )
        )
    if nearest_reps:
        pts = contact_points[np.asarray(nearest_reps, dtype=np.int64)]
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers+text",
                name="nearest representatives",
                text=[f"near_{i}" for i in range(len(pts))],
                marker=dict(size=7, color="red", symbol="circle"),
            )
        )
    if center_reps:
        pts = contact_points[np.asarray(center_reps, dtype=np.int64)]
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers+text",
                name="center representatives",
                text=[f"center_{i}" for i in range(len(pts))],
                marker=dict(size=7, color="blue", symbol="x"),
            )
        )
    fig.add_trace(
        go.Scatter3d(
            x=keypoints[:, 0],
            y=keypoints[:, 1],
            z=keypoints[:, 2],
            mode="markers",
            name="MANO keypoints",
            marker=dict(size=3, color="rgba(20,20,20,0.55)"),
        )
    )
    fig.update_layout(
        title="Synthetic MANO Contact Clustering Test",
        scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    vertices, faces, keypoints = create_random_mano(
        args.mano_root,
        seed=args.seed,
        pose_scale=args.pose_scale,
    )
    samples = sample_mano_surface(
        vertices,
        faces,
        num_points=args.num_surface_points,
        seed=args.seed,
        oversample_factor=args.oversample_factor,
    )
    finger_assignments = assign_samples_to_fingers(samples.points, keypoints)
    contact_indices, contact_points, seed_indices = build_synthetic_contact_set(
        samples.points,
        finger_assignments,
        patch_radius=args.patch_radius,
        min_seed_distance=args.min_seed_distance,
        rng=rng,
    )
    seed_points = np.stack([samples.points[index] for index in seed_indices.values()], axis=0)
    labels = radius_connected_components(
        contact_points,
        radius=args.cluster_radius,
        min_cluster_size=args.min_cluster_size,
    )
    nearest_reps, center_reps = representative_indices(contact_points, labels, seed_points)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_npz = output_dir / f"{args.output_prefix}.npz"
    output_html = output_dir / f"{args.output_prefix}.html"
    np.savez_compressed(
        output_npz,
        vertices=vertices.astype(np.float32),
        faces=faces.astype(np.int64),
        keypoints_3d=keypoints.astype(np.float32),
        surface_points=samples.points.astype(np.float32),
        surface_normals=samples.normals.astype(np.float32),
        surface_face_indices=samples.face_indices.astype(np.int64),
        surface_barycentric=samples.barycentric.astype(np.float32),
        contact_indices=contact_indices.astype(np.int64),
        contact_points=contact_points.astype(np.float32),
        cluster_labels=labels.astype(np.int64),
        seed_indices=np.asarray(list(seed_indices.values()), dtype=np.int64),
        seed_fingers=np.asarray(list(seed_indices.keys())),
        nearest_rep_indices=np.asarray(nearest_reps, dtype=np.int64),
        center_rep_indices=np.asarray(center_reps, dtype=np.int64),
        nearest_rep_sample_indices=contact_indices[np.asarray(nearest_reps, dtype=np.int64)].astype(np.int64),
        center_rep_sample_indices=contact_indices[np.asarray(center_reps, dtype=np.int64)].astype(np.int64),
    )
    fig = make_figure(
        vertices=vertices,
        faces=faces,
        keypoints=keypoints,
        surface_points=samples.points,
        contact_points=contact_points,
        labels=labels,
        seed_points=seed_points,
        nearest_reps=nearest_reps,
        center_reps=center_reps,
    )
    fig.write_html(output_html, include_plotlyjs="cdn")

    print(f"surface samples: {samples.points.shape}")
    print(f"synthetic contact points: {contact_points.shape}")
    print(f"cluster labels: {sorted(int(x) for x in np.unique(labels))}")
    print(f"num valid clusters: {sum(1 for x in np.unique(labels) if x >= 0)}")
    print(f"seed fingers: {list(seed_indices.keys())}")
    print(f"saved data: {output_npz}")
    print(f"visualization: {output_html}")
    if args.show:
        fig.show()


if __name__ == "__main__":
    main()
