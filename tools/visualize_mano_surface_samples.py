from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prosthetic_grasp.geometry import sample_mano_surface, write_mano_surface_samples_html


MANO_TIP_VERTEX_IDS = np.array([744, 320, 443, 554, 671], dtype=np.int64)


def patch_numpy_legacy_aliases() -> None:
    """Provide removed NumPy aliases needed by chumpy-loaded MANO pickles."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample MANO surface points and visualize them as an interactive HTML file."
    )
    parser.add_argument(
        "--phase5-npz",
        default="",
        help="Optional pipeline artifact, e.g. outputs/.../phase5_mano.npz.",
    )
    parser.add_argument("--hand-index", type=int, default=0)
    parser.add_argument("--mano-root", default="models", help="Fallback MANO model root for synthetic geometry.")
    parser.add_argument("--pose", choices=["zero", "random"], default="zero")
    parser.add_argument("--pose-scale", type=float, default=0.45)
    parser.add_argument("--num-points", type=int, default=2000)
    parser.add_argument("--oversample-factor", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-fps", action="store_true", help="Disable farthest-point downsampling.")
    parser.add_argument("--output-dir", default="outputs/mano_surface_samples")
    parser.add_argument("--output-prefix", default="mano_surface")
    parser.add_argument("--show", action="store_true", help="Open an interactive Plotly window after writing HTML.")
    return parser.parse_args()


def load_phase5_npz(path: str | Path, hand_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, str]:
    npz_path = Path(path)
    if not npz_path.exists():
        raise FileNotFoundError(f"phase5 npz does not exist: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    faces = np.asarray(data["faces"], dtype=np.int64)
    prefix = f"hand_{hand_index}"
    vertices_key = f"{prefix}_vertices"
    if vertices_key not in data:
        hand_keys = sorted(k for k in data.files if k.endswith("_vertices"))
        raise KeyError(f"{vertices_key!r} not found in {npz_path}. Available vertex keys: {hand_keys}")
    vertices = np.asarray(data[vertices_key], dtype=np.float64)
    keypoints_key = f"{prefix}_keypoints_3d"
    keypoints = np.asarray(data[keypoints_key], dtype=np.float64) if keypoints_key in data else None
    return vertices, faces, keypoints, f"phase5 hand {hand_index}"

def create_synthetic_mano(
    mano_root: str,
    pose: str,
    seed: int,
    pose_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    patch_numpy_legacy_aliases()
    try:
        import smplx
        import torch
    except ImportError as exc:
        raise ImportError(
            "Synthetic MANO fallback requires smplx and torch. Pass --phase5-npz "
            "to visualize an existing pipeline artifact instead."
        ) from exc

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
    faces = np.asarray(model.faces, dtype=np.int64)
    return vertices, faces, keypoints, f"synthetic {pose} MANO"


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


def main() -> None:
    args = parse_args()
    if args.phase5_npz:
        vertices, faces, keypoints, source_name = load_phase5_npz(args.phase5_npz, args.hand_index)
    else:
        vertices, faces, keypoints, source_name = create_synthetic_mano(
            args.mano_root,
            args.pose,
            args.seed,
            args.pose_scale,
        )

    samples = sample_mano_surface(
        vertices,
        faces,
        num_points=args.num_points,
        seed=args.seed,
        use_farthest_point_sampling=not args.no_fps,
        oversample_factor=args.oversample_factor,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_npz = output_dir / f"{args.output_prefix}_samples.npz"
    output_html = output_dir / f"{args.output_prefix}_samples.html"
    np.savez_compressed(
        output_npz,
        points=samples.points.astype(np.float32),
        normals=samples.normals.astype(np.float32),
        face_indices=samples.face_indices.astype(np.int64),
        barycentric=samples.barycentric.astype(np.float32),
        vertices=vertices.astype(np.float32),
        faces=faces.astype(np.int64),
        keypoints_3d=keypoints.astype(np.float32) if keypoints is not None else np.zeros((0, 3), dtype=np.float32),
    )
    write_mano_surface_samples_html(
        output_html,
        vertices,
        faces,
        samples,
        keypoints=keypoints,
        title=f"MANO Surface Samples - {source_name}",
    )

    print(f"source: {source_name}")
    print(f"vertices: {vertices.shape}, faces: {faces.shape}")
    print(f"samples: {samples.points.shape}")
    print(f"saved samples: {output_npz}")
    print(f"visualization: {output_html}")

    if args.show:
        from prosthetic_grasp.geometry.mano_surface import make_mano_surface_samples_figure

        fig = make_mano_surface_samples_figure(
            vertices=vertices,
            faces=faces,
            samples=samples,
            keypoints=keypoints,
            title=f"MANO Surface Samples - {source_name}",
        )
        fig.show()


if __name__ == "__main__":
    main()
