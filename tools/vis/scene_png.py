#!/usr/bin/env python3
"""Render one HOI4D hand-object grasp frame.

This script uses MANO/manopth when available. If the environment is not ready
yet, it falls back to a simple hand-like skeleton so the object pose pipeline
can still be checked.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation


DEFAULT_SEQUENCE = Path("extracted_dataset_sampled/ZY20210800004/H4/C5/N26/S113/s01/T1")
DEFAULT_FRAME = "74"
DEFAULT_CAD = Path(
    "extracted_dataset_sampled/cad_models/HOI4D_CAD_Model_for_release/rigid/Bottle/026.obj"
)
DEFAULT_HAND = Path(
    "extracted_dataset_sampled/Hand_pose/handpose_right_hand/ZY20210800004/H4/C5/N26/S113/s01/T1/74.pickle"
)
DEFAULT_MANO_ROOT = Path("mano/mano")


@dataclass(frozen=True)
class ManoSceneResult:
    vertices: np.ndarray
    faces: np.ndarray
    keypoints: np.ndarray
    side: str


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    faces = []
    with path.open("r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                idx = []
                for token in line.strip().split()[1:]:
                    idx.append(int(token.split("/")[0]) - 1)
                if len(idx) >= 3:
                    for i in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[i], idx[i + 1]])
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def transform_object(vertices: np.ndarray, obj: dict) -> np.ndarray:
    center = obj["center"]
    dims = obj["dimensions"]
    rot = obj["rotation"]
    target_center = np.array([center["x"], center["y"], center["z"]], dtype=float)
    target_dims = np.array([dims["length"], dims["width"], dims["height"]], dtype=float)

    local = vertices.copy()
    bbox_min = local.min(axis=0)
    bbox_max = local.max(axis=0)
    bbox_center = (bbox_min + bbox_max) / 2.0
    bbox_dims = np.maximum(bbox_max - bbox_min, 1e-9)
    local = (local - bbox_center) * (target_dims / bbox_dims)

    rotation = Rotation.from_euler("XYZ", [rot["x"], rot["y"], rot["z"]]).as_matrix()
    return local @ rotation.T + target_center


def load_objpose(path: Path) -> dict:
    data = json.loads(path.read_text())
    objects = data.get("dataList") or data.get("objects") or []
    if not objects:
        raise ValueError(f"No objects found in {path}")
    return objects[0]


def load_hand(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f, encoding="latin1")


def make_fallback_hand(hand: dict, obj_center: np.ndarray, obj_dims: np.ndarray) -> list[np.ndarray]:
    """Create a simple 3D hand-like skeleton near the object.

    This is not a MANO reconstruction. It only visualizes the hand translation
    and a plausible grasp direction when MANO model files are unavailable.
    """
    wrist = np.asarray(hand["trans"], dtype=float)
    to_obj = obj_center - wrist
    to_obj = to_obj / (np.linalg.norm(to_obj) + 1e-9)
    up = np.array([0.0, 0.0, 1.0])
    side = np.cross(to_obj, up)
    if np.linalg.norm(side) < 1e-6:
        side = np.array([1.0, 0.0, 0.0])
    side = side / np.linalg.norm(side)
    up = np.cross(side, to_obj)
    up = up / (np.linalg.norm(up) + 1e-9)

    palm = wrist + 0.035 * to_obj
    radius = float(max(obj_dims[0], obj_dims[1]) * 0.55)
    length = float(obj_dims[2] * 0.55)

    fingers: list[np.ndarray] = []
    offsets = [-0.028, -0.014, 0.0, 0.014, 0.028]
    for i, off in enumerate(offsets):
        base = palm + off * side + (0.018 if i == 0 else 0.0) * up
        tip = obj_center + off * 0.65 * side - radius * to_obj + (0.02 - 0.01 * abs(i - 2)) * up
        mid1 = base + 0.45 * (tip - base) + 0.025 * up
        mid2 = base + 0.75 * (tip - base) + 0.010 * up
        fingers.append(np.vstack([wrist, base, mid1, mid2, tip]))

    # Add a thumb crossing from the side.
    thumb_base = palm - 0.04 * side - 0.005 * up
    thumb_tip = obj_center + 0.04 * side - radius * to_obj - 0.01 * up
    fingers.append(np.vstack([wrist, thumb_base, (thumb_base + thumb_tip) / 2 + 0.02 * up, thumb_tip]))
    return fingers


def mano_side_from_path(path: Path) -> str:
    lowered = str(path).lower()
    if "left" in lowered:
        return "left"
    return "right"


def try_make_mano_scene(hand: dict, hand_path: Path, mano_root: Path) -> ManoSceneResult | None:
    try:
        # chumpy/manopth still import removed NumPy aliases such as np.int.
        # Restore them before importing manopth so newer NumPy versions work.
        for alias, value in {
            "bool": bool,
            "int": int,
            "float": float,
            "complex": complex,
            "object": object,
            "unicode": str,
            "str": str,
        }.items():
            if not hasattr(np, alias):
                setattr(np, alias, value)
        import torch
        from manopth.manolayer import ManoLayer
    except Exception as exc:
        return None

    side = mano_side_from_path(hand_path)
    model_file = mano_root / f"MANO_{side.upper()}.pkl"
    if not model_file.exists():
        return None

    theta = torch.as_tensor(hand["poseCoeff"], dtype=torch.float32).unsqueeze(0)
    beta = torch.as_tensor(hand["beta"], dtype=torch.float32).unsqueeze(0)
    trans = torch.as_tensor(hand["trans"], dtype=torch.float32).view(1, 1, 3)

    layer = ManoLayer(
        mano_root=str(mano_root),
        use_pca=False,
        ncomps=45,
        flat_hand_mean=True,
        side=side,
    )
    with torch.no_grad():
        verts, joints = layer(theta, beta)
    verts_np = (verts[0] / 1000.0 + trans[0]).detach().cpu().numpy()
    joints_np = (joints[0] / 1000.0 + trans[0]).detach().cpu().numpy()
    faces = layer.th_faces.detach().cpu().numpy()
    return ManoSceneResult(vertices=verts_np, faces=faces, keypoints=joints_np, side=side)


def try_make_mano_mesh(hand: dict, hand_path: Path, mano_root: Path) -> tuple[np.ndarray, np.ndarray, str] | None:
    mano = try_make_mano_scene(hand, hand_path, mano_root)
    if mano is None:
        return None
    return mano.vertices, mano.faces, mano.side


def set_equal_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0) * 1.25
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def render_scene(sequence: Path, frame: str, cad: Path, hand_path: Path, output: Path, mano_root: Path) -> None:
    objpose = load_objpose(sequence / "objpose" / f"{frame}.json")
    hand = load_hand(hand_path)
    vertices, faces = load_obj(cad)
    object_vertices = transform_object(vertices, objpose)

    dims = objpose["dimensions"]
    obj_dims = np.array([dims["length"], dims["width"], dims["height"]], dtype=float)
    center = objpose["center"]
    obj_center = np.array([center["x"], center["y"], center["z"]], dtype=float)
    mano_mesh = try_make_mano_mesh(hand, hand_path, mano_root)
    hand_lines = [] if mano_mesh else make_fallback_hand(hand, obj_center, obj_dims)

    output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(object_vertices[faces], alpha=0.82, linewidths=0.05)
    mesh.set_facecolor((0.72, 0.72, 0.68, 1.0))
    mesh.set_edgecolor((0.35, 0.35, 0.35, 0.18))
    ax.add_collection3d(mesh)

    if mano_mesh:
        hand_vertices, hand_faces, side = mano_mesh
        hand_collection = Poly3DCollection(hand_vertices[hand_faces], alpha=0.88, linewidths=0.02)
        hand_collection.set_facecolor((0.06, 0.58, 0.22, 1.0))
        hand_collection.set_edgecolor((0.02, 0.28, 0.10, 0.08))
        ax.add_collection3d(hand_collection)
    else:
        side = "fallback"
        for line in hand_lines:
            ax.plot(line[:, 0], line[:, 1], line[:, 2], color="#16833a", linewidth=7, solid_capstyle="round")
            ax.scatter(line[:, 0], line[:, 1], line[:, 2], color="#1ea64b", s=45, depthshade=True)

    all_points = np.vstack([object_vertices, mano_mesh[0]] if mano_mesh else [object_vertices] + hand_lines)
    set_equal_axes(ax, all_points)
    ax.view_init(elev=18, azim=-65)
    ax.set_axis_off()
    ax.set_title(f"HOI4D frame {frame}: Bottle CAD + {side} hand", fontsize=11)
    plt.tight_layout()
    fig.savefig(output, dpi=180, transparent=False)
    plt.close(fig)

    meta = {
        "sequence": str(sequence),
        "frame": frame,
        "cad": str(cad),
        "hand_pickle": str(hand_path),
        "objpose": str(sequence / "objpose" / f"{frame}.json"),
        "mano_root": str(mano_root),
        "hand_render_mode": "mano_mesh" if mano_mesh else "fallback_skeleton",
        "hand_note": "MANO mesh is used when torch/manopth and MANO model files are available.",
        "mano_fields": {key: list(np.asarray(value).shape) for key, value in hand.items()},
    }
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--frame", default=DEFAULT_FRAME)
    parser.add_argument("--cad", type=Path, default=DEFAULT_CAD)
    parser.add_argument("--hand", type=Path, default=DEFAULT_HAND)
    parser.add_argument("--mano-root", type=Path, default=DEFAULT_MANO_ROOT)
    parser.add_argument("--output", type=Path, default=Path("outputs/grasp_scene_bottle_frame74.png"))
    args = parser.parse_args()
    render_scene(args.sequence, args.frame, args.cad, args.hand, args.output, args.mano_root)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
