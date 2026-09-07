#!/usr/bin/env python3
"""Generate hand masks by rasterizing recovered HaMeR meshes into the image.

These masks are not semantic hand segmentations. They are mesh-consistent
silhouettes used as a stable fallback for phase5.5 hand-anchored pointmap
alignment when Detectron2/ViTPose/SAM hand masks are unavailable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]

KETTLE_SAMPLE_IDS = [
    "20260713_172042_054_kettle-1",
    "20260713_172116_649_kettle-2",
    "20260713_172149_129_kettle-3",
]


def load_intrinsics(path: Path, fallback: np.ndarray) -> np.ndarray:
    if not path.exists():
        return fallback.astype(np.float64)
    if path.suffix == ".npy":
        return np.load(path).astype(np.float64)
    vals = [float(x) for x in path.read_text(encoding="utf-8").split()]
    if len(vals) == 4:
        fx, fy, cx, cy = vals
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 9:
        return arr.reshape(3, 3)
    raise ValueError(f"Cannot parse intrinsics: {path}")


def load_faces(mesh_path: Path) -> np.ndarray:
    mesh = trimesh.load_mesh(str(mesh_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return np.asarray(mesh.faces, dtype=np.int64)


def project(vertices: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = vertices[:, 2]
    valid = z > 1e-8
    u = k[0, 0] * vertices[:, 0] / z + k[0, 2]
    v = k[1, 1] * vertices[:, 1] / z + k[1, 2]
    return u, v, valid


def rasterize_mask(
    vertices: np.ndarray,
    faces: np.ndarray,
    k: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    u, v, valid_vertices = project(vertices, k)
    face_valid = valid_vertices[faces].all(axis=1)
    face_depth = vertices[faces, 2].mean(axis=1)
    order = np.argsort(face_depth)[::-1]
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for face_idx in order:
        if not face_valid[face_idx]:
            continue
        face = faces[face_idx]
        poly = [(float(u[j]), float(v[j])) for j in face]
        if all((x < -2 or x > width + 2 or y < -2 or y > height + 2) for x, y in poly):
            continue
        draw.polygon(poly, fill=255)
    return np.asarray(mask)


def write_overlay(rgb_path: Path, mask: np.ndarray, output_path: Path) -> None:
    image = Image.open(rgb_path).convert("RGB")
    rgb = np.asarray(image).copy()
    m = mask > 0
    rgb[m] = (rgb[m].astype(np.float32) * 0.55 + np.asarray([255, 180, 0], dtype=np.float32) * 0.45).astype(np.uint8)
    Image.fromarray(rgb).save(output_path)


def collect_lollipop_dirs(output_root: Path) -> list[Path]:
    dirs: list[Path] = []
    for sample_id in KETTLE_SAMPLE_IDS:
        dirs.extend(sorted((output_root / sample_id).glob("lollipop_*")))
    return [d for d in dirs if (d / "phase4_inpaint_full.png").exists()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "outputs/0713test_phase1_4_vlm_qwen37_test"),
    )
    parser.add_argument("--hamer-subdir", default="pose_hamer_official")
    parser.add_argument("--fx", type=float, default=615.0)
    parser.add_argument("--fy", type=float, default=615.0)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    fallback_k = np.array([[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dirs = collect_lollipop_dirs(output_root)
    if not dirs:
        raise SystemExit("No lollipop dirs found.")

    count = 0
    for lollipop_dir in dirs:
        hamer_dir = lollipop_dir / args.hamer_subdir
        vertices_path = hamer_dir / "hand_00_vertices_camera.npy"
        mesh_path = hamer_dir / "hand_00_camera.obj"
        rgb_path = lollipop_dir / "phase4_inpaint_full.png"
        if not vertices_path.exists() or not mesh_path.exists():
            print(f"skip missing HaMeR files: {lollipop_dir}")
            continue
        intr_txt = lollipop_dir / "pointmap" / "phase4_inpaint_full_intrinsics.txt"
        intr_npy = lollipop_dir / "pointmap" / "phase4_inpaint_full_intrinsics.npy"
        k = load_intrinsics(intr_txt if intr_txt.exists() else intr_npy, fallback_k)
        image = Image.open(rgb_path)
        width, height = image.size
        vertices = np.load(vertices_path).astype(np.float64)
        faces = load_faces(mesh_path)
        mask = rasterize_mask(vertices, faces, k, width, height)
        mask_path = hamer_dir / "hand_00_projected_mask.png"
        overlay_path = hamer_dir / "hand_00_projected_mask_overlay.png"
        Image.fromarray(mask).save(mask_path)
        write_overlay(rgb_path, mask, overlay_path)
        count += 1
        print(f"[{count:02d}] {mask_path}")
    print(f"Generated {count} projected hand masks.")


if __name__ == "__main__":
    sys.exit(main())
