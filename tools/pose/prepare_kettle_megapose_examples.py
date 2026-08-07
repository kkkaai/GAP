#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "outputs/0713test_phase1_4_vlm_qwen37_test"
MEGAPOSE_EXAMPLES = REPO_ROOT / "external/megapose6d/local_data/examples"
FOUNDATION_ASSETS = OUTPUT_ROOT / "_pose_foundationpose_assets"
FOUNDATION_MESH = FOUNDATION_ASSETS / "kettle-decimated-100k-scale0.2.obj"

SAMPLES = [
    "20260713_172042_054_kettle-1",
    "20260713_172116_649_kettle-2",
    "20260713_172149_129_kettle-3",
]

K = [[615.0, 0.0, 320.0], [0.0, 615.0, 240.0], [0.0, 0.0, 1.0]]


def bbox_from_mask(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask > 0)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def ensure_mm_mesh(dst: Path) -> None:
    mesh = trimesh.load(FOUNDATION_MESH, force="mesh", process=False)
    mesh_mm = mesh.copy()
    mesh_mm.apply_scale(1000.0)
    dst.parent.mkdir(parents=True, exist_ok=True)
    mesh_mm.export(dst)


def main() -> None:
    if not FOUNDATION_MESH.exists():
        raise FileNotFoundError(f"Run FoundationPose preparation first; missing {FOUNDATION_MESH}")

    prepared = []
    for sample in SAMPLES:
        source_dir = REPO_ROOT / "0713test" / sample
        mask = cv2.imread(str(source_dir / "object_mask.png"), cv2.IMREAD_GRAYSCALE)
        depth = cv2.imread(str(source_dir / "depth.png"), cv2.IMREAD_UNCHANGED)
        bbox = bbox_from_mask(mask)

        for ldir in sorted((OUTPUT_ROOT / sample).glob("lollipop_*")):
            rgb_file = ldir / "phase4_inpaint_full.png"
            if not rgb_file.exists():
                continue
            example_name = f"{sample}_{ldir.name}_kettle"
            ex = MEGAPOSE_EXAMPLES / example_name
            if ex.exists():
                shutil.rmtree(ex)
            (ex / "inputs").mkdir(parents=True, exist_ok=True)
            (ex / "meshes" / "kettle").mkdir(parents=True, exist_ok=True)

            rgb = imageio.imread(rgb_file)[..., :3]
            imageio.imwrite(ex / "image_rgb.png", rgb)
            imageio.imwrite(ex / "image_depth.png", depth.astype(np.uint16))
            ensure_mm_mesh(ex / "meshes" / "kettle" / "kettle.obj")

            camera_data = {
                "K": K,
                "resolution": [int(rgb.shape[0]), int(rgb.shape[1])],
            }
            (ex / "camera_data.json").write_text(json.dumps(camera_data, indent=2), encoding="utf-8")
            object_data = [
                {
                    "label": "kettle",
                    "bbox_modal": bbox,
                    "bbox_amodal": bbox,
                }
            ]
            (ex / "inputs" / "object_data.json").write_text(
                json.dumps(object_data, indent=2), encoding="utf-8"
            )

            prepared.append(
                {
                    "example_name": example_name,
                    "example_dir": str(ex),
                    "rgb_file": str(rgb_file),
                    "depth_file": str(source_dir / "depth.png"),
                    "mask_file": str(source_dir / "object_mask.png"),
                    "mesh_file": str(ex / "meshes" / "kettle" / "kettle.obj"),
                    "mesh_units": "mm",
                    "bbox_modal": bbox,
                    "camera_K": K,
                }
            )

    out = OUTPUT_ROOT / "kettle_megapose_examples_summary.json"
    out.write_text(json.dumps(prepared, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Prepared {len(prepared)} MegaPose examples")
    print(out)


if __name__ == "__main__":
    main()
