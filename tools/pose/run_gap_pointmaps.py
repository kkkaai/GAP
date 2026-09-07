#!/usr/bin/env python3
"""Generate MoGe pointmaps for GAP phase4 images.

Run this inside the pointmap environment, e.g.:
  conda activate gap-dai-pointmap
  python tools/pose/run_gap_pointmaps.py --kettle12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]

KETTLE_SAMPLE_IDS = [
    "20260713_172042_054_kettle-1",
    "20260713_172116_649_kettle-2",
    "20260713_172149_129_kettle-3",
]


def collect_kettle12(output_root: Path) -> list[Path]:
    images: list[Path] = []
    for sample_id in KETTLE_SAMPLE_IDS:
        for lollipop_dir in sorted((output_root / sample_id).glob("lollipop_*")):
            image = lollipop_dir / "phase4_inpaint_full.png"
            if image.exists():
                images.append(image)
    return images


def load_moge_model(device: torch.device):
    from moge.model.v1 import MoGeModel

    model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device).eval()
    return model


def run_pointmap(model, image_path: Path, output_path: Path, device: torch.device) -> None:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_t = torch.as_tensor(image_rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            output = model.infer(image_t)
    points = output["points"].detach().cpu().numpy()
    intrinsics = output["intrinsics"].detach().cpu().numpy().copy()
    h, w = image_rgb.shape[:2]
    intrinsics[0, 0] *= w
    intrinsics[1, 1] *= h
    intrinsics[0, 2] *= w
    intrinsics[1, 2] *= h

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, points)
    np.save(output_path.with_name(output_path.name.replace("_pointmap.npy", "_intrinsics.npy")), intrinsics)
    txt_path = output_path.with_name(output_path.name.replace("_pointmap.npy", "_intrinsics.txt"))
    txt_path.write_text(
        f"{intrinsics[0, 0]}\n{intrinsics[1, 1]}\n{intrinsics[0, 2]}\n{intrinsics[1, 2]}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", action="append", default=[], help="Image path. Can be repeated.")
    parser.add_argument("--image-list", default=None, help="Text file with one image path per line.")
    parser.add_argument("--kettle12", action="store_true", help="Use the 12 generated kettle phase4 images.")
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "outputs/0713test_phase1_4_vlm_qwen37_test"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    images = [Path(p).expanduser().resolve() for p in args.image]
    if args.image_list:
        images.extend(
            Path(line.strip()).expanduser().resolve()
            for line in Path(args.image_list).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if args.kettle12:
        images.extend(collect_kettle12(Path(args.output_root).expanduser().resolve()))
    images = sorted(dict.fromkeys(images))
    if not images:
        raise SystemExit("No images selected.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading MoGe on {device}...")
    model = None if args.dry_run else load_moge_model(device)
    for idx, image in enumerate(images, start=1):
        out_dir = image.parent / "pointmap"
        out_dir.mkdir(exist_ok=True)
        output = out_dir / f"{image.stem}_pointmap.npy"
        intrinsics = out_dir / f"{image.stem}_intrinsics.txt"
        if output.exists() and intrinsics.exists() and not args.overwrite:
            print(f"[{idx:02d}/{len(images):02d}] skip existing {output}")
            continue
        print(f"[{idx:02d}/{len(images):02d}] {image} -> {output}")
        if args.dry_run:
            continue
        run_pointmap(model, image, output, device)


if __name__ == "__main__":
    main()
