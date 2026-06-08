from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prosthetic_grasp.common.io import load_image
from prosthetic_grasp.common.segformer_distill import (
    image_stem,
    iter_image_files,
    save_mask,
    save_overlay,
    set_seed,
)
from prosthetic_grasp.phases.phase1_mask import Phase1Mask, Phase1MaskConfig


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Generate SegGPT pseudo labels for SegFormer mechanical-hand distillation.")
    parser.add_argument("--image-dir", required=True, help="Directory containing raw training images.")
    parser.add_argument("--output-mask-dir", required=True, help="Directory to save binary pseudo masks.")
    parser.add_argument("--overlay-dir", default="", help="Optional directory for mask overlay QA images.")
    parser.add_argument("--recursive", action="store_true", help="Scan image-dir recursively.")
    parser.add_argument("--model-id", default="external/models/seggpt-vit-large")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--support-mask-threshold", type=int, default=127)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--support-image-path", default="")
    parser.add_argument("--support-mask-path", default="")
    parser.add_argument("--support-image-paths", nargs="+", default=None)
    parser.add_argument("--support-mask-paths", nargs="+", default=None)
    parser.add_argument("--write-splits", action="store_true", help="Write train/val split files from generated masks.")
    parser.add_argument("--split-dir", default="", help="Directory to save train.txt and val.txt. Defaults to image-dir parent.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-images", type=int, default=0, help="Only process the first N images. 0 means all images.")
    return parser


def write_splits(stamps: list[str], split_dir: Path, val_ratio: float, seed: int) -> None:
    rng = random.Random(seed)
    stamps = list(stamps)
    rng.shuffle(stamps)
    val_count = max(1, int(round(len(stamps) * val_ratio))) if len(stamps) > 1 else 0
    val_stamps = sorted(stamps[:val_count])
    train_stamps = sorted(stamps[val_count:])
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train.txt").write_text("\n".join(train_stamps) + ("\n" if train_stamps else ""), encoding="utf-8")
    (split_dir / "val.txt").write_text("\n".join(val_stamps) + ("\n" if val_stamps else ""), encoding="utf-8")
    print(f"Wrote splits: {split_dir / 'train.txt'} ({len(train_stamps)}), {split_dir / 'val.txt'} ({len(val_stamps)})")


def main() -> None:
    args = build_argparser().parse_args()
    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError(f"val-ratio must be in [0, 1), got {args.val_ratio}.")
    set_seed(args.seed)

    image_paths = iter_image_files(args.image_dir, recursive=args.recursive)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.image_dir}.")
    if args.max_images < 0:
        raise ValueError(f"max-images must be >= 0, got {args.max_images}.")
    if args.max_images:
        image_paths = image_paths[: args.max_images]

    extractor = Phase1Mask(
        Phase1MaskConfig(
            mode="seggpt",
            model_id=args.model_id,
            support_image_path=args.support_image_path or None,
            support_mask_path=args.support_mask_path or None,
            support_image_paths=args.support_image_paths,
            support_mask_paths=args.support_mask_paths,
            support_mask_threshold=args.support_mask_threshold,
            threshold=args.threshold,
            device=args.device,
        )
    )

    output_mask_dir = Path(args.output_mask_dir)
    overlay_dir = Path(args.overlay_dir) if args.overlay_dir else None
    generated_stamps: list[str] = []
    for index, image_path in enumerate(image_paths, start=1):
        stamp = image_stem(image_path)
        mask_path = output_mask_dir / f"{stamp}.png"
        if args.skip_existing and mask_path.exists():
            generated_stamps.append(stamp)
            print(f"[{index}/{len(image_paths)}] skip existing {mask_path}")
            continue
        image_rgb = load_image(image_path)
        result = extractor.run(image_rgb)
        save_mask(mask_path, result.mask)
        if overlay_dir is not None:
            save_overlay(overlay_dir / f"{stamp}.png", image_rgb, result.mask)
        generated_stamps.append(stamp)
        fg_ratio = float(np.mean(result.mask))
        print(f"[{index}/{len(image_paths)}] {image_path} -> {mask_path} fg_ratio={fg_ratio:.4f}")

    if args.write_splits:
        split_dir = Path(args.split_dir) if args.split_dir else Path(args.image_dir).parent
        write_splits(generated_stamps, split_dir, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
