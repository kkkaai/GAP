from __future__ import annotations

import argparse
import random
import subprocess
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
    parser = argparse.ArgumentParser(
        "Build SegGPT pseudo labels and train SegFormer for the 2026.6.6 RealSense mechanical-hand dataset."
    )
    parser.add_argument("--data-root", default="2026.6.6 realsense")
    parser.add_argument("--support-image-dir", default="seggpt-pre")
    parser.add_argument("--support-mask-dir", default="mask-seggpt")
    parser.add_argument("--segformer-image-dir", default="segformer")
    parser.add_argument("--pseudo-mask-dir", default="segformer-seggpt-masks")
    parser.add_argument("--overlay-dir", default="segformer-seggpt-overlays")
    parser.add_argument("--split-dir", default="", help="Defaults to data-root.")
    parser.add_argument("--train-split-name", default="segformer_train.txt")
    parser.add_argument("--val-split-name", default="segformer_val.txt")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing-masks", action="store_true")
    parser.add_argument("--skip-pseudo-labels", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-pseudo-images",
        type=int,
        default=0,
        help="Only process the first N SegFormer images when generating pseudo labels. 0 means all images.",
    )

    parser.add_argument("--seggpt-model-id", default="external/models/seggpt-vit-large")
    parser.add_argument("--seggpt-threshold", type=float, default=0.5)
    parser.add_argument("--support-mask-threshold", type=int, default=127)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--segformer-model-name", default="external/models/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--img-size", type=int, default=448)
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--class-weight-fg", type=float, default=2.0)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--run-name", default="realsense_mechhand")
    parser.add_argument("--output-root", default="weights")
    parser.add_argument("--color-jitter", action="store_true", default=True)
    return parser


def resolve_under_root(root: Path, path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return root / candidate


def discover_support_pairs(support_image_dir: Path, support_mask_dir: Path) -> tuple[list[str], list[str]]:
    support_images = iter_image_files(support_image_dir)
    image_paths: list[str] = []
    mask_paths: list[str] = []
    for image_path in support_images:
        mask_path = support_mask_dir / f"{image_path.stem}.png"
        if mask_path.exists():
            image_paths.append(str(image_path))
            mask_paths.append(str(mask_path))
    if not image_paths:
        raise FileNotFoundError(
            f"No support image/mask pairs found. Images: {support_image_dir}, masks: {support_mask_dir}."
        )
    return image_paths, mask_paths


def mask_foreground_ratio(mask_path: str | Path, threshold: int) -> float:
    array = np.array(Image.open(mask_path).convert("L"))
    return float(np.mean(array > threshold))


def write_splits(
    stamps: list[str],
    split_dir: Path,
    *,
    train_split_name: str,
    val_split_name: str,
    val_ratio: float,
    seed: int,
) -> tuple[Path, Path]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val-ratio must be in [0, 1), got {val_ratio}.")
    rng = random.Random(seed)
    shuffled = list(stamps)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio))) if len(shuffled) > 1 else 0
    val_stamps = sorted(shuffled[:val_count])
    train_stamps = sorted(shuffled[val_count:])
    split_dir.mkdir(parents=True, exist_ok=True)
    train_path = split_dir / train_split_name
    val_path = split_dir / val_split_name
    train_path.write_text("\n".join(train_stamps) + ("\n" if train_stamps else ""), encoding="utf-8")
    val_path.write_text("\n".join(val_stamps) + ("\n" if val_stamps else ""), encoding="utf-8")
    return train_path, val_path


def generate_pseudo_labels(
    *,
    image_paths: list[Path],
    output_mask_dir: Path,
    overlay_dir: Path,
    support_image_paths: list[str],
    support_mask_paths: list[str],
    support_mask_threshold: int,
    model_id: str,
    threshold: float,
    device: str,
    skip_existing: bool,
) -> list[str]:
    extractor = Phase1Mask(
        Phase1MaskConfig(
            mode="seggpt",
            model_id=model_id,
            support_image_paths=support_image_paths,
            support_mask_paths=support_mask_paths,
            support_mask_threshold=support_mask_threshold,
            threshold=threshold,
            device=device,
        )
    )
    stamps: list[str] = []
    for index, image_path in enumerate(image_paths, start=1):
        stamp = image_stem(image_path)
        mask_path = output_mask_dir / f"{stamp}.png"
        if skip_existing and mask_path.exists():
            stamps.append(stamp)
            print(f"[SegGPT {index}/{len(image_paths)}] skip existing {mask_path}")
            continue
        image_rgb = load_image(image_path)
        result = extractor.run(image_rgb)
        save_mask(mask_path, result.mask)
        save_overlay(overlay_dir / f"{stamp}.png", image_rgb, result.mask)
        stamps.append(stamp)
        print(
            f"[SegGPT {index}/{len(image_paths)}] {image_path.name} -> {mask_path.name} "
            f"fg_ratio={float(np.mean(result.mask)):.4f}"
        )
    return stamps


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)

    data_root = Path(args.data_root)
    support_image_dir = resolve_under_root(data_root, args.support_image_dir)
    support_mask_dir = resolve_under_root(data_root, args.support_mask_dir)
    segformer_image_dir = resolve_under_root(data_root, args.segformer_image_dir)
    pseudo_mask_dir = resolve_under_root(data_root, args.pseudo_mask_dir)
    overlay_dir = resolve_under_root(data_root, args.overlay_dir)
    split_dir = resolve_under_root(data_root, args.split_dir) if args.split_dir else data_root

    support_image_paths, support_mask_paths = discover_support_pairs(support_image_dir, support_mask_dir)
    image_paths = iter_image_files(segformer_image_dir)
    if not image_paths:
        raise FileNotFoundError(f"No SegFormer training images found in {segformer_image_dir}.")
    if args.max_pseudo_images < 0:
        raise ValueError(f"max-pseudo-images must be >= 0, got {args.max_pseudo_images}.")
    if args.max_pseudo_images:
        image_paths = image_paths[: args.max_pseudo_images]

    print(f"Data root: {data_root}")
    print(f"Support pairs: {len(support_image_paths)}")
    for image_path, mask_path in zip(support_image_paths, support_mask_paths):
        fg_ratio = mask_foreground_ratio(mask_path, args.support_mask_threshold)
        print(f"  support image={image_path} | mask={mask_path} | support_fg_ratio={fg_ratio:.4f}")
    print(f"SegFormer images: {len(image_paths)} from {segformer_image_dir}")
    print(f"Pseudo masks: {pseudo_mask_dir}")
    print(f"Overlays: {overlay_dir}")

    if args.dry_run:
        print("Dry run only. No pseudo labels, splits, or training were generated.")
        return

    if args.skip_pseudo_labels:
        stamps = [image_stem(path) for path in image_paths]
    else:
        stamps = generate_pseudo_labels(
            image_paths=image_paths,
            output_mask_dir=pseudo_mask_dir,
            overlay_dir=overlay_dir,
            support_image_paths=support_image_paths,
            support_mask_paths=support_mask_paths,
            support_mask_threshold=args.support_mask_threshold,
            model_id=args.seggpt_model_id,
            threshold=args.seggpt_threshold,
            device=args.device,
            skip_existing=args.skip_existing_masks,
        )
    train_split, val_split = write_splits(
        stamps,
        split_dir,
        train_split_name=args.train_split_name,
        val_split_name=args.val_split_name,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    train_count = len([line for line in train_split.read_text(encoding="utf-8").splitlines() if line.strip()])
    val_count = len([line for line in val_split.read_text(encoding="utf-8").splitlines() if line.strip()])
    print(f"Wrote split files: train={train_split} ({train_count}), val={val_split} ({val_count})")

    if args.skip_training:
        print("Pseudo-label generation and split creation finished. Training skipped.")
        return

    command = [
        sys.executable,
        str(REPO_ROOT / "train_segformer_distill.py"),
        "--data_root",
        str(data_root),
        "--img_subdir",
        str(Path(args.segformer_image_dir)),
        "--mask_dir",
        str(Path(args.pseudo_mask_dir)),
        "--train_split",
        str(train_split.relative_to(data_root) if train_split.is_relative_to(data_root) else train_split),
        "--val_split",
        str(val_split.relative_to(data_root) if val_split.is_relative_to(data_root) else val_split),
        "--model_name",
        args.segformer_model_name,
        "--local_files_only",
        "--img_size",
        str(args.img_size),
        "--mask_threshold",
        str(args.mask_threshold),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--class_weight_fg",
        str(args.class_weight_fg),
        "--dice_weight",
        str(args.dice_weight),
        "--num_workers",
        str(args.num_workers),
        "--run_name",
        args.run_name,
        "--output_root",
        args.output_root,
    ]
    if args.color_jitter:
        command.append("--color_jitter")
    print("Running training command:")
    print(" ".join(command))
    subprocess.run(command, check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
