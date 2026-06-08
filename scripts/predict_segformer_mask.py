from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prosthetic_grasp.common.io import load_image
from prosthetic_grasp.common.segformer_distill import (
    iter_image_files,
    load_segformer_checkpoint,
    load_segformer_model,
    predict_segformer_mask,
    save_mask,
    save_overlay,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Predict mechanical-hand masks with a distilled SegFormer model.")
    parser.add_argument("--image", default="", help="Single image path.")
    parser.add_argument("--image-dir", default="", help="Directory of images.")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overlay-dir", default="")
    parser.add_argument("--checkpoint", required=True, help="best_model.pth or best_checkpoint.pth.")
    parser.add_argument("--model-name", default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--img-size", type=int, default=448)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if not args.image and not args.image_dir:
        raise ValueError("Provide --image or --image-dir.")
    if args.device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    model = load_segformer_model(
        model_name=args.model_name,
        num_labels=2,
        local_files_only=args.local_files_only,
        device=device,
    )
    load_segformer_checkpoint(model, args.checkpoint, device)

    image_paths = [Path(args.image)] if args.image else iter_image_files(args.image_dir, recursive=args.recursive)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.image_dir}.")

    output_dir = Path(args.output_dir)
    overlay_dir = Path(args.overlay_dir) if args.overlay_dir else None
    for index, image_path in enumerate(image_paths, start=1):
        image_rgb = load_image(image_path)
        mask, prob = predict_segformer_mask(
            model,
            image_rgb,
            img_size=args.img_size,
            threshold=args.threshold,
            device=device,
        )
        save_mask(output_dir / f"{image_path.stem}.png", mask)
        if overlay_dir is not None:
            save_overlay(overlay_dir / f"{image_path.stem}.png", image_rgb, mask)
        print(f"[{index}/{len(image_paths)}] {image_path} fg_ratio={float(np.mean(mask)):.4f} prob_mean={float(prob.mean()):.4f}")


if __name__ == "__main__":
    main()
