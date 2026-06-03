from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from prosthetic_grasp.common.io import load_image, save_image, save_mask
from prosthetic_grasp.common.prompts import DEFAULT_PHASE4_INTENTION
from prosthetic_grasp.phases.phase4_inpaint import Phase4Inpaint, Phase4InpaintConfig


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run phase4 human-hand generation through ZenMux GPT-Image-2.")
    parser.add_argument(
        "--image",
        default="outputs/test_phase3_zenmux_refine_handmask/openai__gpt-image-2/erase/erased_full.png",
        help="Path to the object-only RGB image.",
    )
    parser.add_argument("--mask", default="testcase/mask_hand_1.png", help="Path to binary hand-generation mask.")
    parser.add_argument("--output-dir", default="outputs/test_phase4_zenmux_gpt_image_2", help="Output directory.")
    parser.add_argument("--model-id", default="openai/gpt-image-2", help="ZenMux image edit model ID.")
    parser.add_argument("--pad-ratio", type=float, default=0.35)
    parser.add_argument("--intention", default=DEFAULT_PHASE4_INTENTION)
    parser.add_argument("--prompt", default=None, help="Optional full prompt override. If omitted, uses intention template.")
    parser.add_argument(
        "--no-preserve-unmasked",
        action="store_true",
        help="Disable restoring unmasked pixels after generation.",
    )
    return parser


def _save_comparison(output_dir: Path) -> None:
    items = [
        ("input", output_dir / "phase4_input.png"),
        ("mask", output_dir / "phase4_mask.png"),
        ("gpt-image-2 phase4", output_dir / "phase4_inpaint_full.png"),
    ]
    images = []
    labels = []
    for label, path in items:
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        img.thumbnail((360, 360))
        images.append(img.copy())
        labels.append(label)
    if not images:
        return
    w = max(img.width for img in images)
    h = max(img.height for img in images)
    label_h = 36
    canvas = Image.new("RGB", (w * len(images), h + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (img, label) in enumerate(zip(images, labels)):
        x = idx * w + (w - img.width) // 2
        y = label_h + (h - img.height) // 2
        canvas.paste(img, (x, y))
        draw.text((idx * w + 8, 8), label, fill=(0, 0, 0))
    canvas.save(output_dir / "comparison_full.png")


def main() -> None:
    args = build_argparser().parse_args()
    image = load_image(args.image)
    mask = np.array(Image.open(args.mask).convert("L")) > 127
    config = Phase4InpaintConfig(
        mode="api",
        model_name="zenmux",
        model_id=args.model_id,
        intention=args.intention,
        prompt=args.prompt or "",
        pad_ratio=args.pad_ratio,
        preserve_unmasked_pixels=not args.no_preserve_unmasked,
    )
    result = Phase4Inpaint(config).run(image, mask)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_image(output_dir / "phase4_input.png", image)
    save_mask(output_dir / "phase4_mask.png", mask)
    save_image(output_dir / "phase4_rgb_crop.png", result.rgb_crop)
    save_mask(output_dir / "phase4_mask_crop.png", result.mask_crop)
    save_image(output_dir / "phase4_inpaint_crop.png", result.inpaint_crop)
    save_image(output_dir / "phase4_inpaint_full.png", result.inpaint_full)
    (output_dir / "phase4_prompt.txt").write_text(result.prompt, encoding="utf-8")
    (output_dir / "phase4_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    _save_comparison(output_dir)
    print(f"saved {output_dir}")


if __name__ == "__main__":
    main()
