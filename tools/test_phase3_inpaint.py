from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from prosthetic_grasp.common.io import load_image, save_image, save_mask
from prosthetic_grasp.common.image_editing import STABILITY_INPAINT_ENDPOINT
from prosthetic_grasp.phases.phase3_erase import Phase3Erase, Phase3EraseConfig


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run phase3 hand removal on a single image and mask.")
    parser.add_argument("--image", default="testcase/coffeecup.png", help="Path to the RGB input image.")
    parser.add_argument(
        "--mask",
        default="testcase/converted/mask_hand_1_lollipop_phase2_radius110_width110.png",
        help="Path to the binary lollipop mask image.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/test_phase3_coffeecup_inpaint",
        help="Directory for saved artifacts.",
    )
    parser.add_argument("--mode", default="api", help="Image editing mode, e.g. api or local.")
    parser.add_argument(
        "--model-name",
        default="stability-inpaint",
        help="Image editing model name, e.g. stability-inpaint or flux-fill.",
    )
    parser.add_argument(
        "--model-id",
        default="stable-diffusion-xl-1024-v1-0",
        help="Model identifier used by the selected backend.",
    )
    parser.add_argument("--pad-ratio", type=float, default=0.25, help="Mask crop padding ratio.")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional prompt override. Defaults to Phase3EraseConfig.prompt.",
    )
    parser.add_argument(
        "--no-preserve-unmasked",
        action="store_true",
        help="Disable restoring unmasked pixels from the source image.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    image = load_image(args.image)
    mask = np.array(Image.open(args.mask).convert("L")) > 127

    config = Phase3EraseConfig(
        mode=args.mode,
        model_name=args.model_name,
        model_id=args.model_id,
        stability_endpoint=STABILITY_INPAINT_ENDPOINT if args.model_name == "stability-inpaint" else "",
        pad_ratio=args.pad_ratio,
        preserve_unmasked_pixels=False,
    )
    if args.prompt is not None:
        config.prompt = args.prompt

    phase3 = Phase3Erase(config)
    result = phase3.run(image, mask)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_mask(output_dir / "phase3_lollipop_mask.png", mask)
    save_image(output_dir / "phase3_rgb_crop.png", result.rgb_crop)
    save_mask(output_dir / "phase3_mask_crop.png", result.mask_crop)
    save_image(output_dir / "phase3_inpaint_crop.png", result.erased_crop)
    save_image(output_dir / "phase3_inpaint_full.png", result.erased_full)
    (output_dir / "phase3_prompt.txt").write_text(result.prompt, encoding="utf-8")

    print(f"saved {output_dir}")


if __name__ == "__main__":
    main()
