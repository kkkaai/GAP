from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from prosthetic_grasp.common.io import load_image, save_image, save_mask
from prosthetic_grasp.common.prompts import DEFAULT_PHASE4_INTENTION
from prosthetic_grasp.phases.phase2_lollipop import (
    Phase2Lollipop,
    Phase2LollipopConfig,
    mirror_lollipop_params,
    render_lollipop_mask,
    rotate_lollipop_params,
)
from prosthetic_grasp.phases.phase4_inpaint import Phase4Inpaint, Phase4InpaintConfig


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate rotated lollipop masks around the fitted circle center and optionally run phase4 on each angle. "
            "Positive angles are clockwise in image coordinates; negative angles are counterclockwise."
        )
    )
    parser.add_argument(
        "--image",
        default="outputs/test_phase3_zenmux_new_prompt_handmask/openai__gpt-image-2/erase/erased_full.png",
        help="Object-only RGB image used as phase4 input.",
    )
    parser.add_argument("--hand-mask", default="testcase/mask_hand.png", help="Binary hand mask used to fit lollipop.")
    parser.add_argument("--output-dir", default="outputs/test_phase4_lollipop_angle_sweep")
    parser.add_argument("--angles", default="-90,-60,-30,0,30,60,90", help="Comma-separated degrees.")
    parser.add_argument("--model-id", default="openai/gpt-image-2", help="ZenMux image edit model ID.")
    parser.add_argument("--pad-ratio", type=float, default=0.35)
    parser.add_argument("--intention", default=DEFAULT_PHASE4_INTENTION)
    parser.add_argument("--prompt", default=None, help="Optional full prompt override. If omitted, uses intention template.")
    parser.add_argument("--run-phase4", action="store_true", help="Call the phase4 image-edit API for every angle.")
    parser.add_argument(
        "--mirror",
        choices=["none", "horizontal", "vertical", "both"],
        default="none",
        help="Mirror the fitted lollipop direction around the circle center before rotating.",
    )
    parser.add_argument("--palm-radius-scale", type=float, default=1.1)
    parser.add_argument("--strip-width-scale", type=float, default=1.1)
    parser.add_argument("--line-length-scale", type=float, default=2.0)
    return parser


def _parse_angles(value: str) -> list[float]:
    angles = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not angles:
        raise ValueError("At least one angle is required.")
    return angles


def _angle_name(angle: float) -> str:
    rounded = int(round(angle))
    if abs(angle - rounded) < 1e-6:
        value = f"{abs(rounded):03d}"
    else:
        value = f"{abs(angle):06.2f}".replace(".", "p")
    prefix = "cw" if angle >= 0 else "ccw"
    return f"{prefix}_{value}"


def _mask_overlay(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = Image.fromarray(image_rgb).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    alpha = (mask.astype(np.uint8) * 120)
    color = np.zeros((*mask.shape, 4), dtype=np.uint8)
    color[..., 0] = 255
    color[..., 3] = alpha
    overlay = Image.fromarray(color, mode="RGBA")
    return np.array(Image.alpha_composite(base, overlay).convert("RGB"))


def _save_contact_sheet(output_dir: Path, entries: list[tuple[float, Path]]) -> None:
    images = []
    labels = []
    for angle, path in entries:
        img = Image.open(path).convert("RGB")
        img.thumbnail((260, 260))
        images.append(img.copy())
        labels.append(f"{angle:+.0f} deg")
    if not images:
        return
    tile_w = max(img.width for img in images)
    tile_h = max(img.height for img in images)
    label_h = 28
    cols = min(4, len(images))
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (img, label) in enumerate(zip(images, labels)):
        col = idx % cols
        row = idx // cols
        x0 = col * tile_w
        y0 = row * (tile_h + label_h)
        draw.text((x0 + 8, y0 + 6), label, fill=(0, 0, 0))
        canvas.paste(img, (x0 + (tile_w - img.width) // 2, y0 + label_h + (tile_h - img.height) // 2))
    canvas.save(output_dir / "mask_overlay_contact_sheet.png")


def main() -> None:
    args = build_argparser().parse_args()
    image = load_image(args.image)
    hand_mask = np.array(Image.open(args.hand_mask).convert("L")) > 127
    angles = _parse_angles(args.angles)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase2_config = Phase2LollipopConfig(
        line_length_scale=args.line_length_scale,
        palm_radius_scale=args.palm_radius_scale,
        strip_width_scale=args.strip_width_scale,
    )
    phase2_result = Phase2Lollipop(phase2_config).run(hand_mask)
    save_image(output_dir / "phase4_input.png", image)
    save_mask(output_dir / "source_hand_mask.png", hand_mask)
    base_params = mirror_lollipop_params(phase2_result.lolli_params, args.mirror)
    base_mask = render_lollipop_mask(hand_mask.shape, base_params, phase2_config)
    save_mask(output_dir / "fitted_lollipop_mask.png", phase2_result.lollipop_mask)
    save_image(output_dir / "fitted_lollipop_overlay.png", _mask_overlay(image, phase2_result.lollipop_mask))
    save_mask(output_dir / "base_lollipop_mask.png", base_mask)
    save_image(output_dir / "base_lollipop_overlay.png", _mask_overlay(image, base_mask))

    summary = {
        "image": args.image,
        "hand_mask": args.hand_mask,
        "run_phase4": args.run_phase4,
        "mirror": args.mirror,
        "angle_convention": "Positive angles are clockwise in image coordinates; negative angles are counterclockwise.",
        "fitted_lollipop_params": list(phase2_result.lolli_params),
        "base_lollipop_params": list(base_params),
        "wrist_point": list(phase2_result.wrist_point),
        "tip_point": list(phase2_result.tip_point),
        "phase2_config": asdict(phase2_config),
        "angles": [],
    }
    overlay_entries = []

    for angle in angles:
        angle_dir = output_dir / _angle_name(angle)
        angle_dir.mkdir(parents=True, exist_ok=True)
        rotated_params = rotate_lollipop_params(base_params, angle)
        rotated_mask = render_lollipop_mask(hand_mask.shape, rotated_params, phase2_config)
        save_mask(angle_dir / "phase4_mask.png", rotated_mask)
        save_image(angle_dir / "phase4_mask_overlay.png", _mask_overlay(image, rotated_mask))
        overlay_entries.append((angle, angle_dir / "phase4_mask_overlay.png"))

        config = Phase4InpaintConfig(
            mode="api",
            model_name="zenmux",
            model_id=args.model_id,
            intention=args.intention,
            prompt=args.prompt or "",
            pad_ratio=args.pad_ratio,
            preserve_unmasked_pixels=False,
        )
        item = {
            "angle_degrees": angle,
            "output_dir": str(angle_dir),
            "lollipop_params": list(rotated_params),
            "mask_path": str(angle_dir / "phase4_mask.png"),
            "overlay_path": str(angle_dir / "phase4_mask_overlay.png"),
        }

        if args.run_phase4:
            result = Phase4Inpaint(config).run(image, rotated_mask)
            save_image(angle_dir / "phase4_rgb_crop.png", result.rgb_crop)
            save_mask(angle_dir / "phase4_mask_crop.png", result.mask_crop)
            save_image(angle_dir / "phase4_inpaint_crop.png", result.inpaint_crop)
            save_image(angle_dir / "phase4_inpaint_full.png", result.inpaint_full)
            (angle_dir / "phase4_prompt.txt").write_text(result.prompt, encoding="utf-8")
            (angle_dir / "phase4_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
            item["phase4_inpaint_full"] = str(angle_dir / "phase4_inpaint_full.png")

        summary["angles"].append(item)

    _save_contact_sheet(output_dir, overlay_entries)
    (output_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved {output_dir}")
    if not args.run_phase4:
        print("Generated masks only. Add --run-phase4 to call the image-edit API for every angle.")


if __name__ == "__main__":
    main()
