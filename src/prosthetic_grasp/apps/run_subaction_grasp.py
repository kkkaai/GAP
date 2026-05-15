from __future__ import annotations

import argparse
from pathlib import Path

from prosthetic_grasp.common.io import load_depth, load_image, save_image, save_json, save_mask
from prosthetic_grasp.common.types import SensorFrame
from prosthetic_grasp.config.settings import load_settings
from prosthetic_grasp.pipeline import ProstheticGraspPipeline


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the phase-organized prosthetic grasp pipeline.")
    parser.add_argument("--rgb", required=True, help="Path to the RGB image.")
    parser.add_argument("--depth", default=None, help="Optional path to the depth image.")
    parser.add_argument("--config", default="config/default.toml", help="Path to the TOML config file.")
    parser.add_argument("--output-dir", default="outputs/run", help="Directory for artifacts.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    settings = load_settings(args.config)
    rgb = load_image(args.rgb)
    depth = load_depth(args.depth) if args.depth else None
    frame = SensorFrame(rgb=rgb, depth=depth, rgb_path=args.rgb, depth_path=args.depth)

    pipeline = ProstheticGraspPipeline(settings)
    result = pipeline.run(frame=frame)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "result.json", result.to_json_dict())

    if result.phase1_mask is not None:
        save_mask(output_dir / "phase1_mask.png", result.phase1_mask.mask)
    if result.phase2_lollipop is not None:
        save_mask(output_dir / "phase2_lollipop.png", result.phase2_lollipop.lollipop_mask)
    if result.phase3_inpaint is not None:
        save_image(output_dir / "phase3_inpaint_crop.png", result.phase3_inpaint.inpaint_crop)
        save_image(output_dir / "phase3_inpaint_full.png", result.phase3_inpaint.inpaint_full)
    if result.phase4_flux_fill is not None:
        save_image(output_dir / "phase4_flux_fill_crop.png", result.phase4_flux_fill.flux_crop)
        save_image(output_dir / "phase4_flux_fill_full.png", result.phase4_flux_fill.flux_full)

    print(f"Pipeline status: {result.status}")
    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
