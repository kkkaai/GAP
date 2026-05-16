from __future__ import annotations

import argparse

from prosthetic_grasp.common.artifacts import save_pipeline_artifacts
from prosthetic_grasp.common.io import load_depth, load_image
from prosthetic_grasp.common.types import SensorFrame
from prosthetic_grasp.config.settings import load_settings
from prosthetic_grasp.pipeline import ProstheticGraspPipeline


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the phase-organized prosthetic grasp pipeline.")
    parser.add_argument("--rgb", required=True, help="Path to the RGB image.")
    parser.add_argument("--depth", default=None, help="Optional path to the depth image.")
    parser.add_argument("--config", default="config/default.toml", help="Path to the TOML config file.")
    parser.add_argument("--output-dir", default="outputs/run", help="Directory for artifacts.")
    parser.add_argument("--phase3-mode", default=None, help="Override phase3_erase.mode")
    parser.add_argument("--phase3-model-name", default=None, help="Override phase3_erase.model_name")
    parser.add_argument("--phase3-model-id", default=None, help="Override phase3_erase.model_id")
    parser.add_argument("--phase4-mode", default=None, help="Override phase4_inpaint.mode")
    parser.add_argument("--phase4-model-name", default=None, help="Override phase4_inpaint.model_name")
    parser.add_argument("--phase4-model-id", default=None, help="Override phase4_inpaint.model_id")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    settings = load_settings(args.config)
    settings.setdefault("phase3_erase", {})
    settings.setdefault("phase4_inpaint", {})
    if args.phase3_mode is not None:
        settings["phase3_erase"]["mode"] = args.phase3_mode
    if args.phase3_model_name is not None:
        settings["phase3_erase"]["model_name"] = args.phase3_model_name
    if args.phase3_model_id is not None:
        settings["phase3_erase"]["model_id"] = args.phase3_model_id
    if args.phase4_mode is not None:
        settings["phase4_inpaint"]["mode"] = args.phase4_mode
    if args.phase4_model_name is not None:
        settings["phase4_inpaint"]["model_name"] = args.phase4_model_name
    if args.phase4_model_id is not None:
        settings["phase4_inpaint"]["model_id"] = args.phase4_model_id
    rgb = load_image(args.rgb)
    depth = load_depth(args.depth) if args.depth else None
    frame = SensorFrame(rgb=rgb, depth=depth, rgb_path=args.rgb, depth_path=args.depth)

    pipeline = ProstheticGraspPipeline(settings)
    result = pipeline.run(frame=frame)
    output_dir = save_pipeline_artifacts(result, args.output_dir)

    print(f"Pipeline status: {result.status}")
    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
