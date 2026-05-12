from __future__ import annotations

import argparse
from pathlib import Path

from prosthetic_grasp.common.io import load_depth, load_image, save_image, save_json, save_mask
from prosthetic_grasp.common.types import SensorFrame
from prosthetic_grasp.config.settings import load_settings
from prosthetic_grasp.pipeline import ProstheticGraspPipeline


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the minimal prosthetic grasp pipeline.")
    parser.add_argument("--rgb", required=True, help="Path to the RGB image.")
    parser.add_argument("--depth", default=None, help="Optional path to the depth image.")
    parser.add_argument(
        "--instruction",
        required=True,
        help="User instruction, for example: 'pick up the bottle'.",
    )
    parser.add_argument(
        "--config",
        default="config/default.toml",
        help="Path to the TOML config file.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/run",
        help="Directory for artifacts.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    settings = load_settings(args.config)

    rgb = load_image(args.rgb)
    depth = load_depth(args.depth) if args.depth else None
    frame = SensorFrame(rgb=rgb, depth=depth, rgb_path=args.rgb, depth_path=args.depth)

    pipeline = ProstheticGraspPipeline(settings)
    result = pipeline.run(frames=[frame], instruction=args.instruction)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(output_dir / "result.json", result.to_json_dict())

    if result.scene is not None:
        save_mask(output_dir / "prosthesis_mask.png", result.scene.prosthesis.fine_mask)
        save_mask(output_dir / "prosthesis_lollipop.png", result.scene.prosthesis.lollipop_mask)

    if result.candidates:
        for idx, candidate in enumerate(result.candidates):
            save_image(output_dir / f"candidate_{idx:02d}.png", candidate.image)

    print(f"Pipeline status: {result.status}")
    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
