from __future__ import annotations

import argparse
from pathlib import Path

from prosthetic_grasp.common.io import load_image, save_json
from prosthetic_grasp.phases.phase_quality_vlm import PhaseQualityVLM, VLMQualityScoreConfig


def parse_labeled_image(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, path_text = value.split("=", 1)
    label = label.strip()
    path = Path(path_text.strip())
    if not label:
        label = path.stem
    return label, path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score GAP teacher-pipeline visual artifacts with a VLM.")
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "phase2_lollipop",
            "phase4_generation",
            "phase5_mano",
            "phase6_object_pose",
            "phase6_retarget",
        ],
        help="Pipeline stage to evaluate.",
    )
    parser.add_argument(
        "--image",
        action="append",
        required=True,
        help="Image path, or label=path. Repeat for source/overlay/generated images.",
    )
    parser.add_argument("--object-name", default=None, help="Optional target object name.")
    parser.add_argument("--task", default=None, help="Optional task instruction.")
    parser.add_argument("--model-id", default=None, help="Override VLM model id.")
    parser.add_argument("--output-json", default=None, help="Optional path for the scoring JSON.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    config = VLMQualityScoreConfig()
    if args.model_id:
        config.model_id = args.model_id
    scorer = PhaseQualityVLM(config)
    images = []
    for image_arg in args.image:
        label, path = parse_labeled_image(image_arg)
        images.append((label, load_image(path)))
    result = scorer.run(
        stage=args.stage,
        images=images,
        object_name=args.object_name,
        task_instruction=args.task,
    )
    payload = result.to_json_dict()
    if args.output_json:
        save_json(args.output_json, payload)
    print(f"stage: {result.stage}")
    print(f"overall_score: {result.overall_score}")
    print(f"pass: {result.pass_}")
    print(f"failure_tags: {', '.join(result.failure_tags) if result.failure_tags else 'none'}")
    print(f"reason: {result.reason}")


if __name__ == "__main__":
    main()
