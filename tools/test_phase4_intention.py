from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from prosthetic_grasp.common.io import load_image
from prosthetic_grasp.phases.phase4_intention import Phase4Intention, Phase4IntentionConfig


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a Phase4 grasp intention with a ZenMux VLM.")
    parser.add_argument("--image", default="testcase/coffeecup.png", help="First-person scene image.")
    parser.add_argument("--task", default="", help="Optional speech-to-text user task instruction.")
    parser.add_argument("--output-dir", default="outputs/test_phase4_intention", help="Output directory.")
    parser.add_argument("--model-id", default="qwen/qwen3-vl-plus")
    parser.add_argument("--fast-model-id", default="qwen/qwen3-vl-flash")
    parser.add_argument("--fast", action="store_true", help="Use fast_model_id instead of model_id.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    image = load_image(args.image)
    config = Phase4IntentionConfig(
        model_id=args.model_id,
        fast_model_id=args.fast_model_id,
        use_fast_model=args.fast,
    )
    result = Phase4Intention(config).run(image, task_instruction=args.task)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase4_intention.json").write_text(
        json.dumps(asdict(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "phase4_intention.txt").write_text(result.phase4_intention, encoding="utf-8")
    (output_dir / "phase4_intention_prompt.txt").write_text(result.prompt, encoding="utf-8")
    print(result.phase4_intention)
    print(f"saved {output_dir}")


if __name__ == "__main__":
    main()
