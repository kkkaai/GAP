#!/usr/bin/env python3
"""Sample q_goal candidates from a trained CFM checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prosthetic_grasp.learning.q_goal.checkpoint import load_checkpoint
from prosthetic_grasp.learning.q_goal.dataset import QGoalDataset, QGoalDatasetConfig
from prosthetic_grasp.learning.q_goal.encoders import ConditionEncoderConfig
from prosthetic_grasp.learning.q_goal.model import QGoalCFMModel, QGoalCFMModelConfig
from prosthetic_grasp.learning.q_goal.normalization import JointNormalizer
from prosthetic_grasp.learning.q_goal.sampling import euler_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    model_cfg = cfg["model_config"]
    condition = ConditionEncoderConfig(**model_cfg["condition"])
    model_config = QGoalCFMModelConfig(
        q_dim=int(model_cfg["q_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        depth=int(model_cfg["depth"]),
        time_dim=int(model_cfg["time_dim"]),
        dropout=float(model_cfg["dropout"]),
        condition=condition,
    )
    model = QGoalCFMModel(model_config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    normalizer = JointNormalizer.from_bounds(cfg["q_min"], cfg["q_max"], device=device)
    task_vocab = cfg.get("task_vocab", {})
    data_args = cfg["args"]
    dataset = QGoalDataset(
        QGoalDatasetConfig(
            jsonl_path=args.input_jsonl,
            image_size=int(data_args["image_size"]),
            mask_size=int(data_args["mask_size"]),
            require_rgb=data_args["image_encoder"] not in {"none", "precomputed"},
            task_vocab=task_vocab,
            task_embedding_dim=int(data_args["task_embedding_dim"]) if data_args["task_encoder"] == "precomputed" else 0,
            image_embedding_dim=int(data_args["image_embedding_dim"]) if data_args["image_encoder"] == "precomputed" else 0,
            normalizer=normalizer.to("cpu"),
        )
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    outputs: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            samples_norm = euler_sample(
                model,
                batch,
                q_dim=model_config.q_dim,
                num_steps=args.num_steps,
                num_candidates=args.num_candidates,
            )
            samples = normalizer.denormalize(torch.clamp(samples_norm, -1.0, 1.0)).cpu()
            for i in range(samples.shape[0]):
                outputs.append(
                    {
                        "task_text": batch["task_text"][i] if isinstance(batch["task_text"], list) else "",
                        "q_goal_candidates": samples[i].tolist(),
                    }
                )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

