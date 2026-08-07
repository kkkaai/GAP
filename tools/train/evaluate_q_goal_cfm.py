#!/usr/bin/env python3
"""Evaluate sampled q_goal candidates against teacher q_goal labels."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
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


@dataclass
class MetricAccumulator:
    count: int = 0
    single_l1_sum: float = 0.0
    single_l2_sum: float = 0.0
    topk_l1_sum: float = 0.0
    topk_l2_sum: float = 0.0
    diversity_sum: float = 0.0
    violation_sum: float = 0.0

    def update(
        self,
        *,
        single_l1: torch.Tensor,
        single_l2: torch.Tensor,
        topk_l1: torch.Tensor,
        topk_l2: torch.Tensor,
        diversity: torch.Tensor,
        violation: torch.Tensor,
    ) -> None:
        batch_size = int(single_l1.shape[0])
        self.count += batch_size
        self.single_l1_sum += float(single_l1.sum().cpu())
        self.single_l2_sum += float(single_l2.sum().cpu())
        self.topk_l1_sum += float(topk_l1.sum().cpu())
        self.topk_l2_sum += float(topk_l2.sum().cpu())
        self.diversity_sum += float(diversity.sum().cpu())
        self.violation_sum += float(violation.sum().cpu())

    def as_dict(self) -> dict[str, float | int]:
        denom = max(self.count, 1)
        return {
            "count": self.count,
            "single_sample_joint_l1": self.single_l1_sum / denom,
            "single_sample_joint_l2": self.single_l2_sum / denom,
            "topk_best_joint_l1": self.topk_l1_sum / denom,
            "topk_best_joint_l2": self.topk_l2_sum / denom,
            "candidate_diversity_l2": self.diversity_sum / denom,
            "joint_limit_violation_rate": self.violation_sum / denom,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def load_model(checkpoint: Path, device: torch.device) -> tuple[QGoalCFMModel, dict[str, Any], JointNormalizer]:
    ckpt = load_checkpoint(checkpoint, map_location=device)
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
    return model, cfg, normalizer


def make_dataset(input_jsonl: Path, cfg: dict[str, Any], normalizer: JointNormalizer, limit: int | None) -> QGoalDataset:
    data_args = cfg["args"]
    dataset = QGoalDataset(
        QGoalDatasetConfig(
            jsonl_path=input_jsonl,
            image_size=int(data_args["image_size"]),
            mask_size=int(data_args["mask_size"]),
            require_rgb=data_args["image_encoder"] not in {"none", "precomputed"},
            use_lollipop_mask=data_args["lollipop_mask_encoder"] != "none",
            use_lollipop_params=not data_args["no_lollipop_params"],
            use_q_current=not data_args["no_q_current"],
            task_vocab=cfg.get("task_vocab", {}),
            task_embedding_dim=int(data_args["task_embedding_dim"]) if data_args["task_encoder"] == "precomputed" else 0,
            image_embedding_dim=int(data_args["image_embedding_dim"]) if data_args["image_encoder"] == "precomputed" else 0,
            normalizer=normalizer.to("cpu"),
        )
    )
    if limit is not None and limit < len(dataset):
        dataset.samples = dataset.samples[:limit]
    return dataset


def pairwise_diversity(samples: torch.Tensor) -> torch.Tensor:
    if samples.shape[1] < 2:
        return torch.zeros(samples.shape[0], device=samples.device)
    dist = torch.cdist(samples, samples, p=2)
    k = samples.shape[1]
    tri = torch.triu(torch.ones(k, k, dtype=torch.bool, device=samples.device), diagonal=1)
    return dist[:, tri].mean(dim=1)


@torch.no_grad()
def euler_sample_cached_condition(
    model: QGoalCFMModel,
    batch: dict[str, Any],
    *,
    q_dim: int,
    num_steps: int,
    num_candidates: int,
) -> torch.Tensor:
    """Sample q goals while encoding RGB/text/mask conditions only once per batch.

    The training model recomputes condition features inside ``forward``. That is
    fine for training, but K-candidate evaluation would otherwise rerun SigLIP
    for every Euler step and every candidate batch.
    """

    cond = model.condition_encoder(batch)
    batch_size = int(cond.shape[0])
    cond = cond[:, None, :].expand(batch_size, num_candidates, -1).reshape(batch_size * num_candidates, -1)
    x = torch.randn(batch_size * num_candidates, q_dim, device=cond.device)
    dt = 1.0 / float(num_steps)
    for step in range(num_steps):
        t_value = (step + 0.5) / float(num_steps)
        t = torch.full((batch_size * num_candidates,), t_value, device=cond.device)
        time = model.time_encoder(t)
        x = x + model.vector_field(torch.cat([x, time, cond], dim=-1)) * dt
    return x.reshape(batch_size, num_candidates, q_dim)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, cfg, normalizer = load_model(args.checkpoint, device)
    dataset = make_dataset(args.input_jsonl, cfg, normalizer, args.limit)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    q_min = normalizer.q_min.to(device)
    q_max = normalizer.q_max.to(device)
    acc = MetricAccumulator()
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            batch = move_batch(batch, device)
            teacher = batch["q_goal_raw"]
            samples_norm = euler_sample_cached_condition(
                model,
                batch,
                q_dim=normalizer.q_dim,
                num_steps=args.num_steps,
                num_candidates=args.num_candidates,
            )
            samples_norm = torch.clamp(samples_norm, -1.0, 1.0)
            samples = normalizer.denormalize(samples_norm)
            err = samples - teacher[:, None, :]
            l1_per_candidate = err.abs().mean(dim=-1)
            l2_per_candidate = torch.linalg.vector_norm(err, dim=-1)
            single_l1 = l1_per_candidate[:, 0]
            single_l2 = l2_per_candidate[:, 0]
            topk_l1 = l1_per_candidate.min(dim=1).values
            topk_l2 = l2_per_candidate.min(dim=1).values
            diversity = pairwise_diversity(samples)
            violations = ((samples < q_min) | (samples > q_max)).float().mean(dim=(1, 2))
            acc.update(
                single_l1=single_l1,
                single_l2=single_l2,
                topk_l1=topk_l1,
                topk_l2=topk_l2,
                diversity=diversity,
                violation=violations,
            )
            for i in range(teacher.shape[0]):
                rows.append(
                    {
                        "index": batch_index * args.batch_size + i,
                        "task": batch["task_text"][i] if isinstance(batch["task_text"], list) else "",
                        "single_sample_joint_l1": float(single_l1[i].cpu()),
                        "single_sample_joint_l2": float(single_l2[i].cpu()),
                        "topk_best_joint_l1": float(topk_l1[i].cpu()),
                        "topk_best_joint_l2": float(topk_l2[i].cpu()),
                        "candidate_diversity_l2": float(diversity[i].cpu()),
                        "joint_limit_violation_rate": float(violations[i].cpu()),
                    }
                )

    summary = {
        "checkpoint": str(args.checkpoint),
        "input_jsonl": str(args.input_jsonl),
        "num_candidates": args.num_candidates,
        "num_steps": args.num_steps,
        "batch_size": args.batch_size,
        "device": str(device),
        "metrics": acc.as_dict(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.output_dir / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["index"])
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main() -> None:
    summary = evaluate(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
