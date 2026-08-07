#!/usr/bin/env python3
"""Benchmark q_goal CFM inference latency and frequency."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_q_goal_cfm import euler_sample_cached_condition, load_model, make_dataset, move_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-candidates", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--num-steps", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def elapsed_ms(device: torch.device, fn) -> float:
    sync(device)
    start = time.perf_counter()
    fn()
    sync(device)
    return (time.perf_counter() - start) * 1000.0


def summarize(values: list[float]) -> dict[str, float]:
    values_sorted = sorted(values)
    return {
        "mean_ms": float(statistics.fmean(values)),
        "median_ms": float(statistics.median(values)),
        "p90_ms": float(values_sorted[int(0.9 * (len(values_sorted) - 1))]),
        "p95_ms": float(values_sorted[int(0.95 * (len(values_sorted) - 1))]),
        "min_ms": float(values_sorted[0]),
        "max_ms": float(values_sorted[-1]),
        "hz_from_mean": float(1000.0 / max(statistics.fmean(values), 1e-9)),
    }


@torch.no_grad()
def sample_from_condition(
    model,
    cond: torch.Tensor,
    *,
    q_dim: int,
    num_steps: int,
    num_candidates: int,
) -> torch.Tensor:
    batch_size = int(cond.shape[0])
    expanded = cond[:, None, :].expand(batch_size, num_candidates, -1).reshape(batch_size * num_candidates, -1)
    x = torch.randn(batch_size * num_candidates, q_dim, device=cond.device)
    dt = 1.0 / float(num_steps)
    for step in range(num_steps):
        t_value = (step + 0.5) / float(num_steps)
        t = torch.full((batch_size * num_candidates,), t_value, device=cond.device)
        time_features = model.time_encoder(t)
        x = x + model.vector_field(torch.cat([x, time_features, expanded], dim=-1)) * dt
    return x.reshape(batch_size, num_candidates, q_dim)


def make_batch(args: argparse.Namespace, cfg: dict[str, Any], normalizer, device: torch.device) -> dict[str, Any]:
    dataset = make_dataset(args.input_jsonl, cfg, normalizer, None)
    if args.sample_index:
        dataset.samples = dataset.samples[args.sample_index :] + dataset.samples[: args.sample_index]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return move_batch(next(iter(loader)), device)


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    model, cfg, normalizer = load_model(args.checkpoint, device)
    batch = make_batch(args, cfg, normalizer, device)
    q_dim = normalizer.q_dim

    results: list[dict[str, Any]] = []
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model.condition_encoder(batch)
        condition_times = [elapsed_ms(device, lambda: model.condition_encoder(batch)) for _ in range(args.iterations)]
        cond = model.condition_encoder(batch)

        for steps in args.num_steps:
            for candidates in args.num_candidates:
                for _ in range(args.warmup):
                    _ = sample_from_condition(
                        model,
                        cond,
                        q_dim=q_dim,
                        num_steps=steps,
                        num_candidates=candidates,
                    )
                cached_times = [
                    elapsed_ms(
                        device,
                        lambda: sample_from_condition(
                            model,
                            cond,
                            q_dim=q_dim,
                            num_steps=steps,
                            num_candidates=candidates,
                        ),
                    )
                    for _ in range(args.iterations)
                ]
                for _ in range(max(1, args.warmup // 4)):
                    _ = euler_sample_cached_condition(
                        model,
                        batch,
                        q_dim=q_dim,
                        num_steps=steps,
                        num_candidates=candidates,
                    )
                online_times = [
                    elapsed_ms(
                        device,
                        lambda: euler_sample_cached_condition(
                            model,
                            batch,
                            q_dim=q_dim,
                            num_steps=steps,
                            num_candidates=candidates,
                        ),
                    )
                    for _ in range(args.iterations)
                ]
                row = {
                    "batch_size": args.batch_size,
                    "num_steps": steps,
                    "num_candidates": candidates,
                    "condition_encode": summarize(condition_times),
                    "flow_sampling_cached_condition": summarize(cached_times),
                    "online_encode_plus_sampling": summarize(online_times),
                }
                results.append(row)

    summary = {
        "checkpoint": str(args.checkpoint),
        "input_jsonl": str(args.input_jsonl),
        "device": str(device),
        "iterations": args.iterations,
        "warmup": args.warmup,
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "inference_benchmark.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (args.output_dir / "inference_benchmark.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "batch_size",
            "num_steps",
            "num_candidates",
            "condition_mean_ms",
            "flow_cached_mean_ms",
            "online_mean_ms",
            "online_p95_ms",
            "online_hz_from_mean",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "batch_size": row["batch_size"],
                    "num_steps": row["num_steps"],
                    "num_candidates": row["num_candidates"],
                    "condition_mean_ms": row["condition_encode"]["mean_ms"],
                    "flow_cached_mean_ms": row["flow_sampling_cached_condition"]["mean_ms"],
                    "online_mean_ms": row["online_encode_plus_sampling"]["mean_ms"],
                    "online_p95_ms": row["online_encode_plus_sampling"]["p95_ms"],
                    "online_hz_from_mean": row["online_encode_plus_sampling"]["hz_from_mean"],
                }
            )
    return summary


def main() -> None:
    summary = benchmark(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
