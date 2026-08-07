#!/usr/bin/env python3
"""Parse q_goal CFM train.log and optionally plot loss curves."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

_PATTERN = re.compile(r"epoch=(\d+)\s+train_flow_mse=([0-9.eE+-]+)\s+val_flow_mse=([0-9.eE+-]+)")


def parse_log(path: Path) -> list[tuple[int, float, float]]:
    rows: list[tuple[int, float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _PATTERN.search(line)
        if match:
            rows.append((int(match.group(1)), float(match.group(2)), float(match.group(3))))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--png", type=Path)
    args = parser.parse_args()

    rows = parse_log(args.log)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_flow_mse", "val_flow_mse"])
        writer.writerows(rows)

    if args.png and rows:
        import matplotlib.pyplot as plt

        epochs = [r[0] for r in rows]
        train = [r[1] for r in rows]
        val = [r[2] for r in rows]
        args.png.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
        ax.plot(epochs, train, label="train_flow_mse", linewidth=1.8)
        ax.plot(epochs, val, label="val_flow_mse", linewidth=1.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Flow MSE")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.png)
        plt.close(fig)

    if rows:
        best = min(rows, key=lambda r: r[2])
        print(f"parsed={len(rows)} latest_epoch={rows[-1][0]} latest_train={rows[-1][1]:.6f} latest_val={rows[-1][2]:.6f} best_epoch={best[0]} best_val={best[2]:.6f}")
    else:
        print("parsed=0")


if __name__ == "__main__":
    main()
