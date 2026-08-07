#!/usr/bin/env python3
"""Run position-only retargeting on random HOI4D frames."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".conda" / "envs" / "hoi4d-recon" / "bin" / "python"
POSITION = REPO_ROOT / "tools" / "retarget" / "position.py"
DATASET = REPO_ROOT / "extracted_dataset_sampled"
CAD_ROOT = DATASET / "cad_models" / "HOI4D_CAD_Model_for_release" / "rigid"

CATEGORY_BY_C = {
    "C1": "ToyCar",
    "C2": "Mug",
    "C3": "Laptop",
    "C4": "StorageFurniture",
    "C5": "Bottle",
    "C6": "Safe",
    "C7": "Bowl",
    "C8": "Bucket",
    "C9": "Scissors",
    "C10": "Pliers",
    "C11": "Kettle",
    "C12": "Knife",
    "C13": "TrashCan",
    "C14": "TrashCan",
    "C15": "Chair",
    "C16": "Lamp",
}


def sequence_parts(sequence: Path) -> dict[str, str]:
    parts = sequence.parts
    values: dict[str, str] = {}
    for part in parts:
        if part.startswith(("C", "N", "S", "T", "H")):
            values[part[0]] = part
    return values


def hand_path_for(sequence: Path, frame: str) -> Path | None:
    rel = sequence.relative_to(DATASET)
    candidates = [
        DATASET / "Hand_pose" / "handpose_right_hand" / rel / f"{int(frame)}.pickle",
        DATASET / "Hand_pose" / "handpose_left_hand" / rel / f"{int(frame)}.pickle",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def cad_path_for(sequence: Path) -> Path | None:
    parts = sequence_parts(sequence)
    category = CATEGORY_BY_C.get(parts.get("C", ""))
    instance = parts.get("N", "")
    if not category or not instance:
        return None
    cad = CAD_ROOT / category / f"{int(instance[1:]):03d}.obj"
    return cad if cad.exists() else None


def iter_candidates() -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for rgb in DATASET.glob("ZY*/H*/C*/N*/S*/s*/T*/align_rgb/*.jpg"):
        sequence = rgb.parent.parent
        frame = str(int(rgb.stem))
        objpose = sequence / "objpose" / f"{frame}.json"
        if not objpose.exists():
            continue
        hand = hand_path_for(sequence, frame)
        cad = cad_path_for(sequence)
        if hand is None or cad is None:
            continue
        candidates.append(
            {
                "sequence": str(sequence),
                "frame": frame,
                "rgb": str(rgb),
                "hand": str(hand),
                "cad": str(cad),
            }
        )
    return candidates


def case_id_for(sample: dict[str, str]) -> str:
    sequence = Path(sample["sequence"])
    parts = sequence.relative_to(DATASET).parts
    seq_id = "_".join(parts)
    return f"random_{seq_id}_frame{int(sample['frame']):05d}"


def run_one(sample: dict[str, str], robot: str, output_root: Path, restarts: int, max_nfev: int) -> dict:
    case_id = case_id_for(sample)
    cmd = [
        str(PYTHON if PYTHON.exists() else sys.executable),
        str(POSITION),
        "--sequence",
        sample["sequence"],
        "--frame",
        sample["frame"],
        "--rgb",
        sample["rgb"],
        "--cad",
        sample["cad"],
        "--hand",
        sample["hand"],
        "--output-root",
        str(output_root),
        "--case-id",
        case_id,
        "--robot-profile",
        robot,
        "--optimization-restarts",
        str(restarts),
        "--max-nfev",
        str(max_nfev),
        "--overwrite",
    ]
    completed = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)
    result_path = output_root / case_id / f"retargeted_{robot}.json"
    entry = {
        "case_id": case_id,
        "robot_profile": robot,
        "returncode": completed.returncode,
        "result_json": str(result_path),
        "scene_html": str(output_root / case_id / f"scene_{robot}.html"),
    }
    if completed.returncode != 0:
        entry["error"] = completed.stderr[-2000:]
        return entry
    data = json.loads(result_path.read_text())
    errors = data["result"].get("fingertip_error") or []
    errors_mm = [value * 1000.0 for value in errors]
    entry.update(
        {
            "status": data["result"].get("status"),
            "optimization_seconds": data.get("optimization_seconds"),
            "mean_error_mm": sum(errors_mm) / len(errors_mm) if errors_mm else None,
            "max_error_mm": max(errors_mm) if errors_mm else None,
            "per_finger_error_mm": errors_mm,
        }
    )
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs" / "runs_random10_position_only")
    parser.add_argument("--robots", nargs="+", default=["shadow_hand", "folding_hand_right", "inspire_hand"])
    parser.add_argument("--optimization-restarts", type=int, default=2)
    parser.add_argument("--max-nfev", type=int, default=80)
    args = parser.parse_args()

    candidates = iter_candidates()
    if len(candidates) < args.count:
        raise SystemExit(f"Only found {len(candidates)} runnable samples, need {args.count}.")
    rng = random.Random(args.seed)
    samples = rng.sample(candidates, args.count)

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "selected_samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")

    summary: list[dict] = []
    total = len(samples) * len(args.robots)
    done = 0
    for sample in samples:
        print(f"\n[{case_id_for(sample)}] frame {sample['frame']}")
        for robot in args.robots:
            done += 1
            print(f"  ({done}/{total}) {robot} ...", flush=True)
            entry = run_one(sample, robot, args.output_root, args.optimization_restarts, args.max_nfev)
            summary.append({**sample, **entry})
            status = entry.get("status", "failed")
            mean_error = entry.get("mean_error_mm")
            mean_text = f"{mean_error:.2f} mm" if mean_error is not None else "n/a"
            print(f"    {status}, mean={mean_text}")
            (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    failures = [item for item in summary if item.get("returncode") != 0 or item.get("status") != "ok"]
    print(f"\nDone: {len(summary) - len(failures)}/{len(summary)} successful")
    print(f"Summary: {args.output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
