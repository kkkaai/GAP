#!/usr/bin/env python3
"""Build a q_goal CFM JSONL dataset from retargeting summaries."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def as_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: as_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(item) for item in value]
    return value


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_lollipop_params(path: Path) -> dict[str, float]:
    raw = read_json(path)
    width = float(raw.get("image_width", 640) or 640)
    height = float(raw.get("image_height", 480) or 480)
    scale = max(width, height)
    center = raw.get("palm_center_xy", [0.0, 0.0])
    tip = raw.get("tip_xy", [0.0, 0.0])
    radius = float(raw.get("palm_radius", 0.0))
    arm_width = float(raw.get("arm_width", 0.0))
    dx = float(tip[0]) - float(center[0])
    dy = float(tip[1]) - float(center[1])
    arm_length = float((dx * dx + dy * dy) ** 0.5)
    return {
        "center_x": float(center[0]) / max(width, 1.0),
        "center_y": float(center[1]) / max(height, 1.0),
        "theta": float(raw.get("theta_rad", 0.0)),
        "palm_radius": radius / max(scale, 1.0),
        "arm_length": arm_length / max(scale, 1.0),
        "confidence": float(raw.get("priority", 1.0)),
    }


def route_quality(mean_fingertip_error_m: float, *, fc_bonus: float = 0.0) -> float:
    return max(0.0, min(1.0, 1.0 - mean_fingertip_error_m / 0.15 + fc_bonus))


def _result_payload(item: dict[str, Any]) -> dict[str, Any]:
    result_json = item.get("result_json")
    if not result_json:
        return {}
    path = REPO_ROOT / result_json if not Path(result_json).is_absolute() else Path(result_json)
    raw = read_json(path)
    if isinstance(raw.get("result"), dict):
        return raw["result"]
    return raw


def _route_action(item: dict[str, Any]) -> list[float]:
    payload = _result_payload(item)
    action = payload.get("action")
    if action is None:
        raise ValueError(f"No action found for {item.get('frame_id')} {item.get('route')}.")
    return [float(v) for v in action]


def _route_action_names(item: dict[str, Any]) -> list[str]:
    payload = _result_payload(item)
    return [str(v) for v in payload.get("action_names", [])]


def _route_error_m(item: dict[str, Any]) -> float:
    if item.get("mean_fingertip_error_mm") is not None:
        return float(item["mean_fingertip_error_mm"]) / 1000.0
    payload = _result_payload(item)
    if payload.get("mean_fingertip_error_m") is not None:
        return float(payload["mean_fingertip_error_m"])
    fingertip_error = payload.get("fingertip_error")
    if isinstance(fingertip_error, list) and fingertip_error:
        return float(np.mean(np.asarray(fingertip_error, dtype=np.float64)))
    raise ValueError(f"No fingertip error found for {item.get('frame_id')} {item.get('route')}.")


def _group_by_frame(summary: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for item in summary:
        if item.get("returncode") != 0 or item.get("status") not in (None, "ok"):
            continue
        frame_id = str(item.get("frame_id") or item.get("case") or "")
        route = str(item.get("route") or "")
        if frame_id and route:
            grouped.setdefault(frame_id, {})[route] = item
    return grouped


def _selected_items(summary: list[dict[str, Any]], route: str) -> list[dict[str, Any]]:
    if route != "best":
        return [
            item
            for item in summary
            if item.get("returncode") == 0 and item.get("status") in (None, "ok") and item.get("route") == route
        ]
    selected: list[dict[str, Any]] = []
    for route_items in _group_by_frame(summary).values():
        candidates = [item for item in route_items.values() if item.get("route") in {"position_only", "contact_heatmap"}]
        if candidates:
            selected.append(min(candidates, key=_route_error_m))
    return selected


def _frame_dir_from_sample(item: dict[str, Any], dataset_root: Path) -> Path | None:
    sample = item.get("sample")
    if not isinstance(sample, dict):
        return None
    case_id = sample.get("case_id")
    if not case_id:
        return None
    return dataset_root / "frames" / str(case_id)


def _build_hoi4d_record(item: dict[str, Any], dataset_root: Path, q_min: np.ndarray, q_max: np.ndarray) -> dict[str, Any]:
    frame_dir = _frame_dir_from_sample(item, dataset_root)
    if frame_dir is None or not frame_dir.exists():
        raise FileNotFoundError(f"Cannot resolve HOI4D frame directory for {item.get('frame_id')}.")
    params_path = frame_dir / "lollipop_params.json"
    meta_path = frame_dir / "sample_metadata.json"
    params_raw = read_json(params_path)
    meta = read_json(meta_path) if meta_path.exists() else {}
    route = str(item["route"])
    mean_error_m = _route_error_m(item)
    return {
        "sample_id": meta.get("case_id") or params_raw.get("candidate_id") or item["frame_id"],
        "source_domain": "hoi4d",
        "rgb_path": str((frame_dir / "rgb.png").resolve()),
        # Kept for inspection only. Use --lollipop-mask-encoder none to train with parametric lollipop only.
        "lollipop_mask_path": str((frame_dir / "lollipop_mask.png").resolve()),
        "task": str(params_raw.get("task") or meta.get("task") or f"grasp the {meta.get('category', 'object')}".lower()),
        "grasp_type": "grasp",
        "affordance_part": "object",
        "lollipop_params": load_lollipop_params(params_path),
        "q_current": [0.0] * len(q_min),
        "q_goal": _route_action(item),
        "q_min": q_min,
        "q_max": q_max,
        "quality": route_quality(mean_error_m),
        "metadata": {
            "source_summary": str((REPO_ROOT / item["result_json"]).resolve()) if not Path(item["result_json"]).is_absolute() else item["result_json"],
            "teacher_route": route,
            "robot_profile": item.get("robot_profile") or _result_payload(item).get("robot_profile"),
            "action_names": _route_action_names(item),
            "retarget_frame_id": item["frame_id"],
            "mean_fingertip_error_m": mean_error_m,
            "category": meta.get("category"),
            "side": meta.get("side"),
            "sequence": meta.get("rel_sequence"),
            "frame": meta.get("frame"),
        },
    }


def _build_kettle_record(
    item: dict[str, Any],
    route: str,
    q_min: np.ndarray,
    q_max: np.ndarray,
    summary_path: Path,
) -> dict[str, Any]:
    ldir = REPO_ROOT / "outputs/0713test_phase1_4_vlm_qwen37_test" / item["sample_id"] / item["lollipop"]
    params_path = ldir / "lollipop_params.json"
    params_raw = read_json(params_path)
    task = str(params_raw.get("task") or params_raw.get("grasp_type") or item["lollipop"])
    mean_error_m = float(item[route]["mean_fingertip_error_m"])
    fc_bonus = 0.1 if route == "position_force" and item[route].get("force_closure", {}).get("is_force_closure") else 0.0
    return {
        "sample_id": item["sample_id"],
        "source_domain": "0713test",
        "lollipop": item["lollipop"],
        "rgb_path": str((ldir / "phase4_inpaint_full.png").resolve()),
        "lollipop_mask_path": str((ldir / "lollipop_mask.png").resolve()),
        "task": task,
        "grasp_type": params_raw.get("grasp_type", ""),
        "affordance_part": params_raw.get("affordance_part", ""),
        "lollipop_params": load_lollipop_params(params_path),
        "q_current": [0.0] * len(q_min),
        "q_goal": item[route]["action"],
        "q_min": q_min,
        "q_max": q_max,
        "quality": route_quality(mean_error_m, fc_bonus=fc_bonus),
        "metadata": {
            "source_summary": str(summary_path.resolve()),
            "teacher_route": route,
            "robot_profile": item.get("robot_profile"),
            "action_names": item.get("position_only", {}).get("action_names", []),
            "retarget_case": item["case"],
            "mean_fingertip_error_m": mean_error_m,
        },
    }


def build_records(summary_path: Path, route: str, q_bounds: str, dataset_root: Path | None = None) -> list[dict[str, Any]]:
    summary = read_json(summary_path)
    ok = _selected_items(summary, route) if dataset_root is not None else [
        item for item in summary if item.get("status") != "failed" and route in item
    ]
    if not ok:
        raise ValueError(f"No usable {route} records found in {summary_path}.")
    actions = np.asarray([_route_action(item) if dataset_root is not None else item[route]["action"] for item in ok], dtype=np.float64)
    if q_bounds == "unit":
        q_min = np.zeros(actions.shape[1], dtype=np.float64)
        q_max = np.ones(actions.shape[1], dtype=np.float64)
        q_max = np.maximum(q_max, actions.max(axis=0) + 0.05)
    elif q_bounds == "observed":
        margin = np.maximum(0.05, 0.1 * np.maximum(actions.max(axis=0) - actions.min(axis=0), 1e-6))
        q_min = actions.min(axis=0) - margin
        q_max = actions.max(axis=0) + margin
    else:
        raise ValueError(f"Unknown q_bounds {q_bounds!r}.")

    records: list[dict[str, Any]] = []
    for item in ok:
        if dataset_root is not None:
            records.append(_build_hoi4d_record(item, dataset_root, q_min, q_max))
        else:
            records.append(_build_kettle_record(item, route, q_min, q_max, summary_path))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(as_jsonable(record), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=REPO_ROOT
        / "outputs/0713test_phase1_4_vlm_qwen37_test/kettle_retarget_three_baselines_hamer_official_mixed12_phase55_se3/summary.json",
    )
    parser.add_argument("--route", choices=["position_only", "position_force", "contact_heatmap", "best"], default="contact_heatmap")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Unified HOI4D root containing frames/<case_id>. If set, parse batch summary rows instead of old kettle summary.",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/q_goal/0713test_kettle_smoke")
    parser.add_argument("--target-size", type=int, default=1000)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--q-bounds", choices=["observed", "unit"], default="observed")
    args = parser.parse_args()

    base = build_records(args.summary_json, args.route, args.q_bounds, args.dataset_root)
    rng = random.Random(args.seed)
    records = [dict(base[i % len(base)], replica_index=i // len(base)) for i in range(max(args.target_size, len(base)))]
    rng.shuffle(records)
    n_val = max(1, int(round(len(records) * args.val_fraction)))
    val = records[:n_val]
    train = records[n_val:]
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "val.jsonl", val)
    meta = {
        "summary_json": args.summary_json,
        "dataset_root": args.dataset_root,
        "route": args.route,
        "q_bounds": args.q_bounds,
        "base_records": len(base),
        "train_records": len(train),
        "val_records": len(val),
        "target_size": args.target_size,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(as_jsonable(meta), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(as_jsonable(meta), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
