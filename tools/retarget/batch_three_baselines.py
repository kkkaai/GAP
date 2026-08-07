#!/usr/bin/env python3
"""Run the three retargeting baselines on the saved random-10 HOI4D samples."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "extracted_dataset_sampled"
DEFAULT_SAMPLES = REPO_ROOT / "outputs" / "runs_random10_position_only" / "selected_samples.json"
POSITION = REPO_ROOT / "tools" / "retarget" / "position.py"
POSITION_FORCE = REPO_ROOT / "tools" / "retarget" / "position_force.py"
CONTACT_HEATMAP = REPO_ROOT / "tools" / "retarget" / "contact_heatmap.py"
MANO_ROOT = REPO_ROOT / "mano" / "mano"


def remap_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    parts = path.parts
    if "extracted_dataset_sampled" in parts:
        index = parts.index("extracted_dataset_sampled")
        candidate = REPO_ROOT.joinpath(*parts[index:])
        if candidate.exists():
            return candidate
    return path


def load_samples(path: Path) -> list[dict[str, str]]:
    samples = json.loads(path.read_text(encoding="utf-8"))
    normalized: list[dict[str, str]] = []
    for sample in samples:
        item = dict(sample)
        for key in ("sequence", "rgb", "hand", "cad"):
            item[key] = str(remap_path(str(item[key])))
        normalized.append(item)
    return normalized


def case_id_for(sample: dict[str, str]) -> str:
    sequence = Path(sample["sequence"])
    try:
        rel_parts = sequence.relative_to(DATASET).parts
    except ValueError:
        rel_parts = sequence.parts[-7:]
    return f"random_{'_'.join(rel_parts)}_frame{int(sample['frame']):05d}"


def run_command(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)


def metric_from_json(route: str, path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if route == "position_only":
        result = data.get("result", {})
        errors = result.get("fingertip_error") or []
        errors_mm = [float(value) * 1000.0 for value in errors]
        return {
            "status": result.get("status"),
            "optimization_seconds": data.get("optimization_seconds"),
            "mean_fingertip_error_mm": sum(errors_mm) / len(errors_mm) if errors_mm else None,
            "max_fingertip_error_mm": max(errors_mm) if errors_mm else None,
        }
    if route == "position_force":
        return {
            "status": data.get("status"),
            "position_seconds": (data.get("position_initialization") or {}).get("optimization_seconds"),
            "refine_seconds": (data.get("force_refinement") or {}).get("optimization_seconds"),
            "mean_fingertip_error_mm": (data.get("fingertip_error") or {}).get("mean_m", 0.0) * 1000.0,
            "mean_contact_error_mm": (data.get("contact_error") or {}).get("mean_m", 0.0) * 1000.0,
            "strict_force_closure": (data.get("force_closure") or {}).get("is_force_closure"),
        }
    if route == "contact_heatmap":
        return {
            "status": data.get("status"),
            "position_seconds": (data.get("position_initialization") or {}).get("optimization_seconds"),
            "refine_seconds": (data.get("heatmap_refinement") or {}).get("optimization_seconds"),
            "heatmap_mse": (data.get("heatmap_error") or {}).get("mse"),
            "mean_high_contact_distance_mm": (data.get("heatmap_error") or {}).get("mean_high_contact_distance_m", 0.0)
            * 1000.0,
            "mean_fingertip_error_mm": (data.get("fingertip_error") or {}).get("mean_m", 0.0) * 1000.0,
        }
    return {}


def run_baseline(
    sample: dict[str, str],
    frame_dir: Path,
    route: str,
    robot_profile: str,
    python_exe: str,
    overwrite: bool,
    no_html: bool,
) -> dict[str, Any]:
    output_root = frame_dir
    case_id = route
    common = [
        python_exe,
        str({"position_only": POSITION, "position_force": POSITION_FORCE, "contact_heatmap": CONTACT_HEATMAP}[route]),
        "--sequence",
        sample["sequence"],
        "--frame",
        str(sample["frame"]),
        "--rgb",
        sample["rgb"],
        "--cad",
        sample["cad"],
        "--hand",
        sample["hand"],
        "--mano-root",
        str(MANO_ROOT),
        "--output-root",
        str(output_root),
        "--case-id",
        case_id,
        "--robot-profile",
        robot_profile,
    ]
    if route == "position_only":
        cmd = common + ["--optimization-restarts", "2", "--max-nfev", "60"]
        json_path = frame_dir / route / f"retargeted_{robot_profile}.json"
        html_path = frame_dir / route / f"scene_{robot_profile}.html"
    elif route == "position_force":
        cmd = common + [
            "--position-restarts",
            "2",
            "--position-max-nfev",
            "60",
            "--num-robot-samples",
            "900",
            "--stage2-maxiter",
            "80",
        ]
        json_path = frame_dir / route / f"retargeted_{robot_profile}_position_force.json"
        html_path = frame_dir / route / f"scene_{robot_profile}_position_force.html"
    else:
        cmd = common + [
            "--position-restarts",
            "2",
            "--position-max-nfev",
            "60",
            "--num-object-samples",
            "256",
            "--num-mano-samples",
            "900",
            "--num-robot-samples",
            "520",
            "--maxiter",
            "25",
            "--fingertip-prior-weight",
            "20",
        ]
        json_path = frame_dir / route / f"retargeted_{robot_profile}_contact_heatmap.json"
        html_path = frame_dir / route / f"scene_{robot_profile}_contact_heatmap.html"
    if overwrite:
        cmd.append("--overwrite")
    if no_html:
        cmd.append("--no-html")

    started_at = time.perf_counter()
    completed = run_command(cmd, REPO_ROOT)
    elapsed = time.perf_counter() - started_at
    entry: dict[str, Any] = {
        "route": route,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "result_json": str(json_path),
        "scene_html": str(html_path),
        "command": cmd,
    }
    if completed.returncode != 0:
        entry["error"] = (completed.stderr or completed.stdout)[-4000:]
    else:
        entry.update(metric_from_json(route, json_path))
    return entry


def run_sample_baselines(
    sample_index: int,
    sample_count: int,
    sample: dict[str, str],
    output_root: str,
    routes: list[str],
    robot_profile: str,
    python_exe: str,
    overwrite: bool,
    no_html: bool,
) -> dict[str, Any]:
    frame_id = case_id_for(sample)
    frame_dir = Path(output_root) / frame_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for route in routes:
        entry = run_baseline(sample, frame_dir, route, robot_profile, python_exe, overwrite, no_html)
        entries.append(entry)
    write_frame_index(frame_dir, sample, entries)
    return {
        "sample_index": sample_index,
        "sample_count": sample_count,
        "frame_id": frame_id,
        "sample": sample,
        "entries": entries,
    }


def write_frame_index(frame_dir: Path, sample: dict[str, str], entries: list[dict[str, Any]]) -> None:
    (frame_dir / "frame_index.json").write_text(
        json.dumps({"sample": sample, "baselines": entries}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    original = frame_dir / "original.jpg"
    rgb = Path(sample["rgb"])
    if rgb.exists() and not original.exists():
        shutil.copy2(rgb, original)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs" / "runs_random10_three_baselines")
    parser.add_argument("--robot-profile", default="folding_hand_right")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--routes",
        nargs="+",
        choices=["position_only", "position_force", "contact_heatmap"],
        default=["position_only", "position_force", "contact_heatmap"],
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-html", action="store_true", help="Skip per-case Plotly HTML rendering in baseline scripts.")
    parser.add_argument("--workers", type=int, default=1, help="Number of sample-level worker processes.")
    args = parser.parse_args()

    samples = load_samples(args.samples)
    if args.limit is not None:
        samples = samples[: args.limit]
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "selected_samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")

    summary: list[dict[str, Any]] = []
    total = len(samples) * len(args.routes)
    done = 0
    if args.workers <= 1:
        for sample_index, sample in enumerate(samples, start=1):
            frame_id = case_id_for(sample)
            frame_dir = args.output_root / frame_id
            frame_dir.mkdir(parents=True, exist_ok=True)
            entries: list[dict[str, Any]] = []
            print(f"\n[{sample_index}/{len(samples)}] {frame_id}")
            for route in args.routes:
                done += 1
                print(f"  ({done}/{total}) {route} ...", flush=True)
                entry = run_baseline(
                    sample,
                    frame_dir,
                    route,
                    args.robot_profile,
                    args.python,
                    args.overwrite,
                    args.no_html,
                )
                entries.append(entry)
                status = entry.get("status", "failed" if entry["returncode"] else "done")
                metric = entry.get("mean_fingertip_error_mm") or entry.get("heatmap_mse")
                metric_text = f"{metric:.3f}" if isinstance(metric, (int, float)) else "n/a"
                print(f"    {status}, metric={metric_text}, elapsed={entry['elapsed_seconds']:.1f}s")
                summary.append({"frame_id": frame_id, "sample": sample, **entry})
                (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
                write_frame_index(frame_dir, sample, entries)
    else:
        print(f"Running {len(samples)} samples x {len(args.routes)} routes with {args.workers} workers.")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    run_sample_baselines,
                    sample_index,
                    len(samples),
                    sample,
                    str(args.output_root),
                    list(args.routes),
                    args.robot_profile,
                    args.python,
                    args.overwrite,
                    args.no_html,
                )
                for sample_index, sample in enumerate(samples, start=1)
            ]
            for future in as_completed(futures):
                result = future.result()
                done += len(result["entries"])
                print(f"\n[{result['sample_index']}/{result['sample_count']}] {result['frame_id']} ({done}/{total})")
                for entry in result["entries"]:
                    status = entry.get("status", "failed" if entry["returncode"] else "done")
                    metric = entry.get("mean_fingertip_error_mm") or entry.get("heatmap_mse")
                    metric_text = f"{metric:.3f}" if isinstance(metric, (int, float)) else "n/a"
                    print(f"  {entry['route']}: {status}, metric={metric_text}, elapsed={entry['elapsed_seconds']:.1f}s")
                    summary.append({"frame_id": result["frame_id"], "sample": result["sample"], **entry})
                summary.sort(key=lambda item: (item["frame_id"], item["route"]))
                (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    failures = [entry for entry in summary if entry.get("returncode") != 0 or entry.get("status") not in {None, "ok"}]
    print(f"\nDone: {len(summary) - len(failures)}/{len(summary)} successful")
    print(f"Output root: {args.output_root}")
    print(f"Summary: {args.output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
