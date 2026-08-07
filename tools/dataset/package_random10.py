#!/usr/bin/env python3
"""Package the random-10 retargeting subset and project outputs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "extracted_dataset_sampled"
DEFAULT_SELECTED = REPO_ROOT / "outputs" / "runs_random10_position_only" / "selected_samples.json"


def copy_file_preserve_root(src: Path, dst_root: Path) -> None:
    src = src.resolve()
    rel = src.relative_to(REPO_ROOT)
    dst = dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst_root: Path) -> None:
    dst = dst_root / src.relative_to(REPO_ROOT)
    shutil.copytree(
        src,
        dst,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )


def frame_sidecars(sequence: Path, frame: str) -> list[Path]:
    frame_i = int(frame)
    candidates = [
        sequence / "align_rgb" / f"{frame_i:05d}.jpg",
        sequence / "objpose" / f"{frame_i}.json",
        sequence / "2Dseg" / "shift_mask" / f"{frame_i}.png",
        sequence / "2Dseg" / "shift_mask" / f"{frame_i:05d}.png",
    ]
    return [path for path in candidates if path.exists()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--stage-root", type=Path, default=REPO_ROOT / "package_staging")
    parser.add_argument("--name", default="hoi4d_random10_position_only_package")
    args = parser.parse_args()

    package_root = args.stage_root / args.name
    package_root.mkdir(parents=True, exist_ok=True)

    selected = json.loads(args.selected.read_text())
    copied: set[Path] = set()

    for sample in selected:
        sequence = Path(sample["sequence"])
        frame = str(sample["frame"])
        for path in frame_sidecars(sequence, frame):
            copied.add(path.resolve())
        for key in ("hand", "cad"):
            copied.add(Path(sample[key]).resolve())

    for path in sorted(copied):
        copy_file_preserve_root(path, package_root)

    copy_file_preserve_root(args.selected, package_root)
    summary = args.selected.parent / "summary.json"
    if summary.exists():
        copy_file_preserve_root(summary, package_root)

    for dirname in ("outputs", "third_party", "tools"):
        copy_tree(REPO_ROOT / dirname, package_root)

    archive_base = REPO_ROOT / args.name
    archive = shutil.make_archive(str(archive_base), "zip", root_dir=args.stage_root, base_dir=args.name)
    print(archive)


if __name__ == "__main__":
    main()
