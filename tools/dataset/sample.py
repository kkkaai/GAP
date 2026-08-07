#!/usr/bin/env python3
"""Extract a tiny 5-frame sample from one local HOI4D sequence.

The script expects a sequence directory like:
  ZY20210800004/H4/C8/N14/S71/s03/T2

It does not download data. It copies only files that already exist and records
missing optional fields in metadata.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


VIDEO_NAMES = {
    "rgb": Path("align_rgb/image.mp4"),
    "depth": Path("align_depth/depth_video.avi"),
}

DECODED_FRAME_DIRS = {
    "rgb": Path("align_rgb"),
    "depth": Path("align_depth"),
}

MASK_DIR_CANDIDATES = [
    Path("2Dseg/mask"),
    Path("2Dseg"),
    Path("mask"),
]

ANNOTATION_DIRS = {
    "objpose": Path("objpose"),
    "action": Path("action"),
    "scene_3dseg": Path("3Dseg"),
    "handpose": Path("handpose"),
    "hand_pose": Path("hand_pose"),
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
MESH_OR_POSE_EXTS = {".obj", ".ply", ".pcd", ".pickle", ".pkl", ".json", ".txt", ".log"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 5 frames and available annotations from one HOI4D sequence."
    )
    parser.add_argument(
        "--sequence",
        required=True,
        type=Path,
        help="Path to one HOI4D sequence directory ending in .../T*.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hoi4d_sample_5frames"),
        help="Output directory to create. Default: ./hoi4d_sample_5frames",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=5,
        help="Number of frame indices to extract. Default: 5",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting frame index. Default: 0",
    )
    parser.add_argument(
        "--task-definitions",
        type=Path,
        default=Path("HOI4D-Instructions/definitions/task/task_definitions.csv"),
        help="CSV used to resolve C*/T* labels when available.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args()


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def sequence_parts(sequence: Path) -> dict[str, str]:
    parts: dict[str, str] = {}
    for part in sequence.parts:
        if len(part) >= 2 and part[0] in {"H", "C", "N", "S", "T"} and part[1:].isdigit():
            parts[part[0]] = part
        elif len(part) >= 2 and part[0] == "s" and part[1:].isdigit():
            parts["layout"] = part
        elif part.startswith("ZY"):
            parts["camera"] = part
    return parts


def load_task_label(csv_path: Path, category_id: str | None, task_id: str | None) -> str | None:
    if not csv_path.exists() or not category_id or not task_id:
        return None

    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("Category ID") == category_id:
                return row.get(task_id) or None
    return None


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def frame_name_candidates(frame_idx: int) -> list[str]:
    return [
        f"{frame_idx:05d}",
        f"{frame_idx:06d}",
        f"{frame_idx}",
    ]


def find_decoded_frame(directory: Path, frame_idx: int, exts: set[str] = IMAGE_EXTS) -> Path | None:
    for stem in frame_name_candidates(frame_idx):
        for ext in sorted(exts):
            candidate = directory / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    return None


def extract_video_frame(video: Path, frame_idx: int, dst: Path) -> bool:
    if not video.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"select=eq(n\\,{frame_idx})",
        "-frames:v",
        "1",
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return dst.exists()


def copy_or_extract_frame(sequence: Path, kind: str, frame_idx: int, output_frame_dir: Path) -> str | None:
    decoded_dir = sequence / DECODED_FRAME_DIRS[kind]
    decoded_frame = find_decoded_frame(decoded_dir, frame_idx)
    suffix = ".jpg" if kind == "rgb" else ".png"
    dst = output_frame_dir / f"{kind}{suffix}"

    if decoded_frame and copy_if_exists(decoded_frame, dst.with_suffix(decoded_frame.suffix)):
        return str(dst.with_suffix(decoded_frame.suffix))

    video = sequence / VIDEO_NAMES[kind]
    if extract_video_frame(video, frame_idx, dst):
        return str(dst)
    return None


def copy_mask(sequence: Path, frame_idx: int, output_frame_dir: Path) -> str | None:
    for mask_dir in MASK_DIR_CANDIDATES:
        source_dir = sequence / mask_dir
        mask = find_decoded_frame(source_dir, frame_idx)
        if mask:
            dst = output_frame_dir / f"mask{mask.suffix}"
            copy_if_exists(mask, dst)
            return str(dst)
    return None


def copy_annotation_files(sequence: Path, frame_idx: int, output_frame_dir: Path) -> list[str]:
    copied: list[str] = []
    for label, relative_dir in ANNOTATION_DIRS.items():
        src_dir = sequence / relative_dir
        if not src_dir.exists():
            continue

        candidates: list[Path] = []
        for stem in frame_name_candidates(frame_idx):
            candidates.extend(
                p for p in src_dir.rglob(f"{stem}.*") if p.is_file() and p.suffix.lower() in MESH_OR_POSE_EXTS
            )

        if not candidates and label in {"action", "scene_3dseg"}:
            candidates = [p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in MESH_OR_POSE_EXTS]

        for src in sorted(set(candidates)):
            rel = src.relative_to(src_dir)
            dst = output_frame_dir / label / rel
            copy_if_exists(src, dst)
            copied.append(str(dst))
    return copied


def main() -> None:
    args = parse_args()
    sequence = args.sequence.resolve()
    output = args.output.resolve()

    if not sequence.exists():
        raise FileNotFoundError(f"Sequence path does not exist: {sequence}")
    if not sequence.is_dir():
        raise NotADirectoryError(f"Sequence path must be a directory: {sequence}")

    ensure_output_dir(output, args.overwrite)

    parts = sequence_parts(sequence)
    action_json = load_json(sequence / "action/color.json")
    metadata: dict[str, Any] = {
        "source_sequence": str(sequence),
        "sequence_parts": parts,
        "task_label": load_task_label(args.task_definitions, parts.get("C"), parts.get("T")),
        "action_color_json": action_json,
        "requested_frames": args.frames,
        "start_frame": args.start,
        "frames": [],
        "missing_fields": [],
    }

    for frame_idx in range(args.start, args.start + args.frames):
        frame_dir = output / f"frame_{frame_idx:05d}"
        frame_dir.mkdir()

        frame_record: dict[str, Any] = {"frame_index": frame_idx}
        for kind in ("rgb", "depth"):
            extracted = copy_or_extract_frame(sequence, kind, frame_idx, frame_dir)
            frame_record[kind] = extracted
            if extracted is None:
                metadata["missing_fields"].append({"frame": frame_idx, "field": kind})

        mask = copy_mask(sequence, frame_idx, frame_dir)
        frame_record["mask"] = mask
        if mask is None:
            metadata["missing_fields"].append({"frame": frame_idx, "field": "2Dseg/mask"})

        annotations = copy_annotation_files(sequence, frame_idx, frame_dir)
        frame_record["annotations"] = annotations
        if not annotations:
            metadata["missing_fields"].append({"frame": frame_idx, "field": "annotations/meshes/poses"})

        metadata["frames"].append(frame_record)

    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote sample to {output}")


if __name__ == "__main__":
    main()
