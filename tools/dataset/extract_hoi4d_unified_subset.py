#!/usr/bin/env python3
"""Selectively extract a unified HOI4D subset from local zip archives.

This script intentionally does not unpack the full HOI4D archives. It scans zip
indices, samples valid frame-level candidates, then extracts only the files
needed for retargeting and q_goal CFM training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE_DIR = Path("/home/kai/下载")

CATEGORY_BY_ID = {
    "C1": "ToyCar",
    "C2": "Mug",
    "C3": "Laptop",
    "C4": "StorageFurniture",
    "C5": "Bottle",
    "C6": "Safe",
    "C7": "Bowl",
    "C8": "Bucket",
    "C9": "Scissors",
    "C11": "Pliers",
    "C12": "Kettle",
    "C13": "Knife",
    "C14": "TrashCan",
    "C17": "Lamp",
    "C20": "Chair",
}

DEFAULT_CATEGORIES = [
    "ToyCar",
    "Mug",
    "Bottle",
    "Bowl",
    "Bucket",
    "Scissors",
    "Pliers",
    "Kettle",
    "Knife",
    "Lamp",
]


@dataclass(frozen=True)
class Candidate:
    case_id: str
    rel_sequence: str
    frame: int
    side: str
    category_id: str
    category: str
    instance_id: str
    release_video_entry: str
    hand_entry: str
    objpose_entry: str
    mask_entry: str | None
    action_entry: str | None
    cad_entry: str
    camera_entry: str | None


def as_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {key: as_jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {key: as_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(item) for item in value]
    return value


def parse_sequence_parts(rel_sequence: str) -> dict[str, str]:
    parts: dict[str, str] = {}
    for part in Path(rel_sequence).parts:
        if part.startswith("ZY"):
            parts["subject"] = part
        elif len(part) >= 2 and part[0] in {"H", "C", "N", "S", "T"} and part[1:].isdigit():
            parts[part[0]] = part
        elif len(part) >= 2 and part[0] == "s" and part[1:].isdigit():
            parts["layout"] = part
    return parts


def case_id_for(rel_sequence: str, frame: int, side: str) -> str:
    return f"{rel_sequence.replace('/', '__')}__frame{frame:05d}__{side}"


def strip_prefix(name: str, prefix: str) -> str:
    if not name.startswith(prefix):
        raise ValueError(f"{name!r} does not start with {prefix!r}")
    return name[len(prefix) :]


def frame_stems(frame: int) -> list[str]:
    return [str(frame), f"{frame:05d}", f"{frame:06d}"]


def build_zip_sets(
    release_zip: zipfile.ZipFile,
    annotations_zip: zipfile.ZipFile,
    hand_zip: zipfile.ZipFile,
    cad_zip: zipfile.ZipFile,
    camera_zip: zipfile.ZipFile | None,
) -> dict[str, Any]:
    print("Indexing release zip...", flush=True)
    release_videos = {
        strip_prefix(name, "HOI4D_release/").removesuffix("/align_rgb/image.mp4"): name
        for name in release_zip.namelist()
        if name.endswith("/align_rgb/image.mp4")
    }
    print(f"  videos: {len(release_videos)}", flush=True)

    print("Indexing annotations zip...", flush=True)
    objposes: set[str] = set()
    masks: set[str] = set()
    actions: set[str] = set()
    for name in annotations_zip.namelist():
        rel = strip_prefix(name, "HOI4D_annotations/")
        if "/objpose/" in rel and rel.endswith(".json"):
            objposes.add(rel)
        elif "/2Dseg/mask/" in rel and rel.endswith(".png"):
            masks.add(rel)
        elif rel.endswith("/action/color.json"):
            actions.add(rel)
    print(f"  objposes: {len(objposes)}, masks: {len(masks)}, actions: {len(actions)}", flush=True)

    print("Indexing CAD zip...", flush=True)
    cad_entries = {
        strip_prefix(name, "HOI4D_CAD_Model_for_release/"): name
        for name in cad_zip.namelist()
        if name.startswith("HOI4D_CAD_Model_for_release/rigid/") and name.endswith(".obj")
    }
    print(f"  rigid CAD objects: {len(cad_entries)}", flush=True)

    camera_entries: dict[str, str] = {}
    if camera_zip is not None:
        camera_entries = {
            Path(name).parts[1]: name
            for name in camera_zip.namelist()
            if name.startswith("camera_params/") and name.endswith("/intrin.npy")
        }
    print(f"  camera intrinsics: {len(camera_entries)}", flush=True)

    print("Indexing hand pose zip...", flush=True)
    hand_entries = [name for name in hand_zip.namelist() if name.endswith(".pickle")]
    print(f"  hand pickles: {len(hand_entries)}", flush=True)
    return {
        "release_videos": release_videos,
        "objposes": objposes,
        "masks": masks,
        "actions": actions,
        "cad_entries": cad_entries,
        "camera_entries": camera_entries,
        "hand_entries": hand_entries,
        "hand_entry_set": set(hand_entries),
    }


def candidate_from_hand_entry(name: str, indices: dict[str, Any], allowed_categories: set[str]) -> Candidate | None:
    parts = Path(name).parts
    if len(parts) < 10 or parts[0] != "Hand_pose":
        return None
    side_dir = parts[1]
    if side_dir not in {"handpose_left_hand", "handpose_right_hand"}:
        return None
    side = "left" if "left" in side_dir else "right"
    rel_sequence = str(Path(*parts[2:-1]))
    try:
        frame = int(Path(parts[-1]).stem)
    except ValueError:
        return None
    seq = parse_sequence_parts(rel_sequence)
    category_id = seq.get("C", "")
    category = CATEGORY_BY_ID.get(category_id)
    if category is None or category not in allowed_categories:
        return None
    instance_id = seq.get("N", "")
    if not instance_id:
        return None
    release_video_entry = indices["release_videos"].get(rel_sequence)
    if release_video_entry is None:
        return None
    objpose_entry_rel = f"{rel_sequence}/objpose/{frame}.json"
    if objpose_entry_rel not in indices["objposes"]:
        return None
    cad_rel = f"rigid/{category}/{int(instance_id[1:]):03d}.obj"
    cad_entry = indices["cad_entries"].get(cad_rel)
    if cad_entry is None:
        return None
    mask_entry = None
    for stem in frame_stems(frame):
        rel = f"{rel_sequence}/2Dseg/mask/{stem}.png"
        if rel in indices["masks"]:
            mask_entry = f"HOI4D_annotations/{rel}"
            break
    action_entry = None
    action_rel = f"{rel_sequence}/action/color.json"
    if action_rel in indices["actions"]:
        action_entry = f"HOI4D_annotations/{action_rel}"
    subject = seq.get("subject", "")
    camera_entry = indices["camera_entries"].get(subject)
    return Candidate(
        case_id=case_id_for(rel_sequence, frame, side),
        rel_sequence=rel_sequence,
        frame=frame,
        side=side,
        category_id=category_id,
        category=category,
        instance_id=instance_id,
        release_video_entry=release_video_entry,
        hand_entry=name,
        objpose_entry=f"HOI4D_annotations/{objpose_entry_rel}",
        mask_entry=mask_entry,
        action_entry=action_entry,
        cad_entry=cad_entry,
        camera_entry=camera_entry,
    )


def candidate_from_index_row(
    rel_sequence: str,
    frame: int,
    side: str,
    indices: dict[str, Any],
    allowed_categories: set[str],
) -> Candidate | None:
    side_dir = f"handpose_{side}_hand"
    hand_entry = f"Hand_pose/{side_dir}/{rel_sequence}/{frame}.pickle"
    if hand_entry not in indices["hand_entry_set"]:
        return None
    return candidate_from_hand_entry(hand_entry, indices, allowed_categories)


def sample_candidates(
    indices: dict[str, Any],
    allowed_categories: set[str],
    count: int,
    seed: int,
    max_per_sequence: int,
) -> list[Candidate]:
    rng = random.Random(seed)
    hand_entries = list(indices["hand_entries"])
    rng.shuffle(hand_entries)
    selected: list[Candidate] = []
    sequence_counts: dict[str, int] = {}
    seen_case_ids: set[str] = set()
    for entry in hand_entries:
        candidate = candidate_from_hand_entry(entry, indices, allowed_categories)
        if candidate is None:
            continue
        if candidate.case_id in seen_case_ids:
            continue
        if sequence_counts.get(candidate.rel_sequence, 0) >= max_per_sequence:
            continue
        selected.append(candidate)
        seen_case_ids.add(candidate.case_id)
        sequence_counts[candidate.rel_sequence] = sequence_counts.get(candidate.rel_sequence, 0) + 1
        if len(selected) >= count:
            break
    if len(selected) < count:
        raise RuntimeError(f"Only found {len(selected)} candidates, requested {count}.")
    return selected


def sample_candidates_from_index_csv(
    index_csv: Path,
    indices: dict[str, Any],
    allowed_categories: set[str],
    count: int,
    seed: int,
    max_per_sequence: int,
    sides: list[str],
) -> list[Candidate]:
    rng = random.Random(seed)
    with index_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rng.shuffle(rows)

    selected: list[Candidate] = []
    sequence_counts: dict[str, int] = {}
    seen_case_ids: set[str] = set()
    for row in rows:
        rel_sequence = row.get("vid_index", "")
        if not rel_sequence:
            continue
        try:
            frame = int(row.get("frame_number", ""))
        except ValueError:
            continue
        row_sides = list(sides)
        rng.shuffle(row_sides)
        for side in row_sides:
            candidate = candidate_from_index_row(
                rel_sequence=rel_sequence,
                frame=frame,
                side=side,
                indices=indices,
                allowed_categories=allowed_categories,
            )
            if candidate is None:
                continue
            if candidate.case_id in seen_case_ids:
                continue
            if sequence_counts.get(candidate.rel_sequence, 0) >= max_per_sequence:
                continue
            selected.append(candidate)
            seen_case_ids.add(candidate.case_id)
            sequence_counts[candidate.rel_sequence] = sequence_counts.get(candidate.rel_sequence, 0) + 1
            break
        if len(selected) >= count:
            break
    if len(selected) < count:
        raise RuntimeError(
            f"Only found {len(selected)} indexed candidates, requested {count}. "
            f"Try increasing --max-per-sequence or broadening --categories/--sides."
        )
    return selected


def extract_zip_member(zf: zipfile.ZipFile, member: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as src, dst.open("wb") as out:
        shutil.copyfileobj(src, out)


def extract_frame_from_video(video_path: Path, frame: int, dst: Path, ffmpeg_bin: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ffmpeg_bin),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"select=eq(n\\,{frame})",
        "-frames:v",
        "1",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    if not dst.exists():
        raise RuntimeError(f"ffmpeg did not create {dst}")


def valid_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2:
        return np.empty((0, 2), dtype=np.float64)
    points = points[:, :2]
    ok = np.isfinite(points).all(axis=1)
    ok &= (points[:, 0] > 1.0) & (points[:, 1] > 1.0)
    return points[ok]


def lollipop_from_hand(hand_data: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    points = valid_points(np.asarray(hand_data.get("kps2D", []), dtype=np.float64))
    if len(points) == 0:
        center = np.array([width * 0.5, height * 0.5], dtype=np.float64)
        wrist = np.array([width * 0.5, height * 0.95], dtype=np.float64)
        radius = min(width, height) * 0.12
    else:
        if len(points) >= 18:
            palm_indices = [0, 5, 9, 13, 17]
            palm_points = np.asarray(hand_data.get("kps2D"), dtype=np.float64)[palm_indices, :2]
            palm_points = valid_points(palm_points)
            center = palm_points.mean(axis=0) if len(palm_points) else points.mean(axis=0)
            wrist = np.asarray(hand_data.get("kps2D"), dtype=np.float64)[0, :2]
            if not np.isfinite(wrist).all():
                wrist = points[np.argmax(points[:, 1])]
        else:
            center = points.mean(axis=0)
            wrist = points[np.argmax(points[:, 1])]
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        radius = max(28.0, 0.32 * float(max(maxs - mins)))
    direction = wrist - center
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        direction = np.array([0.0, 1.0], dtype=np.float64)
    else:
        direction = direction / norm
    candidates: list[tuple[float, np.ndarray]] = []
    if abs(direction[0]) > 1e-6:
        for x in (0.0, float(width - 1)):
            t = (x - center[0]) / direction[0]
            y = center[1] + t * direction[1]
            if t > 0 and 0 <= y <= height - 1:
                candidates.append((t, np.array([x, y], dtype=np.float64)))
    if abs(direction[1]) > 1e-6:
        for y in (0.0, float(height - 1)):
            t = (y - center[1]) / direction[1]
            x = center[0] + t * direction[0]
            if t > 0 and 0 <= x <= width - 1:
                candidates.append((t, np.array([x, y], dtype=np.float64)))
    edge = min(candidates, key=lambda item: item[0])[1] if candidates else wrist
    dx, dy = edge - center
    theta = math.atan2(float(dy), float(dx))
    return {
        "palm_center_xy": [float(center[0]), float(center[1])],
        "tip_xy": [float(edge[0]), float(edge[1])],
        "palm_radius": float(radius),
        "arm_width": float(max(20.0, radius * 0.58)),
        "theta_rad": theta,
        "image_width": int(width),
        "image_height": int(height),
        "strategy": "hoi4d_hand_keypoints_boundary_extended",
    }


def draw_lollipop_mask(params: dict[str, Any], dst_mask: Path, dst_overlay: Path, rgb_path: Path) -> None:
    width = int(params["image_width"])
    height = int(params["image_height"])
    center = tuple(params["palm_center_xy"])
    tip = tuple(params["tip_xy"])
    radius = float(params["palm_radius"])
    arm_width = int(round(float(params["arm_width"])))
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.line([tip, center], fill=255, width=arm_width)
    draw.ellipse(
        [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius],
        fill=255,
    )
    dst_mask.parent.mkdir(parents=True, exist_ok=True)
    mask.save(dst_mask)

    rgb = Image.open(rgb_path).convert("RGB")
    overlay = Image.new("RGBA", rgb.size, (0, 0, 0, 0))
    overlay.putalpha(mask.point(lambda p: 96 if p > 0 else 0))
    red = Image.new("RGBA", rgb.size, (255, 80, 0, 0))
    red.putalpha(mask.point(lambda p: 96 if p > 0 else 0))
    composed = Image.alpha_composite(rgb.convert("RGBA"), red)
    composed.save(dst_overlay)


def task_for_category(category: str) -> str:
    if category in {"Mug", "Kettle"}:
        return f"grasp the {category.lower()} handle or body"
    if category in {"Knife", "Scissors", "Pliers", "Stapler"}:
        return f"grasp the {category.lower()} handle"
    if category in {"Bottle", "Bowl", "Bucket"}:
        return f"hold the {category.lower()}"
    return f"grasp the {category.lower()}"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(as_jsonable(data), indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(as_jsonable(row), ensure_ascii=False) + "\n")


def resolve_ffmpeg(path: Path | None) -> Path:
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    sibling = Path(sys.executable).resolve().parent / "ffmpeg"
    if sibling.exists():
        return sibling
    found = shutil.which("ffmpeg")
    if found:
        return Path(found)
    raise FileNotFoundError("ffmpeg not found. Pass --ffmpeg-bin /path/to/ffmpeg.")


def extract_subset(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = resolve_ffmpeg(args.ffmpeg_bin)
    archives = {
        "release_zip": args.release_zip,
        "annotations_zip": args.annotations_zip,
        "hand_pose_zip": args.hand_pose_zip,
        "cad_zip": args.cad_zip,
        "camera_zip": args.camera_zip,
        "ffmpeg_bin": ffmpeg_bin,
    }
    write_json(args.output_root / "archives" / "archives.json", archives)

    with zipfile.ZipFile(args.release_zip) as release_zip, zipfile.ZipFile(args.annotations_zip) as annotations_zip, zipfile.ZipFile(args.hand_pose_zip) as hand_zip, zipfile.ZipFile(args.cad_zip) as cad_zip:
        camera_zip = zipfile.ZipFile(args.camera_zip) if args.camera_zip and args.camera_zip.exists() else None
        try:
            indices = build_zip_sets(release_zip, annotations_zip, hand_zip, cad_zip, camera_zip)
            if args.index_csv is not None:
                selected = sample_candidates_from_index_csv(
                    index_csv=args.index_csv,
                    indices=indices,
                    allowed_categories=set(args.categories),
                    count=args.count,
                    seed=args.seed,
                    max_per_sequence=args.max_per_sequence,
                    sides=args.sides,
                )
            else:
                selected = sample_candidates(
                    indices=indices,
                    allowed_categories=set(args.categories),
                    count=args.count,
                    seed=args.seed,
                    max_per_sequence=args.max_per_sequence,
                )
            write_jsonl(args.output_root / "manifests" / "selected_candidates.jsonl", [asdict(c) for c in selected])

            cad_written: dict[str, str] = {}
            camera_written: dict[str, str] = {}
            rows: list[dict[str, Any]] = []
            with tempfile.TemporaryDirectory(prefix="hoi4d_extract_") as tmpdir_text:
                tmpdir = Path(tmpdir_text)
                video_cache: dict[str, Path] = {}
                for index, candidate in enumerate(selected, start=1):
                    print(f"[{index}/{len(selected)}] {candidate.case_id}", flush=True)
                    case_dir = args.output_root / "frames" / candidate.case_id
                    compat_sequence = args.output_root / "dataset" / candidate.rel_sequence
                    compat_hand = (
                        args.output_root
                        / "dataset"
                        / "Hand_pose"
                        / f"handpose_{candidate.side}_hand"
                        / candidate.rel_sequence
                        / f"{candidate.frame}.pickle"
                    )
                    rgb_path = case_dir / "rgb.png"
                    compat_rgb = compat_sequence / "align_rgb" / f"{candidate.frame:05d}.jpg"
                    objpose_path = case_dir / "object_pose.json"
                    compat_objpose = compat_sequence / "objpose" / f"{candidate.frame}.json"
                    mask_path = case_dir / "seg_mask.png"
                    compat_mask = compat_sequence / "2Dseg" / "mask" / f"{candidate.frame:05d}.png"
                    hand_path = case_dir / "hand_pose.pickle"

                    video_path = video_cache.get(candidate.release_video_entry)
                    if video_path is None:
                        video_path = tmpdir / f"video_{len(video_cache):05d}.mp4"
                        extract_zip_member(release_zip, candidate.release_video_entry, video_path)
                        video_cache[candidate.release_video_entry] = video_path
                    extract_frame_from_video(video_path, candidate.frame, rgb_path, ffmpeg_bin)
                    compat_rgb.parent.mkdir(parents=True, exist_ok=True)
                    Image.open(rgb_path).convert("RGB").save(compat_rgb, quality=95)

                    extract_zip_member(hand_zip, candidate.hand_entry, hand_path)
                    compat_hand.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(hand_path, compat_hand)

                    extract_zip_member(annotations_zip, candidate.objpose_entry, objpose_path)
                    compat_objpose.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(objpose_path, compat_objpose)

                    if candidate.mask_entry is not None:
                        extract_zip_member(annotations_zip, candidate.mask_entry, mask_path)
                        compat_mask.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(mask_path, compat_mask)

                    action_path = None
                    if candidate.action_entry is not None:
                        action_path = case_dir / "action.json"
                        extract_zip_member(annotations_zip, candidate.action_entry, action_path)

                    cad_rel = strip_prefix(candidate.cad_entry, "HOI4D_CAD_Model_for_release/")
                    cad_out = args.output_root / "dataset" / "cad_models" / "HOI4D_CAD_Model_for_release" / cad_rel
                    if candidate.cad_entry not in cad_written:
                        extract_zip_member(cad_zip, candidate.cad_entry, cad_out)
                        cad_written[candidate.cad_entry] = str(cad_out)

                    camera_out = None
                    if camera_zip is not None and candidate.camera_entry is not None:
                        subject = parse_sequence_parts(candidate.rel_sequence).get("subject", "")
                        camera_out_path = args.output_root / "dataset" / "camera_params" / subject / "intrin.npy"
                        if candidate.camera_entry not in camera_written:
                            extract_zip_member(camera_zip, candidate.camera_entry, camera_out_path)
                            camera_written[candidate.camera_entry] = str(camera_out_path)
                        camera_out = camera_written[candidate.camera_entry]

                    with hand_path.open("rb") as f:
                        hand_data = pickle.load(f, encoding="latin1")
                    image = Image.open(rgb_path)
                    lollipop_params = lollipop_from_hand(hand_data, image.width, image.height)
                    lollipop_params.update(
                        {
                            "candidate_id": candidate.case_id,
                            "source_domain": "hoi4d",
                            "task": task_for_category(candidate.category),
                            "category": candidate.category,
                            "side": candidate.side,
                        }
                    )
                    lollipop_mask = case_dir / "lollipop_mask.png"
                    lollipop_overlay = case_dir / "lollipop_overlay.png"
                    draw_lollipop_mask(lollipop_params, lollipop_mask, lollipop_overlay, rgb_path)
                    write_json(case_dir / "lollipop_params.json", lollipop_params)

                    row = {
                        **asdict(candidate),
                        "source_domain": "hoi4d",
                        "index_csv": args.index_csv,
                        "task": lollipop_params["task"],
                        "case_dir": case_dir,
                        "rgb_path": rgb_path,
                        "compat_rgb_path": compat_rgb,
                        "hand_path": hand_path,
                        "compat_hand_path": compat_hand,
                        "sequence_path": compat_sequence,
                        "objpose_path": objpose_path,
                        "compat_objpose_path": compat_objpose,
                        "mask_path": mask_path if candidate.mask_entry else None,
                        "action_path": action_path,
                        "cad_path": cad_out,
                        "camera_path": camera_out,
                        "lollipop_mask_path": lollipop_mask,
                        "lollipop_overlay_path": lollipop_overlay,
                        "lollipop_params_path": case_dir / "lollipop_params.json",
                    }
                    write_json(case_dir / "sample_metadata.json", row)
                    rows.append(row)
                    if index % args.summary_interval == 0:
                        write_jsonl(args.output_root / "manifests" / "unified_samples.jsonl", rows)
            write_jsonl(args.output_root / "manifests" / "unified_samples.jsonl", rows)
            write_json(
                args.output_root / "manifests" / "extraction_summary.json",
                {
                    "count": len(rows),
                    "categories": args.categories,
                    "seed": args.seed,
                    "max_per_sequence": args.max_per_sequence,
                    "index_csv": args.index_csv,
                    "sides": args.sides,
                    "output_root": args.output_root,
                    "cad_models_extracted": len(cad_written),
                    "camera_files_extracted": len(camera_written),
                    "archives": archives,
                },
            )
        finally:
            if camera_zip is not None:
                camera_zip.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-zip", type=Path, default=DEFAULT_ARCHIVE_DIR / "HOI4D_release.zip")
    parser.add_argument("--annotations-zip", type=Path, default=DEFAULT_ARCHIVE_DIR / "HOI4D_annotations.zip")
    parser.add_argument("--hand-pose-zip", type=Path, default=DEFAULT_ARCHIVE_DIR / "HOI4D_Hand_pose.zip")
    parser.add_argument("--cad-zip", type=Path, default=DEFAULT_ARCHIVE_DIR / "HOI4D_CAD_Model_for_release.zip")
    parser.add_argument("--camera-zip", type=Path, default=DEFAULT_ARCHIVE_DIR / "camera_params.zip")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "data" / "hoi4d" / "unified_100_smoke")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--max-per-sequence", type=int, default=5)
    parser.add_argument("--summary-interval", type=int, default=25)
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=None,
        help="Optional HOI4D frame index CSV, e.g. Affordance Diffusion preprocess/docs/all_contact.csv.",
    )
    parser.add_argument("--sides", nargs="+", default=["right", "left"], choices=["right", "left"])
    parser.add_argument("--ffmpeg-bin", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists: {args.output_root}. Use --overwrite to replace it.")
        shutil.rmtree(args.output_root)
    for path in (args.release_zip, args.annotations_zip, args.hand_pose_zip, args.cad_zip):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.index_csv is not None and not args.index_csv.exists():
        raise FileNotFoundError(args.index_csv)
    extract_subset(args)
    print(f"Done. Output: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
