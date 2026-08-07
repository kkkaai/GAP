#!/usr/bin/env python3
"""Prepare a reproducible HOI4D retargeting test case folder.

Each run creates a new folder under outputs/runs/ containing:
  - original.jpg
  - input_scene.html
  - input_scene.json
  - retargeted_scene.html

The retargeted_scene.html file is a placeholder unless a retarget result HTML is
passed in. This keeps the output contract stable for future retargeting code.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from scene_html import build_figure, write_html
from scene_png import DEFAULT_CAD, DEFAULT_FRAME, DEFAULT_HAND, DEFAULT_MANO_ROOT, DEFAULT_SEQUENCE


DEFAULT_RGB = DEFAULT_SEQUENCE / "align_rgb" / "00074.jpg"


def slugify(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def default_case_id(sequence: Path, frame: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{slugify(str(sequence))}_frame{int(frame):05d}"


def write_placeholder_retarget_html(path: Path, metadata: dict) -> None:
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Retargeted Scene Placeholder</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: #f6f6f6;
      color: #222;
      display: grid;
      place-items: center;
      min-height: 100vh;
    }}
    main {{
      width: min(900px, calc(100vw - 48px));
      background: white;
      border: 1px solid #ddd;
      padding: 24px;
    }}
    h1 {{ margin-top: 0; font-size: 22px; }}
    pre {{ white-space: pre-wrap; background: #f0f0f0; padding: 16px; overflow: auto; }}
  </style>
</head>
<body>
  <main>
    <h1>Retargeted scene not generated yet</h1>
    <p>Write your robot-hand retargeting result to this file, or pass <code>--retarget-html</code> when preparing the case.</p>
    <pre>{json.dumps(metadata, indent=2)}</pre>
  </main>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def append_index(index_path: Path, record: dict) -> None:
    if index_path.exists():
        data = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        data = []
    data = [item for item in data if item.get("case_id") != record.get("case_id")]
    data.append(record)
    index_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare_case(
    sequence: Path,
    frame: str,
    rgb: Path,
    cad: Path,
    hand: Path,
    mano_root: Path,
    output_root: Path,
    case_id: str | None,
    retarget_html: Path | None,
    overwrite: bool,
) -> Path:
    case_id = case_id or default_case_id(sequence, frame)
    case_dir = output_root / case_id
    if case_dir.exists() and overwrite:
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=False)

    original_dst = case_dir / "original.jpg"
    input_html = case_dir / "input_scene.html"
    input_json = case_dir / "input_scene.json"
    retarget_dst = case_dir / "retargeted_scene.html"

    shutil.copy2(rgb, original_dst)

    fig, scene_meta = build_figure(sequence, frame, cad, hand, mano_root)
    scene_meta.update(
        {
            "case_id": case_id,
            "case_dir": str(case_dir),
            "original_image": str(original_dst),
            "input_scene_html": str(input_html),
            "input_scene_json": str(input_json),
            "retargeted_scene_html": str(retarget_dst),
            "retargeting_status": "provided" if retarget_html else "pending",
        }
    )
    write_html(fig, scene_meta, original_dst, input_html)
    input_json.write_text(json.dumps(scene_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    if retarget_html:
        shutil.copy2(retarget_html, retarget_dst)
    else:
        write_placeholder_retarget_html(retarget_dst, scene_meta)

    append_index(
        output_root / "index.json",
        {
            "case_id": case_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "case_dir": str(case_dir),
            "sequence": str(sequence),
            "frame": frame,
            "original_image": str(original_dst),
            "input_scene_html": str(input_html),
            "input_scene_json": str(input_json),
            "retargeted_scene_html": str(retarget_dst),
            "retargeting_status": scene_meta["retargeting_status"],
        },
    )
    return case_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--frame", default=DEFAULT_FRAME)
    parser.add_argument("--rgb", type=Path, default=DEFAULT_RGB)
    parser.add_argument("--cad", type=Path, default=DEFAULT_CAD)
    parser.add_argument("--hand", type=Path, default=DEFAULT_HAND)
    parser.add_argument("--mano-root", type=Path, default=DEFAULT_MANO_ROOT)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--case-id")
    parser.add_argument("--retarget-html", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    case_dir = prepare_case(
        sequence=args.sequence,
        frame=args.frame,
        rgb=args.rgb,
        cad=args.cad,
        hand=args.hand,
        mano_root=args.mano_root,
        output_root=args.output_root,
        case_id=args.case_id,
        retarget_html=args.retarget_html,
        overwrite=args.overwrite,
    )
    print(f"Wrote test case: {case_dir}")
    print(f"Open: {case_dir / 'input_scene.html'}")


if __name__ == "__main__":
    main()
