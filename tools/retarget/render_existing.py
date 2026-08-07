#!/usr/bin/env python3
"""Render retargeting HTML from an existing retargeted_*.json file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
RETARGET_DIR = REPO_ROOT / "tools" / "retarget"
if str(RETARGET_DIR) not in sys.path:
    sys.path.insert(0, str(RETARGET_DIR))

from position import write_retarget_html


def namespace_result(data: dict) -> SimpleNamespace:
    result = data["result"]
    return SimpleNamespace(**result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--no-scale", action="store_true")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    data = json.loads(args.json_path.read_text())
    case_dir = args.json_path.parent
    base_meta = {
        "case_id": data["case_id"],
        "sequence": data["sequence"],
        "frame": data["frame"],
        "cad": data["cad"],
        "hand_pickle": data["hand_pickle"],
        "mano_root": data["mano_root"],
        "robot_profile": data["robot_profile"],
        "route": data["route"],
    }
    rgb = Path(data["sequence"]) / "align_rgb" / f"{int(data['frame']):05d}.jpg"
    output = write_retarget_html(
        case_dir,
        rgb,
        base_meta,
        namespace_result(data),
        float(data.get("optimization_seconds") or 0.0),
        scale_visualization=not args.no_scale,
        output_suffix=args.suffix,
    )
    print(output)


if __name__ == "__main__":
    main()
