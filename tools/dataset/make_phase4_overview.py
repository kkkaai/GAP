#!/usr/bin/env python3
"""Create a compact phase4 overview image from a manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell-width", type=int, default=180)
    parser.add_argument("--cell-height", type=int, default=135)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_sample: dict[str, list[dict]] = {}
    for row in rows:
        by_sample.setdefault(row["sample_id"], []).append(row)

    cell_w = args.cell_width
    cell_h = args.cell_height
    label_h = 34
    cols = 5
    row_images = []
    for sample_id in sorted(by_sample):
        sample_rows = sorted(by_sample[sample_id], key=lambda item: item["lollipop_id"])[:4]
        source = Image.open(sample_rows[0]["source_rgb"]).convert("RGB").resize((cell_w, cell_h))
        row_img = Image.new("RGB", (cols * cell_w, cell_h + label_h), "white")
        draw = ImageDraw.Draw(row_img)
        row_img.paste(source, (0, label_h))
        draw.text((4, 4), sample_id[:34], (0, 0, 0))
        draw.text((4, 18), "source", (0, 0, 0))
        for col, item in enumerate(sample_rows, start=1):
            out = Image.open(item["phase4_output"]).convert("RGB").resize((cell_w, cell_h))
            row_img.paste(out, (col * cell_w, label_h))
            label = f"{item['lollipop_id']} {item.get('affordance_part', '')}"
            draw.text((col * cell_w + 4, 4), label[:28], (0, 0, 0))
            draw.text((col * cell_w + 4, 18), item.get("grasp_type", "")[:28], (0, 0, 0))
        row_images.append(row_img)

    sheet = Image.new("RGB", (cols * cell_w, (cell_h + label_h) * len(row_images)), "white")
    for index, row_img in enumerate(row_images):
        sheet.paste(row_img, (0, index * (cell_h + label_h)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
