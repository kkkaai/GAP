#!/usr/bin/env python3
"""Create per-sample phase4 contact sheets from a manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def make_contact_sheet(sample_id: str, rows: list[dict], output_name: str, cell_width: int, cell_height: int) -> Path:
    sample_dir = Path(rows[0]["output_dir"]).parent
    source = Image.open(sample_dir / "source_rgb.png").convert("RGB").resize((cell_width, cell_height))
    font_pad = 38
    row_images = []
    for row in sorted(rows, key=lambda item: item["lollipop_id"]):
        candidate_dir = Path(row["output_dir"])
        overlay = Image.open(candidate_dir / "lollipop_overlay.png").convert("RGB").resize((cell_width, cell_height))
        phase4 = Image.open(row["phase4_output"]).convert("RGB").resize((cell_width, cell_height))
        params_path = candidate_dir / "lollipop_params.json"
        params = json.loads(params_path.read_text(encoding="utf-8")) if params_path.exists() else {}

        row_img = Image.new("RGB", (cell_width * 3, cell_height + font_pad), "white")
        row_img.paste(source, (0, font_pad))
        row_img.paste(overlay, (cell_width, font_pad))
        row_img.paste(phase4, (cell_width * 2, font_pad))

        draw = ImageDraw.Draw(row_img)
        label = f"{row['lollipop_id']} {params.get('affordance_part', row.get('affordance_part', ''))} | {params.get('task', '')}"
        draw.text((6, 4), label[:80], (0, 0, 0))
        draw.text((6, 20), params.get("grasp_type", row.get("grasp_type", ""))[:80], (0, 0, 0))
        draw.text((cell_width + 6, 20), "lollipop overlay", (0, 0, 0))
        draw.text((cell_width * 2 + 6, 20), "phase4 output", (0, 0, 0))
        row_images.append(row_img)

    sheet = Image.new("RGB", (cell_width * 3, (cell_height + font_pad) * len(row_images)), "white")
    for index, row_img in enumerate(row_images):
        sheet.paste(row_img, (0, index * (cell_height + font_pad)))
    out_path = sample_dir / output_name
    sheet.save(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-name", default="vlm_phase4_contact_sheet.png")
    parser.add_argument("--cell-width", type=int, default=320)
    parser.add_argument("--cell-height", type=int, default=240)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_sample: dict[str, list[dict]] = {}
    for row in rows:
        by_sample.setdefault(row["sample_id"], []).append(row)

    outputs = []
    for sample_id in sorted(by_sample):
        outputs.append(
            make_contact_sheet(
                sample_id,
                by_sample[sample_id],
                args.output_name,
                args.cell_width,
                args.cell_height,
            )
        )
    print(json.dumps({"num_samples": len(outputs), "outputs": [str(path) for path in outputs]}, indent=2))


if __name__ == "__main__":
    main()
