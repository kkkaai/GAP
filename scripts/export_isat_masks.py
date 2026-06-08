from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Export ISAT polygon annotations to 0/255 binary masks.")
    parser.add_argument("--json-dir", required=True, help="Directory containing ISAT json files.")
    parser.add_argument("--output-dir", required=True, help="Directory to save exported masks.")
    parser.add_argument("--category", default="prosthetic_hand", help="ISAT category to export as foreground.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing masks.")
    return parser


def output_name(payload: dict, json_path: Path) -> str:
    image_name = payload.get("info", {}).get("name") or f"{json_path.stem}.png"
    suffix = Path(image_name).suffix
    if suffix.lower() in IMAGE_SUFFIXES:
        return f"{Path(image_name).stem}.png"
    return f"{json_path.stem}.png"


def export_one(json_path: Path, output_dir: Path, category: str, overwrite: bool) -> tuple[Path, int]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    info = payload.get("info", {})
    width = int(info["width"])
    height = int(info["height"])
    mask_path = output_dir / output_name(payload, json_path)
    if mask_path.exists() and not overwrite:
        return mask_path, -1

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    count = 0
    for obj in payload.get("objects", []):
        if obj.get("category") != category:
            continue
        points = obj.get("segmentation") or []
        if len(points) < 3:
            continue
        polygon = [(float(x), float(y)) for x, y in points]
        draw.polygon(polygon, fill=255)
        count += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    mask.save(mask_path)
    return mask_path, count


def main() -> None:
    args = build_argparser().parse_args()
    json_dir = Path(args.json_dir)
    output_dir = Path(args.output_dir)
    json_paths = sorted(json_dir.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No ISAT json files found in {json_dir}.")

    written = 0
    skipped = 0
    for json_path in json_paths:
        mask_path, count = export_one(json_path, output_dir, args.category, args.overwrite)
        if count < 0:
            skipped += 1
            print(f"skip existing: {mask_path}")
        else:
            written += 1
            print(f"{json_path.name} -> {mask_path.name} objects={count}")
    print(f"Finished. written={written}, skipped={skipped}, output_dir={output_dir}")


if __name__ == "__main__":
    main()
