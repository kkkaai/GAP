#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _draw_overlay(image: np.ndarray, masks: np.ndarray, boxes: np.ndarray, scores: np.ndarray) -> Image.Image:
    overlay = image.copy()
    colors = [(0, 255, 0), (255, 150, 0), (0, 180, 255), (255, 0, 180)]
    for idx, mask in enumerate(masks):
        color = np.zeros_like(overlay)
        color[:, :] = colors[idx % len(colors)]
        alpha = (mask > 0)[..., None].astype(np.float32) * 0.42
        overlay = (overlay * (1 - alpha) + color * alpha).astype(np.uint8)

    pil = Image.fromarray(overlay)
    draw = ImageDraw.Draw(pil)
    for idx, box in enumerate(boxes):
        x0, y0, x1, y1 = [float(v) for v in box]
        color = colors[idx % len(colors)]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        draw.text((x0 + 4, max(0, y0 - 16)), f"hand {float(scores[idx]):.2f}", fill=color)
    return pil


def _make_sheet(image: Image.Image, overlay: Image.Image, mask: Image.Image) -> Image.Image:
    width, height = image.size
    sheet = Image.new("RGB", (width * 3, height + 28), "white")
    draw = ImageDraw.Draw(sheet)
    panels = [("image", image), ("sam3.1 text hand overlay", overlay), ("union mask", mask.convert("RGB"))]
    for idx, (label, panel) in enumerate(panels):
        sheet.paste(panel, (idx * width, 28))
        draw.text((idx * width + 8, 6), label, fill=(0, 0, 0))
    return sheet


def generate(args: argparse.Namespace) -> None:
    gap_root = Path(args.gap_root).expanduser().resolve()
    sam3_root = gap_root / "external" / "sam3"
    if not sam3_root.exists():
        raise FileNotFoundError(f"SAM3 repo not found: {sam3_root}")
    sys.path.insert(0, str(sam3_root))

    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    lollipop_dir = Path(args.lollipop_dir).expanduser().resolve()
    image_path = Path(args.image).expanduser().resolve() if args.image else lollipop_dir / "phase4_inpaint_full.png"
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else lollipop_dir / "pose_hamer_official" / "sam3_1_text_hand"
    )
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    output_dir.mkdir(parents=True, exist_ok=True)
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image_pil)

    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        device=device,
        eval_mode=True,
    )
    processor = Sam3Processor(model, device=device, confidence_threshold=args.confidence_threshold)
    autocast_enabled = device == "cuda"
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        state = processor.set_image(image_pil)
        output = processor.set_text_prompt(state=state, prompt=args.text_prompt)

    masks = output["masks"].detach().cpu().numpy()
    boxes = output["boxes"].detach().float().cpu().numpy().astype(np.float32)
    scores = output["scores"].detach().float().cpu().numpy().astype(np.float32)
    if masks.ndim == 4:
        masks = masks[:, 0]
    masks = masks.astype(bool)

    if len(masks) == 0:
        union_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
        overlay = Image.fromarray(image_np)
    else:
        order = np.argsort(-scores)
        if args.max_detections > 0:
            order = order[: args.max_detections]
        masks = masks[order]
        boxes = boxes[order]
        scores = scores[order]
        union_mask = (np.any(masks, axis=0).astype(np.uint8) * 255)
        overlay = _draw_overlay(image_np, masks, boxes, scores)

    Image.fromarray(union_mask).save(output_dir / "sam3_1_text_hand_mask.png")
    easyhoi = np.where(union_mask > 0, 0, 255).astype(np.uint8)
    Image.fromarray(easyhoi).save(output_dir / "sam3_1_text_hand_mask_easyhoi.png")
    overlay.save(output_dir / "sam3_1_text_hand_overlay.png")
    sheet = _make_sheet(image_pil, overlay, Image.fromarray(union_mask))
    sheet.save(output_dir / "sam3_1_text_hand_debug_sheet.png")

    metadata = {
        "image": str(image_path),
        "text_prompt": args.text_prompt,
        "checkpoint": str(checkpoint),
        "device": device,
        "confidence_threshold": args.confidence_threshold,
        "detections": [
            {
                "box_xyxy": boxes[idx].tolist(),
                "score": float(scores[idx]),
                "mask_pixels": int(masks[idx].sum()),
            }
            for idx in range(len(masks))
        ],
        "union_mask_pixels": int((union_mask > 0).sum()),
    }
    with open(output_dir / "sam3_1_text_hand_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap-root", default=str(_repo_root()))
    parser.add_argument(
        "--lollipop-dir",
        default=(
            "outputs/0713test_phase1_4_vlm_qwen37_test/"
            "20260713_172042_054_kettle-1/lollipop_03"
        ),
    )
    parser.add_argument("--image")
    parser.add_argument("--output-dir")
    parser.add_argument("--checkpoint", default="models/sam3_1/sam3.1_multiplex.pt")
    parser.add_argument("--text-prompt", default="human hand")
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--max-detections", type=int, default=3)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
