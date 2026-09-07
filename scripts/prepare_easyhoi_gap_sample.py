#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _bbox_from_mask(mask: np.ndarray, pad: int = 0) -> list[int]:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        raise ValueError("Mask is empty; cannot compute bbox.")
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(mask.shape[1] - 1, int(xs.max()) + pad)
    y1 = min(mask.shape[0] - 1, int(ys.max()) + pad)
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def _load_phase5_prediction(gap_root: Path, image_path: Path):
    sys.path.insert(0, str(gap_root / "src"))
    from prosthetic_grasp.phases.phase5_mano import Phase5Mano, Phase5ManoConfig

    image_rgb = np.asarray(Image.open(image_path).convert("RGB"))
    runner = Phase5Mano(
        Phase5ManoConfig(
            hamer_root=str(gap_root / "external" / "hamer"),
            hand_side="right",
            download_models=False,
            batch_size=1,
        )
    )
    result = runner.run(image_rgb)
    if result.status != "ok" or not result.hands:
        raise RuntimeError(f"HaMeR failed for EasyHOI sample: {result.status} {result.message}")
    return result.hands[0]


def prepare_sample(args: argparse.Namespace) -> None:
    gap_root = Path(args.gap_root).expanduser().resolve()
    lollipop_dir = Path(args.lollipop_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    sample_id = args.sample_id

    image_path = lollipop_dir / "phase4_inpaint_full.png"
    # EasyHOI needs a visible hand-region mask in its inpainting convention
    # (hand pixels are black, background is white). Prefer the text-grounded
    # SAM3.1 hand mask; fall back to older masks only for smoke tests.
    hand_mask_candidates = [
        lollipop_dir / "pose_hamer_official" / "sam3_1_text_hand" / "sam3_1_text_hand_mask.png",
        lollipop_dir / "pose_hamer_official" / "sam3_1_text_hand_mask.png",
        lollipop_dir / "pose_hamer_official" / "sam2_hand_mask.png",
        lollipop_dir / "pose_hamer_official" / "official_hand_bbox_mask.png",
        lollipop_dir / "lollipop_mask.png",
    ]
    hand_mask_src = next((path for path in hand_mask_candidates if path.exists()), hand_mask_candidates[0])
    object_mask_src = lollipop_dir / "pose_foundationpose" / "rendered_mesh_mask.png"
    if not image_path.exists() or not hand_mask_src.exists() or not object_mask_src.exists():
        raise FileNotFoundError(
            "Missing image or masks. Expected phase4_inpaint_full.png, "
            "pose_hamer_official/sam3_1_text_hand/sam3_1_text_hand_mask.png, and "
            "pose_foundationpose/rendered_mesh_mask.png."
        )

    mesh_src = Path(args.mesh).expanduser().resolve()
    if not mesh_src.exists():
        raise FileNotFoundError(mesh_src)

    hand = _load_phase5_prediction(gap_root, image_path)

    image_dir = out_dir / "images"
    hand_mask_dir = out_dir / "obj_recon" / "hand_mask"
    obj_mask_dir = out_dir / "obj_recon" / "obj_mask"
    inpaint_mask_dir = out_dir / "obj_recon" / "inpaint_mask"
    hoi_box_dir = out_dir / "obj_recon" / "inpaint" / "hoi_box"
    mesh_dir = out_dir / "obj_recon" / "results" / "instantmesh" / "instant-mesh-large" / "meshes" / sample_id
    hamer_dir = out_dir / "hamer"
    for directory in [image_dir, hand_mask_dir, obj_mask_dir, inpaint_mask_dir, hoi_box_dir, mesh_dir, hamer_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(image_path, image_dir / f"{sample_id}.png")
    shutil.copyfile(mesh_src, mesh_dir / "full.obj")

    hand_mask = np.asarray(Image.open(hand_mask_src).convert("L"))
    obj_mask = np.asarray(Image.open(object_mask_src).convert("L"))

    # EasyHOI convention: hand mask uses 0 for hand and nonzero for background.
    easyhoi_hand_mask = np.where(hand_mask > 0, 0, 255).astype(np.uint8)
    easyhoi_obj_mask = np.where(obj_mask > 0, 255, 0).astype(np.uint8)
    # For a smoke test, use the object silhouette as the inpaint/object target mask.
    easyhoi_inpaint_mask = easyhoi_obj_mask.copy()

    Image.fromarray(easyhoi_hand_mask).save(hand_mask_dir / f"{sample_id}.png")
    Image.fromarray(easyhoi_obj_mask).save(obj_mask_dir / f"{sample_id}.png")
    Image.fromarray(easyhoi_inpaint_mask).save(inpaint_mask_dir / f"{sample_id}.png")

    hoi_bbox = _bbox_from_mask(easyhoi_obj_mask, pad=20)
    with open(hoi_box_dir / f"{sample_id}.json", "w", encoding="utf-8") as f:
        json.dump(hoi_bbox, f)

    width, height = Image.open(image_path).size
    focal_px = float(hand.focal_length)
    hand_cam = {
        "fx": focal_px / width,
        "fy": focal_px / height,
        "cx": 0.5,
        "cy": 0.5,
        "extrinsics": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
    }
    with open(hamer_dir / f"{sample_id}_cam.json", "w", encoding="utf-8") as f:
        json.dump(hand_cam, f, indent=2)

    mano_params = {}
    for key, value in hand.mano_params.items():
        arr = np.asarray(value)
        if key == "global_orient":
            arr = arr.reshape(1, 1, 3, 3)
        elif key == "hand_pose":
            arr = arr.reshape(1, 15, 3, 3)
        if key == "betas":
            arr = arr.reshape(1, 10)
        mano_params[key] = torch.from_numpy(arr.astype(np.float32))

    hamer_pt = {
        "batch_size": 1,
        "boxes": torch.from_numpy(np.asarray(hand.bbox_xyxy, dtype=np.float32)).reshape(1, 4),
        "is_right": torch.tensor([1.0 if hand.is_right else 0.0], dtype=torch.float32),
        "cam_transl": torch.from_numpy(np.asarray(hand.pred_cam_t_full, dtype=np.float32)).reshape(1, 3),
        "mano_params": mano_params,
    }
    torch.save(hamer_pt, hamer_dir / f"{sample_id}.pt")

    # Avoid expensive mesh_to_sdf during a one-image smoke test. The no_penetr config
    # does not use this field, but EasyHOI still loads it unconditionally.
    np.save(mesh_dir / "sdf.npy", np.zeros((64, 64, 64), dtype=np.float32))

    np.save(out_dir / "test_filtered.npy", np.asarray([{"img_id": sample_id, "hamer_info": [{"id": 0}]}], dtype=object))

    manifest = {
        "sample_id": sample_id,
        "image": str(image_dir / f"{sample_id}.png"),
        "mesh": str(mesh_dir / "full.obj"),
        "hand_mask": str(hand_mask_dir / f"{sample_id}.png"),
        "object_mask": str(obj_mask_dir / f"{sample_id}.png"),
        "inpaint_mask": str(inpaint_mask_dir / f"{sample_id}.png"),
        "hamer_pt": str(hamer_dir / f"{sample_id}.pt"),
        "hand_cam": str(hamer_dir / f"{sample_id}_cam.json"),
        "hoi_box": str(hoi_box_dir / f"{sample_id}.json"),
        "note": "Smoke-test adapter for EasyHOI; not a final-quality EasyHOI preprocessing pipeline.",
    }
    with open(out_dir / "gap_easyhoi_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap-root", default="/home/kai/kai_vs_projects/GAP")
    parser.add_argument(
        "--lollipop-dir",
        default=(
            "/home/kai/kai_vs_projects/GAP/outputs/0713test_phase1_4_vlm_qwen37_test/"
            "20260713_172042_054_kettle-1/lollipop_03"
        ),
    )
    parser.add_argument(
        "--mesh",
        default=(
            "/home/kai/kai_vs_projects/GAP/objects/2026.7.14 3D test hunyuan/processed/"
            "kettle-decimated-50k.obj"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="/home/kai/kai_vs_projects/GAP/outputs/easyhoi_gap_one_image_test/data",
    )
    parser.add_argument("--sample-id", default="gap_kettle_l03")
    prepare_sample(parser.parse_args())


if __name__ == "__main__":
    main()
