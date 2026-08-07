#!/usr/bin/env python3
"""Run HaMeR on generated kettle phase4 images using lollipop masks as hand boxes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[2]
HAMER_DIR = REPO_ROOT / "external" / "hamer"
sys.path.insert(0, str(HAMER_DIR))


def bbox_from_mask(mask: np.ndarray, pad: float = 0.10) -> list[float]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        h, w = mask.shape[:2]
        return [0, 0, w - 1, h - 1]
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    bw, bh = x2 - x1 + 1, y2 - y1 + 1
    x1 -= bw * pad
    x2 += bw * pad
    y1 -= bh * pad
    y2 += bh * pad
    h, w = mask.shape[:2]
    return [float(max(0, x1)), float(max(0, y1)), float(min(w - 1, x2)), float(min(h - 1, y2))]


def collect_cases(output_root: Path) -> list[dict]:
    sample_ids = [
        "20260713_172042_054_kettle-1",
        "20260713_172116_649_kettle-2",
        "20260713_172149_129_kettle-3",
    ]
    cases = []
    for sample_id in sample_ids:
        for ldir in sorted((output_root / sample_id).glob("lollipop_*")):
            rgb = ldir / "phase4_inpaint_full.png"
            mask = ldir / "lollipop_mask.png"
            if rgb.exists() and mask.exists():
                cases.append(
                    {
                        "sample_id": sample_id,
                        "lollipop": ldir.name,
                        "rgb": rgb,
                        "mask": mask,
                        "out_dir": ldir / "pose_hamer",
                    }
                )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "outputs/0713test_phase1_4_vlm_qwen37_test"),
    )
    parser.add_argument("--rescale-factor", type=float, default=1.6)
    parser.add_argument("--mask-pad", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    # HaMeR checkpoint paths are relative to external/hamer.
    os.chdir(HAMER_DIR)
    from hamer.configs import CACHE_DIR_HAMER
    from hamer.models import DEFAULT_CHECKPOINT, download_models, load_hamer
    from hamer.utils import recursive_to
    from hamer.datasets.vitdet_dataset import DEFAULT_MEAN, DEFAULT_STD, ViTDetDataset
    from hamer.utils.renderer import Renderer, cam_crop_to_full

    output_root = Path(args.output_root)
    cases = collect_cases(output_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)
    model = model.to(device)
    model.eval()
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    faces = np.asarray(model.mano.faces)
    light_blue = (0.65098039, 0.74117647, 0.85882353)

    summary = []
    for idx, case in enumerate(cases):
        print(f"[{idx + 1:02d}/{len(cases):02d}] {case['sample_id']} {case['lollipop']}", flush=True)
        out_dir = case["out_dir"]
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        img_cv2 = cv2.imread(str(case["rgb"]))
        mask = cv2.imread(str(case["mask"]), cv2.IMREAD_GRAYSCALE)
        bbox = bbox_from_mask(mask, pad=args.mask_pad)
        boxes = np.asarray([bbox], dtype=np.float32)
        right = np.asarray([1], dtype=np.float32)

        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        all_keypoints = []
        mesh_records = []
        rendered_patches = []

        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                pred = model(batch)

            multiplier = 2 * batch["right"] - 1
            pred_cam = pred["pred_cam"]
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(
                pred_cam, box_center, box_size, img_size, scaled_focal_length
            ).detach().cpu().numpy()

            bs = batch["img"].shape[0]
            for n in range(bs):
                is_right = float(batch["right"][n].detach().cpu().numpy())
                verts = pred["pred_vertices"][n].detach().cpu().numpy()
                verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
                keypoints = pred["pred_keypoints_3d"][n].detach().cpu().numpy()
                keypoints[:, 0] = (2 * is_right - 1) * keypoints[:, 0]
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)
                all_keypoints.append(keypoints)

                mesh = trimesh.Trimesh(vertices=verts + cam_t.reshape(1, 3), faces=faces, process=False)
                mesh.export(out_dir / f"hand_{n:02d}_camera.obj")
                np.savetxt(out_dir / f"hand_{n:02d}_cam_t.txt", cam_t.reshape(1, 3))
                np.save(out_dir / f"hand_{n:02d}_vertices_camera.npy", verts + cam_t.reshape(1, 3))
                np.save(out_dir / f"hand_{n:02d}_keypoints_camera.npy", keypoints + cam_t.reshape(1, 3))
                np.save(out_dir / f"hand_{n:02d}_keypoints_mano_local.npy", keypoints)
                mesh_records.append(
                    {
                        "mesh_file": str(out_dir / f"hand_{n:02d}_camera.obj"),
                        "vertices_camera_npy": str(out_dir / f"hand_{n:02d}_vertices_camera.npy"),
                        "keypoints_camera_npy": str(out_dir / f"hand_{n:02d}_keypoints_camera.npy"),
                        "keypoints_mano_local_npy": str(out_dir / f"hand_{n:02d}_keypoints_mano_local.npy"),
                        "cam_t": cam_t.tolist(),
                        "is_right": bool(is_right),
                    }
                )

                input_patch = batch["img"][n].detach().cpu() * (DEFAULT_STD[:, None, None] / 255) + (
                    DEFAULT_MEAN[:, None, None] / 255
                )
                input_patch = input_patch.permute(1, 2, 0).numpy()
                regression_img = renderer(
                    pred["pred_vertices"][n].detach().cpu().numpy(),
                    pred["pred_cam_t"][n].detach().cpu().numpy(),
                    batch["img"][n],
                    mesh_base_color=light_blue,
                    scene_bg_color=(1, 1, 1),
                )
                rendered_patches.append(np.concatenate([input_patch, regression_img], axis=1))

        if all_verts:
            misc_args = dict(
                mesh_base_color=light_blue,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(
                all_verts,
                cam_t=all_cam_t,
                render_res=img_size[0],
                is_right=all_right,
                **misc_args,
            )
            input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
            overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
            overlay_bgr = (255 * overlay[:, :, ::-1]).clip(0, 255).astype(np.uint8)

            mask_overlay = img_cv2.copy()
            m = mask > 0
            mask_overlay[m] = (0.65 * mask_overlay[m] + 0.35 * np.array([0, 255, 255])).astype(np.uint8)
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            cv2.rectangle(mask_overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)

            comparison = np.concatenate([mask_overlay, overlay_bgr], axis=1)
            cv2.imwrite(str(out_dir / "input_bbox_overlay.png"), mask_overlay)
            cv2.imwrite(str(out_dir / "hand_mesh_overlay.png"), overlay_bgr)
            cv2.imwrite(str(out_dir / "comparison.png"), comparison)
            if rendered_patches:
                patch = (255 * np.concatenate(rendered_patches, axis=0)[:, :, ::-1]).clip(0, 255).astype(np.uint8)
                cv2.imwrite(str(out_dir / "crop_regression.png"), patch)

        meta = {
            "sample_id": case["sample_id"],
            "lollipop": case["lollipop"],
            "rgb": str(case["rgb"]),
            "mask": str(case["mask"]),
            "bbox_xyxy": bbox,
            "rescale_factor": args.rescale_factor,
            "mesh_records": mesh_records,
            "note": "HaMeR run with direct lollipop-mask hand bbox; detector/ViTPose were bypassed.",
        }
        (out_dir / "hamer_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        summary.append(meta)

    summary_file = output_root / "kettle_hamer_direct_bbox_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {summary_file}", flush=True)


if __name__ == "__main__":
    main()
