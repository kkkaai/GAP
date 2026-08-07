#!/usr/bin/env python3
"""Run HaMeR on generated kettle phase4 images with the official detector path.

The official HaMeR demo first detects people with Detectron2, then estimates
whole-body keypoints with ViTPose and derives hand boxes from those keypoints.
This script applies the same flow to the 12 kettle phase4 images and saves the
same artifact names as the direct-bbox pilot under ``pose_hamer_official``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[2]
HAMER_DIR = REPO_ROOT / "external" / "hamer"
sys.path.insert(0, str(HAMER_DIR))

SAMPLE_IDS = [
    "20260713_172042_054_kettle-1",
    "20260713_172116_649_kettle-2",
    "20260713_172149_129_kettle-3",
]


def as_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: as_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(item) for item in value]
    return value


def bbox_from_mask(mask: np.ndarray, pad: float = 0.10) -> list[float]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        h, w = mask.shape[:2]
        return [0.0, 0.0, float(w - 1), float(h - 1)]
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    bw, bh = x2 - x1 + 1, y2 - y1 + 1
    h, w = mask.shape[:2]
    return [
        float(max(0, x1 - bw * pad)),
        float(max(0, y1 - bh * pad)),
        float(min(w - 1, x2 + bw * pad)),
        float(min(h - 1, y2 + bh * pad)),
    ]


def bbox_area(box: np.ndarray) -> float:
    return float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))


def bbox_mask_overlap(box: np.ndarray, mask: np.ndarray) -> float:
    x1, y1, x2, y2 = np.round(box).astype(int)
    h, w = mask.shape[:2]
    x1, x2 = int(np.clip(x1, 0, w - 1)), int(np.clip(x2, 0, w - 1))
    y1, y2 = int(np.clip(y1, 0, h - 1)), int(np.clip(y2, 0, h - 1))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(np.count_nonzero(mask[y1 : y2 + 1, x1 : x2 + 1] > 0))


def collect_cases(output_root: Path) -> list[dict[str, Path | str]]:
    cases: list[dict[str, Path | str]] = []
    for sample_id in SAMPLE_IDS:
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
                        "out_dir": ldir / "pose_hamer_official",
                    }
                )
    return cases


def load_body_detector(body_detector: str):
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy

    if body_detector == "vitdet":
        from detectron2.config import LazyConfig
        import hamer

        cfg_path = Path(hamer.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = (
            "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
            "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        )
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        return DefaultPredictor_Lazy(detectron2_cfg)

    from detectron2 import model_zoo

    detectron2_cfg = model_zoo.get_config(
        "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py",
        trained=True,
    )
    detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
    detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
    return DefaultPredictor_Lazy(detectron2_cfg)


def detect_hand_boxes(
    img_cv2: np.ndarray,
    mask: np.ndarray,
    detector: Any,
    cpm: Any,
    *,
    person_score_thresh: float,
    keypoint_thresh: float,
    allow_left: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    det_out = detector(img_cv2)
    det_instances = det_out["instances"]
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > person_score_thresh)
    pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores = det_instances.scores[valid_idx].cpu().numpy()
    candidates = []
    if len(pred_bboxes):
        img_rgb = img_cv2[:, :, ::-1]
        vitposes_out = cpm.predict_pose(img_rgb, [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)])
        for person_id, vitposes in enumerate(vitposes_out):
            hands = []
            if allow_left:
                hands.append(("left", vitposes["keypoints"][-42:-21], 0.0))
            hands.append(("right", vitposes["keypoints"][-21:], 1.0))
            for handedness, keyp, is_right in hands:
                valid = keyp[:, 2] > keypoint_thresh
                if int(np.count_nonzero(valid)) > 3:
                    box = np.asarray(
                        [
                            keyp[valid, 0].min(),
                            keyp[valid, 1].min(),
                            keyp[valid, 0].max(),
                            keyp[valid, 1].max(),
                        ],
                        dtype=np.float32,
                    )
                    candidates.append(
                        {
                            "bbox_xyxy": box,
                            "is_right": is_right,
                            "handedness": handedness,
                            "person_id": person_id,
                            "valid_keypoints": int(np.count_nonzero(valid)),
                            "mask_overlap": bbox_mask_overlap(box, mask),
                            "area": bbox_area(box),
                        }
                    )
    if not candidates:
        meta = {
            "person_bboxes": pred_bboxes,
            "person_scores": pred_scores,
            "hand_candidates": [],
            "selected_indices": [],
        }
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), meta

    candidates.sort(key=lambda item: (item["mask_overlap"], item["area"]), reverse=True)
    selected = candidates[0]
    meta = {
        "person_bboxes": pred_bboxes,
        "person_scores": pred_scores,
        "hand_candidates": candidates,
        "selected_indices": [0],
    }
    return selected["bbox_xyxy"].reshape(1, 4).astype(np.float32), np.asarray([selected["is_right"]], dtype=np.float32), meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs/0713test_phase1_4_vlm_qwen37_test")
    parser.add_argument("--body-detector", choices=["vitdet", "regnety"], default="regnety")
    parser.add_argument("--rescale-factor", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--person-score-thresh", type=float, default=0.5)
    parser.add_argument("--keypoint-thresh", type=float, default=0.5)
    parser.add_argument("--allow-left", action="store_true")
    parser.add_argument("--fallback-to-lollipop", action="store_true")
    parser.add_argument("--fallback-mask-pad", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    os.chdir(HAMER_DIR)
    from hamer.configs import CACHE_DIR_HAMER
    from hamer.datasets.vitdet_dataset import DEFAULT_MEAN, DEFAULT_STD, ViTDetDataset
    from hamer.models import DEFAULT_CHECKPOINT, download_models, load_hamer
    from hamer.utils import recursive_to
    from hamer.utils.renderer import Renderer, cam_crop_to_full
    from vitpose_model import ViTPoseModel

    output_root = args.output_root
    cases = collect_cases(output_root)
    if args.limit > 0:
        cases = cases[: args.limit]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)
    model = model.to(device).eval()
    detector = load_body_detector(args.body_detector)
    cpm = ViTPoseModel(device)
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    faces = np.asarray(model.mano.faces)
    light_blue = (0.65098039, 0.74117647, 0.85882353)

    summary = []
    for idx, case in enumerate(cases, start=1):
        print(f"[{idx:02d}/{len(cases):02d}] {case['sample_id']} {case['lollipop']}", flush=True)
        out_dir = Path(case["out_dir"])
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        img_cv2 = cv2.imread(str(case["rgb"]))
        mask = cv2.imread(str(case["mask"]), cv2.IMREAD_GRAYSCALE)
        boxes, right, detect_meta = detect_hand_boxes(
            img_cv2,
            mask,
            detector,
            cpm,
            person_score_thresh=args.person_score_thresh,
            keypoint_thresh=args.keypoint_thresh,
            allow_left=args.allow_left,
        )
        used_fallback = False
        if len(boxes) == 0 and args.fallback_to_lollipop:
            boxes = np.asarray([bbox_from_mask(mask, pad=args.fallback_mask_pad)], dtype=np.float32)
            right = np.asarray([1.0], dtype=np.float32)
            used_fallback = True

        mesh_records = []
        rendered_patches = []
        all_verts: list[np.ndarray] = []
        all_cam_t: list[np.ndarray] = []
        all_right: list[float] = []
        scaled_focal_length = None
        img_size = None

        if len(boxes):
            dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            hand_index = 0
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
                    pred_cam,
                    box_center,
                    box_size,
                    img_size,
                    scaled_focal_length,
                ).detach().cpu().numpy()

                for n in range(batch["img"].shape[0]):
                    is_right = float(batch["right"][n].detach().cpu().numpy())
                    verts = pred["pred_vertices"][n].detach().cpu().numpy()
                    verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
                    keypoints = pred["pred_keypoints_3d"][n].detach().cpu().numpy()
                    keypoints[:, 0] = (2 * is_right - 1) * keypoints[:, 0]
                    cam_t = pred_cam_t_full[n]
                    all_verts.append(verts)
                    all_cam_t.append(cam_t)
                    all_right.append(is_right)

                    mesh = trimesh.Trimesh(vertices=verts + cam_t.reshape(1, 3), faces=faces, process=False)
                    mesh.export(out_dir / f"hand_{hand_index:02d}_camera.obj")
                    np.savetxt(out_dir / f"hand_{hand_index:02d}_cam_t.txt", cam_t.reshape(1, 3))
                    np.save(out_dir / f"hand_{hand_index:02d}_vertices_camera.npy", verts + cam_t.reshape(1, 3))
                    np.save(out_dir / f"hand_{hand_index:02d}_keypoints_camera.npy", keypoints + cam_t.reshape(1, 3))
                    np.save(out_dir / f"hand_{hand_index:02d}_keypoints_mano_local.npy", keypoints)
                    mesh_records.append(
                        {
                            "mesh_file": out_dir / f"hand_{hand_index:02d}_camera.obj",
                            "vertices_camera_npy": out_dir / f"hand_{hand_index:02d}_vertices_camera.npy",
                            "keypoints_camera_npy": out_dir / f"hand_{hand_index:02d}_keypoints_camera.npy",
                            "keypoints_mano_local_npy": out_dir / f"hand_{hand_index:02d}_keypoints_mano_local.npy",
                            "cam_t": cam_t,
                            "is_right": bool(is_right),
                        }
                    )
                    hand_index += 1

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

        bbox_overlay = img_cv2.copy()
        for box in boxes:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            cv2.rectangle(bbox_overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.imwrite(str(out_dir / "detected_hand_bbox_overlay.png"), bbox_overlay)

        if all_verts and img_size is not None and scaled_focal_length is not None:
            cam_view = renderer.render_rgba_multiple(
                all_verts,
                cam_t=all_cam_t,
                render_res=img_size[0],
                is_right=all_right,
                mesh_base_color=light_blue,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
            overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
            overlay_bgr = (255 * overlay[:, :, ::-1]).clip(0, 255).astype(np.uint8)
            cv2.imwrite(str(out_dir / "hand_mesh_overlay.png"), overlay_bgr)
            cv2.imwrite(str(out_dir / "comparison.png"), np.concatenate([bbox_overlay, overlay_bgr], axis=1))
            if rendered_patches:
                patch = (255 * np.concatenate(rendered_patches, axis=0)[:, :, ::-1]).clip(0, 255).astype(np.uint8)
                cv2.imwrite(str(out_dir / "crop_regression.png"), patch)

        meta = {
            "sample_id": case["sample_id"],
            "lollipop": case["lollipop"],
            "rgb": case["rgb"],
            "mask": case["mask"],
            "bbox_xyxy": boxes[0].tolist() if len(boxes) else None,
            "bbox_source": "lollipop_fallback" if used_fallback else "official_detectron2_vitpose",
            "used_fallback": used_fallback,
            "body_detector": args.body_detector,
            "rescale_factor": args.rescale_factor,
            "person_score_thresh": args.person_score_thresh,
            "keypoint_thresh": args.keypoint_thresh,
            "detection": detect_meta,
            "mesh_records": mesh_records,
        }
        (out_dir / "hamer_metadata.json").write_text(
            json.dumps(as_jsonable(meta), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        summary.append(meta)

    summary_file = output_root / "kettle_hamer_official_summary.json"
    summary_file.write_text(json.dumps(as_jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {summary_file}", flush=True)


if __name__ == "__main__":
    main()
