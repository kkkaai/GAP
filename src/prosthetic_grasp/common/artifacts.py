from __future__ import annotations

from pathlib import Path

import numpy as np

from prosthetic_grasp.common.io import save_image, save_json, save_mask
from prosthetic_grasp.common.types import PipelineResult


def save_pipeline_artifacts(result: PipelineResult, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_json(output_path / "result.json", result.to_json_dict())

    if result.phase1_mask is not None:
        save_mask(output_path / "phase1_mask.png", result.phase1_mask.mask)
    if result.phase2_lollipop is not None:
        save_mask(output_path / "phase2_lollipop.png", result.phase2_lollipop.lollipop_mask)
    if result.phase3_erase is not None:
        save_image(output_path / "phase3_rgb_crop.png", result.phase3_erase.rgb_crop)
        save_mask(output_path / "phase3_mask_crop.png", result.phase3_erase.mask_crop)
        save_image(output_path / "phase3_erase_crop.png", result.phase3_erase.erased_crop)
        save_image(output_path / "phase3_erase_full.png", result.phase3_erase.erased_full)
    if result.phase4_inpaint is not None:
        save_image(output_path / "phase4_rgb_crop.png", result.phase4_inpaint.rgb_crop)
        save_mask(output_path / "phase4_mask_crop.png", result.phase4_inpaint.mask_crop)
        save_image(output_path / "phase4_inpaint_crop.png", result.phase4_inpaint.inpaint_crop)
        save_image(output_path / "phase4_inpaint_full.png", result.phase4_inpaint.inpaint_full)
    if result.phase5_mano is not None and hasattr(result.phase5_mano, "hands"):
        payload = {}
        if result.phase5_mano.faces is not None:
            payload["faces"] = result.phase5_mano.faces
        for hand in result.phase5_mano.hands:
            prefix = f"hand_{hand.hand_index}"
            payload[f"{prefix}_is_right"] = np.array(hand.is_right)
            payload[f"{prefix}_bbox_xyxy"] = hand.bbox_xyxy
            payload[f"{prefix}_keypoints_2d"] = hand.keypoints_2d
            payload[f"{prefix}_vertices"] = hand.vertices
            payload[f"{prefix}_keypoints_3d"] = hand.keypoints_3d
            payload[f"{prefix}_pred_cam"] = hand.pred_cam
            payload[f"{prefix}_pred_cam_t_crop"] = hand.pred_cam_t_crop
            payload[f"{prefix}_pred_cam_t_full"] = hand.pred_cam_t_full
            for name, value in hand.mano_params.items():
                payload[f"{prefix}_mano_{name}"] = value
        if payload:
            np.savez_compressed(output_path / "phase5_mano.npz", **payload)

    if result.phase6_prosthetic_action is not None and hasattr(result.phase6_prosthetic_action, "action"):
        phase6 = result.phase6_prosthetic_action
        payload = {
            "action": phase6.action,
            "action_names": np.asarray(phase6.action_names),
        }
        if phase6.mano_wrist is not None:
            payload["mano_wrist"] = phase6.mano_wrist
        if phase6.mano_fingertips is not None:
            payload["mano_fingertips"] = phase6.mano_fingertips
        if phase6.target_fingertips_wrist is not None:
            payload["target_fingertips_wrist"] = phase6.target_fingertips_wrist
        if phase6.prosthetic_fingertips_wrist is not None:
            payload["prosthetic_fingertips_wrist"] = phase6.prosthetic_fingertips_wrist
        if phase6.fingertip_error is not None:
            payload["fingertip_error"] = phase6.fingertip_error
        np.savez_compressed(output_path / "phase6_prosthetic_action.npz", **payload)

    return output_path
