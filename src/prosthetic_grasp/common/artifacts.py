from __future__ import annotations

from pathlib import Path

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
        save_image(output_path / "phase3_erase_crop.png", result.phase3_erase.erased_crop)
        save_image(output_path / "phase3_erase_full.png", result.phase3_erase.erased_full)
    if result.phase4_inpaint is not None:
        save_image(output_path / "phase4_inpaint_crop.png", result.phase4_inpaint.inpaint_crop)
        save_image(output_path / "phase4_inpaint_full.png", result.phase4_inpaint.inpaint_full)

    return output_path
