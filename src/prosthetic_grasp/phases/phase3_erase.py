from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prosthetic_grasp.common.image_editing import (
    FLUX_FILL_MODEL_ID,
    ImageEditConfig,
    STABILITY_ERASE_ENDPOINT,
    crop_from_mask,
    paste_crop_back,
    run_masked_image_edit,
)
from prosthetic_grasp.common.types import Phase3EraseResult


@dataclass
class Phase3EraseConfig(ImageEditConfig):
    mode: str = "local"
    model_name: str = "flux-fill"
    model_id: str = FLUX_FILL_MODEL_ID
    stability_endpoint: str = STABILITY_ERASE_ENDPOINT
    prompt: str = (
        "Remove the current hand or prosthetic hand from the masked region and complete the hidden background "
        "and object appearance naturally. Do not add any human hand. Preserve scene layout, object identity, "
        "and content outside the masked region."
    )


class Phase3Erase:
    """Use an image-edit model to erase the current hand and complete the occluded background."""

    def __init__(self, config: Phase3EraseConfig) -> None:
        self.config = config

    def run(self, image_rgb: np.ndarray, lollipop_mask: np.ndarray) -> Phase3EraseResult:
        rgb_crop, mask_crop, roi_box = crop_from_mask(image_rgb, lollipop_mask, self.config.pad_ratio)
        erased_crop = run_masked_image_edit(self.config, rgb_crop, mask_crop)
        erased_full = paste_crop_back(image_rgb, erased_crop, roi_box)
        return Phase3EraseResult(
            roi_box=roi_box,
            rgb_crop=rgb_crop,
            mask_crop=mask_crop,
            erased_crop=erased_crop,
            erased_full=erased_full,
            mode=self.config.mode,
            model_name=self.config.model_name,
            prompt=self.config.prompt,
        )
