# from __future__ import annotations

# from dataclasses import dataclass

# import numpy as np

# from prosthetic_grasp.common.image_editing import (
#     FLUX_FILL_MODEL_ID,
#     ImageEditConfig,
#     STABILITY_INPAINT_ENDPOINT,
#     crop_from_mask,
#     paste_crop_back,
#     run_masked_image_edit,
# )
# from prosthetic_grasp.common.types import Phase4InpaintResult


# @dataclass
# class Phase4InpaintConfig(ImageEditConfig):
#     mode: str = "local"
#     model_name: str = "flux-fill"
#     model_id: str = FLUX_FILL_MODEL_ID
#     stability_endpoint: str = STABILITY_INPAINT_ENDPOINT
#     prompt: str = (
#         "Generate a realistic healthy adult right hand in first-person view, naturally interacting with the "
#         "target object inside the masked region. Preserve the object identity and scene layout. Avoid extra "
#         "fingers, deformed anatomy, and changes outside the masked region."
#     )


# class Phase4Inpaint:
#     """Use an image-edit model to generate the healthy human-hand interaction image."""

#     def __init__(self, config: Phase4InpaintConfig) -> None:
#         self.config = config

#     def run(self, image_rgb: np.ndarray, lollipop_mask: np.ndarray) -> Phase4InpaintResult:
#         rgb_crop, mask_crop, roi_box = crop_from_mask(image_rgb, lollipop_mask, self.config.pad_ratio)
#         inpaint_crop = run_masked_image_edit(self.config, rgb_crop, mask_crop)
#         inpaint_full = paste_crop_back(image_rgb, inpaint_crop, roi_box)
#         return Phase4InpaintResult(
#             roi_box=roi_box,
#             rgb_crop=rgb_crop,
#             mask_crop=mask_crop,
#             inpaint_crop=inpaint_crop,
#             inpaint_full=inpaint_full,
#             mode=self.config.mode,
#             model_name=self.config.model_name,
#             prompt=self.config.prompt,
#         )


from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prosthetic_grasp.common.image_editing import (
    FLUX_FILL_MODEL_ID,
    ImageEditConfig,
    STABILITY_INPAINT_ENDPOINT,
    crop_from_mask,
    paste_crop_back,
    run_masked_image_edit,
)
from prosthetic_grasp.common.types import Phase4InpaintResult

@dataclass
class Phase4InpaintConfig(ImageEditConfig):
    mode: str = "local"
    model_name: str = "flux-fill"
    model_id: str = FLUX_FILL_MODEL_ID
    stability_endpoint: str = STABILITY_INPAINT_ENDPOINT
    prompt: str = (
        "Generate a realistic healthy adult right hand in first-person view, naturally interacting with the "
        "target object inside the masked region. Preserve the object identity and scene layout. Avoid extra "
        "fingers, deformed anatomy, and changes outside the masked region."
    )

class Phase4Inpaint:
    """Use an image-edit model to generate the healthy human-hand interaction image."""

    def __init__(self, config: Phase4InpaintConfig) -> None:
        self.config = config

    def run(self, image_rgb: np.ndarray, lollipop_mask: np.ndarray) -> Phase4InpaintResult:
        rgb_crop, mask_crop, roi_box = crop_from_mask(image_rgb, lollipop_mask, self.config.pad_ratio)
        inpaint_crop = run_masked_image_edit(self.config, rgb_crop, mask_crop)
        inpaint_full = paste_crop_back(image_rgb, inpaint_crop, roi_box)
        return Phase4InpaintResult(
            roi_box=roi_box,
            rgb_crop=rgb_crop,
            mask_crop=mask_crop,
            inpaint_crop=inpaint_crop,
            inpaint_full=inpaint_full,
            mode=self.config.mode,
            model_name=self.config.model_name,
            prompt=self.config.prompt,
        )
