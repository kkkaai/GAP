from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from prosthetic_grasp.common.types import Phase4FluxFillResult, ROIBox
from prosthetic_grasp.phases.phase3_inpaint import Phase3Inpaint


@dataclass
class Phase4FluxFillConfig:
    mode: str = "local"
    model_id: str = "black-forest-labs/FLUX.1-Fill-dev"
    prompt: str = "A highly realistic background behind the object, no human hand, seamless continuation of the surface, extremely detailed, 8k"
    guidance_scale: float = 30.0
    num_inference_steps: int = 50
    pad_ratio: float = 0.25


class Phase4FluxFill:
    def __init__(self, config: Phase4FluxFillConfig) -> None:
        self.config = config
        self._pipe = None

    def _ensure_local_pipe(self):
        if self._pipe is not None:
            return self._pipe
        import torch
        from diffusers import FluxFillPipeline

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = FluxFillPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        pipe = pipe.to(device)
        self._pipe = pipe
        return pipe

    def run(self, image_rgb: np.ndarray, lollipop_mask: np.ndarray) -> Phase4FluxFillResult:
        rgb_crop, mask_crop, roi_box = Phase3Inpaint.crop_from_mask(image_rgb, lollipop_mask, self.config.pad_ratio)

        if self.config.mode == "api":
            raise NotImplementedError(
                "phase4_flux_fill API mode is intentionally left blank. "
                "Add a BFL API client later while keeping this interface stable."
            )

        pipe = self._ensure_local_pipe()
        crop_pil = Image.fromarray(rgb_crop)
        mask_pil = Image.fromarray(mask_crop.astype(np.uint8) * 255)
        result = pipe(
            prompt=self.config.prompt,
            image=crop_pil,
            mask_image=mask_pil,
            guidance_scale=self.config.guidance_scale,
            num_inference_steps=self.config.num_inference_steps,
        ).images[0]
        flux_crop = np.array(result)
        flux_full = Phase3Inpaint.paste_crop_back(image_rgb, flux_crop, roi_box)
        return Phase4FluxFillResult(
            roi_box=roi_box,
            flux_crop=flux_crop,
            flux_full=flux_full,
            prompt=self.config.prompt,
        )
