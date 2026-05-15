from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from prosthetic_grasp.common.types import Phase3InpaintResult, ROIBox


@dataclass
class Phase3InpaintConfig:
    model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    prompt: str = "background preserved, object preserved, realistic completion behind removed hand, no human hand visible"
    guidance_scale: float = 8.0
    strength: float = 0.99
    num_inference_steps: int = 30
    pad_ratio: float = 0.25


class Phase3Inpaint:
    def __init__(self, config: Phase3InpaintConfig) -> None:
        self.config = config
        self._pipe = None

    def _ensure_pipe(self):
        if self._pipe is not None:
            return self._pipe
        import torch
        from diffusers import AutoPipelineForInpainting

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = AutoPipelineForInpainting.from_pretrained(
            self.config.model_id,
            torch_dtype=dtype,
            variant="fp16" if device == "cuda" else None,
        )
        pipe = pipe.to(device)
        pipe.enable_attention_slicing()
        self._pipe = pipe
        return pipe

    @staticmethod
    def crop_from_mask(image, mask, pad_ratio=0.25):
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        h, w = image.shape[:2]
        bh = y1 - y0 + 1
        bw = x1 - x0 + 1
        pad_y = max(int(bh * pad_ratio), 16)
        pad_x = max(int(bw * pad_ratio), 16)
        x0 = max(0, x0 - pad_x)
        y0 = max(0, y0 - pad_y)
        x1 = min(w, x1 + pad_x)
        y1 = min(h, y1 + pad_y)
        return image[y0:y1, x0:x1], mask[y0:y1, x0:x1], ROIBox(x0=x0, y0=y0, x1=x1, y1=y1)

    @staticmethod
    def paste_crop_back(full_rgb, crop_rgb, crop_box: ROIBox):
        out = full_rgb.copy()
        out[crop_box.y0:crop_box.y1, crop_box.x0:crop_box.x1] = crop_rgb
        return out

    def run(self, image_rgb: np.ndarray, lollipop_mask: np.ndarray) -> Phase3InpaintResult:
        pipe = self._ensure_pipe()
        rgb_crop, mask_crop, roi_box = self.crop_from_mask(image_rgb, lollipop_mask, self.config.pad_ratio)
        crop_pil = Image.fromarray(rgb_crop)
        mask_pil = Image.fromarray(mask_crop.astype(np.uint8) * 255)
        result = pipe(
            prompt=self.config.prompt,
            image=crop_pil,
            mask_image=mask_pil,
            guidance_scale=self.config.guidance_scale,
            strength=self.config.strength,
            num_inference_steps=self.config.num_inference_steps,
        ).images[0]
        inpaint_crop = np.array(result)
        inpaint_full = self.paste_crop_back(image_rgb, inpaint_crop, roi_box)
        return Phase3InpaintResult(
            roi_box=roi_box,
            rgb_crop=rgb_crop,
            mask_crop=mask_crop,
            inpaint_crop=inpaint_crop,
            inpaint_full=inpaint_full,
            prompt=self.config.prompt,
        )
