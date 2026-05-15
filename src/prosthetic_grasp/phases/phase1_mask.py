from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from prosthetic_grasp.common.io import load_image
from prosthetic_grasp.common.types import Phase1MaskResult


@dataclass
class Phase1MaskConfig:
    text_prompt: str = "hand. coffee cup."
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    mode: str = "precomputed"
    precomputed_mask_path: str | None = None


class Phase1Mask:
    """Phase 1 placeholder.

    The validated notebook currently runs Grounded-SAM-2 directly in Colab.
    In local src/, we only keep a minimal adapter that can load a precomputed
    mask saved from the notebook. The real local Grounded-SAM-2 integration is
    intentionally left blank for now.
    """

    def __init__(self, config: Phase1MaskConfig) -> None:
        self.config = config

    def run(self, image_rgb: np.ndarray) -> Phase1MaskResult:
        if self.config.mode != "precomputed" or not self.config.precomputed_mask_path:
            raise NotImplementedError(
                "phase1_mask local Grounded-SAM-2 integration is not implemented yet. "
                "Use mode='precomputed' with a saved mask from the notebook."
            )

        mask = load_image(self.config.precomputed_mask_path)
        if mask.ndim == 3:
            mask = mask[..., 0]
        return Phase1MaskResult(mask=mask > 127, metadata={"source": str(Path(self.config.precomputed_mask_path))})
