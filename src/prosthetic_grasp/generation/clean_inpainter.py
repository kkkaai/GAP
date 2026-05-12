from __future__ import annotations

import numpy as np


class CleanInpainter:
    """Stub local inpainting.

    Replace with LaMa integration in the next iteration.
    """

    def __init__(self, model_name: str = "stub") -> None:
        self.model_name = model_name

    def inpaint(self, rgb: np.ndarray, removal_mask: np.ndarray) -> np.ndarray:
        output = rgb.copy()
        if not removal_mask.any():
            return output

        mean_color = rgb[~removal_mask].mean(axis=0) if (~removal_mask).any() else np.array([127, 127, 127])
        output[removal_mask] = mean_color.astype(np.uint8)
        return output

