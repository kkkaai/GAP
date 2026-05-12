from __future__ import annotations

import numpy as np


class ProsthesisSegmentor:
    """Stub segmentor.

    Replace this class with a custom-trained binary segmentation model.
    """

    def __init__(self, model_name: str = "stub") -> None:
        self.model_name = model_name

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        height, width = rgb.shape[:2]
        mask = np.zeros((height, width), dtype=bool)

        # Heuristic placeholder:
        # assume the prosthesis enters from the bottom-right region.
        y0 = int(height * 0.58)
        x0 = int(width * 0.62)
        mask[y0:, x0:] = True

        # Taper the region toward the center.
        for row in range(y0, height):
            extent = int((row - y0) * 0.6)
            start = max(x0 - extent, width // 2)
            mask[row, :start] = False

        return mask

