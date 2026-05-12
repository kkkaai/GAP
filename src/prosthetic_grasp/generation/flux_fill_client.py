from __future__ import annotations

import numpy as np

from prosthetic_grasp.common.types import GeneratedCandidate


class FluxFillClient:
    """Stub grasp candidate generator.

    Replace with a FLUX Fill API client or a self-hosted backend.
    """

    def __init__(self, model_name: str = "stub") -> None:
        self.model_name = model_name

    def generate(
        self,
        rgb_clean: np.ndarray,
        prompt: str,
        num_samples: int,
        roi: tuple[int, int, int, int],
    ) -> list[GeneratedCandidate]:
        candidates: list[GeneratedCandidate] = []
        x0, y0, x1, y1 = roi
        height, width = rgb_clean.shape[:2]

        for seed in range(num_samples):
            image = rgb_clean.copy()
            palm_x = min(max((x0 + x1) // 2 + seed * 4, 0), width - 1)
            palm_y = min(max((y0 + y1) // 2, 0), height - 1)
            rr = max((y1 - y0) // 8, 10)

            yy, xx = np.indices((height, width))
            palm = (xx - palm_x) ** 2 + (yy - palm_y) ** 2 <= rr**2
            image[palm] = np.array([220, 180, 160], dtype=np.uint8)

            candidates.append(
                GeneratedCandidate(
                    image=image,
                    prompt=prompt,
                    seed=seed,
                    metadata={"roi": [x0, y0, x1, y1]},
                )
            )

        return candidates

