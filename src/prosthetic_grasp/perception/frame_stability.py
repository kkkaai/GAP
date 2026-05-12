from __future__ import annotations

import numpy as np

from prosthetic_grasp.common.types import Burst, SensorFrame


def blur_score(rgb: np.ndarray) -> float:
    gray = rgb.mean(axis=2).astype(np.float32)
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    return float(np.mean(gx**2) + np.mean(gy**2))


def is_stable_burst(burst: Burst, max_blur_score: float) -> bool:
    if not burst.frames:
        return False
    scores = [blur_score(frame.rgb) for frame in burst.frames]
    return all(score <= max_blur_score for score in scores)


def select_best_frame(burst: Burst) -> SensorFrame:
    return max(burst.frames, key=lambda frame: blur_score(frame.rgb))

