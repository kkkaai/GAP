from __future__ import annotations

import numpy as np

from prosthetic_grasp.common.types import GeneratedCandidate, IntentSpec


def score_candidate(candidate: GeneratedCandidate, base_rgb: np.ndarray, intent: IntentSpec) -> float:
    diff = np.mean(np.abs(candidate.image.astype(np.float32) - base_rgb.astype(np.float32)))
    intensity_bonus = 0.0
    if intent.task == "pick_up":
        intensity_bonus += 0.1
    return float(max(0.0, 1.0 - diff / 255.0) + intensity_bonus)


def rank_candidates(
    candidates: list[GeneratedCandidate],
    base_rgb: np.ndarray,
    intent: IntentSpec,
) -> list[GeneratedCandidate]:
    for candidate in candidates:
        candidate.score = score_candidate(candidate, base_rgb, intent)
    return sorted(candidates, key=lambda item: item.score, reverse=True)

