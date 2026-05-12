from __future__ import annotations

from prosthetic_grasp.common.types import ProstheticAction


def optimize_action(action: ProstheticAction) -> ProstheticAction:
    # Placeholder local optimization.
    # Keep the interface so a true neighborhood search can replace this later.
    action.aperture = max(0.1, min(1.0, action.aperture))
    action.force_level = max(0.05, min(1.0, action.force_level))
    action.confidence = min(1.0, action.confidence + 0.02)
    return action

