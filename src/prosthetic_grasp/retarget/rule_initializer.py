from __future__ import annotations

from prosthetic_grasp.common.types import HumanPrior2D, IntentSpec, ProstheticAction


def initialize_action(
    prior: HumanPrior2D,
    intent: IntentSpec,
    default_force_level: float,
    default_closure_speed: float,
    min_aperture: float,
    max_aperture: float,
) -> ProstheticAction:
    grasp_type = prior.grasp_family_hint
    thumb_mode = "oppose" if grasp_type != "lateral" else "side"

    aperture = prior.hand_span_px / 256.0
    aperture = max(min_aperture, min(max_aperture, aperture))

    wrist_rotation = 0.0
    if grasp_type == "lateral":
        wrist_rotation = 25.0
    elif grasp_type == "pinch":
        wrist_rotation = 10.0

    force_level = default_force_level
    if intent.task == "handover":
        force_level = min(default_force_level, 0.25)
    elif intent.task == "use":
        force_level = max(default_force_level, 0.45)

    return ProstheticAction(
        grasp_type=grasp_type,
        thumb_mode=thumb_mode,
        wrist_rotation=wrist_rotation,
        aperture=aperture,
        closure_speed=default_closure_speed,
        force_level=force_level,
        confidence=prior.confidence,
    )

