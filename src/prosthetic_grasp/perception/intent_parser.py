from __future__ import annotations

from prosthetic_grasp.common.types import IntentSpec


def parse_intent(text: str) -> IntentSpec:
    lowered = text.lower().strip()

    task = "pick_up"
    if "handover" in lowered or "hand over" in lowered:
        task = "handover"
    elif "use" in lowered:
        task = "use"
    elif "reposition" in lowered or "move" in lowered:
        task = "reposition"

    target = "unknown"
    for token in ("bottle", "mug", "cup", "spray", "can", "tool", "box"):
        if token in lowered:
            target = token
            break

    grasp_part = "unspecified"
    for token in ("handle", "body", "cap", "side", "top"):
        if token in lowered:
            grasp_part = token
            break

    return IntentSpec(
        raw_text=text,
        target=target,
        task=task,
        grasp_part=grasp_part,
    )

