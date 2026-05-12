from __future__ import annotations

from prosthetic_grasp.common.types import IntentSpec


def build_generation_prompt(intent: IntentSpec) -> str:
    return (
        "Given the first-person scene image, generate a realistic image where a healthy adult "
        "right hand performs the intended interaction with the target object.\n\n"
        f"Target object: {intent.target}\n"
        f"Intent: {intent.task}\n"
        f"Preferred grasp part: {intent.grasp_part}\n"
        "Camera viewpoint: first-person head-mounted camera\n\n"
        "Requirements:\n"
        "- preserve the original object identity and local scene layout\n"
        "- only add a realistic healthy right hand\n"
        "- the hand should make physically plausible contact with the target object\n"
        "- avoid extra fingers, deformed hand anatomy, broken wrists, or impossible grasping\n"
        "- avoid changing object pose, object scale, or unrelated scene content\n"
    )

