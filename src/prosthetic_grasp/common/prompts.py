from __future__ import annotations


PHASE3_ERASE_PROMPT = """
# Restore Object-Only Daily-Use Scene

Edit the input image by removing only the visible human hand, prosthetic hand, glove, wrist, sleeve, or arm from the masked region. Reconstruct the occluded object parts and background so the image becomes a realistic object-only scene.

## Task and Scene Scope

- The scene is a daily-use manipulation scene for prosthetic grasp research.
- The target object should remain available for a future prosthetic or human grasp action.
- Complete only the content hidden by the removed hand or prosthetic device.
- If a reference image is provided, use it only to infer the original object, material, geometry, texture, and background continuity.

## Object and Background Preservation

- Keep the target object in its original position, scale, orientation, inclination, material, color, shape, topology, and identity.
- Restore occluded object structures such as handles, rims, edges, holes, buttons, grips, caps, labels, and contact boundaries with continuous geometry.
- Preserve the table, support surface, background, lighting, shadows, reflections, camera viewpoint, and all surrounding objects.
- Do not move, deform, replace, redesign, or duplicate the object.
- Do not change the number, position, or layout of surrounding objects.

## Masked Region Editing

- Use the mask as the editable region.
- Remove every visible hand/prosthetic-related part inside the mask.
- Reconstruct the missing object and background naturally inside the mask.
- Keep all unmasked content visually unchanged.

## Forbidden Content

- Do not add any human hand, prosthetic hand, robotic hand, glove, wrist, forearm, sleeve, tool, gripper, claw, or extra object.
- Do not add new attachments or structures connected to the object.
- Avoid blurry patches, repeated textures, warped object edges, broken handles, missing rims, inconsistent shadows, and obvious inpainting artifacts.

## Output

Return a realistic image of the same scene with the hand removed and the occluded regions completed naturally.
""".strip()


PHASE4_INTENTION_PROMPT_TEMPLATE = """
You are a prosthetic grasp intention planner.

You will receive:
1. A first-person scene image, possibly showing a prosthetic hand.
2. A short user task instruction transcribed from speech, if available.

## User Task Instruction

{task_instruction}

## Planning Goal

Generate a concise English intention for a downstream image-editing model that will synthesize a healthy adult right human hand grasping or interacting with the target object.

## Reasoning Rules

- If the user task instruction is available and clearly specifies the task, prioritize it.
- If the user task instruction is empty, vague, or incomplete, infer the most plausible daily-use grasp intention from the scene image.
- Use the image to identify the target object, its affordance, usable contact region, handle, opening, button, grip surface, or support surface.
- Prefer daily-use prosthetic grasp tasks such as drinking from a cup, picking up an object, holding a utensil, using a tool, opening a container, pressing a button, carrying an item, or stabilizing an item.
- The intention must specify the right-hand grasp style, contact region, and task purpose.
- Keep the intention concrete, physically plausible, and useful for image generation.
- Do not mention masks, image editing, inpainting, prompts, VLMs, or models.

## Output Format

Output only valid JSON with this schema:

{{
  "target_object": "short English object name",
  "daily_task": "short English daily-use task",
  "grasp_type": "short English grasp type",
  "phase4_intention": "one concise English sentence describing the right-hand grasp style, contact region, and task purpose"
}}

## Example

If the image shows a mug and the user says "I want to pick up the cup and drink water", a good output is:

{{
  "target_object": "mug",
  "daily_task": "drink from the mug",
  "grasp_type": "right-hand handle grasp",
  "phase4_intention": "Grasp the mug handle with the right hand for drinking, with the thumb on one side of the handle and four fingers curling naturally through or around the handle."
}}
""".strip()


def build_phase4_intention_prompt(task_instruction: str | None = None) -> str:
    task = (task_instruction or "").strip()
    if task:
        task_block = task
    else:
        task_block = "No user task instruction was provided. Infer the most plausible daily-use grasp intention from the image."
    return PHASE4_INTENTION_PROMPT_TEMPLATE.format(task_instruction=task_block)


DEFAULT_PHASE4_INTENTION = """
Use a natural daily-use grasp that is appropriate for the visible target object. If the object has a handle, the right hand should grasp the handle. If the object is meant to be picked up, held, pressed, opened, carried, or stabilized, choose a plausible grasp for that action.
""".strip()


PHASE4_HAND_GENERATION_PROMPT_TEMPLATE = """
# Generate First-Person Right-Hand Grasp for Daily-Use Object

Generate a photorealistic healthy adult human right hand inside the masked region, naturally grasping or interacting with the target object for a daily-use prosthetic grasp task.

## Grasping Style Consistent with Object Use

{intention}

- The grasp should match the object's intended daily function and the requested task intention above.
- Use a physically plausible right-hand grasp for common daily activities such as holding a cup, picking up a utensil, using a tool, opening a container, pressing a button, carrying an object, or stabilizing an item.
- The hand should approach from a natural first-person viewpoint, as if the viewer is seeing their own right hand interacting with the object.
- If a reference image is provided, use it only as guidance for hand pose, grasp style, skin tone, lighting, or occlusion. Do not copy unrelated objects or background changes from the reference image.

## Object Placement and Scene Consistency

- Keep the object in its original position, scale, orientation, inclination, material, color, shape, topology, and identity.
- Do not move, deform, replace, redesign, duplicate, or resize the object.
- Preserve object details such as handles, rims, edges, holes, labels, buttons, caps, and contact surfaces.
- Preserve the table, support surface, background, lighting, shadows, reflections, camera viewpoint, and all surrounding objects.
- Do not change the number, position, or layout of surrounding objects.

## Hand Anatomy and Occlusion

- Generate only one right hand.
- The hand must be anatomically correct with five fingers total: one thumb and four fingers.
- The thumb and fingers should form a plausible grasp with correct joint bends, contact points, and occlusion relationships.
- Fingers may be partially occluded by the object or by other fingers, but there must be no missing finger, extra finger, fused finger, duplicated nail, broken joint, or deformed anatomy.
- The hand and object must not intersect or penetrate unnaturally.
- Optionally include part of the wrist or forearm only if it improves the first-person spatial relationship.

## Masked Region Editing

- Use the mask as the hand-generation region.
- Place the hand primarily inside the mask, while allowing natural contact and occlusion with the object.
- Keep the interaction focused on the target object.

## Visual Style and Integration

- Match the original image style, camera perspective, resolution, lighting direction, color tone, texture detail, shadows, and depth of field.
- The generated hand should blend seamlessly into the scene.
- Avoid cartoon-like rendering, synthetic skin, glossy plastic skin, distorted shadows, blurred boundaries, and unnatural composition.

## Forbidden Content

- Do not generate a left hand, second hand, prosthetic hand, robotic hand, glove, sleeve-dominant arm, tool, gripper, claw, or extra object.
- Do not alter the scene outside the intended interaction.

## Output

Return a realistic image of the same scene with a natural right-hand grasp of the target object.
""".strip()


def build_phase4_hand_generation_prompt(intention: str | None = None) -> str:
    return PHASE4_HAND_GENERATION_PROMPT_TEMPLATE.format(
        intention=(intention or DEFAULT_PHASE4_INTENTION).strip()
    )


PHASE4_HAND_GENERATION_PROMPT = build_phase4_hand_generation_prompt()


VLM_QUALITY_STAGE_GUIDES = {
    "phase2_lollipop": """
Evaluate whether the lollipop prior and its short task are suitable for first-person prosthetic grasp data generation.
Focus on:
- mask_affordance_score: the lollipop circle is centered on a functional contact region of the target object.
- tail_direction_score: the lollipop tail extends from the contact region toward a plausible wrist/forearm entry direction and reaches the image boundary.
- task_mask_alignment: the short task matches the selected object part, such as handle, lid, cap, rim, body, or side.
- object_relevance: the lollipop is attached to the target object rather than background or a distractor object.
- diversity_score: if multiple candidates are shown, they cover distinct useful affordances instead of repeating the same grasp.
""".strip(),
    "phase4_generation": """
Evaluate whether the generated healthy right-hand interaction image is suitable for MANO recovery and retargeting.
Focus on:
- hand_completeness: exactly one right hand with one thumb and four fingers; no missing, extra, fused, or broken fingers.
- anatomical_plausibility: realistic finger bends, palm shape, wrist connection, and first-person hand anatomy.
- contact_plausibility: the hand visibly contacts or wraps the target object at the intended region without obvious floating or impossible penetration.
- task_alignment: the grasp matches the task, for example grasp the handle, hold the body, pinch the lid, or stabilize the rim.
- egocentric_consistency: the hand/forearm enters naturally from the image boundary as a first-person right hand.
- object_preservation: the target object identity, geometry, scale, pose, and important affordance parts are preserved.
- artifact_level: low visual artifacts, texture corruption, duplicated structures, or obvious generation defects.
""".strip(),
    "phase5_mano": """
Evaluate whether the MANO/HaMeR hand reconstruction overlay correctly matches the visible hand.
Focus on:
- mesh_image_alignment: the rendered hand mesh or skeleton aligns with the visible hand silhouette and joints.
- hand_orientation: palm/back orientation is correct and not flipped.
- left_right_consistency: the detected hand side is consistent with a right hand when expected.
- finger_pose_consistency: reconstructed fingers match the visible finger positions and bends.
- wrist_direction: wrist and forearm direction match the image.
- flip_risk: 0 means no flip risk; 5 means obvious 180-degree, palm/back, or left/right flip.
""".strip(),
    "phase6_object_pose": """
Evaluate whether the object detection, object mask, and projected 6D pose overlay are visually trustworthy.
Focus on:
- object_mask_alignment: the object mask or contour covers the actual target object and excludes background/distractors.
- pose_overlay_alignment: the projected 3D box/axes align with the visible object silhouette, orientation, and perspective.
- scale_consistency: the rendered pose has plausible size relative to the image object.
- object_identity_consistency: the detected object corresponds to the intended target object.
- occlusion_consistency: the pose remains plausible under hand/object occlusion.
""".strip(),
    "phase6_retarget": """
Evaluate whether the retargeted prosthetic hand result is visually suitable as a q_goal teacher label.
Focus on:
- contact_geometry_plausibility: prosthetic fingers/palm contact the intended object region.
- penetration_visual_risk: 0 means no obvious penetration; 5 means severe visual interpenetration.
- retarget_task_alignment: the final prosthetic grasp still matches the requested task.
- grasp_stability_plausibility: the contact pattern looks stable for the object and action.
- hand_object_scale_consistency: prosthetic hand size and wrist pose are plausible relative to the object.
""".strip(),
}


VLM_QUALITY_PROMPT_TEMPLATE = """
You are a strict visual quality evaluator for a prosthetic grasp teacher-data pipeline.

The input may include the source RGB image, lollipop or object overlays, generated hand images, MANO/HaMeR overlays, FoundationPose overlays, or retargeting renders.

Stage: {stage}
Target object: {object_name}
Task instruction: {task_instruction}

Stage-specific evaluation guide:
{stage_guide}

Scoring rules:
- Use integer scores from 0 to 5 only.
- 5 = excellent and directly usable.
- 4 = usable with minor issues.
- 3 = marginal but usable with caution.
- 2 = likely wrong or risky.
- 1 = severe error.
- 0 = invalid, missing, or impossible to judge.
- For risk metrics such as flip_risk and penetration_visual_risk, 0 is best and 5 is worst.
- Be strict. A visually obvious failure should not receive an overall score above 2.
- Return only valid JSON. Do not output markdown.

Required JSON schema:
{{
  "stage": "{stage}",
  "overall_score": 0,
  "pass": false,
  "scores": {{
    "metric_name": 0
  }},
  "failure_tags": [
    "short_machine_readable_tag"
  ],
  "reason": "one short sentence explaining the decision"
}}
""".strip()


def build_vlm_quality_prompt(
    *,
    stage: str,
    object_name: str | None = None,
    task_instruction: str | None = None,
) -> str:
    stage_key = stage.strip()
    stage_guide = VLM_QUALITY_STAGE_GUIDES.get(
        stage_key,
        "Evaluate whether this visual result is suitable for downstream prosthetic grasp teacher-data generation.",
    )
    return VLM_QUALITY_PROMPT_TEMPLATE.format(
        stage=stage_key,
        object_name=(object_name or "unknown target object").strip() or "unknown target object",
        task_instruction=(task_instruction or "No task instruction was provided.").strip()
        or "No task instruction was provided.",
        stage_guide=stage_guide,
    )
