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
