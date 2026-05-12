# Prosthetic Grasp Generation

Minimal V1 project scaffold for the pipeline described in:

- [prosthetic_grasp_generation_plan.md](/Users/bigstepper/VscodeProjects/GAP/prosthetic_grasp_generation_plan.md)
- [prosthetic_grasp_generation_plan_zh.md](/Users/bigstepper/VscodeProjects/GAP/prosthetic_grasp_generation_plan_zh.md)

This codebase implements a runnable skeleton for:

`RGB-D observation -> prosthesis mask -> lollipop -> local ROI -> clean inpaint -> human grasp candidate generation -> grasp prior extraction -> rule/optimization retarget -> execute`

## Scope

This is a **minimal V1 scaffold**:

- explicit target object segmentation is optional and disabled by default
- perception / generation modules are wired through stable interfaces
- most model-backed modules currently ship with safe placeholder implementations
- the package is structured so real models can be dropped in without rewriting the pipeline

## Layout

```text
src/prosthetic_grasp/
  apps/
  common/
  config/
  control/
  extraction/
  generation/
  perception/
  retarget/
```

## Quick Start

Install the package:

```bash
python -m pip install -e .
```

Run the demo app on a single RGB image:

```bash
python -m prosthetic_grasp.apps.run_subaction_grasp \
  --rgb path/to/image.png \
  --instruction "pick up the bottle" \
  --output-dir outputs/demo_run
```

If you do not want to install the package yet, run it directly from the repo root:

```bash
PYTHONPATH=src python -m prosthetic_grasp.apps.run_subaction_grasp \
  --rgb path/to/image.png \
  --instruction "pick up the bottle" \
  --output-dir outputs/demo_run
```

Optional depth image:

```bash
python -m prosthetic_grasp.apps.run_subaction_grasp \
  --rgb path/to/image.png \
  --depth path/to/depth.png \
  --instruction "pick up the bottle" \
  --output-dir outputs/demo_run
```

## Current Placeholders

The following modules are intentionally simple and need real models later:

- `perception.prosthesis_segmentor`
- `generation.clean_inpainter`
- `generation.flux_fill_client`
- `extraction.hand_proxy_extractor`
- `control.hand_executor`

The current scaffold exists to make later integration straightforward:

- replace the placeholder implementation
- keep the input/output schema unchanged
- preserve the pipeline contract

## Recommended Integration Order

1. Replace prosthesis segmentation with the custom lightweight binary model.
2. Replace clean inpainting with LaMa.
3. Connect FLUX Fill API or a self-hosted image-editing backend.
4. Replace hand-prior extraction with MediaPipe Hand Landmarker or HaMeR-backed logic.
5. Replace the executor stub with the real prosthesis hardware interface.
