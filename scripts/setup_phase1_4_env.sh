#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-gap-phase1-4}"
ENV_FILE="${2:-environment.phase1-4.yml}"

conda env create -f "$ENV_FILE"
conda run -n "$ENV_NAME" python -m pip install -e ".[flux-stage,seggpt]"
conda run -n "$ENV_NAME" python - <<'PY'
import cv2
import diffusers
import numpy
import PIL
import requests
import torch
import transformers

from prosthetic_grasp.phases.phase1_mask import Phase1Mask, Phase1MaskConfig
from prosthetic_grasp.phases.phase2_lollipop import Phase2Lollipop, Phase2LollipopConfig
from prosthetic_grasp.phases.phase3_erase import Phase3Erase, Phase3EraseConfig
from prosthetic_grasp.phases.phase4_inpaint import Phase4Inpaint, Phase4InpaintConfig

print("gap phase1-4 env ok")
print("numpy:", numpy.__version__)
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("diffusers:", diffusers.__version__)
print("phase classes:", Phase1Mask.__name__, Phase2Lollipop.__name__, Phase3Erase.__name__, Phase4Inpaint.__name__)
PY
