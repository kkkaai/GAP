#!/usr/bin/env bash
set -euo pipefail

# Lightweight runtime deps for GAP phase5.5 hand-anchored pointmap alignment.
# This is separate from scripts/setup_dai_pointmap_env.sh, which installs the
# heavy MoGe/Fast-SAM3D pointmap generator.

ENV_NAME="${1:-gap-phase6-retargeting}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

python -m pip install --upgrade pip
python -m pip install numpy scipy opencv-python trimesh plotly rtree

python - <<'PY'
import cv2
import numpy
import plotly
import scipy
import trimesh
import rtree
print("phase5.5 Do-as-I-Do alignment deps OK")
PY

