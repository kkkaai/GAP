#!/usr/bin/env bash
set -euo pipefail

# Environment for the Do-as-I-Do/MoGe pointmap stage used by GAP phase5.5.
# This installs only MoGe pointmap inference dependencies. GAP reuses the
# Do-as-I-Do hand-anchored pointmap alignment method, but does not need the
# full Fast-SAM3D object-mesh generation stack for this phase.

ENV_NAME="${1:-gap-dai-pointmap}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Run this from a shell where conda is initialized." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Updating existing conda env: $ENV_NAME"
  conda activate "$ENV_NAME"
else
  echo "Creating conda env: $ENV_NAME"
  conda create -y -n "$ENV_NAME" python=3.11
  conda activate "$ENV_NAME"
fi

python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
conda install -y -c conda-forge ffmpeg git

# The pointmap script calls MoGe directly. Do not install Fast-SAM3D's full
# requirements.txt here: it pulls large mesh-generation and Blender dependencies
# that are not needed for this stage.
python -m pip install \
  pillow numpy opencv-python scipy trimesh einops huggingface_hub safetensors timm \
  "git+https://github.com/microsoft/MoGe.git"

python -m pip uninstall -y notebook || true

echo
echo "Environment ready: $ENV_NAME"
echo "MoGe weights will be downloaded automatically from Hugging Face on first use:"
echo "  Ruicheng/moge-vitl"
