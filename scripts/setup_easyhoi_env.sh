#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-easyhoi}"
PYTHON_VERSION="${PYTHON_VERSION:-3.9}"
CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0+PTX}"

conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"

CONDA_BASE="$(conda info --base)"
ENV_PREFIX="${CONDA_PREFIX:-${CONDA_BASE}/envs/${ENV_NAME}}"
PYTHON="${ENV_PREFIX}/bin/python"

conda install -y -n "${ENV_NAME}" --override-channels -c nvidia -c defaults \
  cuda-nvcc=12.8 cuda-cudart-dev=12.8 cuda-cccl=12.8

"${PYTHON}" -m pip install --upgrade pip setuptools wheel
"${PYTHON}" -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0+cu128 torchvision==0.23.0+cu128

conda install -y -n "${ENV_NAME}" -c conda-forge fvcore iopath

"${PYTHON}" -m pip install \
  "numpy==1.23.5" "opencv-python<5" \
  hydra-core hydra-optuna-sweeper hydra-colorlog rootutils rich pre-commit pytest \
  lightning torchmetrics trimesh pillow mesh-to-sdf pymeshfix scipy scikit-learn \
  scikit-image pandas pyrender geomloss IPython deprecation \
  chumpy torchgeometry manotorch ninja

export CUDA_HOME="${ENV_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${CUDA_ARCH_LIST}"
export CPATH="${CUDA_HOME}/targets/x86_64-linux/include:${CPATH:-}"
export LIBRARY_PATH="${CUDA_HOME}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/targets/x86_64-linux/lib:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

"${PYTHON}" -m pip install --no-build-isolation --no-cache-dir \
  "git+https://github.com/NVlabs/nvdiffrast.git"

if [ ! -d /tmp/pytorch3d-src ]; then
  git -c http.version=HTTP/1.1 clone --depth 1 \
    https://github.com/facebookresearch/pytorch3d.git /tmp/pytorch3d-src
fi
"${PYTHON}" -m pip install --no-build-isolation --no-cache-dir /tmp/pytorch3d-src

"${PYTHON}" -m pip install --no-build-isolation --no-cache-dir \
  "git+https://github.com/otaheri/chamfer_distance"

"${PYTHON}" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    print("cuda_sum", torch.ones(4, device="cuda").sum().item())
PY
