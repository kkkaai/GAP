#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-gap-q-goal-cfm}"
ENV_FILE="${2:-environment.q_goal_cfm.yml}"

conda env create -f "$ENV_FILE"
conda run -n "$ENV_NAME" python -m pip install -e .
conda run -n "$ENV_NAME" python - <<'PY'
import torch
from transformers import SiglipModel, SiglipProcessor
import flow_matching

print("gap q_goal CFM env ok")
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("SigLIP classes:", SiglipModel.__name__, SiglipProcessor.__name__)
print("flow_matching:", getattr(flow_matching, "__version__", "installed"))
PY
