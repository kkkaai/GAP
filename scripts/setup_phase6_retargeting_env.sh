#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-gap-phase6-retargeting}"

conda env list | awk '{print $1}' | grep -qx "$ENV_NAME" || conda env create -f environment.yml

conda run -n "$ENV_NAME" python -m pip install --no-build-isolation chumpy
conda run -n "$ENV_NAME" python -m pip install yourdfpy
conda run -n "$ENV_NAME" python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
conda run -n "$ENV_NAME" python -m pip install smplx
conda run -n "$ENV_NAME" python -m pip install opencv-python-headless==4.10.0.84 --no-deps
conda run -n "$ENV_NAME" python -m pip install --no-deps git+https://github.com/hassony2/manopth.git
conda run -n "$ENV_NAME" python - <<'PY'
from pathlib import Path
import manopth.manolayer

path = Path(manopth.manolayer.__file__)
text = path.read_text()
replacements = {
    "torch.Tensor(smpl_data['betas'].r)": "torch.Tensor(smpl_data['betas'].r.copy())",
    "torch.Tensor(smpl_data['shapedirs'].r)": "torch.Tensor(smpl_data['shapedirs'].r.copy())",
    "torch.Tensor(smpl_data['posedirs'].r)": "torch.Tensor(smpl_data['posedirs'].r.copy())",
    "torch.Tensor(smpl_data['v_template'].r)": "torch.Tensor(smpl_data['v_template'].r.copy())",
    "torch.Tensor(smpl_data['weights'].r)": "torch.Tensor(smpl_data['weights'].r.copy())",
}
for old, new in replacements.items():
    text = text.replace(old, new)
path.write_text(text)
PY

conda run -n "$ENV_NAME" python -c "import numpy, scipy, trimesh, plotly, yourdfpy, torch, smplx, cv2; print('gap phase6 retargeting env ok')"
