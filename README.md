# Installation

## 1. Create Conda Environment

```bash
conda create -n gap python=3.10 -y
conda activate gap
```

## 2. Install PyTorch

Install `torch` and `torchvision` matching your local CUDA version.

Example for CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 3. Install GAP FLUX Dependencies

From the GAP repository root:

```bash
pip install -e .[flux-stage]
```

If you prefer requirements directly:

```bash
pip install -r requirements/flux-stage.txt
```

## 4. Install HaMeR from the Official Repository

```bash
git clone --recursive https://github.com/geopavlakos/hamer.git
cd hamer
pip install -e .[all]
pip install -v -e third-party/ViTPose
```

## 5. Download HaMeR Assets

From the `hamer` repository root:

```bash
bash fetch_demo_data.sh
```

Then place:

```text
MANO_RIGHT.pkl
```

under:

```text
hamer/_DATA/data/mano/MANO_RIGHT.pkl
```
