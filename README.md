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

## 4. Install SegGPT for Phase 1

```bash
pip install -e .[seggpt]
```

`phase1_mask` can use SegGPT as a few-shot segmentation backend:

```toml
[phase1_mask]
mode = "seggpt"
model_id = "BAAI/seggpt-vit-large"
support_image_path = "data/support/prosthetic_hand_001.png"
support_mask_path = "data/support/prosthetic_hand_001_mask.png"
threshold = 0.5
device = "auto"
```

Model weights are downloaded automatically by Hugging Face Transformers on the first run. For offline machines, pre-cache `BAAI/seggpt-vit-large` in the Hugging Face cache before running the pipeline.

## 5. Install HaMeR from the Official Repository

```bash
git clone --recursive https://github.com/geopavlakos/hamer.git
cd hamer
pip install -e .[all]
pip install -v -e third-party/ViTPose
```

Official repository:

- [HaMeR](https://github.com/geopavlakos/hamer)

## 6. Download HaMeR Assets

From the `hamer` repository root:

```bash
bash fetch_demo_data.sh
wget https://github.com/JonathanLehner/Colab-collection/releases/download/MANO/mano_v1_2.zip
unzip mano_v1_2.zip
mv mano_v1_2/models/MANO_RIGHT.pkl _DATA/data/mano/
rm -r mano_v1_2
```

## Retargeting / Phase 6 Tools

The MANO-to-prosthetic retargeting experiments from `mano2prosthetic` are merged into this repository under:

- `tools/retarget/`: single-frame and batch retargeting baselines.
- `tools/vis/`: HOI4D/MANO/object visualization helpers.
- `tools/report/`: baseline report generation.
- `docs/retargeting_method_notes_cn.md`: method notes.
- `docs/retargeting_baselines_report.md`: three-baseline experiment report.

Install the retargeting dependencies with:

```bash
pip install -e .[retargeting]
```

Or create the conda environment:

```bash
conda env create -f environment.retargeting.yml
conda activate gap-phase6-retargeting
pip install -e .[retargeting]
```

The tools expect local HOI4D sample data under `extracted_dataset_sampled/` and MANO model files under `mano/mano/`.
