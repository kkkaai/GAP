# Colab Pipeline Models Summary

本文档总结 `/Users/bigstepper/Downloads/Untitled10 (1).ipynb` 中实际使用的模型、checkpoint、关键依赖与版本信息。  
目标是为后续复现实验、迁移到实验室 PC、以及改用 API 调用时提供统一参考。


## 1. Pipeline 总览

当前 Colab pipeline 为：

1. `Grounding DINO` 检测手与相关目标
2. `SAM 2` 根据检测框分割手部 mask
3. 将手部 mask 转换为 `lollipop mask`
4. 使用 `SDXL Inpainting` 先去掉手部区域
5. 使用 `FLUX.1-Fill-dev` 在 lollipop ROI 中做生成/补全
6. 使用 `HaMeR` 从生成图像中提取 `MANO`


## 2. 各步骤模型汇总

| 步骤 | 任务 | 模型 / 方法 | 备注 |
| --- | --- | --- | --- |
| Step 1 | 文本驱动目标检测 | `Grounding DINO SwinT OGC` | 用于根据 prompt 找到 hand / coffee cup 等框 |
| Step 1 | 框引导分割 | `SAM 2.1 Hiera Large` | 根据 Grounding DINO 框输出 mask |
| Step 2 | 手 mask 转 lollipop | 自定义 PCA + 几何规则 | 思路参考 Affordance Diffusion，代码为 notebook 内重写 |
| Step 3 | 去手 inpaint | `SDXL Inpainting` | 使用 diffusers 的 inpainting pipeline |
| Step 4 | ROI 区域生成/编辑 | `FLUX.1-Fill-dev` | 在 lollipop mask 区域中生成语义手操作图像 |
| Step 5 | MANO 提取 | `HaMeR` | 输出 hand mesh / MANO 参数 |


## 3. 各步骤详细配置

### 3.1 Grounding DINO

- 作用：
  - 从图像中根据文本 prompt 检测手和杯子
- Notebook 中的 prompt：
  - `TEXT_PROMPT = "hand. coffee cup."`
- 关键阈值：
  - `BOX_THRESHOLD = 0.25`
  - `TEXT_THRESHOLD = 0.25`
- 配置文件：
  - `/content/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py`
- checkpoint：
  - `/content/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth`

### 3.2 SAM 2

- 作用：
  - 将 Grounding DINO 的检测框转换为精细 mask
- 模型配置：
  - `configs/sam2.1/sam2.1_hiera_l.yaml`
- checkpoint：
  - `/content/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt`
- 推理类：
  - `SAM2ImagePredictor`

### 3.3 Lollipop Mask

- 作用：
  - 将分割出的手部 mask 转换为 Affordance Diffusion 风格的手代理表示
- 方法：
  - PCA 提取主轴
  - 根据“靠近图像边缘的一端更可能是手臂/手腕”来确定 wrist side
  - 渲染成掌心圆 + 向图像边缘延伸的前臂条带
- 依赖：
  - `numpy`
  - `opencv-python`
  - `matplotlib`

### 3.4 Inpainting

- 作用：
  - 在 lollipop 区域内先移除原始人手，为后续 FLUX Fill 提供干净底图
- 模型：
  - `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`
- pipeline：
  - `AutoPipelineForInpainting`
- 关键参数：
  - `guidance_scale = 8.0`
  - `strength = 0.99`
  - `num_inference_steps = 30`

### 3.5 FLUX Fill

- 作用：
  - 在 ROI patch 的 lollipop 区域内做语义生成 / 补全
- 模型：
  - `black-forest-labs/FLUX.1-Fill-dev`
- pipeline：
  - `FluxFillPipeline`
- 关键参数：
  - `guidance_scale = 30.0`
  - `num_inference_steps = 50`
- 当前 notebook 中的 prompt 偏“背景补全”：
  - `A highly realistic background behind the object, no human hand, seamless continuation of the surface, extremely detailed, 8k`
- 如果后续要生成“理想交互人手”，应替换成更明确的 task-aware prompt

### 3.6 HaMeR

- 作用：
  - 从生成结果中提取 MANO / hand mesh
- 仓库：
  - `https://github.com/geopavlakos/hamer.git`
- 安装方式：
  - `pip install -e .[all]`
  - `pip install -v -e third-party/ViTPose`
- 额外资源：
  - `MANO_RIGHT.pkl`
- notebook 中的当前执行命令：
  ```bash
  python demo.py \
  --img_folder example_data --out_folder demo_out \
  --batch_size=48 --side_view --save_mesh
  ```
- 注意：
  - 当前 HaMeR demo 仍然是官方示例路径，尚未改成直接处理前面生成的 `flux_full` 图像


## 4. 关键依赖与版本

以下版本信息来自 notebook 中的显式安装命令。  
如果某个包未显式固定版本，则标记为“未显式固定”。

| 包 | 版本 / 约束 | 来源 |
| --- | --- | --- |
| `Grounded-SAM-2` | GitHub latest at clone time | `git clone https://github.com/IDEA-Research/Grounded-SAM-2.git` |
| `grounding_dino` | 仓库内源码安装 | `pip install -e grounding_dino` |
| `transformers` | `<=4.38.2` | notebook 显式安装 |
| `diffusers` | `==0.27.2`（前半段） | notebook 显式安装 |
| `huggingface_hub` | `<=0.25.2` | notebook 显式安装 |
| `peft` | `<=0.10.0`（前半段） | notebook 显式安装 |
| `supervision` | 未显式固定 | notebook 显式安装 |
| `addict` | 未显式固定 | notebook 显式安装 |
| `yapf` | 未显式固定 | notebook 显式安装 |
| `torch` | 未显式固定 | notebook 中直接 `pip install torch` |
| `torchvision` | 未显式固定 | 通过环境已有或与 torch 配套 |
| `opencv-python` / `cv2` | 未显式固定 | notebook 使用但未显式安装 |
| `numpy` | 未显式固定 | notebook 使用但未显式安装 |
| `matplotlib` | 未显式固定 | notebook 使用但未显式安装 |
| `accelerate` | 前半段通过 diffusers 依赖，后半段 `pip install -U` 升级 | notebook 显式升级 |
| `diffusers`（后半段） | 被 `pip install -U diffusers transformers accelerate peft` 升级到最新兼容版 | notebook 后半段显式升级 |
| `transformers`（后半段） | 被升级到最新兼容版 | notebook 后半段显式升级 |
| `peft`（后半段） | 被升级到最新兼容版 | notebook 后半段显式升级 |
| `HaMeR` | GitHub latest at clone time | `git clone --recursive https://github.com/geopavlakos/hamer.git` |
| `ViTPose` | HaMeR third-party 源码安装 | `pip install -v -e third-party/ViTPose` |


## 5. 版本一致性注意事项

当前 notebook 有一个重要问题：

- 前半段为了 `Grounded SAM 2`，显式安装了：
  - `transformers<=4.38.2`
  - `diffusers==0.27.2`
  - `huggingface_hub<=0.25.2`
  - `peft<=0.10.0`
- 后半段为了 `FLUX Fill`，又执行了：
  - `pip install -U diffusers transformers accelerate peft`

这意味着：

- notebook 前后环境实际上**不再完全一致**
- 后半段升级依赖后，理论上可能影响前半段 `Grounded SAM 2 / Grounding DINO` 的兼容性
- 当前 notebook 通过“重启前保存 mask / 图片”规避了这个问题

因此当前 notebook 的真实运行策略其实是：

### 阶段 A：Grounded SAM 2 / mask / lollipop / SDXL inpaint

- 使用偏旧且更适合 `Grounded SAM 2` 的依赖组合

### 阶段 B：重启会话后加载 FLUX Fill

- 升级 `diffusers / transformers / accelerate / peft`
- 使用更新环境运行 `FLUX.1-Fill-dev`

### 阶段 C：HaMeR

- 再进入新的依赖链

这说明当前 Colab workflow 仍然是**多阶段环境切换**，还不是完全统一环境。


## 6. 当前 notebook 中间产物

当前 notebook 明确保存了这些中间结果：

- `/content/image_source_saved.png`
- `/content/lollipop_mask.npy`
- Grounded SAM 2 导出的若干 `mask_*.png`

建议后续统一保存为：

- `input_rgb.png`
- `hand_mask.png`
- `hand_mask_overlay.png`
- `lollipop_mask.png`
- `roi_crop.png`
- `inpaint_crop.png`
- `flux_crop.png`
- `flux_full.png`
- `hamer_input.png`
- `summary.json`


## 7. 推荐后续整理方向

### 7.1 如果继续留在 Colab

建议拆成三个 notebook：

1. `grounded_sam2_mask_lollipop_inpaint.ipynb`
2. `flux_fill_generation.ipynb`
3. `hamer_mano_extraction.ipynb`

原因：

- 避免依赖升级互相污染
- 每段更容易调试
- 便于分别迁移到本地或服务器

### 7.2 如果迁移到实验室 PC

建议的长期形态：

- 本地：
  - 机械手 / 人手分割
  - lollipop
  - 可选轻量 inpaint
  - HaMeR
- API：
  - `FLUX Fill`

这样会比在单机上同时维护所有重模型环境更稳。


## 8. 快速清单

### 当前实际使用的核心模型

- `Grounding DINO SwinT OGC`
- `SAM 2.1 Hiera Large`
- `SDXL Inpainting`
- `FLUX.1-Fill-dev`
- `HaMeR`

### 当前实际使用的核心 pipeline

- `Grounded SAM 2` 得到手 mask
- `PCA + geometry` 得到 lollipop
- `SDXL Inpaint` 去掉原始手
- `FLUX Fill` 做 ROI 生成
- `HaMeR` 提取 MANO

