# FLUX Fill 与 HaMeR 本地 Conda 环境总结

本文档只总结当前项目后半段需要的两个模型环境：

- `FLUX.1-Fill-dev`
- `HaMeR`

不讨论前面的分割模型环境。

---

## 1. 总体建议

虽然从理论上可以尝试把 `FLUX Fill` 和 `HaMeR` 塞进同一个本地 conda 环境，但**不建议这样做**。

原因：

- `FLUX Fill` 依赖较新的 `diffusers / transformers / accelerate`
- `HaMeR` 安装会引入自己的完整 3D 手部恢复依赖链
- 两者都比较重，后续调试时更容易遇到版本冲突或显存管理问题

因此建议采用：

- `env-gap-flux`：负责 `Step 3-4`
- `env-gap-hamer`：负责 `Step 5`

这样更利于后续迁移到实验室 PC，并且便于单独维护。

---

## 2. 当前技术路线中的模型职责

### 2.1 FLUX Fill

负责：

- 接收修补后的 ROI 图像
- 接收 lollipop mask
- 接收任务语义 prompt
- 在局部区域生成人手操作图像

当前 notebook 中对应：

- `Step 4`
- 使用 `FLUX.1-Fill-dev`

### 2.2 HaMeR

负责：

- 接收 `Step 4` 生成的人手操作图像
- 恢复 3D MANO 手模型
- 输出 mesh / MANO 参数 / 手部关键点

当前 notebook 中对应：

- `Step 5`
- 使用 `HaMeR`

---

## 3. FLUX Fill 环境建议

### 3.1 角色

该环境主要负责：

- ROI 图像加载与裁剪
- inpaint / fill 输入准备
- `FLUX.1-Fill-dev` 推理
- 图像保存与结果缓存

### 3.2 推荐 Python 版本

- `Python 3.10`

原因：

- 与大多数 PyTorch / diffusers 生态兼容性最好
- 后续如果需要和 HaMeR 共用部分脚本，也更稳

### 3.3 建议核心依赖

建议至少包含：

- `pytorch`
- `torchvision`
- `python-dotenv`
- `diffusers`
- `transformers`
- `accelerate`
- `peft`
- `huggingface_hub`
- `safetensors`
- `Pillow`
- `numpy`
- `opencv-python`

### 3.4 notebook 中已经体现出的依赖关系

从当前 `notebooks/GAP_Colab.ipynb` 看，`FLUX Fill` 相关步骤实际依赖：

- 先前为 inpaint 安装过：
  - `diffusers==0.27.2`
  - `huggingface_hub<=0.25.2`
  - `peft<=0.10.0`
- 到 `FLUX Fill` 阶段又执行了升级：
  - `pip install -U diffusers transformers accelerate peft`

这说明一个事实：

- 当前 Colab notebook 中，`FLUX Fill` 阶段依赖的是**更新后的 diffusers 栈**
- 因此本地环境不要照搬 `0.27.2` 作为最终 FLUX 环境

### 3.5 本地环境建议版本策略

推荐策略：

- `torch`、`torchvision` 与你实验室 GPU CUDA 版本匹配
- `diffusers`、`transformers`、`accelerate` 采用相互兼容的较新版本

更重要的是：

- 以 `FluxFillPipeline` 能正常导入并推理为准
- 不建议把旧版 `SDXL inpaint` 和新版 `FLUX Fill` 强行绑死在一个完全固定的超旧版本集合上

### 3.5 密钥管理建议

建议在项目根目录使用 `.env` 保存本地密钥，而不是手工执行登录命令。

例如：

```dotenv
HF_TOKEN=your_huggingface_token_here
BFL_API_KEY=your_bfl_api_key_here
```

推荐读取方式：

```python
from dotenv import load_dotenv
from huggingface_hub import login
import os

load_dotenv()
hf_token = os.getenv("HF_TOKEN", "")
if hf_token:
    login(token=hf_token)
```

### 3.6 推荐环境用途

`env-gap-flux` 推荐负责：

- `Step 3`：背景修补
- `Step 4`：语义人手生成

如果后续你把 `Step 3` 的 inpaint 也换成 `FLUX Fill`，这个环境仍然适用。

---

## 4. HaMeR 环境建议

### 4.1 角色

该环境主要负责：

- 加载 `HaMeR`
- 下载并加载 checkpoint
- 准备 `MANO_RIGHT.pkl`
- 对输入图像运行 3D 手恢复
- 输出 mesh / MANO / 关键点

### 4.2 推荐 Python 版本

- `Python 3.10`

原因：

- 与 `HaMeR` 官方仓库默认安装路径更一致

### 4.3 notebook 中实际使用的安装步骤

当前 notebook 中 `HaMeR` 部分实际做了：

- `git clone --recursive https://github.com/geopavlakos/hamer.git`
- `pip install torch`
- `pip install -e .[all]`
- `pip install -v -e third-party/ViTPose`
- 下载 `MANO_RIGHT.pkl`
- 运行 `demo.py`

因此，本地环境至少要支持：

- `torch`
- `HaMeR` 主包
- `ViTPose`
- `MANO` 资产文件

### 4.4 需要单独关注的点

#### MANO 资产

除了 pip 依赖，你还需要准备：

- `MANO_RIGHT.pkl`

这是运行 `HaMeR` 的必要模型资产，不属于普通 Python 包。

#### 显存与进程管理

在 Colab 里你已经遇到过：

- `torch.OutOfMemoryError`

这说明：

- `FLUX Fill` 和 `HaMeR` 不适合在同一进程里连续常驻大模型后再直接跑
- 本地运行时也建议：
  - 单独进程
  - 单独环境
  - 或者至少在执行前释放前一阶段 GPU 占用

### 4.5 推荐环境用途

`env-gap-hamer` 推荐只负责：

- 接收 `Step 4` 输出图像
- 运行 3D MANO 恢复
- 导出 MANO 参数与关键点结果

不要把 `FLUX Fill` 也放在这个环境里常规运行。

---

## 5. 推荐的本地拆分方式

### 5.1 方案 A：推荐方案

拆成两个 conda 环境：

- `env-gap-flux`
- `env-gap-hamer`

优点：

- 版本冲突少
- GPU 内存管理更清晰
- 后续调试时定位问题容易

缺点：

- 需要在两个环境之间做文件传递

### 5.2 方案 B：不推荐但可尝试

单一 conda 环境：

- 同时安装 `FLUX Fill` 和 `HaMeR`

风险：

- `diffusers / transformers` 与 `HaMeR` 的依赖链更容易互相影响
- 后续升级某一方时，另一方容易被破坏

只有在你确认实验室 PC 环境非常稳定，且确实需要一键式运行时，再考虑此方案。

---

## 6. 本地运行阶段建议

建议将后半段拆成两个离线阶段：

### 阶段一：生成阶段

环境：

- `env-gap-flux`

输入：

- 修补后的 ROI 图像
- lollipop mask
- 任务 prompt

输出：

- 人手操作图像
- 日志
- prompt
- 采样参数

### 阶段二：MANO 恢复阶段

环境：

- `env-gap-hamer`

输入：

- 阶段一输出的人手操作图像

输出：

- MANO 参数
- 手 mesh
- 指尖关键点

这两阶段之间只通过文件交互即可。

---

## 7. 与当前 notebook 的对应关系

当前 `notebooks/GAP_Colab.ipynb` 已经可以理解成：

- `Step 1-3`：前处理
- `Step 4`：`FLUX Fill`
- `Step 5`：`HaMeR`
- `Step 6`：尚未实现

因此本地环境封装时，建议直接对应到：

- `Step 4` 一个环境
- `Step 5` 一个环境

而不是试图把整个 notebook 原样搬到一套本地环境中。

---

## 8. 后续建议

后续如果你要开始在实验室 PC 上真正封装环境，建议按这个顺序做：

1. 先单独封装 `env-gap-flux`
2. 验证 `FLUX.1-Fill-dev` 在本地可推理
3. 再单独封装 `env-gap-hamer`
4. 验证 `demo.py` 能对单张图输出 MANO 结果
5. 最后再写一个脚本，把两阶段串起来

这样更稳，也更符合你当前研究推进节奏。
