# Undo

本文档列出当前项目中**还没有完成**、但后续必须推进的工作。  
目标不是记录所有想法，而是把剩余任务拆成可执行清单。


## P0 当前最优先

这些任务决定项目能否从“骨架”进入“真实可用”阶段。

### 1. 建立最小测试体系

- [ ] 创建 `tests/` 目录
- [ ] 为 `intent_parser.py` 编写单元测试
- [ ] 为 `lollipop_fitter.py` 编写单元测试
- [ ] 为 `interaction_roi.py` 编写单元测试
- [ ] 为 `rule_initializer.py` 编写单元测试
- [ ] 为 `local_optimizer.py` 编写单元测试
- [ ] 编写一个最小 CLI 集成测试，验证主流程在 stub 模式下可以跑通

完成标准：

- `pytest` 能运行
- 至少有一组单元测试和一组集成测试
- 修改核心逻辑后能快速回归


### 2. 建立调试数据与目录结构

- [ ] 创建调试数据目录，例如 `data/debug/`
- [ ] 准备一批固定的 RGB 输入样例
- [ ] 如有需要，准备对应深度图
- [ ] 定义统一的输出 artifacts 目录格式
- [ ] 为调试样例建立固定运行脚本或命令

完成标准：

- 有一个固定的小规模 debug set
- 每次运行都能稳定产出中间结果
- 不同版本结果可直接对比


### 3. 替换假肢分割占位实现

当前占位文件：

- [src/prosthetic_grasp/perception/prosthesis_segmentor.py](/Users/bigstepper/VscodeProjects/GAP/src/prosthetic_grasp/perception/prosthesis_segmentor.py)

待完成任务：

- [ ] 设计假肢分割数据目录结构
- [ ] 采集第一批第一视角假肢 RGB 数据
- [ ] 完成 mask 标注方案
- [ ] 选择第一版分割模型
- [ ] 编写训练脚本
- [ ] 编写推理封装
- [ ] 将推理结果接入当前主流程
- [ ] 保存推理 mask 可视化结果

完成标准：

- 不再使用启发式右下角 mask
- 对固定测试图能稳定输出合理的假肢 mask
- lollipop 拟合结果基本合理


### 4. 让 lollipop 拟合进入可视化调试状态

相关文件：

- [src/prosthetic_grasp/perception/lollipop_fitter.py](/Users/bigstepper/VscodeProjects/GAP/src/prosthetic_grasp/perception/lollipop_fitter.py)

待完成任务：

- [ ] 增加 lollipop 参数调试输出
- [ ] 增加 lollipop 覆盖在原图上的可视化
- [ ] 评估不同假肢姿态下的拟合稳定性
- [ ] 调整 palm 半径和 arm strip 宽度规则

完成标准：

- 能从 artifact 一眼看出 lollipop 是否贴合假肢
- 拟合不再只是“代码能跑”，而是“几何上可用”


### 5. 固定最小 demo 命令

- [ ] 明确一个统一 demo 命令
- [ ] 明确一个固定输入样例
- [ ] 明确一个固定输出目录
- [ ] 在 README 中补充 demo 运行说明

完成标准：

- 任何人进入仓库后都知道如何跑最小演示


## P1 下一阶段必须完成

这些任务会把当前系统从“只有占位图像流”推进到“真实生图管线”。

### 6. 替换假肢去除占位实现

当前占位文件：

- [src/prosthetic_grasp/generation/clean_inpainter.py](/Users/bigstepper/VscodeProjects/GAP/src/prosthetic_grasp/generation/clean_inpainter.py)

待完成任务：

- [ ] 选定 LaMa 接入方式
- [ ] 编写本地推理封装
- [ ] 加入 mask 外保持策略
- [ ] 保存 clean inpaint 输出
- [ ] 对比占位实现与真实 inpaint 效果

完成标准：

- 假肢区域能被清理
- mask 外区域基本保持不变
- 输出图可供下一步生成使用


### 7. 替换抓取图生成占位实现

当前占位文件：

- [src/prosthetic_grasp/generation/flux_fill_client.py](/Users/bigstepper/VscodeProjects/GAP/src/prosthetic_grasp/generation/flux_fill_client.py)

待完成任务：

- [ ] 接入 `FLUX.1 Fill [pro]` API
- [ ] 定义 API key 读取方式
- [ ] 支持 mask + prompt + image 输入
- [ ] 支持多候选生成
- [ ] 支持错误处理和超时处理
- [ ] 保存原始返回结果和最终候选图

完成标准：

- 不再生成假的“肤色圆形手掌”
- 能真实返回候选抓取图
- 多候选逻辑能稳定工作


### 8. 完善 prompt 与候选图输出

相关文件：

- [src/prosthetic_grasp/generation/prompt_builder.py](/Users/bigstepper/VscodeProjects/GAP/src/prosthetic_grasp/generation/prompt_builder.py)
- [src/prosthetic_grasp/generation/candidate_ranker.py](/Users/bigstepper/VscodeProjects/GAP/src/prosthetic_grasp/generation/candidate_ranker.py)

待完成任务：

- [ ] 整理 prompt 模板版本
- [ ] 针对不同任务定义 prompt 变体
- [ ] 完善候选打分规则
- [ ] 保存候选排序分数
- [ ] 记录失败案例

完成标准：

- prompt 不再是单一模板
- 候选图排序不再只是非常粗糙的像素差评分


## P2 决策与控制闭环

这些任务对应“从生成图得到假肢动作”。

### 9. 替换抓取先验提取占位实现

当前占位文件：

- [src/prosthetic_grasp/extraction/hand_proxy_extractor.py](/Users/bigstepper/VscodeProjects/GAP/src/prosthetic_grasp/extraction/hand_proxy_extractor.py)

待完成任务：

- [ ] 选定 `MediaPipe Hand Landmarker` 接入方案
- [ ] 评估其在生成图上的表现
- [ ] 设计从关键点到 `HumanPrior2D` 的转换逻辑
- [ ] 定义接近方向估计方法
- [ ] 定义 contact patch 提取方法
- [ ] 定义 `grasp_family_hint` 提取方法

完成标准：

- 不再直接从 ROI 人工构造 prior
- 能从真实生成图中提取出稳定的抓取先验


### 10. 细化规则型重定向

相关文件：

- [src/prosthetic_grasp/retarget/rule_initializer.py](/Users/bigstepper/VscodeProjects/GAP/src/prosthetic_grasp/retarget/rule_initializer.py)
- [src/prosthetic_grasp/retarget/local_optimizer.py](/Users/bigstepper/VscodeProjects/GAP/src/prosthetic_grasp/retarget/local_optimizer.py)

待完成任务：

- [ ] 明确 `pinch / tripod / power_wrap / lateral` 判定规则
- [ ] 补充 `thumb_mode` 规则
- [ ] 补充 `wrist_rotation` 规则
- [ ] 补充 `aperture` 估计逻辑
- [ ] 补充 `force_level` 估计逻辑
- [ ] 定义局部优化搜索空间
- [ ] 定义局部优化目标函数

完成标准：

- `ProstheticAction` 的输出不再只是非常粗糙的启发式
- 能根据不同任务输出可区分的动作参数


### 11. 替换执行器占位实现

当前占位文件：

- [src/prosthetic_grasp/control/hand_executor.py](/Users/bigstepper/VscodeProjects/GAP/src/prosthetic_grasp/control/hand_executor.py)

待完成任务：

- [ ] 定义执行器接口协议
- [ ] 先实现“记录指令到日志”的假执行器
- [ ] 再实现真实硬件适配层
- [ ] 增加执行结果与 telemetry 记录
- [ ] 定义失败码和错误信息

完成标准：

- 不再只返回 `simulated_success`
- 至少具备真实控制接口的统一抽象


## P3 工程化与质量保障

这些任务保证仓库能持续维护。

### 12. 建立 GitHub Actions 基础 CI

- [ ] 创建 `.github/workflows/ci.yml`
- [ ] 在 CI 中安装包
- [ ] 在 CI 中运行 `compileall`
- [ ] 在 CI 中运行 CLI `--help`
- [ ] 在 CI 中运行最小测试集

完成标准：

- push / PR 时自动做基础健康检查


### 13. 整理配置系统

相关文件：

- [config/default.toml](/Users/bigstepper/VscodeProjects/GAP/config/default.toml)

待完成任务：

- [ ] 把更多硬编码参数迁移到配置文件
- [ ] 明确模型选择配置
- [ ] 明确 API 配置
- [ ] 明确输出目录配置
- [ ] 明确调试开关配置

完成标准：

- 主流程代码中尽量少写死参数


### 14. 统一日志与产物保存

- [ ] 保存输入图
- [ ] 保存假肢 mask
- [ ] 保存 lollipop mask
- [ ] 保存 clean inpaint 图
- [ ] 保存候选图
- [ ] 保存 prior 可视化
- [ ] 保存最终 action
- [ ] 保存执行结果

完成标准：

- 每次运行都有完整 artifact
- 方便调试和论文图整理


### 15. 整理文档

- [ ] 更新 README 中的项目结构说明
- [ ] 更新 README 中的运行方式
- [ ] 增加模型接入说明
- [ ] 增加数据准备说明
- [ ] 增加测试运行说明

完成标准：

- 仓库对后续开发者是可读的


## P4 可选增强

这些任务不是当前最短闭环必需，但后续很可能需要。

### 16. 可选目标物体检测/分割模块

- [ ] 设计目标检测/分割接口
- [ ] 接入开放词汇原型模型
- [ ] 定义何时启用该模块
- [ ] 用于杂乱/多目标场景增强

完成标准：

- 该模块作为可选增强存在，不破坏最小 v1 路径


### 17. 更强的候选验证模块

- [ ] 设计更强的候选打分策略
- [ ] 评估是否需要额外视觉模型或 VLM
- [ ] 与启发式 scorer 做对比

完成标准：

- 候选图排序在真实数据上明显优于当前简单策略


### 18. 更丰富的手部重建

- [ ] 评估 `HaMeR`
- [ ] 比较 `MediaPipe` 与 `HaMeR`
- [ ] 决定第一篇工作是否需要 3D 手部信息

完成标准：

- 明确最终采用 2D prior 还是更丰富的 3D prior


## 当前明确的占位模块

以下模块目前仍是占位实现，必须后续替换：

- [ ] [src/prosthetic_grasp/perception/prosthesis_segmentor.py](/Users/bigstepper/VscodeProjects/GAP/src/prosthetic_grasp/perception/prosthesis_segmentor.py)
- [ ] [src/prosthetic_grasp/generation/clean_inpainter.py](/Users/bigstepper/VscodeProjects/GAP/src/prosthetic_grasp/generation/clean_inpainter.py)
- [ ] [src/prosthetic_grasp/generation/flux_fill_client.py](/Users/bigstepper/VscodeProjects/GAP/src/prosthetic_grasp/generation/flux_fill_client.py)
- [ ] [src/prosthetic_grasp/extraction/hand_proxy_extractor.py](/Users/bigstepper/VscodeProjects/GAP/src/prosthetic_grasp/extraction/hand_proxy_extractor.py)
- [ ] [src/prosthetic_grasp/control/hand_executor.py](/Users/bigstepper/VscodeProjects/GAP/src/prosthetic_grasp/control/hand_executor.py)


## 当前建议的推进顺序

建议严格按这个顺序做，避免同时开太多分支：

1. 建测试
2. 建 debug 数据
3. 替换假肢分割
4. 打通 lollipop 可视化
5. 接入 LaMa
6. 接入 FLUX Fill API
7. 接入抓取先验提取
8. 细化规则型重定向
9. 接入执行器
10. 加 CI


## 当前项目里程碑

### Milestone 1：视觉生成闭环

- [ ] 输入 RGB 图
- [ ] 输出假肢 mask
- [ ] 输出 lollipop
- [ ] 输出 clean 图
- [ ] 输出多张真实抓取候选图

### Milestone 2：控制决策闭环

- [ ] 从候选图输出 `HumanPrior2D`
- [ ] 从 `HumanPrior2D` 输出 `ProstheticAction`

### Milestone 3：执行闭环

- [ ] 将 `ProstheticAction` 发送到真实或仿真的执行接口

