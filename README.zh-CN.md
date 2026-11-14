# RIND / G2-02

[完整研究链路报告](docs/RIND_RESEARCH_CHAIN.zh-CN.md) · [实验状态](docs/PROJECT_STATUS.zh-CN.md) · [English](README.md)

目标是在学生不知道全场障碍物几何的情况下，从局部响应反演驱动源位置，并通过同场景多视窗观测逐步减少不确定性。物理模型为正强度、二值可见性、无距离衰减的线性叠加。

## 当前主线

1. **Phase1定位**：单贡献响应与窗口位置 → 全场源位置概率。已有正式测试结果。
2. **Phase2分层**：原始多源响应与窗口位置 → 有效层数、无序二值贡献掩码。当前v6在学生内部自主减去单层贡献，只监督最终层数和掩码；新实验尚无最终测试结论。
3. **后续闭环**：学生分层 → Phase1逐层定位 → 多视窗融合与主动选窗。环境已有原型，完整多源定位收益仍需验证。

当前关注层数预测准确率与单层IoU≥0.90通过率，两项等权。完整响应是守恒约束，不是v6重建loss。教师剥离顺序、中间残差、强度和真源编号不进入学生前向。有效层数R不等于物理源数K；32为兼容已有复杂标签的执行上限。

方法、证据、负结果、指标口径与研究限制见[完整报告](docs/RIND_RESEARCH_CHAIN.zh-CN.md)。

## 目录职责

| 路径 | 内容 |
|---|---|
| `Phase1_Single_Source_Single_Intensity/` | 单贡献物理教师、模型、训练、评价与位置场浏览器 |
| `Phase2_Multi_Source/src/rind_phase2/` | 多源数据、几何遮挡、加权响应和候选工具 |
| `Phase2_Multi_Source/research/implicit_slots/` | 当前分层学生、标签缓存、训练、指标和旧权重对比 |
| `Phase2_Multi_Source/research/layered_residual/` | 逐层教师、共享候选渲染、历史定位实验与浏览器 |
| `Phase2_Multi_Source/research/active_observation/` | 新增像素预算、移动/扩窗环境及参考策略 |
| `Phase2_Multi_Source/dataset_update/` | 安装数据、随机强度派生、重复几何安全拼接 |
| 各阶段的 `configs/`、`tests/`、`docs/` | 版本化实验、回归测试、方法及运行文档 |
| `docs/` | 跨阶段研究报告、状态和紧凑证据 |
| `tools/` | 仓库检查、从正式源码构建服务器包 |
| `Literature_Review/`、`G2_02_Initial_Proposal.pdf` | 团队文献与原proposal |

Phase1的 `part_e/`、`student_baseline/` 仍是运行依赖。历史训练入口和配置用于复现旧权重；当前版本无需复制另一套源码目录。

## 环境

在G2-02根目录运行：

```bash
./setup.sh
export RIND_PYTHON="$PWD/.venv/bin/python"
```

安装器在用户目录准备Python 3.12和固定依赖，无需sudo/snap。默认CUDA 12.4；CPU安装可用 `RIND_TORCH_BACKEND=cpu ./setup.sh`。已有环境直接设置 `RIND_PYTHON=/absolute/path/to/python`。

原始数据需单独安装，见[多源数据说明](Phase2_Multi_Source/dataset_update/README.zh-CN.md)及[Phase1服务器说明](Phase1_Single_Source_Single_Intensity/docs/SERVER_100K_MULTISCALE.zh-CN.md)。已有完整分层缓存和旧v5最佳权重时，v6训练无需原始ZIP或重新渲染位置教师。

## 当前训练与评价

以下源目录是服务器上一轮大实验示例；必须包含完整 `mask_records.json`、`mask_cache/` 与v5 `models/seed_0/best.pt`。缺失时明确报错，不回退到小数据集。

```bash
unset RIND_MASKS_CONFIG RIND_DECOMPOSITION_CONFIG RIND_TMUX_SESSION
./run_implicit_masks_tmux.sh start all \
  --source-root outputs/masks_disentangled_v5_complex245746 \
  --expected-source-epoch 16 \
  --run-name masks_implicit_v6_best16 \
  --session rind_v6_best16 \
  --gpus auto --auto-batch
./run_implicit_masks_tmux.sh status --session rind_v6_best16
```

全部原训练记录及追加复杂窗保留，验证/测试名单不变；真实数量以缓存索引和日志为准。任务在tmux后台运行，可断开SSH/VPN。再次启动复用配置与epoch末的 `last.pt`，不承诺恢复到中断batch。

```bash
./run_implicit_masks_tmux.sh logs --session rind_v6_best16
./run_implicit_masks_tmux.sh attach --session rind_v6_best16
./run_implicit_masks_tmux.sh start evaluate --session rind_v6_best16 --gpus auto
```

最后一条只做正式评价。上述命令的新输出位于 `Phase2_Multi_Source/outputs/masks_implicit_v6_best16/`，保留旧权重，并检查继承的最佳轮次为16：

- `models/seed_0/best.pt`、`last.pt`：最佳模型与恢复断点。
- `training_summary.json`、`models/seed_0/training.json`：训练记录。
- `evaluation/seed_0/primary_metrics.json`：两个主要指标及分层结果。
- `evaluation/seed_0/reference_comparison.zh-CN.md`：相同名单上的旧最终best/新best对比；旧训练未完成时记录pending。
- `tmux_runs/`：持久日志、执行命令、退出码。

当前运行和加速细节见 [best16续训说明](Phase2_Multi_Source/docs/V6_BEST16_RUN.zh-CN.md)，完整协议见 [v6说明](Phase2_Multi_Source/docs/SEQUENTIAL_COUNT_SHAPE.zh-CN.md)。历史v1–v5、数据扩充、诊断、槽数搜索与动态观测入口集中在 [Phase2文档索引](Phase2_Multi_Source/docs/README.zh-CN.md)。旧定位浏览器仍可用于对应旧产物，当前纯分层结果不自动获得完整位置场可视化。

## 检查与服务器包

```bash
"$RIND_PYTHON" tools/check_repo.py
"$RIND_PYTHON" tools/package_update.py
```

默认生成 `dist/RIND_Phase2_Update.tar.gz`；解压前缀固定为 `RIND_Phase2_Training_Minimal`。可以用 `--output /path/to/RIND_Phase2_Update.tar.gz` 指定同名目标。源码包不含原始数据、缓存、虚拟环境、权重或逐窗结果。

本地的这些实验产物仍保留，由 `.gitignore` 排除。GitHub保存源码、配置、测试、方法与紧凑证据。维护规则见 [仓库整理约定](docs/REPOSITORY_MAINTENANCE.zh-CN.md)。

## 单源RL动态选窗（独立实验）

冻结已有Phase1定位器，训练选路策略。新版从一个16×16块开始，每次在整个已观测区域的四邻接边缘添加一个新块；独立查询Phase1，保留原始小窗概率的逐点最小值，再归一化。新增全量配置，继承全部80,000/10,000/10,000场景划分，测试排除上轮审阅的128个场景；以实际解析数量为准。每个训练场景至少完成一条轨迹后才允许结束。原64块乘积融合实验保留。

```bash
./run_single_source_rl_tmux.sh start all --gpus auto \
  --run-name single_source_rl_frontier16_minimum_full \
  --config configs/active_rl_frontier16_minimum_full.json \
  --phase1-run outputs/experiments/phase1_100k_multiscale_h20_memmap \
  --data-root data/raw/phase1-single-source-100k
./run_single_source_rl_tmux.sh status --run-name single_source_rl_frontier16_minimum_full
```

运行、融合定义、参数与逐步HTML可视化见 [16块最小值版指南](Phase1_Single_Source_Single_Intensity/docs/FRONTIER16_MINIMUM_RL.zh-CN.md)。服务器原v1已完成128场景测试：观测扩展有效，但PPO未证明优于随机或直接扩窗；详见 [审核报告](Phase1_Single_Source_Single_Intensity/docs/results/rl_single_source_v1_review.zh-CN.md)。v2完成本地真实数据流程验证，100k模型的选路收益和H20效率待服务器评价。真源只用于训练奖励及离线评价，不进入策略。
