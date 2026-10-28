# RIND / G2-02

[English](README.md) · [项目进展与结果](docs/PROJECT_STATUS.zh-CN.md) · [Phase1](Phase1_Single_Source_Single_Intensity/README.zh-CN.md) · [Phase2](Phase2_Multi_Source/README.zh-CN.md)

目标是在学生不知道全场障碍物几何的情况下，根据局部响应推断驱动源位置，并研究如何用多个视窗逐步缩小不确定性。物理模型是无距离衰减的正强度、二值可见性线性叠加。

当前主线分为两部分：Phase1学习单贡献的全场位置概率；Phase2先学习把原始多源响应解耦成若干有效贡献层，再复用Phase1定位。当前服务器已完成2万窗口的解耦训练；10万窗口扩展流程已交付，完整多源定位与多视窗闭环仍需验证。

## 目录职责

```text
G2-02/
├── README.md / README.zh-CN.md           项目入口
├── setup.sh / requirements.txt          用户目录Python环境和固定训练依赖
├── run_masks*_tmux.sh                   当前分层解耦训练、扩展与后台管理
├── run_decomposition*.sh                保留完整位置教师/定位训练入口
├── run_phase1_100k.sh                   单贡献定位训练入口
├── Phase1_Single_Source_Single_Intensity/
│   ├── src/rind_phase1/                 数据、物理教师、能量模型、训练和浏览器
│   ├── part_e/                         正式评价与自适应对照，仍被主流程引用
│   ├── student_baseline/               旧模型兼容适配，保留复现依赖
│   ├── configs/ scripts/ tests/ docs/   配置、命令、回归检查与方法文档
│   └── research/                       边界统计与几何反例研究
├── Phase2_Multi_Source/
│   ├── src/rind_phase2/                 多源数据、可见性、教师与几何邻接
│   ├── research/implicit_slots/         当前解耦模型、标签、训练和评估
│   ├── research/layered_residual/       残差教师、旧定位实验与结果浏览
│   ├── dataset_update/                 数据安装、随机强度派生与几何安全拼接
│   └── configs/ scripts/ tests/ docs/   运行配置、辅助命令与实验记录
├── tools/                              检查仓库与构建服务器更新包
├── docs/                               跨阶段进展、结果摘要和整理记录
├── Literature_Review/                  组员文献与讨论
└── G2_02_Initial_Proposal.pdf           原始proposal
```

各阶段的 `data/`、`outputs/` 与虚拟环境保留在本地，GitHub不包含原始数据、教师缓存、权重和逐窗输出。服务器包也不包含这些文件。

## 环境与数据

从仓库根目录运行：

```bash
./setup.sh
export RIND_PYTHON="$PWD/.venv/bin/python"
```

安装脚本在用户目录管理uv/Python 3.12，无需sudo或snap；默认安装CUDA 12.4版PyTorch，CPU环境可用 `RIND_TORCH_BACKEND=cpu ./setup.sh`。已有环境可直接设置 `RIND_PYTHON=/path/to/python`，不必重装。脚本不会覆盖已有的阶段独立虚拟环境。

数据需单独安装。多源数据步骤见 [数据更新说明](Phase2_Multi_Source/dataset_update/README.zh-CN.md)；单源数据见 [Phase1服务器说明](Phase1_Single_Source_Single_Intensity/docs/SERVER_100K_MULTISCALE.zh-CN.md)。已有缓存可直接训练/评估；只有准备新窗口时才需要原始数据。

## 当前推荐运行流程

复现2万窗口解耦训练：

```bash
./run_masks_tmux.sh start all --gpus auto
./run_masks_tmux.sh status
```

在已有 `outputs/masks_only_v1` 缓存和最佳权重基础上扩展到10万窗口：

```bash
./run_masks_expanded_tmux.sh start all --gpus auto
./run_masks_expanded_tmux.sh status
```

扩展复用旧2万训练窗，仅新增8万；原验证和测试各2千窗保持不变。训练结束自动生成完整指标。配置、恢复命令和输出位置见 [扩展训练说明](Phase2_Multi_Source/docs/MASKS_EXPANDED.zh-CN.md)。后台任务启动后可断开SSH/VPN。

已完成标签准备时用 `start train`，只生成指标时用 `start evaluate`。默认最多选取所有符合空闲阈值的GPU，也可传 `--gpus 6` 或 `--gpu-ids 0,2,3`。

单源定位与旧结果浏览继续从Phase1目录使用原入口；完整定位流程见 [完整训练说明](Phase2_Multi_Source/docs/DECOMPOSITION_TRAINING.zh-CN.md)。新分层模型的可视化接入情况以对应说明为准，旧定位浏览器展示的是已有定位实验。

## 检查与服务器打包

```bash
"$RIND_PYTHON" tools/check_repo.py
"$RIND_PYTHON" tools/package_update.py
```

默认生成 `dist/RIND_Phase2_Update.tar.gz`，始终使用同一文件名；也可用 `--output /path/to/RIND_Phase2_Update.tar.gz`。更新包按白名单从正式目录构建，不再依赖临时更新副本。解压到服务器现有安装的父目录即可覆盖代码，保留已有数据、缓存和模型。

未完成的研究包括有效层的单源物理可实现性、严格整窗解耦成功率、多源定位端到端验证，以及多视窗信息融合与动态扩窗。分层标签目前是既有BFS规则的有效贡献，不应直接解释成真实源逐个恢复。
