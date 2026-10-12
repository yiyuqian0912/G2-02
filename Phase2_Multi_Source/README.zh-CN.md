# RIND Phase II：多源位置与强度推断

以 Yushu 的 `yushu/phase1-multi-source` 分支为基础，目录与单源 Phase I 平级，模块职责保持一致。[来源与移植记录](docs/upstream.md)。

**已有：**多源数据安装/读取、原始数据浏览、整组源重渲染、联合教师，以及 `research/layered_residual/` 中的逐层教师、单一全场学生训练、统计报告和交互式结果浏览器。完整多源联合后验学生、正式联合评价和自适应联合搜索仍待实现。当前逐层学生输出明确标为贡献位置概率场，不冒充完整联合后验。

## 实际数据设定

- 发布包：10,000 个场景、367,109 个 source-free 局部窗口；安装后由数据校验器核对这些计数。
- 每场景随机 1–4 个源，每源强度独立采样于 [0,1)。可见强度相加，响应允许大于 1。
- 动态四分树最小 size=16；含任意源节点继续分，无源节点停止，含源最小叶子不作为观测。每场景窗口数量不固定。
- 一个逆问题候选是一整组 `(x,y,intensity)`，允许考虑不同源数。交换源顺序不改变物理含义。
- 参考解包括同位置强度拆分，可能有 5 个源；生成范围 1–4 不自动等于所有逆解的源数量上限。

## 目录与 Phase I 对应

| 文件夹/入口 | 作用与当前状态 |
|---|---|
| `src/rind_phase2/` | 主实现；包名独立，不修改 `rind_phase1` |
| `data.py` / `install_data.py` / `browser.py` | 已导入多源读取、安装和原始数据浏览 |
| `physics.py` / `search.py` / `teacher.py` | 已导入整组源物理代价、联合采样、教师分布与保存 |
| `diagnostics.py` / `checks.py` | 已导入源数量/二维投影、参考解和数据/教师检查 |
| `interfaces.py` | 新增学生交接草案，校验强度、变长源数量及零 padding；不代表学生已实现 |
| `model.py` / `train.py` / `predict.py` | 预留多源集合能量模型、软分布训练及联合预测；显式待实现 |
| `protocol.py` / `sampling.py` / `experiments.py` | 冻结实验、学生侧联合查询与编排；仅配置查看已可用 |
| `evaluate.py` / `part_e/` | 保留 Phase I 的评价/联调结构；联合指标与无序源匹配待实现 |
| `results_browser.py` / `templates/` / `view_results.sh` | 浏览已训练的逐层整合实验；自选测试视窗和层，显示局部响应、教师/学生全场概率及逐层剥离 |
| `research/layered_residual/` / `run_combined_training.sh` | 当前逐层教师、单一全场概率学生、断点续训、统计与可视化的正式研究入口 |
| `run_server_full.sh` / `configs/layered_combined_server_full.json` | 8000/1000/1000 scene 划分的服务器全量入口；扫描选定尺寸的全部真实窗口 |
| `scripts/` / `run_training.sh` | 原始联合源集合教师与尚未完成的联合后验学生草案；不用于当前逐层学生正式训练 |
| `configs/phase2.json` | 上游多源数据与教师约定，未确定的正式参数留空 |
| `configs/experiment_multisource.json` | 新实验草案；源数量先验、采样参数、训练参数待定 |
| `configs/splits/` | 等待新数据身份与正式 scene 清单；不复制单源 IDs |
| `tests/` | 上游小场景数据/物理/联合教师测试及新增接口/骨架测试 |
| `vendor/` | 分支自带读取器/渲染器快照，保留来源 |
| `student_baseline/` | 仅预留多源基线位置，没有单源旧代码 |
| `data/` / `outputs/` | 隔离的多源数据与结果目录，仅占位文件进入 Git |

## 安装与开始验证

```bash
cd ../Phase2_Multi_Source
uv sync --locked --extra train --extra browser --extra diagnostics
# 多源 ZIP 放在本项目目录，或传入完整路径；建议保留下载包。
./scripts/install_data.sh /path/to/RIND-adaptive-v3-10000scenes.zip --keep-archive
./scripts/browser.sh --no-open
```

原始数据浏览默认 **http://127.0.0.1:8768**，避免与 Phase I 结果浏览的 8767 冲突。多源数据目录为 `data/raw/phase2-multi-source/`；已有多源数据可通过 `RIND_MULTI_DATA_ROOT` 指向，无需重复安装。不要指向单源发布包。

训练完成后打开 Phase II 结果浏览器：

```bash
./view_results.sh
```

浏览器默认地址为 **http://127.0.0.1:8768**。它会自动发现包含 `records.json` 和 `models/seed_*/best.pt` 的实验。页面可以按窗口 size、恢复层数和测试视窗筛选，并逐层查看剥离前残差、二值层响应、实际减量、剥离后残差与该层 Phase1 全场概率。若原始数据浏览器正在使用同一端口，可运行 `./view_results.sh --port 8769`。

下载链接及安装空间、读取代码示例见 [数据指南](docs/dataset-guide.zh-CN.md)。没有自动下载或解压完整数据。

无需完整数据即可运行仓库的小场景测试：

```bash
PYTHONPATH=.:src .venv/bin/python -m unittest discover -s tests -q
./run_training.sh --check
```

当前逐层整合学生可以直接训练；完整多源联合后验学生仍未实现。服务器全量流程、记录筛选和恢复方式见[逐层残差说明](research/layered_residual/README.zh-CN.md#服务器全量训练)。

## 方法边界

学生输入草案延续当前 Phase I 的 **response + window**；数据保留 obstacle 供验证和可选消融，源真值/分源通道不作输入。上游教师采样会使用隐藏几何筛选位置，因此未来学生若沿用其候选范围，必须标为候选/几何辅助；公开域推理需要独立设计。

教师 `q_i ∝ base_mass_i × exp(-C_i/tau)`。这里 base_mass 表示源数量先验与采样测度，**不是空间面积权重**。跨不同 K 的比较不能删掉它，也不能分别 softmax 后直接比较。

[方法](docs/method.md) · [接口与字段用途](docs/interfaces.md) · [实验计划](docs/experiments.zh-CN.md) · [多源图像解释](docs/dense_probability.zh-CN.md)


## 逐层残差教师

当前方法先在局部残差区域图中逐层恢复二值贡献响应并完成剥离；所有层确定后，只遍历一次全场均匀候选位置。每个候选的单源可见性只渲染一次，再分别与所有层比较、独立归一化，最后形成一张等权层混合教师概率场。

```bash
./run_combined_training.sh all --config configs/layered_combined_2000.json
```

完整算法、训练参数、服务器命令、逐层可视化和失败语义见[逐层残差说明](research/layered_residual/README.zh-CN.md)。旧的联合随机教师入口继续保留作为对照。
