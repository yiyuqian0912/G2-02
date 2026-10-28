# RIND Phase2：多源响应分层与定位

[项目总览](../README.zh-CN.md) · [当前解耦训练](docs/MASKS_ONLY.zh-CN.md) · [10万窗口扩展](docs/MASKS_EXPANDED.zh-CN.md) · [完整定位流程](docs/DECOMPOSITION_TRAINING.zh-CN.md)

Phase2的目标是处理正强度、多源线性叠加响应。当前主线先让学生从完整局部响应预测有效贡献的二值掩码、强度和存在性；然后研究复用Phase1单贡献定位器。学生默认不读取全场障碍物几何。

## 当前工作流

局部总响应 → 最低正残差/BFS标签 → 重叠槽位掩码、强度与存在性监督 → 验证集选模 → 固定测试集评估。

在G2-02根目录执行：

```bash
./run_masks_tmux.sh start all --gpus auto
./run_masks_tmux.sh status
```

2万窗口版本已完成；配置为 `configs/masks_only.json`，输出为 `outputs/masks_only_v1/`。已收到的测试摘要：软Dice=0.8902，有效层数准确率=79.95%，响应重建绝对MAE=0.01773；最佳epoch=37。强度损失不等同于强度MAE，软Dice也不等同于整窗解耦成功率。

已存在旧缓存和最佳权重时，用扩展版本：

```bash
./run_masks_expanded_tmux.sh start all --gpus auto
./run_masks_expanded_tmux.sh status
```

扩展目标为100000/2000/2000个实际完成的train/validation/test窗口，复用旧训练前缀并锁定旧评估集。从旧最佳模型微调，训练完成后自动生成 `evaluation/decomposition_metrics.json` 和中文报告。更多参数见扩展说明；10万窗口的正式训练结果尚待回收。

## 目录与文件职责

| 路径 | 做什么 |
|---|---|
| `research/implicit_slots/decomposition_model.py` | 原始多源响应编码、可重叠槽位、强度/存在性输出、匹配和监督损失 |
| `research/implicit_slots/mask_only_data.py` | 低成本局部残差标签、二值掩码和连续训练缓存，不生成位置概率 |
| `research/implicit_slots/expand_masks.py` | 复用旧缓存，追加新窗口，保持评估集固定并支持恢复 |
| `research/implicit_slots/train_decomposition.py` | 解耦与完整定位两种模式的训练、选模和断点 |
| `research/implicit_slots/evaluate_masks.py` | 解耦掩码、层数、强度与重建指标 |
| `research/implicit_slots/evaluate_decomposition.py` | 完整定位模式的分层/位置指标，区别实际预测和oracle诊断 |
| `research/implicit_slots/slot_*.py`、`train_slots.py` | 早期概率槽位基线与共享接口，保留实验复现 |
| `research/layered_residual/` | 残差教师、逐层物理评分、旧层定位实验和结果浏览 |
| `src/rind_phase2/` | 多源读取、正确加权叠加、几何遮挡、联合候选教师和检查工具 |
| `dataset_update/` | 新数据安装、随机强度派生、重复几何安全划分与联合读取 |
| `configs/` | 当前和历史实验配置；当前默认以masks开头的配置为准 |
| `tests/`、研究模块中的 `test_*.py` | 数据、物理、标签、缓存、模型和命令入口的回归检查 |
| `docs/` | 当前运行说明、方法限制、数据说明及历史实验记录 |

`src/rind_phase2`负责物理和数据，并不再保留无法运行的联合学生占位模块；当前学生在 `research/implicit_slots` 中。早期 `part_e/` 空接口与空基线目录已移除。原实现可从Git历史恢复，研究结论见保留的文档。

## 数据与评估约定

- 已有旧1万随机强度多源配置；新发布包含10万配置、每配置256个随机窗口，原发布为单位强度。
- 安装器保留原始数据，并创建随机(0,1)强度派生版本；正式联合目录为 `data/raw/phase2-merged-110k`。
- 同一障碍物几何在不同发布出现时保持同split。11万配置不是11万独立几何。
- 总响应必须逐源按强度加权再相加；底层多源OR渲染不能直接当作加权响应。
- 当前窗口筛选保留同时有自由零响应与正响应的窗口；全正响应窗口沿用原协议跳过。
- 局部分层不读真实源编号或位置来决定传播。有效贡献强度可以大于1，有效层数不等于真实源数。
- 算术重建正确不保证每层可由单个物理源实现。严格整窗成功率与多源定位闭环尚需验证。

数据安装见 [dataset_update](dataset_update/README.zh-CN.md)。已有缓存可直接训练/评估，不需要重渲染位置教师。

## 已有可视化与历史实验

`./view_layer_results.sh`、`./view_results.sh` 保留旧定位实验的测试窗口浏览和逐层剥离可视化；需对应的原数据、教师缓存和checkpoint。它们不应被当成当前纯解耦模型的自动可视化。

早期层定位、几何消融、集合训练探索及联合穷举工具保留用于研究复现；历史报告见 [v4–v6记录](docs/RIND_Phase2_v4_完整实验设计与结果.md)。这些流程不替代当前推荐入口。
