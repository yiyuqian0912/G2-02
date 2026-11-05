# RIND Phase2：多源响应分层与定位

[完整研究链路](../docs/RIND_RESEARCH_CHAIN.zh-CN.md) · [文档索引](docs/README.zh-CN.md) · [项目状态](../docs/PROJECT_STATUS.zh-CN.md) · [English](README.md)

Phase2研究如何从完整局部多源响应恢复有效贡献层，再复用Phase1定位器。当前主线是v6：学生输入原始响应、窗口位置与size；内部自主逐层减法；教师最终层数与无序掩码只进入监督，不提供教师过程或强度答案。

两个主要目标为层数预测准确率、单层IoU≥0.90通过率，分别监督、等权。完整响应是守恒约束；不以重建loss代替分层正确性。当前v6最终测试结果尚未回收。

## 使用已有大缓存训练

在G2-02根目录执行；示例源目录需换成服务器真实v5大实验目录：

```bash
unset RIND_MASKS_CONFIG RIND_DECOMPOSITION_CONFIG RIND_TMUX_SESSION
./run_implicit_masks_tmux.sh start all \
  --source-root outputs/masks_disentangled_v5_complex245746 \
  --gpus auto --auto-batch
./run_implicit_masks_tmux.sh status
```

需要源目录的 `mask_records.json`、完整 `mask_cache/` 与v5 `models/seed_0/best.pt`。全部原训练记录和追加复杂窗保留，验证/测试不变；无需原始ZIP、无需重新渲染位置教师。缺缓存明确报错，不回退小数据。

```bash
./run_implicit_masks_tmux.sh logs
./run_implicit_masks_tmux.sh attach
./run_implicit_masks_tmux.sh start evaluate --gpus auto --auto-batch
```

正式评价在新输出 `outputs/<源实验名>_implicit_v6/evaluation/seed_0/`。保留旧best/last，旧训练完成后统一名单比较；未完成写pending。具体监督、约束、断点和输出说明见 [v6运行文档](docs/SEQUENTIAL_COUNT_SHAPE.zh-CN.md)。

## 文件职责

| 文件/目录 | 任务 |
|---|---|
| `research/implicit_slots/sequential_count_shape.py` | 当前自主残差模型、二值减法、正强度可行性验收 |
| `sequential_objective.py` | 两项等权监督、最终无序匹配、两个主要指标 |
| `optimization_batches.py` | 跨微批/跨GPU的优化步归一化 |
| `configure_sequential.py` | 冻结旧权重、绑定完整原缓存、固定运行配置 |
| `train_decomposition.py` | 多版本训练、选模、进度和恢复 |
| `evaluate_sequential.py`、`model_comparison.py` | 当前评价、同名单旧最终best对比、配对统计 |
| `evaluation_snapshot.py` | checkpoint角色/epoch/SHA与只读模型快照 |
| `gpu_runtime.py` | 自适应选卡、共同微批探测和运行参数 |
| `mask_only_data.py` | 低成本局部残差标签与连续缓存，候选渲染数为0 |
| `research/layered_residual/` | 历史逐层教师、共享候选位置渲染、层定位与浏览器 |
| `src/rind_phase2/` | 数据、加权可见性叠加、几何邻接与物理评价 |
| `dataset_update/` | 新发布安装、随机强度派生、重复几何同split |
| `research/active_observation/` | 像素预算、移动/扩窗环境及参考路径 |

表内单独文件均位于 `research/implicit_slots/`，除非明确写出其他目录。旧v1–v5、槽数搜索与数据扩充模块保留复现用途，入口见文档索引。

## 数据和结论边界

- 正式随机强度派生响应必须逐源乘强度求和，不能用多源OR响应代替。
- 当前静态分层过滤全暗与所有自由像素全正的窗口；主动观测需另行适配这些情况。
- 有效层R不等于物理源K；32是缓存兼容的执行上限。
- mask-only教师保证算术闭合，不检查每层是否是完整物理单源可见性；不能据此宣称真实源逐个恢复。
- v6只保证接受的输出通过守恒检查，不保证所有输入都有可接受分解。未完成窗口留在指标分母。
- `view_layer_results.sh`、`view_results.sh`服务对应历史位置实验，不能默认展示新纯分层学生。

数据安装见 [dataset_update](dataset_update/README.zh-CN.md)。原数据、缓存、模型与逐窗输出保留本地，不进入GitHub或源码tar。
