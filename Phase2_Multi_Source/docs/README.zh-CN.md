# Phase2 文档索引

更新：2026-11-04。当前主线是v6自主残差解耦；历史文档保留当时设定与实验语境，不代表当前默认方案。

## 当前阅读顺序

1. [完整研究链路报告](../../docs/RIND_RESEARCH_CHAIN.zh-CN.md)：物理、数据、Phase1/Phase2演进、结果与下一步。
2. [当前项目状态](../../docs/PROJECT_STATUS.zh-CN.md)：哪些结果已回收，哪些仍待验证。
3. [v6两项等权监督、自主减法与守恒](SEQUENTIAL_COUNT_SHAPE.zh-CN.md)：当前模型、loss、约束、缓存复用、tmux和旧权重对比。
4. [从v5 best epoch16继续训练](V6_BEST16_RUN.zh-CN.md)：当前启动命令、旧权重保护、GPU缓存、自动批量与复核。
5. [数据安装与拼接](../dataset_update/README.zh-CN.md)：随机强度派生、重复几何绑定和联合读取。

## 历史流程和研究工具

| 文档 | 用途/入口 |
|---|---|
| [v5独立监督与复杂窗扩充](DISENTANGLED_TRAINING.zh-CN.md) | `run_disentangled_masks_tmux.sh`；上一轮大缓存与权重 |
| [失败因素诊断](FAILURE_DIAGNOSTICS.zh-CN.md) | `run_failure_diagnostics_tmux.sh`；区分计数、形状、强度和旧strict |
| [均衡微调](BALANCED_FINETUNE.zh-CN.md) | `run_balanced_masks_tmux.sh`；100k训练、12.5k/12.5k自然评估 |
| [边界细化模型](REFINED_MASKS_RUN.zh-CN.md) | `run_refined_masks_tmux.sh`；v2历史实验 |
| [复杂窗口盘点](COMPLEX_WINDOW_INVENTORY.zh-CN.md) | `run_count_masks_tmux.sh start prepare`；仅盘点，不自动训练 |
| [计数优先训练草案](COUNT_FIRST_TRAINING.zh-CN.md) | 历史训练模板，不代替当前v6配置 |
| [槽位/结构搜索](SLOT_SEARCH.zh-CN.md) | `run_slot_search_tmux.sh`；历史自动选型 |
| [2万窗口分层](MASKS_ONLY.zh-CN.md) | `run_masks_tmux.sh`；初始mask-only实验 |
| [10万窗口扩展](MASKS_EXPANDED.zh-CN.md) | `run_masks_expanded_tmux.sh`；扩展旧2万缓存 |
| [完整分层定位](DECOMPOSITION_TRAINING.zh-CN.md) | `run_decomposition*_tmux.sh`；历史位置教师与定位学生 |
| [动态观测设计](ACTIVE_OBSERVATION.zh-CN.md) | `run_active_observation.sh`；环境与三种参考路径，尚无策略训练结论 |
| [多视窗集合训练探索](VIEW_SET_TRAINING.zh-CN.md) | 早期被动集合蒸馏与后续历史融合方向 |
| [旧定位实验汇报](RIND_Phase2_v4_完整实验设计与结果.md) | 早期模型和全场几何消融结果 |

方法与接口历史定义见 [method.md](method.md)、[interfaces.md](interfaces.md)、[dataset-guide](dataset-guide.zh-CN.md)。局部算术层、单源物理位置教师、学生预测层和最终展示图是不同对象；对照时核对实验与指标协议。
