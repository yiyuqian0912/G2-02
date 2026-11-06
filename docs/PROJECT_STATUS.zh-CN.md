# 项目状态与已确认结果

更新：2026-11-05。完整背景、方法演进、结果解释和后续实验见 [研究链路报告](RIND_RESEARCH_CHAIN.zh-CN.md)；数值与来源SHA见 [紧凑证据](results/research_chain_evidence.json)。

## 当前主线

未知全场几何下，先提升局部多源响应的有效层数与单层形状预测，再接Phase1单贡献定位。v6模型内部自主逐层减法；教师只提供最终R和无序二值层集合进行监督。两项任务各占0.5，完整响应作为守恒条件。

当前v6尚无最终验证/测试结论；best16新运行的损失einsum中间量OOM已定位并修复，补齐检查点/非检查点显存探测，待服务器重跑。不能把代码回归通过、训练前缀准确率或旧v5指标当作v6正式成绩。

## 已完成实验

| 实验 | 样本与checkpoint | 已确认结果 |
|---|---|---|
| Phase1多尺度单贡献定位 | 80,343训练；4,530验证；4,714测试；best epoch18 | scene-balanced Test CE=7.316051，KL=0.680941；窗口等权核心区Dice@75=0.551211 |
| 边界嵌套实证 | 原10k发布按K各100，共400配置、14,706窗 | 像素跃迁边99.9019%嵌套；共享段96.5442%全段嵌套；有边界窗口15.6632%含混合边 |
| 2万mask-only | 20k/2k/2k，best epoch37 | Test count=79.95%，软Dice=89.02% |
| v2边界细化 | 100k/2k/2k，best epoch32 | 同2000测试窗count=92.25%；历史严格整窗成功81.65%，R≥3仅8.51% |
| v3均衡微调 | 100k/12.5k/12.5k，训练10轮 | best仍是epoch0旧权重；所选模型测试count=90.94%，不能声明新微调best提升 |
| v5失败诊断 | epoch14 **last验证集**，12,500窗 | count=92.616%；旧部署匹配的活跃层IoU通过率82.544%，不是最终测试结果 |
| v5最终结果 | 345,746/12,500/12,500；best epoch16 | 测试count=93.480%；历史部署匹配的单层IoU通过率82.744%；当前v6从此权重初始化 |

历史软Dice、平均硬IoU、单层通过率与严格整窗成功的定义不同，不能互换。不同阶段候选池及评价总体也不同，不能直接比较CE大小。

已核对的10万mask训练窗：R1=60,900，R2=23,548，R3=9,290，R4=3,763，R≥5=2,499。当前v6完整继承上一轮大缓存及追加复杂窗；具体总数以服务器索引为准，不由实验目录名猜测。

## 当前实现及入口

| 功能 | 入口/文档 | 状态 |
|---|---|---|
| v6自主残差模型 | `run_implicit_masks_tmux.sh`；[方法与运行](../Phase2_Multi_Source/docs/SEQUENTIAL_COUNT_SHAPE.zh-CN.md) | 已实现，待最终新旧配对评估 |
| 旧v5训练/复杂窗扩充 | `run_disentangled_masks_tmux.sh`；[说明](../Phase2_Multi_Source/docs/DISENTANGLED_TRAINING.zh-CN.md) | 保留原权重与完整缓存 |
| 失败因素诊断 | `run_failure_diagnostics_tmux.sh`；[说明](../Phase2_Multi_Source/docs/FAILURE_DIAGNOSTICS.zh-CN.md) | 已回收epoch14验证和epoch16最终测试；新旧统一shape-only比较待v6 |
| 旧权重保护与对比 | `model_comparison.py` | 相同完整名单；冻结旧最终best/last；未完成则pending |
| 动态观测 | `run_active_observation.sh`；[说明](../Phase2_Multi_Source/docs/ACTIVE_OBSERVATION.zh-CN.md) | 环境、三基线和18条真实路径核对完成；无定位收益结论 |

单源RL v1服务器结果已回收：冻结Phase1 epoch18，128个测试场景；25%面积预算下PPO Recall@32=41.41%，随机42.19%，优先扩窗44.53%，未证实RL收益。详见 [审核](../Phase1_Single_Source_Single_Intensity/docs/results/rl_single_source_v1_review.zh-CN.md)。新增v2采用16×16全边缘逐块扩展、原始概率逐点最小值融合、空间动作头及10,000/1,000/1,000场景；已完成本地流程检查，正式100k模型收益待验证。[运行指南](../Phase1_Single_Source_Single_Intensity/docs/FRONTIER16_MINIMUM_RL.zh-CN.md)。

## 仍需完成

1. 从已确认的v5 best16继续训练v6，用统一shape-only口径完成固定验证/测试比较；[当前运行指南](../Phase2_Multi_Source/docs/V6_BEST16_RUN.zh-CN.md)。
2. 分别检验count与单层准确率，重点看R≥3，目标至少90%须按明确分母提出。
3. 审计有效层的单源物理可实现性；算术闭合不能代替物理完整支持。
4. 学生层、oracle层分别连接Phase1，量化端到端定位损失。
5. 固定路径验证多视窗去重融合，再训练选窗收益；必要时再做RL。

当前学生默认不读全场障碍几何。物理K、教师有效R与32步容量必须分别报告。联合数据按几何组划分；只收到scene_id的评估包不能独立重证几何隔离。主指标保留全部窗口，约束失败不从分母删除。
