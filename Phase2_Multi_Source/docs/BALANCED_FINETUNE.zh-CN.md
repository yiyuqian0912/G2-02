# 现有权重的均衡微调与自然评估集扩充

日期：2026-11-01。保留模型、损失和现有10万条训练标签；从已完成的`masks_boundary_refined_v2/models/seed_0/best.pt`权重继续微调。验证/测试各12,500，按窗口数量实现80%/10%/10%。不重新划分场景或几何组。

## 上传与启动

Mac本地：

```bash
scp -P 10688 ~/Desktop/RIND_Phase2_Update.tar.gz test@58.199.162.221:/home/qq/yiyu/
```

服务器：

```bash
cd /home/qq/yiyu
tar -xzf RIND_Phase2_Update.tar.gz
cd RIND_Phase2_Training_Minimal
./run_balanced_masks_tmux.sh start all --gpus auto
```

也可使用`--gpus 8`表示至多8张空闲卡，或显式`--gpu-ids 0,1`手动指定。默认自动排除已有计算进程的卡。后台启动后可断开SSH/VPN。

```bash
./run_balanced_masks_tmux.sh status
./run_balanced_masks_tmux.sh logs
./run_balanced_masks_tmux.sh attach
```

attach后按Ctrl+B，再按D离开而不停止任务。会话名为`rind_masks_balanced`，与旧版训练会话分开。

## 必须保留的服务器文件

- `Phase2_Multi_Source/outputs/masks_boundary_refined_v2/mask_records.json`和`mask_cache`来源链接。
- `Phase2_Multi_Source/outputs/masks_boundary_refined_v2/models/seed_0/best.pt`，用于微调和同测试集上的初始模型对照。
- `Phase2_Multi_Source/outputs/masks_only_100k/mask_cache`及`mask_protocol.json`，现有10万训练和旧评估缓存不覆盖。
- 原始`Phase2_Multi_Source/data/raw/phase2-merged-110k`及其引用的数据组件，用来读取**新增评估视窗**。现有2000个窗口的缓存无法提供未缓存的新窗口。

tar只带源码，不包含数据、缓存、环境或权重。扩充完成后，后续train/evaluate只读取新缓存；无需再读取原始RIND数据，但原缓存链接及参考权重须保留。

## 执行流程

1. 校验源缓存实际有100,000条训练记录，标签算法、数据版本和原场景划分一致。
2. 训练文件使用只读来源链接，不复制或重新生成10万训练标签。旧验证和测试各2000条按原顺序、原字节保留。
3. 从原验证/测试场景池分别补10,500条完成标签。新增窗口排除对应旧2000窗口已出现的几何组，不从训练池取场景；在原筛选条件下随机选择，不按R挑选。继续跳过全亮、全暗、无自由像素和无法完整分层的窗口。
4. 默认8个CPU进程并行读取已有响应并执行便宜的局部Numba分层；不做全场候选光源渲染。按确定顺序接收结果，不按哪个进程先完成选样本。分块保存、校验和及游标支持中断恢复。
5. 从旧best权重初始化新模型，在扩充后的自然验证集上先测基线并保留为epoch0。新建AdamW优化器，学习率2e-5，最多40轮，patience10；不继承旧实验已早停的计数。验证仍用自然分布的同定义总loss选best。
6. 只有训练启用层数分组采样：1层30%、2层30%、3–4层30%、≥5层10%。不截断32层长尾，不改标签；各组内部保留其原有native-size比例。
7. 每轮名义抽取100,000次，**有放回**。不意味着每轮有100,000个不同窗口，也不意味着新增了独立复杂场景。每轮随机种子状态随checkpoint恢复。多卡尺寸尾批会少量补样，真实四组draw计数写入训练history。
8. 训练完成后，在同一扩充验证/测试集上分别评估所选微调模型和原始best权重。测试不参与模型选择。分别报告全部12,500窗口、旧2000窗口及新增10,500窗口，以便判断分布变化和微调效果。

保留初始权重作为epoch0意味着：如果微调没有改善自然验证loss，best.pt仍可对应epoch0。此时对比报告会明确显示未选中新的微调权重，而不会把较差模型当成提升。

## 配置

`Phase2_Multi_Source/configs/masks_balanced_finetune.json`：

| 字段 | 当前设置 | 用途 |
|---|---|---|
| `cache_source_root` | `outputs/masks_boundary_refined_v2` | 读取原完整标签索引和来源 |
| `output_root` | `outputs/masks_boundary_balanced_v3` | 新实验独立输出 |
| `training.initial_decomposition_checkpoint` | 原模型best.pt | 只加载现有权重 |
| `training.learning_rate` | 0.00002 | 微调学习率 |
| `training.epochs` / `patience` | 40 / 10 | 新微调预算与早停 |
| `training.sampling` | 30/30/30/10，100000draws | 训练均衡采样 |
| `evaluation_expansion.record_counts` | validation/test各12500 | 要求实际完成的评估数量 |
| `evaluation_expansion.workers` | 8 | 仅新增局部标签的CPU准备并发数 |
| `evaluation_expansion.chunk_size` | 64 | 可恢复准备块大小 |
| `evaluation_expansion.max_views_per_scene` | 32 | 自然扫描每场景的候选窗口上限 |
| `training.batch_size` | 每卡32 | 保留原微批大小 |
| `training.target_global_batch_size` | 256 | 梯度累积目标，实际值随卡数向上取整并打印 |

模型仍为32槽、256维、4层解码、3轮共享权重细化，12,325,731参数。几何输入与Phase1定位保持关闭。均衡采样的收益需要本次实验验证，没有90%保证。

## 结果位置

根目录：`Phase2_Multi_Source/outputs/masks_boundary_balanced_v3/`。

- `evaluation_extension_protocol.json`：来源、参数和固定划分约定。
- `mask_records.json`：100000/12500/12500记录、原/新增窗口边界及层数/size统计。
- `sampling_plan.json`：各训练组的原样本数、目标draw数和采样约定。
- `models/seed_0/training.json`：学习曲线和每轮实际四组draw数；`best.pt`/`last.pt`分别用于结果和断点。
- `evaluation/decomposition_metrics.json`、中文md、`windows.jsonl`：微调后完整测试指标；JSON的`by_cohort`分旧/新增子集，`by_layer_count`分有效层数。
- `evaluation/validation_metrics.json`：扩充验证集同定义指标。
- `evaluation/initial_reference/`：原权重在相同扩充评估集上的指标。
- `evaluation/finetuning_comparison.json`和中文md：同一测试集上的微调前后差异。
- `evaluation/validation_finetuning_comparison.json`及中文md：验证集前后差异。

扩充后的同一场景可能有多个窗口，因此同时提供窗口等权和场景等权统计；置信区间按scene_id聚类。不能把12500个窗口一律当作12500个独立几何场景。

## 恢复与重算指标

中断后，使用相同配置重新运行`start all`可继续准备/训练；已完成缓存不会重新生成。已有完整新缓存时，也可：

```bash
./run_balanced_masks_tmux.sh start train --gpus auto
```

从最后完整epoch恢复，未保存的中断epoch重跑。配置中的采样概率/训练目标发生变化时应使用新的output_root，避免静默混合实验。

```bash
./run_balanced_masks_tmux.sh start evaluate --gpus auto --bootstrap-repetitions 1000
```

会重新生成自然验证和测试的前后对照。使用现有相同文件名tar更新，不需要额外上传缓存或权重。

## 检查范围

本地检查覆盖：训练配额和native-size分桶、各rank对齐、非均衡评估保留全部记录、旧缓存字节不变、扩充断点恢复与损坏块拒绝、旧几何排除、warm-start与epoch0保留、相同扩充测试集的前后对照、场景bootstrap优化与原定义一致。另用真实RIND、默认完整模型和两个准备进程跑通小规模流程；这是执行检查，不是全量准确率证据。

H20上的CUDA/NCCL吞吐和全量微调收益需本次服务器运行确认。原2000测试窗已参与设计反馈；新增几何子集应独立查看，不能把合并后的全部结果都称为全新盲测。
