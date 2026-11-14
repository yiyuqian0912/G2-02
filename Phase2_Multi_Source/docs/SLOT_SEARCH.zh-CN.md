# 自动探索槽位数与解耦模型参数

这是已有局部分层缓存上的新实验，不覆盖原32槽模型，不需要重新生成位置教师，也不改变最低正残差/BFS分层规则。目标是在既定候选范围和预算中找到较好的结构；不保证数学意义的全局最优或达到90%。

## 一条命令运行

解压同名更新包到现有安装的父目录后，在安装目录运行：

```bash
cd /home/qq/yiyu/RIND_Phase2_Training_Minimal
./run_slot_search_tmux.sh start all --gpus auto
```

也可限制最多6卡：`--gpus 6`；指定卡：`--gpu-ids 0,1,2,3,4,5`。默认排除已有计算进程的卡，再要求显存使用≤2048MiB、利用率≤10%；全量训练前重新检查一次。显式指定卡号代表你已确认可使用这些卡。不会在没有空闲卡时自动退回CPU。

tmux会话为 `rind_slot_search`，启动后可断开SSH/VPN：

```bash
./run_slot_search_tmux.sh status
./run_slot_search_tmux.sh logs
./run_slot_search_tmux.sh attach
./run_slot_search_tmux.sh stop
```

中断后再次执行 `start all`，会复用固定样本、已完成试验、晋级结果和训练断点。正在训练的试验从最后一个完整epoch恢复。所有搜索子进程也会响应停止操作。相同输出目录有文件锁，防止两个任务同时写入。

## 执行顺序

1. **固定标签来源。** 优先读取完成的 `outputs/masks_only_100k`，其次 `outputs/masks_only_v1`。第一次运行后锁定来源；途中扩展缓存完成不会更换实验数据。只有已有连续缓存即可运行，无需原始ZIP、逐窗NPZ或Phase1权重。
2. **审计容量。** 统计完整训练集、验证集的有效层数。一个槽数只要装不下其中任一完整标签，就标记不适用，不删除该窗口、不截断层数。实际4源不等于BFS答案最多4层；若所有小容量都被排除，需要检查分层碎片，而不是宣称6槽足够。
3. **固定共同开发样本。** 默认按有效层数和窗口大小成比例抽取8000训练窗、1500验证窗，每个非空分层至少保留一个；不足则全部使用。这样显存探测能看到完整训练/验证中的最大原生窗口。保存场景/窗口名单及随机种子。所有配置用相同窗口、相同标签和相同训练种子0，开发试验目标全局batch128，增加每轮优化步数。打包字节只复制一次；各试验共享这份缓存，不重新渲染、不复制多份大数组。
4. **分阶段探索。** 默认候选槽数6/8/12/16/32，宽度128/192/256，解码深度2/4，学习率5e-5/1e-4，8注意力头。在有效组合中选最多18组，强制包含小/大两组结构各自的槽数对照，其余固定随机抽样。累计训练预算依次为5、15、40epoch，每轮保留约前三分之一；晋级继续同一模型和优化器。未晋级不等于充分收敛后一定更差。
5. **锁定优胜配置。** 使用场景等权验证质量分数：`0.50×层数准确率 + 0.35×硬掩码IoU + 0.15×max(0,1−响应相对L1)`。同时报告严格整窗成功率。在最高分0.005以内优先较少参数，再比较槽数。权重和容差在运行前固定，不用测试集调整。
6. **自动全量训练。** 用选定结构与学习率，从头训练所选来源缓存的全部训练窗口，验证仍用完整原验证集。默认100epoch、patience15、seed0。不是直接拿开发子集试验的权重当最终模型；也不会裁切旧32槽权重或强行加载不同宽度权重。
7. **最后测试一次。** 所有选型与全量早停完成后，才评估锁定测试集。若测试中有超过所选容量的窗口，保留完整标签，把缺少的预测层按未预测处理；层数/严格成功率计失败、掩码指标包含漏层。不会删除困难窗口来提高指标。

## GPU执行方式

开发试验每张空闲卡独立运行一个配置，避免小模型全部堆在DDP上；结束后立即安排下一个配置。最后全量训练用选中卡做DDP。训练复用现有bf16、TF32、激活检查点和GPU Hungarian匹配；匹配、损失、前向和反向都在GPU上。

首次单卡试验在最大原生视窗尺寸上探测batch16/32/64/128，比较前向/损失/反向的吞吐，并预留Adam状态和20%显存余量。探测不更新正式训练权重，OOM会停止尝试更大batch。所有单卡候选的目标全局batch保持一致，仅采用该目标的整除微batch。全量DDP根据探测容量和卡数选微batch；实际全局batch由现有梯度累积公式确定并写入日志，可能略高于请求值。

探测不包含磁盘与DDP通信开销，不能保证某个GPU利用率数字或固定加速倍数。标签读取、解包、配置与调度仍需CPU和磁盘；这里没有新增CPU射线渲染或CPU张量匹配。

## 参数在哪里修改

`Phase2_Multi_Source/configs/slot_search.json`：

| 参数 | 用处 |
|---|---|
| `source_candidates` | 已有缓存目录，按顺序选择；实际窗口数写入plan |
| `output_root` | 新搜索及全量结果目录，原实验不变 |
| `slot_counts` | 待比较的最大有效贡献数，未启用槽位允许为空 |
| `hidden_widths` / `decoder_depths` | 共享特征宽度和槽解码层数，不是堆叠多个独立模型 |
| `learning_rates` | 每个候选从头训练时的学习率 |
| `max_trials` | 初始小实验数量上限，必须容纳所有有效槽数的小/大结构控制对照；两种结构相同时只需一组 |
| `pilot_records` | 共同开发训练/验证窗口数，不是正式全量训练规模 |
| `pilot_global_batch_size` | 每个开发试验的目标全局batch，默认128；正式训练沿用模板的目标256 |
| `epoch_budgets` / `reduction_factor` | 累计训练预算和晋级比例 |
| `autobatch` | 探测开关、微batch候选、显存预算；可关闭后沿用模板batch |
| `final_training` | 全量训练epoch、早停与种子；多个种子会独立训练并分别保存评价 |
| `success_thresholds` | 严格整窗：所有实际启用层与教师一一对应，每层IoU≥0.9、强度相对误差≤0.1、总响应相对L1≤0.05 |

仅做计划或分阶段运行：

```bash
./run_slot_search.sh plan
./run_slot_search_tmux.sh start search --gpus auto
./run_slot_search_tmux.sh start fit --gpus auto
```

改候选集、开发样本、损失、预算或模型定义后使用新 `output_root`，避免混用不同实验的断点。可通过 `RIND_SEARCH_CONFIG=/absolute/path/custom.json` 指定自定义搜索配置。已有搜索不能只编辑槽数来恢复旧模型。

## 输出在哪里

默认 `Phase2_Multi_Source/outputs/slot_architecture_search_v1/`：

| 文件/目录 | 结果 |
|---|---|
| `capacity_audit.json` | 完整训练/验证层数直方图、每个槽数的覆盖率和排除原因 |
| `plan.json` / `search_protocol.json` | 实际数据规模、参数、来源与代码指纹 |
| `pilot_data/sample_manifest.json` | 可复现的共同窗口名单 |
| `trials/*/config.json` / `epoch_*.log` | 每个结构的配置、日志、断点和验证报告 |
| `trials/*/batch_probe.json` | GPU微batch探测与所选batch |
| `rung_*.json` / `search_status.json` | 每轮结果、运行/失败状态 |
| `selected.json` / `search_report.zh-CN.md` | 锁定结构、参数量和验证分数 |
| `final_summary.json` | 优胜结构、正式样本数与全部最终种子的测试指标汇总 |
| `best_full/models/seed_0/best.pt` / `last.pt` | 全量最佳模型与断点 |
| `best_full/training_summary.json` | 全量训练摘要；训练中不计算测试分数 |
| `best_full/evaluation/decomposition_metrics.json` | 最后测试的完整指标与场景bootstrap区间 |
| `best_full/evaluation/decomposition_metrics.zh-CN.md` | 中文测试报告 |

多个最终种子时，测试指标放在 `best_full/evaluation/seed_<N>/`，避免覆盖。`final_status.json` 与 `best_full/fit_complete.json` 标记整体完成。搜索不能替代独立重复实验；最终是否优秀，要看锁定测试集的严格成功率、层数准确率、掩码IoU、强度与重建指标，不能只看综合分数。

层掩码仍是算术BFS有效贡献，不自动等于物理单源可实现掩码；本次没有新增位置概率场训练或端到端Phase1定位结论。
