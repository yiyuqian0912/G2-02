# 先训练局部分层解耦

二次代码审查与验证范围见 [MASKS_ONLY_AUDIT.zh-CN.md](MASKS_ONLY_AUDIT.zh-CN.md)。扩展到10万窗口的入口见 [MASKS_EXPANDED.zh-CN.md](MASKS_EXPANDED.zh-CN.md)。当前启动日志版本为 `v9-expanded`。

保留完整训练入口 `run_decomposition_union_tmux.sh`。新增 `run_masks_tmux.sh` 只训练从原始多源响应预测有效层掩码、强度和层数；不生成或读取全场候选位置、物理评分、位置概率场，也不加载或实例化 Phase1 定位网络。

## 更新和运行

更新包仍名为 `RIND_Phase2_Update.tar.gz`。它是代码覆盖包，不包含原始数据、缓存、模型和虚拟环境。请解压到现有安装目录的父目录，保留服务器已有数据与环境：

```bash
cd /home/qq/yiyu
tar -xzf RIND_Phase2_Update.tar.gz
cd RIND_Phase2_Training_Minimal
./run_masks_tmux.sh start all --gpus 8
```

`all` 顺序执行轻量标签准备、训练及最佳模型测试集损失评估；完整指标用下面的 `evaluate`。后台运行后可以断开 SSH/VPN。已有重型渲染任务若仍在运行，先用它原来的 tmux 入口发送停止命令，避免两项任务争用资源。

```bash
./run_masks_tmux.sh status
./run_masks_tmux.sh logs
./run_masks_tmux.sh attach
# 手动停止；下次 start train 从上个完整 epoch 的 last.pt 恢复
./run_masks_tmux.sh stop
./run_masks_tmux.sh start train --gpus 8
# 训练结束后，使用锁定测试集计算完整解耦指标
./run_masks_tmux.sh start evaluate --gpus 8
```

`--gpus 8` 表示最多选取 8 张空闲卡，默认排除显存占用超过 2048 MiB 或利用率超过 10% 的卡；也可以明确传 `--gpu-ids 0,1,2`。GPU 选取发生在轻量标签准备之后。

## 数据、样本量和标签

配置为 `Phase2_Multi_Source/configs/masks_only.json`。默认读取已经安装的 `data/raw/phase2-merged-110k`，包括旧场景和新随机强度场景，沿用按几何隔离的 train/validation/test 划分。默认实际完成的样本数为训练 20,000、验证 2,000、测试 2,000；跳过的全亮、全暗、无自由像素或未完成窗口不计入数量。可以修改 `label_preparation.record_counts` 扩大规模。更改标签协议、数据或模型请使用新的 `output_root`，避免混用旧缓存；只想重启训练则保留配置并用 `train`。

标签准备从数据中读取窗口总响应与局部障碍物信息，按既有的最低正残差、BFS传播、逐层减法生成二值层掩码与正强度，不主动把最低值拆成真实源。相同强度的断开区域不直接合并。保留全部算术层及原始响应，逐窗验证叠加重建，限制最多32层。选窗和标签推导不使用真实源位置、源编号、真实强度列表或每源真值掩码来决定传播。数据读取器可以从数据已有可见性与强度重建**观测总响应**，这不产生新的候选渲染。

这仍是有监督训练，需要准备轻量的局部标签。它省去的是昂贵的全场物理教师采样，不是完全免除监督。局部标签在 CPU 上做少量分割、邻接和减法，训练前向、匹配、损失、反向和优化在 GPU 上执行。没有候选×像素的射线渲染，也不为每层生成全场概率数组。

标签是当前规则给出的**有效贡献分解**。未知/冲突区域不通过真实源补全；当前 BFS 确认区域作为本轮有效掩码，其余响应留在残差后续处理，并记录冲突/未知诊断。完整重建只说明算术一致，尚未检查每个掩码能否由单个物理源实现；此阶段不能声称已经学会准确源定位。

## 模型和监督

保持完整流程的解耦网络：多尺度卷积编码局部响应，32个可学习槽位通过 Transformer 解码得到可重叠的二值掩码概率、正强度和存在概率。强度由 softplus 输出，允许有效贡献强度大于1；二值掩码为单位响应，不代表原始物理源都是单位强度。全局障碍物几何不输入学生；默认局部障碍物也不作为输入通道，仅用于监督屏蔽。

保持精确 Hungarian 一对一匹配、掩码 BCE/Dice、强度、存在性、有效层数、完整响应重建、逐层残差和负残差损失。关闭位置 KL 与位置匹配代价，没有定位梯度或全场位置输出。默认 bf16、多卡 DDP、每卡 micro batch=32、目标全局 batch=256、激活检查点和连续缓存。实际有效 batch 随可用卡数与累积步数取整，日志显示真实配置。没有实测 H20 性能保证，需以日志中的耗时和显存为准。

## 保存位置和评估

默认目录 `Phase2_Multi_Source/outputs/masks_only_v1/`：

| 文件/目录 | 内容 |
|---|---|
| `mask_protocol.json` | 标签规则、源代码指纹、随机种子、分割与样本协议 |
| `local_labels/` | 每窗局部标签、原始响应、强度和未知/冲突诊断，可重复使用 |
| `mask_records.json` | 样本清单、场景/窗口、连续缓存偏移、实际数量和数据指纹 |
| `mask_cache/` | float32响应、压位障碍物/掩码；后续训练只需这个缓存和索引 |
| `models/seed_0/{last,best}.pt` | 恢复训练与最佳模型；不含 Phase1 定位权重 |
| `models/seed_0/training.json` | 每轮各项损失、掩码 Dice、层数准确率、测试指标 |
| `progress.jsonl` | 训练进度、耗时与峰值显存 |
| `evaluation/` | `evaluate`生成完整 JSON、中文报告、逐窗指标及场景聚类区间 |
| `tmux_runs/` | 离线任务完整日志、起止时间、退出状态 |

重点看测试集掩码 Dice/IoU、存在性 precision/recall、有效层数准确率/MAE、强度 MAE、响应与逐层残差重建误差。掩码指标基于匹配，不等于槽位全部检出；两者分开报告。评估只读缓存与最佳权重，不需要原始数据或教师概率文件。

## 后续恢复完整训练

完整流程仍用 `decomposition_union.json` / `run_decomposition_union_tmux.sh`，其旧输出不受影响。之后准备好位置教师，向完整配置的 `training` 加入：

```json
"initial_decomposition_checkpoint": "outputs/masks_only_v1/models/seed_0/best.pt"
```

同时保留原来的 `initial_phase1_checkpoint`，解耦模型配置必须一致。选择新的完整训练 `output_root`，再从完整入口训练，即可复用本轮解耦权重与 Phase1 定位权重；不会自动把旧优化器或旧位置头混进来。
