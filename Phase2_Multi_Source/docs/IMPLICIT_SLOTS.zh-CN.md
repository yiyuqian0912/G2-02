# Phase2 隐式贡献槽位学生

## 目标

教师继续使用逐层残差剥离和 Phase1 物理采样生成多个单贡献位置分布。学生不读取残差、层掩码、层顺序、真实源编号或全场障碍物几何，只读取原始多源响应与窗口坐标，并在一次前向传播中输出多个无序贡献槽位。

每个槽位输出存在概率、有效强度和一张 Phase1 风格的全场位置分布。槽位使用独立 cross-attention 读取同一组响应 token；同一像素可以同时为多个贡献提供证据，不进行像素排他分配。

## 为什么使用 8 个槽位

现有 2,605 个测试窗口中，94.89% 的窗口不超过 4 个信息层，98.66% 不超过 6 层，99.42% 不超过 8 层。真实源数与有效残差层数并不相同，所以不直接使用真实源上限 4。超过 8 层的窗口只参与聚合占据场和截断计数监督，不截断后伪装成完整槽位标签。

## 教师缓存

默认复用：

```text
outputs/full_training_h20_v2
```

`index` 只读取已有教师缓存并建立窗口级槽位索引，不重新渲染教师。它同时把分散的压缩层文件转换为连续训练缓存；后续 epoch 直接读取共享 memmap，不再反复解压 NPZ。

## 运行

```bash
cd /home/qq/yiyu/RIND_Phase2_Training_Minimal/Phase2_Multi_Source

./run_implicit_slots.sh index
./run_implicit_slots.sh train --gpus 8
```

### tmux 后台运行

从最小包根目录启动：

```bash
cd /home/qq/yiyu/RIND_Phase2_Training_Minimal
./run_implicit_slots_tmux.sh start --gpus 8
```

命令返回 `Started background training` 后即可退出 SSH 和 VPN。训练进程属于服务器上的 tmux 会话，不依赖本地网络连接。默认会话名为 `rind_phase2_slots`，不会使用容易与系统账号混淆的 `test`。

重新连接服务器后可使用：

```bash
./run_implicit_slots_tmux.sh status
./run_implicit_slots_tmux.sh logs
./run_implicit_slots_tmux.sh attach
```

在 attach 界面按 `Ctrl-b`，松开后按 `d`，只会退出界面，不会停止训练。确实需要中止时执行：

```bash
./run_implicit_slots_tmux.sh stop
```

训练结果仍保存在 `Phase2_Multi_Source/outputs/implicit_contribution_slots_v1/`。每次后台任务的完整终端日志、命令、开始／结束时间和退出码保存在其 `tmux_runs/<timestamp>/` 子目录。服务器重启会终止普通 tmux 会话，但单纯关闭终端、SSH 或 VPN 不会。

启动时应首先打印：

```text
RIND build: phase2-implicit-slots-2026-10-27-metrics-v4
```

如果看不到该行，说明服务器仍在运行同名 tar 的旧副本；不要根据旧日志判断新显存修复是否生效。

`ddpfix-v3` 关闭了 PyTorch 2.4 中与梯度累积及 non-reentrant activation checkpoint 冲突的 DDP `static_graph` 和 `no_sync` 优化。该修改不改变模型、损失或累积后的梯度，只改用保守的 reducer 执行路径。

### 完整评估

训练完成后从 Phase2 目录运行：

```bash
./run_implicit_slots.sh evaluate --gpus 8 --bootstrap-repetitions 1000
```

评估固定读取 `models/seed_0/best.pt`，使用精确的一对一槽位匹配，并同时输出：有效层数量 accuracy/MAE/confusion matrix、presence precision/recall/F1、单层 CE/KL/JS/信息恢复率/HDR 几何指标/强度误差、整合 occupancy 的 HDR Dice/Recall/Precision/IoU、无效候选质量、信息性真源距离、按窗口大小和层数分层的结果，以及按独立场景聚类 bootstrap 的 95% 置信区间。

结果写入：

```text
outputs/implicit_contribution_slots_v1/evaluation/implicit_slots/
├── comprehensive_metrics.json
├── REPORT.zh-CN.md
├── windows.jsonl
├── windows.csv
└── shards/
```

`--gpus` 表示最多使用多少张空闲卡，不再假定使用物理卡 `0...N-1`。脚本默认同时检查 GPU 利用率和已用显存；如果只有物理卡 2、3 空闲，即使执行 `--gpus 8`，也会自动设置 `CUDA_VISIBLE_DEVICES=2,3` 并启动两进程 DDP。也可以明确指定：

```bash
./run_implicit_slots.sh train --gpu-ids 2,3
```

默认空闲标准是利用率不超过 10%、已用显存不超过 2048 MiB，可以通过 `--max-utilization` 和 `--max-used-memory-mib` 调整。

自动选择发生在启动 DDP 之前。物理卡号通过 `CUDA_VISIBLE_DEVICES` 映射成训练进程内的连续逻辑卡号，因此例如只选中物理卡 2、3 时，两个进程分别使用逻辑 `cuda:0`、`cuda:1`；不会与配置文件中的旧卡号发生冲突。如果最终只剩一张空闲卡，也会安全退化为单卡训练。

也可以连续执行：

```bash
./run_implicit_slots.sh all --gpus 8
```

输出：

```text
outputs/implicit_contribution_slots_v1/
├── slot_records.json
├── protocol.json
├── progress.jsonl
├── training_summary.json
└── models/seed_0/
    ├── best.pt
    ├── last.pt
    └── training.json
```

## 匹配与输出

训练使用 GPU Sinkhorn 一对一匹配教师层和学生槽位。未匹配槽位学习“不存在”。最终同时输出：

```text
contribution_density = Σ presence × slot_probability
occupancy = 1 - Π(1 - presence × slot_probability)
```

前者保留预期贡献数量，后者表示每个候选位置至少被一个有效贡献支持的软占据概率。

## 教师答案怎样进入训练

假设一个窗口被教师剥离为三层。教师为每层保存同一候选网格上的 Phase1 分布 `q1/q2/q3`、层强度和 HDR@75 核心掩码。组成 batch 时，它们只被放入监督张量：

```text
teacher_probability: [B, 8, N]，前三行是 q1/q2/q3，其余补零
layer_mask:          [B, 8]，前三项为真
teacher_intensity:   [B, 8]
count_target:        3
teacher_occupancy:   1 - (1-q1)(1-q2)(1-q3)
```

学生的输入张量只有原始完整响应、局部障碍掩码、窗口坐标和候选坐标。学生独立产生八张 `p1...p8`。训练在 GPU 上计算三张教师图与八张学生图的 `3×8` 代价矩阵，再用一对一 Sinkhorn 匹配决定本批次中谁监督谁。例如 `q1→p5, q2→p2, q3→p7`。匹配后的交叉熵和核心区域损失训练位置图；其余槽位训练为不存在。数量损失训练 `count=3`，聚合损失比较所有槽位形成的 occupancy 与教师 occupancy。

教师层不会拼接到学生输入中，也不会按剥离顺序固定对应槽位编号。它只在损失函数一侧提供目标。

多卡训练使用一张卡一个进程的 DDP。安全默认值为每卡 `micro-batch=8`、候选分块 1024，并以梯度累积把有效全局 batch 保持在约 128：1/2/4/8 卡分别累积 16/8/4/2 个 micro-batch。服务器使用的 PyTorch 2.4 在 activation checkpoint、`static_graph` 和 `no_sync` 同时启用时会触发 reducer 内部断言，因此兼容配置同时关闭 `static_graph` 和 `no_sync`。每个 micro-batch 会正常同步梯度，累积后的数学结果不变。

8 个槽位仍然展平到 GPU batch 维度并行评分。候选分块使用 non-reentrant activation checkpoint：前向只保留每块输出，反向时逐块重算评分。这不改变概率、匹配或损失定义，但不会再让 16,384 个候选的所有大型计算图同时占据显存。每个 epoch 会在日志记录各 rank 中最大的 allocated/reserved GPU GiB。损失匹配和 Sinkhorn 全部留在 GPU，CPU 只负责读取共享 memmap 和准备小批次。

训练硬件参数（空闲卡数、micro-batch、候选分块）不改变已有 `index` 缓存的语义。更新本版本后可以直接执行 `train`，不需要重新渲染教师；若已经存在有效的 `last.pt`，模型仍按检查点逻辑续训。
