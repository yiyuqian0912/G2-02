# 完整局部响应监督：训练与新数据拼接

本版新增独立实验 `response_decomposition_v2`，保留旧槽位实验和教师缓存。模型先预测各层局部掩码、强度和存在性，再将预测掩码输入共享 Phase1 定位器；教师掩码只用作监督和上限对照。掩码之间允许重叠。

2026-10-28 `v6-geometryfix`：区域邻接改用显式 `@njit` 几何调用链，避免跨模块动态注册在服务器 Numba 环境下编译失败。随机线段和闭合边界与 canonical 几何一致性已验证。启动教师前先单线程编译并自检，候选响应仍使用 GPU 渲染。Numba 缓存默认位于 `runtime_cache/numba-regions-v2`，无需手动清理环境缓存。

若之前已经完成新数据安装与拼接、在教师生成阶段失败，覆盖解压新 tar 后直接运行 `./run_decomposition_union_tmux.sh start all --gpus 8`。该命令复用安装结果、已有教师文件和训练检查点，不需要再次上传数据 ZIP。

## 使用现有教师缓存开始训练

更新代码后在最小包根目录执行：

```bash
cd /home/qq/yiyu/RIND_Phase2_Training_Minimal
./run_decomposition_tmux.sh start all --gpus 8
```

`all` 在后台依次读取旧教师完整标签、生成新训练缓存、开始训练；不重新渲染旧场景。后台默认会话为 `rind_decomposition`。看到 Started background training 后可退出 SSH/VPN。

```bash
./run_decomposition_tmux.sh status
./run_decomposition_tmux.sh logs
./run_decomposition_tmux.sh attach
./run_decomposition_tmux.sh stop
```

attach 后按 Ctrl-b 再按 d 可退出界面并保持运行。stop 发送 Ctrl-C；续训从上一个完整 epoch 保存的 last.pt 开始，不保存被中断的半个 epoch。

训练已有新缓存时，续训执行：

```bash
./run_decomposition_tmux.sh start train --gpus 8
```

评估也使用后台会话：

```bash
./run_decomposition_tmux.sh start evaluate --gpus 8 --bootstrap-repetitions 1000
```

输出目录：

```text
Phase2_Multi_Source/outputs/response_decomposition_v2/
  decomposition_records.json  完整层数、定位资格、过滤原因与内容指纹
  decomposition_cache/       原始响应、全部层掩码/强度/概率的连续缓存
  models/seed_0/             best.pt、last.pt、training.json
  evaluation/                端到端指标与真实掩码 Phase1 上限
  tmux_runs/                 完整终端日志、命令、退出码
```

## 训练逻辑和参数

参数位于 `Phase2_Multi_Source/configs/decomposition_student.json`。

- 原始多源响应与窗口坐标是学生输入。默认仍不输入局部或全场障碍物信息；障碍掩码仅按物理协议屏蔽损失与定位输入的无效像素。
- max_slots=32 对齐教师 max_layers=32。高置信定位层不再被当作完整层集合：全部算术层都保留，只有物理兼容层进行定位蒸馏。
- 一对一匹配使用 GPU Hungarian；存在性正负分别归一化，掩码/强度/定位按实际层数归一化。
- 存在性直接决定层数分布，避免独立 count head 与实际槽位互相矛盾。
- 第1–3 epoch 学掩码、强度、存在性及响应/残差重建；从第4 epoch 加入逐层定位 KL。Phase1 前10 epoch 冻结参数但保留输入梯度；第11 epoch 小学习率联合微调。
- 每卡 micro-batch=2，目标全局 batch=128，按空闲卡数自动累积；BF16、TF32、GPU匹配和候选/编码器 activation checkpoint。所有训练使用兼容 PyTorch 2.4 的非 static_graph DDP。
- warmup epoch 不参加最佳模型选择。选择最佳模型与 patience 使用 post-warmup 的一致验证目标。
- 指定样本数是实际完成的教师窗口数；被跳过或教师未完成的窗口不计入配额。

真实源编号与可见性通道不进入推断。教师 complete 只说明算术剥离完成；索引额外校验每步等式，并分别记录单源物理定位资格。

## 新数据的真实设定

新 ZIP：`RIND-random-v3-100000scenes-multi-unit-256views.zip`

```text
SHA256 44015d37e6abb2093228380c237beeea1e71266b87640a5e8cdc05fe4a36d174
压缩文件 1,669,364,063 bytes
解压约 62.4 GiB
100,000 场景，每场景 256 个随机视窗，共 25,600,000 视窗
光源强度固定为 1；视窗 size 为 64、128、256、512
```

压缩包实际为单位强度，与本次要求不符。安装器会保留原始发布，使用固定 seed=20261028 为每个有效源重新独立采样 (0,1) 强度，建立 `phase2-random-strength-100k` 派生目录。响应直接按每源二值可见性重新加权求和，无需重新渲染几何。原始单位强度参考解被停用；新配置的教师分层和位置概率重新计算。

正式训练的新旧两个分量均使用随机强度。新视窗仍来自原包的随机选择，旧视窗来自四分树。已逐元素核实新前10,000场景的障碍几何、源位置和源数与旧数据一致；不能独立重随机 split。

拼接采用虚拟索引，原数组保留。旧配置 ID 0–9999 不变，新配置 ID 10000–109999，总计110,000个场景配置、100,000组独立几何。相同几何强制同split，旧split原样继承；新几何按固定seed分配80/10/10。

## 上传、安装与拼接后的训练

本机连接服务器需要 VPN。连接可用后在本机执行包内提供的 `upload_rind_update.sh`，其 rsync 支持中断续传，并校验新数据 ZIP 的 SHA256。脚本把代码 tar 上传到 `/home/qq/yiyu`，把 ZIP 放进最小包根目录，然后覆盖解压代码；不会删除服务器数据或旧输出。

服务器上执行一条后台命令：

```bash
cd /home/qq/yiyu/RIND_Phase2_Training_Minimal
./run_decomposition_union_tmux.sh start bootstrap --gpus 8
```

bootstrap 顺序：安装并校验新 ZIP → 生成固定种子的随机强度派生数据 → 建立防几何泄漏的虚拟拼接 → 复用旧教师、渲染新配置教师 → 建立完整层缓存 → 新实验训练。默认会话为 `rind_decomposition_union`。

```bash
./run_decomposition_union_tmux.sh status
./run_decomposition_union_tmux.sh logs
./run_decomposition_union_tmux.sh start evaluate --gpus 8
```

union 参数位于 `configs/decomposition_union.json`，继承原参数，仅改变数据/教师/输出目录并增加新教师配额。第一轮默认新完成教师 train=100000、validation=10000、test=10000，此外保留原教师。

第一轮渲染 size64/128，每场景最多抽8个视窗。它使用完整拼接数据作为抽样总体，并未穷尽2560万视窗。可调整 `new_teacher.record_counts`、`window_sizes`、`max_views_per_scene`；修改选样或物理设置应使用新的教师输出目录，防止误复用。

新教师目录：`outputs/decomposition_teachers_union`。
新训练目录：`outputs/response_decomposition_union_v2`。
仅续训该实验执行：`./run_decomposition_union_tmux.sh start train --gpus 8`。

正式训练不会使用原始单位强度响应。随机强度下仍须报告物理兼容层覆盖率，不能把所有完整算术层都称作真实单源。

普通 tmux 在断开 SSH/VPN 后继续工作；服务器重启、管理员终止进程或任务自身报错仍会停止。完整日志和退出码可用于判断是否成功。
