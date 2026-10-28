# 10 万场景多尺度单源训练

本实验仍训练与 Phase1 相同的单源条件能量模型。Phase2 多源窗口先由残差方法拆成二值层，再调用这个模型；本次没有把多层混合目标重新放回学生训练。

## 本版模型

- **多尺度边界编码**：从二值响应计算原始、2×、4×尺度的 Sobel 边界；这些量只由学生可见的局部响应产生。Sobel 使用复制边界，不会把视窗裁剪边缘误当成亮暗边界。
- **显式边界分支**：边界特征保留到较高分辨率，区域特征在 1/4 尺度编码，全局特征在 1/8 尺度编码。
- **注意力分工**：总计固定 8 个头，其中 4 个只读取边界 token，2 个读取区域 token，2 个读取全局 token；三组使用独立 Q/K/V，并由候选相关门控融合。配置若不是 8 个头会直接报错，模型中不再保留未使用的旧 attention 参数。
- **候选相对几何**：候选查询包含相对窗口中心的 Fourier 特征，以及到窗口左、右、上、下边界的有符号距离；此外，三组注意力都使用二维 RoPE，将候选全场坐标与每个观察 token 的全场坐标旋转到 Q/K 中，使注意力分数显式依赖 `candidate_position - token_position`。这种实现不构造候选×像素×特征的巨大张量。
- **几何不可见主设定**：`use_obstacle=false`，学生不知道全场障碍物；隐藏几何仅供 CUDA 教师重渲染和候选合法性计算。

## 数据与重新渲染

配置读取 `data/raw/phase1-single-source-100k/`，并使用固定的 80,000/10,000/10,000 scene 划分。每个场景各取一个 16、32、64、128 窗口，因此过滤前最多为：

- train：320,000 个窗口；
- validation：40,000 个窗口；
- test：40,000 个窗口。

无自由像素、全暗、全亮窗口不会进入教师记录。`teachers.json` 会保存实际成功记录数和跳过清单，训练使用的是真实保留数量。

新数据使用新的 `data_root`、split 清单、输出目录和协议哈希。新版场景仍需重新执行 CUDA 物理重渲染；旧缓存通过下面两种受检路径追加，不会把不同场景的数组误认成新版场景缓存。

## 空服务器运行

把代码包和 `RIND-adaptive-v3-100000scenes-single.zip` 放到同一目录，解压后执行：

```bash
./setup.sh
./install_100k_data.sh /path/to/RIND-adaptive-v3-100000scenes-single.zip

# 先用独立输出做 100/20/20 scene 的端到端接口检查
./run_phase1_100k.sh smoke --gpus 8

# 推荐分阶段运行；每一步均可用同一命令续跑
./run_phase1_100k.sh prepare --gpus 8
./run_phase1_100k.sh train --gpus 8
./run_phase1_100k.sh evaluate --gpus 8
```

如果 `/home` 是网络盘，最快方式是把数据和输出都放到服务器本地 NVMe。首次运行会
复制一次数据集，之后复用同一副本；教师分片、checkpoint 和日志也都写入高速盘：

```bash
./run_phase1_100k.sh prepare --gpus 8 --scratch-root /local_nvme/$USER/rind
./run_phase1_100k.sh train --gpus 8 --scratch-root /local_nvme/$USER/rind
./run_phase1_100k.sh evaluate --gpus 8 --scratch-root /local_nvme/$USER/rind
```

将 `/local_nvme` 换成服务器实际的本地 SSD/NVMe 挂载点，例如 `/scratch`、`/nvme`
或 `/local`。三个阶段必须使用同一个 `--scratch-root`。

`setup.sh` 会在包目录内安装 uv、Python 3.12 和 CUDA PyTorch，不需要 sudo 或 snap。启动器会依次使用 `RIND_PYTHON`、当前激活的 `VIRTUAL_ENV`、Phase1 内的 `.venv`、包根目录的 `.venv`。

若服务器上保留了旧 1 万场景 Phase1 教师，可在 prepare 时追加：

```bash
./run_phase1_100k.sh prepare --gpus 8 \
  --append-cache /path/to/old_experiment/response
```

路径也可以指向包含 `response/teachers/` 的旧实验根目录。程序按 `scene_id/view_id` 查找旧记录，并逐条核验旧、新数据的窗口响应、障碍掩码、完整障碍几何、候选网格、温度和物理代价设置。完全相同才导入；网格间距或任何物理参数不同的记录会被拒绝并在新 GPU 上重渲染。导入不是软链接：验证后的记录会写入新实验目录，随后继续 append 新场景。`teachers.json` 中分别记录 `reused`、`imported`、`rendered` 和拒绝原因。

服务器原来的 `Phase2_Multi_Source/outputs/full_training_h20_v2` 属于另一类缓存：其中每个已完成残差层本身就是一个二值单贡献反演样本。它不替代新版 Phase1 场景的教师渲染，而是在训练阶段追加到训练集：

```bash
./run_phase1_100k.sh train --gpus 8 \
  --append-layer-cache /path/to/Phase2_Multi_Source/outputs/full_training_h20_v2
```

默认配置已经包含同一最小包内的相对路径 `../Phase2_Multi_Source/outputs/full_training_h20_v2`；缓存位于该位置时不用额外传参。程序只读取旧缓存的 `train` split，并检查完整状态、二值层非恒定、概率归一化、最低物理代价、温度及 `alpha/beta/boundary` 设置。旧 validation/test 不会追加，所以新版 validation/test 仍然独立。

旧层若使用 spacing=8、新教师使用 spacing=16，会按“窗口大小 + 候选数”进入不同 batch；两种离散分布均保持各自归一化，不会为了组成 batch 相互填充。旧缓存为 `beta=0` 时只追加给 response 分支；edge 分支会报告参数不兼容并仅使用新版数据。首次训练会生成 `response/appended_layer_cache.json`，记录实际接受数量、来源和拒绝计数。改变追加缓存会改变断点签名，程序不会误续跑旧 checkpoint。

也可以一次运行：

```bash
./run_phase1_100k.sh all --gpus 8
```

`--gpus 8` 使用当前进程可见的 GPU 0–7。若服务器用 `CUDA_VISIBLE_DEVICES` 重新编号，只填写可见卡数。

## H20 设置

默认配置针对 8×H20 96GB：八张卡各保留一个物理渲染槽。候选合法性、逐像素
可见性、边界距离权重、物理代价和教师 softmax 归一化全部由 CUDA 完成，过程中
不把候选批次搬回 CPU。每个窗口完成后只做一次最终 GPU→主机传输，用于持久化
教师记录。

正式配置设置 `record_storage=memmap`。教师不再为每个窗口创建一个 NPZ，而是按
`split × window size` 预分配连续 NPY memmap 分片。候选坐标由窗口和固定网格确定，
读取时重建，不在每条记录中重复保存。正式规模由约 40 万个小文件降至几十个连续
数组，显著减少文件打开、目录元数据和压缩开销。每条记录以独立完成标志提交，
Ctrl-C 后可从未完成 slot 继续。float32 代价、概率和布尔有效掩码预计约需 15–25 GB，
具体取决于过滤后的窗口数量。旧逐窗口 NPZ 仍可作为受检导入源。

每卡 2 层流水线让下一窗口的 GPU 渲染和上一批连续写盘重叠。CPU 只保留无法消除
的数据读取、位图存储格式解码、任务调度、完整性检查和写盘，不参与批量物理计算。

配置中的 `gpu_batch_size=8192` 是每个窗口的候选 batch 上限。spacing=16 时全场
候选总数约为 4096，所以一个窗口本来就只提交一次候选渲染；把上限从 8192
继续增大不会减少 kernel 次数。旧日志每完成 32 条记录打印一次，其中的 32 只是
日志间隔，不是渲染 batch。本版将日志间隔明确设为 128，并在启动日志中分别打印
`render slots`、`pipeline workers` 和 `configured candidate batch`；每条教师记录还保存
实际候选数和实际提交 batch。

学生使用全局 batch 768、BF16、fused AdamW、TF32、cuDNN benchmark、
channels-last、16 批预取和 PyTorch DataParallel。模型包含 20,030,020 个可训练参数；
审计确认没有未接入前向图的可训练参数。教师索引直接读取 `teachers.json`，不会为
建立索引逐个读取分片；第一轮把实际训练字段缓存到主机内存，后续 epoch 不再重复
读取，预计需要数十 GiB 主机内存。先运行 `smoke`；它使用独立输出目录，不污染正式
教师。正式教师开始生成后不要原地修改 batch 或缓存策略，因为冻结协议会正确拒绝
混用两套实验配置。

## 输出与恢复

输出根目录：

```text
Phase1_Single_Source_Single_Intensity/outputs/experiments/phase1_100k_multiscale_h20_memmap/
```

- `protocol.json`：冻结的数据、代码和窗口选择；
- `response/teacher_shards/*/*.npy`：按 split 和视窗大小组织的连续教师数组；
- `response/teachers.json`：教师统计、跳过清单与哈希；
- `response/appended_layer_cache.json`：旧 Phase2 逐层训练样本索引与验收统计；
- `response/seed_0/last.pt`：最近 checkpoint；
- `response/seed_0/best.pt`：验证 CE 最优 checkpoint；
- `response/seed_0/training.json`：逐 epoch 训练曲线及新版/追加/总训练记录数；
- `response/seed_0/test_metrics.json`：训练结束后的测试指标。

Ctrl-C 后重新执行相同 stage 命令。教师会复用分片中已提交的 slot；学生从 `last.pt`
继续。使用 `--scratch-root` 时恢复命令必须继续给出同一路径。不要更改已经开始运行
的配置；要改变实验定义时改 `name` 和 `output_root`。
