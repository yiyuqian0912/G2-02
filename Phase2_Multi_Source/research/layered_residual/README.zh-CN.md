# 逐层残差搜索原型

本目录实现固定顺序：最低正残差整体贡献 → 符号状态 BFS 得到当前层二值响应 → 按当前层响应剥离 → 重新分区；全部层恢复完成后，只进行一次全场均匀候选可见性采样，并用同一批候选分别给各层分配 Phase1 概率。它不读取真实源信息决定传播，也不联合恢复全部源。

完成逐层剥离后，教师只遍历一次统一全场候选网格。每个候选的单位可见性只渲染一次；同一个可见性掩码先接受共享阴影筛选，再同时与所有层的二值响应比较。对于原始观测中“自由像素且总响应为 0”的位置，正强度加性模型保证任意真实源都必须不可见；因此，任何会照亮至少一个这类像素的候选位置都被永久排除。障碍物内部的零值不参与该约束。筛选只依赖原始观测和教师侧正向渲染，不读取真实源标签。

服务器配置把候选端的整条批量流水线放到 CUDA：候选障碍物合法性、二值可见性、原始阴影筛选、所有层的加权响应代价和温度 softmax 都在 GPU 上完成，只把最终合法性、代价和概率小数组传回主进程保存。多个 GPU 同时处理不同真实视窗，每张卡内部再批量处理候选。CPU 只负责读取数据、逐层区域图控制和文件写入，不再遍历候选掩码或逐候选计算代价。每个窗口最先渲染的一个候选仍由原 CPU 渲染器复核一次；一旦不一致就立即停止。没有 CUDA 时，明确选择 `backend=auto` 可继续使用原 Numba/CPU 路径。

## 算法

每轮只在自由像素中选择严格大于容差的最低正残差 `a`。响应为 0 的阴影／背景区域不是贡献层，不参与最低值和种子选择；但它仍作为当前层响应为 0 的观测约束参与 Phase1 候选评分。同值不连通区域保留不同 ID，种子取行优先最早的一个。BFS 只维护“当前层是否仍然存在”，不恢复其他贡献的标签集合：

```text
residual = observation
shadow = free pixels where original observation <= tolerance
while residual has positive free pixels:
    rebuild equal-value connected regions and adjacency graph
    a, seed = lowest positive value and deterministic seed region
    queue = [(seed, target(a))]
    while queue:
        rising/equal edge: current layer remains
        falling edge with drop == a: current layer exits
        falling edge ending below a: current layer must exit
        other falling edge: attribute the drop to other contributions; current layer remains
        target removed: mark destination negative and stop this branch
        target retained: continue this branch
        inconsistent paths: mark unknown
    layer_response = pixels in BFS-positive regions
    keep disconnected equal-valued regions separate for later rounds
    subtract a * layer_response; stop if this would make residual negative
if every layer was recovered and residual is zero:
    for each geometrically valid candidate on one uniform full-field grid:
        render its unit visibility exactly once
        reject it if the visibility is positive anywhere in shadow
        compare the retained visibility with every recovered layer_response
    independently normalize each layer's candidate costs with the Phase1 temperature rule
combined_probability(s) = (1 / number_of_layers) * sum_t layer_probability_t(s)
```

途中其他贡献只通过边的净增减量被记录在轨迹中，不建立身份标签，也不枚举它们的退出组合。若下降量等于当前层强度，按当前层退出处理；否则只要目标区域仍能容纳强度 `a`，按其他贡献退出处理。不同路径若分别得到“存在”和“退出”，该区域为冲突未知。未访问区域同样为未知，不当作负约束。精确同强度的其他贡献退出会与当前层退出混淆，这是显式保留的启发式误差来源。

单个种子无法凭 BFS 穿过“当前层已退出”的区域。按照当前算法约定，无歧义的 BFS 正区域组成这一轮的完整二值层响应；未访问、负状态和路径冲突区域不并入这一层，继续保留在残差中，之后单独处理。这与“不相邻的同强度区域分开处理”采用同一规则。冲突数量写入报告，供评估这一确定性选择的影响。

Phase1 教师不要求候选渲染与层响应完全一致，也不要求只有一种完整物理掩码。它像 Phase1 原流程一样计算候选响应与每层响应的物理代价，再通过温度归一化得到每层全场概率。物理渲染只发生在一次共享的候选遍历中；不同层仍必须分别计算代价和 softmax，因为它们的目标掩码不同。零误差候选数量仅作为诊断记录，不作为接受或拒绝视窗的门槛。若逐层剥离本身失败，则不会启动昂贵的全场候选渲染。

最低整体贡献大于 1 时仍可作为有效贡献用单位掩码定位和剥离，但记录 `effective_above_physical_source_limit=true`；它不被称为一个合法物理源，也不会被拆成多个强度。

## Phase1 复用

- 候选：`rind_phase1.search.CandidateDomain` 与 `uniform_grid`，全局半网格中心、窗口外候选、等网格点概率质量。每个视窗只生成并遍历一次全场网格，基于原始零响应阴影执行共享筛选，所有层复用同一批候选可见性与同一个候选支持。
- 物理评分：Phase1 的 `boundary_weight_map`、`response_cost` 和可选 `edge_cost`；Phase2 教师侧渲染只提供二值单源可见性和障碍物合法性。
- 概率：`rind_phase1.teacher.teacher_probabilities`，`q ∝ exp(-cost/temperature)`，每层离散质量和为 1。
- 可视化：与 Phase1 相同的世界坐标、y 轴向下、概率色图、灰色无支持位置、青色窗口框和色条。所有层使用同一候选网格，因此可以逐点进行等权混合。

整合图使用等权层混合：`P_combined(s)=(1/r) sum_t P_t(s)`。由于每层 `P_t` 的质量和都是 1，每层对最终标签严格贡献 `1/r` 的概率质量。它表示“从恢复出的有效贡献中等概率选一层，再从该层的位置分布采样”，适合继续使用 Phase1 的单输出 softmax 学生。它不是“该位置至少存在一个源”的占用概率，也不是完整多源联合后验。部分失败时，图只使用已成功层并在报告标记 `partial_combined_distribution=true`。

## 运行

从 Phase2 目录运行：

```bash
./run_layered_residual.sh \
  --scene-id 39 --view-id 8 \
  --spacing 16 --temperature 0.01 \
  --max-layers 8 \
  --joint-samples 1000 \
  --output outputs/layered_residual/scene39-view8
```

`--joint-samples 0` 关闭对照。大于零时，使用现有 Phase2 联合 Monte Carlo，在完全相同的有效位置池、恢复出的有效贡献数下抽样；其强度仍是现有实现的 iid Uniform[0,1)，所以只比较实际候选评估数、时间和最低 MAE，不把它称为穷举或同一个后验。

Phase1 可调参数直接暴露：`--alpha`、`--beta`、`--boundary-lambda`、`--boundary-sigma`、`--temperature`、`--spacing`。第一版默认保持 Phase1 baseline：`alpha=1, beta=0, boundary_lambda=0`。

## 生成完整教师答案并训练学生

学生不接收逐层残差、层编号或逐层掩码。教师先离线完成所有层的搜索和剥离，再用一次共享的全场候选遍历给所有层分配 Phase1 概率，最后把成功层等权混合成一张全场概率图。训练记录只有一个原始局部观测和一张全场教师答案，接口与 Phase1 的单输出能量模型一致。

配置文件为 `configs/layered_combined_train.json`。其中：

- `scene_counts` 控制按 `scene_id` 隔离的训练、验证、测试场景池大小；它只决定可扫描的独立场景范围，不再被当作样本数；
- `record_counts` 可以是正整数或字符串 `"all"`。例如 `train=2000` 表示 `records.json` 中必须有 2000 条可训练记录；`train="all"` 表示扫描该 split 所有符合 `window_sizes` 的真实窗口，并保留过滤后教师成功的全部记录；
- `window_sizes` 控制允许扫描的窗口规格。各规格的候选池独立打乱后轮流扫描，避免只由某一种尺寸先填满配额；
- `teacher` 控制逐层教师的候选间距、温度、损失和最多层数；
- `teacher.shadow_screening=true` 启用严格的原始阴影候选筛选；当前无噪声、正强度模型建议保持开启；
- `teacher.backend=cuda` 启用 GPU 候选流水线；`gpu_batch_size` 控制每卡每批候选数，`gpu_pixel_chunk` 控制每次处理的窗口自由像素数，`gpu_verify_candidates` 控制每个窗口用 CPU 复核的候选数；实际卡数通过运行命令的 `--gpus N` 指定；
- `teacher.save_plots=false` 不为每条教师记录自动生成 PNG；全量服务器实验应关闭，否则图片会占用大部分磁盘，训练目标和数值记录仍完整保存；
- `training` 控制 Phase1 `SourceEnergyField` 的网络宽度、Fourier 特征、批量、学习率和早停；
- `combined_rule` 必须是 `normalized_elementwise_max`。各层先分别归一化，随后逐点取最大值并对结果重新归一化。

学生训练接口与 Phase1 对齐。正式大模型配置使用 spacing=8 全场候选网格，每个视窗最多有 16,384 个候选，是 spacing=16 的四倍；有限网格不要求存在零误差候选。Fourier 频率设为 7，最高频率在该网格上仍满足每周期至少 4 个采样点。

| 参数 | Phase1 | Phase2 |
|---|---:|---:|
| candidate spacing | 64 | 8 |
| teacher temperature | 0.05 | 0.05 |
| `alpha / beta` | 1.0 / 0.0 | 1.0 / 0.0 |
| `boundary_lambda / boundary_sigma` | 2.0 / 1.0 | 2.0 / 1.0 |
| epochs / patience | 30 / 8 | 150 / 25 |
| batch size | 4 | 96（六卡，每卡约 16） |
| AdamW learning rate / weight decay | 0.001 / 0.0001 | 0.0004 / 0.0001 |
| hidden / Fourier frequencies | 64 / 4 | 256 / 7 |
| candidate chunk | 256 | 512 |
| student obstacle input | false | false |
| pixel-coordinate channels | true | true |
| student normalization support | world | world |

`world` 表示学生在完整的 world-minus-window 候选集合上归一化，不使用隐藏障碍物几何替学生排除位置；教师仍使用几何判断合法候选并令非法位置的目标质量为零。训练现在与 Phase1 一样保存 `last.pt` 的优化器和随机数状态，再次运行相同命令会校验协议后从该 checkpoint 继续。

准备教师记录：

```bash
./run_combined_training.sh prepare --config configs/layered_combined_train.json
```

训练：

```bash
./run_combined_training.sh train --config configs/layered_combined_train.json
```

也可以连续执行：

```bash
./run_combined_training.sh all --config configs/layered_combined_train.json
```

本机快速检查使用相同学生模型和优化器参数，只缩小场景数量并只取 32×32 窗口。`split_seed=19` 是在不查看教师代价和真源的前提下，根据窗口是否为全亮／全暗选择的可运行 smoke 划分；真实检查得到 2 个训练、1 个验证和 1 个测试教师记录：

```bash
./run_combined_training.sh all --config configs/layered_combined_smoke.json
```

窗口筛选只检查自由像素：自由像素全部大于容差的全亮窗口、全部不大于容差的全暗窗口，以及没有自由像素的窗口都不会生成训练记录；教师失败的窗口同样不进入训练集合。整数 `record_counts` 会继续扫描直到收集到精确目标；`"all"` 会耗尽该 split 的全部真实窗口。

`"all"` 不是把原始窗口总数直接当成训练样本数。每个 split 开始时，日志会显示 `raw windows` 及各 size 的数量；扫描日志显示 `raw checked/总数`、`eligible seen` 和 `completed training records`；split 结束时才得到实际进入学生训练的精确记录数。`records.json` 的 `counts` 保存最终训练／验证／测试记录数，`split_statistics` 保存各 split 的原始总数、实际检查数、合格数、教师未完成数和跳过数，`skip_counts` 提供全局跳过汇总。

输出位于配置的 `output_root`：教师书在 `teachers/`，划分和跳过原因在 `records.json`，训练协议在 `protocol.json`，模型在 `models/seed_*/best.pt`，训练统计在 `training_summary.json`。

终端会持续显示以下进度：当前数据划分、已检查窗口数、全亮／全暗跳过数、正在计算的 scene/view、教师完成状态与层数，以及每个 epoch 内的 batch、running CE、验证 CE、早停计数和耗时。相同信息以一行一个 JSON 事件写入实验目录的 `progress.jsonl`，中断后也能检查。`training.progress_every_batches` 控制每隔多少个 batch 打印一次；正式配置默认 10，smoke 配置默认 1。

另开终端实时查看日志：

```bash
tail -f outputs/experiments/layered_combined_2000_v4_one_pass/progress.jsonl
```

## 服务器全量训练

全量配置为 `configs/layered_combined_server_full.json`：按 scene_id 固定划分 8000/1000/1000 个场景，扫描 16、32、64、128 四种窗口的全部真实记录，过滤后保留所有成功教师。原始扫描总体分别为 235145、28683、28233 个窗口；最终训练记录数由真实过滤与教师完成状态决定，写入 `records.json`，不会用原始窗口数冒充样本数。配置关闭逐记录 PNG，使用 CUDA、六卡总 batch size 96、150 epochs 和 patience 25。教师阶段每卡独立处理一个视窗；学生阶段使用 PyTorch DataParallel 将 batch 分到同一组 GPU。checkpoint 始终保存基础模型参数，不带多卡包装前缀。

正式学生为可配置的 `large` 架构：256 维隐藏特征、深层残差卷积观察编码器、三次下采样、8 头候选注意力，以及加深的窗口、坐标和能量网络。`baseline` 仍保留 Phase1 原结构并可严格加载旧 checkpoint。当前 `record_counts` 已是三个 split 的 `"all"`，所以数据量的增加来自更密的物理监督以及全部合格真实窗口，不通过复制记录增加表面样本量。

服务器准备环境和安装数据：

```bash
cd Phase2_Multi_Source
uv sync --locked --extra train --extra diagnostics --extra browser
./scripts/install_data.sh RIND-adaptive-v3-10000scenes.zip
```

推荐把 GPU 教师准备和 GPU 学生训练分开执行，并放在 `tmux` 中：

```bash
tmux new -s rind-phase2
./run_server_full.sh prepare --gpus 6
./run_server_full.sh train --gpus 6
```

也可以连续运行：

```bash
./run_server_full.sh all --gpus 6
```

实时查看进度：

```bash
tail -f outputs/experiments/layered_combined_h20_v2/progress.jsonl
```

中断后重新运行同一命令即可。教师阶段复用已完成的 scene/view 目录；训练阶段从 `last.pt` 恢复优化器、epoch 和随机数状态。正式配置针对最大 128×128 视窗使用 `gpu_batch_size=16384`、`gpu_pair_budget=134217728`（按自由像素数自适应，OOM 自动最低退到 1024） 和 `gpu_pixel_chunk=16384`。教师显存不足时会自动将候选批量从当前自适应批量依次减半，最低 1024；学生显存不足时把总 `training.batch_size` 从 96 依次降到 48、24。`--gpus N`、批量和像素块都进入实验协议，恢复时必须使用相同参数。

对测试集第 0 个窗口生成 Phase1 风格的教师／学生对比，并额外查询逐像素学生分布：

```bash
./visualize_combined.sh \
  --config configs/layered_combined_train.json \
  --seed 0 --index 0 --dense-spacing 1
```

默认不绘制真源；加 `--show-truth` 后，真源只作为评估标注。图和数值数组保存在相应模型目录的 `visuals/test_0/`。

同一命令还会生成教师对该局部视窗的完整逐层剥离图：

- `teacher_peeling_page_XX.png`：每页最多六层，每层依次显示剥离前残差、当前层二值响应、实际减去的贡献和剥离后残差；
- `teacher_layers/layer_XX_peeling.png`：单独查看某一层的高分辨率版本；
- `teacher_probability_layers_page_XX.png`：整合教师以及每一层通过 Phase1 得到的全场位置概率；
- `teacher_layers.json`：每层强度、最低 Phase1 代价、零误差网格候选数、路径冲突区域数及图片清单。

局部剥离图使用视窗对应的世界坐标。`Current layer response` 是严格的二值掩码：黑色为 0、白色为 1、蓝色为障碍物；蓝色不是响应的中间值。`Subtracted contribution` 严格等于“层强度 × 当前层响应”，并在绘图前检查 `residual_after = residual_before - contribution`，防止可视化与教师实际剥离不一致。

如果只完成了 `prepare`、还没有训练学生，可以直接查看测试记录中的教师剥离过程，不需要 checkpoint：

```bash
./visualize_combined.sh \
  --config configs/layered_combined_2000.json \
  --index 0 --teacher-only
```

此时图片写入 `outputs/experiments/layered_combined_2000_v4_one_pass/visuals/test_0/`。

## 输出

- `report.json`：完成状态、失败原因、逐层符号轨迹、冲突、阴影筛选前后候选数、节省的逐层评分次数、运行时间、最终 MAE、联合采样对照和事后真值诊断。
- `layer-XX.npz`：剥离前后残差、区域标签、正／负／未知约束、当前层响应、候选坐标、Phase1 代价及归一化概率。
- `combined.npz`：共同网格、各层等权混合概率、原始响应和最终残差。
- `probability.png`：第一幅为整合分布，其后为各层分布。
- `layers.png`：每层的残差、约束、验证掩码和剥离后残差。

真值通道只在所有推断结束后计算每层与单个生成源的最佳 IoU，位于 `truth_diagnostics_not_used_for_inference`。最低整体贡献没有对应单个真实源时，低 IoU 不判定分解失败。

## 目前成立范围

已由代码检查：每个被剥离的层响应来自 BFS 正区域；剥离不产生超过容差的负残差；原始自由零响应像素不会被保留候选照亮；Phase1 对全部筛选后候选计算物理代价；每层与整合概率归一化。

条件性成立：在边界满足嵌套，并且“下降量等于 `a` 当且仅当当前层退出”的条件下，传播符合目标规则。它没有证明实际边界一定满足该条件，也没有证明“最低值整体贡献”一定物理可实现。

经验性步骤：把一次上升记为一个匿名整体跳变量；把 BFS 正区域视为当前轮的完整层响应；未访问的同强度区域留给后续轮次。边界长度阈值尚未接入传播，也没有用 17 像素阈值过滤短边；阈值需要开发场景定参和独立场景验证。

尚未解决：目标贡献退出后在远处重现时是否应与当前层关联；同强度退出和混合切换造成的误判；把路径冲突区域留给后续轮次是否是最佳选择；有效贡献大于 1 时物理源数量后验。有限候选网格可能没有零误差位置，但这只会形成较高代价和较软的 Phase1 分布，不再导致整窗失败。

复杂度包含每轮重新分割与建图、BFS 路径冲突处理、共享候选渲染、逐层评分和残差检查。若有 `r` 层和 `n` 个合法候选，昂贵的物理部分是 `n` 次单源可见性渲染，与层数无关；得到这些掩码后仍需约 `n r` 次掩码—层代价比较及 `r` 次 softmax。实现将候选置于外层循环，在一次遍历中完成所有层比较。区域数量、每次几何渲染成本和歧义处理仍会变化，因此完整算法不简单宣称为 `O(n)`。

## 已运行的小规模真实检查

spacing=16、默认 Phase1 代价：

| scene/view | 生成 K（只作事后诊断） | 恢复有效层 | 状态 | 最终 MAE |
|---|---:|---:|---|---:|
| 993/11 | 1 | 1 | complete | 0 |
| 39/8 | 2 | 2 | complete | 0 |
| 132/3 | 3 | 2 | complete | 0 |
| 48/15 | 4 | 2 | complete | 0 |

这四个是接口与行为检查，不是总体完成率。K=3/4 只恢复两层符合“有效贡献不等于真实源数量”的定义。scene 39/view 8 的 200 个同池联合 Monte Carlo 候选最低 MAE 为 0.005792，而逐层重建为 0；样本规模太小，不能用于性能结论。

### H20 执行优化

正式配置为六张 96 GB H20 启用 16384 最大候选批量、134,217,728 pair budget 和 1024 最低自动降档。教师的二值可见性加权 L1 代价通过等价的 float64 GEMM 同时计算全部层；几何判断仍为 float64。学生使用 BF16 autocast、float32 概率归一化、fused AdamW、TF32、cuDNN benchmark 和总 batch size 96。为追求吞吐关闭 bitwise deterministic 慢路径，但 scene split、样本顺序种子和教师目标仍被记录。正式 H20 输出写入 `outputs/experiments/layered_combined_h20_v2/`，不与旧实现混合。

训练数据加载使用 8 个有序预取线程，在当前 CUDA batch 运行时读取和解压下一批 `combined.npz`。教师状态及全亮/全暗检查只在训练集合初始化时验证一次，不再在每个 epoch 重复读取和重算。
