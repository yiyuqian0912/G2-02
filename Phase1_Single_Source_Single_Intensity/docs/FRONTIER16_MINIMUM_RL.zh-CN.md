# 16×16逐块扩展与最小值融合

2026-11-06。本版是独立选路实验：复用并冻结已有Phase1单源定位器，每次只新增一个16×16小窗，再训练PPO选择下一个小窗。没有重训Phase1，也没有给学生输入全场障碍物或未来响应。

## 全量训练配置（新增推荐入口）

`configs/active_rl_frontier16_minimum_full.json`将三个scene_counts设置为`all`。它从冻结Phase1权重继承的独立划分中取全部可用场景，不把验证/测试并入训练。100k单源发布包预期为：

- 训练80,000个场景，验证10,000个场景；
- 测试10,000减去上轮已审阅的128个场景及同几何组，预期9,872；最终以启动日志和protocol的实际数字为准；
- 定期选优仍只用预先固定的128个验证场景；最佳策略最后评价完整验证集和测试集。

```bash
./run_single_source_rl_tmux.sh start all \
  --run-name single_source_rl_frontier16_minimum_full \
  --config configs/active_rl_frontier16_minimum_full.json \
  --gpus auto

./run_single_source_rl_tmux.sh status \
  --run-name single_source_rl_frontier16_minimum_full
```

使用原100k定位器及原始单源数据的默认路径。仅全量训练、暂不启动最终评价时将`start all`换成`start train`；之后可`start evaluate`。tmux断网后继续运行。旧1万场景配置和输出保留，新全量实验使用上面的独立名称。

`minimum_scene_visits=1`要求每个训练场景至少完成一条观测轨迹。轨迹可以耗尽预算或由策略主动停止；这不表示穷举每场景所有16×16块，更不表示每个场景都可准确定位。场景先打乱排队，优先调度尚未达到目标的场景，不用重复轨迹冒充新增独立场景。

`updates=500`在全量模式中是**最低更新次数，不是上限**。训练同时满足“更新不少于500”和“全部训练场景达到访问目标”才结束；若500次后还没覆盖完，会继续训练并显示真实进度。`best.pt`只从达到全覆盖目标后的权重中选择；此前持续保存`last.pt`用于恢复。

日志和`training.json.training_scene_coverage`明确保存：

```text
training_scenes                 全部训练场景数
started_unique_scenes           已开始观测的独立场景数
completed_unique_scenes         至少完成一条轨迹的独立场景数
scenes_reaching_target          已达到目标访问次数的场景数
minimum_completed_visits        全部场景中最少的完成次数
coverage_complete              是否已完成覆盖目标
```

未完成轨迹不会计为完成，恢复时重新开始；完成次数不会因重新启动清零。自动选到的GPU数量变化时，覆盖记录按scene_id重新分配。覆盖完成前的定期验证分数仅供监控；这些权重不冒充全量最佳模型。

全量续训：

```bash
./run_single_source_rl_tmux.sh start train \
  --run-name single_source_rl_frontier16_minimum_full --gpus auto
```

想让每个训练场景至少完成两条轨迹，可额外指定`--set minimum_scene_visits=2`；恢复保留已完成次数。增加的访问目标和更新次数同时存入断点和`training_schedule.json`，下次不重复参数也不会回退。恢复优先使用`last.pt`；若它缺失，使用已有`best.pt`，只恢复到该断点保存的进度。两者都没有时才重新开始，不覆盖已有的`initial.pt`。

同一实验目录使用进程锁，防止两个训练/评估任务并发写入；进程退出后锁自动释放，无需删除锁文件。多卡准备、保存和汇总阶段的主进程异常会同步给其他进程。完整测试结果仅在主进程汇总，并流式写入JSON，避免每张卡持有全部结果或创建大型JSON字符串副本；逐场景记录本身仍需内存。

本地已用旧小定位器和小预算跑通`all`训练分片的2,500个真实场景，覆盖达到2,500/2,500后结束；另已验证双进程训练及恢复为单进程后累计到每场景两次。这些是流程验证，不是服务器80,000场景的训练效果或H20性能结果。

注意：此处“全量”是100k**单源**发布包。另一个100k多源扩展包用于Phase2，不能直接将其多源总响应当作当前单源RL输入。单源定位权重继续复用，不重新渲染位置教师。

## 观测顺序

1. 从一个起点所属的16×16网格块开始，首次花费256像素预算。
2. 单独调用Phase1，得到该块在全场候选网格上的位置概率场。
3. 在全部已观测块的四邻接边缘中，选一个尚未观测的新块。动作可以出现在任一分支的边缘，不局限于上一块的四邻居；不截断候选边缘。
4. 新增块只调用一次定位器，并将其预测加入最小值融合。
5. 重复，直到预算耗尽、步数上限或策略主动停止。重复块禁止采集、禁止重复融合。

区域可形成L形、条带及分支，定位器始终只接收完整16×16响应。这里的路径是传感器ROI查询顺序，不代表机器人已经能够跨障碍通行；本版没有机械移动成本。

指定起始像素时，首次读取包含它的对齐16×16块，不是只测量一个像素。默认起点随机且独立于源；训练每次重采样，评估使用协议中固定的起点。

## 最小值究竟怎样融合

所有小窗使用同一个spacing=16的全场网格，在1024场景中为4,096个候选点。每个有效小窗单独输出已归一化的概率质量P_i。

\[
 m_t(s)=\min_{i\le t}P_i(s),\qquad
 P_t(s)=\frac{m_t(s)}{\sum_u m_t(u)}.
\]

实现保留**原始小窗预测的逐点最小值**，而不是把前一次归一化结果继续与新场取min。这使相同观测集合的融合结果与到达顺序无关。

计算在log空间进行：`log_min=min(log_min, log_P_i)`，展示时`softmax(log_min)`，避免小概率相乘/存储时的下溢。最小值版本不使用`fusion_discount`，该兼容参数设为1。

- 若输入某位置是明确的零（log概率为负无穷），后续窗口不能将其恢复。
- 很小的正概率仍是正概率，不自动阈值化。图中黑色也不自动等于硬排除。当前Phase1 softmax没有独立的“物理不可能”判定头。
- 若明确零值掩码的交集为空，返回`FusionConflictError`，不改成均匀分布掩盖冲突。当前有限能量softmax通常不会产生这种真正空交集。
- 全亮/全暗块仍然收费、进入覆盖掩码，但本版按中性证据处理，不让均匀分布截平已有预测峰；直到出现有效非恒定响应前，保持均匀先验。
- 归一化会提高一些位置的相对概率，但没有额外的正向累乘奖励。

例如三点空间中P1=(0.8,0.1,0.1)、P2=(0.4,0.3,0.3)、P3=(0.2,0.7,0.1)：最终原始min=(0.2,0.1,0.1)，归一化为(0.5,0.25,0.25)。若错误地在每轮对归一化后的结果取min，会得到不同答案。

这是“各窗口共同支持程度”的共识分布，不是经证明的多视窗贝叶斯后验。某个错误且过度自信的小窗也可能压低真源位置，所以必须继续评价真源质量和错误集中，不能只看场变尖。

## 训练目标和选路模型

冻结定位器仍是服务器100k Phase1的`response/seed_0/best.pt`（上轮最佳epoch18）。本版只训练新选路器，旧64块PPO权重与新空间动作头不兼容，不自动加载；已有Phase1权重完整复用。

新选路器以16×16块为分辨率，保留空间特征，直接给整个网格的每个块输出动作分数，再屏蔽不在四邻接边缘的位置。1024场景的动作数为4,096个块加停止；没有构造`候选动作数×隐藏宽度`的大型拼接张量。输入只有已观测响应均值、覆盖、当前融合概率场、最后一块位置、坐标、预算与步数。它没有读取待选块的响应。

奖励仍为：

\[
 r_t=\Delta[\tfrac12\log(M_{16}+10^{-8})+\tfrac12\log(M_{32}+10^{-8})].
\]

这里的M来自**最小值融合后的场**。真源仅用于训练奖励、离线评价和可视化标注；Actor与Critic都不接收真源。位置熵不进入奖励；PPO动作熵用于探索。

## 先前1万场景配置与预算

配置：`configs/active_rl_frontier16_minimum.json`。

| 参数 | 默认值与含义 |
|---|---|
| 独立场景 | train=10,000，validation=1,000，test=1,000；分别继承Phase1的三个划分 |
| 测试隔离 | 排除上一轮已审阅的128个测试场景及其相同几何组 |
| 单块 | 16×16；初始块和之后每块都收256像素 |
| 训练预算 | 每次轨迹在4,096、16,384、65,536中抽取，分别是16、64、256块 |
| 测试预算 | 同上，分别占全场0.39%、1.56%、6.25%；与旧25%预算不能直接比较 |
| 最大动作数 | 256；足够花完默认最大预算 |
| PPO | 每卡32个环境，每次64步，500次更新；每卡1,024,000条交互，总量乘实际GPU进程数 |
| 验证选优 | 预先固定128个验证场景，每25次更新检查；最终最佳权重另外评估完整1,000个验证场景 |
| 测试 | 完整1,000个场景、三档预算；不参与选优 |
| 对照 | 随机边缘、紧凑扩展、沿已观测响应变化扩展、训练前策略；共用相同动作/预算/定位/融合 |

`compact`优先选择离起点最近的边缘块；`boundary`使用相邻已观测块的响应混合程度与差异，信息不足时采用紧凑扩展。两者都不预读未来响应，也不读取真实几何。

## 服务器命令：tmux后台训练

本节保留先前1万场景命令；全量请使用文档开头的`_full`配置与实验名。

在`RIND_Phase2_Training_Minimal`根目录运行（沿用已有.venv）：

```bash
./run_single_source_rl_tmux.sh start all \
  --run-name single_source_rl_frontier16_minimum_v2 \
  --config configs/active_rl_frontier16_minimum.json \
  --gpus auto \
  --phase1-run outputs/experiments/phase1_100k_multiscale_h20_memmap \
  --data-root data/raw/phase1-single-source-100k
```

```bash
./run_single_source_rl_tmux.sh status --run-name single_source_rl_frontier16_minimum_v2
./run_single_source_rl_tmux.sh logs --run-name single_source_rl_frontier16_minimum_v2
```

断点续训（恢复该轮选路权重及优化器，未完成的部分轨迹重新开始）：

```bash
./run_single_source_rl_tmux.sh start train \
  --run-name single_source_rl_frontier16_minimum_v2 --gpus auto
```

单独重做评价：

```bash
./run_single_source_rl_tmux.sh start evaluate \
  --run-name single_source_rl_frontier16_minimum_v2 --gpus auto
```

参数改变需要新run-name；总`updates`和`minimum_scene_visits`可在同一轮中增加。不要用v1旧run-name运行新配置。

固定起始像素的独立实验可在首次启动时额外传：

```bash
--run-name frontier16_fixed_start --set 'initial_pixel=[512,512]'
```

如需复现旧三档预算，要在新run-name中同时覆盖：

```bash
--set 'train_budgets=[65536,131072,262144]' \
--set 'evaluation_budgets=[65536,131072,262144]' \
--set budget=262144 --set max_steps=1024
```

这最多需要1,024个小窗，计算和轨迹长度会增加，建议先完成默认规模的流程检查。

## 输出与逐步可视化

默认输出：

```text
Phase1_Single_Source_Single_Intensity/outputs/active_rl/
  single_source_rl_frontier16_minimum_v2/
```

- `protocol.json`：场景、起点、数据及冻结权重身份、验证选优子集。
- `training.json`：奖励、验证分数、动作KL、全局交互数、吞吐、rank0显存峰值与定位缓存命中。
- `training_schedule.json`：该实验已要求的最低更新次数和场景访问目标；续训不会静默降低目标。
- `best.pt`、`last.pt`、`initial.pt`：选路权重；`phase1_frozen.pt`是已有定位器的快照。
- `evaluation/validation_final.json`：最佳选路器在完整验证集上的结果。
- `evaluation/metrics.json`：三预算五策略汇总，按独立几何组的配对bootstrap区间。
- `evaluation/episodes.json`：原始场景统计和观测块的完整顺序。总体曲线每8步及终点保存一次，避免巨型JSON。
- `figures/*.html`：可离线打开、播放、拖动每一步，比较累计观测、本块响应、本块独立概率与累计融合概率；代表案例保存每一步。
- `figures/*.npz`：逐步原始和融合概率数组及同一全场候选坐标；没有低概率阈值裁剪。
- `figures/*.png`：路径、初始/最终概率场与真源质量曲线的快速预览。
- `index.html`：上述图与轨迹页面入口。默认保存3个代表场景，不能替代完整测试统计。

原始单源数据仍需保留：任意新块需要读取`visibility_bits`。无需重新生成位置教师；已有离散视窗教师缓存不能替代原始响应位图。

## 效率与验证范围

自动GPU发现、一进程一卡DDP；PPO、定位查询、批量位图解码、最小值和概率归一化在所选GPU上。CPU保留文件读取和轻量边缘/路径 bookkeeping。评估也并行多个环境，合并Phase1批量查询；指标批量计算后一次传回，避免逐场景串行查询和逐标量同步。

最小值只保留一个全场log-min向量，每增加一块为O(N)更新；不会每步重新堆叠所有历史场。全亮/全暗避免模型调用但照常计费；定位预测有GPU上的有界LRU缓存。局部位图的全场packed缓存仍仅对模拟器私有，Actor看不到。

已经覆盖的检查包括：边缘连通、精确预算、禁止重复、融合顺序、真实零值不恢复、空交集报错、中性块不截峰、真值隔离、串行与并行评价一致、PPO反传，以及真实本地数据的小规模完整流程。正式100k模型的训练收益、最小值是否优于乘积、H20吞吐仍需服务器实验；不作保证。

2026-11-06再次审核：27项RL专项测试通过，另完成Phase1回归检查；真实本地16个训练场景、4个验证场景、4个测试场景跑通训练、全覆盖、五策略评价和逐步可视化。双进程测试覆盖主进程错误同步，以及双进程转单进程续训并提高到每场景两次。另专门验证删除`last.pt`后的`best.pt`恢复、访问目标不回退和重复写入拒绝。这些运行使用本地CPU及旧小定位器，仅证明已测试路径可运行，不代替H20或完整100k实验。

最小值与旧乘积的对照应保持相同16块环境和预算。可在另一个run-name下使用同一配置并覆盖`--set 'fusion_method="product"' --set fusion_discount=0.5`。随机/紧凑/边界基线的固定路径有助于区分融合变化和选路变化。
