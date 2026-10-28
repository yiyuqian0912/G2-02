# 同场景多视窗集合训练（v7）

## 1. 当前定位

集合训练不会把问题改成联合枚举多个源，也不会让网络一次输出一个源集合。学生仍然执行已经验证过的 Phase1 风格任务：

```text
一个二值残差层响应 + 当前窗口坐标 → 一张全场候选位置概率分布
```

新增部分只发生在训练损失和多视窗推理阶段。同一场景的不同局部视窗先分别完成残差解耦和单层预测；能够被保守关联为同一个有效贡献的层组成一条跨视窗轨迹，再融合它们的全场位置证据。

这个设计适用于实际应用中的“完整障碍物几何未知”模式。学生不读取真实源编号、逐源真值掩码或完整障碍物 primitive。

## 2. 为什么需要集合训练

单个局部视窗通常只给出一条带状或扇形可行区域。不同视窗对同一贡献产生不同约束，若它们的高概率区域在全场坐标中相交，融合后应当保留交集并降低位置分布的散度。

已有 v6 代码只实现了训练后的跨视窗关联与概率融合，没有让学生在训练时看到“融合后也应与教师一致”这一目标。v7 在保留单层监督的同时增加集合蒸馏，使学生输出更适合后续证据融合。

## 3. 数据和轨迹构造

数据职责分为两部分：扩充后的 10 万场景单源包只用于学习更稳定的“二值层响应 → 全场位置分布”基础能力；本实验的多源包负责生成残差层、跨视窗轨迹与多源验证。单源包不能直接替换 `data/raw/phase2-multi-source`，否则源强度、分源通道和教师记录接口会不匹配。

1. 复用 v4 的 `layer_records.json` 和高置信残差层，不重新生成教师。
2. 严格按 `scene_id` 保持 train、validation、test 隔离。
3. 在每个场景内部，以教师单层概率场、有效贡献强度和 `view_id` 建立保守轨迹。
4. 一条轨迹在同一视窗内最多包含一个层；聚类采用 complete-linkage，所有跨簇层对都必须满足阈值。
5. 只保留至少覆盖两个不同视窗的轨迹作为集合训练单位。
6. 训练关联不读取真实源编号。教师概率场参与关联属于蒸馏监督；测试推理时必须改用学生概率场执行同一关联规则。

默认关联参数与 v6 一致：普通层对阈值为 `0.55`；连续强度在相对容差 `1e-6` 内相等时阈值为 `0.15`；概率场接近均匀分布时会通过信息量因子降低关联分数。

## 4. 训练目标

设一条轨迹包含 (m) 个局部层，教师和学生单层分布分别为 (q_i(s)) 与 (p_i(s))。单层损失继续使用 Phase1 交叉熵：

\[
L_{\mathrm{layer}}
=\frac{1}{m}\sum_{i=1}^{m}H(q_i,p_i).
\]

集合证据采用折扣后的 log 概率融合。默认相关性折扣为 \(\rho=0.5\)，总有效证据权重为

\[
W_m=1+\rho(m-1),\qquad w_i=\frac{W_m}{m}.
\]

教师和学生使用完全相同的融合规则：

\[
Q_{\mathcal T}(s)\propto\exp\!\left(\sum_iw_i\log q_i(s)\right),
\qquad
P_{\mathcal T}(s)\propto\exp\!\left(\sum_iw_i\log p_i(s)\right).
\]

集合蒸馏损失为

\[
L_{\mathrm{set}}=H(Q_{\mathcal T},P_{\mathcal T}),
\qquad
L=L_{\mathrm{layer}}+0.5L_{\mathrm{set}}.
\]

保留 `L_layer` 可以防止集合微调破坏已经有效的单视窗定位。集合项的目标是让多个近似正确的几何形貌在融合后仍与教师融合形貌一致。

## 5. 场景平衡和训练策略

- 每个训练 epoch 对每个场景随机选择一条多视窗轨迹，因此场景拥有更多窗口或更多残差层时不会自动获得更大权重。
- 每条轨迹最多随机使用 4 个视窗，控制显存并增加视窗组合变化。
- validation 和 test 使用各场景的全部保留轨迹，然后先在场景内平均、再跨场景平均。
- 从 v4 的 `best.pt` warm start；epoch 0 的原模型会先评估并成为可选最佳检查点。集合微调若没有带来验证收益，不会覆盖 v4 基线。
- 最佳模型按 validation 集合 CE 选择，同时要求单层 CE 相对 epoch 0 的增加不超过 `0.05`；集合变好但破坏单视窗能力的 checkpoint 不会被接受。
- NPZ 读取和解压使用 16 个有序预取线程、提前准备 2 个 track batch；候选分块留在每张 GPU 内部执行，六卡按照层样本维度并行。
- 模型结构和候选网格保持不变，仍可使用现有 Phase1/Phase2 单层可视化。

## 6. 可运行命令

从 `Phase2_Multi_Source` 目录执行：

```bash
# 只需执行一次：从 v4 层索引构建同场景跨视窗轨迹
./run_view_set_training.sh index \
  --config configs/layer_student_view_set.json

# 六张 GPU 正式训练；相同命令可从 last.pt 恢复
./run_view_set_training.sh train \
  --config configs/layer_student_view_set.json \
  --gpus 6

# 使用最佳检查点重新评估 validation/test
./run_view_set_training.sh evaluate \
  --config configs/layer_student_view_set.json \
  --gpus 6 --seed 0
```

连续建立索引并训练：

```bash
./run_view_set_training.sh all \
  --config configs/layer_student_view_set.json \
  --gpus 6
```

小规模流程检查可限制每个 split 的场景数。应使用单独的 `output_root`，避免与正式实验 checkpoint 混用：

```bash
./run_view_set_training.sh train \
  --config configs/layer_student_view_set.json \
  --device cpu --epochs 1 --max-scenes 2
```

上述命令仍要求 v4 的层索引、教师文件和初始 checkpoint 已存在。

## 7. 输入和输出

必要输入：

- `outputs/full_training_h20_v2/`：逐层教师文件；
- `outputs/layer_student_v4/layer_records.json`：经过高置信筛选的单层索引；
- `outputs/layer_student_v4/models/seed_0/best.pt`：单视窗学生初始化；
- `configs/layer_student_view_set.json`：关联、集合损失和算力参数。

输出位于 `outputs/layer_student_v7_view_set/`：

- `view_set_index.json`：各 split 的场景、轨迹、视窗和关联质量；
- `models/seed_0/best.pt`：验证集合目标最优的模型；
- `models/seed_0/last.pt`：可恢复训练状态；
- `models/seed_0/training.json`：每个 epoch 的单层 CE、集合 CE、融合 overlap 和熵；
- `training_summary.json`：正式训练汇总；
- `evaluation.json`：最佳模型在 validation/test 的集合评价。

`evaluation.json` 同时区分两种口径：预先给定教师轨迹时的融合分布质量，以及完全由学生概率场建立轨迹时的 pair precision、recall、F1。后者更接近实际推理流程；教师轨迹只是代理标签，不等同于真实源身份。

## 8. 判断训练是否有效

集合训练至少同时满足以下条件才算有效：

1. validation/test 的 `set_cross_entropy` 低于 epoch 0 的 v4 基线；最佳 checkpoint 直接按此项选择；
2. `fused_probability_overlap` 提高；
3. `layer_cross_entropy` 不出现明显退化；
4. 改善在场景等权统计下成立，而不是由一个窗口很多的场景主导；
5. 推理时用学生场关联后仍能观察到相同趋势。

其中第 5 项由 `student_driven_association_validation/test` 报告。若条件于教师轨迹的集合 CE 很好、但学生关联 recall 很低，说明定位场可以融合，实际系统却无法稳定找到该融合关系，仍不能判为成功。

`student_entropy_reduction` 只能说明融合后分布变尖，不能单独证明结果更准确。错误关联也会产生非常低的熵，因此必须与集合 CE、概率重叠和真实源距离诊断一起看。

## 9. 审核后的边界

已经由代码保证：统一候选网格、场景隔离、每条轨迹每个视窗至多一层、教师/学生使用相同融合公式、视窗相关性折扣、场景等权评价、v4 epoch 0 回退和 checkpoint 恢复。

经验性部分：关联阈值、强度相等的低阈值、`rho=0.5`、每轨迹最多 4 个视窗。这些参数需要用开发场景确定，并在独立测试场景上冻结验证。

尚未解决：残差层与真实物理源并非严格一一对应；同一源在不同视窗中可能被拆成不同有效贡献；学生场驱动的推理关联可能比教师场关联更差。v7 是可执行的集合蒸馏基线，不宣称恢复完整多源联合后验。
