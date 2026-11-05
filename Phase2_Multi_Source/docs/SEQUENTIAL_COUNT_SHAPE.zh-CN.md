# Phase2 v6：两项监督、自主逐层减法与响应守恒

## 本轮约定

- 模型输入只有完整原始局部视窗响应 `y` 与窗口位置/size。
- 教师只在 loss 中提供最终有效层数和一组最终二值掩码；层的顺序不作为答案。
- 模型不接收教师中间残差、教师种子、传播轨迹、每步强度或真实光源编号。
- 训练与验证均自主逐步执行。没有 teacher forcing，也不使用教师层数决定执行步数。
- 层数和单层形状分别监督，两个任务权重各为 0.5。
- 没有强度、重建、残差或轨迹 loss。完整响应是已知约束条件。
- 只有二值掩码 × 每层一个正标量强度的叠加才是合法的响应分解。

## 模型实际执行什么

原始多尺度响应/边界编码器只编码一次，继承 v5 权重。计数模块从原始响应预测 R，随后同一个单层提取器执行最多 R 步。

1. 初始化模型内部残差 `r = y` 与循环记忆 `h = 0`。
2. 共享神经提取器读取原始响应特征、自己的当前残差和记忆，预测本层完整掩码；没有教师的最低值/种子/BFS规则。
3. 将当前层掩码二值化。原始响应为零的位置不能包含正贡献；不同层的掩码可以重叠。
4. 对该模型掩码，计算不超过当前像素预算的暂定标量强度，即其支持上残差最小值。掩码是模型先预测的，强度预算不是用教师最低区域决定掩码。
5. 明确执行 `r_next = r - a*M`，以这个自主残差及循环记忆读取下一层；没有截断负残差。
6. 执行完模型预测的 R 步后，固定这组模型掩码，仅用原始已知响应求解各层正强度的可行性。
7. 若能逐像素守恒且每层正强度，则返回有效分解；否则明确标为未完成。不会修改掩码、层数或原始观测来凑等式，不任意补最后一层。

最后的强度可行性计算允许调整内部暂定强度，避免相同掩码集合仅因生成顺序不同而被惩罚。它不是神经网络loss，也不使用教师强度。GPU上的仿射/非负迭代是有限预算求解器，未找到可行解不等于证明无解。掩码线性相关时强度可能不唯一。

单层提取器的边界/区域/全局注意力读取不同记忆。只有一个查询时，注意力在空间位置上归一化，避免“对一个槽做 softmax”退化成常数。

32 是沿用大缓存的安全执行上限，不要求同时提取32个贡献。有效层数 R 不等于真实物理源数量 K。

v5 强度头冻结，仅保留在继承的计数特征精炼中。内部暂定强度由已知残差和模型掩码决定；最终强度由守恒可行性计算确定。不由该头提供，也不读取教师强度。

## 为什么不会出现负残差

对模型预测的当前掩码 M，暂定强度不超过该支持上任意像素的剩余响应。因此 `r_next = r - a*M >= 0`。空掩码或已耗尽预算会使暂定强度为0，不能冒充一个有效正贡献。

该性质在减法前成立，不是对结果 clamp。原始响应非负时，全部内部步骤的残差都非负。最终固定掩码求得正强度后，再逐像素检查其和等于 y，所有层的任何部分和也不超过 y。数值求解产生的微小超额只能通过对标量强度的舍入级修正处理，不能逐像素改变层亮度。

代码用 float64 做残差减法。缓存响应为 float32，默认仅允许 `8 * float32_eps * window_max` 的逐像素舍入误差。最大响应为1时约为9.54e-7；不是5%的物理重建容差。`roundoff_ulps=0` 可要求数值完全相等，但分别舍入的缓存未必能通过。

`constraint_accepted` 同时要求：

- 所有残差非负；
- 末尾每个像素残差不超过舍入容差；
- 最终固定掩码可行解的正强度层数等于预测 R，且每层掩码非空。

若最终固定掩码的可行性计算未通过守恒、非负或非空检查，则标为未完成。`accepted_layers()` 对未完成窗口返回 None。原始候选仍保留供诊断，但不能作为满足守恒的完整分解使用。

这保证接受的输出满足约束，不保证任意输入都会得到可接受的分解。约束不能替代形状学习，也不能证明教师有效层对应物理单源。

## 训练监督与两个指标

训练中教师答案只进入独立 loss 模块。模型完整自主运行后，预测层与教师最终掩码进行无序匹配：优先使用实际启用层，训练按同一 BCE+Dice 形状代价进行 Hungarian 匹配；评价按硬掩码 IoU 匹配。不读取教师强度。即使硬IoU打平，训练也不会因为教师数组顺序改变而改用另一套不相同的soft形状代价。

`L_total = 0.5 * L_count + 0.5 * L_shape`。

- `L_count` 为继承的融合/全局/槽计数分类的加权平均，以 `log(S+1)` 明确归一化；内部相对权重1、0.5、0.25。
- `L_shape` 为 BCE/log(2) 与 Dice error 的等权平均，按所有教师层求平均。
- 没有预测到的教师层计为形状失败；多预测的层由层数监督处理。
- 两个任务等权不意味着每一步数值或梯度大小严格相等。
- 前向使用完全相同的二值掩码和非负减法；反向对掩码使用 straight-through 代理梯度，强度预算停止梯度。后续层监督可经残差与循环记忆传回早期掩码。它是近似梯度，不能称为精确离散求导；实际收益需实证检查。只监督最终无序集合，没有教师过程监督。

核心评价只有：

1. 层数预测准确率：预测 R 与教师 R 一致的视窗比例。
2. 单层形状准确率：实际输出层与教师匹配后，IoU≥0.90的教师层比例。漏层按失败处理。

未通过守恒验收的窗口也保留在指标分母中。约束验收仅单列为输出合法性审计，不再使用原来的 strict、强度准确率或重建误差作为模型优劣评分。

最佳权重按 R=1、2、3、4、≥5 分组的 count/shape 宏平均等权选择，以免单层数量掩盖复杂层表现；整体 count/shape 与单层 count 设0.5个百分点的退化保护。模型不以重建误差选 best。

## 本次修正：梯度路径与跨卡归一化

`model.residual_gradient=straight_through` 明确标记代理梯度协议。推断不使用代理梯度，前向二值结果与非负性质保持不变。

`training.loss_normalization=optimizer_step`：先统计整个优化步的全部微批次和全部GPU上的加权窗口数、加权教师层数，再分别归一化两个任务。每个rank的反向贡献乘world_size，抵消DDP的梯度平均；梯度累积不再额外除以微批次数。

同一组样本合成大batch或拆成微批次，梯度应在浮点容差内一致。epoch日志也使用全局累计分子/分母计算任务均值。验证分区即使不等长，也只在epoch末汇总，不在逐批验证里执行归一化collective。

新入口是 `run_implicit_masks_tmux.sh`，使用独立会话和 `<原大实验名>_implicit_v6` 目录，避免混用上一版的权重、优化器和训练协议。旧入口与旧协议保留。

## 复用原大训练集

入口默认读取 `rind_masks_disentangled_more` 的原配置指针，找到该轮完整的 `mask_records.json` 和 `mask_cache`。

- 所有原训练窗口，包括追加的复杂窗口，全部保留。
- validation/test 名单与原缓存一致。
- 不退回默认10万，不重新挑窗口，不修改教师答案，不做全场教师采样。
- 只建立只读缓存引用，不复制整套缓存；不需要原始数据包。
- 自动冻结原实验 `best.pt` 的模型快照，只迁移权重，不沿用旧优化器。
- 原实验、原缓存与原 tmux 会话保持独立；新会话为 `rind_masks_implicit_v6`。

启动日志会打印实际 train/validation/test 数量。以服务器真实缓存为准，不用目录名猜数量。

## 服务器执行

```bash
cd /home/qq/yiyu
tar -xzf RIND_Phase2_Update.tar.gz
cd RIND_Phase2_Training_Minimal
unset RIND_MASKS_CONFIG RIND_DECOMPOSITION_CONFIG RIND_TMUX_SESSION

./run_implicit_masks_tmux.sh start all \
  --source-session rind_masks_disentangled_more \
  --gpus auto --auto-batch
```

原配置指针不存在时可以显式指定现有大实验：

```bash
./run_implicit_masks_tmux.sh start all \
  --source-root outputs/masks_disentangled_v5_complex245746 \
  --gpus auto --auto-batch
```

这里的源目录示例来自上一轮诊断；如服务器目录不同，以实际大实验路径替换。若缺少源缓存或权重会报出具体缺失路径，不会重新渲染或偷偷使用小数据集。

```bash
./run_implicit_masks_tmux.sh status
./run_implicit_masks_tmux.sh logs
./run_implicit_masks_tmux.sh attach
```

关闭SSH/VPN不影响 tmux 作业。重新执行 `start all --gpus auto --auto-batch` 会复用原配置与 last.pt 续训；原会话仍运行时不会启动第二份任务。

GPU选择、BF16、DDP、共享原始编码特征及数据预取沿用现有实现。内部循环每一步只计算尚有预测层需要生成的活跃窗口，避免一个长序列让整批简单窗口空转。原生图像计算和残差减法在GPU执行；没有CPU几何渲染。自动batch会探测完整自主执行上限的前向/反向显存，避免仅以较短步骤估算后在复杂窗口OOM。实测吞吐由服务器自动校准，不能由本地CPU测试保证H20峰值利用率。

## 输出位置

默认新输出是 `Phase2_Multi_Source/outputs/<原大实验名>_implicit_v6/`。

- `models/seed_0/best.pt`：按两个主要指标选择的最佳权重。
- `models/seed_0/last.pt`：断点续训。
- `models/seed_0/training.json`、`training_summary.json`：训练记录。
- `evaluation/seed_0/validation_primary_metrics.json`：验证集两个指标。
- `evaluation/seed_0/primary_metrics.json` 与 `.zh-CN.md`：锁定测试集两个指标、分层结果和场景bootstrap区间。
- `evaluation/seed_0/primary_metrics_windows.jsonl.gz`：逐窗口原始计数与约束状态。
- `tmux_runs/`：后台命令、日志、退出码。

测试集只在训练结束后的正式评价读取一次，不重复执行训练循环内的测试。

训练频率权重仅由原训练集 R=1/2/3/4/≥5 的计数计算：平方根逆频率并限制在[0.25,4]，不读取验证/测试频率、不重复或删减训练名单。层数和形状任务分别按窗口数、教师层数计算权重。两项任务归一化后仍各占0.5。

固定随机种子为0，默认60轮、patience=12。继承模块学习率1e-5，新循环状态/残差模块3e-5；2轮预热后cosine衰减至初始学习率的10%，恢复训练按epoch确定性续接。运行配置保存在 `outputs/launch_configs/<原大实验名>_implicit_v6.json`；续训会核对配置/缓存签名，修改训练定义需指定新的 `--run-name`。


## 保留原权重，完成后对比两个指标

旧实验默认是 `cache_source_root` 所指的原大实验；可以用 `comparison_reference_root` 明确指定另一个共享完整缓存的参考实验。该选项默认无需修改。

- 原目录的 `best.pt`、`last.pt` 不覆盖、不清理，原last中的优化器状态保留。
- 旧训练的 `models/seed_0/training.json` 出现最终 `best_epoch` 与 `trained_through_epoch` 后，才将其视为已完成；中途epoch记录不会当成最终训练。
- 对比时保留旧最终best和last的模型副本、epoch与SHA；副本只含模型，不复制优化器。旧最终best可能不同于新训练启动时的warm-start快照，对比不使用中途快照代替最终权重。
- 新训练结束后自动在相同验证/测试名单上重新推理旧最终best和新best。两个指标统一为层数准确率和单层IoU≥0.90比例，实际启用层优先匹配，掩码阈值0.5。旧软概率不能直接转bool当二值掩码。
- 不剔除新模型守恒失败的窗口，不根据置信度挑样本。检查两模型使用同一完整缓存与名单，不通过取交集凑对比。
- 报告总体、R=1/2/3/4/≥5结果、变化百分点与以scene_id为单位的配对bootstrap区间。不同best沿用各自验证集选择策略，属于完整流程对比，不宣称只证明某一结构改动有效。

新实验的 `evaluation/seed_0/` 下新增：

```text
preserved_reference_weights.json                  # 旧最终best/last副本、原路径、epoch、SHA
reference_final/primary_metrics.json              # 旧最终best按统一口径重新评估
reference_comparison.json                         # 锁定测试集新旧对比
reference_comparison.zh-CN.md                     # 可读对比表
reference_comparison_paired_windows.jsonl.gz       # 每个窗口的配对差值
validation_reference_comparison.json              # 同口径验证集对比
```

模型副本位于新实验的 `reference_weights/seed_0/`，当前轮best/last仍保留在当前轮 `models/seed_0/`。不需要把旧权重移动或改名。

如果新训练结束时旧训练仍未完成，当前轮指标会正常保存，并写入 `reference_comparison_pending.json`，不伪造最终对比。等旧训练结束后，离线执行：

```bash
./run_implicit_masks_tmux.sh start evaluate --gpus auto --auto-batch
./run_implicit_masks_tmux.sh status
```

只重新评估，不重新训练、不重新渲染。已训练好的本轮权重也可直接使用这条命令补做对比。更新该分析代码不改变模型或训练协议，无需重训。
