# Phase2 训练有效性验收

当前学生预测的是逐层教师等权整合后的**有效贡献位置边缘分布**。它不是完整多源联合后验，不预测源数量、强度或层间位置组合，也不能解释为“该位置至少存在一个真实源”的概率。

训练结果必须依次通过三个层次。

## 1. 教师有效性

- 被接受记录的最终响应重建 MAE 应不超过 `1e-10`，且不能靠截断负残差继续运行；
- `complete / eligible` 总体建议至少 90%，各窗口 size、真实 K 分层建议至少 80%；
- 至少 95% 的恢复层建议满足最小 Phase1 响应代价不超过 0.01；
- 必须报告失败原因、路径冲突和只保留 complete 教师造成的选择偏差；
- 在可联合穷举的小问题上，应比较逐层教师边缘分布和联合参考边缘分布。建议 median JS 不超过 0.05、90% 分位不超过 0.15。

## 2. 学生分布拟合

对教师 `q` 和学生 `p`：

```text
H(q)       = -sum q log q
CE(q,p)    = -sum q log p
KL(q||p)   = CE(q,p) - H(q)
CE_uniform = log(number of candidates)
skill      = (CE_uniform - CE_model) / (CE_uniform - H(q))
```

`skill=1` 表示匹配教师，`skill=0` 等同均匀预测，负值比均匀预测更差。教师接近均匀时分母很小，应把这些低信息窗口单独报告。

第一版内部标准：

| 等级 | uniform skill | 强基线 skill | JS | TV | 学生在教师 90% 区域的质量 |
|---|---:|---:|---:|---:|---:|
| 失败 | ≤0.1 | ≤0 | >0.20 | >0.40 | <0.60 |
| 初步可用 | ≥0.5 | ≥0.3 | ≤0.10 | ≤0.25 | ≥0.75 |
| 较好 | ≥0.7 | ≥0.5 | ≤0.07 | ≤0.20 | ≥0.85 |
| 优秀 | ≥0.8 | ≥0.6 | ≤0.05 | ≤0.15 | ≥0.90 |

强基线至少包括：window-only、按窗口 size/位置统计的 mean-teacher，以及 shuffled-response。只超过 uniform 不足以证明模型使用了局部响应。

在任何全量训练前，模型必须先通过 128 条记录过拟合检查：无 NaN/Inf，最低 train CE 接近 train teacher entropy，uniform-normalized skill 建议至少 0.95。

## 3. 实际搜索价值

在测试集上用学生 top 1%、5%、10% 候选重新做物理验证，报告覆盖的教师质量、最低物理代价、遗漏区域和实际重渲染次数。建议 top 10% 候选覆盖至少 90% 教师质量，并在至少 90% 的窗口中接近完整教师的最低物理误差，同时减少至少 10 倍候选评估。

所有正式指标按独立 scene 聚类汇总并做 scene bootstrap；至少按窗口 size、真实 K、恢复层数和教师熵分层，运行三个训练 seed。不同 spacing、temperature 或候选数量的原始 CE 不直接比较。

## 命令

完整审计已有教师和训练：

```bash
./run_validation.sh audit
```

如果 `best.pt` 存在，该命令会实际加载学生模型，在 validation 和 test 记录上重新预测完整候选概率场，并把 CE、KL、JS、TV、分布重叠、熵差、uniform skill 和教师 90% 集合覆盖质量写入 `student_distribution_evaluation`。这不是仅从训练日志推算指标。

快速抽取每个 split 最多 1000 条记录：

```bash
./run_validation.sh audit --max-records 1000
```

显式使用 GPU 并调整评估 batch：

```bash
./run_validation.sh audit --max-records 1000 --student-device cuda --student-batch-size 32
```

复用正式教师缓存，运行 128/32/32 的 FP32 稳定性实验：

```bash
./run_validation.sh overfit --gpus 1
```

审计结果分别保存在对应实验目录的 `diagnostics/teacher_training_audit.json`。
