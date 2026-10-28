# 概率核心区域评价更新

本更新只评价已经保存的原始测试预测，不重新训练学生模型，不重新运行教师渲染，也不修改 checkpoint。

默认读取原实验的：

```text
outputs/experiments/phase1_100k_multiscale_h20_memmap/response/seed_0/
```

因此不要添加 `--run-label sharpened`。如果此前运行的是 `train_phase1_sharpened.sh`，它产生的 `seed_0_sharpened` 不会被本命令读取。

运行：

```bash
cd /home/qq/yiyu/RIND_Phase2_Training_Minimal
./evaluate_phase1_100k.sh comprehensive --bootstrap-repetitions 1000
```

脚本使用教师和学生各自的最高密度区域（HDR）进行评价。主结果为 HDR@75：分别保留每个分布中累计概率质量达到 75% 的最高概率候选，剔除其余低概率尾部，再比较核心区域的 Dice、Precision、Recall、IoU、概率 overlap 和 Jensen-Shannon 距离。

同时计算 HDR@50、HDR@90 和 HDR@95，用于检查结论对核心阈值是否敏感。完整分布的 CE、KL、JS、熵和 invalid mass 仍保留在报告中，用于校准与物理有效性审计，但不会取代 HDR@75 的核心形状结论。

输出位置：

```text
outputs/experiments/phase1_100k_multiscale_h20_memmap/response/seed_0/evaluation/
├── comprehensive_metrics.json
├── comprehensive_metrics.md
└── comprehensive_observations.jsonl
```

其中 JSON 是完整汇总，Markdown 是易读报告，JSONL 保存每个测试视窗的全部指标。该命令会覆盖旧版综合评价文件，但不会覆盖测试预测或模型权重。
