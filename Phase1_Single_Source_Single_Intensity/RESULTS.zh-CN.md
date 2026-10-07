# 真实数据 Part E pilot 结果

使用用户提供的单源 ZIP。C 盘不足以完整安装，因此校验原始数组完整 SHA256 后提取前 32 个真实场景；不是生成的 mock 数据。原 ZIP 保留。

训练/验证/测试按原 scene ID 固定为 0–23 / 24–27 / 28–31，每场景一个 L=16 观测。uniform spacing=128、tau=0.05、兼容阈值=0.01；使用 Thomas 原模型训练 80 epochs，checkpoint 仅按验证 CE 选择。这里是探索性 pilot，不是最终研究协议。

端到端单观测联调通过：True。

| 4 个测试场景平均指标 | 结果 |
|---|---|
| Teacher→Student KL | 0.853032 |
| L1 distance | 1.047210 |
| Student 低成本区域质量 | 47.74% |
| Teacher 低成本区域质量 | 99.14% |
| Student 期望物理成本 | 0.486845 |
| Teacher 期望物理成本 | 0.000602 |
| uniform-probability baseline KL | 1.020955 |
| uniform-probability baseline 低成本质量 | 37.50% |

结果显示真实链路已接通，但 Student 分布与物理 Teacher 仍有明显差距，不能声明已恢复正确解域。和均匀预测基线比较应按全部指标判断，不以单个图像或训练损失证明泛化。

四个测试观测都是全 1 响应，图中白色是实际观测。相同局部响应对应不同隐藏几何下的 Teacher 解域，是重要的可辨识性限制；不能把所有差异归因于训练不足。

## Adaptive 与边界消融

在 scene=0/view=4 上，uniform 渲染 63 次，adaptive 45 次，减少 28.57%；参考兼容格点召回 62.96%，参考 Teacher 质量覆盖 62.96%。省计算同时漏掉兼容区域，后续需增加 exploration/retention，并在多个场景验证。

计时：uniform 0.019077s，adaptive 0.014737s。single CPU run; renderer warmed using one excluded diagnostic render; not a controlled repeated benchmark；不据单次计时声明稳定加速比。

已生成匹配支持上的 Lresp 与 Lresp+0.1*Ledge Teacher 对比。两者 lambda=0、sigma=1、alpha=1、tau=0.05，候选完全对齐；对应 Student 消融结果见后文，Teacher 的变化不等于 Student 改善。

## 分析限制与下一步

- 主测试仅 4 个场景，训练仅 24 条真实观测；前 32 场景选择未经随机化。
- spacing=128 是粗网格，只能描述采样解域，不能证明连续多解拓扑。
- 评价使用 Teacher 的 hidden-geometry validity mask；不是无几何部署结果。
- 训练仅 L=16；其他窗口尺寸的固定模型结果属于尺寸外推诊断。
- obstacle type multiset 分组不是未见连续几何证明；若测试没有新组合，不能报告新组合成功。
- 正式研究还需更大场景集、细网格、多随机种子、多尺寸训练、匹配 Student edge 消融和公开候选集部署检查。

## 文件

- figures/teacher_student_test.png：四场景原观测、物理成本与 Teacher/Student。
- figures/training_validation.png：真实训练/验证曲线。
- reports/metrics.json：逐观测、场景均衡指标与 held-out/组合分组。
- end_to_end/end_to_end_result.json：逐阶段证据。
- ../real_comparisons/comparisons.json：边界和搜索结果。

## 六种尺寸诊断（scene 28，模型仅在 L=16 训练）

| L | KL | 参考兼容格点数 | Student 兼容质量 |
|---|---|---|---|
| 16 | 0.8844 | 20 | 45.51% |
| 32 | 1.1490 | 16 | 31.39% |
| 64 | 1.6435 | 9 | 15.47% |
| 128 | 8.2631 | 0 | 0.00% |
| 256 | 7.1604 | 0 | 0.00% |
| 512 | 3.4827 | 0 | 0.00% |

较大窗口在粗网格上可能没有满足阈值的候选；这时兼容质量为零不能单独说明 Student 错误，需要细化网格。尺寸结果属于外推诊断，非匹配多尺寸训练实验。

## 匹配 Student 边界消融已完成

same splits, candidates, tau, seed, optimizer, epochs; independently validation-selected checkpoints; lambda=0, beta=0 vs .1。

| 共同 response-only 物理诊断 | Lresp 模型 | Lresp+0.1Ledge 模型 |
|---|---|---|
| 期望 Lresp | 0.486845 | 0.491847 |
| 兼容区域质量 | 47.74% | 47.26% |

仅一组 beta 和单随机种子的 pilot，不能据此断言边界项普遍有效或无效。
