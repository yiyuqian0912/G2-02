# 事件几何与可迁移反演

## 我理解的目标

利用 G1 数据集设计一个能够在潜空间中学习由静态结构决定的空间遮挡关系的模型，也就是这个模型应该学习到真正的空间理解而不是模式匹配，从而在未见结构上保持较强的泛化空间理解能力。

RIND 问题的非局部性与不连续性，使其天然成为检验空间关系理解的问题。因此，经过适当数据与评估设计的 RIND 任务，会系统性削弱非空间捷径，从而对可迁移空间关系表示施加更强的学习压力。

项目成功的核心是：

- **空间理解：** 在未见结构上保持泛化能力，为此需要合理设计Mask和target，以及合适的JEPA和Training Dynamic
- **正反统一：** 正向预测和反向搜索必须依赖同一套中间表示和物理预测权重，而非两个可各自学习捷径的网络。

> 学习可迁移的 ray-event latent geometry。

## 我认为最重要的研究问题

### RQ1：Mask 和 target 应如何设计，才能迫使 latent 编码可迁移的 RIND 几何？

随机遮掉阴影像素，可能只要求模型完成局部插值或补全场景特有的图案，未必需要理解“哪段结构通过哪族射线造成了切换”。因此，需要同时考虑遮掉什么、保留什么，以及预测目标本身包含多少关系信息。

[I-JEPA（Assran et al.，CVPR 2023）](https://arxiv.org/pdf/2301.08243v3#page=3)提供了一个具体思路：target 编码器先处理完整图像，再取出目标区域的表示；context 编码器只能看到未被遮掉的区域。这样，缺失区域的预测目标也可以包含周围结构的信息。作者强调：

> “the target blocks are obtained by masking the output of the target-encoder, not the input.”
>
> [PDF 第 14 页，附录 C，*Masking at the output of the target-encoder*](https://arxiv.org/pdf/2301.08243v3#page=14)。

这一选择有直接消融支持：ViT-H/16 预训练 300 epochs，冻结编码器并用 1% ImageNet-1K 标签训练线性分类器，先编码完整图像再选取目标的准确率为 **67.3%**，分别编码各目标区域则为 **56.1%**。[PDF 第 15 页，表 11](https://arxiv.org/pdf/2301.08243v3#page=15)。对 RIND 来说，这提示我们比较单点亮暗 target 与包含跨位置、跨光源信息的一组射线 target，检验更丰富的上下文是否有利于学习事件关系。

目标更丰富，也需要保留足够的推理依据。[Point-JEPA（Saito et al.，WACV 2025）](https://arxiv.org/pdf/2404.16432v6#page=3)先将点云分成局部块，分别处理局部形状与位置，再按空间接近关系选择成组的 context 和 target。其消融固定 4 个目标块及 context 比例设置，将目标块比例由 **15%–20%** 增至 **35%–40%**，ModelNet40 线性分类准确率从 **93.3% 降至 84.6%**。作者据此指出：

> “Point-JEPA does not require a large size for the target blocks”
>
> [PDF 第 11 页，补充材料 B，*Ratio of Targets*、表 8](https://arxiv.org/pdf/2404.16432v6#page=11)。

两篇论文共同提示，mask 的关键是预测难度与剩余证据之间的平衡。我们可以比较随机遮挡、连续光源区间遮挡，以及按射线关系组织的遮挡；同时保留与目标有关的远端几何或其他射线观测。具体哪种组合能够改善跨结构迁移，需要实验判断。

target 的内容还可以进一步关注切换边界。[DeepSDF（Park et al.，CVPR 2019）](https://arxiv.org/pdf/1901.05103v1#page=4)通过连续距离函数表示形状，并用截断距离损失及表面附近的密集采样集中学习边界细节；相关设计见 PDF 第 4 页第 3 节、第 5 页 *Data Preparation*。这一思路启发我们研究连续的事件表示，使其保留亮暗切换附近的信息。但 DeepSDF 有距离监督；G1 只有亮暗标签，就不能直接将模型分数解释为真实事件距离。

**反演的输入也需要在这里明确。** 初步可以考虑已知结构几何、观测位置及对应的亮暗观测，光源作为待求变量；再比较完整阴影与稀疏观测两种设置。输入必须区分“没有观测”与“观测为暗”，观测编码也不能依赖反演时未知的光源。观测不足时应允许多个合理解释，而非强制唯一答案。

### RQ2：正向和反向推理如何证明模型学到了空间关系？

两套独立网络可能分别利用不同的统计捷径，使正反结果难以共同支持空间理解的判断。可以让两个方向共享关系表示与核心权重，但仍需区分“如何实现反演”与“如何验证模型使用了正确的关系”。

一条路线是**前向模型加反向搜索**。[Neural-Adjoint（Ren et al.，NeurIPS 2020）](https://arxiv.org/pdf/2009.12919v4#page=4)先学习前向模型，再固定权重，通过梯度调整输入，使预测结果接近目标观测：

> “the parameters of the neural network are fixed, and we are only adjusting the input to the network”
>
> [PDF 第 4 页，第 3 节 *The Neural-Adjoint Method*，式 4 后的解释](https://arxiv.org/pdf/2009.12919v4#page=4)。

实验中，作者从多个初始点得到 1,000 个候选，用前向模型排序后，再将选出的候选交给模拟器评估。搜索可能利用近似模型在训练范围外的错误，因此作者加入 boundary loss 约束输入范围。[PDF 第 4–5 页，第 3–3.1 节](https://arxiv.org/pdf/2009.12919v4#page=4)。在 RIND 中，这对应固定结构与观测、搜索光源；其局限是反演依赖前向模型在新结构上的准确性，搜索机制本身不能补足泛化能力。

[V-JEPA 2（Assran et al.，2025）](https://arxiv.org/pdf/2506.09985v1#page=10)也采用了相近思路：用潜空间预测器评估候选动作序列，比较预测终点与目标图像的表示距离，再通过 Cross-Entropy Method 搜索动作，执行第一步后重新规划。[PDF 第 10–11 页，第 3.2 节 *Inferring Actions by Planning*，式 5、图 7](https://arxiv.org/pdf/2506.09985v1#page=10)。这说明共享潜表示可以用于搜索与规划，为方案1提供了进一步的方法依据。

另一条路线是**共享掩码模型直接推断未知变量**。[UniMASK（Carroll et al.，NeurIPS 2022）](https://papers.neurips.cc/paper_files/paper/2022/file/e58fa6a7b431e634e0fd125e225ad10c-Paper-Conference.pdf#page=3)把状态、动作和回报等信息组成序列，通过 mask 指定已知条件和预测目标，同一个双向 Transformer 据此完成不同任务：

> “we formulate tasks in sequential decision problems as input masking schemes.”
>
> [PDF 第 3 页，第 3.1 节 *Tasks as Masking Schemes*](https://papers.neurips.cc/paper_files/paper/2022/file/e58fa6a7b431e634e0fd125e225ad10c-Paper-Conference.pdf#page=3)。

这对应我们的方案2：给定结构与光源、遮掉阴影，做正向预测；给定结构与阴影、遮掉光源，做反向推断。训练需要覆盖两个方向，反向输出应允许候选或分布。不过，UniMASK 的图 7 显示，随机掩码训练只在一半任务上优于单任务，指定多任务训练也不稳定占优。[PDF 第 7 页，第 4.3 节 *Measuring Single-Task Performance*](https://papers.neurips.cc/paper_files/paper/2022/file/e58fa6a7b431e634e0fd125e225ad10c-Paper-Conference.pdf#page=7)。因此，共享模型可行，并不意味着共享后必然更好；将其改为 JEPA 的潜表示预测，并验证未见结构上的泛化，仍是我们的研究内容。

这两条路线都需要更有针对性的评估。[Shortcut Learning（Geirhos et al.，Nature Machine Intelligence，2020）](https://arxiv.org/pdf/2004.07780v5#page=4)用星形／月牙分类说明：当类别与位置相关时，网络可以在同分布测试中表现良好，却在控制位置因素后降至随机猜测。图注明确指出：

> “The network has learned to associate object location with a category.”
>
> [PDF 第 5 页，图 2；相关论证见第 4–6 页第 3 节](https://arxiv.org/pdf/2004.07780v5#page=5)。

因此，除了按完整结构划分训练与测试集，还应构造成对测试：固定查询点、光源与局部几何，只改变会影响遮挡的远端结构；再用不影响相关射线的结构变化作对照。对反演，则同时检查独立射线验证结果、多解覆盖和计算预算。**共享权重是方法上的约束，测试为证明模型学习到可迁移空间关系提供证据。**

### RQ3：采用哪一种 JEPA 与 RIND inductive bias，怎样避免 collapse？

前两个问题规定了模型应该利用什么信息、完成什么推断；这里还需要考虑如何稳定地学到这些表示。Representation collapse 指不同输入得到相同或近乎相同的表示，使模型通过输出常量降低表示匹配损失。I-JEPA 采用非对称编码器设计，target 编码器通过 context 编码器权重的指数移动平均更新，可作为初始训练基线。[I-JEPA，PDF 第 2–4 页，第 2–3 节](https://arxiv.org/pdf/2301.08243v3#page=2)。

[Bardes 等人的 VICReg（ICLR 2022）](https://arxiv.org/pdf/2105.04906v3#page=4)提供了显式约束的思路：在编码器后的投影头输出上匹配不同视图，为各维度的批内标准差设置下限，并抑制维度之间的相关性。作者特别强调：

> “Using the standard deviation and not directly the variance is crucial.”
>
> [PDF 第 5 页，第 4.1 节 *Method*，式 2 后的解释](https://arxiv.org/pdf/2105.04906v3#page=5)。

但这些组件不能简单叠加。在 ResNet-50、100 epochs 的 ImageNet 线性评估中，复现的 BYOL 原始设置为 **69.3%**，加入方差正则为 **70.2%**，同时加入方差和协方差正则为 **69.5%**；作者也报告，正则施加在 expander 输出上优于 predictor 输出。[PDF 第 8 页，第 6 节 *Analysis*、表 4](https://arxiv.org/pdf/2105.04906v3#page=8)。因此，应分别比较正则组件和施加位置，并检查表示的变化是否对应遮挡关系，而不只是坐标或场景身份。

[LeJEPA（Balestriero & LeCun，2025）](https://arxiv.org/pdf/2511.08544v3#page=9)则提出 SIGReg：沿随机方向投影表示，约束投影分布接近标准高斯，并在训练中重新采样方向。图 6 通过直接优化合成样本，展示它能够修正退化维度。完整 LeJEPA 还加入视图之间的表示匹配：

> “The prediction loss is then given by having all views predict the global views”
>
> [PDF 第 12 页，第 5.1 节 *The Prediction Loss*；实现见第 11 页算法 2](https://arxiv.org/pdf/2511.08544v3#page=12)。

这一目标将各视图拉向全局视图表示的均值，并结合 SIGReg 训练，不使用 teacher–student 或 stop-gradient。对 RIND 来说，借用 SIGReg 与采用完整 LeJEPA 应分开比较：不同光源可能导致不同遮挡，将它们直接当作应获得相同表示的视图，可能压掉需要保留的变化。因此，可以保留带光源条件的预测任务，单独检验 SIGReg 是否改善训练

除了防坍塌，还需规定模型如何学习光源变化。前面用于规划的 V-JEPA 2，在动作条件训练阶段冻结视频编码器，训练 predictor 根据视觉表示、机器人状态与动作预测下一步，并把自己的预测继续作为输入，加入两步 rollout：

> “We also compute a two-step rollout loss to improve the model’s ability to perform autoregressive rollouts at inference time.”
>
> [PDF 第 9 页，第 3.1 节 *Loss function*](https://arxiv.org/pdf/2506.09985v1#page=9)。实际训练为两步，第 10 页图 6 采用四步示意。

这启发我们把光源变化作为明确的预测条件，并在需要连续预测时检查跨越遮挡事件后的误差积累。可以将事件附近的训练采样、光源条件和 rollout 分别做消融，判断哪些设计改善了迁移，而不是一次加入所有约束。

最后，训练稳定之后，还要检验新结构需要多少观测才能适配。RQ1 提到的 DeepSDF 为每个形状学习 latent code，与共享解码器共同训练；在推断新形状时：

> “During inference, decoder weights are fixed, and an optimal latent vector is estimated.”
>
> [PDF 第 4 页，图 4；优化过程见第 5 页第 4.2 节、式 9–10](https://arxiv.org/pdf/1901.05103v1#page=4)。

该机制还用于从单视角深度观测补全未见形状。[PDF 第 6–7 页，第 6.3 节 *Shape Completion*、表 4](https://arxiv.org/pdf/1901.05103v1#page=6)。我们可以据此比较不适配、固定共享模型后只拟合场景 code，以及小规模参数适配，并记录性能随观测数量的变化。DeepSDF 提供了适配方式的依据；它并未证明少量亮暗观测足以让模型适配新结构，这仍需在 G1 上检验。

## References

1. Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., & Ballas, N. [**Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture**](https://arxiv.org/abs/2301.08243). *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2023.

2. Saito, A., Kudeshia, P., & Poovvancheri, J. [**Point-JEPA: A Joint Embedding Predictive Architecture for Self-Supervised Learning on Point Cloud**](https://arxiv.org/abs/2404.16432). *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*, 2025.

3. Park, J. J., Florence, P., Straub, J., Newcombe, R., & Lovegrove, S. [**DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation**](https://arxiv.org/abs/1901.05103). *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2019.

4. Ren, S., Padilla, W., & Malof, J. [**Benchmarking Deep Inverse Models over Time, and the Neural-Adjoint Method**](https://arxiv.org/abs/2009.12919). *Advances in Neural Information Processing Systems (NeurIPS)*, 2020.

5. Assran, M., Bardes, A., Fan, D., Garrido, Q., Howes, R., Komeili, M., et al. [**V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning**](https://arxiv.org/abs/2506.09985). *arXiv preprint arXiv:2506.09985*, 2025.

6. Carroll, M., Paradise, O., Lin, J., Georgescu, R., Sun, M., Bignell, D., Milani, S., Hofmann, K., Hausknecht, M., Dragan, A., & Devlin, S. [**Uni[MASK]: Unified Inference in Sequential Decision Problems**](https://proceedings.neurips.cc/paper_files/paper/2022/hash/e58fa6a7b431e634e0fd125e225ad10c-Abstract-Conference.html). *Advances in Neural Information Processing Systems (NeurIPS)*, 2022.

7. Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. [**Shortcut Learning in Deep Neural Networks**](https://doi.org/10.1038/s42256-020-00257-z). *Nature Machine Intelligence*, 2, 665–673, 2020.

8. Bardes, A., Ponce, J., & LeCun, Y. [**VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning**](https://arxiv.org/abs/2105.04906). *International Conference on Learning Representations (ICLR)*, 2022.

9. Balestriero, R., & LeCun, Y. [**LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics**](https://arxiv.org/abs/2511.08544). *arXiv preprint arXiv:2511.08544*, 2025.
