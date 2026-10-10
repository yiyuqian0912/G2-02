# 多源教师：怎么理解，怎么使用

[English README](../README.md) · [中文 README](../README.zh-CN.md) · [数学约定](method.md) · [接口表](interfaces.md)

## 1. 它做什么？

给教师一个局部观测：响应、局部障碍物和窗口位置。教师在完整场景几何中尝试很多组窗口外的源，检查每组能否产生相近的局部响应，最后给每组参数一个概率。

它是物理计算流程，当前无需训练一个教师神经网络。多源时，一组答案包含多个位置和各自强度；这些参数需要一起评价，因为它们的贡献会叠加。同一局部响应可以对应不同位置、不同数量和不同强度组合。

学生输入仍只有局部响应、局部障碍物和窗口 `(x,y,L)`。完整场景几何只供教师和评价使用。教师记录是监督目标；其中的全局有效性、真值和诊断信息不能变成学生的额外观测输入。

## 2. 常用名词

| 名词 | 大白话 |
|---|---|
| 候选 / hypothesis | 一整组可能的源，例如两个源及其位置、强度 |
| 联合分布 / joint target | 每一整组参数的概率，保留哪些源相互搭配 |
| 物理代价 C | 候选总响应与观测逐像素的平均绝对强度差；越小越接近 |
| 位置间距 / spacing | 候选源位置网格的世界单位间距，与观测像素精度不是同一件事 |
| 每种源数的抽样数 | 每个 K 抽多少整组参数；不是抽多少个源坐标 |
| 源数先验 / count prior | 算响应前，分别给 K=1、2、3……多少总权重 |
| 基础权重 / base mass | 每个样本代表多少先验权重，通常为该 K 的先验除以该 K 的抽样数 |
| 温度 tau | 控制代价转概率的锐利程度；较小更偏向低误差组合 |
| 二维投影 / marginal | 把整组参数的概率汇总到位置上，查看某处是否可能有源 |
| 完成 / complete | 声明要抽的候选全算完了；连续世界里的所有组合仍可能没有覆盖 |

响应是强度相加，允许大于 1。比如两个可见源强度为 .7 和 .8，总响应就是 1.5；教师按 1.5 比较。

## 3. 开始运行

先安装本项目的多源数据。生成目标本身不需要 PyTorch，也不需要绘图库。需要出图时安装 diagnostics：

```bash
uv sync --locked --extra diagnostics
uv run --extra diagnostics rind-multi-teacher \
  --scene-id 1 --view-id 13 \
  --spacing 64 --samples-per-count 1024 --temperature 0.05 \
  --counts 1 2 3 4 --count-prior 0.25 0.25 0.25 0.25 \
  --seed 20260923 --diagnostics --plot \
  --output outputs/teachers/example-scene1-view13.npz
```

上面是演示参数，配置没有把它们设为正式实验默认值。输出三个文件：

| 文件 | 用途 |
|---|---|
| `.npz` | 全部候选参数、代价、概率和可复现元数据 |
| `.diagnostics.json` | 源数分布、歧义指标、前十组参数、参考解检查 |
| `.png` | 原始／候选响应、局部掩码、空间投影与源数概率 |

只生成目标时省略 `--diagnostics --plot`，运行 `uv run rind-multi-teacher ...` 即可。模块入口 `python -m rind_phase1_multi.teacher` 也可用。合并环境需求时可使用 `uv sync --locked --extra train --extra diagnostics`。

源数已由实验条件明确指定时，例如 K=2，用 `--counts 2` 并省略先验。源数未知时显式列出 K 与先验；教师不会读取真实数量、位置、强度或参考解来提出候选。候选 K 可超过生成上限四，是否这样做由实验明确决定。

所有候选源都在世界内、观测窗口外、障碍物外。quadtree 只用于产生窗口，没有父节点限制。

## 4. 在代码中调用和取值

```python
from rind_phase1_multi.data import Phase1MultiDataset
from rind_phase1_multi.teacher import (
    generate_joint_teacher, save_teacher_record, load_teacher_record,
)

ds = Phase1MultiDataset()
t = generate_joint_teacher(
    ds, 1, 13,
    spacing=64, temperature=.05, seed=20260923,
    source_counts=[1, 2, 3, 4],
    samples_per_count=1024, count_prior=[.25, .25, .25, .25],
)
save_teacher_record(t, "outputs/teachers/example.npz")
t = load_teacher_record("outputs/teachers/example.npz")

for i in range(len(t["teacher_prob"])):
    K = int(t["source_counts"][i])
    sources = t["sources"][i, :K]  # float64 [K,3]，(x,y,强度)
    C = t["physical_cost"][i]     # 平均绝对强度误差
    q = t["teacher_prob"][i]      # 这整组参数的概率
    valid = t["valid"][i]         # 是否满足物理定义域
```

`sources` 保存为补零数组 `[N,K_max,3]`。只读前 K 行，其余没有源的含义。用相同的 i 配对参数、代价和概率。交换一组内部源的排列会去重；同位置强度拆分成不同数量的源仍保留为不同假设。

`metadata["scene_id"]`、`metadata["view_id"]` 对应原始观测；`window` 保留世界坐标。元数据还记录数据包与清单摘要、代码摘要、抽样规则、随机数版本、温度、配置／运行标识与实际计算次数。NPZ 只含数值／字符串数组，可以用 `np.load(path,allow_pickle=False)` 读取。

### 二维图怎么取？

```python
from rind_phase1_multi.diagnostics import source_space_projections
maps = source_space_projections(t)
presence = maps["presence"]
expected_count = maps["expected_count"]
expected_intensity = maps["expected_intensity"]
conditional_min_cost = maps["minimum_joint_cost"]
coverage = maps["hypotheses_per_position"]
```

这些图为 `[Ny,Nx]`，轴坐标在 `x_axis`、`y_axis` 中：

- `presence`：这个网格位置至少有一个源的概率，单格为 0–1，整图之和不要求为 1。
- `expected_count`：这个位置的预期源数量；整图求和为预期 K。同位置两个源会计两次。
- `expected_intensity`：这个位置的预期总源强度；同位置的强度会叠加。
- `minimum_joint_cost`：抽到的、包含这个位置的整组候选中，最小的代价。不能理解成只放一个源的代价。
- `coverage`：多少个有效组合用到这个位置。0 表示没有抽到，-1 表示该网格位置被窗口／几何排除。

浮点图中被排除的格子是 NaN。未抽到的位置在经验概率图里为 0、代价图里为 NaN；不能据此判定连续世界里那里不可能有源。投影丢失源之间的关联，训练完整组合分布时使用 `teacher_prob`。

## 5. 这个真实实例说明什么？

上面命令使用 scene 1、view 13：窗口为 `(128,384,128)`，观测中有障碍物，总强度可达约 2.119。

实测抽了 4096 组参数（每个 K 抽 1024 组），使用 244 个不同位置的可见性计算。教师总概率按 K 汇总约为：

| 候选源数 | 教师概率 |
|---|---:|
| 1 | 0.007% |
| 2 | 5.887% |
| 3 | 25.493% |
| 4 | 68.613% |

概率最高的一组有四个源，位置／强度约为：

```text
(32, 288, .596)
(224, 800, .122)
(288, 288, .758)
(352, 864, .795)
```

这组的平均响应误差约为 .09075，在这份有限抽样上的概率约 5.02%。输出仍保留全部 4096 组。联合熵约 6.056 nats，有效候选数量约 125.8，可以看到多种参数仍分担概率。

这个粗位置网格加随机强度的样本没有零代价组合；另外六组数据参考组合物理检查代价为零。参考组合没有加入抽样。有限抽样会漏掉精确解，使用目标前需要检查采样规模和间距是否足够。

## 6. 计算预算与自己的候选

`--max-evaluations` 限制新计算的整组代价数量；`--max-source-renders` 限制不同位置的单位可见性重渲染数量。多个组合／强度共用位置时可以复用缓存，缓存每个采样像素只占一 bit。预算不够时明确失败，不生成被截断的最终教师目标。

自己提出候选时，同样要声明它们的采样权重：

```python
from rind_phase1_multi.physics import JointResponseEvaluator
from rind_phase1_multi.search import make_joint_support, evaluate_joint_support
from rind_phase1_multi.teacher import target_from_search

support = make_joint_support(
    [
        [[32, 288, .6], [352, 864, .8]],
        [[288, 288, .7], [224, 800, .2]],
    ],
    base_mass=[.5, .5],  # 这里明确声明：这两个自定义候选同权
    metadata={"proposal": "caller_declared_two_set_demo"},
)
evaluator = JointResponseEvaluator(ds, 1, 13)
result = evaluate_joint_support(support, evaluator)
q = target_from_search(result, temperature=.05)
```

若混合 K 或不同采样规模，不能靠“所有行同权”默认决定源数先验。内置采样器给每次抽样 `pi_K/N_K` 的权重，再按整组物理误差一起归一化；如果所有候选代价一样，汇总后的 K 概率就是指定先验。

## 7. 正式实验前还要决定什么？

配置中的间距、抽样数、温度、候选 K 和先验保持为空，运行时必须明确提供。需要比较更细位置网格、更大样本量、多个种子，以及各窗口大小的稳定性。熵会随支持点规模改变，应在可比采样策略下比较。

当前实现是透明的联合抽样基线。自适应搜索、强度的其他先验或拟合方式、连续解区域的连通性估计，以及学生网络／训练仍属后续研究工作。数据参考组合是物理正确性的检查证据，不是教师标签或完整逆解集合。

## English summary

The teacher is a physical compatibility evaluator over complete source sets. It uses reproducible stratified Monte Carlo draws: iid valid uniform-grid positions with replacement, iid uniform strengths, and an explicit count prior. Normalize `base_mass * exp(-MAE/tau)` across all valid sets, preserving raw costs and ordering. Completion applies to the declared samples, not exhaustive inverse coverage. Source-space projections are diagnostic marginals, not the joint target. See the [English README](../README.md) and [method](method.md) for the same semantics.
