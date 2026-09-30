# RIND Phase I

[English](README.md) | 中文

**当前状态：**项目结构与职责占位文件已建立，以下内容定义待完成的研究任务，并不表示功能已实现。原始 RIND 生成器与数据仍需接入。

补充文档：[数学定义与方法细节](docs/method.md) · [模块数据接口](docs/interfaces.md)。两份文档目前为英文，已与当前八个核心文件的职责和字段对齐。

## 项目目标

RIND Phase I 研究二维遮挡环境中的逆问题：

> 给定局部响应和观察窗口的位置，预测能够解释当前观测的源位置分布。

我们不直接回归唯一源坐标，而是学习连续二维源空间中的兼容性：

$$
O=(R,v), \qquad v=(x,y,\text{size})
$$

其中：

- $R$：局部响应；
- $v$：窗口位置和大小；
- $s=(s_x,s_y)$：候选源位置。

整体流程：

$$
O
\rightarrow
C(s;O)
\rightarrow
q^*(s\mid O)
\rightarrow
E_\theta(O,s)
\rightarrow
p_\theta(s\mid O)
$$

- $C(s;O)$：物理模型给出的候选代价；
- $q^*(s\mid O)$：物理教师分布；
- $E_\theta(O,s)$：学生条件能量；
- $p_\theta(s\mid O)$：学生预测分布。

物理教师可以访问完整场景和障碍物，并通过 RIND 重渲染候选位置；学生只使用局部响应、窗口信息和候选坐标。训练完成后，学生不再需要逐候选调用物理渲染。

Phase I 只考虑：**单源、固定强度、二维连续空间、遮挡，无距离衰减、反射、噪声和材料差异。**

---

## 数据与观察窗口

完整场景范围为：

$$
[0,1024]\times[0,1024]
$$

其中 $x$ 向右、$y$ 向下，数组索引为 `[row, col]`。窗口内像素 `[row, col]` 对应世界坐标中心 `(x + col + 0.5, y + row + 0.5)`。全文 `size` 均表示窗口在世界坐标中的边长。

观察窗口由**源驱动的动态四分树**生成：

1. 从完整场景开始；
2. 含源区域继续四等分；
3. 不含源区域停止，并作为 observation；
4. 最小边长为 16；
5. 最终仍含源的最小叶节点不作为 observation。

分割线上的源归右侧或下侧子块，世界最外边界归末端子块；窗口是否包含源使用同一规则，避免边界样本被重复计数。

因此 observation 定义为：

```text
scene_id
view_id
response
window = (x, y, size)
```

真实源坐标和完整场景几何只用于教师监督和评价，不作为学生输入。窗口大小可以不同，必须保留原始空间尺度。单源场景按上述规则从 1024 分割到 16，应产生 18 个无源窗口，可作为检查依据。

本阶段使用**新生成的单源、固定强度数据**，后续运行可读取这份已生成数据。训练、验证、测试始终按 `scene_id` 划分，同一场景的窗口不得跨集合。

---

## 项目结构

```text
.
├── README.md
├── README.zh-CN.md
├── pyproject.toml
├── .gitignore
├── docs/
│   ├── method.md
│   └── interfaces.md
├── configs/
│   └── phase1.json
├── src/rind_phase1/
│   ├── __init__.py
│   ├── data.py
│   ├── physics.py
│   ├── search.py
│   ├── teacher.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── checks.py
├── data/
│   ├── raw/
│   └── splits/
├── outputs/
│   ├── teachers/
│   ├── checkpoints/
│   ├── figures/
│   └── reports/
└── archive/
```

---

## 核心文件

### `data.py`

**输入**

```text
RIND 数据或生成器
scene 数量
固定 source intensity
random seed
quadtree 设置
split 设置
```

**这些输入分别做什么、怎么用**

- **RIND 数据或生成器**：原始 RIND 项目路径、生成接口或已生成实例路径。 提供场景几何和真实响应的来源。
  **怎么用：**首次调用生成器创建新单源数据；之后读取同一实例。不能用读取已有响应的接口代替改变源后的重渲染。

- **scene 数量**：需要新生成的独立场景数量，正整数。 确定数据规模和生成成本。
  **怎么用：**例如设置为 100，就生成 100 个独立场景；每个场景再产生多个观察窗口。这里的 100 是示例，不是已确定的实验规模。生成后核对场景数量。

- **固定 source intensity**：所有场景共用的源强度，取 RIND 接受的正值。 隔离强度变化，只研究位置和遮挡。
  **怎么用：**生成真实响应和候选重渲染时使用完全相同的值；不逐场景随机取值。

- **random seed**：生成器的随机数种子。 让相同设置的数据生成可复现。
  **怎么用：**初始化场景生成及相关随机步骤，并随数据保存；种子本身不保证跨划分无重复场景，仍需检查场景身份。

- **quadtree 设置**：世界大小、`min_window_size`、含源分割规则及边界归属规则。 定义哪些区域可以成为观测。
  **怎么用：**当前从 1024 × 1024 全场开始，含源区域继续四等分，无源区域保存为观察窗口；到 16 × 16 仍含源的区域不保存为观测。

- **split 设置**：训练、验证、测试的比例或显式场景名单。 避免同场景几何泄漏。
  **怎么用：**按场景分配后保存名单，再让窗口继承所属集合；比例总和应为 1，小规模试验应确认各必要集合非空。

**处理**

- 接入 RIND 生成新的单源、固定强度场景，之后支持读取，不覆盖旧实例；
- 构造动态四分树 observation；
- 按 `scene_id` 划分 train / validation / test；
- 保留窗口的世界坐标和尺度。

**输出**

```python
{
    "scene_id": ...,
    "view_id": ...,
    "response": ...,
    "window": [x, y, size]
}
```

数据写入：

```text
data/raw/
data/splits/
```

**目的**

建立统一的：

$$
\text{scene}
\rightarrow
\text{local observation}
$$

数据接口。

---

### `physics.py`

**输入**

```text
scene geometry
observed response
window
candidate source coordinate
fixed source intensity
physics loss settings
```

**这些输入分别做什么、怎么用**

- **scene geometry**：通过 `scene_id` 取得的连续障碍物类型、位置及形状参数。 决定候选源到窗口各位置的遮挡关系。
  **怎么用：**传给 RIND 重渲染并检查候选位置是否合法；仅教师和离线评价使用，不送入学生编码器。

- **observed response**：观测数组 `response[size,size]`。 作为候选解释要匹配的目标。
  **怎么用：**与候选响应逐位置比较，并从它提取边界权重；权重不能由候选响应决定。

- **window**：`[x,y,size]`，世界坐标中的左上角与边长。 指定要重渲染哪一块区域。
  **怎么用：**所有候选都渲染同一窗口并保持像素中心约定，使候选和真实响应逐像素对齐。

- **candidate source coordinate**：一个世界坐标 `[sx,sy]`，批量时为 `[N,2]`。 代表待验证的源位置假设。
  **怎么用：**先检查在世界内、窗口外及障碍物外，再将唯一源放到该位置渲染；不移动窗口或障碍物。

- **fixed source intensity**：与生成真实响应完全一致的强度。 保证代价反映位置差异，而非强度不一致。
  **怎么用：**将它与候选坐标组合为 RIND 所需的源参数，每个候选保持不变。

- **physics loss settings**：`alpha`、`beta`、`boundary_lambda`、`boundary_sigma`，以及边界提取、距离单位和空边界约定。 决定像素差异与边界几何差异如何评价。
  **怎么用：**`lambda` 控制边界附近加权幅度，`sigma` 控制范围，`alpha/beta` 汇总两项代价；`beta=0` 时跳过边界项。所有候选使用同一设置。

**处理**

将候选源 $s$ 放入原场景重新渲染：

$$
s \rightarrow \hat R(s)
$$

再比较真实响应 $R$ 与候选响应 $\hat R(s)$。

基础代价是边界加权的响应误差：

$$
L_{\text{resp}}(s)
=\frac{\sum_u w(u)|\hat R(s)(u)-R(u)|}{\sum_u w(u)}
$$

其中权重仅由真实响应的边界确定：

$$
w(u)=1+\lambda\exp\left(-\frac{d(u)^2}{2\sigma^2}\right)
$$

$d(u)$ 是像素到最近真实响应边界的距离。没有有效边界时使用均匀权重。固定强度应满足 `0 < intensity ≤ 1` 且为 RIND 接口接受的值；`α>0`、`β≥0`、`λ≥0`、`σ>0`。强度为零会使所有候选响应退化为零，不用于本阶段。

可选边界项直接比较两组边界的空间位置：

$$
L_{\text{edge}}(s)=D(E_R,E_s)+D(E_s,E_R)
$$

$E_R$ 与 $E_s$ 分别为真实响应和候选响应的边界；$D$ 表示单向边界距离。需统一距离单位，并明确边界为空、缺失或额外出现时的结果。

最终：

$$
C(s;O)
=
\alpha L_{\text{resp}}(s)
+
\beta L_{\text{edge}}(s)
$$

其中 `β=0` 对应 response-only baseline。先验证真实源能够重现观测；无效源不参与重渲染，`valid=False`，其代价不进入分布归一化。边界项关闭时 `L_edge` 可为空，并记录是否计算。

**输出**

```text
candidate_xy
candidate_response
L_resp
L_edge
physical_cost
valid
```

**目的**

建立：

$$
s \rightarrow C(s;O)
$$

即用物理模型评价某个候选源能否解释当前观测。

---

### `search.py`

**输入**

```text
world bounds
observation window
candidate spacing
search budget
physics evaluator
optional scene geometry
```

**这些输入分别做什么、怎么用**

- **world bounds**：允许搜索的全场边界，当前为 `[0,1024]²`。 限定候选位置的外部范围。
  **怎么用：**建立初始网格与空间单元；所有细分单元都不得越界。

- **observation window**：当前观测的 `[x,y,size]`。 排除已知不含源的观察区域。
  **怎么用：**从搜索范围中排除窗口内部并处理跨边界单元；默认不追加父节点范围限制。

- **candidate spacing**：世界坐标单位下的初始网格步长，对应配置 `candidate_spacing`。 控制初始覆盖精度和候选数量。
  **怎么用：**按此步长构造粗网格及单元边界；步长越小通常候选越多。它不是响应图的像素分辨率。

- **search budget**：自适应模式下允许的物理候选评价次数上限，对应 `adaptive_budget`。 控制教师构造成本。
  **怎么用：**在选择继续细分的区域时检查剩余预算；按评价过的候选计数，批量调用不算作一次候选评价。重算和边界项成本另行记录。

- **physics evaluator**：由 `physics.py` 提供、绑定了当前场景、观测和强度的评价能力。 告诉搜索过程哪些区域更能解释观测。
  **怎么用：**给它候选坐标，取得有效性与代价，用于筛选和细化；缓存结果供教师复用。

- **optional scene geometry**：当前场景的障碍物信息；也可封装在物理评价器内部。 提前排除非法候选并估计有效单元面积。
  **怎么用：**只在教师搜索中使用；若不直接传入，应由评价器完成几何检查。无几何部署不能复用这条教师搜索路径。

**处理**

先提供均匀网格候选，再扩展到 coarse-to-fine 自适应搜索：

```text
coarse sampling
→ physics evaluation
→ select promising regions
→ refine
```

对于非均匀候选，需要记录每个候选代表的空间面积：

$$
a_i
$$

以避免细采样区域因为候选更多而获得额外概率质量。这里采用均匀空间先验和基于空间单元的候选表示：`area_weight` 是候选所代表的单元面积，父单元细分后不能再与子单元重复计入。还需保存 `cell_bounds=[x_min,y_min,x_max,y_max]`，使单元覆盖范围可复查。跨越观察窗口或障碍物边界的单元，应裁剪、细分或记录有效面积近似；仅检查中心点合法，不能把整格面积当作精确有效面积。若改用其他采样方式，需要同步明确其权重含义。

候选应在世界范围内、观察窗口外；教师使用场景几何排除障碍物内部位置。保持必要的粗覆盖并检查遗漏的可行区域；仅对保留区域归一化时，应明确结果针对的是该候选覆盖范围。最终候选使用一致的物理代价，不能直接混合仅响应项与已加入边界项的评分。

**输出**

```text
candidate_xy
cell_bounds
valid
area_weight
physical_cost
search_level
num_physics_evaluations
```

**目的**

把连续源空间转换为有限候选集合，同时降低物理评价成本。

---

### `teacher.py`

**输入**

```text
candidate_xy
cell_bounds
physical_cost
valid
area_weight
temperature
scene_id / view_id
config_id
```

**这些输入分别做什么、怎么用**

- **candidate_xy**：`[N,2]` 候选世界坐标，由搜索模块提供。 明确每个教师概率对应哪个位置。
  **怎么用：**按原顺序保存；与所有候选数组一一对应，不单独重排。

- **cell_bounds**：`[N,4]`，每行 `[x_min,y_min,x_max,y_max]`。 记录候选代表的空间单元，便于核查覆盖和绘图。
  **怎么用：**随候选保存，检查父子单元不重复；跨障碍物单元的实际有效面积由 `area_weight` 和面积估计规则说明。

- **physical_cost**：`[N]` 最终物理代价。 确定候选对观测的解释程度。
  **怎么用：**在有效候选上计算 `exp(-cost/temperature)`；保留原代价用于绝对一致性评价。

- **valid**：`[N]` 布尔值，标记该候选是否属于所声明的有效范围。 排除不能作为源假设的位置。
  **怎么用：**只对有效项归一化，无效项概率为零；全无效时报告失败。

- **area_weight**：`[N]`，单元代表的有效空间面积或明确记录的近似。 避免细采样区域因候选更多而获得额外质量。
  **怎么用：**与 `exp(-cost/temperature)` 相乘后归一化；有效项权重为正，保留给学生使用。

- **temperature**：正标量 `tau`。 控制相近物理代价之间的概率差异。
  **怎么用：**用于 `cost/tau`；较小值更突出低代价区域。用验证数据选择，并随教师记录保存。

- **scene_id / view_id**：原始观测的身份标识。 将教师结果准确匹配回训练样本。
  **怎么用：**用于读取、缓存和检查划分归属，不作为神经网络输入。

- **config_id**：指向本次教师生成配置的可解析标识。 保证代价和分布结果可追溯。
  **怎么用：**随记录保存；对应配置应包含渲染版本、代价设置、搜索范围及面积估计规则，不能只是找不到来源的标签。

**处理**

下面的求和均仅包含 `valid=True` 的候选，且温度 `τ>0`。无效候选概率为 0；没有有效候选时应报告失败，不能生成伪造的均匀目标。

等面积的均匀候选时：

$$
q_i^*
=
\frac{\exp(-C_i/\tau)}
{\sum_j \exp(-C_j/\tau)}
$$

非均匀候选时：

$$
q_i^*
=
\frac{a_i\exp(-C_i/\tau)}
{\sum_j a_j\exp(-C_j/\tau)}
$$

这样多个物理上合理的位置可以同时保留较高概率。有效候选的面积必须为正；教师与学生必须使用相同的候选顺序、有效性和面积权重。

需要同时保存原始 `physical_cost`，因为概率总和始终为 1，并不能单独表示“当前观测是否存在好的解释”。

**输出**

保存到：

```text
outputs/teachers/
```

主要字段：

```text
scene_id
view_id
candidate_xy
cell_bounds
valid
area_weight
physical_cost
teacher_prob
temperature
config_id
```

**目的**

建立：

$$
C(s;O)
\rightarrow
q^*(s\mid O)
$$

把昂贵的物理评价转换成学生可以学习的监督分布。

---

### `model.py`

**输入**

```text
model config（初始化时使用）
local response
window = (x, y, size)
candidate_xy
valid
area_weight
```

**这些输入分别做什么、怎么用**

- **model config（初始化时使用）**：`coordinate_scale`、`fourier_frequencies`，以及实现时确定的网络结构设置。 定义坐标单位、频率范围和模型结构。
  **怎么用：**创建模型时读取并保存；训练与加载检查点必须一致。它是初始化设置，不是每条样本的额外观测。

- **local response**：`[size,size]` 的观测数组。 提供遮挡形状与有效边界等局部信息。
  **怎么用：**输入观测编码器；不同大小样本可按大小分组，或填充并屏蔽填充值。不能让填充区域被当作零响应证据。

- **window = (x, y, size)**：`[x,y,size]`。 告诉模型局部响应在世界中的位置和实际范围。
  **怎么用：**将位置及大小交给窗口编码部分，与响应表示组合；内部坐标按统一 `coordinate_scale=1024` 归一化，保存记录仍用世界坐标。

- **candidate_xy**：待查询的 `[N,2]` 世界坐标。 指定模型需要判断的假设位置。
  **怎么用：**使用与窗口一致的坐标尺度，再进行 Fourier 表示；逐候选输出能量，不重新运行物理过程。

- **valid**：当前预测候选集合的有效性标记。 规定分布归一化范围。
  **怎么用：**仅用于将能量转为概率，不送入编码器。教师辅助评价可用教师掩码；无几何预测只能使用公开可知的范围约束。

- **area_weight**：与当前候选同顺序的 `[N]` 空间权重。 将能量对应的密度评分转成单元概率质量。
  **怎么用：**按 `area_weight * exp(-energy)` 归一化；不影响能量函数本身。

**不作为学生特征：**真实源、场景 ID、完整场景几何、障碍物掩码、教师物理代价和完整四分树。`valid` 与 `area_weight` 是归一化元数据，不能作为额外的几何编码输入。

**处理**

首先编码 observation：

$$
R \rightarrow h_R
$$

同时编码窗口位置和大小：

$$
v \rightarrow h_v
$$

得到：

$$
h_O=f(h_R,h_v)
$$

候选坐标使用 Fourier features：

$$
s\rightarrow\gamma(s)
$$

最后输出条件能量：

$$
E_\theta(O,s)
$$

能量越低表示候选越合理。能量函数本身只接收响应、窗口信息和候选坐标；`valid` 与 `area_weight` 用于随后将评分转成分布，不作为编码特征。模型应支持不同窗口大小，并能查询新的连续坐标。

在有效候选集合上归一化，无效候选概率为 0：

$$
p_i
=
\frac{a_i e^{-E_i}}
{\sum_j a_j e^{-E_j}}
$$

仅当有效单元等面积时，所有 $a_i$ 才相同；规则网格在经过边界裁剪后也可能需要不同权重。

**输出**

```text
energy
probability
```

**目的**

学习一个快速的候选兼容性函数：

$$
(R,v,s)
\rightarrow
E_\theta(O,s)
$$

从而替代推理阶段的逐候选物理重渲染。

---

### `train.py`

**输入**

```text
observation
teacher candidate set
teacher_prob
valid
area_weight
student model
training config
```

**这些输入分别做什么、怎么用**

- **observation**：训练或验证集合中的响应、窗口及样本标识。 提供学生条件输入，并确定监督记录。
  **怎么用：**用标识匹配教师记录，只将响应和窗口送入模型；训练集用于更新，验证集用于选择模型。

- **teacher candidate set**：教师记录中的 `candidate_xy`，附带对应空间单元信息。 确定这一训练样本需要比较的位置。
  **怎么用：**把同一候选坐标输入学生，保持与教师概率完全一致的顺序。

- **teacher_prob**：`[N]` 的软监督概率，来自 `teacher.py`。 表达多个源假设的相对兼容性。
  **怎么用：**作为 KL 或软标签交叉熵的目标，不改成单个最大概率位置的标签。

- **valid**：教师记录中的候选有效性。 保证教师和学生比较的是同一范围。
  **怎么用：**对双方使用同一掩码；不将无效代价或无效位置加入损失。

- **area_weight**：教师记录中的空间权重。 保证两种分布使用同一空间度量。
  **怎么用：**用于学生概率归一化；`teacher_prob` 已经包含面积因素，不能再把它额外乘一次面积。

- **student model**：`model.py` 定义的可训练条件能量模型。 承载待学习的逆向兼容性函数。
  **怎么用：**从观测和候选生成能量与概率，通过分布损失更新参数；保存模型权重。

- **training config**：设备、批量大小、轮数、学习率、随机种子及检查点选择规则等。 控制训练资源、更新过程和结果复现。
  **怎么用：**按配置运行并保存实际设置；使用验证指标选检查点，测试数据不参与训练或选择。

**处理**

学生在与教师相同的候选集合上计算：

$$
E_i=E_\theta(O,s_i)
$$

并得到学生分布：

$$
p_\theta(s_i\mid O)
$$

训练目标是让：

$$
p_\theta(s\mid O)
\approx
q^*(s\mid O)
$$

例如使用：

$$
D_{\mathrm{KL}}(q^*\Vert p_\theta)
$$

或 soft-label cross entropy。

这里不要求：

$$
E_i=C_i
$$

也不直接把真实源坐标作为唯一回归目标。先验证小规模样本上的学习，再扩大训练；模型选择只使用验证集，测试场景不参与调参。

**输出**

保存到：

```text
outputs/checkpoints/
```

包括：

```text
model weights
optimizer state
training history
validation history
experiment config
```

**目的**

把物理教师的源空间判断能力蒸馏到学生 energy model 中。

---

### `evaluate.py`

**输入**

```text
test observations
teacher records
trained model
experiment config
```

**这些输入分别做什么、怎么用**

- **test observations**：已固定测试场景中的观测及标识。 检验未见场景上的表现。
  **怎么用：**输入模型并按场景、窗口大小汇总；真实源可作图像参考，但不能作为模型输入。

- **teacher records**：同一批观测在评价候选集合上的代价、概率、有效性和面积。 提供物理参照与分布参照。
  **怎么用：**让学生在相同候选上评分后比较。若保存的是不同自适应候选，需要先通过物理评价补齐共同参考网格上的教师记录，不直接比较不同向量。

- **trained model**：指定检查点及其模型配置。 生成待评价的学生结果。
  **怎么用：**加载后固定参数进行预测；不在测试集上继续拟合。无几何预测单独使用公开候选集。

- **experiment config**：候选范围、网格/面积约定、指标、消融设置、检查点标识和运行环境。 保证结果可比且可追溯。
  **怎么用：**冻结评价规则；区分教师辅助范围和无几何结果，记录各次实验变动项及实际耗时。

**处理**

主要比较：

$$
q^*(s\mid O)
$$

与：

$$
p_\theta(s\mid O)
$$

并分析：

- teacher / student distribution agreement；
- 学生高概率位置是否具有低 physical cost；
- 多峰和不确定区域是否被保留，尤其是单条有效边界与两条独立边界的对比；
- unseen scene / obstacle configuration 泛化；
- 不同窗口尺度表现；
- `Lresp` 与 `Lresp + Ledge`；
- uniform 与 adaptive search 的质量和计算成本。

分布指标使用共同的评价候选范围与面积约定。热图若比较不同单元大小，显示 `probability / area_weight` 得到的分段密度，而非直接把单元概率当作点密度；区域概率则对单元质量求和。跨搜索策略比较时优先使用共同的均匀参考网格，不能直接比较不同候选点数下的概率向量。报告实际未见的障碍物组合；若尚未构造此划分，应明确其评估未完成。

**输出**

```text
outputs/figures/
outputs/reports/
```

典型图像：

```text
Observation
→ Physical Cost
→ Teacher Distribution
→ Student Distribution
```

**目的**

判断学生是否真正恢复了物理模型定义的源空间约束，而不仅仅是预测一个接近真实源的坐标。

---

### `checks.py`

**输入**

```text
少量 RIND scenes
phase1 config
```

**这些输入分别做什么、怎么用**

- **少量 RIND scenes**：可读取的真实场景、观测与真实源参数。 提供能够核对物理正确性的具体例子。
  **怎么用：**检查窗口与划分，使用真实源重渲染，再串联候选、教师和学生；未训练输出必须标注。

- **phase1 config**：当前实际使用的完整配置文件。 使联调与各模块使用同一约定。
  **怎么用：**检查必需项已填写，按同一强度、窗口、坐标和权重规则执行；记录失败模块和配置标识。

**处理**

串联：

```text
data
→ search
→ physics
→ teacher
→ model
```

检查数据划分无场景重叠、窗口不含源、真实源重渲染一致，以及教师和学生的候选顺序、有效性与概率总和正确。必要时加入一个小规模训练过程，并调用评价模块输出一张对比图。

**输出**

```text
sample observation
candidate set
physical cost
teacher probability
student energy
student probability
check results / failing stage
```

**目的**

提供一个统一的小规模 end-to-end 入口，方便各模块开发和接口联调。

---

## 其余目录

### `configs/phase1.json`

保存整个实验的统一配置，包括：

```text
data settings
quadtree settings
physics weights
teacher temperature
search settings
model settings
training settings
```

`null` 表示尚未确定，不是可运行默认值。至少在运行前确定 RIND 路径、固定强度、场景数量及必要的搜索和训练参数。

### `data/raw/`

保存本阶段新生成的 RIND 场景，沿用原生数据格式；响应可能以压缩可见性保存并在读取时恢复：

```text
scene geometry
obstacles
source
source intensity
rendered response
metadata
```

### `data/splits/`

保存：

```text
train.json
validation.json
test.json
```

按 `scene_id` 划分。

### `outputs/teachers/`

保存每个 observation 的物理候选评价和教师分布。

### `outputs/checkpoints/`

保存训练后的学生模型和训练记录。

### `outputs/figures/`

保存物理代价、教师分布、学生分布和搜索过程可视化。

### `outputs/reports/`

保存实验指标、消融结果和阶段总结。

### `archive/`

保存旧版设计、旧模块和不再参与当前 Phase I 的内容。旧文档中的路径和职责不再作为当前任务要求。

`pyproject.toml` 保存项目定义和依赖；`.gitignore` 排除大型产物与缓存；`__init__.py` 标记 Python 包；空目录中的 `.gitkeep` 仅保留目录。

---

## 标准数据接口

### Observation

```python
{
    "scene_id": ...,
    "view_id": ...,
    "response": ...,
    "window": [x, y, size]
}
```

### Teacher record

```python
{
    "scene_id": ...,
    "view_id": ...,
    "candidate_xy": ...,
    "cell_bounds": ...,
    "valid": ...,
    "area_weight": ...,
    "physical_cost": ...,
    "teacher_prob": ...,
    "temperature": ...,
    "config_id": ...
}
```

候选相关数组按同一顺序排列：`candidate_xy` 为 `[N, 2]`，`cell_bounds` 为 `[N, 4]`，`valid`、`area_weight`、`physical_cost` 和 `teacher_prob` 为 `[N]`。`temperature` 是标量，`config_id` 是配置记录标识。`response` 为 `[size, size]`，`window` 为 `[3]`。

### Student output

```python
{
    "candidate_xy": ...,
    "energy": ...,
    "probability": ...
}
```

---

## 研究解释与边界

- **教师目标的含义：**当前目标是固定窗口下、给定场景的响应兼容性 Gibbs 分布，而非已校准的真实源后验。候选不必重新生成相同的四分树叶节点。单源规则下，相同叶节点的生成确实要求源位于其父节点内且在该窗口外；当前关闭父节点限制，是明确选择只评价响应兼容性。如果以后要研究包含窗口选择机制的完整条件后验，需要加入该规则，并重新生成教师目标。
- 教师依赖隐藏场景几何，严格来说其目标为 $q^*(s\mid O,S)$。学生只看到 $O$；相同观测可能对应不同场景的兼容性分布，因此不能保证逐场景精确恢复教师结果。
- **评价与部署分开：**教师有效范围上的一致性用于判断蒸馏是否成功。无几何预测必须在预先声明的完整公开候选范围上独立运行，不能复用依赖场景几何或物理代价的有效掩码/自适应候选集。仅在教师保留范围内训练，不会自动约束被排除位置的能量；无几何结果需单独报告，不能由教师有效范围上的表现推断。

## 首轮交付与认领

首轮检查：**2026 年 10 月 1 日（周四）**。人员与算力待定，组员自行认领；可一人负责多个文件。

| 工作 | 文件 | 负责人 | 最小交付 |
|---|---|---|---|
| 数据 | `data.py` | 待认领 | 小份新数据、窗口示例及场景划分 |
| 物理监督 | `physics.py`、`teacher.py` | 待认领 | 真实源重渲染验证、一个观测的代价与教师分布 |
| 候选搜索 | `search.py` | 待认领 | 均匀候选基线及自适应搜索的后续交付计划 |
| 模型与训练 | `model.py`、`train.py` | 待认领 | 不同窗口大小的评分，监督就绪时完成小规模学习 |
| 评价 | `evaluate.py` | 待认领 | 教师/学生对比图或已完成阶段的图像与指标 |
| 联调 | `checks.py` | 各模块负责人 | 已接通环节与失败环节清单 |

先完成一个观测的完整链路；最终仍需完成自适应搜索质量/成本对比、边界项消融及未见场景评价。实际结果与待实现内容应分别标注。分工变更同步英文 README。

## 核心研究问题

Phase I 研究两个核心问题：

$$
\boxed{
\text{Can a neural energy model learn physics-based source-space compatibility?}
}
$$

$$
\boxed{
\text{Can it preserve multimodal solutions and uncertainty without rerendering every candidate?}
}
$$
