# RIND Phase I

[English](README.md) | 中文

**当前状态：**ZIP 安装、数据读取、本地可视化、场景子集、按尺寸组批和参考解重渲染已实现。均匀源空间搜索、响应物理代价、稳定的软教师分布、记录存取和逆空间诊断也已实现。自适应剪枝、边界损失、学生模型、训练与学生评价仍待完成。

补充文档：[数学定义与方法细节](docs/method.md) · [模块数据接口](docs/interfaces.md)。另有 [教师基线使用与解释](docs/teacher-baseline.md)，包含可运行命令、Python 调用、记录读取和诊断解释。

## 项目目标

RIND Phase I 研究二维遮挡环境中的逆问题：

> 给定局部响应、局部障碍物掩码和观察窗口的位置，预测能够解释当前观测的源位置分布。

我们不直接回归唯一源坐标，而是学习连续二维源空间中的兼容性：

$$
O=(R,M,v), \qquad v=(x,y,\text{size})
$$

其中：

- $R$：局部响应；
- $M$：仅当前窗口内的障碍物掩码；
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

物理教师可以访问完整场景和障碍物，并通过 RIND 重渲染候选位置；学生只使用局部响应、局部障碍物掩码、窗口信息和候选坐标，不接收窗外障碍物或完整场景几何。训练完成后，学生不再需要逐候选调用物理渲染。

Phase I 只考虑：**单源、固定强度、二维连续空间、遮挡，无距离衰减、反射、噪声和材料差异。**

---

## 数据集：下载、安装与使用

已生成的**单源、强度固定为 1 的 RIND 数据集**含 10,000 个场景、180,000 个局部观测，每场景 18 个。每个场景是 1024 × 1024 的连续二维世界，包含障碍物和一个源。源到像素中心的线段未受遮挡时响应为 1，否则为 0；障碍物内部也为 0，并另有障碍物掩码。栅格只规定采样精度，源坐标和障碍物边界始终保持连续。

### 首次安装

1. 从 [Google Drive](https://drive.google.com/file/d/1_zerwmggBskUtPtkhWTb6Xqs4kIU5L5I/view?usp=sharing) 下载 ZIP，直接放到**本 README 与 `pyproject.toml` 所在的项目根目录**，不用手动解压。
2. 若尚未安装 uv，按 [uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/)安装。
3. 在项目根目录运行：

```bash
./scripts/install_data.sh
```

项目声明 **Python >=3.10**，不固定小版本。uv 会选择符合要求的解释器及锁文件中对应的依赖版本；NumPy、Numba 和可选的 PyTorch 仍须支持该解释器和平台。基础读取和重渲染已在 Python 3.12、3.13 实测。已有 `.venv` 会继续使用原来的解释器；需要明确选择其他版本时，例如运行 `UV_PYTHON=3.11 ./scripts/install_data.sh`，uv 可能据此重建环境。

脚本根据 `uv.lock` 在本项目的 `.venv` 中安装模型项目和数据读取器，校验 ZIP 内部的 SHA256 校验和，仅解压数据到 `data/raw/phase1-single-source/`，检查数组形状与源强度，成功后删除 ZIP。校验失败时保留 ZIP 并清理临时解压目录。下载包中的 Python 代码和环境不会被安装。重复运行会检查已有数据，不会重复解压。当前数据及安装过程请预留约 3 GB 空间，基础读取环境还需额外空间；大数组使用内存映射，只在读取时解码需要的窗口。

成功解压后也会删除旁边对应的 `.zip.sha256` 文件（若存在）。已安装数据、`.venv` 和 uv 的共享依赖缓存会保留。若数据已经安装，脚本会跳过解压，不删除后来放入的 ZIP；首次安装时加 `--keep-archive` 也可保留 ZIP。

根目录有多个 ZIP 时，可指定文件；希望保留 ZIP 时加 `--keep-archive`：

```bash
./scripts/install_data.sh ./RIND-adaptive-v3-10000scenes-single.zip --keep-archive
```

数据需要放到另一块硬盘时，安装和运行模型均使用同一设置：

```bash
export RIND_DATA_ROOT=/mnt/data/rind-phase1-single
./scripts/install_data.sh
```

也可在安装时使用 `--data-root /path/to/data`，在代码中使用 `Phase1Dataset(root="/path/to/data")`。脚本接受带参考位置的单源发布包，会拒绝多源包。bash 入口适用于 Linux/macOS；等价的 Python 入口是先运行 `uv sync --locked`，再运行 `uv run --no-sync rind-install-data`。

### 浏览已安装的数据

```bash
./scripts/browser.sh
```

脚本自动安装可选的 Flask/Pillow 依赖，并在 `http://127.0.0.1:8766` 打开原来的 RIND 可视化页面。它与 `Phase1Dataset` 共用 `data/raw/phase1-single-source/` 或 `RIND_DATA_ROOT` 指定的数据，不需要另一份数据或独立的数据集环境。

页面保留场景缩略图、连续障碍物和源位置叠加、不同尺寸的局部窗口、响应通道选择、诊断筛选和像素响应数值检查。当前为单源、强度固定为 1 的版本，因此总响应与唯一源的贡献相同。

```bash
./scripts/browser.sh --no-open --port 8770
./scripts/browser.sh --data-root /path/to/data
# 安装 browser 可选依赖后的等价 Python 入口：
uv run --extra browser rind-browser
```

### 在代码里读取观测

用 `uv run python` 启动 Python，或在 IDE 中选择本项目的 `.venv`：

```python
from rind_phase1.data import Phase1Dataset

ds = Phase1Dataset()  # 默认安装位置，或 RIND_DATA_ROOT
print(ds.num_scenes, len(ds))  # 10000 个场景、180000 个观测
sample = ds[0]
R = sample["response"]
M = sample["obstacle"]        # bool [L,L]，仅局部窗口
window = sample["window"]
s, v = sample["scene_id"], sample["view_id"]
print(R.shape, R.dtype)       # (L, L)，float32
print(window)                # 世界坐标 [x, y, L]
```

| 观测字段 | 含义 |
|---|---|
| `scene_id`、`view_id` | 整数标识，用于匹配教师记录，不作为模型特征 |
| `response` | NumPy `float32 [L,L]`，原始采样精度下的 0/1 响应 |
| `obstacle` | NumPy `bool [L,L]`，true 表示窗口内障碍物 |
| `window` | NumPy `int64 [3]`，左上角世界坐标 `(x,y)` 和边长 `L` |

`ds[i]` 返回这五个字段。学生使用响应、局部掩码和窗口；真实源、参考解、完整几何与窗外障碍物仅供教师/评价，通过独立方法取得。数组索引为 `[row,column]`，x 向右、y 向下；像素 `(row,column)` 的采样中心为 `(x+column+0.5,y+row+0.5)`。窗口尺寸有 16、32、64、128、256、512 六种。即使编码器内部缩放图像，也必须保留窗口的原始位置和尺度。

### 按场景划分与 PyTorch DataLoader

训练前确定并保存场景划分。下面的比例只是使用示例，不是已决定的实验协议：

```python
import json
from pathlib import Path
from rind_phase1.data import Phase1Dataset, split_scene_ids, make_dataloader

all_data = Phase1Dataset()
splits = split_scene_ids(all_data.num_scenes, ratios=(0.8, 0.1, 0.1), seed=20260923)
Path("data/splits").mkdir(parents=True, exist_ok=True)
for name, ids in splits.items():
    Path(f"data/splits/{name}.json").write_text(json.dumps(ids))
train_data = Phase1Dataset(scene_ids=splits["train"])
loader = make_dataloader(train_data, batch_size=8, num_workers=0)
for batch in loader:
    R = batch["response"]      # torch.float32 [B,L,L]
    M = batch["obstacle"]      # torch.bool [B,L,L]，仅局部
    window = batch["window"]   # torch.int64 [B,3]
    break
```

PyTorch 为可选依赖，用 `uv sync --locked --extra train` 安装，并用 `uv run --extra train python` 运行上述例子。默认安装仅包含读取器和 CPU 渲染器。`make_dataloader` 按窗口尺寸组批，同一批次可直接堆叠，不需要填充或改变采样精度。每个尺寸组的最后一批可能不足 `batch_size`；每轮调用 `loader.batch_sampler.set_epoch(epoch)` 可得到可复现的新顺序。在使用 spawn 的平台上增大 `num_workers` 时，将 DataLoader 的创建放在 `if __name__ == "__main__":` 内。同一场景的所有窗口必须在同一集合；后续实验读取保存的场景名单，避免每次重新改变划分。

### 参考解、重新渲染与树状读取

```python
import numpy as np

references = ds.get_candidates(s, v)  # 10 组，每组 float64 [1,3]
source_xy = references[1][0, :2]      # (x,y)，第 3 列为强度 1
fresh = ds.rerender(s, sample["window"], source_xy)
assert np.array_equal(fresh, sample["response"])
scene = ds.get_scene(s)              # 真实源和连续障碍物几何
region = ds.get_region(s, 0, 0, 32)  # 独立通道、总响应、障碍物掩码
root = ds.get_view_tree(s)           # 仅恢复树的元数据，不解码图像
for leaf in ds.iter_view_leaves(s):
    observation = ds.get_observation(s, leaf["view_id"])
```

第 0 组参考解是真实源，其余 9 组位置不同，均位于原含源的 16 × 16 格子内，已逐组验证与当前窗口的采样响应完全一致。它们是兼容位置示例，不是全部逆解，也不是教师概率分布；部分位置差异很小。它们不保证在窗口外或采样中心之间的所有连续位置都产生相同响应。教师搜索仍需覆盖实验规定的候选范围。不能把这十个位置追加到均匀网格后归一化，也不能把它们直接当作概率目标。

`rerender` 在原场景中放置一个强度为 1 的假设源，返回 NumPy `float32 [L,L]`，拒绝世界范围外及障碍物内部的源。搜索模块还需按协议排除观察窗口内部位置。该 CPU NumPy/Numba 渲染器用于教师代价与评价，不能通过它对源坐标反向传播。首次调用可能因 Numba 编译而较慢。几何和参考坐标保留 `float64`，观测响应由适配层转换为 `float32`。

树节点的 `kind` 为 `split`、`view` 或 `occupied`。`view` 叶节点带 `view_id`；`occupied` 是含源的最小格子，不对应观测。四个子节点依次为左上、右上、左下、右下。`iter_view_leaves` 默认仅遍历观测叶节点；使用 `include_occupied=True` 可同时取得含源叶节点。

### 检查安装及以后更新

```bash
uv run python -m rind_phase1.checks --data-only
# 可选：检查真正的张量批次和多进程读取
uv run --extra train python -m rind_phase1.checks --data-only --torch --workers 2
```

这些命令检查全部窗口的父格含源/观测窗无源规则、局部掩码、场景子集、组批、树状恢复和样本参考解。教师链路使用下面独立的 `--teacher` 检查；模型训练仍待实现。`installation.json` 保存压缩包哈希和环境锁文件哈希。读取器版本保存在 `vendor/rind-dataset/`，运行代码在本仓库更新，并由 uv 锁定依赖。以后的数据版本应安装到新的数据目录并显式选择，以便追溯实验。需要更完整的生成与几何介绍时，可单独阅读下载包中的 Word 指南。

## 均匀物理教师：生成与读取

本任务明确研究**窗口外的源**。默认候选范围为 `world \ W`，即整个世界减去观测窗。quadtree 用于提供数据窗口，主教师不要求候选源能够重新生成相同的叶节点。完整几何由教师用于排除非法源；`--prior parent` 仅用于显式开启父格减观测窗的可选消融。十个参考解只用于独立检查和叠加显示。

```bash
uv sync --locked --inexact --extra diagnostics
# 间距和温度只是运行示例，不是已确定的实验默认值：
NUMBA_NUM_THREADS=2 uv run --no-sync rind-teacher \
  --scene-id 0 --view-id 4 --spacing 64 --temperature 0.05 --plot \
  --output outputs/teachers/example-world.npz
NUMBA_NUM_THREADS=2 uv run --no-sync python -m rind_phase1.checks --teacher
```

输出包括压缩 NPZ 教师记录、JSON 诊断报告和六面板 PNG。物理代价为响应不一致的采样像素比例；完成全部候选评价后，在有效候选上稳定归一化 `exp(-cost/tau)`。不加面积权重，也不使用真实源标签。预算不足导致未完成、或没有有效候选时明确失败。配置中的间距和温度仍为 `null`，必须显式选择。

```python
from rind_phase1.teacher import load_teacher_record
from rind_phase1.diagnostics import grid_from_record
record = load_teacher_record("outputs/teachers/example-world.npz")
xy = record["candidate_xy"]     # [N,2] 世界坐标，固定顺序
cost = record["physical_cost"]  # [N] 原始物理代价
q = record["teacher_prob"]      # [N] 软教师目标，无效候选为 0
valid = record["valid"]
cost_map = grid_from_record(record).as_map(cost)
```

同一教师支持 16、32、64、128、256、512 六种窗口：按原始尺寸重渲染，以 `L²` 归一化不一致像素数，不需要为不同大小另建教师。大窗口的计算量更高。[详细使用说明](docs/teacher-baseline.md) 包含 Python 直接生成、网格恢复、元数据、参考解检查、零代价连通区域、熵和不同窗口大小的比较。粗网格没有零代价点时，不能据此认为连续逆解为空，也不能通过追加参考解修补教师分布。

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
obstacle = local obstacle mask
window = (x, y, size)
```

真实源坐标和完整场景几何只用于教师监督和评价，不作为学生输入。窗口大小可以不同，必须保留原始空间尺度。单源场景按上述规则从 1024 分割到 16，应产生 18 个无源窗口，可作为检查依据。

本阶段可直接使用上述**已生成的单源、强度固定为 1 的数据**。实验需要新场景或新协议时再单独生成，并记录来源。训练、验证、测试始终按 `scene_id` 划分，同一场景的窗口不得跨集合。

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
│   ├── interfaces.md
│   └── teacher-baseline.md
├── configs/
│   └── phase1.json
├── src/rind_phase1/
│   ├── __init__.py
│   ├── install_data.py
│   ├── browser.py
│   ├── data.py
│   ├── physics.py
│   ├── search.py
│   ├── teacher.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── diagnostics.py
│   └── checks.py
├── scripts/install_data.sh
├── scripts/browser.sh
├── vendor/rind-dataset/
├── uv.lock
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

已安装的数据目录、场景名单或划分比例、随机种子和加载参数。当前发布数据为 10,000 个场景，每场景一个强度为 1 的源和 18 个无源窗口。

**已实现**

`Phase1Dataset`、`split_scene_ids`、`SizeBucketBatchSampler`、`make_dataloader`，以及参考解、场景几何、四分树和重渲染接口。`install_data.py` 负责校验、解压和压缩包清理；读取代码使用本仓库的版本快照。具体调用见前面的使用说明。

**仍需完成的研究接入**

确定并保存实验场景划分，向教师和学生模块提供观测，并记录额外生成的评价场景。先用现有数据接通链路；需要新场景或不同协议时再调用生成器。固定强度和窗口规则以数据清单为准，不能在读取时改成另一套物理假设。

**输出与目的**

`data/raw/phase1-single-source/` 保存已安装的原生数组，`data/splits/` 保存约定的场景名单。每条观测为 `scene_id`、`view_id`、`response [L,L]`、局部 `obstacle [L,L]` 和 `window [x,y,L]`。建立统一的场景到局部观测接口，并保持世界尺度与场景划分。

---

### `physics.py`

**输入与用途**

- 当前场景的连续障碍物几何：教师用它判断源是否合法和射线遮挡，不交给学生。
- 观测响应与 `[x,y,L]`：确定重渲染区域、像素中心及比较目标。
- 一个候选 `[sx,sy]`：在原场景放置唯一强度为 1 的源。

**当前实现**

`PhysicalEvaluator(ds,scene_id,view_id).evaluate(xy)` 返回几何有效性、物理代价、是否缓存命中和本次新增渲染次数。候选范围/观测窗排除由搜索模块负责；非法几何位置不渲染。完全相同坐标复用缓存，渲染计数只包含真正的新评价。

```text
C(s) = mean(abs(rerendered_response(s) - observed_response))
     = 不一致的采样像素数量 / L²
```

所有像素等权，包含双方响应都为零的障碍物像素。二值数据的代价在 `[0,1]`，零表示当前采样观测完全相等。`check_reference_solutions` 单独验证十个参考解全部为零代价；这些位置不进入教师候选构造。

`CandidateEvaluator.evaluate(xy) -> CandidateEvaluation` 是小型显式接口，可用合成评价器独立测试搜索和教师。

**后续研究**

提案的边界加权响应与边界距离尚未实现。本基线为 `alpha=1,lambda=0,beta=0`。启用更复杂代价之前，需约定边界提取、距离单位、裁剪边缘与空/缺失/额外边界处理；一条教师记录中的所有候选必须使用相同代价定义。

---

### `search.py`

**输入与用途**

- `CandidateDomain(window,world_size,quadtree_prior=False)`：默认采用世界减观测窗。显式设为 true 时启用父格减观测窗的可选消融。
- `spacing`：世界坐标中的候选间距，独立于响应采样精度。
- 物理评价器：提供每个坐标的有效性和响应代价。
- 可选 `max_evaluations`：实际新增物理渲染预算；未完成结果不能生成最终教师目标。

**当前实现**

构造全局锚定的 `((kx+0.5)*spacing,(ky+0.5)*spacing)` 网格，按 y 后 x 升序排列。先排除空间范围之外的候选，再评价几何与响应。返回坐标、有效性、已评价标记、原始代价、完成状态、新渲染/缓存次数与时间。`grid.as_map(values)` 恢复源空间图，观测窗位置为空值。

同一间距下父格候选是世界候选的子集。分割线归右/下，世界最外边界归末端格。`deduplicate_candidates` 对完全相同坐标按首次出现去重。

**后续研究**

先通过均匀代价图了解多解形状，再决定 coarse-to-fine 策略。不能假设单峰、只细化最好区域，或把粗细过程点直接混合进教师 softmax。搜索轨迹与最终支持集需要分开；声明完成时最终间距应一致，缓存/去重并按真实渲染计数。质量必须与共同均匀参考支持比较。当前没有自适应剪枝算法。

---

### `teacher.py`

**已实现：**完成的均匀代价转稳定的有效候选软分布、NPZ 存取、实际配置和数据/代码版本追溯，以及 `rind-teacher` 命令。`diagnostics.py` 提供局部响应/掩码、代价/零代价/概率图、参考解叠加、采样连通区域与熵。

**输入**

```text
candidate_xy
physical_cost
valid
temperature
scene_id / view_id
config_id
```

**这些输入分别做什么、怎么用**

- **candidate_xy**：`[N,2]` 候选世界坐标，由搜索模块提供。 明确每个教师概率对应哪个位置。
  **怎么用：**按原顺序保存；与所有候选数组一一对应，不单独重排。

- **physical_cost**：`[N]` 最终物理代价。 确定候选对观测的解释程度。
  **怎么用：**在有效候选上先减去最小代价，再计算 `exp(-(cost-min_cost)/temperature)`，避免小温度下溢；保留原代价用于绝对一致性评价。

- **valid**：`[N]` 布尔值，标记该候选是否属于所声明的有效范围。 排除不能作为源假设的位置。
  **怎么用：**只对有效项归一化，无效项概率为零；全无效时报告失败。

- **temperature**：正标量 `tau`。 控制相近物理代价之间的概率差异。
  **怎么用：**用于 `cost/tau`；较小值更突出低代价区域。用验证数据选择，并随教师记录保存。

- **scene_id / view_id**：原始观测的身份标识。 将教师结果准确匹配回训练样本。
  **怎么用：**用于读取、缓存和检查划分归属，不作为神经网络输入。

- **config_id**：指向本次教师生成配置的可解析标识。 保证代价和分布结果可追溯。
  **怎么用：**随记录保存；对应配置应包含渲染版本、代价设置、搜索范围及采样间距，不能只是找不到来源的标签。

**处理**

下面的求和均仅包含 `valid=True` 的候选，且温度 `τ>0`。无效候选概率为 0；没有有效候选时应报告失败，不能生成伪造的均匀目标。

在最终保留的有效候选集合上：

$$
q_i^*
=
\frac{\exp(-C_i/\tau)}
{\sum_j \exp(-C_j/\tau)}
$$

这样多个物理上合理的位置可以同时保留较高概率。教师与学生必须使用相同的候选坐标、顺序和有效性；不在 softmax 中加入面积因子。

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
valid
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
local obstacle mask
window = (x, y, size)
candidate_xy
valid
```

**这些输入分别做什么、怎么用**

- **model config（初始化时使用）**：`coordinate_scale`、`fourier_frequencies`，以及实现时确定的网络结构设置。 定义坐标单位、频率范围和模型结构。
  **怎么用：**创建模型时读取并保存；训练与加载检查点必须一致。它是初始化设置，不是每条样本的额外观测。

- **local response**：`[size,size]` 的观测数组。 提供遮挡形状与有效边界等局部信息。
  **怎么用：**输入观测编码器；不同大小样本可按大小分组，或填充并屏蔽填充值。不能让填充区域被当作零响应证据。

- **local obstacle mask**：`obstacle[size,size]` 布尔数组，仅包含当前窗口。
  **怎么用：**与局部响应共同作为观测条件；禁止读取窗外掩码或完整几何作为学生特征。

- **window = (x, y, size)**：`[x,y,size]`。 告诉模型局部响应在世界中的位置和实际范围。
  **怎么用：**将位置及大小交给窗口编码部分，与响应表示组合；内部坐标按统一 `coordinate_scale=1024` 归一化，保存记录仍用世界坐标。

- **candidate_xy**：待查询的 `[N,2]` 世界坐标。 指定模型需要判断的假设位置。
  **怎么用：**使用与窗口一致的坐标尺度，再进行 Fourier 表示；逐候选输出能量，不重新运行物理过程。

- **valid**：当前预测候选集合的有效性标记。 规定分布归一化范围。
  **怎么用：**仅用于将能量转为概率，不送入编码器。教师辅助评价可用教师掩码；无几何预测只能使用公开可知的范围约束。

**不作为学生特征：**真实源、场景 ID、完整场景几何、窗外障碍物掩码、教师物理代价和完整四分树。`valid` 是归一化元数据，不能作为额外的几何编码输入。

**处理**

首先编码 observation：

$$
(R,M) \rightarrow h_{RM}
$$

同时编码窗口位置和大小：

$$
v \rightarrow h_v
$$

得到：

$$
h_O=f(h_{RM},h_v)
$$

候选坐标使用 Fourier features：

$$
s\rightarrow\gamma(s)
$$

最后输出条件能量：

$$
E_\theta(O,s)
$$

能量越低表示候选越合理。能量函数本身只接收响应、局部掩码、窗口信息和候选坐标；`valid` 用于随后将评分转成分布，不作为编码特征。模型应支持不同窗口大小，并能查询新的连续坐标。

在有效候选集合上归一化，无效候选概率为 0：

$$
p_i
=
\frac{e^{-E_i}}
{\sum_j e^{-E_j}}
$$

求和仅包含有效候选，概率表示当前候选集合内的相对兼容性。

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
student model
training config
```

**这些输入分别做什么、怎么用**

- **observation**：训练或验证集合中的响应、局部掩码、窗口及样本标识。 提供学生条件输入，并确定监督记录。
  **怎么用：**用标识匹配教师记录，只将响应和窗口送入模型；训练集用于更新，验证集用于选择模型。

- **teacher candidate set**：教师记录中的 `candidate_xy`，附带对应空间单元信息。 确定这一训练样本需要比较的位置。
  **怎么用：**把同一候选坐标输入学生，保持与教师概率完全一致的顺序。

- **teacher_prob**：`[N]` 的软监督概率，来自 `teacher.py`。 表达多个源假设的相对兼容性。
  **怎么用：**作为 KL 或软标签交叉熵的目标，不改成单个最大概率位置的标签。

- **valid**：教师记录中的候选有效性。 保证教师和学生比较的是同一范围。
  **怎么用：**对双方使用同一掩码；不将无效代价或无效位置加入损失。

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

- **teacher records**：同一批观测在评价候选集合上的代价、概率和有效性。 提供物理参照与分布参照。
  **怎么用：**让学生在相同候选上评分后比较。若保存的是不同自适应候选，需要先通过物理评价补齐共同参考网格上的教师记录，不直接比较不同向量。

- **trained model**：指定检查点及其模型配置。 生成待评价的学生结果。
  **怎么用：**加载后固定参数进行预测；不在测试集上继续拟合。无几何预测单独使用公开候选集。

- **experiment config**：候选范围、候选间距与覆盖约定、指标、消融设置、检查点标识和运行环境。 保证结果可比且可追溯。
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

分布指标使用共同的评价候选集合。跨搜索策略比较时使用共同的均匀参考网格，不能直接比较不同候选点数或不同位置上的概率向量。热图标明网格间距；不要把自适应候选的离散概率解释为连续密度。被搜索排除的参考位置，在搜索质量评价中记为未覆盖，并报告合理区域的遗漏，不能只比较保留区域。报告实际未见的障碍物组合；若尚未构造此划分，应明确其评估未完成。

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

已实现 `--data-only` 数据检查和 `--teacher` 均匀教师联调检查；后者覆盖六种窗口、六十个精确参考解、坐标/代价对应、归一化、几何排除、缓存计数与记录存取。学生链路检查仍待接入。

**输入**

```text
少量 RIND scenes
phase1 config
```

**这些输入分别做什么、怎么用**

- **少量 RIND scenes**：可读取的真实场景、观测与真实源参数。 提供能够核对物理正确性的具体例子。
  **怎么用：**检查窗口与划分，使用真实源重渲染，再串联候选、教师和学生；未训练输出必须标注。

- **phase1 config**：当前实际使用的完整配置文件。 使联调与各模块使用同一约定。
  **怎么用：**检查必需项已填写，按同一强度、窗口、坐标和候选归一化规则执行；记录失败模块和配置标识。

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

`null` 表示尚未确定，不是可运行默认值。数据路径、固定强度 1 和场景数量 10,000 已按当前发布数据填写；主实验关闭 `quadtree_prior`，研究窗口外源的全世界兼容性；基线使用均匀响应代价。候选间距、温度、正式场景划分与训练参数仍待确定。数据类读取数据清单，研究模块需显式读取配置。

### `data/raw/`

默认在 `data/raw/phase1-single-source/` 保存安装后的 RIND 原生数组；响应按位压缩保存，并在读取窗口时恢复：

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

`pyproject.toml` 与 `uv.lock` 管理模型及读取器环境；`scripts/install_data.sh` 负责安装，`scripts/browser.sh` 负责安装可视化依赖并打开数据，`vendor/rind-dataset/` 保存读取器和可视化页面的版本；`.gitignore` 排除大型产物与缓存；`__init__.py` 标记 Python 包；空目录中的 `.gitkeep` 仅保留目录。

---

## 标准数据接口

### Observation

```python
{
    "scene_id": ...,
    "view_id": ...,
    "response": ...,
    "obstacle": ...,  # 仅局部，bool [L,L]
    "window": [x, y, size]
}
```

### Teacher record

```python
{
    "scene_id": ...,
    "view_id": ...,
    "candidate_xy": ...,
    "valid": ...,
    "physical_cost": ...,
    "teacher_prob": ...,
    "temperature": ...,
    "config_id": ...
}
```

候选相关数组按同一顺序排列：`candidate_xy` 为 `[N, 2]`，`valid`、`physical_cost` 和 `teacher_prob` 为 `[N]`。`temperature` 是标量，`config_id` 是配置记录标识。`response` 和局部 `obstacle` 为 `[size,size]`，`window` 为 `[3]`。实际教师记录另有 `evaluated`、`metadata`、`run_id`、网格轴与索引，见接口文档。

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

- **教师目标的含义：**研究窗口外的源。默认 `quadtree_prior=false`，主教师在整个世界减观测窗内评价响应兼容性，不要求候选源重新生成相同 quadtree 叶节点。父格限制仅是显式开启的可选消融。教师目标是给定场景、声明均匀支持上的离散兼容性分布，尚不是已校准的完整生成后验。
- 教师依赖隐藏场景几何，严格来说其目标为 $q^*(s\mid O,S)$。学生只看到 $O$；相同观测可能对应不同场景的兼容性分布，因此不能保证逐场景精确恢复教师结果。
- **评价与部署分开：**教师有效范围上的一致性用于判断蒸馏是否成功。无几何预测必须在预先声明的完整公开候选范围上独立运行，不能复用依赖场景几何或物理代价的有效掩码/自适应候选集。仅在教师保留范围内训练，不会自动约束被排除位置的能量；无几何结果需单独报告，不能由教师有效范围上的表现推断。

## 首轮交付与认领

首轮检查：**2026 年 10 月 1 日（周四）**。人员与算力待定，组员自行认领；可一人负责多个文件。

| 工作 | 文件 | 负责人 | 最小交付 |
|---|---|---|---|
| 数据 | `data.py` | Yushu He | 安装已有数据、窗口示例及约定的场景划分 |
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
