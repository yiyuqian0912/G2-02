# RIND Phase I — 多源叠加观测

[English](README.md) | 中文

这个版本把单源任务扩展为多个不同强度的源。研究对象是**固定观测窗口外的源**：哪些完整的源参数组合，能够解释窗口里看到的响应？同一幅观测可能对应很多种源组合。

**已实现：**发布包安装、内存映射读取、局部障碍物掩码、分源通道、源组合重渲染、物理代价、树状读取、按尺寸组批的 DataLoader、browser，以及可复现的**多源联合教师**。
**待实现：**自适应搜索、学生模型、训练和评价。教师保留每一整组源参数的概率，也提供便于查看的二维投影。

[设计与研究约定](docs/method.md) · [教师使用指南](docs/teacher.md) · [接口一览](docs/interfaces.md) · [为什么先用 1–4 个源](docs/source-counts.md) · [实测记录](docs/verification.md)

## 1. 这是一个什么数据集？

| 内容 | 当前多源版 |
|---|---|
| 场景与观测数量 | 10,000 个场景，367,109 个局部观测 |
| 世界 | 连续二维 1024 × 1024；x 向右，y 向下 |
| 每场景的源 | 随机 1–4 个；每个强度在 [0,1) 均匀采样 |
| 响应 | 能看见的源强度相加，**允许大于 1** |
| 障碍物 | 连续矩形、椭圆、三角形、多边形及其组合 |
| 窗口边长 | 16、32、64、128、256、512 个世界单位 |
| 每个窗口的参考组合 | 场景只有 1 个源时为 5 组；2–4 个源时为 6 组 |
| 存储 | 每个源的可见性按位压缩，另存 float64 强度；读取时恢复 |

源到采样点的线段不被障碍物挡住，就贡献自己的强度。没有距离衰减、反射或噪声。障碍物内部响应为零。窗口像素 `[row,col]` 对应世界坐标 `(x+col+0.5,y+row+0.5)`；栅格是采样精度，源位置和障碍物几何仍然连续。

窗口由源驱动的四分树产生：含源区域继续十字均分；无源区域立即记为观测窗口；分到 16 为止，最小的含源块不保存观测。多源场景的窗口数可变，不能按“源数 × 18”计算。

## 2. 安装与浏览

向数据维护者获取多源包 **`RIND-adaptive-v3-10000scenes.zip`**。这里尚未登记多源下载链接；单源版的 Drive 链接对应另一份数据。把 ZIP 放到本 README 所在目录，保持压缩状态：

```bash
./scripts/install_data.sh
./scripts/browser.sh
```

需要先安装 uv。安装脚本在本项目建立 `.venv`，校验发布包 SHA256，只解压原生数组到 `data/raw/phase1-multi-source/`。成功后删除这份 ZIP 和旁边对应的 `.zip.sha256`；失败时保留 ZIP，并清理临时解压内容。重复执行会检查已安装数据，不重复解压，也不删除后来新放入的 ZIP。希望首次安装也保留压缩包时，加 `--keep-archive`。

原生数据约需 **7 GB**，环境和临时压缩包另算。下载包自己的代码和环境不会安装。这个版本使用自己的数据目录和 **`RIND_MULTI_DATA_ROOT`**，与单源版分开。

```bash
./scripts/install_data.sh /path/to/RIND-adaptive-v3-10000scenes.zip --keep-archive
export RIND_MULTI_DATA_ROOT=/mnt/data/rind-multi
./scripts/install_data.sh /path/to/RIND-adaptive-v3-10000scenes.zip
./scripts/browser.sh --no-open --port 8767
```

browser 默认地址为 `http://127.0.0.1:8767`。可以切换总响应和每个源的独立贡献；像素检查显示实际强度和各源贡献。显示亮度按场景总强度缩放，数值不会截断到 1，也不会转换为 0/1。

Python 范围为 **>=3.10**，不固定小版本；依赖仍须支持所选解释器和平台。uv 选择兼容解释器，已有 `.venv` 会沿用原解释器。需要明确选择时可运行 `UV_PYTHON=3.12 ./scripts/install_data.sh`。在代码里使用 `uv run python`，或在编辑器中选择本项目 `.venv`。

## 3. 读取一条观测

```python
from rind_phase1_multi.data import Phase1MultiDataset

ds = Phase1MultiDataset()  # 或 root="/path/to/native/data"
print(ds.num_scenes, len(ds))  # 10000, 367109
sample = ds[0]
s, v = sample["scene_id"], sample["view_id"]
R = sample["response"]       # float32 [L,L]，总叠加强度
M = sample["obstacle"]       # bool [L,L]，True 表示障碍物
W = sample["window"]         # int64 [3]，世界坐标 (x,y,L)
```

`ds[i]` 只返回 ID、`response`、`obstacle` 和 `window`。学生条件输入为 `(R,M,W)`；ID 用于匹配记录。真实源数、坐标、强度、分源通道、完整几何和窗口外障碍物都通过独立接口供教师或评价使用。即使编码器内部缩放图像，也要保留原始窗口位置与世界尺度。

## 4. 数据划分和 DataLoader

需要训练时安装 PyTorch：

```bash
uv sync --locked --extra train
```

用 `uv run --extra train python` 执行：

```python
import json
from pathlib import Path
from rind_phase1_multi.data import Phase1MultiDataset, split_scene_ids, make_dataloader

all_data = Phase1MultiDataset()
# 示例比例；正式实验先约定并保存一次。
splits = split_scene_ids(all_data.num_scenes, ratios=(.8,.1,.1), seed=20260923)
Path("data/splits").mkdir(parents=True, exist_ok=True)
for name, ids in splits.items():
    Path(f"data/splits/{name}.json").write_text(json.dumps(ids))
train_data = Phase1MultiDataset(scene_ids=splits["train"])
loader = make_dataloader(train_data, batch_size=8, num_workers=0)
for batch in loader:
    R = batch["response"]  # torch.float32 [B,L,L]
    M = batch["obstacle"]  # torch.bool [B,L,L]
    W = batch["window"]    # torch.int64 [B,3]
    break
```

同尺寸窗口在一批，不同批的 L 可以不同。不需要缩放或补零。学生批次里没有源标签，因此源数不同也不需要为源数补齐。每个 epoch 调用 `loader.batch_sampler.set_epoch(epoch)` 更新可复现的打乱顺序。使用 spawn 多进程时，在 `if __name__ == "__main__":` 内建立 DataLoader。

同一场景的所有窗口、教师目标必须在同一集合。按观测均匀抽样会让窗口较多的多源场景权重更高；正式实验需声明场景平衡策略，或至少按源数分别报告。

## 5. 教师和评价怎么取值？

```python
import numpy as np
from rind_phase1_multi.physics import evaluate_sources

sources = ds.get_source_params(s)  # float64 [K,3]：x,y,强度
channels = ds.get_channels(s, v)   # float64 [K,L,L]，各源贡献
raw = ds.get_region(s, *map(int, W))["response"]  # float64 [L,L]
assert np.array_equal(channels.sum(axis=0), raw)

fresh = ds.rerender(s, W, sources)  # float64 [L,L]
parts = ds.rerender(s, W, sources, return_channels=True)
# parts 含 channels [K,L,L]、response [L,L]、obstacle [L,L]
result = evaluate_sources(ds, s, v, sources)
print(result["valid"], result["physical_cost"])

scene = ds.get_scene(s)             # 连续几何与生成源
references = ds.get_candidates(s, v)  # 多个 float64 [K_candidate,3] 数组
```

每行必须明确提供强度。一整个源组合才是一个逆问题候选；物理代价是 float64 响应之间的平均绝对强度差，不能把转为 float32 的学生图像当作重渲染精度检查目标。

`evaluate_sources` 要求**每个源都在世界内、窗口外、障碍物外**；没有 quadtree 父节点限制。`rerender` 是通用正向渲染接口，也可以渲染窗口内的源；研究范围的限制由物理评价或搜索实施。

### 参考解到底是什么？

第 0 组是真实生成源。多源场景还提供源行顺序反转的一组。另有四组把一个源的强度拆成两个**同位置**源的强度，保留总响应。因此参考组可能有 **K+1 个源**，最多 5 个，虽然生成场景最多 4 个。

这些参考组说明源顺序和强度拆分的多解性，并不像单源包那样提供九个新位置。它们不是完整逆解集，也不是教师概率标签；不能追加到教师采样网格后参与归一化。浮点加法的排列或拆分可能引起末尾几位误差，物理一致性检查使用绝对容差 `1e-12`；这不是训练超参数或搜索接受阈值。

### 类树状恢复 local view

```python
root = ds.get_view_tree(s)
for leaf in ds.iter_view_leaves(s):
    print(leaf["x"], leaf["y"], leaf["size"], leaf["view_id"])
    local = ds.get_observation(s, leaf["view_id"])
```

节点的 `kind` 为 `split`、`view` 或 `occupied`。`view` 叶节点有对应 `view_id`；最小含源叶节点没有观测。子节点顺序为左上、右上、左下、右下；传 `include_occupied=True` 可包含含源叶节点。树来自源驱动的分割，属于教师或诊断接口，不作为学生输入。

## 6. 生成与读取多源教师

一个候选是一**整组源参数**：所有位置和强度一起解释观测。教师比较整组源叠加后的响应，再给每组一个概率。每个源都在世界内、窗口外、障碍物外，没有 quadtree 父节点限制。第一版在均匀位置网格上可复现地抽样组合，强度独立均匀抽样，支持全部六种窗口尺寸。

不知道源数时，要明确给出考虑哪些数量以及各自的先验。下面同时考虑 1–4 个源，各占四分之一。这些间距、抽样数和温度是**演示参数，不是已经论证好的研究默认值**：

```bash
uv sync --locked --extra diagnostics
uv run --extra diagnostics rind-multi-teacher \
  --scene-id 1 --view-id 13 \
  --spacing 64 --samples-per-count 1024 --temperature 0.05 \
  --counts 1 2 3 4 --count-prior 0.25 0.25 0.25 0.25 \
  --seed 20260923 --diagnostics --plot \
  --output outputs/teachers/example-scene1-view13.npz
```

输出为压缩 NPZ、诊断 JSON 和 PNG。图里有原始观测、候选重渲染、局部障碍物、空间投影和源数量概率。如果任务外部已经明确源数，可以用 `--counts 2` 并省略 `--count-prior`。教师不会偷读真实源数来决定候选。

```python
from rind_phase1_multi.teacher import load_teacher_record
from rind_phase1_multi.diagnostics import source_space_projections

t = load_teacher_record("outputs/teachers/example-scene1-view13.npz")
q = t["teacher_prob"]         # float64 [N]：每一整组参数的概率
C = t["physical_cost"]        # float64 [N]：原始平均强度误差
i = int(q.argmax())          # 查看一组；分布训练仍需保留所有候选
K = int(t["source_counts"][i])
parameters = t["sources"][i, :K]  # [K,3]，每行 (x,y,强度)，后面是补零
maps = source_space_projections(t)
presence = maps["presence"]   # [Ny,Nx]，某网格位置至少有一个源的概率
```

真正的**联合目标**是整组参数上的 `q`。二维存在概率图只是投影，所有格子加起来可以大于 1，也无法保留源与源之间的搭配关系，不能直接替代联合目标。这里的“完成”表示指定的候选全部算完，未穷举连续解空间；预算不足会明确报错，不会把只算完的前半部分当作最终教师。参考解另做物理检查，不加入采样或概率归一化。

直接在 Python 中生成、评价自己给定的候选、缓存和预算、实例解释，见[教师使用指南](docs/teacher.md)。正式实验的间距、抽样数、温度、候选源数和先验仍需明确指定，配置中保留为空。

## 7. 检查与后续开发

```bash
uv run python -m unittest discover -s tests -v
uv run python -m rind_phase1_multi.checks --data-only
uv run python -m rind_phase1_multi.checks --data-only --full-audit --report outputs/reports/data-check.json
uv run --extra train python -m rind_phase1_multi.checks --data-only --torch --workers 2
uv run python -m rind_phase1_multi.checks --teacher \
  --spacing 128 --samples-per-count 64 --temperature 0.05 \
  --counts 1 2 3 4 --count-prior 0.25 0.25 0.25 0.25 \
  --report outputs/reports/joint-teacher-check.json
```

全量检查覆盖所有场景和窗口的元数据、位图与参考组的解析等价性。物理重渲染抽查各源数的首尾场景，并各取一个可用尺寸的窗口，未逐个重渲染整份发布数据。

教师检查覆盖实际 K=1–4、L=16–512 的 24 个观测，另检参考解，并核对缓存代价与直接重渲染。粗采样检查能验证流程，不能证明完整解空间覆盖或训练质量。正式运行前仍需确定场景划分和采样／温度参数，比较不同采样规模与随机种子的稳定性，再确定学生模型和训练协议。自适应搜索留作后续工作。
