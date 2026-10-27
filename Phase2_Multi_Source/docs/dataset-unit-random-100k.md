# 100,000 场景：固定强度多源与随机窗口

[中文入口](../README.zh-CN.md) · [English README](../README.md)

## 数据内容

| 内容 | 固定强度随机窗口版 |
|---|---|
| 场景数 | 100,000 |
| 每场景窗口 | 256 个不同的 `(x,y,L)`；合计 25,600,000 个 |
| 源数量与强度 | 随机 1–4 个源；每源强度为 1 |
| 窗口边长 | 64、128、256、512，随机位置 |
| 世界 | 连续二维 1024×1024；源和障碍物参数为 float64 |
| 采样 | 像素 `[row,col]` 对应 `(x+col+0.5,y+row+0.5)` |
| 窗口约束 | 所有真实源在窗口外；非障碍物采样点至少有两种不同总响应值 |
| 参考解 | 每窗口可读取真值加 9 组不同位置的固定强度源组合 |
| 发布包 | `RIND-random-v3-100000scenes-multi-unit-256views.zip`，1.55 GiB |
| 解压空间 | 约 62.41 GiB |

能看见几个源，响应就是几，因此总响应为 0–4，不截断成 0/1。没有距离衰减、反射或噪声。全暗、全亮、恒定中间亮度都被排除；障碍物内部的零值不参与信息判断。例如自由区域同时出现 1 和 2 的窗口有效。没有额外设置少数像素比例门槛。

窗口不再来自四分树。每种尺寸在全部合法的整数起点中无放回均匀抽样，通常各 64 个；某个尺寸合法位置不足时，由其他尺寸补足，总数仍为 256。窗口可以重叠，同一场景不重复保存同一个 `(x,y,L)`。整数起点只用于共享采样网格，连续几何不会吸附到栅格。

## 安装与 Browser

需要本次更新后的 **Phase2_Multi_Source**，包括 `src/` 和 `vendor/`。命令、类名和依赖保持原用法；不需要独立配套项目或生成器环境。数据 ZIP 不会自动更新旧仓库代码，`uv sync` 也不会替代代码更新。

新版 Drive 链接待发布后补充；原 10k Drive 链接对应随机强度四分树数据。

将 ZIP 放到项目根目录：

```bash
./scripts/install_data.sh
./scripts/browser.sh
```

默认安装到 `data/raw/phase2-multi-source/`。安装器逐文件校验 SHA256；成功后删除 ZIP 和 sidecar，失败时保留。需要保留下载包时加 `--keep-archive`。Browser 默认地址为 `http://127.0.0.1:8768`，支持总响应和各源通道，像素检查显示实际叠加强度。

与已有旧数据并存：

```bash
./scripts/install_data.sh /path/to/RIND-random-v3-100000scenes-multi-unit-256views.zip \
  --data-root data/raw/phase2-multi-source-unit-random-100000 --keep-archive
./scripts/browser.sh --data-root data/raw/phase2-multi-source-unit-random-100000
```

已有数据与新 ZIP 不同时，安装器提示选择另一目录或显式使用 `--replace`。替换先完整提取并校验新包，再把旧目录保存在相邻的 `.backup-...` 后切换；切换失败恢复原目录。备份不会自动删除。

Python 范围仍为 `>=3.10`。旧的随机强度四分树版本继续可读，manifest 用来区分两种格式。

## Python 读取与 DataLoader

```python
from rind_phase2.data import Phase2Dataset, make_dataloader

ds = Phase2Dataset()  # 默认目录；也可传 root="/path/to/native/data"
print(ds.num_scenes, len(ds))  # 100000, 25600000
item = ds[-1]                 # scene_id=99999，view_id=255
R = item["response"]          # float32 [L,L]，总响应 0..K
M = item["obstacle"]          # bool [L,L]，True 为障碍物
W = item["window"]            # int64 [3]，(x,y,L)
s, v = item["scene_id"], item["view_id"]

# 需要 uv sync --locked --extra train；先用小的场景子集演示。
subset = Phase2Dataset(ds.root, scene_ids=[0, 10000, 99999])
loader = make_dataloader(subset, batch_size=8, num_workers=2)
batch = next(iter(loader))    # response/obstacle [B,L,L]，window [B,3]
```

相同尺寸自动分桶。学生样本只返回 ID、局部总响应、局部障碍物掩码和窗口元数据；真实源数、位置、强度、分源通道及完整几何通过独立接口访问。`RIND_MULTI_DATA_ROOT` 可改变默认目录，训练配置的显式 `data_root` 仍需指向实际安装位置。

## 窗口、通道、参考解与重渲染

```python
for view_id, window in ds.iter_windows(s):
    print(view_id, window)

channels = ds.get_channels(s, v)  # float64 [K,L,L]，逐源贡献
truth = ds.get_source_params(s)  # float64 [K,3]，(x,y,1)
refs = ds.get_candidates(s, v)    # 10 组 [K,3]，各源强度均为 1
pred = ds.rerender(s, W, refs[1], backend="numba")  # float64 [L,L]
```

`channels.sum(axis=0)` 就是总响应。随机窗口没有树，使用 `iter_windows()`；调用 `get_view_tree()` 会明确报错。旧四分树数据仍支持树状读取。

十组参考解按场景共享存储。每组改变一个源的位置，并完整验证其 1024×1024 采样可见性不变，因此所有窗口的响应也一致。这是采样点上的精确一致，不宣称整个连续世界处处相同。参考解是靠近真值的不完整见证，不是完整逆解集或教师概率标签。

可见性按源逐位保存。全部窗口坐标约 154 MB，场景共享参考参数约 64 MB；不重复保存浮点窗口图像。`candidate_layout="scene-source-sets"` 对应 `reference_sources.npy`，用户代码应通过 `get_candidates()` 读取。

## 教师与训练配置

联合教师根据 manifest 的 `driver_strength_mode` 选择已知强度先验：本版候选各源强度固定为 1，旧版仍为 [0,1) 随机强度。源数量、数量先验、位置间距和温度仍由调用方明确指定；不读取真实源数或参考解来构造候选集。

必须按 **scene ID** 划分训练、验证和测试。同一场景窗口会重叠，共享源和几何，不能随机拆到不同集合。

旧训练配置若使用 16、32 等尺寸，应在新的实验配置中选择本版尺寸，使用新的输出目录，并重新生成相应教师记录。已有教师缓存和冻结实验记录不能直接复用；模型架构和训练超参数没有自动改变。

## 数据检查与发布校验

常规数据接口检查：

```bash
uv run python -m rind_phase2.checks --data-only --data-root /path/to/native/data
```

完整的新版逐场景审计可用：

```bash
uv run python -m rind_phase2.unit_checks /path/to/native/data --workers 4
```

完整审计检查全部窗口是否有信息、无源、无重复，以及源强度与参考解结构；按固定种子抽取 24 个场景重渲染参考解。此命令会更新数据目录中的生成统计及校验报告。只需常规读取时不必每次执行完整审计。

正式数据已通过全部 100,000 场景、25,600,000 窗口的结构和信息检查。24 个物理抽检场景的 240 组整图参考解一致；另行验证了最高编号场景、四种尺寸的零物理代价、双 worker DataLoader、Browser 及窗口重现。源数 1/2/3/4 的场景分别为 24,898/25,081/24,974/25,047 个。

| 边长 | 实际窗口数 |
|---|---:|
| 64 | 6,403,003 |
| 128 | 6,402,868 |
| 256 | 6,402,866 |
| 512 | 6,391,263 |

132 个场景没有合法的 512 起点，配额已按上述规则分配。发布包内有生成源码与记录，消费数据不需要运行生成器。

ZIP 的 SHA256：

```text
44015d37e6abb2093228380c237beeea1e71266b87640a5e8cdc05fe4a36d174
```

若同时下载 sidecar，可用 `sha256sum -c RIND-random-v3-100000scenes-multi-unit-256views.zip.sha256` 复核。

## English summary

This update supports the 100k unit-strength random-window release through the existing Phase II installer, Browser, `Phase2Dataset` and DataLoader APIs. Each scene has 1–4 unit sources and 256 distinct informative source-free windows, with sizes 64/128/256/512. There is no quadtree for this variant. Read shared scene-level references through `get_candidates()`; these incomplete witnesses are not teacher labels. Joint teacher proposals follow the manifest's strength policy, while count priors remain explicit. Update the project code before installing the new ZIP; no generator environment or new dependency is required. The ZIP is 1.55 GiB and native storage is 62.41 GiB. Split by scene and regenerate teacher records for new experiments. The old random-strength quadtree release remains supported.
