# RIND3D-Lite v0.1：发布契约与实现

Lite 与原完整版本共用 RINDPack v1、信息索引和 D 窗口选择器。它新增显式 profile `rind3d-lite-v0.1` 和状态 `READY_LITE`，不会把原 `READY` 的 4096 窗口、1–64 参考源改成较小数量。

每个 Lite geometry/source 场景必须包含完整的 1024³ 响应和实体掩码、无损 canonical 连续几何、一个强度为 1 的真实源，以及恰好 256 个信息窗口。边长 64、128、256 的配额为 85、85、86；原点是任意合法整数，允许重叠但不允许重复。真实源必须在半开窗口外，每窗必须同时有亮、暗两类自由体素；实体中的 0 不算暗自由空间。原有至少两个八分区的覆盖门禁保留。

每窗保存 1–8 个完整验证的参考源，包含真实源，任意两源距离至少 32/1024。没有最低额外解数量，不补齐、不复制。参考解只是有限见证，数量受搜索预算影响；只有一个不代表唯一解。验证的是窗口所有采样点上的精确响应，不声称连续空间等价。

## 渲染与独立物理复核

`rendering/polyhedra.py` 精确解码 canonical binary64 参数；逆矩阵和平面代数使用有理数。`context.py` 与 `native/convex.cpp` 提供源平面缓存、保守区间与 GMP 精确回退。`cuda_runtime.py` 管理 NVRTC/driver 生命周期，`cuda_backend.py` 管理有界 packed 砖块批次；`rendering/stream.py` 使用原有 writer、CRC、SHA、journal、seal 与 resume 协议。没有通过导入路径切换接管实验代码；历史实验保持独立。

四个后端名称为 `cpu_reference`、`cpu_native_certified`、`cuda_certified`、`auto`。原参考后端始终可独立调用。原生/CUDA 快路径仅接受已证明有界、满维的凸体顶层并集；不支持的结构交回原 canonical oracle。`auto` 还检查本构建的认证能力、CUDA 可用性和真实设备的定向运算自检。缺少 CUDA 时使用 CPU；编译错误、数值失败和未决查询不会被掩盖成有效标签。

CUDA 使用严格 binary64 定向区间，输出可证明亮、可证明暗或 UNKNOWN。UNKNOWN 仅在凸体数学前提已证明时调用区间/GMP 谓词；不把近似网格、float32 射线或容差当真值。点、接触、切线和闭线段的端点都保留闭边界语义。

`physical_evidence.py` 对所需的物理砖块独立重算两遍，同时比较响应和实体位。A 为参数区间裁剪、必要时 GMP；B 为有理 Fourier–Motzkin 阴影投影和另一套区间符号计算、必要时 GMP。B 不调用 A 的射线裁剪。两者共享 canonical 参数解码，属于明确披露的共同依赖；另外用未修改的 CGAL canonical oracle 做认证夹具对照。

每个任意对齐窗口必须被这些已复核的 64³ 砖块完整覆盖，包括所有边缘。证据绑定几何、源的原始 float64 字节、数值政策、两个实现身份和两个字段 SHA。重叠窗口可以复用物理证据；哈希检查本身不算物理验证。`verify --physical` 会重新执行物理比较，而非只读取缓存。

## 几何与源的发布认证

数值渲染通过不等于场景可发布。v0.1 对随机生成器采用一个明确的、较窄的支持域：所有凸体分量都严格处于世界内部，与六个世界面有可证明的正间隙。这是记录在 recipe 中的生成分布约束，不是对任意截断几何已经认证的宣称。

每场景重新检查 canonical 编解码逐字节一致、引用与类型、可逆变换、每个凸体的严格内部球，以及六个世界坐标界。坐标界使用非负 Farkas/线性规划对偶证书：原始平面的非负组合精确等于正/负坐标方向，组合常数给出整个凸体的界。Python Fraction 独立核查这些等式。只检查列出的顶点不作为世界包含证明。满维有界闭凸体的有限并集保留 regular-closed 拓扑；遇到不能由简单证明解决的并集边界拓扑，仍调用原 canonical 服务。

继承 acceptance-core 的 25 个点、20 条线段（含反向和零长度）以及世界/左右半世界区域查询必须确定且一致。真实源独立均匀提议，经过精确 FREE 和 `clearance_GT`；半径继承原工程的 **2^-26**，不是源间距，也不是最小障碍物厚度。未决结果与预算耗尽分别记录，不能当作几何物理拒绝。所有保留额外源也要求同样净空。

`certify-lite` 先对九类生成器的冻结随机夹具，运行原 canonical 点/线段/区域/净空检查、完整 32³/64³/128³ A/B/CUDA 对照和接触/薄片/拓扑困难例。认证绑定科学实现指纹；改变相关代码后必须重新认证。逐场几何、源、字段、窗口和参考解检查仍不可省略。`random_field` 及不支持的函数几何不在 Lite 家族名单内，原 `ready_capable=False` 限制保持。

## 随机分布、搜索与私有数据

九类生成器包含旋转盒、薄片、棱柱、楔体、随机凸多面体、多尺度混合、走廊、屏障和遮挡簇。参数和实际平面/面数均记录。方向使用均匀四元数提议后保存为 canonical binary64 矩阵；中心在保守可容纳范围内提议，结构化家族使用相关布局。薄片半厚度按 log2 区间 [-20,-5] 提议，没有基于体素宽度删除薄几何。详细范围见 `lite_generators.DISTRIBUTION` 与每场景 `recipe.private.json`。不按渲染时间或参考解数量挑选场景。

额外源来自空间分散的 Sobol 自由源池，先精确净空、间距、分域，再平衡亮/暗自由探针筛除，最后逐体素/区域覆盖整窗，并用 B 独立重查。默认每窗首遍搜索预算 0.05 秒、最多 8 个源；时间预算使保留数量可能随硬件和负载变化，提议顺序和所有实际停止原因都有记录。额外源尚未独立复查完则丢弃。真源由两遍物理砖块证据验证，不用“源身份相同”代替物理工作。

`geometry_id` 是无损几何的 canonical 身份，`source_id` 绑定几何和源原始字节，场景 ID 还绑定物理协议。所有共享 geometry_id 的源使用同一组 train/val/test 划分。场景、见证、来源、选择分数和参考解不混入默认模型输入；显式 `private_oracle()` 或 `view="candidate_conditioned"` 才能访问私有标签。

## 操作

```bash
# 先正常安装/编译本工程；需要已有相邻 canonical 依赖，未新增 Python CUDA 依赖
uv sync --extra dev --extra train

# 有可用 NVIDIA 驱动和 NVRTC 时；默认路径找不到可设置 RIND_NVRTC_LIBRARY
PYTHONDONTWRITEBYTECODE=1 uv run rind3d certify-lite
# CPU-only 安装可用 --cpu-only，配置 renderer_backend: cpu_native_certified 或 auto

uv run rind3d lite-pilot --config configs/lite-v0.1.yaml \
  --scenes 10 --output outputs/my-lite-10

# 仅在匹配构建/配置的十场景完整通过后，才允许有上限的一百场景试产
uv run rind3d lite-pilot --config configs/lite-v0.1.yaml --scenes 100 \
  --pilot-report outputs/my-lite-10/report.json --output outputs/my-lite-100

uv run rind3d verify --root outputs/my-lite-10 --full-index-check
uv run rind3d verify --root outputs/my-lite-10 --physical
# 使用完全相同配置恢复；错误身份、损坏文件或过期证据会明确失败
uv run rind3d lite-pilot --config configs/lite-v0.1.yaml --scenes 10 \
  --output outputs/my-lite-10 --resume
```

当前入口限制最多 100 场景，不授权或启动量产。`attempts.jsonl` 保存每次尝试的几何/源身份和终止原因，`.staging` 保留未完成输出，已发布场景以原子目录提升提交。`report.json`、逐场报告和每秒遥测分别保留阶段耗时、候选数量、峰值资源和 GPU 温度。不要把未封存场或采窗成功等同于 READY_LITE。

```python
from rind3d import RINDDataset
from rind3d.data import make_dataloader

with RINDDataset("outputs/my-lite-10", split=None) as ds:
    example = ds[0]  # 默认只读取局部 packed 响应；不展开 1024³
    y = example["observed_response"].unpack()  # ZYX bool
    batch = next(iter(make_dataloader(ds, batch_size=1, num_workers=0)))
    refs = ds.private_oracle().get_candidate_sources(example["scene_id"], example["window_id"])
```

使用多 worker 时将启动代码放入 `if __name__ == "__main__":`。普通 DataLoader 不创建 GPU 上下文。`candidate_conditioned` 批次的参考源是变长列表。完整格式与现有读取选项仍见 README 和 `docs/format.md`。
