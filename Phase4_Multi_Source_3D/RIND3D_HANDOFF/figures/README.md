# Figures of `docs/RIND3D_数据集设计与技术说明.md`

**What this directory holds:** editable sources, compact data and outputs for the nine figures of the Chinese design document.
* **Real data only.** Every panel shows real data or a real solver run. There are no decorative Gaussians or schematic "probability blobs".
* **Drawn at document size.** The figures are made at the document's text width, 14.8 cm (5.83 in, `config.json:width_in`), so the font sizes are the printed sizes.
* **No chrome.** No titles, headers or decorative bands; the captions live in the document.

## Reproduce

```bash
source RIND3D_HANDOFF/env.sh
# 1. data (labeller venv, CUDA; ~2 min). Needs a RIND3D-100K scene container, e.g. slot 0 of contract v1 rendered
#    with the 100K branch into a scratch directory (not a campaign):
#    run_scene(cfg, 0, plan["k"], plan["c_table"], BackendFactory(), 0) -> slot_000000.npz
$RIND3D_LABEL_PY RIND3D_HANDOFF/figures/extract_data.py --container <scratch>/slot_000000.npz
# 2. figures (any Python with matplotlib >= 3.8, scipy, Pillow; LXGW WenKai font)
$RIND3D_TORCH_PY RIND3D_HANDOFF/figures/make_figures.py            # all; --only 04,08 for a subset
```

**Outputs.**
* `out/<name>.svg`, `.pdf` and `.png` (300 dpi). The 3D renders are raster inside the SVG/PDF; all text is vector.
* `out/contact_sheet.png`: every figure at document width, 620 px.
* `docs/assets/<name>.png`: a copy of each PNG for the Markdown document.

Reruns are byte-identical: no timestamps, fixed SVG hash salt, deterministic data.

**Files.**

| file | role |
|---|---|
| `config.json` | scene, channels, rays, the 100K window and source of figure 05, font, width, dpi |
| `extract_data.py` | data extraction. 1K scene 7 (train split): geometry, crops, rays, exact tau-pencil label, instrumented FastLabel run. The 100K slot-0 container: logs, work counters, one full-world plane from the certified renderer |
| `render3d.py` | small orthographic ray caster: convex bodies from exact half-spaces, voxel cutaways, first-hit voxels, depth-tested overlays |
| `make_figures.py` | the nine figures and the contact sheet |
| `data/` | `LOG.json`, the extraction record (tracked), and compact arrays (`*.npz`, `*.npy`, ≈ 0.2 MB). The arrays are **not tracked**, because the repository ignores `*.npz` / `*.npy`; regenerate them with `extract_data.py` |

**Checks recorded in `data/LOG.json`.**
* Every displayed 1K crop re-renders bit for bit with the certified CUDA renderer (`rerender_mismatch = 0`).
* The instrumented FastLabel run reproduces the stored release label byte for byte (`same_as_release = true`).
* TV(q_B, q_F) of the main task = 0.0118 against the exact tau-pencil label.

## Figures and captions

| file | caption (中文，放在文档中) | data |
|---|---|---|
| `01_research_task` | 研究任务：完整世界（几何与全部源，仅标注阶段可见）、Student 输入（单个窗口、单个源的响应）与监督标签（128³ 网格上的全局后验 q_F） | 1K 场景 7 窗口 12 源 1；精确 tau-pencil 标签 |
| `02_dataset_spec` | 一个 100K 场景：凸多面体、源与 12 个窗口按真实比例绘制；三种窗口尺寸相对 1024 体素世界与 128³ 标签网格的尺度 | 100K 合同 v1 slot 0 |
| `03_sampling_coverage` | 合同 v2 的配额安排（暂定 3 + 1，仅 K = 4 场景的一个 64³ 配额为 Coverage）；自然窗口中 m = 4 的比例 | 1K 世界模拟（E19）中的比例 |
| `04_response_types` | 四类局部响应：三维剖切、二维切片与 6-邻接定义；底部为已存标签的支持格数与耗时 | 1K 场景 7 窗口 6（源 1、2）与窗口 12（源 1、3）；逐位复核 |
| `05_local_rendering` | 完整响应场与 100K 局部按需计算：一个源的整面响应、全部配额候选的结果、每场景接收点评估数 | 100K slot 0：认证渲染器整面，`log_L*`，`meta.work` |
| `06_storage` | 100K 存储：每场景容器、窗口内分源响应、每个 mixed 通道一份标签、loader 展开 | 结构图（`rind3d_100k/storage.py`） |
| `07_posterior_field` | 一个真实的精确后验：F 是极薄的楔形片，跨 3,862 个网格；每格质量正比于 Vol(F ∩ C_i)，相差数个数量级 | 主任务的精确 tau-pencil 标签 |
| `08_fastlabel` | FastLabel 的一次真实运行：B₁ ⊇ F 与失败样本、细化后的 B₂、两阶段二项检验、聚合到 128³ 的 q_B；确定性出口 | 主任务的生产 FastLabel 运行（插桩） |
| `09_student_training` | 参考 Student：真实窗口汇聚到 32³ 的四个输入特征、模型、输出与交叉熵监督 | 主任务窗口 |

## Audit of the previous figures (2026-11-13)

| previous file | problems | replacement |
|---|---|---|
| `01_research_task` | decorative: sources outside the world box, flat "obstacles", a Gaussian blob as the posterior; no visibility | real 3D scene with ray-cast occlusion, the actual rays of the source, the real window crop and the real 3D posterior |
| `02_dataset_spec` | window sizes not to scale (64 vs 256 drawn ≈ 1.5×); title band | 12 windows of a real 100K scene to scale; scale bar against 1024 voxels and the 128³ grid |
| `03_response_types` | 2D toy grids; "SOLID-separated" defined only by a 2D picture, although the definition is 3D 6-adjacency | real windows in 3D cutaway and 2D slice, an explicit 6-adjacency definition, lit/dark contact faces drawn; stored label cost per type |
| `04_sampling_coverage` | title band; no acceptance probabilities; did not say the policy is a recommendation | 3 + 1 layout with c(m), the natural m = 4 shares that motivate it, "推荐候选" |
| `05_local_rendering` | random checkerboard as the "full field"; a single arrow as local rendering | a real full-world plane from the certified renderer; every real proposal of the scene; real work counters |
| `06_storage` | title band; mostly text | the same content as a layout, with mixed vs unlabelled channels and the RINDPack distinction |
| `07_fastlabel` | missing glyph (`q□`), right panel cropped, B ⊇ F drawn as a polygon in a circle, unlabelled dots, decorative Gaussian output | `08_fastlabel`: real rounds, samples and test counts of one production label |
| `08_student_training` | overflowing title, title band | `09_student_training`: real input features |
| (none) | the document had no figure of a real posterior field | `07_posterior_field` |
