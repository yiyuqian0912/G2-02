# RIND Phase I：局部响应驱动的源定位

[English](README.md) · [方法定义](docs/method.md) · [统一接口](docs/interfaces.md) · [训练与实验](docs/experiments.zh-CN.md) · [结果浏览](docs/dense_probability.zh-CN.md)

目标是根据**局部响应与窗口位置/size**，预测二维源空间中的可能性分布，保留多解，而不是回归唯一源坐标。物理教师用隐藏场景重渲染候选源，学生学习条件能量，推理时不调用重渲染。

目前已实现数据读取、边界加权响应代价、可选显式边界代价、均匀教师、条件能量训练、断点恢复、测试评价及交互式结果浏览。自适应搜索已有独立对照实现；其完整的“便宜响应筛选 → 不确定区域细化 → 精细边界评价”尚未接入正式训练。受控的一条/两条独立边界实验也尚未完成，不能把现有结果当作这些研究问题的验证。

## 先运行什么

在项目根目录准备环境和数据：

```bash
uv sync --locked --extra train --extra browser --extra diagnostics
# 首次使用：将单源数据 ZIP 放在项目根目录，然后运行
./scripts/install_data.sh
```

已有数据保存在 `data/raw/phase1-single-source/`。数据包下载：[单源 RIND](https://drive.google.com/file/d/1_zerwmggBskUtPtkhWTb6Xqs4kIU5L5I/view?usp=sharing)。安装器校验数据；默认在首次安装成功后删除原 ZIP，需保留时加 `--keep-archive`。重复安装不覆盖已安装实例。数据生成器不是本仓库的正式训练入口，当前训练读取已生成的单源发布包。

查看**已有训练结果**：

```bash
./view_results.sh
```

打开 **http://127.0.0.1:8767**。选实验、seed、测试场景和局部视窗，点击“生成对比”。三栏分别是局部响应、模型预测、教师真实采样。默认查询间距为 1，即 1024×1024 个源像素中心的真实模型计算；也可降低查询密度。局部数据浏览器仍通过 `./scripts/browser.sh` 启动，端口为 8766。

启动新的 **5000 个训练观测**实验：

```bash
./run_training.sh --run-name consistent_5000 \
  --scene-counts 1250 125 125 --window-sizes 16 32 64 128 \
  --set views_per_size=1 --epochs 30 --patience 8 --seeds 0 \
  --spacing 64 --fourier-frequencies 4 --device cpu --threads 4
```

这是 1250×4=5000 个训练观测，加上各 500 个验证/测试观测。默认模板是 `configs/experiment_consistent.json`，输出为 `outputs/experiments/consistent_5000/`。先加 `--check` 可检查参数而不运行。修改已启动实验的参数应换 `--run-name`；详情见训练文档。

## 与 proposal 对齐后的约定

| 内容 | 当前正式约定 |
|---|---|
| 物理环境 | 单源、强度 1、连续二维硬遮挡；无衰减、反射、噪声和材料差异 |
| 观察窗口 | 使用后续讨论确定的动态四分树，取代最初固定窗口设定；最小 size=16 |
| 学生输入 | `response`、`window=[x,y,size]`、查询坐标 `candidate_xy`；默认不使用障碍掩码 |
| 教师输入 | 上述信息及隐藏完整场景；只用于物理监督和评价 |
| 学生主概率范围 | 世界减去观测窗口。不能用隐藏障碍物掩码帮学生删掉错误预测 |
| 几何辅助结果 | 可额外排除障碍物，但单独标记为 `geometry`，不是公开输入主结果 |
| 教师概率 | 每个实际候选点等权的 `softmax(-C/temperature)`；不乘面积 |
| 正式训练候选 | 完整均匀网格，作为可检查的物理参考；自适应搜索单独比较遗漏与成本 |
| 划分 | 固定、互斥的 scene 池；改变训练规模不会把原验证/测试场景移入训练 |
| 坐标 | 世界坐标，x 向右、y 向下；学生内部统一归一化 |
| 模型选择 | 验证 CE 选择 best checkpoint；训练完成后正式测试 |

旧 `train5000` / `round1_cpu` 使用局部障碍掩码与几何辅助归一化，且部分配置的 Fourier 频率与候选间距不匹配。旧权重保留可读，结果浏览器会显示提示；**新约定必须重新训练，无法靠绘图修复旧权重**。v2 固定池沿用 `train5000` 的验证/测试场景便于对照；这些场景已经看过，不能宣称为全新的最终盲测集。

## 数据与输入分别用来做什么

发布包含 10,000 个场景、180,000 个 source-free 观测。每场景 18 个窗口：size=16、32、64、128、256、512 各 3 个。仅含源的四分树节点继续细分，无源节点立即保留；达到最小 size 的含源叶子被排除。模型不读取整棵树或真实源坐标。

| 输入 | 作用 | 怎么使用 |
|---|---|---|
| RIND 数据路径 `data_root` | 指明响应、窗口和教师几何来自哪个发布包 | 在实验配置中填数据目录；已有数据直接读取，不重新生成或覆盖 |
| `scene_counts` | 控制三集合各包含多少个独立场景 | 按 train/validation/test 的顺序设置；不是 observation 数量 |
| `window_sizes`、`views_per_size` | 决定每个场景取哪些原始窗口 | 观测数 = 场景数 × size 种类数 × 每种 size 的窗口数；不缩放响应图 |
| 固定 source intensity | 保持正向过程可比，避免把强度变化误判为空间变化 | 当前发布包和重渲染均固定为 1，不作为待估参数 |
| random seed | 重现模型初始化、batch 顺序和窗口选择 | 训练 seeds 负责重复实验；v2 窗口选择使用固定清单中的 window_seed |
| quadtree 设置 | 决定数据生成阶段有哪些 source-free 窗口 | 当前发布包最小 size=16；训练只选择现成窗口。修改生成规则需要新数据版本 |
| split 设置/清单 | 隔离场景，防止同一几何泄漏 | v2 读取 `configs/splits/phase1_fixed_v2.json`；取各池前 N 个场景 |
| `response[size,size]` | 提供可见/遮挡响应与边界形状 | 原始分辨率送入学生；教师与重渲染结果比较 |
| `window=[x,y,size]` | 将局部像素放回世界，提供位置和尺度 | 局部像素 `[row,col]` 对应 `[x+col+0.5,y+row+0.5]`；size 也用于组批 |
| `candidate_xy[N,2]` | 指定要检验的源假设 | 教师逐点重渲染，学生逐点给能量；不是源坐标标签 |
| `obstacle[size,size]` | 数据检查及可选输入消融 | 保留在记录里；默认 `use_obstacle=false`，不会作为主模型特征 |
| `valid[N]` | 标识教师几何上允许的候选 | 教师无效点概率为 0；主学生仍需为其输出能量并接受概率质量惩罚 |
| `support[N]` | 明确学生在哪些候选上归一化 | `world` 为当前 world-minus-window 候选全部；`geometry` 才等于 valid |

## 项目结构

| 目录 | 职责 |
|---|---|
| `src/rind_phase1/` | 数据、物理教师、模型、训练编排与结果浏览 |
| `part_e/` | 正式评价、接口检查、自适应搜索对照 |
| `configs/` | 共享实验模板与固定 scene 划分 |
| `scripts/` | 安装、训练启动、报告与静态导出 |
| `tests/` | 单元测试及真实数据集成检查 |
| `vendor/rind-dataset/` | 固定版本的 RIND 读取与渲染依赖 |
| `student_baseline/` | 仍被 legacy 适配器引用的兼容实现 |

[每个核心文件的任务与验收目标](docs/modules.zh-CN.md) · [提交范围](docs/integration.md)

## 什么算实验结论，什么还不能说明

模型与教师都在候选集合内归一化。即使所有候选都不匹配，仍会得到总和为 1 的分布，所以必须同时看物理代价和兼容点覆盖。查询更多学生点只改变推理采样密度，不增加物理监督。

教师知道隐藏场景 `S`，学生不知道。同一个局部输入可能来自不同场景，因此学生未必能恢复每个场景独有的教师分布；结果是给定局部信息的兼容性预测，不是严格可辨识性的保证。常量/非恒定响应分组不等于独立边界数量；障碍物类型组合分组也不等于全部几何泛化。

检查代码：

```bash
PYTHONPATH=.:src NUMBA_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests -q
PYTHONPATH=.:src NUMBA_NUM_THREADS=2 .venv/bin/python -m unittest part_e.test_part_e part_e.test_integration -q
```
