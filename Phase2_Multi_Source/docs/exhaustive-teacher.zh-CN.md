# 阴影筛查后的穷举教师

独立脚本 `scripts/exhaustive_teacher.py`，本轮不运行逐层教师、不修改训练或原始数据文件。用已有四分树窗口建立本次实验的有效窗口名单；不为全亮/全暗窗口生成教师。

## 窗口和物理约束

只看非障碍物像素：全零为全暗、全部大于零为全亮，两者跳过。即使全正窗口有强度变化，本轮也按要求跳过。没有自由像素的窗口跳过。保留同时具有零值和正值的窗口。float64 正强度加和的零值是精确零，默认不使用将很小正数当成零的阈值。

在全场规则网格上排除观测窗口内部、障碍物内部位置，然后逐候选计算单位强度可见性。任何照亮至少一个自由零响应像素的候选都排除，因为正强度线性叠加不能抵消该贡献。其他正响应区域不能作为零值阴影约束。全窗口不可见的候选仍保留，避免无依据排除隐藏源。筛查只证明网格点是否满足零响应必要条件，不证明整个网格单元可行，也不是充分条件。

后续穷举只在保留位置上进行，每个 k 枚举 C(n,k) 个不重复位置组合。每组强度由线性规划优化全窗口 MAE，界限 [min-strength,1]。每个 k 单独按 exp(-MAE/temperature) 归一化。该分布以零响应约束为硬筛查，区别于原来允许少量阴影误差的全域软 Gibbs 分布；也不是强度积分后的后验。

## 运行

在 Phase2 目录执行。首次运行 uv 自动提供 scipy、matplotlib。

列出场景中可用的真实窗口（终端只显示 eligible；views.json 保存所有窗口和跳过原因）：

```bash
./run_exhaustive.sh --scene-id 993 --list-views --output outputs/exhaustive/views993
```

先用 16 像素间距筛查，输出候选区域图和各 k 的真实组合数：

```bash
./run_exhaustive.sh --scene-id 993 --view-id 11 --spacing 16 \
  --counts 1 2 3 4 --screen-only --output outputs/exhaustive/s993-v11-grid16
open outputs/exhaustive/s993-v11-grid16/screening.png
```

移除 --screen-only 即可执行同一设置；预算超限会停止，不会截断，也不会自动降低密度。若要改变 counts 或 spacing，使用新的输出目录。

先完整运行一轮高密度 k=1：

```bash
./run_exhaustive.sh --scene-id 993 --view-id 11 --spacing 16 \
  --counts 1 --output outputs/exhaustive/s993-v11-grid16-k1
open outputs/exhaustive/s993-v11-grid16-k1/distributions.png
```

较细网格的 k=1..4 实验可先尝试间距 64，预算显式设置；仍应先用 --screen-only 查看规模：

```bash
./run_exhaustive.sh --scene-id 993 --view-id 11 --spacing 64 \
  --counts 1 2 3 4 --max-combinations 20000000 \
  --output outputs/exhaustive/s993-v11-grid64
```

该命令不保证快速完成：组合数、不同掩码数和窗口面积共同决定成本。不要把提高预算误认为加速。真实例子 spacing=16：3874 个合法位置筛到 1842 个，C(1842,4)=478114290780，仍不适合逐配置穷举。更细间距可设 8 或 4，但尤其要检查预算。

## 输出和恢复

- `parameters.json`：所有实验参数；目录不允许混用不同参数。
- `report.json`：筛查前后数量、组合数、每个完成 k 的最佳 MAE、时间和状态。
- `screening.png`：响应、零响应筛查像素、全场候选保留区域。
- `distributions.png`：各 k 的源位置概率投影和最佳重建。青框是窗口，青色加号是最佳组合位置。
- `screen.npz`：全部网格坐标、保留标记、观测等。
- `visibility-*.npz`：带几何、观测、代码哈希的逐源可见性缓存。
- `k*.npz`：所有位置组合、最优强度、MAE、概率和空间投影。
- `working-*.npy`：正在运行 k 的磁盘工作数组；正常完成后删除。

图的每个网格点表示 E[该点源数]/k，和为 1；不插值成虚假的像素精度。k 倍该值是无重复位置条件下该点出现源的概率。不同 k 分别归一化，不在 k 间比较证据。

Ctrl+C 停止；同一命令再次运行会重用筛查缓存和已完成 k。正在运行但未完成的 k 重新开始，不是逐组合断点恢复。可视化重画不需要重新渲染：

```bash
./run_exhaustive.sh --plot-only --output outputs/exhaustive/s993-v11-grid16-k1
```

已有存量全亮/全暗窗口不会被删除，原数据生成器不改变；过滤只作用于本实验入口。旧 compare_teachers.py 保留其旧行为，之后比较两种教师时需要统一采用本轮筛查域。
