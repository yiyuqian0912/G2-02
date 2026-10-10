# 正式实验与训练流程

当前默认配置是 [`configs/experiment_consistent.json`](../configs/experiment_consistent.json)，协议 v2。旧 `experiment_round1.json` / `experiment_train5000.json` 保留用于解释已有实验，不代表新默认方法。

## 5000 个训练观测的建议起点

在项目根目录运行：

```bash
./run_training.sh --run-name consistent_5000 \
  --scene-counts 1250 125 125 --window-sizes 16 32 64 128 \
  --set views_per_size=1 --epochs 30 --patience 8 --seeds 0 \
  --batch-size 4 --lr 0.001 --hidden 64 \
  --spacing 64 --fourier-frequencies 4 --device cpu --threads 4
```

观测数：训练 1250×4=5000，验证 125×4=500，测试 125×4=500。这不等于全部 180000 个窗口。默认 response variant 使用边界加权响应代价、beta=0；学生输入为 response/window，归一化范围为 world-minus-window。

可先在同一命令后加 `--check`，检查参数且不写文件；加 `--configure-only` 只保存配置。`--set` 允许覆盖已有配置字段，例如 `--set training.use_obstacle=true`，布尔值用 JSON 的 true/false。若重新运行同一个已存在的 run-name，启动器读取已经保存的配置，不用新模板覆盖它。

| 参数 | 控制什么、如何选择 |
|---|---|
| `--run-name` | 实验独立目录与配置名；变更参数时取新名字 |
| `--scene-counts TRAIN VAL TEST` | 各集合场景数，不是窗口数；不能超过固定池上限 |
| `--window-sizes`、`--set views_per_size=...` | 选择现成四分树窗口；每种 size 最多 3 个；大 size 显著增加计算量 |
| `--spacing` | 教师实际候选间距；64 约为 16×16 个世界网格点，排除窗口后更少；减半约增为四倍候选 |
| `--temperature` | 教师软分布温度；默认 0.05，小值更集中；只影响监督 q |
| `--fourier-frequencies` | 候选坐标频率数量；v2 要求最短周期至少覆盖 4 个训练网格间距 |
| `--hidden` | 学生表示宽度；默认 64；增加会提高模型和注意力成本 |
| `--epochs`、`--patience` | 最大 epoch 与验证 CE 无改善早停轮数；默认 30/8 |
| `--batch-size` | 一批同 size 的观测数；CPU 先用 4，内存紧张用 1/2 |
| `--lr` | AdamW 学习率；默认 0.001 |
| `--set training.candidate_chunk=256` | 单次注意力计算的候选数；降低峰值内存，不改变全局 softmax |
| `--seeds 0 1 2` | 独立初始化/训练重复；先跑 0，再做多种子比较 |
| `--device cpu`、`--threads 4` | 当前机器可用的训练设备和线程数；不自动假设 CUDA |
| `--set training.use_obstacle=true` | 额外使用局部掩码的输入消融，必须新训练并单独命名 |
| `--set training.normalization_support='"geometry"'` | 隐藏几何辅助归一化消融；主结果仍应报告 world |

v2 固定池共有 train=9750、validation=125、test=125 个场景。1250→更大训练规模不会移动留出场景，原 scene/size 的窗口选择也保持不变。若要新的划分，创建新的清单/协议；不要修改已冻结实验正在使用的清单。现有固定池继承已检查过的 `train5000` 留出集合，不是未见过的最终盲测集。

## 输出到哪里

```text
configs/experiment_<run-name>.json           启动器保存的参数
outputs/experiments/<run-name>/
  protocol.json                             冻结配置、代码/数据身份、scene/view 清单
  training.log                              持续追加的终端日志
  response/                                 beta=0 的正式基线
    records.json                            train/validation/test 教师文件列表
    teachers.json                           教师来源、耗时、兼容网格覆盖审计
    teachers/s<scene>_v<view>.npz             物理代价、q、valid、候选坐标
    seed_0/
      best.pt                               验证 CE 最优 checkpoint
      last.pt                               最近完整 epoch 的恢复状态
      training.json                         每轮 train/validation CE 与早停信息
      test_metrics.json                     训练结束后的正式测试报告
      predictions/*.npz                     教师原始候选上的学生能量/概率
      example_*.svg                         少量旧式候选概览，不是逐像素图
      sampling_probe.json                   手动运行 probe 后的验证细网格结果
    summary.json、RESULTS.md                 各 seed 的场景平均指标汇总
    search_comparison.json                  启用搜索对照时的保留/遗漏与成本
  edge/                                     显式边界项消融，结构相同
```

在结果页面生成图会保留最近结果于服务内存，不改原实验；点击下载才导出 PNG/NPZ 到浏览器的下载位置。静态导出脚本默认保存在选定 seed 的 `dense/` 子目录。

## 停止、恢复、改变参数

终端按 **Ctrl+C** 停止。参数、数据和代码未改变时，重新运行同一 run-name，会恢复 `last.pt` 的模型、优化器、随机状态和已完成 epoch。中断的半轮从上一完整 epoch 后重新执行；首次完整 epoch 前没有恢复点。达到早停或最大轮数的实验再次运行不会凭空增加训练。

要换参数，用新的 `--run-name`。**本次方法/代码更新后，旧实验目录会因来源哈希不同拒绝续训**，这是避免把两套方法混成一个结果；已有 checkpoint 仍可用于查看和推理。不要改旧 `protocol.json` 绕过检查。需要旧实验原样续训时使用与它匹配的原始代码版本。

## 分阶段运行

一键脚本默认执行 `all`：prepare → train → evaluate → summarize，以及配置启用的 search 对照。研究阶段也可分开执行。以下用默认配置，其输出目录是 `outputs/experiments/consistent_v2`，与上例的 consistent_5000 是不同实验：

```bash
export PYTHONPATH="$PWD:$PWD/src:$PWD/vendor/rind-dataset"
.venv/bin/python -m rind_phase1.experiments plan --config configs/experiment_consistent.json
.venv/bin/python -m rind_phase1.experiments prepare --config configs/experiment_consistent.json
.venv/bin/python -m rind_phase1.experiments train --config configs/experiment_consistent.json
.venv/bin/python -m rind_phase1.experiments evaluate --config configs/experiment_consistent.json
.venv/bin/python -m rind_phase1.experiments summarize --config configs/experiment_consistent.json
```

使用 CPU 时可设置 `NUMBA_NUM_THREADS=4`，并保持配置中的线程数相同。一键启动器会设置它。正式 evaluate 会检查训练完成条件和 checkpoint 来源，不接受未训练完成的临时模型作为最终结果。

## 与 proposal 对应的实验

1. **教师拟合与泛化基线：**先完成 response。看 KL/JS/L1 相对同范围 uniform baseline 的改善，同时看物理兼容质量、无兼容网格点比例和 invalid mass。不能只看坐标误差或漂亮的密集图。
2. **显式边界项：**在同一个冻结配置下追加 `--variant edge`。保持 scene/view、候选、训练超参及 seeds 一致。edge 既与自己的教师比较，也用共同 response 教师评价，避免把不同代价定义直接当性能优劣。
3. **源空间采样：**运行 validation-only 的 `probe`，默认从 spacing=64 到 32，重新调用物理教师检验新坐标。使用验证结果决定下一次新实验的频率/spacing；不要按测试图调参。该检查只覆盖配置的前 N 个验证视窗，不证明逐像素收敛。
4. **搜索成本：**运行 `search` 或启用默认 all 中的对照。当前是独立 block scouting/refinement 原型，报告遗漏教师/兼容质量与实际重渲染次数；不能声称已完成不确定性驱动的级联教师。
5. **模糊性与边界数量：**常量/非恒定分组已实现；一条和两条独立边界的受控配对数据及其评价尚需单独设计。当前熵变化不能替代这个实验。
6. **障碍物组合：**现有报告按类型 multiset 区分训练中是否出现过，只是有限的组合诊断；更严格的几何组合留出需要新的预注册划分。

```bash
# 以下命令针对前面的 consistent_5000 实验
./run_training.sh --run-name consistent_5000 --variant edge
PYTHONPATH=.:src .venv/bin/python -m rind_phase1.experiments probe \
  --config configs/experiment_consistent_5000.json --variant response --seed 0
PYTHONPATH=.:src .venv/bin/python scripts/report_experiment.py \
  --root outputs/experiments/consistent_5000
```

报告中的 true expected physical cost 在 invalid mass>0 时为无穷，以 null 加标志记录；valid-conditional cost 和有限惩罚 cost 是另列指标，不应混称。dense 页面下方 KL 等指标始终在教师原始共同候选上计算，不把低分辨率教师插值成逐像素真值。
