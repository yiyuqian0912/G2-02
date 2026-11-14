# 先盘点真实复杂窗口，再决定训练规模

更新：2026-11-02。当前默认只盘点，不增加训练记录，不加载模型，不自动训练。原10万训练窗、验证/测试各12,500窗保持不变。

## 统计什么

只扫描原标签协议冻结的训练场景池。默认检查每个配置已有的最多256个窗口，size为16/32/64/128；不人为生成新的ROI。先运行原有最低正残差/BFS分层，跳过全亮、全暗、无自由像素、分层不完整等情况。只有完整算术重建、有效层数R≥3的窗口进入新增复杂候选名单。

R是有效贡献层数，不是物理源数K。盘点不做逐层单源物理可实现性验证，也不执行全场候选光源渲染；“可用”指符合当前分层标签规则，不能直接等同于已验证的物理源答案。

会分别报告两种数量：

1. **去重后的新增复杂窗口总量**：排除原训练窗口，再按“几何＋真实源配置＋窗口位置/size”去重。重复view ID或重复场景副本不凑数。
2. **当前多样性限额下的可选量**：在上述名单中，按固定种子与顺序，每个源配置最多选4个新窗、每个几何组最多选16个新窗。限额只用于单独报告，不会提前停止完整盘点，也不会把超限窗口从总量中删掉。这是当前贪心选择规则的结果，不声称是限额下的组合最优值。

同时保存原训练层数分布、新窗口按R/size及R×size分布、配置/几何覆盖、跳过原因和耗时。“总量”始终限定在上述训练池、size和遍历上限内，不代表所有可能连续窗口的总数。

真实源参数只用于离线去重，不用于决定残差传播，也不输入学生。冻结验证/测试几何不能进入新增训练名单。统计不会读取验证/测试响应来调整训练筛选规则。

## 运行

Mac上传同名包：

```bash
scp -P 10688 ~/Desktop/RIND_Phase2_Update.tar.gz \
  test@58.199.162.221:/home/qq/yiyu/
```

服务器解压并启动后台盘点：

```bash
cd /home/qq/yiyu
tar -xzf RIND_Phase2_Update.tar.gz
cd RIND_Phase2_Training_Minimal
./run_count_masks_tmux.sh start prepare
./run_count_masks_tmux.sh status
./run_count_masks_tmux.sh logs
```

tmux会话为`rind_masks_count_first`。启动后可断开SSH/VPN。若此前同名会话还在运行，先确认其任务；需要停止旧任务时用`./run_count_masks_tmux.sh stop`，不会自动停止其他训练。

当前默认配置下，`start all`同样只盘点；`start train`和`start evaluate`会被拒绝，防止规模尚未确定就启动训练。盘点不需要指定GPU数量，也不申请GPU。局部区域分割与BFS使用现有Numba代码及8个准备进程；不把这部分几何拓扑运算描述为GPU训练。

## 依赖与参数

配置为`Phase2_Multi_Source/configs/masks_count_first.json`。

| 字段 | 当前值 | 用途 |
|---|---|---|
| `training_expansion.mode` | `inventory` | 只盘点的明确开关 |
| `training_expansion.additional_complex_windows` | `"all"` | 扫完声明的范围，不设置新增训练目标 |
| `label_preparation.record_counts.train` | `"all"` | 与盘点模式一致；不是自动训练全部数据 |
| `training_expansion.min_layers` | 3 | 新复杂窗口的最低有效层数 |
| `training_expansion.window_sizes` | 16/32/64/128 | 本次盘点的原生窗口size |
| `training_expansion.max_views_per_scene` | 256 | 每个配置最多检查的已有窗口数，旧训练窗会排除 |
| `training_expansion.max_added_per_configuration` | 4 | 仅用于第二种可选量统计 |
| `training_expansion.max_added_per_geometry` | 16 | 仅用于第二种可选量统计 |
| `training_expansion.workers` | 8 | 准备进程数，非GPU数 |
| `training_expansion.chunk_size` | 256 | 每个名单块最多保存的新复杂窗口数 |
| `training_expansion.seed` | 20261102 | 决定固定的遍历和限额选择顺序 |

服务器需要已安装的`data/raw/phase2-merged-110k`及引用组件、`outputs/masks_boundary_balanced_v3/mask_records.json`及其缓存来源、`outputs/masks_only_100k/mask_protocol.json`。**不需要模型权重。** 只保留旧缓存无法得知新窗口的实际层数，因此仍须读取新数据的局部响应和几何。

已提交名单块可断点恢复；只会重算尚未提交或预取但未提交的工作。修改seed、size、限额或数据协议后应换新的输出目录；workers/chunk_size可调整而不改变盘点语义。

## 输出与解读

目录：`Phase2_Multi_Source/outputs/masks_count_inventory_v4/`。

- `inventory_protocol.json`：冻结原索引哈希、数据发布、标签算法和扫描范围。
- `inventory_chunks/*.json.gz`：去重后的真实复杂窗口名单、R、各层强度、教师标签摘要和限额选择标记；只保存元数据，不保存响应/层掩码数组。
- `inventory_report.json`：各类原始计数、层数/size分布、覆盖、限额内数量和耗时。
- `tmux_runs/`：启动命令、后台日志、退出码。

先确认`scan_complete=true`，再将`new_unique_complex_windows`作为本轮完整盘点数量；运行中的`false`只表示当前已发现数量。`available_with_current_diversity_caps`是同一范围内按当前限额选出的数量。`possible_total_training_windows_*`只是“原训练数＋可用新增数”的参考上限，尚未选定训练规模。

`raw_completed_candidates_by_R`是去重和复杂度筛选前的诊断计数，不能当作新增独立样本量。没有边界/失败/重复/限额的原因见`statistics`。

完成后可把报告下载回本地：

```bash
mkdir -p ~/Desktop/G2-02/outputs/evaluation/complex_inventory
scp -P 10688 \
  test@58.199.162.221:/home/qq/yiyu/RIND_Phase2_Training_Minimal/Phase2_Multi_Source/outputs/masks_count_inventory_v4/inventory_report.json \
  ~/Desktop/G2-02/outputs/evaluation/complex_inventory/
```

盘点不会创建`mask_records.json`、`mask_cache/`或`models/`。选择训练规模后，再准备标签缓存。原有计数优先训练实现保留，未来配置模板在`configs/masks_count_first_train_template.json`；模板里的10万新增是之前的草案，不是本次已确认规模，也不是当前默认入口。

本地检查覆盖中断恢复、去重、限额内外分开计数、冻结原缓存、协议/校验和拒绝，以及Python/Shell/tmux入口不会意外训练。小规模真实旧RIND数据用于检查执行，不作为110k数据可用总量的估计；正式数量以服务器完整盘点为准。
