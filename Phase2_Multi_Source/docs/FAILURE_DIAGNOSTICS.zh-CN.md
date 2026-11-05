# 分层解耦失败诊断：层数、启用槽、形状、强度、重建

2026-11-03。只增加评价和统计，不改模型、训练loss、选模规则、数据标签或checkpoint签名。已有权重可以直接评估，无需重训、重新盘点或候选源渲染。

## 1. 本次要回答什么

针对低strict，分别检查：

1. 层数预测是否正确。
2. 层数正确时，实际启用的槽是否与原全槽匹配得到的教师对应集合一致。
3. 每个教师有效层的形状是否全部达标。
4. 每个教师有效层的强度是否全部达标。
5. 模型实际部署重建是否达标。

层数错误时，“数对后是否选对槽”记为NA，不当作通过，也不将它重复记成第二个独立计数原因。形状、强度和重建可以同时失败；重叠失败不是唯一因果分摊。

报告包含两套分母：全部窗口，以及“已经数对层数/启用槽完全匹配”的条件窗口。重点查看`count_partition`中数错、数对但不完整、完整通过三类；这三类互斥且合计全部窗口。

## 2. 同时检查两种匹配口径

- **reported**：保留原评价的全槽Hungarian对应。未启用槽也参与形状匹配；严格成功还要求对应真层的槽均被启用。
- **deployed**：优先将真层匹配到实际启用槽、空标签匹配到未启用槽；数量不同时保留全部教师层，未检出的层以零掩码/强度处理，数量错误仍必定失败。

“reported失败而deployed通过”可能说明原匹配偏好未启用的重复或略优槽，影响了原strict。它是评价匹配敏感性，不是新权重取得的性能提升。原reported strict不被覆盖。

`oracle_pool`仅为教师指导的槽池参考：在所有原始槽中匹配R个教师层，不要求这些槽真的被模型启用。若它通过而实际集合失败，证明槽池里至少存在这一组较好的匹配；若它失败，不证明所有其它组合都失败。它不是穷举最优上界，不是可部署输出。

所有对应仅针对本窗口内的教师有效贡献层，不赋予slot跨窗口固定源身份，也不将R当作物理源数K。

## 3. 原strict和放宽参考并行

| 条件 | strict | relaxed（开发参考） |
|---|---:|---:|
| 层数、启用集合一一对应 | 完全正确 | 完全正确 |
| 每层二值IoU | ≥0.90 | ≥0.85 |
| 每层相对强度误差 | ≤10% | ≤15% |
| 实际重建相对L1 | ≤5% | ≤8% |

掩码先以0.5二值化，屏蔽障碍物；强度使用未乘存在概率的原始预测强度。重建使用模型原本的实际输出，分母是自由像素上的观测响应总和，不是窗口峰值。

CSV同时保存软Dice、硬Dice、IoU和相对强度误差。训练日志的平均软Dice不能等同于所有层通过IoU阈值。放宽通过率是另一评价口径，不能当成模型改善。

## 4. 具体统计

`failure_analysis.json`：

- `overall.reported/deployed.strict/relaxed.rates`：各项通过数、分母、窗口等权、场景等权和场景聚类bootstrap 95%CI。
- `success_given_correct_count`、`selection_given_correct_count`，以及形状/强度/重建在数对后的通过率。
- `exact_failure_patterns`：互斥的完整失败组合及pass，总数必须等于窗口数。
- `only_one_failure`、`multiple_failures`、两两失败交集矩阵。
- `sequential_first_failure`：按count→slot→shape→strength→reconstruction筛查的首次失败量，依赖顺序，不作唯一因果贡献。
- `by_R`、`by_R_group`（1/2/3/4/≥5）、`by_size`：各层级原始计数与比例。总表提供聚类CI；分组表提供点估计。
- `minimum_layer_iou`、`maximum_layer_strength_relative_error`：每窗最差层的均值、分位数和极值。
- `layer_strata`：按教师支持面积占全窗口比例、层强度占窗口峰值比例分组，保存每组分母及形状/强度失败率。
- `matching_sensitivity`：原规则与实际启用集合匹配的差异、教师匹配池与实际启用集合差距。

置信区间以scene_id聚类，包含同场景多窗相关性；仅凭缓存无法额外确认不同scene_id的几何独立性，报告明确保留此限制。零分母条件率为NA，不伪造0%。

## 5. 形状和强度的替换敏感性

只在层数正确、使用二值槽激活的v5窗口上，用同一批窗口计算：

- 原始实际输出的响应相对L1。
- 固定预测掩码，换成对应教师强度后的误差。
- 固定预测强度，换成教师掩码后的误差。
- 将预测掩码硬阈值化后的误差。

这些只用于诊断，不进入推理或训练，也不保证误差一定下降。两项替换都可能因误差补偿而变差。

`softmask_strength_compensation`专门统计：硬形状过关、原始强度不过关，但“预测强度×教师支持内平均掩码概率”接近教师强度的层。这是软掩码/幅度补偿候选，不自动认定模型学到了正确物理强度。

同时检查教师掩码×教师强度自身的闭合重建误差；若缓存标签本身不闭合，不能全部归因于学生。

## 6. Checkpoint与正在运行的训练

默认读取来源训练会话保存的配置，保留实际扩充N；评价**最近完整保存的last.pt**，不是GPU里正在更新到半轮的参数。

rank0一次读取并冻结该checkpoint的模型快照，然后所有评价rank读取同一份不可变快照，避免last.pt在不同rank加载时刚好更新。输出明确记录best/last、epoch、split、源checkpoint SHA256、快照SHA256、缓存/窗口名单摘要及诊断代码摘要。

快照只在服务器留存，用于复现；下载包不含权重或原始响应。独立诊断不会覆盖正式best评价文件，也不会覆盖正在训练会话的日志指针。

## 7. 更新和执行

Mac上传同名代码包：

```bash
scp -P 10688 ~/Desktop/RIND_Phase2_Update.tar.gz \
  test@58.199.162.221:/home/qq/yiyu/
```

服务器：

```bash
cd /home/qq/yiyu
tar -xzf RIND_Phase2_Update.tar.gz
cd RIND_Phase2_Training_Minimal
unset RIND_MASKS_CONFIG RIND_DECOMPOSITION_CONFIG

./run_failure_diagnostics_tmux.sh start \
  --source-session rind_masks_disentangled_more \
  --checkpoint last --split validation

./run_failure_diagnostics_tmux.sh status
./run_failure_diagnostics_tmux.sh logs
```

若你的训练会话名称不同，只改`--source-session`。诊断会话默认`rind_failure_diagnostics`，默认选择一张符合条件的GPU、每卡batch64；不启动训练，不做自动反向batch探测。tmux启动后可断开SSH/VPN。

没有合格空闲卡时不会回退CPU或干扰其它进程。可以等训练结束后运行；若明确允许短时共享计算，额外传`--max-utilization 100 --min-free-memory-mib 32768`，仍自动选卡，但会与同卡任务竞争算力。

训练完成、选型结束后，固定测试集上的best诊断：

```bash
./run_failure_diagnostics_tmux.sh start \
  --source-session rind_masks_disentangled_more \
  --checkpoint best --split test
```

检查checkpoint角色和epoch后再解释结果，不把last与best混为一谈，不用训练期间的测试结果选择模型。

## 8. 结果与下载

固定根目录：`Phase2_Multi_Source/outputs/failure_diagnostics/`。

- `jobs/rind_failure_diagnostics/`：本次诊断日志、退出码和来源配置。
- `latest.json`：最近完成的统计目录及checkpoint身份。
- `latest_diagnostics.tar.gz`：可直接下载的轻量结果包。
- 按实验/seed/role/split/epoch/hash划分的目录保留每次结果。

结果包内：

| 文件 | 内容 |
|---|---|
| `failure_analysis.zh-CN.md` | 中文诊断报告及原始计数、比例、CI |
| `failure_analysis.json` | 完整结构化统计和分层结果 |
| `failure_breakdown.png` | 通过率、失败组合、按R成功率、替换敏感性 |
| `windows.csv.gz` | 每个窗口的分项通过与失败原因 |
| `layers.csv.gz` | 每个有效层的两种匹配、是否启用、IoU、软/硬Dice、强度与支持概率 |
| `diagnostic_windows.jsonl.gz` | 完整窗口诊断与匹配信息 |
| `checkpoint_metadata.json` | 本次权重身份、epoch、split与摘要 |
| `BUNDLE_MANIFEST.json` | 文件校验和，明确不含权重和响应 |

状态显示本次`exit code: 0`后，在Mac下载并解压：

```bash
mkdir -p ~/Desktop/G2-02/outputs/evaluation/failure_diagnostics
scp -P 10688 \
  test@58.199.162.221:/home/qq/yiyu/RIND_Phase2_Training_Minimal/Phase2_Multi_Source/outputs/failure_diagnostics/latest_diagnostics.tar.gz \
  ~/Desktop/G2-02/outputs/evaluation/failure_diagnostics/

tar -xzf ~/Desktop/G2-02/outputs/evaluation/failure_diagnostics/latest_diagnostics.tar.gz \
  -C ~/Desktop/G2-02/outputs/evaluation/failure_diagnostics/
```

先核对`checkpoint_metadata.json`对应本次任务；未完成/失败的任务不会更新latest结果包，不能把上一次的包误作新结果。
