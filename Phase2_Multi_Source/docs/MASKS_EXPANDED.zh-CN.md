# 从2万扩展到10万训练视窗

当前2万版本已经读取 `data/raw/phase2-merged-110k`。这个联合目录包含旧1万配置与新10万配置；2万是训练视窗的采样上限，并不是只使用了旧数据。新增数据发布包为 `RIND-random-v3-100000scenes-multi-unit-256views.zip`，实际是10万配置、每配置256个随机窗口，原始强度为1。已有安装器保留原发布，创建随机(0,1)强度派生版本，联合读取器使用派生总响应。扩展不更换这套数据和分层规则。

默认扩展规模：**训练100,000个有效视窗，验证2,000个，测试2,000个**。保留旧20,000训练窗，仅新增80,000训练窗；验证/测试记录和缓存字节完全沿用旧版本。按既有几何隔离划分，只在train场景中追加。新窗口先生成局部总响应，再运行同样的最低正残差/BFS减法，得到全部有效层掩码和强度；不生成位置教师、候选池或全场概率场。

## 运行

使用同名 `RIND_Phase2_Update.tar.gz` 覆盖代码即可；已有数据、2万版本缓存和最佳权重都要保留，不需要重复传输数据ZIP。

```bash
cd /home/qq/yiyu
tar -xzf RIND_Phase2_Update.tar.gz
cd RIND_Phase2_Training_Minimal
./run_masks_expanded_tmux.sh start all --gpus auto
```

新tmux会话为 `rind_masks_expanded`，旧 `rind_masks_only` 会话独立保留。可以断开SSH/VPN：

```bash
./run_masks_expanded_tmux.sh status
./run_masks_expanded_tmux.sh logs
./run_masks_expanded_tmux.sh attach
./run_masks_expanded_tmux.sh stop
```

准备中断后再次用 `start all` 或 `start prepare` 继续，复用已提交的标签块。准备完成后用 `start train` 恢复训练；训练从上一个完整epoch继续。配置和目标规模改变时使用新的输出目录，避免覆盖已绑定的缓存与断点。

## 怎样扩展

配置：`Phase2_Multi_Source/configs/masks_expanded.json`。

| 配置项 | 当前设置与用途 |
|---|---|
| `data_root` | 原有新旧联合数据目录 |
| `output_root` | `outputs/masks_only_100k`，独立保存扩展结果 |
| `label_preparation.reuse_from` | `outputs/masks_only_v1`，复用原训练前缀和固定评估集 |
| `record_counts.train` | **100000个实际完成的视窗**，不是扫描量，也不是层数 |
| `record_counts.validation/test` | 必须等于原缓存实际数量，默认各2000 |
| `max_views_per_scene` | 最多扫描每场景32个打乱顺序后的合格尺寸原始窗口；优先轮询更多场景 |
| `window_sizes` | 保持16/32/64/128；新发布中使用64/128窗口，尚未加入256/512，避免同时改变输入尺寸和显存需求 |
| `chunk_size` | 每64个新成功窗口提交一个压缩标签块；也在扫描500窗时保存进度 |
| `initial_decomposition_checkpoint` | 原2万版本的 `best.pt`，只载入解耦权重 |
| `learning_rate` | `5e-5`，从已有模型做扩展微调 |
| `batch_size` | 每卡32；目标全局batch256，日志给出实际卡数和累积结果 |
| `evaluate_initial_checkpoint` | 微调前先评估旧模型，保存为epoch0候选；新验证损失更低才替换最佳权重 |
| `evaluate_after_training` | 训练结束自动生成完整测试指标；默认开启 |

以后可以在新实验目录把训练目标设为200000或更大；标签规则和评估集保持一致。`"all"`表示扫描范围内的全部成功窗口，范围仍受 `max_views_per_scene` 与 `window_sizes` 限制，并非无条件读取所有256个窗口。

全暗、全亮、无自由像素、超过层容量或分层失败的窗口不计入目标数量。全亮口径沿用原配置：所有自由像素均为正响应就排除，即使这些正值不相同。没有达到目标会明确报错，不会把扫描量或未完成窗口算作训练样本。

## 缓存与成本

原20,000训练窗直接按连续缓存复制，不重新分层，也不依赖原 `local_labels/*.npz` 仍然存在。新增窗口按块压缩保存，减少逐窗口小文件；最终顺序追加到连续训练缓存。元数据缓存避免每一轮重复过滤/打乱同一场景的256个窗口。标签块记录恢复游标、原始计数和数组校验和，中断最多重做尚未提交的一块。

局部标签准备仍使用CPU完成几何邻接、BFS和减法；开销会随窗口数和分层数量增长，但没有候选位置×像素的昂贵射线渲染。学生训练仍在GPU上做前向、匹配、损失、反向和优化。没有实测10万窗口在H20服务器上的总耗时，不承诺固定加速倍数。

扩展流程检查原数据发布、完整场景划分和分层源代码指纹一致；不同发布不能凭相同scene_id直接拼接标签。如果原缓存位置与默认不同，需要同时修改 `reuse_from` 和初始化权重路径。

## 保存与评估

新目录为 `Phase2_Multi_Source/outputs/masks_only_100k/`：

- `expansion_protocol.json`：扩展参数和原缓存指纹。
- `append_chunks/`：新增局部分层标签块、扫描进度和失败计数。
- `mask_cache/`、`mask_records.json`：最终100000/2000/2000训练数据和索引；索引中的 `expansion` 记录原/新增窗口量、按数据组件/实际源数/有效层数/窗口大小的统计及场景配置数。
- `models/seed_0/best.pt`、`last.pt`：最佳模型和恢复训练断点。
- `training_summary.json`：新训练结果、初始验证指标和最佳验证损失。
- `evaluation/decomposition_metrics.json`：完整测试指标。
- `evaluation/decomposition_metrics.zh-CN.md`：中文指标报告。

若最佳epoch为0，表示扩展训练已执行，但还未在固定验证集上超过旧模型，因此保留旧权重；不能据训练损失下降就声称泛化提升。最终比较使用同一组测试窗口的Dice/IoU、层数、强度和响应重建，测试集不参与最佳模型选择。这是“扩展数据后继续微调”实验，不是从头训练的纯数据量消融。

需要单独重新生成指标时：

```bash
./run_masks_expanded_tmux.sh start evaluate --gpus auto
```

分层标签仍是现有BFS规则的有效贡献，不保证等于真实物理源分解。扩展主要帮助学生学习更多局部响应与分层形态，是否改善未见场景指标仍以固定评估集结果为准。
