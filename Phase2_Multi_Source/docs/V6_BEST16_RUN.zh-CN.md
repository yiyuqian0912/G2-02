# 从已完成的 v5 最佳权重继续：隐式残差训练

本轮从 `masks_disentangled_v5_complex245746` 的 **best epoch 16** 初始化，不从随机权重开始。保留旧实验和缓存，新结果写入 `outputs/masks_implicit_v6_best16`。v6 是新结构的微调实验，目前尚无本轮训练结果，不能预先承诺指标提高。

二次正确性审核、有效层匹配加速及指标修正见 [运行审核](V6_RUNTIME_AUDIT.zh-CN.md)。

## 继承什么、学习什么

- 继承 v5 编码器、层数预测、槽位读取和共用掩码解码参数；只允许新增的 `state_encoder`、`context_feature`、`residual_pixels` 参数缺失。其他参数不匹配直接报错。
- 原强度头保留并冻结，用于兼容原计数特征，不以教师强度训练新过程。
- 学生只看到原始局部响应及窗口位置、size；教师最终无序二值层掩码和有效层数仅参与监督。不给教师剥离顺序、中间残差或真实光源信息。
- 学生自行决定层数并迭代提取；训练目标只有等权的层数与层形状监督。原始响应守恒是最终输出的验收条件，不添加重建损失。
- 迁移的是模型参数，优化器重新建立以适应新增模块。基础学习率 `1e-5`，新增模块为其 3 倍；2 轮预热后余弦衰减。再次启动同一个新实验则恢复它自己的模型和优化器。

完整复用既有 **345,746 / 12,500 / 12,500** 个训练／验证／测试窗口。实际数量以源实验 `mask_records.json` 为准；启动日志和配置保存完整数量，不筛减、不重复凑数、不重新渲染教师。缓存训练不需要原始场景数据，但必须保留源实验的 `mask_records.json`、`mask_cache/` 和最佳权重。

## 上传和启动

Mac 本地执行：

```bash
scp -P 10688 ~/Desktop/RIND_Phase2_Update.tar.gz test@58.199.162.221:/home/qq/yiyu/
```

服务器执行：

```bash
cd /home/qq/yiyu
tar -xzf RIND_Phase2_Update.tar.gz
cd RIND_Phase2_Training_Minimal
unset RIND_MASKS_CONFIG RIND_DECOMPOSITION_CONFIG RIND_TMUX_SESSION

./run_implicit_masks_tmux.sh start all \
  --source-root outputs/masks_disentangled_v5_complex245746 \
  --expected-source-epoch 16 \
  --run-name masks_implicit_v6_best16 \
  --session rind_v6_best16 \
  --gpus auto --auto-batch --gpu-reserve-gib 12
```

`all` 复用完整缓存，训练后运行正式指标与同测试集旧权重对比。启动前会冻结源 `best.pt`，校验它确实是第 16 轮；若源权重或既有同名新实验冻结的轮次不同，会停止并给出原因，避免误用之前第 14 轮的启动快照。不要通过更改检查绕过它。

原来的 v5 `best.pt`、`last.pt` 和所有输出都保留。上传包只包含代码和文档，不包含缓存、原始数据或权重，不会通过解压删除服务器旧输出。

## 查看、继续与单独评估

```bash
./run_implicit_masks_tmux.sh status --session rind_v6_best16
./run_implicit_masks_tmux.sh logs --session rind_v6_best16
tmux attach -t rind_v6_best16
```

退出终端或 VPN 后 tmux 继续运行；从会话脱离用 `Ctrl-b` 再按 `d`。终止后重新执行完整启动命令，会从新实验最近保存的 epoch 继续；尚未保存的半轮需要重跑。

训练结束后单独重新生成指标：

```bash
./run_implicit_masks_tmux.sh start evaluate --session rind_v6_best16 --gpus auto
```

结果位于 `Phase2_Multi_Source/outputs/masks_implicit_v6_best16/`：模型在 `models/seed_0/`，评估在 `evaluation/`，后台日志和退出码在 `tmux_runs/`，冻结的旧最佳模型在 `warm_start/`。主要看精确层数准确率和实际启用层的 IoU≥0.90 通过率；未完成窗口仍计入分母。正式新旧对比重新使用相同的形状匹配规则，不直接混用历史的强度参与匹配指标。

## 本轮加速

- 自动寻找符合空闲显存及近期利用率条件的 GPU，使用 DDP；不固定卡号，不停止其他用户进程。可用卡不足时明确报告。
- 压缩掩码在 GPU 解码、填充和组批。显存足够时缓存常驻 GPU；否则流式传输压缩字节，仍在 GPU 解码。CPU 保留文件读取、元数据管理和集合匹配等必要工作。
- 默认每卡缓存预算不超过当前空闲显存的 55%，且至少预留 32 GiB；自动 microbatch 探测在缓存占用之后运行，为模型、优化器和最坏 32 步展开保留空间。
- 仅长展开（≥9 层）启用激活检查点，避免普通短窗口反复重算；BF16、融合优化器继续使用。自动批量调整维持同卡数下原有有效全局 batch，避免单纯增大 batch 改变优化步骤数。目标为256；实际值按卡数和整数累积向上取整并写入日志，例如8卡为256、6卡为384。
- 每批训练及例行模型选择不运行没有梯度、也不参与这两项损失的最终强度可行性求解。**正式评估和推断仍执行完整求解及非负／守恒验收**；未验收的训练输出不能被当作成功的完整分解。
- 正式评估复用同一次集合匹配结果，避免重复计算。
- 加权像素对比改为显式bmm，避免三输入einsum形成巨大的四维外积；自动批量同时覆盖32层检查点路径及8层无检查点路径。共享卡建议使用上面的12 GiB显存余量。

启动会用少量真实缓存窗口比较 GPU 解码与原 CPU 路径是否逐项一致，检查成功后才开始优化。缓存模式与检查数量写入 `runtime/gpu_inputs/`，自动批量探测结果写入运行时报告。CPU 本地测试验证了输入、损失、梯度和训练语义一致性；H20 实际吞吐、CUDA 内核和 NCCL 仍需服务器运行确认，不能保证每张卡持续 100% 利用率。
