# 8 卡学生训练修复

两次崩溃都属于原生 `Floating-point exception`，不是教师缓存不足，
也不是内存不足。第二次 DDP 运行将问题进一步定位到首个 batch 的底层
GPU 计算路径。

更新后的训练方式是一张 GPU 对应一个进程的
`DistributedDataParallel`（DDP）：

- 冻结配置保留原始 `batch_size=768`，稳定恢复入口默认使用运行时全局
  micro-batch 128，8 卡时每卡 16 条，避免 128×128 视窗在 FP32 下超过
  96 GB；默认累计 6 次梯度后更新，等效全局 batch 仍约为 768；
- 每个进程只在自己的 GPU 上执行完整 forward/backward；
- 每一步各进程获得相同窗口尺寸、候选数和 batch 大小；
- 验证损失在所有 GPU 上按记录数汇总；
- 只有 rank 0 写入 `best.pt`、`last.pt` 和 `training.json`；
- 多卡训练不使用会绕过 DDP `forward` 的候选分块 checkpoint；
- 默认启用稳定 CUDA 模式：FP32/TF32、普通 AdamW、固定 cuDNN 算法；
- 用分块的 FP32 `matmul + softmax` 实现同一注意力公式，避开 PyTorch
  2.4 在目标 H20 环境中发生 SIGFPE 的 fused SDPA backward；
- 首个 batch 会分别打印加载、前向、反向和优化器完成位置，并在每个位置
  同步 GPU，若仍有环境级故障可以精确定位；
- 已生成的教师缓存、数据划分和 80,343 条训练记录不需要重做。

在服务器原目录覆盖解压更新包后，直接运行：

```bash
cd /home/qq/yiyu/RIND_Phase2_Training_Minimal
./resume_phase1_100k.sh train --gpus 8 --global-batch 128 --accumulation 6
```

该入口读取已有实验目录中的冻结协议和教师索引，专门用于代码修复后的
续训。训练会从已有 `last.pt` 继续；若此前在第一个 batch 就崩溃、尚未生成
checkpoint，则从 epoch 1 开始，但不会重新渲染教师。

运行时 micro-batch 和梯度累计次数都会写入 checkpoint 元数据。确认训练稳定
后可以测试 `--global-batch 192 --accumulation 4`；不要直接恢复到 768。
入口同时默认设置
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，减少不同视窗尺寸切换时
的显存碎片。

训练完成后：

```bash
./resume_phase1_100k.sh evaluate --gpus 1
./resume_phase1_100k.sh summarize --gpus 1
```

新实验仍可使用普通入口；它会自动使用 DDP：

```bash
./run_phase1_100k.sh train --gpus 8
```
