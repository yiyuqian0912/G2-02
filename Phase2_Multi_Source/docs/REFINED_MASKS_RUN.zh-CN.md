# 新版完整响应→分层答案：运行说明

2026-10-29。新模型已实现；未进行服务器全量训练，尚不能宣称效果优于旧版。结构详见 [模型说明](DECOMPOSITION_ARCHITECTURE_V2.zh-CN.md)，实现复查见 [二次审核](REFINED_MASKS_AUDIT.zh-CN.md)。

## 直接更新

在Mac本地运行：

```bash
scp -P 10688 ~/Desktop/RIND_Phase2_Update.tar.gz test@58.199.162.221:/home/qq/yiyu/
```

在服务器运行：

```bash
cd /home/qq/yiyu
tar -xzf RIND_Phase2_Update.tar.gz
cd RIND_Phase2_Training_Minimal
./run_refined_masks_tmux.sh start all --gpus auto
./run_refined_masks_tmux.sh status
```

`auto`只使用无计算进程且低占用的空闲卡；也可 `--gpus 8` 使用至多8张空闲卡。`--gpu-ids 0,1` 是用户手动指定，会跳过自动空闲筛选。

后台会依次：验证既有标签缓存→创建只读缓存视图→从头训练新结构→评估最佳验证集checkpoint→生成全部分层指标。tmux启动后可断开SSH/VPN。不会读取旧模型权重强行续训新结构。

## 需要的已有数据

服务器须保留：

```text
Phase2_Multi_Source/outputs/masks_only_100k/
├── mask_records.json
└── mask_cache/
    ├── train_response.f32 / train_obstacle.u8 / train_masks.u8
    ├── validation_response.f32 / validation_obstacle.u8 / validation_masks.u8
    └── test_response.f32 / test_obstacle.u8 / test_masks.u8
```

每行上列3个名称实际是3个独立文件。tar只包含源码，不带缓存、数据或checkpoint；解压不会删除上述原有缓存。训练和评估不需要原始RIND数据或重新渲染教师。

若缓存放在别处，编辑 `configs/masks_boundary_refined.json` 的 `cache_source_root`。实际样本数以缓存index为准，全部保留；不会按label_preparation中的历史参数重新生成样本。默认应为100000/2000/2000。首次绑定后锁定来源；不允许中途悄悄切换缓存。

## 配置与输出

- 配置：`Phase2_Multi_Source/configs/masks_boundary_refined.json`。
- 学生：32槽、隐藏256、4层解码、8头分工4/2/2、3轮共享权重细化；12,325,731参数。
- 输入：完整局部叠加响应、窗口位置与size。教师掩码、强度和残差只参与损失，不进入前向输入。几何输入关闭，Phase1定位关闭。
- 默认100epochs、patience15、每卡batch32、目标全局batch256、lr=1e-4、bf16；实际全局batch按卡数/累积向上取整并打印。
- 结果根目录：`Phase2_Multi_Source/outputs/masks_boundary_refined_v2`，与旧实验分开。
- `models/seed_0/best.pt`：最佳验证checkpoint；`last.pt`：最后完整epoch断点。
- `training_summary.json`、`models/seed_0/training.json`：训练状态/曲线。
- `evaluation/decomposition_metrics.json`、同名`.zh-CN.md`、`windows.jsonl`：完整统计、按有效层数/窗口size分层、严格整窗成功率和场景bootstrap。
- `tmux_runs/latest_run.txt`指向本次后台日志目录；退出码在该目录`exit_code.txt`。

新目标含辅助轮与归一化损失，不能直接把总loss和旧版总loss比较。沿用同定义的层数准确率、每层IoU、强度误差、重建误差、严格整窗成功率判断效果。当前沿用原测试集作历史比较；严格最终确认仍需新几何组独立留出集。

## 查看、恢复、单独评估

```bash
./run_refined_masks_tmux.sh status
./run_refined_masks_tmux.sh logs
./run_refined_masks_tmux.sh attach
```

attach后按`Ctrl+B`再按`D`离开界面，不停止训练。

中断后在没有同名运行任务时：

```bash
./run_refined_masks_tmux.sh start train --gpus auto
```

从最后完整epoch恢复；中断epoch的未保存batch会重跑。若已触发early stopping，再运行不会自动重置patience或从头训练。

单独重算指标：

```bash
./run_refined_masks_tmux.sh start evaluate --gpus auto --bootstrap-repetitions 1000
```

## 验证范围

本地检查涵盖尺寸8/16/32/64/128、幅度尺度等变、重叠掩码、bf16有限梯度、激活检查点、一次最终匹配与多轮辅助监督、缓存复用、两epoch训练/恢复/评估、简例优化和双进程DDP同步。H20 GPU吞吐与完整数据效果须由服务器本次运行测量，不据本地CPU测试推断利用率或90%成功率。
