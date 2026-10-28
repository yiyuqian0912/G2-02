# 新版多源数据安装与安全拼接

本次新文件：`RIND-random-v3-100000scenes-multi-unit-256views.zip`。

- 压缩包：1,669,364,063 bytes，约 1.55 GiB。
- SHA-256：`44015d37e6abb2093228380c237beeea1e71266b87640a5e8cdc05fe4a36d174`。
- 解压数据约 62.4 GiB，安装会先检查空间并逐文件验证发布包 SHA256SUMS。
- 新数据为 100,000 个场景配置，每场景 256 个随机、有信息、无源视窗，大小为 64/128/256/512。
- 下载文件的实际源强度全部为 1。安装器会先完整保留该原始发布，再以固定 seed=20261028 为每个有效源独立采样 (0,1) 强度，建立随机强度派生版本。正式拼接和训练使用派生版本。
- 派生响应直接由保存的每源二值可见性计算 `sum(a_j * V_j)`，不修改可见性或障碍物。原始单位强度参考解可能不再成立，派生版本停用这些参考解。新配置的教师位置概率必须重新计算。

实测新文件前 10,000 场景的 obstacle_counts、obstacle_types、obstacle_params 与旧数据逐元素完全相同。直接重新随机划分 110,000 个 scene_id 会造成几何泄漏。本模块保留旧场景 ID 0–9999，新场景配置从 10000 开始，并将相同连续障碍物几何绑定到相同 train/validation/test split。旧分配来自旧教师 `protocol.json` 中完整的 `scene_splits`。新增几何组采用固定种子的 80/10/10 哈希划分。

拼接后有 110,000 个场景配置，但不能将它们称为 110,000 个独立几何；最终独立几何组数量见 `append_report.json`。重复几何的派生场景使用新采样强度和随机视窗，必须重新生成其教师，不能套用旧缓存。

## 服务器运行

拼接会读取旧教师目录的 `protocol.json`，只使用其中完整的 `scene_splits`；不需要复制教师概率缓存。复现已有实验时应保留该划分元数据。全新安装如果没有旧协议，可以分别运行本模块的 `install`、`randomize` 子命令，将新实验的 `data_root` 指向独立的 `phase2-random-strength-100k`；窗口标签会按几何哈希划分。该独立实验应使用新输出目录，不与旧联合数据实验混用缓存或指标。

把数据 ZIP 单独传到最小包根目录。代码 tar 不含这份 1.55 GiB 数据。

```bash
cd /home/qq/yiyu/RIND_Phase2_Training_Minimal
./Phase2_Multi_Source/dataset_update/install_new_dataset.sh
```

如 ZIP 放在别处，可显式给出路径和旧教师目录：

```bash
./Phase2_Multi_Source/dataset_update/install_new_dataset.sh \
  /home/qq/yiyu/RIND-random-v3-100000scenes-multi-unit-256views.zip \
  /home/qq/yiyu/RIND_Phase2_Training_Minimal/Phase2_Multi_Source/outputs/full_training_h20_v2
```

生成：

```text
Phase2_Multi_Source/data/raw/
├── phase2-multi-source/          原数据，不改写
├── phase2-unit-random-100k/       新数据，逐文件校验解压
├── phase2-random-strength-100k/  (0,1) 强度派生版本，仅强度数组和元数据
└── phase2-merged-110k/            仅元数据，不重复复制约 62.4 GiB 数组
    ├── manifest.json
    ├── scene_splits.json
    ├── geometry_groups.npy
    └── append_report.json
```

训练新流程通过 `from dataset_update import open_dataset` 读取联合目录。原来的严格 `Phase2Dataset` 仍用于旧流程；新适配器同时处理随机视窗布局和可复现的强度派生版本。

## 物理和实验差异

正式训练全部使用随机 (0,1) 源强度。新视窗仍沿用原 ZIP 的随机选择结果，不能将其称为重新按随机强度生成的四分树数据。教师完成和单源物理兼容仍分别检查，报告应按视窗抽样分布分层。

全亮筛选沿用当前训练协议：有自由零响应与正响应的窗口进入训练；新数据自己的“informative”规则只要求至少两种总响应值，因此其中部分全正但不同值窗口仍会被训练协议排除。这是选择机制差异，统计时应报告。

## 校验

```bash
cd Phase2_Multi_Source
python -m unittest dataset_update.test_dataset_union -v
```

测试覆盖旧 ID 保留、随机强度正确叠加读取、原始单位数组不变、旧参考解停用、重复几何同 split、旧 split 既有泄漏拒绝、ZIP 逐文件校验、保留原 ZIP 和幂等拼接。
