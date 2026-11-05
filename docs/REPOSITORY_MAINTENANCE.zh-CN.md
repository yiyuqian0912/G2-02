# 仓库整理约定

更新：2026-11-04。

GitHub保存正式源码、配置、测试、团队文献、原proposal、方法与紧凑结果证据。根README提供当前推荐入口；完整研究演进集中在 [研究链路报告](RIND_RESEARCH_CHAIN.zh-CN.md)，实验状态由 [PROJECT_STATUS](PROJECT_STATUS.zh-CN.md)维护，Phase2的历史流程由 [文档索引](../Phase2_Multi_Source/docs/README.zh-CN.md)导航。

## 保留什么

- 当前v6和原v5、旧mask训练/诊断、Phase1定位及已有可视化。
- 历史配置与模型实现：旧checkpoint仍需要对应结构，不能仅因文件名旧就删除。
- Phase1的 `part_e/`、`student_baseline/` 和Phase2物理教师/联合候选模块：仍有实际依赖与复现实验用途。
- 已运行实验的原始数据、模型、缓存、逐窗统计及虚拟环境：保留在本地/服务器，由Git忽略规则排除。

## 清理什么

- 源码目录中的 `__pycache__`、`.pytest_cache` 与 `.DS_Store` 等可重新生成的状态。
- README中相互冲突、过期的“当前下一步”与重复实验描述；历史结论在对应文档/报告保留。
- 重复导出源码树、临时服务器副本、重复传输包不作为正式源码提交。

本次不通过 `git clean -fdx` 清理项目，也不把数据、权重或研究结果当垃圾删除。已有 `.local/cleanup_20261028/` 是历史保留区，不进入源码包。

## 后续规则

新实验建立新配置/输出目录，不复制整个训练源码。已启动实验保持配置和缓存签名，用checkpoint恢复；改变训练定义时建立新实验。原权重比较前冻结带SHA的快照，不覆盖旧best/last。

服务器包从正式目录通过 `tools/package_update.py` 构建，固定名为 `RIND_Phase2_Update.tar.gz`，固定解压前缀 `RIND_Phase2_Training_Minimal`。白名单仅包含源码、配置、测试、文档，不含数据、环境、缓存或权重。不维护第二套更新源码。
