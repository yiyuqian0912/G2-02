# Integration 提交范围

本版本合并数据接口、物理监督、条件能量训练、正式评价与测试集可视化。主入口是 `run_training.sh` 和 `view_results.sh`，默认模板为 `configs/experiment_consistent.json`。

## 包含

- `src/rind_phase1/`、`part_e/`：统一后的主流程及评价代码。
- `student_baseline/`：legacy 适配器仍引用，暂不删除。
- `configs/experiment_consistent.json`：v2 默认模板。
- `configs/experiment_round1.json`：旧协议回归测试模板。
- `configs/experiment_train5000.json`：已讨论的旧基线参数，便于追溯。
- `configs/splits/phase1_fixed_v2.json`：必要的固定场景池/窗口清单，不是原始数据。
- 脚本、测试、方法/接口/使用文档、依赖清单和 RIND vendor 来源信息。

## 本地保留，不提交

- `.venv/`、缓存、数据 ZIP、`data/` 数据内容。
- `outputs/` 下的教师缓存、权重、图表、日志和统计报告；仅保留目录占位文件。
- `configs/experiment_consistent_10000.json` 等个人运行配置。新运行通过启动器生成。
- `archive/` 历史草稿和本次本地一致性复核记录。

历史草稿和原来误跟踪的 pilot 输出仅从 Git 索引移除，本机副本保留。它们仍在旧提交历史中；本次不重写历史。

## 组员拉取后

```bash
uv sync --locked --extra train --extra browser --extra diagnostics
# 先运行代码测试；缺数据/旧 checkpoint 的集成检查会跳过。
PYTHONPATH=.:src NUMBA_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests -q
PYTHONPATH=.:src .venv/bin/python -m unittest part_e.test_part_e part_e.test_integration -q
# 安装数据后，再跑完整检查和实验。
./scripts/install_data.sh
./run_training.sh --run-name my_first_run --check
```

未安装数据时的跳过不等于完成端到端验证。浏览器依赖本地训练产物，干净仓库没有模型成绩或预置 checkpoint。旧结果不能代表 v2 模型效果。

本轮只准备提交范围，不自动 commit 或 push。请从 Git 仓库根目录检查暂存差异后提交；本项目位于父仓库 `G2-02/` 内。
