# Part E：评价与端到端联调

此目录是独立的 E 实现，不覆盖 src/rind_phase1 中已有文件。

## 查看提交的结果

- outputs/figures/part_e_pilot/：3 张主要图片。
- outputs/reports/part_e_pilot/RESULTS.zh-CN.md：真实数据 pilot 报告。
- 同目录 JSON：测试、场景划分、边界消融、搜索、窗口尺寸和端到端检查结果。

这次使用前 32 个真实场景，24 train / 4 validation / 4 test。模型仅在 L=16 训练；六尺寸结果是外推诊断，粗网格和单随机种子成绩不能代表最终性能。

## 最小验证

在 Phase1_Single_Source_Single_Intensity 目录中运行：

```bash
python -m pip install numpy
python -m unittest part_e.test_part_e part_e.test_integration -v
```

已有 records 的评价入口：

```bash
python -m part_e.evaluate --manifest PATH_TO_MANIFEST.json --output outputs/reports/new_run
python -m part_e.checks --teacher PATH_TO_TEACHER.npz --student PATH_TO_STUDENT.npz
```

manifest 的 observations 列表每条含 teacher、student 和可选 observation 的 NPZ 相对路径，另需 compatibility_threshold；测试集验证还需 splits 和实际 trained_scene_ids。

## 真实联调与对比

主分支的 B/C/D 尚未全部合入。pipeline_config.json 中 teacher_source 指向 C 完整 Phase I 目录，student_source 指向 D 的 student_baseline，data_root 指向已安装数据；使用真实模型时填写 checkpoint。这些产物不在本上传包内，必须先填自己的实际路径。路径相对配置文件。

```bash
python -m part_e.checks --pipeline-config part_e/pipeline_config.json --output outputs/reports/end_to_end
python -m part_e.run_comparisons --config part_e/pipeline_config.json --output outputs/reports/comparisons
```

训练模型预测需要 PyTorch，真实 renderer 推荐 Numba；原实验完整环境版本见 requirements-pilot.lock。本精简包保留 E 评价和联调入口；数据提取、训练复现和旧图表重建脚本仍在完整结果包中。
