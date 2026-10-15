# 第二轮几何实验：v5 诊断与 v6 残差几何模型

## v5 已完成结论

v5 将最多 48 个障碍物 primitive 输入几何 Transformer，并由候选位置通过一次 cross-attention 读取几何。完整测试中，v5 test CE 为 8.827647，略差于 v4 的 8.813449；Precision@90、Dice@90 和物理无效候选质量也没有改善。

几何置乱诊断固定局部响应、视窗和候选位置，只交换场景几何：正确几何 CE=8.949366，置乱几何 CE=8.949612，差值仅 +0.000246；120 个样本中正确几何更好的比例为 47.5%。置乱使概率场 TV 平均变化 0.1035，因此模型会响应几何，但这种响应没有与正确场景建立稳定关系。

v5 配置和 checkpoint 保留用于复现，不再作为推荐训练方案。

## v6 方法

v6 使用：

\[
E_{\mathrm{geo}}(s)=E_{\mathrm{v4}}(s)+\Delta E_{\mathrm{geometry}}(s).
\]

- 从 v4 最佳 checkpoint 加载所有共享参数；
- 几何修正头最后一层初始化为零，epoch 0 与 v4 完全一致；
- epoch 0 可以成为最终最佳 checkpoint；
- 前 8 个 epoch 冻结 v4；
- 之后 v4 学习率 `1e-4`，几何分支学习率 `1e-3`；
- geometry transformer 为 1 层、dropout 0.1；候选到几何使用 4-head cross-attention，局部 v4 注意力保持 1 head；
- batch 384、180 epochs，在六卡上约提供 20,000 次优化更新。

配置：`configs/layer_student_geometry_residual.json`。

## 运行

```bash
cd /home/qq/yiyu/RIND_Phase2_Training_Minimal

./run_training.sh Phase2_Multi_Source/configs/layer_student_geometry_residual.json index
./run_training.sh Phase2_Multi_Source/configs/layer_student_geometry_residual.json train --gpus 6
./run_training.sh Phase2_Multi_Source/configs/layer_student_geometry_residual.json metrics \
  --seed 0 --device cuda --bootstrap-repetitions 1000
```

几何诊断：

```bash
cd Phase2_Multi_Source
../.venv/bin/python research/layered_residual/audit_geometry_conditioning.py \
  --config configs/layer_student_geometry_residual.json \
  --device cuda --max-layers 512 --batch-size 16
```

## 验收

1. epoch 0 验证 CE 与 v4 一致；
2. 最佳验证 CE 不差于 epoch 0；
3. 置乱几何后的 CE 稳定升高；
4. Precision@90、Dice@90 或 invalid-candidate mass 改善，Recall@90 不明显下降。

完整实验记录见 `RIND_Phase2_v4_完整实验设计与结果.md` 的第 17–20 节。

## 几何不可见的多视窗融合

实际部署继续使用 v4 单层定位器。不同视窗中疑似同一贡献的概率场先通过 Bhattacharyya overlap、去均匀背景的 centered cosine shape similarity、相对均匀场的信息量以及强度软一致性进行保守关联。均匀或近均匀概率场不能仅靠高 overlap 或同向数值扰动建立关联。轨迹合并使用 complete-linkage，并禁止一个轨迹包含同一视窗的两个层。

一般关联阈值为 0.55；连续有效强度在相对容差 `1e-6` 内完全相等时阈值为 0.15，但仍需通过位置场形状和信息量检查。本地 100 窗口恢复包中，10 个重复场景形成 22 个窗口和 33 个跨视窗层对。以精确强度相等作为不完美代理标签时，默认规则接受 5/11 个代理正对且没有接受代理负对，对应代理 precision 100%、recall 45.5%。这是小规模开发诊断，不是真实源身份的正式测试结果。

融合采用相关性折扣 log opinion pool。默认 `redundancy_discount=0.5`；它介于不增加证据和完整独立概率乘积之间。由于重叠视窗共享未知障碍几何，融合结果是组合证据分布，只有在条件独立假设成立时才能解释为严格联合后验。
