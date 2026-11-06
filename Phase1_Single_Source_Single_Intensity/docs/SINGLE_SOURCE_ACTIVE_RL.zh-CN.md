# 单源RL动态选窗：首版实验与运行指南

2026-11-06新增：16×16边缘扩展与逐点最小值融合已作为独立v2实验接入，参见 [新版指南](FRONTIER16_MINIMUM_RL.zh-CN.md)。下文继续记录原64块、折扣乘积的v1，不覆盖旧实验。

2026-11-05建立。本实验独立于Phase2分层训练。冻结已有Phase1定位网络，只训练一个PPO选窗策略，检验有限新增像素预算能否使源位置分布向真实坐标收敛。算法参考 [PPO-Clip](https://spinningup.openai.com/en/latest/algorithms/ppo.html)。服务器100k定位器的v1结果已回收，见 [128场景审核](results/rl_single_source_v1_review.zh-CN.md)；未证实PPO优于随机或直接扩窗。

## 奖励与未知场部署

每步奖励是：

\[
 r_t=\Phi(P_{t+1})-\Phi(P_t),\qquad
 \Phi(P)=\tfrac12\log(M_{16}+10^{-8})+\tfrac12\log(M_{32}+10^{-8}).
\]

`M16/M32`分别为真源16/32像素欧氏邻域内的离散概率质量；全部步骤使用同一个全场候选网格。折扣gamma=1，因此累计奖励等于最终与初始定位分数之差。没有概率场熵奖励，也不因变尖就判成功。PPO的`action_entropy_bonus`仅用于动作探索，与位置场熵无关。

真源只在训练环境计算奖励以及离线评价时读取。演员和价值网络仅接收：已经观测的响应、覆盖掩码、当前信念、当前窗口、剩余预算和动作坐标/新像素成本。没有scene_id、真源坐标、完整响应或障碍物几何输入，也不预先读取候选动作的未来响应。

未知场执行`UnknownFieldAgent`即可。它加载策略和对应的冻结Phase1快照，输出窗口请求，接收传感器提供的已观测块；不需要数据集或源真值。代码位于`src/rind_phase1/active_rl/deploy.py`：

```python
from rind_phase1.active_rl.deploy import UnknownFieldAgent

agent = UnknownFieldAgent("best.pt", "phase1_frozen.pt",
                          initial_window=[448, 448, 128], budget=262144,
                          device="cuda")
while (request := agent.request()) is not None:
    # 设备读取request中的新64×64块；返回{tile_id: 二维响应数组}。
    acquired = {tile_id: sensor.read_roi(*roi)
                for tile_id, roi in request["tiles"].items()}
    probability = agent.observe(acquired)
```

这仍假设无噪声、无距离衰减、单源正强度物理模型；正响应转成单位二值掩码。真实带噪传感器需要额外标定，不能把本接口当作已验证的实机方案。

## 观测、预算及去重融合

默认初始窗口128×128，动作是半边长移动、五种锚点扩为两倍，以及停止。所有窗口与固定64×64原子块对齐。这样移动最小为64像素，采集的每个块都是完整的，不需要用填零图冒充Phase1完整窗口输入。

预算按块覆盖并集精确折算为原生新增像素，首次观测和障碍物内部像素也收费。默认训练预算262,144像素（全场25%），允许多步移动和两次扩窗；测试另外比较65,536和131,072预算。零成本重访不产生新证据，独立max_steps防止循环。全亮/全暗不被真值筛选或免费跳过，只对冻结模型采用中性证据，因为它原先主要训练非均匀响应。包含真源的观测不按真值删除，另行保存`source_was_observed`诊断。

每个新64×64块复用Phase1的`encode_observation/score_encoded`，在固定spacing=16的4096点全场网格上查询和softmax。64×64块只评价一次；融合按固定块编号排序，重访和路径顺序不改变同一观测集合的结果。采用折扣log意见池：

\[
 P(s)\propto P_0(s)\prod_{i\in\text{unique tiles}}[P_i(s)/P_0(s)]^{0.5}.
\]

它处理了重复像素，但**尚未完全处理不同块因共享隐藏几何产生的相关性，不是精确多视窗后验**。可能出现错误集中，必须同时检验真源质量、定位距离和可信区覆盖率。固定小块也可能损失跨块长边界特征。本轮将融合固定，单独比较选路策略，不同时训练融合器。

## 数据和权重

默认读取服务器已有的：

- `outputs/experiments/phase1_100k_multiscale_h20_memmap/protocol.json`
- 同目录`response/seed_0/best.pt`
- 原始单源100k RIND数据（需要visibility_bits等文件）。

路径均相对于Phase1文件夹。首次启动从已有best复制模型参数，不重新训练Phase1，也不生成位置教师。新任意视窗仍需原始响应位图；旧的离散教师缓存不能替代原始数据。

RL默认从定位模型原有train/validation/test划分中分别取2,048/128/128个场景，不够会报错，不跨集合借样本。准备时检查所有提供的Phase1 split中相同连续障碍物参数集合是否跨集；测试bootstrap以这种几何组为单位。若Phase1曾追加Phase2残差层，本程序记录该事实，但不能独立确认那些外部几何与本次集合绝无重合，需要额外审计。

训练起点每次从对齐网格均匀重采样，独立于真源；验证/测试起点固定在协议清单。冻结源快照、样本名单、原始manifest和权重SHA存入新输出，不覆盖旧实验。策略best仅按验证集最终Phi选择，锁定测试集不参与选择。

## 服务器运行：全部使用tmux

更新同名`RIND_Phase2_Update.tar.gz`并解压后，在包根目录运行：

```bash
./run_single_source_rl_tmux.sh start all \
  --gpus auto \
  --phase1-run outputs/experiments/phase1_100k_multiscale_h20_memmap \
  --data-root data/raw/phase1-single-source-100k

./run_single_source_rl_tmux.sh status
./run_single_source_rl_tmux.sh logs
```

默认会自动使用近期低利用率且至少8 GiB空闲的GPU；一进程一卡，各rank分担训练场景并用DDP同步策略。ROI位图解码、Phase1批量查询、融合、策略和PPO计算在所选GPU上。CPU保留路径/覆盖网格、文件读取、元数据与输出工作；未在本机H20测量吞吐。大部分开销来自冻结Phase1查询，有块级缓存，并为均匀块避免模型调用。不会占用正在高负载计算的卡或终止其他进程。

已有运行无需重复参数即可继续；恢复权重/优化器/更新号，未保存的部分rollout与环境状态会重新开始，不承诺逐动作精确续跑。可以增加总更新数：

```bash
./run_single_source_rl_tmux.sh start train --gpus auto --updates 500
./run_single_source_rl_tmux.sh start evaluate --gpus auto
```

若要做另一组配置，应换`--run-name`；之后status/evaluate都带同一个名字。CPU可显式`--device cpu`，不会自动从CUDA退到CPU。

参数可在首次启动前修改`configs/active_rl.json`，或通过`--set KEY=JSON`覆盖，例如：

```bash
./run_single_source_rl_tmux.sh start all --gpus auto \
  --run-name single_source_rl_trial \
  --scene-counts 2048 128 128 --updates 200 --num-envs 32 \
  --set locator_batch_size=32 --set fusion_discount=0.5
```

## 输出和评价

默认输出为`Phase1_Single_Source_Single_Intensity/outputs/active_rl/single_source_rl_v1/`：

| 文件 | 用途 |
|---|---|
| `protocol.json` | 场景、固定评估起点、参数、权重与数据身份 |
| `phase1_frozen.pt` | 只包含冻结定位模型的必要状态 |
| `initial.pt`、`best.pt`、`last.pt` | 初始策略、验证最佳策略、PPO恢复断点 |
| `training.json` | 更新曲线、奖励、动作熵、KL和时间 |
| `evaluation/metrics.json` | 各预算的四策略指标与按几何组配对区间；含RL update和快照SHA |
| `evaluation/episodes.json` | 每个场景/策略/预算的原始曲线和实际像素支出 |
| `figures/*.png/json/npz` | 采集路径、初始/最终概率场、真源质量曲线和逐步概率数组 |
| `index.html` | 打开图片与报告的入口 |
| `tmux_runs/` | 命令、日志和退出码 |

对比策略包括PPO、随机、优先扩窗、训练前初始策略。它们使用相同初始窗、预算、动作和冻结定位/融合模型。提前停止时保留最后分布，明确报告实际支出，不把未用像素写成已观测。重点看M16/M32、MAP距离、Recall@32及90%区域面积/真源覆盖率，不能单看熵。

## 本地试跑的真实结果与限制

已使用本地旧`consistent_10000`小定位模型（hidden=64，best epoch16）跑通64/16/16场景、40次PPO更新、三档预算及四策略评估。它与服务器20M的100k模型不是同一定位器。262,144预算下，PPO的M32约0.0435，随机约0.0448；MAP距离约374与352像素，配对区间未显示稳定改善。PPO的90%区域真源覆盖率约68.8%，说明融合后的集中度还不够可靠。

因此这次本地结果只支持“流程可以运行”，**不支持RL已经优于基线，也不支持已收敛到真源**。后续服务器100k定位器的128场景测试也未证实PPO优于基线，且发现固定扩窗偏好；完整数字及探测限制见上述审核。新增16块最小值v2另设实验名，不覆盖这些负结果。另已跑通双进程Gloo真实数据训练/评估，以及去重、预算、奖励、终止和真值隔离测试。v2的H20吞吐与定位收益仍需服务器验证。
