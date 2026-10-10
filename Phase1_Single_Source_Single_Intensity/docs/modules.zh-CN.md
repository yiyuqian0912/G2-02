# 模块职责与交付目标

认领按下面模块即可，不需要再建立一套并行接口。负责人自选；交付应是可运行结果、配置和相应检查，不能只有函数占位。

| 模块/文件 | 要完成什么 | 交付与验收目标 |
|---|---|---|
| `src/rind_phase1/data.py` | 读取单源发布包、按 ID 取原始窗口、暴露教师重渲染接口 | 能定位任意 scene/view；响应、掩码、window 一致；按 size 组批 |
| `src/rind_phase1/install_data.py` | 校验和安装数据，记录来源 | 正确拒绝不符合单源约定的数据；已有数据不被覆盖 |
| `src/rind_phase1/browser.py` | 接入原始数据浏览器 | 可检查场景、四分树窗口和响应；不是训练预测页面 |
| `src/rind_phase1/physics.py` | 给出边界加权响应误差和可选对称边界距离 | 权重只由真实观测确定；无边界及无效候选处理有明确定义；统计真实渲染次数 |
| `src/rind_phase1/search.py` | 定义源域和可重复的均匀候选网格 | 候选顺序、spacing、窗口排除规则明确；未完成搜索不能输出完整教师 |
| `src/rind_phase1/teacher.py` | 将完整物理代价转为软分布并保存 | 保存坐标、valid、cost、q、配置及来源；不注入真实源、不乘面积 |
| `src/rind_phase1/interfaces.py` | 拼接教师与观测并阻止错配 | IDs、window、候选顺序和教师分布检查通过；不私自转换坐标 |
| `src/rind_phase1/protocol.py` | 固定场景/视窗选择，检查新配置 | 扩大训练集不改变留出集合；拒绝不匹配的 Fourier 频率/候选间距 |
| `src/rind_phase1/model.py` | 输出观测与候选位置的条件能量 | 只读取声明输入；保留空间位置信息；支持任意连续查询坐标 |
| `src/rind_phase1/train.py` | 按 size 训练、延迟读取记录、加载 checkpoint | 不一次载入所有响应；候选分块不改变全局 softmax；支持完整 batch 训练 |
| `src/rind_phase1/predict.py` | 在教师原始候选集合上统一推理 | 分块前后结果一致；输出归一化范围、能量和 student_prob |
| `src/rind_phase1/sampling.py` | 密集源空间查询与频率检查 | 每个显示点都有真实能量计算；全局归一化一次，不插值概率、不乘面积 |
| `src/rind_phase1/experiments.py` | 串起冻结协议、教师、训练、测试、汇总和搜索对照 | config/数据/代码改变时拒绝混入旧实验；保存可恢复状态与正式报告 |
| `src/rind_phase1/results_browser.py` | 读取已有测试清单，提供异步推理和导出 | 用户可选 test scene/view；旧新 checkpoint 均可读；输入错误不启动无效任务 |
| `src/rind_phase1/templates/results.html` | 展示局部响应、学生热图、教师采样 | 可筛选、切换共同候选/密集模式、悬停读数、放大和下载 PNG/NPZ |
| `src/rind_phase1/diagnostics.py` | 教师逆空间诊断与静态图 | 区分候选代价、教师概率及参考解；不把示意图当稠密物理验证 |
| `src/rind_phase1/checks.py` | 数据/教师阶段自检 | 独立检查窗口、参考解重渲染和记录约定 |
| `src/rind_phase1/evaluate.py` | 保留通用检查和早期诊断 API | 使用统一概率范围；正式实验指标以 `part_e/evaluate.py` 为准 |
| `part_e/checks.py` | 检查教师、预测及场景划分的交接 | 验证声明的 support，而不是强制隐藏几何辅助 |
| `part_e/evaluate.py` | 正式分布、兼容性、模糊性及泛化评价 | 与 uniform baseline 对照；报告障碍物内质量；按 scene 和 size 汇总 |
| `part_e/adaptive.py` | 在同一最终网格上做粗到细保留对照 | 记录预算和遗漏；只有完整保留块可形成条件教师；不把搜索轨迹当最终分布 |
| `part_e/pipeline.py`、`record_io.py`、`predict.py` | 读取各组交付并运行联调 | 统一记录可直接走通；失败或缺少学生结果要明确报告 |
| `part_e/adapters.py` | 显式读取旧模型接口 | 兼容只在指定 legacy 路径启用，不混淆坐标/输入 |
| `part_e/physical_cost.py` | 复用正式物理代价 | 不维护另一套损失定义 |
| `part_e/run_comparisons.py` | 运行独立 pilot 比较 | 保留 pilot 配置和数据范围；不作为新正式训练入口 |
| `scripts/launch_training.py`、`run_training.sh` | 提供可自定义参数的一键训练入口 | 打印观测数量、保存配置、拒绝覆盖不同实验 |
| `scripts/report_experiment.py` | 从已完成报告画学习曲线和 size 对照图 | 不重新选模型或改测试结果；正确标注概率范围和代价定义 |
| `scripts/dense_probability.py` | 保留静态逐像素导出入口 | 输出真实模型查询的 PNG/NPZ/独立 HTML；交互选择优先用结果浏览器 |
| `view_results.sh` | 在项目环境启动结果页面 | 终端一条命令启动本机 8767 服务；Ctrl+C 停止 |
| `configs/experiment_consistent.json` | 新实验默认参数 | 与上面输入、域及采样约定一致；无需改 Python 就能配置 |
| `configs/splits/phase1_fixed_v2.json` | 全部场景的固定池及已选窗口 | 与数据发布包匹配；训练/验证/测试永久互斥 |
| `configs/experiment_round1.json`、`experiment_train5000.json` | 保留旧实验参数 | 明确是 legacy 协议，不能据此代表 v2 效果 |
| `configs/phase1.json` | 低层无权重教师 CLI 默认参数与项目约定索引 | 保持教师基线命令可运行；正式训练使用 active_experiment 指向的配置 |
| `tests/`、`part_e/test_*.py` | 检查物理、接口、训练恢复及可视化数值 | 有意义的回归测试可重复运行；不以截图替代数值检查 |
| `docs/` | 定义方法、输入含义、实验命令和解释边界 | README 中的链接存在，术语与代码一致 |
| `vendor/rind-dataset/` | 固定版本的数据读取/几何渲染依赖 | 保留来源快照，避免为学生需求悄悄改物理规律 |
| `student_baseline/` | 保留旧模型兼容实现 | 用作历史参考，不作为新实验权威实现 |
| `data/raw/` | 安装后的数据 | 不纳入模型特征、不覆盖或重生成已有发布包 |
| `outputs/experiments/<name>/` | 单次冻结实验的全部生成物 | protocol、教师、checkpoint、报告、预测一起保留 |

详细数学和字段约束分别在 `docs/method.md` 和 `docs/interfaces.md`（英文）。每位组员认领时说明文件、输入/输出、验收结果；首轮交付日期按组内约定的周四安排，不在代码中硬编码过期日期。
