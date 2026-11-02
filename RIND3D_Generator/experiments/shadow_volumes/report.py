"""Human-readable report generated from measured, status-aware artifacts."""
import json
from pathlib import Path
from .benchmark import ROOT


def table(header,rows):return '\n'+'\n'.join(['| '+' | '.join(header)+' |','| '+' | '.join(['---']*len(header))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'


def main():
    s=json.loads((ROOT/'summary.json').read_text());full=s['full_table'];small=s['small'];div=s['diversity']['scenes'];lite=s['lite']
    smallrows=[]
    for v,d in zip(small,div):
        c=[next(r for r in v['rows'] if r['method']=='C' and r['resolution']==n)['wall_seconds'] for n in (32,64,128)]
        smallrows.append([v['family'],d['obstacle_count'],','.join(str(o['actual_clipped_facets']) for o in d['objects']),sum(v['shadow']['shadow_plane_counts']),f"{v['geometry_init_seconds']:.2f}",*['%.3f'%x for x in c]])
    fullrows=[]
    for v in full:
        fullrows.append([v['family'],v['method'],v['seconds_budget'],'完成' if v['complete'] else '预算不完整',v['committed_brick_pairs'],f"{v['total_wall_including_setup']:.2f}",f"{v['total_cpu_including_setup']:.2f}",f"{v['peak_rss_bytes']/2**20:.1f}",f"{v['certificate_coverage']*100:.2f}%" if v['complete'] else '—'])
    reuse=[]
    for v in s['reuse']:
        reuse.append([v['family'],v['resolution'],f"{v['cold_total_wall']:.2f}",f"{v['shared_total_wall']:.2f}",f"{100*(1-v['shared_total_wall']/v['cold_total_wall']):.1f}%",f"{v['cold_total_cpu']:.2f}",f"{v['shared_total_cpu']:.2f}"])
    geom=s['geometry_only_reuse'];rot=next(v for v in s['reuse'] if v['resolution']==1024)
    prediction=[]
    for count in (5000,10000):
        v=s['projections'][str(count)];fmt=lambda x:' / '.join(f'{t:.1f}' for t in x)
        prediction.append([count,fmt(v['field_only_cpu_hours_min_median_max']),fmt(v['illustrative_truth_only_Lite_cpu_hours_min_median_max']),fmt([x/1e9 for x in v['compressed_bytes_min_median_max']])])
    cost=[]
    for v in full:
        if not v['complete']:continue
        cost.append([v['family'],v['method'],f"{v['first_brick_including_setup']:.3f}",f"{v['mean_worker_job_seconds']*1000:.3f}",f"{v['file_bytes']:,}",v['canonical_fallback_rays'] if v['method']=='C' else '—'])
    verified_voxels=sum(v['checked_voxels'] for v in s['stratified'])
    text=f'''# RIND3D 凸多面体阴影体、几何多样性与复用实验

2026-10-31。独立原型与有限规模基准，不是 5k–10k 数据集量产。

**当前不建议把这版阴影体实现设为统一主渲染器。** 它能精确而快速地渲染旋转盒和一般凸多面体，但没有胜过同样使用 C++ 遍历的原八角证书控制组。主要限制是投影后保留的冗余平面和逐平面区间计算。已有八角证书的数学机制本来就是正确的，不能把减少 Python 调用的收益算成新阴影体表示的收益。

本轮实现了精确平面投影、区域分类/细分/原 oracle 回退、12 类冻结几何、几何及实体掩码复用、独立参考和完整字段对照。完整回归 **217 项通过**，另有追加源与困难构型检查。原生产代码、原编译扩展、两个相邻工程和旧实验均保持不变；新家族和 Lite 演示都没有发布 READY。

## 数学设计与现有机制审计

原 `native/filter.cpp` 已经针对同一凸障碍物检查八个接收角点；全被遮挡即可证明整个接收盒在其凸阴影内。不同障碍物分别挡住不同角点不能证明整块被挡，本轮专门测试了这种“角点都暗、中心仍亮”的反例。原可见区域证书使用源到区域的保守线段束包围；原快速路径限于正对角变换的盒/椭球，旋转盒和半空间体会回到昂贵 CGAL。

新原型把光源固定后的一维存在条件预先消掉。设 `K={{q:a_i·q≤b_i}}`、`c_i=b_i−a_i·p`。`x∈S_p(K)` 当且仅当存在 λ≥1，使 `a_i·(x−p)≤λc_i`。精确消元得到负 c_i 对应的原前盖平面、零 c_i 对应的过源平面，以及每个 `c_i<0<c_j` 的平面 `(c_j a_i−c_i a_j)·(x−p)≤0`。等号始终保留，不能把有限障碍物前方也误当阴影。

所有 canonical binary64 输入先按精确 dyadic rational 解释，以 Fraction/GMP 计算逆变换、平面和光源符号。区域测试使用逐操作向外舍入的区间。只有所有平面上界均≤0 才证明在阴影内；有平面下界>0 才证明与该阴影分离。其余状态为 UNKNOWN，细分到点后调用原 oracle。闭线段端点、面/边/顶点切触、源在扩展面上、极薄体和 nearly-coplanar 情况均有测试，不使用 epsilon 判断。

快路径还要求完整维度、canonical 已接受的有界凸体、顶层并集，以及世界内部的严格中心见证。这避免把世界外仅贴住 Ω 的面误认为 regularization 后仍存在的障碍物。不能证明该条件的输入使用原 evaluator。几何 GT、CGAL/Arb、函数 `ready_capable=False` 均未改。完整推导和证明义务见 [数学设计](shadow-volume-design.md)。

## 几何多样性

包含随机方向盒、很薄的旋转片、六边棱柱、楔体、不同面数凸体、大小混合、墙体/走廊与遮挡簇。随机方向由 float64 参数生成；真正的几何是这些参数精确解释后的 affine/halfspace 对象，不是理想旋转或近似网格。几何完全独立于 1024³ 网格。

下表每类只有一个冻结几何，不能当作总体分布。面数通过**有理数三平面交点枚举及有效面判定**得到，并包括世界裁剪；因此 poly16 实际是 14 面，poly32 实际是 28 面。阴影列是保留的约束数，包含冗余，并非最小面数。尺寸、长宽比、方向角和源到物体中心距离逐对象记录在 `diversity-statistics.json`；这些中心距离不是净空证书。极薄片的半厚度为 `2^-18`，困难测试另用 `2^-40`。
'''
    text+=table(['家族','障碍物数','实际面数/物体','阴影约束总数','CGAL 初始化 s','C 32³ s','C 64³ s','C 128³ s'],smallrows)
    text+='''
![几何示例](../outputs/shadow-volume-experiment/figures/geometry-diversity.png)

图中线框顶点仅在绘图时舍入，红星表示一份独立单源响应的源。显示网格从未作为真值。设计重点是朝向、尺度、位置关系和遮挡依赖，不是表面纹理复杂度。

## 正确性与验证覆盖

36 个完整小场（12 类 × 32³/64³/128³）共比较 28,704,768 个位置的响应和实体位，差异均为 0。独立实现使用原 canonical 半空间的参数线段裁剪，区间不能决定时用 GMP 精确比较，**不读取新阴影平面**。掩码复用版本另与这 36 个已核对字段逐位一致。追加的六个 geometry/source pair 又完成各三个小尺度和每源十个 1024³ 分层砖的独立核对。

必须区分参考覆盖：上述裁剪实现共享 canonical 平面提取和区间算子，不是对每个体素重跑原 CGAL。主实验另做 288 条 CGAL 随机射线及其实体点检查，困难审计增加 12 条 close-source/极薄/nearly-coplanar 射线，追加源增加 48 条；单元测试还覆盖精确切触、nextafter、源扩展面、未对齐 33³ ROI、错误缓存身份和调用者修改源数组。全部通过，源缓存现有独立只读副本。

原 canonical-only A 在 5 秒小场预算内未能完成全场，对旋转/一般凸体的逐体素 CGAL 全场复算仍有覆盖缺口。**没有把独立裁剪全场通过改称为穷举 CGAL 通过。** 对原快速路径支持的轴对齐盒和走廊，C 与原 B、原证书 native 控制的完整 1024³ 响应及实体掩码各有逐位对照，四组全相同。新家族不因这些有限验证而获得生产 READY。

分层 1024³ 砖测试按空间八分区取样，并加入障碍物邻域，原主源共核对 '''+f'{verified_voxels:,}'+''' 个位置；追加源另核对 60 个砖。它们是 ROI 检查，不是完整场完成数，也不用于从便宜首砖外推整场耗时。

## 性能方法与完整字段结果

机器为 Ryzen 7 5800H、16 逻辑 CPU，Clang 23.1.4 / Release，GMP 精确谓词，禁止 fast-math 与 FMA contraction。每字段两个 worker、队列四；最多两个字段基准同时进行，测试/分析可另有进程。没有锁频、独占机器或清页缓存；这些是单次受控结果，不是统计置信区间。

A 是强制直接使用原 canonical oracle 的原流式流程；B 是原 Oracle/原 Python 遍历及已有凸区域证书；B_native 在相同 native 遍历内直接调用**未改的**原八角过滤器；C 是预构造阴影平面与 native 细分。C_python 小场另隔离同一平面表示在原 Python 遍历中的成本。B 的旋转/一般凸体快速路径不存在，其实际行为就是 A 的回退，不能写成已完成的 B 对照。

下表总 wall/CPU 从已加载的 canonical bundle 开始，计入 CGAL 初始化、原生几何与阴影准备、生成、压缩、写入、封存、独立文件校验和信息索引。原始参数采样、首次 NPZ 归档、公共 Python 启动和跨方法逐位对照另计；因此是场阶段成本，尚不是完整训练数据生产成本。小场 C 预算 60 秒，B/C_python 30 秒，A 5 秒；1024³ 首轮统一 60 秒。单次精确调用不可中断，超时会稍后返回，状态不被当作物理无效或数值失败。A 早版未持久化的详细回退计数保持“未记录”，不写成零。

完成要求始终是 4096 对砖、两文件封存和全索引校验。32 输入约束案例首轮只完成 3996 对，另一次**单独的 90 秒有界重测**完成了全场；先前失败记录保留，未通过加预算宣称算法更快。
'''
    text+=table(['家族','方法','预算 s','状态','砖对','总 wall s','总 CPU s','峰值 MiB','区域证书覆盖'],fullrows)
    text+='''
![性能控制](../outputs/shadow-volume-experiment/figures/performance-controls.png)

轴对齐盒的 C 场阶段约 7.66 秒，B 为 7.94 秒，但 B_native 为 6.11 秒；走廊 C/B 约 16.52/16.51 秒，B_native 为 8.70 秒。小差异无重复统计支持，不能宣称优于 B；相同 native 控制明确更快。减去 Python 遍历后，平面表示没有获得原先预期的收益。

投影保留的平面从 poly8 的 20 增到 poly16 的 72、poly32 的 260，对 128³ 成本也明显增加。这里有输入冗余面和非轮廓的投影组合。最小轮廓表示未实现，不能从本次负结果断言所有阴影体算法都无效。下一步应比较精确冗余消除/轮廓边构造与一般凸体的缓存角点/参数裁剪，而不是先增加线程。

首砖包含准备成本。均值砖时间来自所有 4096 个实际 worker job 的总计时，包含计算和打包，不是完成事件间隔；写入、校验、索引另外包含在总场时间内。文件字节仅为 response+solid，不含元数据。C 普通场没有发生必须回退的射线；困难边界审计则实际调用了原 oracle。
'''
    text+=table(['家族','方法','首砖总 s','均值 job ms','两字段 B','C 射线回退数'],cost)
    text+='''
## 多源复用

复用保持三份**独立单点源响应**，不叠加光源。相同几何共享 canonical Oracle、exact plane/native 几何；每个源重新构造其阴影平面。实体掩码按几何/原源/数值政策的 RINDPack 身份和 SHA 校验后只读复用，缓存有界并加锁。所有源 ID 不同，几何 ID 相同；源记录按 geometry_id 的 hash 分组 train/val/test，禁止按每源 scene_id 拆开。

以下是三源的实际总成本。冷路径每源重建几何；共享路径只建一次并复用实体值。两条路径都含同样的文件完整性校验，跨方法逐位比较单独计时，不混进生产吞吐。初期带额外比较成本的结果也保存在 archive，表中采用明确分离后的 timing 运行。
'''
    text+=table(['几何','分辨率','冷 wall s','共享 wall s','wall 降幅','冷 CPU s','共享 CPU s'],reuse)
    text+=f'''
1024³ 旋转盒仅共享几何、不读取旧实体文件的控制为 {geom['wall_seconds']:.2f} wall s / {geom['cpu_seconds']:.2f} CPU s（三源）。这说明“复用”不是免费收益：读取、解码和重新写入实体也有成本。128³ 几何初始化占比高时，复用收益更明显；不能把该比例直接套到 1024³ 或所有场景。该优化也不依赖阴影体，原 Oracle 本身就可以跨源保留。

当前 RINDPack 头把两个字段都绑定到 geometry+source 身份，故新源仍输出独立 solid 文件。我们复用了正确的实体值和初始化，**没有宣称磁盘上只存一个 mask**，更没有把旧文件硬链接成错误源身份。若要消除这部分磁盘重复，需要另行设计兼容的共享身份约定。

## Lite 兼容与生产阻碍

实机演示已通过：完整 1024³ 响应/实体，256 窗口（85/85/86），每窗一条真源参考、声明上限八条；每窗均用新表示与原 canonical 凸过滤器/fallback 两种实现对整个观测核对，不使用 source-identity shortcut 或仅探针一致。原 pool/IDs/offsets、索引、选择器和 loader 均直接复用。默认样本无候选、真源或见证，显式 private_oracle 才能读参考源。

演示索引重建 {lite['index_seconds']:.3f} 秒、选择 {lite['selection_seconds']:.3f} 秒、整个演示阶段 {lite['total_seconds']:.2f} wall s / {lite['process_seconds']:.2f} CPU s。这只验证一条参考/窗，不代表七条额外候选搜索和严格验证已经测过。异源候选通过当前 adapter 的原 oracle 路径仍可能很慢，也未以 source-only 前向基准掩盖。

checkout 的严格发布器实际上仍要求 4096 窗口/1–64 参考。没有偷偷降低它的计数门禁；Lite 示例为 INCOMPLETE，默认严格 loader 排除，需显式 `require_ready=False` 查看。将来正式发布至少还需显式 Lite profile、geometry-group split 进入生产管道、新家族生成接受规则及更充分 canonical 对照。函数 READY 限制继续关闭。这些是明确阻碍，不是已解决事项。

## 5,000 / 10,000 规模的条件估算

以下“场景”定义为一个独立 geometry/source pair，首先假设每对都重新生成几何。三列为本轮**完成家族**的最小/中位/最大样本，不是总体界限或置信区间。仅场 CPU 包括初始化、写入、校验和索引；Lite 模型再保守加上单盒、一条真源/窗的实测演示 CPU，索引重建因此重复计了一次。把单盒验证成本移用于其他家族只是预算示例，不能作为产量承诺。
'''
    text+=table(['场景数','仅场 CPU h（低/中/高）','示例真源 Lite CPU h（低/中/高）','压缩 GB（低/中/高）'],prediction)
    text+=f'''
10k 的两个字段未压缩 RINDPack 硬上界约 {s['projections']['10000']['pack_raw_upper_bound_two_fields_bytes']/2**40:.2f} TiB，另加索引/几何/标签；实测多面体场压缩很好，不能拿这七个几何保证任意复杂场也只占几 GB。额外最多七条参考的坐标/IDs 本身较小，但搜索和验证 CPU 未测，场景接受/失败重试、真实几何分布、冷磁盘和量产并行效率也未测。CPU h 是累计进程 CPU，不能直接除以 16 当作墙钟承诺。

复用三源时，5k pair 只有约 1667 个独立几何；如果目标是 5k 个独立几何各三个源，就有 15k pair，必须按组成本估算，不能混淆场景多样性和源多样性。源坐标是冻结对照用的三点，不是本轮定义的生产源分布。

## 结论与复现

保留原凸证书策略；native 遍历控制比当前阴影投影更有证据。旋转/一般凸体的可认证快路径和内容寻址几何复用值得继续，但当前完整证据不足以选择这版投影作为 5k–10k 的统一默认，也不应进入量产。下一步优先处理冗余平面、一般凸角点对照、候选验证成本和显式 Lite 发布适配，随后用更多独立几何/源做统计对照。

所有代码在 `experiments/shadow_volumes/`，数学设计、原型配置和 README 已更新。重现时先按 README 构建独立 GMP 扩展，再设置新输出目录，避免覆盖现有实验：

```bash
export RIND_SHADOW_OUTPUT=outputs/shadow-replay-new
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -m experiments.shadow_volumes.benchmark --workers 2
.venv/bin/python -m experiments.shadow_volumes.evaluate stratified --case box
.venv/bin/python -m experiments.shadow_volumes.evaluate full --case box --method C
# 其他几何也必须先通过同一 small/stratified 门禁
.venv/bin/python -m experiments.shadow_volumes.evaluate reuse --case rotated_box --variant timing
```

新实验产物在 `outputs/shadow-volume-experiment/`：[汇总](../outputs/shadow-volume-experiment/summary.json)、[完整字段逐案 CSV](../outputs/shadow-volume-experiment/full-fields.csv)、[精确几何统计](../outputs/shadow-volume-experiment/diversity-statistics.json)、[保全检查](../outputs/shadow-volume-experiment/preservation.json)。217 测试日志与 JUnit、所有失败/部分场、v1/v2 二进制与源码、额外源/困难案例检查都保留。此前 8511 个实验文件逐一校验不变，生产源码/主扩展指纹和两个相邻工程也不变。
'''
    Path('docs/shadow-volume-report.md').write_text(text);print('Wrote measured shadow-volume report')


if __name__=='__main__':main()
