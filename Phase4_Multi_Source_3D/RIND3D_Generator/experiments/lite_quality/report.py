"""Write measured result tables; incomplete pilots never count as successes."""
import json,math
from pathlib import Path
from collections import Counter
from fractions import Fraction as Q
import numpy as np
from .audit import OUT

def read(p):return json.loads(Path(p).read_text())
def quantiles(values):
    a=np.asarray(values,dtype=float)
    return dict(zip(('p50','p95','p99','max'),map(float,[*np.percentile(a,[50,95,99]),a.max()])))
def table(headers,rows):return '\n'.join(['| '+' | '.join(map(str,headers))+' |','| '+' | '.join('---' for _ in headers)+' |',*['| '+' | '.join(map(str,row))+' |' for row in rows]])
def fmt(x):return f'{x:.3f}'

def write():
    natural=read(OUT/'frozen-distribution-audit.json');strat=read(OUT/'sampling/summary.json')['stratified'];mixed=read(OUT/'sampling-mixed-current/summary.json')['mixed'];baseline=read('outputs/lite-v01-production/pilot-100/report.json')
    facts={'frozen_bins':natural['bins'],'stratified_bins':strat['bins'],'mixed_bins':mixed['bins'],'studies':{}}
    text=['# Lite 观察质量、验证、逆源几何与持续生产实验','',
          '原 100 个 READY_LITE 场景和历史输出保持冻结；没有启动 5k–10k 量产。新产物位于 `outputs/lite-v02-research`。Lite v0.1 的完整 1024³ 双字段、256 窗口、85/85/86 尺寸配额、无损连续几何、窗外真源、亮暗自由体素、1–8 严格参考源、净空和所有发布门禁不变。原 4096 窗口 profile 未改。',
          '', '数学推导、算术误差界和每个认证结论的边界见 [设计说明](lite-quality-inverse-design.md)。下面只把完成声明数量、通过标准和私有加载检查的试产计为成功。','',
          '## 观察分布','',
          '全部 25,600 个冻结窗口逐项精确计数；1,600 个窗口另外以完整 Python packed-bit popcount 对照。没有全亮、全暗或把实体零误算为阴影的窗口。暗主导 8,641 个，亮主导 16,959 个。十档以精确整数 `10*lit//free` 划分，精确比例 0 和 1 不允许。','',
          table(['亮/自由比例','自然','分层','30/50/20 混合'],[(f'{i*10}–{i*10+10}%',natural['bins'][i],strat['bins'][i],mixed['bins'][i]) for i in range(10)]),'',
          '![亮比例](lite-quality-assets/lit-ratio-distribution.png)','',
          '混合实际分量为自然 7,680、均衡 12,800、极端 5,120。这些是软选择分量，最终直方图不等于三个理想密度的机械相加。按数据集累计缺额选择可用档；场景缺档时回填并报告，候选池缺档不等于该场景在数学上不存在该档。自然路径调用原 D，不改变选择坐标；十场景显式复现原坐标，整个匹配 100 场景的窗口文件也完全相同。','',
          '![家族分布](lite-quality-assets/family-lit-ratios.png)','',
          table(['窗口边长']+[f'{i/10:.1f}–{(i+1)/10:.1f}' for i in range(10)],[[side,*natural['by_side'][str(side)]] for side in (64,128,256)]),'',
          '自然集相关性仅作描述：场景平均亮比例与阴影占据率 Pearson r=-0.936，与障碍数量 -0.623，与实际面数 -0.544。窗口级相关性更弱；同一场景内窗口不独立，不能据此报告 IID 显著性或因果关系。完整家族表、空间覆盖、IoU、源位置相关性在 `frozen-distribution-audit.json` 和 `frozen-window-counts.parquet`。']
    counts=read(OUT/'current-count-benchmark/summary.json');cr=counts['rows'];facts['counts']={'build_seconds':quantiles([r['build']['wall_seconds'] for r in cr]),'persistent_bytes':quantiles([r['persistent_bytes'] for r in cr])}
    text+=['','## 精确计数索引','',
           '16³ 叶计数加 65³ 粗前缀，常驻数组 3,245,576 字节，另有最多 8 MiB 双字段 packed-brick 缓存。非对齐边缘分成互不重叠的六个长方体，读取实际位；没有 1024³ 前缀或全世界 bool 张量。18 个场景、5,400 个随机位置查询全部与完整 packed-window 扫描相同。构建中位 '+fmt(facts['counts']['build_seconds']['p50'])+' 秒，压缩索引文件中位 '+f"{facts['counts']['persistent_bytes']['p50']/1024:.1f}"+' KiB。','',
           table(['边长','索引跨场景中位 μs','直接扫描跨场景中位 μs'],[[s,f"{np.median([r['query_seconds'][str(s)]['p50'] for r in cr])*1e6:.1f}",f"{np.median([r['direct_seconds'][str(s)]['p50'] for r in cr])*1e6:.1f}"] for s in (64,128,256)]),'',
           '64³ 直接扫描更便宜，128³ 接近；256³ 的索引明显有收益。没有声称索引在所有尺度都更快。一次构建和每次查询的文件检查、解码、内存及缓存计数保存在 `current-count-benchmark/summary.json`。']
    vr=read(OUT/'verification-v3/summary.json')['scenes'];facts['verification']={'before_B_cpu':quantiles([r['baseline_frozen_stages'][1]['cpu_seconds'] for r in vr]),'new_B_cpu':quantiles([r['B2']['cpu_seconds'] for r in vr])}
    text+=['','## 独立物理验证','',
           'A 的参数裁剪/GMP 实现没有改动。B 使用精确真实面/轮廓边去除冗余 FME 约束、规范化平面的严格 binary64 误差包络和父子区域证书继承；不调用 A 的裁剪决定。近零仍用原紧 nextafter 区间及 GMP。2^-46 是从系数和运算误差推导的包络，绝不是几何容差。','',
           table(['冻结 slot','家族','A 原墙钟 s','B 原墙钟 s','新 B 墙钟 s','FME → 轮廓约束'],[[r['slot'],r['family'],f"{r['baseline_frozen_stages'][0]['wall_seconds']:.2f}",f"{r['baseline_frozen_stages'][1]['wall_seconds']:.2f}",f"{r['B2']['wall_seconds']:.2f}",f"{sum(r['planes']['original_fme_planes'])} → {sum(r['planes']['retained_planes'])}"] for r in vr]),'',
           '原完整场复核用两个原生线程，新 B 独立对照用一个线程；CPU 时间另存，不能把线程数差异隐藏为算法收益。三例重复执行了未改变 A/B 的全部所需砖，20 例都执行相同空间分层控制砖。新 B 对 20 例全部所需砖重新物理计算，响应和实体位全部一致，另有三个完整 1024³、每例 4,096 砖逐位对照。所有家族的完整 32³/64³/128³、切触、薄体、非对齐区域也通过。','',
           '![验证器](lite-quality-assets/independent-verifier.png)','',
           '新 B 的改进是整套实现对照，不把全部差异归于平面数量一个因素。冻结随机场的 GMP 点回退为零，主要旧成本是海量区间平面/裁剪运算；源相关有理准备单独记录。A 仍是新生产长尾的主要成本。两套方法共享 canonical 平面、几何证明和 GMP，但不共享可见性决定；不称为完全独立的软件栈。']
    names=('core-smoke-10','core-matched-100','core-sequential-10','core-pipeline-1-10','core-pipeline-4-10','core-fast-natural-10','core-fast-mixed-10','core-fast-mixed-100','core-final-smoke-10','core-final-100','core-fast-mixed-300');study_rows=[]
    for name in names:
        p=OUT/name/'report.json'
        if not p.exists():continue
        r=read(p)
        if not r['complete']:study_rows.append([name,f"进行中 {r['ready_scenes']}/{r['requested_scenes']}",'—','—']);continue
        elapsed=r['wall_seconds_this_run'];stages={k:quantiles([s['stages'][k]['wall_seconds'] for s in r['scenes']]) for k in r['scenes'][0]['stages']}
        facts['studies'][name]={'ready':r['ready_scenes'],'wall_seconds':elapsed,'ready_per_hour':r['ready_scenes']/elapsed*3600,'stages':stages,'telemetry':r['telemetry'],
                               'parent_cpu_seconds':r.get('parent_cpu_seconds_this_run',0),'child_cpu_seconds':r.get('children_cpu_seconds_this_run',0),'scientific_bytes':sum(s['scene_bytes'] for s in r['scenes']),
                               'families':dict(Counter(s['family'] for s in r['scenes'])),'failures':r['failures']}
        study_rows.append([name,f"{r['ready_scenes']}/{r['requested_scenes']}",f'{elapsed:.1f}',f'{r["ready_scenes"]/elapsed*3600:.1f}'])
    text+=['','## 端到端 READY 吞吐与配对消融','',
           table(['研究','完成 READY','总墙钟 s','READY/小时'],[['冻结原 100', '100/100','7211.3',f'{3600*100/7211.31:.1f}'],*study_rows]),'',
           'Core 匹配 100 场景的几何、真源、响应文件、实体文件和自然窗口文件全部逐文件 SHA 相同。改动的是可选引用搜索关闭、独立 B 验证实现、计数诊断和调度；没有用不同的容易场景冒充加速。旧 100 的单场中位 62.63 秒不应与流水线的输出间隔混淆。新匹配 100 的平均输出约 12.77 秒一个，原平均约 72.11 秒，组合持续吞吐约 5.65 倍。','',
           '十场景同配置下，顺序单验证进程、流水线 1/2/4 进程分别单独测量。快几何和混合观察改变生成/选择分布，另行命名；不把整个配置包的效果归因于单一组件。十场景快几何未显示明显额外端到端收益，不能预设面数上限必然改善吞吐。','',
           '![吞吐](lite-quality-assets/ready-throughput.png)']
    for name in ('core-matched-100','core-fast-mixed-100','core-final-100','core-fast-mixed-300'):
        if name not in facts['studies']:continue
        r=facts['studies'][name];text+=['',f'### {name} 阶段延迟','',table(['阶段','P50 s','P95 s','P99 s','最大 s'],[[k,*[fmt(v[q]) for q in ('p50','p95','p99','max')]] for k,v in r['stages'].items()]),'',
            '阶段服务时间会重叠，不能相加当作持续输出间隔。30 秒是砖间软调度阈值，长场景仍完成、没有速度筛选。所有失败、超时和相同身份重试写入 ledger；它们是不完整尝试，不是物理拒绝。',
            '',f"资源：PSS 峰值 {r['telemetry']['max_family_pss_bytes']/2**30:.2f} GiB；RSS 求和峰值 {r['telemetry']['max_family_rss_bytes']/2**30:.2f} GiB；GPU 全局内存峰值 {r['telemetry']['max_global_gpu_used_MiB']:.0f} MiB（含桌面）；GPU 最高温度 {r['telemetry']['max_temperature_C']:.0f}°C。CPU 为父进程 {r['parent_cpu_seconds']:.1f} + 子进程 {r['child_cpu_seconds']:.1f} 秒。"]
    text+=['','## 非局部遮挡与数据质量','',
           '每个暗自由采样点用 GMP 精确比较首进入参数与窗口面，区分窗内、窗外和边界。首命中在窗外不表示窗外几何必需；只有证明整段在闭窗口内没有任何障碍接触，才认证该点的外部必要性。540 个自然选择代表窗口中，14,304 点有 12,692 个首命中在窗外、1,612 个在窗内。等权代表窗口平均“外部首命中”约 0.882，“外部必要”约 0.793；这是样本描述，不是整个数据集的精确比例。','',
           '分层和混合观察分别另做同样的空间分层精确点审计，保存 family/side/ratio 分层扩展估计及按阴影体素数加权的比率估计。每个点的结论是认证结果，整体比例仍是有限样本估计，不作 IID 置信度或穷尽因果声明。',
           '', '## 可选引用搜索与逆源几何','',
           'Core 的额外搜索预算默认为零，不创建候选池、读取窗口探针或做可选验证。每窗仍有一个完整双重物理验证的真源，不声明唯一性。离线 enrichment 最多八个总引用，全部窗外、FREE、净空、最小间距 32/1024，并通过完整 A/B 响应及实体比较。外置版本化产物绑定原场景文件哈希，默认私有，原 READY 场景未修改。','',
           '27 个冻结窗口覆盖九类家族、三个大小和不同亮比例，其中 9 个有旧参考替代解；参考坐标仅用于选案例，没有加入提议池。0/0.01/0.05/0.2 秒标称预算样本中未完成额外源认证；记录的是包含准备的实际墙钟，单个不可中断调用可能超过标称软预算，超时不表示无物理解。','',
           '九个 64³ 窗口的 2 秒预算初版空间探针对照：随机 11、局部扰动 10、逆约束 13、混合 12 个额外窗口引用，总墙钟分别约 10.83/15.65/9.95/12.41 秒。改用精确相邻亮暗自由体素后，在相同窗口上再对所有方法配对，仍不把采样见证当作完整逆解集。','',
           '![搜索](lite-quality-assets/inverse-search-yield.png)','',
           '反向阴影约束保留每个暗点的凸阴影体之并、每个亮点的并集之补。预算耗尽的盒保留在外覆盖，不假设单峰，也不将粗细搜索点混成概率标签。外覆盖只是必要条件的体积上界，不是实际可行体积、认证内部盒或完整连通分支。','',
           '![源空间外界](lite-quality-assets/inverse-outer-volumes.png)','',
           '有三个 256³ 窗口得到更强结论：整个源空间的保留/删除 dyadic 单元经独立完整八叉分区重建，无遗漏；所有删除盒由未改变的全 FME 验证器独立复查。保留区域内任意源到真源的距离严格小于 32/1024，因而在声明的窗外域内，不存在满足参考间距的第二源。此结论适用于完整窗口，因为完整解集必为所用 64 个真实观测约束的子集。它不表示连续位置唯一。','',
           '![间距证书](lite-quality-assets/separation-certificates.png)','',
           '每个案例又找到了 6 个距离约 2^-40 的不同源，全部通过完整原 256³ 窗口 A/B 双重验证。这些点太近，明确只作研究见证，不写入数据集参考；没有因此放宽最小间距，也没有凭六点推断已证明一个连续内部盒。源空间证书和近邻证据分别在 `inverse-outer-certificates`、`nearby-ambiguity-witnesses.json`。','',
           '现有 CUDA points 内核的反向批量探针对照逐项与原裁剪一致，但源相关准备、编码和传输的总成本未胜过 CPU；未采用。逆源搜索仅有小样本收益和三例有用的认证歧义结果，仍是离线研究工具，不进入 Core 必经路径。']
    text+=['','## 成本外推与采用建议','']
    estimates=[]
    for name in ('core-matched-100','core-fast-mixed-100','core-final-100','core-fast-mixed-300'):
        if name not in facts['studies']:continue
        r=facts['studies'][name];n=r['ready'];cpu=r['parent_cpu_seconds']+r['child_cpu_seconds']
        for target in (5000,10000):estimates.append([name,target,f"{r['wall_seconds']/n*target/3600:.1f}",f'{cpu/n*target/3600:.1f}',f"{r['scientific_bytes']/n*target/2**30:.2f}"])
    text+=[table(['测量配置','目标场景','墙钟小时','CPU 小时','科学数据 GiB'],estimates),'',
           '这是当前硬件与声明混合的线性外推，包含真实端到端预处理与已记录失败尝试，不是已执行量产。科学数据包含几何、完整双字段、窗口、私有物理证据与引用；外部工作日志、报告、试验重复字段另算。温度/功耗、并发任务、长尾和未遇到的罕见几何仍有不确定性，未承诺线性延伸到所有家族或任意复杂度。','',
           '建议进入下一版生产候选：精确计数、可选透明分层、默认零额外搜索、独立 B 加速、有界单 GPU/CPU 队列及安全恢复。原自然路径与九类几何保留；快几何混合只有实际质量/吞吐结果支持时才选择，不因名称叫 fast 而默认采用。GPU 探针优化和逆源搜索仍保留为实验；认证歧义分析可离线使用。未开启 5k–10k。','',
           '## 使用与复现','',
           '新入口是 `rind3d core-pilot`，配置在四个 `configs/lite-core-*-research-v1.yaml`；旧 `lite-pilot` 不接受这些研究配置，以免忽略新设置。先运行原 `certify-lite` 和 `python -m experiments.lite_quality.certify`，再运行 10 场景；100 需匹配的 10 场景报告，300 需匹配的 100 场景报告。代码/构建变化会关闭旧夹具门禁。生产生成目前从 checkout 运行，配置和无损夹具随仓库；独立 wheel 的读取/计数已另外测试。','',
           '```python\nfrom rind3d import RINDDataset\nfrom rind3d.data import make_dataloader\n\nwith RINDDataset("outputs/my-core-10", split=None) as ds:\n    sample = ds[0]\n    batch = next(iter(make_dataloader(ds, batch_size=1, num_workers=2)))\n    private = ds.private_oracle()\n    counts = private.get_window_counts(sample["scene_id"], sample["window_id"])\n    print(counts.lit, counts.shadow, counts.solid, counts.lit_ratio)\n```','',
           '重复任意 ROI 查询可显式复用 `CountIndex.build(y,m)` 或 `CountIndex.load(path,y,m)`，调用 `query((x,y,z), side)`。旧场景只在内存建临时索引，不写回。命令行有 `build-count-index`、`window-counts`。外置引用通过 `enrichment_research.read_artifact(artifact,scene)` 显式私有读取，默认再次完整双重物理重查，不改变普通模型输入。','',
           '## 验证和保护','',
           '完整回归、真实 READY 集合与普通/私有 PyTorch 加载、原数值构建可移植读取、源空间分区反例、闭边界/薄体、失去 CUDA、进程锁、OS 硬看门狗和原子恢复均有针对性检查。复制的发布中断演练恢复后科学文件逐字节相同，原数据未变。最终测试日志、wheel 读取结果、源码快照和保存审计均保留在输出目录。原有四个 fork 弃用警告来自历史隔离诊断，未改动那些实验。','',
           '18,249 个历史文件 SHA 与 52,104 个兄弟工程文件的大小/修改时间审计无变化，原科学实现身份相同。只新增独立研究模块，并扩展 CLI、公共验证包装、私有计数入口和文档；没有修改原科学判定、旧场景、RINDRay 或相邻数据集。']
    append_extended(text,facts)
    (OUT/'measured-facts.json').write_text(json.dumps(facts,indent=2)+'\n');Path('docs/lite-quality-throughput-report.md').write_text('\n'.join(text)+'\n')


def append_extended(text,facts):
    from .results import run
    run();extra=read(OUT/'extended-results.json');facts['extended']=extra
    spatial=read(OUT/'matched-selection-quality.json')
    natural=read(OUT/'frozen-distribution-audit.json')
    text+=['','## 分布收益的代价和实际可用性','',
        table(['相同冻结 100 场景','4096 控制点覆盖估计','占用 128³ 中心单元','窗口对平均 IoU','平均最大邻窗 IoU'],
              [[name,*[f"{spatial[name]['scene_mean_metrics'][key]:.4f}" for key in ('fixed_4096_control_point_union_coverage','occupied_centre_cells_128','mean_pair_iou','mean_max_other_iou')]] for name in ('natural','stratified','mixed')]),'',
        '覆盖是固定控制点估计，不是认证精确并集体积。均衡采样提高暗窗覆盖，但增加重叠并降低空间覆盖；混合是可选折中，不能宣传为所有方面都更好。相同十场景当前实现的选择墙钟：自然 28.43 秒、分层 59.34 秒、混合 60.32 秒，包含精确计数与失败提议。分层的原较慢读位版本也保留，未冒充当前速度。', '',
        table(['冻结自然选择的几何家族','暗主导窗口比例（lit/free < 0.5）'],[[name,f'{100*sum(values[:5])/sum(values):.1f}%'] for name,values in natural['by_family'].items()]),'',
        '在现有冻结种子和自然选择下，barriers 与 multiscale_convex 提供的暗主导比例较高；thin_slabs 并非最高。家族、光源和空间布置共同影响结果，不能单凭“薄墙”名称保证更暗，也不能把这个小规模相关性当成生成因果实验。新快配比只取认证凸并，并完整记录丢弃/保留组件与父几何身份，不按实测速度挑选场景。']
    for label,q in extra['quality'].items():
        text+=['',f'### {label}','',
            table(['亮比例档']+[str(s) for s in (64,128,256)]+['合计'],
                [[f'{i/10:.1f}–{(i+1)/10:.1f}',*[q['by_side'][str(s)][i] for s in (64,128,256)],q['bins'][i]] for i in range(10)]),'',
            f"精确有效窗口 {q['windows']:,} 个；发现池均匀提议 {q['uniform_pool_proposed']:,} 次，接受 {q['uniform_pool_accepted']:,} 次（{100*q['uniform_pool_acceptance']:.2f}%），独立均匀接受控制 {q['independent_control_windows']:,} 个未参与选择。首次拒绝计数：`{q['first_rejections']}`。缺档回填：`{q['selection_fallbacks']}`；缺档场景/尺寸/比例单元 {sum(r['scene_count'] for r in q['unavailable_pool_cells'])} 个。混合模式的全均匀目标累计差额 `{q['final_cumulative_uniform_target_deficits']}` 包含有意保留的自然/极端分量，不能全解释为物理供给不足。",'',
            table(['家族','场景数','障碍数量 P50/P95','实际总面数 P50/P95','实体占据 P50','暗自由占据 P50'],
                [[family,r['scenes'],f"{r['obstacles']['p50']:.0f}/{r['obstacles']['p95']:.0f}",f"{r['actual_total_faces']['p50']:.0f}/{r['actual_total_faces']['p95']:.0f}",f"{r['solid_occupancy']['p50']:.4f}",f"{r['shadow_free_occupancy']['p50']:.4f}"] for family,r in q['geometry_statistics'].items()]),'',
            '障碍参数、方向矩阵和中心位置逐场保存在质量审计 JSON；这是真实面数，不是生成器声明的平面数。几何/光源失败种子保留在 ledger，未按速度替换。']
    text+=['','## 非局部依赖与亮比例的联合审计','',
        table(['选择/试产','抽样场景数 / slot 范围','按窗口总体分层扩展：外部首命中','按窗口总体分层扩展：外部必要','按暗体素分层比率：外部必要'],
            [[name,f"{r['sampled_scenes']} / {r['sampled_slot_range'][0]}–{r['sampled_slot_range'][1]}",*[f"{r['population_estimates'][key][estimate]:.4f}" for key,estimate in (('weighted_external_first','window_population_stratum_weighted_estimate'),('weighted_external_necessary','window_population_stratum_weighted_estimate'),('weighted_external_necessary','shadow_voxel_stratum_ratio_estimate'))]] for name,r in extra['nonlocal'].items() if r['population_estimates']]),'',
        '这些是对确定性选出的 family/size/ratio 分层窗口及其分层点样本的描述性扩展估计。每个点通过精确有理区间进入/离开参数认证；总体估计没有穷尽认证，也不提供 IID 置信度。首命中在窗外和窗外几何因果必要分别统计。']
    if 'sustained-300-nonlocal-hashed' in extra['nonlocal']:
        text+=['','原取样按 slot 的简单评分偏向早期场景，因此另外保存一份 SHA256 优先序的 family/side/ratio 分层抽样。它覆盖 215 个不同场景、slot 0–299，360 个窗口及 9,434 个精确暗自由点。原结果未改写；两种审计的差异不能当作采样算法的配对因果改善。哈希优先序审计也仍是描述性抽样，不将点级认证推广为总体精确比例。']
    latest=next((extra['nonlocal'][name] for name in ('sustained-300-nonlocal-hashed','sustained-300-quality-audit','final-100-quality-audit','mixed-nonlocal') if name in extra['nonlocal']),None)
    if latest:
        text+=['',table(['亮比例档','总体窗口','抽样窗口','外部首命中估计','外部必要估计'],
            [[f"{r['ratio_bin']/10:.1f}–{(r['ratio_bin']+1)/10:.1f}",r['population_windows'],r['sampled_windows'],f"{r.get('weighted_external_first',0):.4f}",f"{r.get('weighted_external_necessary',0):.4f}"] for r in latest['by_ratio_bin']]),'',
            '增加暗窗后仍保留大量认证外部依赖，但不能据有限审计声称全体窗口的因果比例已精确确定。样本坐标、首个阻挡组件、精确参数、窗内反事实段和仅用于展示的距离都可追溯。']
    joint=extra['inverse_geometry_joint']
    text+=['','### 观测比例、非局部性和源空间外覆盖的同窗对照','',
        table(['家族','边长','lit/free','外部必要估计','全源外覆盖体积上界','达到间距的源外覆盖上界'],
              [[r['family'],r['side'],f"{r['lit_ratio']:.4f}",f"{r['external_necessary_estimate']:.4f}",f"{r['all_source_outer_volume_upper']:.3g}",f"{r['separated_source_outer_volume_upper']:.3g}"] for r in joint['records']]),'',
        '这是 27 个特意挑选的冻结窗口的联合诊断。表中体积是预算受限、稀疏必要约束的外覆盖上界，不是实际可行体积；不同几何的求界成本也不同。亮比例、外部遮挡比例或外覆盖大小都不能单独推出源唯一性或保证存在达到间距的第二源。描述性秩相关和逐例搜索工作量另存 `extended-results.json`，不报告 IID 显著性或因果关系。']
    search_path=next((OUT/name/'summary.json' for name in ('inverse-five-method-cold-v2','inverse-five-method-cold') if (OUT/name/'summary.json').exists()),OUT/'inverse-five-method-cold/summary.json')
    if search_path.exists():
        search=read(search_path);facts['final_search_comparison']={k:v for k,v in search.items() if k!='records'}
        cpu_label='CPU 含共同准备 s' if all('cpu_seconds_including_full_common_geometry_cost' in r for r in search['methods'].values()) else 'CPU 搜索 s（共同准备未计）'
        text+=['','## 五种逆源方法的冷缓存匹配对照','',
            '九个相同冻结 64³ 窗口，每种方法名义 2 秒；逐方法清空解码砖及源模型缓存，以种子旋转方法顺序。所有约束构建、筛选和完整原 A/B 验证计入实际时间；另把每窗共同必需的几何准备完整计入每种独立方法的总成本。OS 文件页缓存没有伪称为冷磁盘。', '',
            table(['方法','额外窗口引用','不同非真源坐标','成功窗口/9','墙钟含共同准备 s',cpu_label,'引用/总墙钟秒','评估提议'],
                [[name,r['additional_window_references'],r['unique_nontruth_coordinates'],r['successful_windows'],f"{r['wall_seconds_including_full_common_geometry_cost']:.3f}",f"{r.get('cpu_seconds_including_full_common_geometry_cost',r['cpu_seconds']):.3f}",f"{r['references_per_total_wall_second']:.3f}",r['stats'].get('proposals_evaluated',0)] for name,r in search['methods'].items()]),'',
            'silhouette 方法的 SVD 仅生成提议，不是求解器或几何证书。inverse 的不确定/未访问盒保留，不以近似相交发布解。逐 family、side、亮比例、外部遮挡水平及已知替代解的完整小样本记录见 `inverse-v4-cold`、`inverse-nonlocal`、`inverse-five-method-cold`；小于等于 0.2 秒预算常被准备成本耗尽，不能推断不存在解。', '',
            '一次软预算可被不可中断的有理构建/完整物理调用超出，表格使用实际耗时；未完成的额外候选不保留。这个九窗试验不足以保证整体搜索收益，默认 Core 继续零搜索。包含共同几何准备 CPU 计时的最终重复在 `inverse-five-method-cold-v2`；前一次五法对照也保留，没有用更有利的引用数替换历史结果。']
    text+=['','## 最终生命周期验收与持续运行','',
        '在进程锁、OS 看门狗基础上补齐了协调进程死亡时工作进程的内核退出信号及启动竞态检查；防止旧工作进程在恢复后继续写入。此改动后重新生成当前代码身份的夹具、十场景和百场景门禁，不把旧实现的成功报告直接用作新版本 300 场通行证。复制恢复演练和真实强制终止单元测试分别保留。', '']
    for name in ('core-final-100','core-fast-mixed-300'):
        if name not in extra['pilots']:continue
        r=extra['pilots'][name]
        text+=[f'### {name} 持续运行细项','',
            f"{r['ready']} 个 READY，{r['unique_geometry_ids']} 个不同 geometry_id，{r['source_geometry_pairs']} 个 geometry/source 对。不同试验重复的前缀场景不能相加当作更多独立几何。事件计数 `{r['event_counts']}`；源提议拒绝 `{r['source_proposal_rejection_reasons']}`；硬看门狗 `{r['watchdogs']}`。Core 引用分布 `{r['candidate_histogram']}`，只说明零可选预算下保留真源，不说明唯一性。",'',
            table(['验证器','单线程墙钟 P50/P95/P99/max s','CPU P50/P95/max s','每场唯一物理砖 P50/max'],
                [[backend,' / '.join(f"{row['wall_seconds'][k]:.3f}" for k in ('p50','p95','p99','max')),' / '.join(f"{row['cpu_seconds'][k]:.3f}" for k in ('p50','p95','max')),f"{row['unique_bricks_per_scene']['p50']:.0f}/{row['unique_bricks_per_scene']['max']:.0f}"] for backend,row in r['verification'].items()]),'',
            f"冷 CUDA 初始化 {r['cold_cuda_setup_seconds']:.3f} 秒；每场 kernel P50/P95 {r['kernel_seconds']['p50']:.3f}/{r['kernel_seconds']['p95']:.3f} 秒，双向传输 {r['transfer_seconds']['p50']:.3f}/{r['transfer_seconds']['p95']:.3f} 秒。GPU 利用率按 1 秒采样均值 {r['gpu_utilization_percent_mean']:.1f}%，不能将短 kernel 的利用率当作 READY 吞吐；平均使用 {r['cpu_cores_mean']:.2f} 个 CPU 核。渲染器自有设备内存最大 {r['max_cuda_owned_bytes']/2**20:.2f} MiB，桌面/GPU 全局占用另见资源表。",'',
            table(['READY 顺序批次','含前一 READY/冷启动至本批末 s','READY/小时'],[[f"{b['start_ready']}–{b['end_ready']}",f"{b['seconds']:.2f}",f"{b['ready_per_hour']:.2f}"] for b in r['ready_batches']]),'',
            f"全部科学场景压缩文件 {r['scientific_bytes']:,} 字节；根目录含工作 journal、遥测、报告和索引表总计 {r['root_total_bytes']:,} 字节。逐阶段 P50/P95/P99/最大值在前表和 `extended-results.json`。",'',
            f"若按此实测平均文件大小保留全部工作材料，5k/10k 的运行根目录约需 {r['root_total_bytes']/r['ready']*5000/2**30:.2f}/{r['root_total_bytes']/r['ready']*10000/2**30:.2f} GiB；其中科学场景文件约 {r['scientific_bytes']/r['ready']*5000/2**30:.2f}/{r['scientific_bytes']/r['ready']*10000/2**30:.2f} GiB。额外保留的本轮对照副本、源码和 wheel 不在这个生产根目录外推内。"]
        if name=='core-fast-mixed-300':text+=['','![持续吞吐](lite-quality-assets/sustained-throughput.png)']
    status=read(OUT/'campaign-status.json')
    test_runs=[s for s in status['stages'] if 'regression' in s['name']]
    for s in test_runs:
        lines=Path(s['log']).read_text().splitlines()
        s['pytest_summary']=next((line for line in reversed(lines) if ' passed' in line),'见原始日志')
    facts['regression_runs']=test_runs
    text+=['','## 最终检查记录与可复现入口','',
        table(['检查','通过','pytest 结果','总进程墙钟 s','日志'],[[s['name'],s['passed'],s['pytest_summary'],f"{s['wall_seconds']:.2f}",f"`{s['log']}`"] for s in test_runs]),'',
        '普通和私有 DataLoader 在每个场景原子发布前、整批结束后都实际执行。独立 wheel 读取、计数、强制退出与恢复、旧 CUDA 缺失路径、SHA/CRC/manifest 损坏、几何组划分和私有标签隔离均另有检查；未以 kernel microbenchmark 代替整套验收。','',
        '```bash\n# 两个认证夹具匹配当前构建后，依次通过 10/100/300；不得跳过先决报告。\nuv run rind3d core-pilot --config configs/lite-core-fast-research-v1.yaml \\\n  --scenes 300 --verification-workers 4 \\\n  --pilot-report outputs/lite-v02-research/core-final-100/report.json \\\n  --output outputs/my-bounded-300\n\n# 只读原 READY，单独发布不可覆盖的私有补充版本。\nuv run rind3d enrich --scene DATA_ROOT/scenes/SCENE_ID --windows 0 17 \\\n  --config configs/ambiguity-enrichment-research-v1.yaml \\\n  --output outputs/enrichment/SCENE_ID-v1.private.json\n```','',
        '最终源文件清单、构建身份、冻结场哈希、测试原始日志和逐次失败记录全部保留。复现时设置 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`。推荐四个单线程 CPU 验证进程配一个 GPU 生产者；当前剩余长尾主要在 A 的独立参数裁剪。快速几何配比只带来有限额外收益，保留完整九家族自然 Core 作为可比较配置；分层/混合是有明确分布代价的质量选项。', '',
        '硬看门狗覆盖隔离 CPU 验证进程；GPU 生产者仍使用原 streaming 预算和砖级检查点。若 CUDA 驱动调用本身卡死，需要外部终止协调器并显式恢复，当前没有驱动级自动抢占保证。罕见硬件故障、未遇到的几何长尾、热环境变化，以及 5k–10k 的实际失败率仍未测量。']
    text+=['',
        '另一个量产前需单独验证的工程边界是目录规模：当前每次发布仍重写累计 catalog 和运行报告。三百场持续测试包括了这些实际成本，但没有测量单一根目录增长到一万场后的重写成本，不能假设它完全线性。正式开放 5k–10k 前应验证增量/分片目录方案及其几何组划分；本轮没有为此扩展新的数据框架，也没有开放量产入口。']
    wheel=OUT/'wheel-final/loader-test.json'
    if wheel.exists():
        record=read(wheel);facts['standalone_wheel_read']=record
        text+=['',f"最终 wheel 只读验收：`{record['wheel']}`，实际读取 {record['scenes']} 个场景、{record['windows']:,} 个窗口的索引；普通及私有 PyTorch DataLoader 均通过。父进程和 spawn 工作进程都断言模块来自解包 wheel，而不是本地 editable 源码；精确计数也由 wheel 中扩展完成。读取测试没有修改原数据。"]

if __name__=='__main__':write()
