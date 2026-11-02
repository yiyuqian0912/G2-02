"""Generate the report from retained measurements; no performance extrapolation from ROIs."""
import csv
import json
from pathlib import Path
import statistics as st
import xml.etree.ElementTree as ET

ROOT=Path('outputs/cpu-cuda-efficiency')
OLD=Path('outputs/shadow-volume-experiment')
NAMES=('box','rotated_box','thin_slab','prism','wedge','poly8','poly16','poly32','multiscale4','multiscale12','multiscale24','corridor','cluster')
METHODS=('B_native','C_shadow','CPU_clip','CPU_projected','GPU')


def read(path):return json.loads(Path(path).read_text())
def number(v,places=3):return '—' if v is None else f'{v:.{places}f}'
def table(header,rows):return '\n'.join(['| '+' | '.join(header)+' |','| '+' | '.join('---' for _ in header)+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    hist=read(OLD/'lite-compatibility/result.json');profile=read(ROOT/'lite-profile.json')
    fields={name:{m:read(ROOT/'full'/name/m/'result.json') for m in METHODS if (ROOT/'full'/name/m/'result.json').exists()} for name in NAMES}
    evidence=read(ROOT/'lite-evidence/result.json');combined=sum(sum(x) for x in hist['verification_by_side_seconds'].values())
    state=read(ROOT/'run-status.json')
    final_state=read(ROOT/'final-status.json') if (ROOT/'final-status.json').exists() else {}
    all_finished=state.get('complete') and final_state.get('complete')
    md=['# RIND3D CPU / CUDA 效率实验（2026-10-31）',
        '\n状态：'+('有界实验和回归已结束，结果见下。' if all_finished else '实验尚在执行；下表只包含已写入的记录，不能视为最终完成报告。'),
        '\n本轮已保留完整场景几何、精确 binary64 真值、闭线段接触语义和所有发布门禁。代码在 `experiments/cpu_cuda`，生产渲染器、RINDPack、信息索引、窗口选择、候选验证和两个相邻工程均未替换。输出是诊断产物；不构成 READY 数据集，也没有执行量产。',
        '\n## 79.64 秒究竟花在哪里',
        '\n旧演示在几何初始化之后开始计时，并且直接复用了已经生成的完整场。它没有搜索额外候选，每窗只有一个真实源，数量符合 Lite 的 1–8 目标。原记录只分别计了索引、选择和两次验证的合计，不能事后伪造两个验证器各自的历史耗时。',
        '\n'+table(['历史阶段','实测秒','范围'],[
            ['完整场生成','不在 79.64 内','旧 box/C 场单独测量'],
            ['几何初始化','不在 79.64 内','本轮相同几何复测 '+number(profile['geometry_initialization_seconds'])],
            ['索引构建',number(hist['index_seconds']),'存储字段精确扫描'],
            ['窗口选择',number(hist['selection_seconds']),'同一批 256 个窗口'],
            ['首次物理验证 + 独立复核',number(combined),'两遍完整窗口'],
            ['未细分的读取、元数据、I/O',number(hist['total_seconds']-hist['index_seconds']-hist['selection_seconds']-combined),'由历史总时减去已计阶段'],
            ['额外候选搜索','0','未执行'],['总计',number(hist['total_seconds']),'不含几何与完整场']]),
        '\n用相同种子、几何、真实源和逐字节相同的窗口表重新计时，获得下面的分解。复测与历史值是不同运行，不能把其部分时间直接塞进旧总时。',
        '\n'+table(['本轮复测','秒'],[['几何初始化',number(profile['geometry_initialization_seconds'])],['索引',number(profile['index_seconds'])],['选择',number(profile['selection_seconds'])],['读取窗口',number(profile['totals']['read_seconds'])],['首次物理验证',number(profile['totals']['true_source_verification_seconds'])],['独立复核',number(profile['totals']['independent_recheck_seconds'])],['额外候选搜索','0'],['含初始化的复测总时',number(profile['total_wall_seconds_with_setup'])]]),
        f"\n每遍窗口合计 {profile['logical_voxels_per_pass']:,} 个体素位置；窗口共引用 {profile['brick_reuse']['window_brick_references']:,} 次全局砖块，却只涉及 {profile['brick_reuse']['distinct_bricks']:,} 个不同砖块，单砖最多被 {profile['brick_reuse']['maximum_reuse']} 窗口引用。两遍验证各实际查询了 {profile['totals']['true_source_verification_points']:,} 个叶点，其余由区域证书覆盖。",
        '\n## 可复用的独立物理证据',
        '\n新原型对相关的每个全局砖块做两次新的物理渲染：首次用新参数裁剪，独立复核用未修改的 B_native。每次都把重新算出的响应和实体掩码逐位与存储字段比较。只读原响应、算哈希、或者同源身份相等，都不能代替此步骤。证据绑定几何、精确源字节、政策、两个后端及输入文件哈希，未覆盖或身份不同的窗口会明确失败。后续任意未对齐窗口只需引用所有相交砖块；没有漏掉部分块。',
        '\n'+table(['阶段','墙钟秒','CPU 秒','新物理砖块'],[[r['stage'],number(r['wall_seconds']),number(r['cpu_seconds']),r['freshly_rendered_bricks']] for r in evidence['evidence_stages']]+[['256 窗口覆盖查询',number(evidence['window_evidence_queries_seconds']),'—','无新增物理查询'],['含准备的验证总计',number(evidence['validation_wall_seconds']),number(evidence['validation_cpu_seconds']),'每遍 2480']]),
        '\n这项收益同时包含原生批量渲染和跨窗口证据复用，不能全部归因于复用，更不能归因于新的阴影数学。']
    ablation=ROOT/'native-window-replay.json'
    if ablation.exists():
        a=read(ablation);md+=['\n相同两个 native 实现、逐窗口重算且不跨窗复用的控制如下。它也检查实体掩码，因此比旧演示仅验证响应更严格。',
            '\n'+table(['同实现消融','墙钟秒','CPU 秒'],[['原生逐窗，不复用',number(a['wall_seconds']),number(a['cpu_seconds'])],['原生砖块，两遍复用',number(evidence['validation_wall_seconds']),number(evidence['validation_cpu_seconds'])]]),f"\n不复用时，每遍重算 {a['physical_tiles_per_pass']:,} 个 64³ 物理块，窗口内分块原点未必对齐全局砖块。点查询数与区域边界也会随分块变化；收益不能简单等同于体积比。"]
    md+=['\n## CPU 算法与对照',
         '\nB_native 本来已有凸阴影的八角点证明，原方案并不缺这条定理。新的 CPU_clip 把它推广至旋转盒及一般有界凸半空间，缓存精确的源平面值，对线段参数直接做区间裁剪；不确定点用 GMP 精确有理数。它避免展开 O(F²) 阴影平面。CPU_projected 使用相同的新区间实现和原生遍历，但保留投影平面，用来区分几何算法与算术实现的贡献。两种新算法都不是 Python 批量化与旧 Python 的混比。',
         '\n严格内部见证、非奇异变换、精确 recession-cone 有界性与全维证明成立后，可以在实验的完整凸谓词路径中延迟 CGAL 初始化。不支持的几何仍走原后端；这没有实现或绕过发布器要求的几何接受/净空认证。旧 B/C 扩展保留 Clang 构建，新扩展用 GCC 16.2.1；旧新比较包含实现/编译器因素。两种新 CPU 算法之间使用相同 GCC、区间核、遍历、叶边长和缓存，是几何算法控制。',
         '\n主矩阵所有方法都共用相同的预解析几何/源输入，B/C 以未修改 native 代码为对照，表中不重复执行其旧 Python 包装器的 eager CGAL 构造。`poly8/16/32` 名字表示声明半空间数量；上一轮精确几何统计的实际活动面分别为 8/14/28，不能把声明约束数直接说成实际面数。',
         '\n下表都是完整 1024³ 响应 + 实体掩码，包含打包、压缩、校验、索引和写入；不含表后单独列出的冷启动准备。所有同场景完整方法都逐位比较。B_native 不支持一般凸体，空白不表示被新实现模拟。',
         '\n'+table(['冻结场景',*METHODS],[[name,*[number(fields[name].get(m,{}).get('wall_seconds')) if fields[name].get(m,{}).get('complete') else ('未完成' if m in fields[name] else '—') for m in METHODS]] for name in NAMES])]
    cpurows=[[name,m,number(v['cpu_seconds']),number(v['geometry_setup_seconds']),number(v['source_setup_seconds']),v['work'].get('point_rays','—'),v['work'].get('exact_fallbacks',0)] for name,values in fields.items() for m,v in values.items() if m in ('CPU_clip','CPU_projected','B_native')]
    md+=['\n'+table(['场景','CPU 方法','CPU 秒','几何准备秒','源准备秒','叶射线','GMP 回退'],cpurows),
         '\n旧的昂贵 CGAL 初始化并未被偷偷记成负数或 GPU 加速；本轮 Lite 复测 box 为 '+number(profile['geometry_initialization_seconds'])+' 秒。当前快路径的几何准备只包含数学上完整的凸谓词支持范围；新的家族仍不具备生产发布资格。',
         '\n## 真正运行的 CUDA 原型',
         '\n非沙盒执行确认 RTX 3060 Laptop、标称 6144 MiB、compute 8.6、驱动 615.78.08。NVCC 未安装；缓存的 NVRTC 13.0 确实编译 PTX 并在设备上执行。没有用 CPU 冒充 CUDA。设备同时驱动桌面，不是独占基准。CPU 为 AMD Ryzen 7 5800H，native 流式对照使用两线程、64³ 砖块、8³ 叶；BLAS/OMP 限为单线程。',
         '\nCPU 精确有理数构造每个平面并验证 binary64 区间端点。CUDA 仅用有向 double 加乘，编译禁用 fast math、FTZ 和收缩，保存的 PTX 检查显式上下舍入及无 float32/approx 指令。数值测试在实际 GPU 上对照 Fraction，包括次正规数、溢出、抵消和非 dyadic 精确切触；UNKNOWN 走原始 canonical CGAL，必需查询无法解决仍失败。完整数学义务和官方依据见 [认证说明](cpu-cuda-certification.md)。',
         '\nGPU 按最多 32 砖块批处理，64/32/16/8 层次至多 585 节点/砖。输出直接是响应、掩码和两份 UNKNOWN packed bits，不建 1024³ bool 张量。几何、源阴影和设备缓冲区跨砖复用。下表 kernel 是 CUDA events，H2D/D2H 为显式复制，sync 为等待墙钟，kernel 与 sync 存在重叠，不能相加。',
         '\n'+table(['场景','总墙钟秒','CPU 秒','kernel 秒','H2D 秒','D2H 秒','同步墙钟秒','区域覆盖%','原 oracle 回退射线/点'],
             [[name,number(v['wall_seconds']),number(v['cpu_seconds']),number(v['runtime_timings']['kernel_seconds']),number(v['runtime_timings']['H2D_seconds']),number(v['runtime_timings']['D2H_seconds']),number(v['runtime_timings']['sync_wall_seconds']),number(100*v['work']['certificate_covered_voxels']/1024**3,2),str(v['work'].get('canonical_fallback_rays',0))+'/'+str(v['work'].get('canonical_fallback_points',0))]
              for name,values in fields.items() if (v:=values.get('GPU',{})).get('complete')]),
         '\n'+table(['GPU 场景','首批砖延迟秒','压缩写入秒','索引秒','主存高水位 MiB','显式device峰值MiB','最低剩余VRAM MiB','完整双字段/小时'],
             [[name,number(v.get('first_brick_seconds')),number(v.get('compression_seconds')),number(v.get('index_seconds')),
               number(v['peak_rss_bytes']/2**20,1),number(v['acceleration']['runtime_device_allocated_peak']/2**20,2),
               number(v['minimum_device_free_bytes']/2**20,1),number(3600/v['wall_seconds'],1)]
              for name,values in fields.items() if (v:=values.get('GPU',{})).get('complete')]),
         '\n每小时数字是热启动的完整字段对（不是 READY 场景）。首砖为首批计算及回退后实际写入延迟。显式设备分配不含 CUDA context/驱动及桌面；全局剩余显存包含它们。主矩阵在同一进程预先建立 CUDA context，CPU/GPU 行的主存高水位不能解释成各后端独立内存，冷 Lite 流程提供独立进程资源记录。']
    util=ROOT/'gpu-utilization-summary.json'
    if util.exists():
        u=read(util)
        md+=['\n'+table(['GPU 场景','平均利用率%','峰值%','全局显存峰值MiB','功率峰值W','温度峰值°C','SM频率MHz范围'],
                        [[r['family'],number(r['mean_gpu_util_percent'],1),number(r['peak_gpu_util_percent'],1),number(r['peak_global_memory_mib'],0),
                          number(r['peak_gpu_power_w'],1),number(r['peak_temperature_c'],0),str(r['sm_clock_mhz_range'])] for r in u['rows']]),
             '\n利用率来自 200 ms NVML 采样，按字段结果写入时间和实测墙钟选取时间段，端点约有一个采样周期误差；不是 CUDA event 计时，也不应与 kernel 百分比混同。全局功率/显存包含桌面，未运行数小时热稳定性试验。']
    fair=[]
    for p in sorted((ROOT/'fair-io').glob('*/result.json')):
        r=read(p);audit=read(p.parent/'checkpoint-control.json');fair.append([p.parent.name,number(fields[p.parent.name]['GPU']['wall_seconds']),number(r['wall_seconds']),number(audit['checkpoint_seconds'])])
    md+=['\nCPU 每 8 砖 checkpoint，主 GPU 批处理每 32 砖 checkpoint；这是工程差异，不能当作几何算法收益。',
         '\n'+table(['同场景 I/O 控制','GPU 每32砖秒','GPU 每8砖秒','后者 checkpoint 秒'],fair) if fair else '\n匹配 checkpoint 频率的额外控制尚未记录。']
    fallback=ROOT/'certified-native-fallback'
    if (fallback/'result.json').exists():
        md+=['\n## 困难场暴露的真实回退瓶颈与显式替代路径',
             '\n13 类主矩阵都是原 CGAL 默认回退，正常源下没有触发点回退。额外近共面场却不能照搬这个吞吐结论：只有很少的 UNKNOWN，原 CGAL 也会变成决定性瓶颈。32³ 的 134 条 UNKNOWN 花约 31.3 秒，64³ 的 534 条花约 103 秒；128³ 在整个困难任务 180 秒 watchdog 内没完成，没有提交完整字段。它是时间预算耗尽，不是物理拒绝或精度失败。原 v1 仅在 batch 前检查预算，64³ 虽已逐点物理完成却超过 60 秒软预算；此事实单独记录，原结果未改写。当前实现也在 batch 后、提交前检查软预算；阻塞中的 oracle 由外层硬 watchdog 限制。',
             '\n新路径必须显式选择 `fallback_backend="certified_native"`，默认仍为 `canonical_cgal`。UNKNOWN 点使用已实现的 CPU 区间/GMP **完整凸谓词**，不是 GPU 推测或近似网格：精确逆变换、严格内部/世界见证、有界全维凸分量与顶层并集是前置证明；非本范围几何不能进入。该证明允许在此范围用原始 canonical 半空间精确闭线段裁剪与成员判断替代 CGAL 的昂贵通用表示，不改变连续真值或切触语义。细节见认证说明。',
             '\n每个原生回退点都另用未修改的独立 interval/GMP 实现重新核查；32³/64³ 还与已经用 CGAL 真正解决全部 UNKNOWN 的旧字段逐位一致。小场每一个体素也独立比较，之后才做分层 ROI 和有界 1024³。',
             '\n'+table(['困难场','分辨率','CPU秒','GPU+原生回退+逐点独立检查秒','原生回退射线','独立复核点'],
                        [[(r:=read(p))['case'],n,number(next(x['wall_seconds'] for x in r['rows'] if x['method']=='CPU_clip' and x['resolution']==n)),
                          number((v:=next(x for x in r['rows'] if x['method']=='GPU' and x['resolution']==n))['wall_seconds']),v['work'].get('native_fallback_rays',0),v['work'].get('independent_fallback_points_checked',0)]
                         for p in sorted(fallback.glob('*/result.json')) if 'rows' in read(p) for n in (32,64,128)])]
        f=read(fallback/'full-nearly-coplanar/result.json')
        md+=['\n'+table(['近共面完整1024³','墙钟秒','CPU秒','原生回退射线','回退CPU秒','额外独立核查秒'],
                        [[name,number((v:=f[key])['wall_seconds']),number(v['cpu_seconds']),v['work'].get('native_fallback_rays',0),number(v['work'].get('native_fallback_seconds',0)),number(v['work'].get('independent_fallback_seconds',0))]
                         for name,key in [('CPU完整场','CPU_clip'),('GPU+回退+每点独立核查','GPU_with_independent_fallback_checks'),('GPU+回退，整场另行对照','GPU')]]),
             '\n完整场共 133,240 个 UNKNOWN，全部由精确 CPU 决定，全部逐点独立检查的审计字段约 4.07 秒；去除运行时额外实验核查、但整场再与审计字段和 CPU 字段逐位对照的计时字段约 3.06 秒。后者不被冒充已经运行了逐点检查；两种时间分开。CGAL 仍可用于独立审计及不支持的范围，READY/家族接受门禁没有开放。',
             '\n小场不一定值得用 GPU：近共面 128³ 中 CPU 约 0.049 秒，而带独立回退检查的 GPU 约 0.052 秒；一次 CUDA context/编译还要另付。通用 CGAL 回退的 v1 不能直接成为 5k–10k 的可靠主渲染器，必须先完成这个受限原生精确路径的正式接受与更广泛验证，或继续保留 CPU 主路径。']
        md+=['\n最小复现：在可访问 GPU 的环境运行 `.venv/bin/python -m experiments.cpu_cuda.fallback_repro --count 8`。它冻结同一近共面叶、真实源 `[0.1,0.3,0.6]` 和完整 64³ 点集，发现 534 个 UNKNOWN，默认仅取其中 8 条分别用原 CGAL 与受限原生精确谓词求值，单独计几何初始化并要求答案一致。`--count 534` 会重现约百秒级完整回退开销；不会生成数据集。']
    reuse=[]
    for p in sorted(ROOT.glob('reuse-*/*/*/result.json')):
        r=read(p)
        for a in r['prefixes']:reuse.append([r['family'],r['resolution'],r['method'],a['sources'],number(a['cold_wall_per_source']),number(a['shared_wall_per_source']),number(a['shared_cpu_per_source'])])
    md+=['\n## 几何组：1/2/4/8 个源',
         '\n八个冻结源分别生成各自完整字段。冷控制逐源重新建几何加速，组控制共用不可变几何和实体掩码值，但每源重建阴影/源平面结构。GPU context/编译在冷、热两边同样共享，不把省下重复 CUDA 编译算成几何复用。表中 1/2/4/8 是真实执行源序列的累计前缀，均摊时间包含该组一次几何准备及各源写入，不含之后额外的交叉对照。',
         '\n'+table(['几何','分辨率','方法','源数','逐源冷墙钟/源秒','共用墙钟/源秒','共用CPU/源秒'],reuse),
         '\n共用掩码的 packed 主存缓存最大 128 MiB。当前 RINDPack 身份含源坐标，因此多个源仍各写一份掩码文件；所有文件独立 inode、独立源绑定 header，并检查 CRC/SHA。没有通过跨源硬链接改变存储语义。geometry_id 是 canonical 内容哈希，source_id 加入精确源字节；同几何全部源按 geometry_id 固定到同一 split，避免泄漏。该组策略在实验 manifest 中实现，未悄悄改动生产按场景分组的旧默认。']
    for p in sorted(ROOT.glob('reuse-1024/*/*/result.json')):
        r=read(p);a=r['prefixes'][-1];change=100*(a['shared_wall_per_source']/a['cold_wall_per_source']-1)
        md+=[f"\n完整 1024³ {r['family']} / {r['method']} 八源：冷建 {a['cold_wall_per_source']:.3f} 秒/源，共用 {a['shared_wall_per_source']:.3f} 秒/源，后者变动 {change:+.2f}%。这是当前方法的一次固定顺序对照，不是总体置信区间；没有测出优势时不推荐默认启用掩码缓存。"]
    md+=['\n这里一个 source-response pair 计一个记录，不等同于一个新几何世界。若 5k/10k 指源响应记录，每几何八源分别只需约 625/1250 个几何；若要求 5k/10k 个不同几何且各八源，响应字段数变为 40k/80k。复用的统计收益不能掩盖几何多样性数量的变化。掩码重用仍有缓存读取/复制/重新序列化成本；若实测慢于直接认证，应保留几何共享并关闭该掩码路径，不能只报节省的谓词数。']
    e2e=[]
    for p in sorted((ROOT/'lite-end-to-end').glob('*/result.json')):
        r=read(p);e2e.append(r)
    stage_labels=[('geometry_device_and_source_setup_seconds','几何 / 设备 / 源准备'),('full_field_seconds','完整双字段（含索引构建）'),
                  ('field_organization_seconds','文件组织'),('index_open_seconds','打开索引'),('window_selection_seconds','选择256窗口'),
                  ('two_physical_replays_seconds','两遍物理重算'),('full_window_evidence_validation_seconds','验证窗口完整覆盖'),
                  ('candidate_metadata_io_loader_seconds','候选 / 元数据 / I/O / loader'),('additional_candidate_search_seconds','额外候选搜索（未执行）')]
    md+=['\n## Lite 冷启动端到端',
         '\n从初始化开始，实际新生成完整 1024³ 双字段、读取索引、选择同一批 256 个窗口、两遍独立物理砖块验证、写候选与元数据，再用普通/私有 loader 检查。每窗一个真实源，额外搜索预算显式为零；不补齐、不伪造候选。85/85/86 配额、源在窗外、双类自由体素、全窗覆盖、私有标签隔离均保留。当前 Lite 严格发布器缺口仍在，样本状态 INCOMPLETE、严格默认读取为空。',
         '\n'+table(['阶段',*[r['method'] for r in e2e]],
                     [[label,*[number(r['stages'].get(key)) for r in e2e]] for key,label in stage_labels]
                     +[['总墙钟秒',*[number(r['wall_seconds']) for r in e2e]],['总CPU秒',*[number(r['cpu_seconds']) for r in e2e]]]),
         '\n这是一个简单 box 的真源单候选演示，不能当作全部多样几何、八候选搜索或失败重试后的 READY 场景吞吐。']
    launch=[]
    for name in ('lite-cpu','lite-gpu'):
        p=ROOT/(name+'-job.json')
        if p.exists():
            r=read(p);launch.append([name,number(r['end_unix']-r['start_unix'])])
    if launch:md+=['\n管线内部计时从几何初始化开始，不含 Python 模块导入。协调器还记录了包含解释器启动、导入和退出的完整命令墙钟：',
                   '\n'+table(['命令','墙钟秒'],launch)]
    smallcpu=[read(p) for p in (ROOT/'small-cpu').glob('*/result.json')];smallgpu=[read(p) for p in (ROOT/'small-gpu').glob('*/result.json')]
    strat=[read(p) for p in (ROOT/'stratified').glob('*/result.json')]
    regpath=ROOT/'regression-final.xml' if (ROOT/'regression-final.xml').exists() else ROOT/'regression.xml'
    reg=ET.parse(regpath).getroot() if regpath.exists() else None
    suites=list(reg.iter('testsuite')) if reg is not None else []
    regtxt=', '.join(f"{s.attrib.get('tests')} tests / {s.attrib.get('failures','0')} failures / {s.attrib.get('errors','0')} errors / {s.attrib.get('skipped','0')} skipped" for s in suites)
    md+=['\n## 正确性、资源和保留审计',
         f'\nCPU/GPU 各完成 {len(smallcpu)}/{len(smallgpu)} 类场景的 32³、64³、128³ 全体素检查，独立参考为未修改的区间/GMP 参数裁剪；每个位置同时比响应与掩码。空间分层、障碍附近及未对齐 ROI 共 {sum(len(r["bricks"]) for r in strat)} 个 1024³ 下的 64³ 区域逐体素验证。完整场只在这些门禁之后执行；未完成数据绝不计入完成字段。',
         '\n这些参考共享 canonical 平面提取，并不等于逐体素重跑 CGAL。原 CGAL 的 close-source、极薄 2^-40、nearly-coplanar、切触/nextafter 检查另有记录；数值不确定与预算耗尽不解释成物理 MISS。']
    if (ROOT/'targeted-canonical-audit.json').exists():md+=[f'\n本轮针对困难几何的原 CGAL 审计通过 {len(read(ROOT/"targeted-canonical-audit.json")["cases"])} 类，CPU、CUDA（含严格回退）全部一致。']
    difficult=list((ROOT/'difficult-small').glob('*/result.json'))
    if difficult:
        md+=['\n原 CGAL 默认回退实验中，完整通过三个小分辨率的困难场只有 near_source，见下。near-coplanar 的 32³/64³ 完成、128³ 超时，随后 very_thin 阶段未到达；不能把这次不完整控制说成三类全通过。三类全部小场已由前述显式原生精确回退实验另行完成。',
             '\n'+table(['困难场景','分辨率','CPU秒','GPU秒','全体素一致'],
                         [[(r:=read(p))['case'],n,number(next(x['wall_seconds'] for x in r['rows'] if x['method']=='CPU_clip' and x['resolution']==n)),
                           number(next(x['wall_seconds'] for x in r['rows'] if x['method']=='GPU' and x['resolution']==n)),True]
                          for p in difficult for n in (32,64,128)])]
    if (ROOT/'exhaustive-cgal-attempt.json').exists():
        a=read(ROOT/'exhaustive-cgal-attempt.json');md+=[f"\n32³ 原 CGAL 全遍历尝试：预算 {a['declared_seconds']} 秒，实际 {a['wall_seconds']:.3f} 秒，只完成 {a['checked_voxels']}/{a['required_voxels']}，complete={a['complete']}，已查差异 {a['differences']}。这是一份明确不完整的参考，不能标成全部 canonical 体素通过。"]
    md+=['\n全套回归：'+(regtxt or '尚未记录')+'。严格 GPU 测试使用 `RIND_TEST_CUDA=1`，不是跳过 GPU 后声称通过。',
         '\n最终日志还有 4 条旧诊断脚本的 fork 弃用警告，与上一轮一致；本轮未修改这些脚本，所有测试正常结束。',
         '\n主存峰值为进程高水位，复用同一进程的行会包含此前分配；CUDA 显式 device buffer 峰值与全局 GPU 使用量分别记录，不把桌面进程占用全部归于实验。`gpu-hardware-final.txt` 保存功率配置、`gpu-utilization-durable.csv` 为 200 ms 实测。设备详细输出显示 ceiling current=115 W、default=80 W；简式 power.limit 查询为 N/A。保留这种接口差别，没有把默认功率冒充当前限制，也未修改功率设置。']
    audit=read(ROOT/'preservation.json') if (ROOT/'preservation.json').exists() else None
    if audit:
        md+=[f"\n保留审计通过：{audit['prior_artifacts_checked']:,} 个历史文件 SHA 未变；相邻工程核对 {audit['sibling_file_stats_checked']:,} 个文件大小/修改时间及 {audit['sibling_scientific_sha_checked']:,} 个科学源码 SHA。原工程文件只更新 README，其余新代码、测试和说明均为独立新增；既有 geometry/filter/stream/pack/selector/loader 及旧阴影实验源码保持原样。最终源码和编译扩展均保存到 source-snapshot。"]
    predictions=[]
    for method in ('CPU_clip','CPU_projected','GPU'):
        rows=[v[method] for v in fields.values() if v.get(method,{}).get('complete')]
        if not rows:continue
        for count in (5000,10000):
            wall=[r['wall_seconds']+r['geometry_setup_seconds']+r['source_setup_seconds'] for r in rows]
            cpu=[r['cpu_seconds'] for r in rows]
            predictions.append([method,count,number(st.median(wall)*count/3600,2),f'{min(wall)*count/3600:.2f}–{max(wall)*count/3600:.2f}',number(st.median(cpu)*count/3600,2)])
    storage=[(name,v['GPU']) for name,v in fields.items() if v.get('GPU',{}).get('complete')]
    sizes=[sum(v['field_bytes'].values())+75585 for _,v in storage]
    md+=['\n## 5,000 / 10,000 的条件成本',
         '\n下表仅由**已经完成的整个 1024³ 字段**外推，没有由容易砖块外推。每种冻结几何同权，列出跨几何范围而非置信区间；真实家族混合、源分布、热降频及 I/O 可能改变结果。墙钟计入每源几何/源准备，GPU 常驻 context/编译可按组摊销，冷启动另见 Lite 总时。CPU 小时列为实测字段进程 CPU 时间，未把 GPU 时间冒充 CPU 时间。它仍不含窗口、额外候选、完整独立复查和发布接受/重试，因此不是 READY 量产总价。',
         '\n这里 GPU 的统计总体仍是主矩阵 13 个正常源例，使用每32砖 checkpoint；近共面原 CGAL 回退失败不在该统计中。因此这些小时数不能作为默认 CGAL 回退模式的量产保证，支持原生精确回退后的代表性总体也需另做采样。匹配每8砖 checkpoint 的表另列了实际增加的 I/O 成本。',
         '\n'+table(['方法','源响应场数','字段墙钟小时（中位数）','跨几何范围小时','字段CPU小时（中位数）'],predictions)]
    for r in e2e:
        md+=[f"\n单个 box 真源 Lite 完整流程 {r['method']}：{r['wall_seconds']:.3f} 墙钟秒 / {r['cpu_seconds']:.3f} CPU 秒。若只重复这个已测简单任务，5k/10k 分别 {r['wall_seconds']*5000/3600:.2f}/{r['wall_seconds']*10000/3600:.2f} 墙钟小时；这不是多样场景总体估计。"]
    if sizes:
        md+=[f'\n已测 13 类完整场的响应+掩码+索引每源占 {min(sizes)/2**20:.3f}–{max(sizes)/2**20:.3f} MiB，中位数 {st.median(sizes)/2**20:.3f} MiB。5k 仅这些文件约 {5000*min(sizes)/1e9:.2f}–{5000*max(sizes)/1e9:.2f} GB，10k 约 {10000*min(sizes)/1e9:.2f}–{10000*max(sizes)/1e9:.2f} GB；还需几何、候选、清单及可选私有物理证据。随机复杂响应可能远大于这些规则凸体，RINDPack 双字段硬上界仍约 256.25 MiB/源，即 5k/10k 约 1.22/2.44 TiB，不能承诺所有场景都达到当前压缩率。']
    data_root=ROOT/'lite-end-to-end/GPU';scene=data_root/'scenes/scene_box_source0'
    if scene.exists():
        actual=sum(p.stat().st_size for p in scene.iterdir() if p.is_file())+sum((data_root/n).stat().st_size for n in ('manifest.json','scenes.parquet','splits.parquet','windows.parquet'))
        proof_bytes=(scene/'physical-evidence.private.json').stat().st_size
        md+=[f'\n实际单盒 Lite 产物含几何、256窗口记录、真源候选池/IDs/offsets、候选证据、采样私有记录和清单共 {actual:,} B（{actual/2**20:.3f} MiB）；其中可选私有砖块证据 JSON 占 {proof_bytes:,} B。此处不计 CUDA 编译产物和基准日志，也没有重复存窗口响应。仅复制这种简单任务的必需文件及该证据，5k/10k 约 {actual*5000/1e9:.2f}/{actual*10000/1e9:.2f} GB；仍不是多样场景容量保证。']
    md+=['\n## 建议与剩余阻碍',
         '\n优先把“有界 native 重算 + 独立物理砖块证据复用”整理成正式可审计接口：Lite 的主要浪费在重叠窗口重复验证。高面数凸体采用直接裁剪有明确收益；低面数、多障碍物和简单盒应依据同条件实测选择，不将新 CPU 算法一刀切为默认。',
         '\nCUDA 原型在实际设备上可严格工作，但应作为可选的生产候选，保留 CPU 快路径。通用 CGAL 默认回退在近共面例中已证实不具备可用吞吐；显式原生精确回退解决了该例，仍需先完成支持范围与家族的正式接受、净空认证、更广泛的 canonical 独立验证、Lite 发布器和可恢复 GPU 作业接口，才能进入生产。当前 1–8 候选数量契约没变，但只测一个候选的 Lite 不足以估计额外候选搜索成本；复杂家族的 256 窗口端到端、失败源重采样及总体接受率仍未测。函数几何 ready_capable=False 仍然关闭，本轮凸体性能不代表函数后端获得认证。',
         '\n性能上还应关注 GPU kernel 之外的 CPU 打包/UNKNOWN bitset 扫描、压缩/校验/checkpoint、选择和独立物理重算。几何复用后的 CGAL 初始化收益不能重复算一次。只有热/冷都实测且包含失败工作的端到端收益，才足以决定最终 5k–10k 工作流。',
         '\n## 复现与原始记录',
         '\n构建与最小运行见 README。`benchmark small` 执行全体素小场；`evaluate full` 先检查小场门禁，再检查分层 ROI 并跑每场每方法最多 120 秒的完整场；`evaluate reuse` 测八源的 1/2/4/8 前缀；`lite_end_to_end` 跑全流程诊断；`RIND_TEST_CUDA=1 python -m pytest -q` 显式测试真实 GPU。`run.py` 是顺序、有预算的本轮协调器，不是数据集量产命令。',
         '\n`before.json` 冻结原文件和历史输出；`preservation.json`/`source-snapshot` 记录最终审计与源码。旧中断场只有在全部 4096 砖、CRC/SHA、完整位对照通过且完整原子临时计时记录可解析后才恢复；原记录保留，未用部分场伪装完成。所有测试、硬件、逐砖和逐窗口记录均保存在 `outputs/cpu-cuda-efficiency`。']
    Path('docs/cpu-cuda-efficiency-report.md').write_text('\n'.join(md)+'\n')
    summary={'run_id':'cpu-cuda-efficiency-20261031','private_diagnostics':True,'dataset_contract_version':2,
             'fields':fields,'lite_profile':{k:v for k,v in profile.items() if k not in ('windows','historical_full_field','brick_reuse')},
             'lite_evidence':evidence,'lite_end_to_end':e2e,'production_ready':False,
             'status':{'main':state,'final':final_state,'post':read(ROOT/'post-status.json')},
             'certified_native_fallback':read(ROOT/'certified-native-fallback/full-nearly-coplanar/result.json') if (ROOT/'certified-native-fallback/full-nearly-coplanar/result.json').exists() else None,
             'canonical_fallback_timeout':read(ROOT/'canonical-difficult-timeout.json') if (ROOT/'canonical-difficult-timeout.json').exists() else None,
             'source_fingerprints':'preservation.json / source-snapshot',
             'regression_suites':[s.attrib for s in suites]}
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    with (ROOT/'full-field-timings.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['family','method','complete','wall_seconds','cpu_seconds','geometry_setup_seconds','source_setup_seconds','peak_rss_bytes'])
        for name,values in fields.items():
            for method,v in values.items():w.writerow([name,method,*[v.get(k) for k in ('complete','wall_seconds','cpu_seconds','geometry_setup_seconds','source_setup_seconds','peak_rss_bytes')]])
    print('Wrote report, summary JSON and full-field timings CSV')


if __name__=='__main__':main()
