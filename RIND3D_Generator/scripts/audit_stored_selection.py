"""Independent uniform controls and common stored-bit distribution measurements.

Composition is counted over the complete packed window, on an explicitly
seeded subset. Common audit witnesses are uniform within each FREE response
class. Construction proofs and these audit witnesses are reported separately.
No oracle, generation, candidate-source search, or READY publication occurs.
"""
import argparse,json,os,resource,subprocess,time
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from rind3d.information import InformationIndex
from rind3d.pack import RINDPack,atomic_json
from rind3d.stored_windows import uniform_candidates,keyed_seed
from rind3d.window_contract import WINDOW_SIDES
from run_stored_selection_pilot import BENCH,validate_rows

POPCOUNT=np.array([i.bit_count() for i in range(256)],dtype=np.uint8)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def describe(values):
    a=np.asarray(values,dtype=float)
    return None if not len(a) else {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),
        'p05':float(np.quantile(a,.05)),'p95':float(np.quantile(a,.95)),'min':float(a.min()),'max':float(a.max())}


def uniform_class_point(bits,origin,rng):
    flat=bits.ravel();counts=POPCOUNT[flat];cumulative=np.cumsum(counts,dtype=np.uint32)
    if not len(cumulative) or not cumulative[-1]:raise AssertionError('Missing response class')
    rank=int(rng.integers(int(cumulative[-1])));byte=int(np.searchsorted(cumulative,rank,side='right'))
    within=rank-(int(cumulative[byte-1]) if byte else 0);value=int(flat[byte]);positions=[i for i in range(8) if value&(1<<i)]
    z,y,bx=np.unravel_index(byte,bits.shape)
    return tuple(int(o+v) for o,v in zip(origin,(bx*8+positions[within],y,z)))


def face_margin(point,origin,side):
    local=np.asarray(point)-np.asarray(origin)
    return int(np.minimum(local,side-1-local).min())


def spatial(values,source,side,seed):
    if not values:return {'n':0}
    origins=np.asarray([r['origin'] for r in values],dtype=float);centers=origins+side/2;truth=np.asarray(source)*1024
    buckets=np.floor(centers/128).astype(int);distance=np.linalg.norm(centers-truth,axis=1)
    outside=np.maximum(np.maximum(origins-truth,truth-(origins+side)),0);near=np.linalg.norm(outside,axis=1)
    n=len(values);rng=np.random.default_rng(seed);pair_iou=[];nearest_iou=[];nn=[]
    if n>1:
        d,which=cKDTree(centers).query(centers,k=2);nn=d[:,1].tolist()
        delta=np.abs(origins-origins[which[:,1]]);intersection=np.maximum(0,side-delta).prod(axis=1)
        nearest_iou=(intersection/(2*side**3-intersection)).tolist()
        pairs=rng.integers(n,size=(min(8192,n*(n-1)),2));pairs=pairs[pairs[:,0]!=pairs[:,1]]
        delta=np.abs(origins[pairs[:,0]]-origins[pairs[:,1]]);intersection=np.maximum(0,side-delta).prod(axis=1)
        pair_iou=(intersection/(2*side**3-intersection)).tolist()
    construction=[face_margin(p,row['origin'],side) for row in values for p in row['witnesses']]
    pair_counts=Counter(row.get('witness_pair_id') for row in values if row.get('witness_pair_id') is not None)
    regions={tuple(row['witness_region']) for row in values if row.get('witness_region') is not None}
    return {'n':n,'unique_origins':len({tuple(row['origin']) for row in values}),
        'occupied_center_cells_128':len({tuple(v) for v in buckets}),
        'max_windows_per_center_cell_128':max(Counter(map(tuple,buckets)).values()),
        'occupied_octants':len({tuple(v) for v in np.floor(centers/512).astype(int)}),
        'center_span_xyz':np.ptp(centers,axis=0).tolist(),'center_std_xyz':centers.std(axis=0).tolist(),
        'center_source_distance':describe(distance),'window_source_distance':describe(near),
        'world_boundary_margin':describe(np.minimum(origins,1024-side-origins).min(axis=1)),
        'nearest_center_distance':describe(nn),'nearest_iou':describe(nearest_iou),'sampled_pair_iou':describe(pair_iou),
        'sampled_pair_iou_gt_half':float(np.mean(np.asarray(pair_iou)>.5)) if pair_iou else None,
        'construction_witness_face_margin':describe(construction),
        'construction_witness_within_one_voxel_face':float(np.mean(np.asarray(construction)<=1)),
        'selected_guided_pairs':len(pair_counts),'selected_guided_regions':len(regions),
        'maximum_selected_pair_reuse':max(pair_counts.values(),default=0),
        'provenance_counts':dict(Counter(row['provenance'] for row in values))}


def composition(values,r,m,seed,maximum):
    rng=np.random.default_rng(seed);sample=sorted(rng.choice(len(values),min(maximum,len(values)),replace=False).tolist())
    measurements=[];start=time.perf_counter();cpu=time.process_time()
    # Brick-local ordering improves IO without changing the preselected subset.
    for i in sorted(sample,key=lambda i:tuple(v//64 for v in values[i]['origin'])):
        row=values[i];o=row['origin'];side=row['side'];y=r.read_window(*o,side,packed=True).data;s=m.read_window(*o,side,packed=True).data
        lit=y&~s;shadow=~y&~s;a=int(POPCOUNT[lit].sum(dtype=np.uint64));b=int(POPCOUNT[shadow].sum(dtype=np.uint64))
        if not a or not b:raise AssertionError('Independent full packed scan found a noninformative accepted window')
        witnesses=[uniform_class_point(c,o,rng) for c in (lit,shadow)]
        margins=[face_margin(p,o,side) for p in witnesses]
        measurements.append({'row_index':i,'origin':o,'side':side,'lit_free_count':a,'shadow_free_count':b,
            'solid_count':side**3-a-b,'lit_fraction_free':a/(a+b),'solid_fraction':1-(a+b)/side**3,
            'uniform_class_audit_witnesses':witnesses,'audit_witness_face_margins':margins})
    margins=[v for row in measurements for v in row['audit_witness_face_margins']]
    return {'n':len(measurements),'kind':'complete packed-bit counts on seeded window subset; no probes',
        'lit_fraction_free':describe([v['lit_fraction_free'] for v in measurements]),
        'solid_fraction':describe([v['solid_fraction'] for v in measurements]),
        'common_audit_witness_face_margin':describe(margins),
        'common_audit_witness_within_one_voxel_face':float(np.mean(np.asarray(margins)<=1)) if margins else None,
        'wall_seconds':time.perf_counter()-start,'process_seconds':time.process_time()-cpu,'measurements':measurements}


def control(case):
    out=BENCH/case['id']/'independent-control';out.mkdir(exist_ok=True)
    if (out/'result.json').exists():return
    values=[];reports={};start=time.perf_counter();cpu=time.process_time()
    with RINDPack(case['response']) as r,RINDPack(case['solid']) as m:
        index=InformationIndex.load(case['index_path'],r,m)
        for side in WINDOW_SIDES:
            pool,report=uniform_candidates(index,case['source'],side,128,65536,keyed_seed(case['seed'],case['scene'],19001,side))
            for row in pool:
                proof=index.query(row['origin'],side,witnesses=True,witness_seed=1);row['witnesses']=[proof.lit,proof.shadow]
                row.update(control_only=True,control_id=len(values));values.append(row)
            reports[str(side)]={**report,'complete':len(pool)==128,'target':128}
    validation=validate_rows(values,case)
    (out/'selected.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in values))
    atomic_json(out/'result.json',{'by_side':reports,'validation':validation,'wall_seconds':time.perf_counter()-start,
        'process_seconds':time.process_time()-cpu,'independent_seed_stream':19001,'used_for_selection':False,
        'index_build_wall_not_included':case['index_build']['wall_seconds'],'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'oracle_calls':0})


def audit(case,maximum):
    root=BENCH/case['id'];output=root/'distribution.json'
    if output.exists():return
    results=[];start=time.perf_counter();cpu=time.process_time()
    with RINDPack(case['response']) as r,RINDPack(case['solid']) as m:
        for path in sorted(root.glob('*/selected.jsonl')):
            if not (path.parent/'result.json').exists():continue
            selected=rows(path);group=[]
            for side in WINDOW_SIDES:
                values=[v for v in selected if v['side']==side]
                measurement=composition(values,r,m,keyed_seed(case['seed'],case['scene'],20002,side),maximum)
                group.append({'side':side,'spatial':spatial(values,case['source'],side,123+side),'composition':measurement})
            results.append({'run':path.parent.name,'by_side':group})
    atomic_json(output,{'case':case['id'],'results':results,'maximum_composition_windows_per_side_run':maximum,
        'wall_seconds':time.perf_counter()-start,'process_seconds':time.process_time()-cpu,'oracle_calls':0,
        'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'audit_controls_not_training_support':True,'common_witness_policy':'uniform voxel in each actual free class; distinct from construction proof'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case');p.add_argument('--mode',choices=['control','features'])
    p.add_argument('--composition-per-side',type=int,default=16);p.add_argument('--all',action='store_true');p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if a.all:return run_all(a.workers,a.composition_per_side)
    case=json.loads((BENCH/a.case/'case.json').read_text())
    if a.mode=='control':control(case)
    else:audit(case,a.composition_per_side)


def run_all(workers,maximum):
    cohort=json.loads((BENCH/'cohort.json').read_text());pending=[(case,mode) for case in cohort['cases'] for mode in ('control','features')];running=[]
    start=time.perf_counter();done=[]
    while pending or running:
        for case,mode in pending[:]:
            root=BENCH/case['id'];destination=root/('independent-control/result.json' if mode=='control' else 'distribution.json')
            if destination.exists():pending.remove((case,mode));continue
            if len(running)>=workers:break
            if mode=='features' and (not (root/'independent-control/result.json').exists() or any(not (root/f'{method}-{target}'/'result.json').exists() for method in cohort['methods'] for target in cohort['targets'])):continue
            log=(root/f'audit-{mode}.log').open('w');cmd=['.venv/bin/python',__file__,'--case',case['id'],'--mode',mode,'--composition-per-side',str(maximum)]
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
            pending.remove((case,mode));running.append((case['id'],mode,process,log))
        for item in running[:]:
            case_id,mode,process,log=item;code=process.poll()
            if code is None:continue
            log.close();running.remove(item);done.append({'case':case_id,'mode':mode,'exit_code':code});print(done[-1],flush=True)
            if code:raise RuntimeError(f'Independent audit failed: {case_id} {mode}; inspect its log')
        atomic_json(BENCH/'audit-progress.json',{'done':done,'pending':len(pending),'running':[(c,m) for c,m,*_ in running],'elapsed_seconds':time.perf_counter()-start,'composition_per_side':maximum})
        if pending or running:time.sleep(.5)


if __name__=='__main__':main()
