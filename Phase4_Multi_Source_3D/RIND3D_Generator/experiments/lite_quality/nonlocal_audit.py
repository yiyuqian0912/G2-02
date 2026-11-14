"""Certified first-hit/window-relative occlusion, with local-only counterfactual."""
import argparse,hashlib,json,time
from collections import Counter
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from rind3d import _research
from rind3d.geometry import load_geometry
from rind3d.rendering.context import GeometryContext
from rind3d.rendering.polyhedra import rational_rows
from rind3d.pack import RINDPack,atomic_json
from rind3d.windows import Window
from rind3d.observation_research import sample_class
from rind3d.pipeline import parquet_write
from .audit import BASE,OUT

def run(base=BASE,counts=OUT/'frozen-window-counts.parquet',output=OUT,selection_order='legacy'):
    if selection_order not in ('legacy','hash'):raise ValueError('Unknown audit window ordering')
    base=Path(base);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'nonlocal-audit.json').exists():raise FileExistsError('Preserve the previous nonlocal audit; choose a new output directory')
    rows=pq.read_table(counts).to_pylist();scene_map={s['scene_id']:s for s in json.loads((base/'report.json').read_text())['scenes']};selected=[]
    # Bounded, explicitly stratified across family/size/ratio, max 2 windows/cell.
    for family in sorted({r['family'] for r in rows}):
        for side in (64,128,256):
            for bin in range(10):
                group=[r for r in rows if r['family']==family and r['side']==side and r['ratio_bin']==bin]
                if selection_order=='hash':
                    group.sort(key=lambda r:hashlib.sha256(f"nonlocal-audit-v2:17:{r['scene_id']}:{r['window_id']}".encode()).digest())
                else:group.sort(key=lambda r:(r['slot']*101+r['window_id']*17)%65537)
                selected.extend(group[:2])
    selected.sort(key=lambda r:(r['scene_id'],r['window_id']));results=[];start=time.perf_counter();count=Counter();component_maps={}
    for sid in sorted({r['scene_id'] for r in selected}):
        path=base/'scenes'/sid;s=scene_map[sid];meta=json.loads((path/'verification.json').read_text());ctx=GeometryContext(load_geometry(path/'geometry.npz'),meta['numerical_policy']);source=np.array(s['source']);model=_research.Nonlocal(rational_rows(ctx.bodies),source.tolist())
        component_maps[sid]={'geometry_id':ctx.geometry_id,'source_hex':[float(x).hex() for x in source],
                             'components':[{'index':i,'leaf_id':c.leaf_id,'codec':c.codec,'anchor':[str(v) for v in c.anchor]} for i,c in enumerate(ctx.components)]}
        with RINDPack(path/'response.rindb',verify_file=True) as y,RINDPack(path/'solid.rindb',verify_file=True) as m:
            for row in [r for r in selected if r['scene_id']==sid]:
                w=Window(tuple(row['origin']),row['side']);indices,sampling=sample_class(y,m,w,False,count=32,seed=row['slot']*1000+row['window_id'])
                if not len(indices):raise AssertionError('Informative window without sampled shadow voxel')
                points=(indices+.5)/1024;t=time.perf_counter();out=model.analyze(points,w.origin,w.size,1024);elapsed=time.perf_counter()-t
                if not np.all(~ctx.bind(source).response(source,points)):raise AssertionError('Exact first-hit sample is not shadow')
                counters=Counter(r['first_location'] for r in out);necessary=sum(r['external_necessary'] for r in out)
                # Octant-stratified counts give weighted window-shadow fractions.
                weighted_external=weighted_necessary=0.;total=sum(sampling['octant_class_counts'])
                for k,n in enumerate(sampling['octant_class_counts']):
                    sample=[r for r,o in zip(out,sampling['sample_octants']) if o==k]
                    if n and not sample:raise AssertionError('Nonempty octant unsampled')
                    if sample:
                        weighted_external+=n/total*sum(r['first_location']=='OUTSIDE' for r in sample)/len(sample)
                        weighted_necessary+=n/total*sum(r['external_necessary'] for r in sample)/len(sample)
                result={k:row[k] for k in ('scene_id','slot','window_id','family','side','origin','lit_ratio','ratio_bin')}
                result.update(samples=len(points),first_location=dict(counters),external_necessary_samples=necessary,weighted_external_first=weighted_external,
                              weighted_external_necessary=weighted_necessary,sampling=sampling,exact_samples=[dict(r,receiver=points[i].tolist()) for i,r in enumerate(out)],wall_seconds=elapsed)
                for entry,point in zip(result['exact_samples'],points):
                    if 't_float' in entry:
                        distance=float(np.linalg.norm(point-source));entry['source_to_first_hit_distance_display']=entry['t_float']*distance;entry['first_hit_to_receiver_distance_display']=(1-entry['t_float'])*distance
                result['certified_fraction_lower']=necessary/total
                result['certified_fraction_upper']=1-sum(r['local_blocking']=='inside' for r in out)/total
                results.append(result);count.update(counters)
    parquet_write(output/'nonlocal-windows.parquet',[{k:v for k,v in r.items() if k not in ('exact_samples','sampling','first_location')} for r in results])
    value={'complete':True,'windows':len(results),'samples':sum(r['samples'] for r in results),'first_locations':dict(count),
           'window_stratum_selection':selection_order,'unique_sampled_scenes':len({r['scene_id'] for r in results}),
           'sampled_scene_slots':sorted({r['slot'] for r in results}),
           'first_outside_is_not_causality':'external_necessary additionally proves no closed-segment obstacle contact inside the closed observation cube',
           'certified_points_estimated_fraction':'Each sampled result exact GMP. Window fractions use population-weighted stratified samples; not exhaustive causal fractions.',
           'boundary_policy':'All six faces reported separately. Boundary-only local contact is not certified externally necessary.',
           'wall_seconds':time.perf_counter()-start,'records':results}
    def group(r):return (r['family'],r['side'],r['ratio_bin'])
    populations=Counter(group(r) for r in rows);sizes=Counter(group(r) for r in results);shadow_population=Counter()
    for r in rows:shadow_population[group(r)]+=r['shadow_free']
    weighted={}
    for key in ('weighted_external_first','weighted_external_necessary'):
        window_mean=sum(populations[group(r)]/sizes[group(r)]*r[key] for r in results)/len(rows)
        group_shadow={}
        for g in sizes:
            sample=[r for r in results if group(r)==g];denom=sum(sum(r['sampling']['octant_class_counts']) for r in sample)
            group_shadow[g]=sum(sum(r['sampling']['octant_class_counts'])*r[key] for r in sample)/denom
        voxel_mean=sum(shadow_population[g]*group_shadow[g] for g in group_shadow)/sum(shadow_population.values())
        weighted[key]={'window_population_stratum_weighted_estimate':window_mean,'shadow_voxel_stratum_ratio_estimate':voxel_mean,
                       'sample_strata_equal_window_mean':float(np.mean([r[key] for r in results]))}
    value['population_estimates']=weighted;value['population_weighting_caveat']='Two deterministically selected windows per family/side/ratio stratum. Window expansion and shadow-population ratio estimates are descriptive, not exhaustive or IID confidence claims.'
    atomic_json(output/'nonlocal-audit.json',dict(value,base=str(base),count_records=str(counts),private_scene_component_maps=component_maps))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',default=str(BASE));p.add_argument('--counts',default=str(OUT/'frozen-window-counts.parquet'));p.add_argument('--output',default=str(OUT));p.add_argument('--selection-order',choices=('legacy','hash'),default='legacy');a=p.parse_args();run(a.base,a.counts,a.output,a.selection_order)
