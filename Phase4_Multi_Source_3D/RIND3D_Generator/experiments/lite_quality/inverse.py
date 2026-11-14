"""Matched optional-search budgets on frozen, ratio/size/family spread windows."""
import argparse,json,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from rind3d.geometry import load_geometry
from rind3d.rendering.context import GeometryContext
from rind3d.pack import RINDPack,atomic_json
from rind3d.windows import Window
from rind3d.enrichment_research import search,SharedCandidateVerifier,save_artifact
from rind3d.inverse_research import InverseConstraints,explore
from rind3d.observation_research import sparse_observations
from .audit import BASE,OUT

def run(label='inverse-v2',probe_strategy='spatial',cold_caches=False):
    rows=pq.read_table(OUT/'frozen-window-counts.parquet').to_pylist();old=json.loads((BASE/'report.json').read_text())['scenes'];selected=[]
    # One fixed window of each size/family; alternate lit-ratio bins and known
    # alternatives where available. References select evaluation cases only;
    # no supplied reference coordinate is inserted into any proposal pool.
    counts={s['scene_id']:np.fromfile(BASE/'scenes'/s['scene_id']/'candidate_counts.bin',dtype='<u2') for s in old}
    for r in rows:r['previous_reference_count']=int(counts[r['scene_id']][r['window_id']])
    for fi,family in enumerate(sorted({r['family'] for r in rows})):
        for si,side in enumerate((64,128,256)):
            group=[r for r in rows if r['family']==family and r['side']==side]
            with_alternatives=[r for r in group if r['previous_reference_count']>1]
            if (fi+si)%2 and with_alternatives:group=with_alternatives
            target=(fi+si*3)%10;group.sort(key=lambda r:(abs(r['ratio_bin']-target),r['slot'],r['window_id']));selected.append(group[0])
    folder=OUT/label;folder.mkdir(parents=True,exist_ok=True);records=[];start=time.perf_counter()
    for row in selected:
        sid=row['scene_id'];wid=row['window_id'];target=folder/f'{sid}-{wid}.json'
        if target.exists():records.append(json.loads(target.read_text()));continue
        path=BASE/'scenes'/sid;meta=json.loads((path/'verification.json').read_text());ctx=GeometryContext(load_geometry(path/'geometry.npz'),meta['numerical_policy']);truth=np.fromfile(path/'driver.f64',dtype='<f8');w=Window(tuple(row['origin']),row['side'])
        tests=[];common_geometry_seconds=0.
        with RINDPack(path/'response.rindb',verify_file=True) as y,RINDPack(path/'solid.rindb',verify_file=True) as m:
            for seconds in (0,.01,.05,.2):
                if cold_caches and seconds>0 and common_geometry_seconds==0:
                    tick=time.perf_counter();ctx.geometry_proof();common_geometry_seconds=time.perf_counter()-tick
                for method in ('random','local','silhouette','inverse','hybrid'):
                    # Cold method-local cache: no unreported cross-method warmup.
                    ctx._cache.clear();verifier=SharedCandidateVerifier(ctx)
                    if cold_caches:y.cache.clear();m.cache.clear()
                    points,proofs,report=search(ctx,truth,w,y,m,seconds=seconds,method=method,seed=31000+row['slot']*100+wid,verifier=verifier,probe_strategy=probe_strategy)
                    tests.append({'budget':seconds,'method':method,'positions':points.tolist(),'proofs':proofs,'report':report})
            if row['side']==64:
                for method in ('random','local','inverse','hybrid'):
                    ctx._cache.clear()
                    if cold_caches:y.cache.clear();m.cache.clear()
                    p,q,r=search(ctx,truth,w,y,m,seconds=2,method=method,seed=31000+row['slot']*100+wid,probe_strategy=probe_strategy)
                    tests.append({'budget':2.,'method':method,'positions':p.tolist(),'proofs':q,'report':r})
            t=time.perf_counter();sparse,labels=sparse_observations(y,m,w,count=32,seed=5);constraints=InverseConstraints(ctx,sparse,labels)
            _,outer=explore(constraints,truth,w,seconds=.5,max_regions=512,max_depth=5,seed=12,separation=0)
            _,separated=explore(constraints,truth,w,seconds=.5,max_regions=512,max_depth=5,seed=12,separation=32/1024)
            ambiguity={'construction_seconds':time.perf_counter()-t-outer['wall_seconds']-separated['wall_seconds'],'all_source_outer':outer,'separated_source_outer':separated,
                       'bounds_are_sparse_necessary_constraints_only':True,'candidate_count_is_not_uniqueness':True}
        record={'window':row,'tests':tests,'ambiguity':ambiguity,'reference_coordinates_never_used_as_proposals':True,
                'cold_decoded_and_source_model_caches_per_method':cold_caches,'common_required_geometry_vertex_preparation_seconds':common_geometry_seconds,
                'common_geometry_cost_scope':'Recorded once, valid geometry cache shared by every method; not charged repeatedly to each proposal or presented as zero.'}
        atomic_json(target,record);records.append(record);atomic_json(folder/'progress.json',{'complete':False,'windows':len(records),'elapsed_seconds':time.perf_counter()-start})
    atomic_json(folder/'summary.json',{'complete':True,'windows':len(records),'records':records,'wall_seconds':time.perf_counter()-start})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',default='inverse-v2');p.add_argument('--probe-strategy',choices=('spatial','transition'),default='spatial');p.add_argument('--cold-caches',action='store_true');a=p.parse_args();run(a.label,a.probe_strategy,a.cold_caches)
