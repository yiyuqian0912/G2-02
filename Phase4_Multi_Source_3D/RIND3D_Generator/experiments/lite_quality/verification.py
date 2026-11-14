"""Frozen slow-scene independent-verifier ablation; all new writes isolated."""
import argparse,json,time,resource
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from rind3d.geometry import load_geometry
from rind3d.rendering.context import GeometryContext
from rind3d.projection_research import ProjectionSource
from rind3d.pack import RINDPack,atomic_json
from rind3d.windows import Window
from rind3d.physical_evidence import bricks_for
from .audit import BASE,OUT

def replay(path,coords,renderer,*,workers=1):
    start=time.perf_counter();cpu=time.process_time();before=dict(renderer.work);latencies=[]
    with RINDPack(path/'response.rindb',verify_file=True,cache_bricks=0) as y,RINDPack(path/'solid.rindb',verify_file=True,cache_bricks=0) as m:
        def job(b):
            t=time.perf_counter();v=renderer.packed_batch([tuple(k*64 for k in b)],64,1024)[0];return b,v,time.perf_counter()-t
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for b,pair,elapsed in pool.map(job,coords):
                if pair!=(y.read_brick_packed(b),m.read_brick_packed(b)):raise AssertionError(f'Physical bits differ: {b}')
                latencies.append(elapsed)
    return {'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,'unique_bricks':len(coords),
            'all_response_and_solid_bits_match':True,'workers':workers,'work':{k:v-before.get(k,0) for k,v in renderer.work.items()},
            'brick_p50_seconds':float(np.median(latencies)),'brick_p95_seconds':float(np.percentile(latencies,95)),'brick_max_seconds':max(latencies)}

def run(limit=20,controls=3,label='verification'):
    old=json.loads((BASE/'report.json').read_text())['scenes'];scenes=sorted(old,key=lambda s:s['stages']['shared_independent_physical_verification']['wall_seconds'],reverse=True)[:limit]
    folder=OUT/label;folder.mkdir(parents=True,exist_ok=True);rows=[]
    for pos,s in enumerate(scenes):
        result_path=folder/(s['scene_id']+'.json')
        if result_path.exists():rows.append(json.loads(result_path.read_text()));continue
        path=BASE/'scenes'/s['scene_id'];meta=json.loads((path/'verification.json').read_text());ctx=GeometryContext(load_geometry(path/'geometry.npz'),meta['numerical_policy']);p=np.fromfile(path/'driver.f64',dtype='<f8')
        windows=[Window(tuple(map(int,r[:3])),int(r[3])) for r in np.fromfile(path/'window_records.bin',dtype='<u2').reshape(-1,4)]
        coords=sorted({b for w in windows for b in bricks_for(w)});b2=ProjectionSource(ctx,p)
        row={'slot':s['slot'],'scene_id':s['scene_id'],'family':s['family'],'baseline_frozen_stages':s['physical'],
             'B2_setup_seconds':b2.setup_seconds,'planes':b2.plane_report,'B2':replay(path,coords,b2),
             'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        if pos<controls:
            row['A_replay']=replay(path,coords,ctx.bind(p),workers=2)
            row['B_replay']=replay(path,coords,ctx.bind(p,independent=True),workers=2)
        # Same spatially spread control bricks for both unmodified algorithms.
        selected=[coords[i] for i in np.linspace(0,len(coords)-1,64,dtype=int)]
        row['A_stratified_bricks']=replay(path,selected,ctx.bind(p),workers=1)
        row['B_stratified_bricks']=replay(path,selected,ctx.bind(p,independent=True),workers=1)
        atomic_json(result_path,row);rows.append(row)
        atomic_json(folder/'progress.json',{'completed':len(rows),'requested':limit,'last_slot':s['slot']})
    atomic_json(folder/'summary.json',{'complete':True,'scenes':rows,'frozen_baseline_unchanged':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=20);p.add_argument('--controls',type=int,default=3);p.add_argument('--label',default='verification');a=p.parse_args();run(a.limit,a.controls,a.label)
