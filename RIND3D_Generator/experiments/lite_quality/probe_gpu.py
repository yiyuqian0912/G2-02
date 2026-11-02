"""Batched reverse-visibility screening using the unchanged certified kernel.

One probe is the source of an inverse cone; all source proposals are GPU
receivers. This is screening only. UNKNOWN is resolved by exact projection.
"""
import json,time
from pathlib import Path
import numpy as np
from rind3d.geometry import load_geometry
from rind3d.rendering.backends import BackendFactory
from rind3d.rendering.cuda_backend import DeviceGeometry,encode
from rind3d.windows import Window,sobol_points,excludes
from rind3d.pack import RINDPack,atomic_json
from rind3d.observation_research import sparse_observations
from rind3d.inverse_research import InverseConstraints
from .audit import BASE,OUT

def run():
    scenes=json.loads((BASE/'report.json').read_text())['scenes'];rows=[]
    with BackendFactory() as f:
        cold=time.perf_counter();runtime=f._cuda()
        if runtime is None:raise RuntimeError('Actual CUDA hardware required for this explicit experiment')
        cold=time.perf_counter()-cold
        for slot in (0,4,50):
            s=next(s for s in scenes if s['slot']==slot);path=BASE/'scenes'/s['scene_id'];meta=json.loads((path/'verification.json').read_text());ctx=f.context(load_geometry(path/'geometry.npz'),meta['numerical_policy']);truth=np.asarray(s['source'])
            raw=np.fromfile(path/'window_records.bin',dtype='<u2').reshape(-1,4)[0];w=Window(tuple(map(int,raw[:3])),int(raw[3]))
            with RINDPack(path/'response.rindb',verify_file=True) as y,RINDPack(path/'solid.rindb',verify_file=True) as m:points,labels=sparse_observations(y,m,w,count=16,seed=8)
            pool=sobol_points(106,slot,812,512);pool=pool[~ctx.contains(pool)];pool=pool[[excludes(w,p,1024) for p in pool]]
            ctx._cache.clear();t=time.perf_counter();c=time.process_time();expected=np.array([ctx.bind(p).response(p,points) for p in pool]);cpu_wall=time.perf_counter()-t;cpu_seconds=time.process_time()-c
            t=time.perf_counter();c=time.process_time();constraints=InverseConstraints(ctx,points,labels);g=DeviceGeometry(ctx,runtime);buffers=[];before=dict(runtime.timings)
            try:
                proposal_buffer=runtime.upload(np.ascontiguousarray(pool,dtype='<f8'));buffers.append(proposal_buffer)
                yy=runtime.alloc(len(pool));mm=runtime.alloc(len(pool));buffers.extend((yy,mm));columns=[];unknown=0
                setup=time.perf_counter()-t;kernel_start=time.perf_counter()
                for model in constraints.models:
                    planes,offsets=encode(model.planes);pr=runtime.upload(planes);po=runtime.upload(offsets)
                    try:
                        runtime.launch('points',((len(pool)+255)//256,1,1),(256,1,1),[g.rows,g.offsets,pr,po,len(ctx.bodies),proposal_buffer,len(pool),yy,mm])
                        actual=runtime.download(yy,(len(pool),),np.int8);solid=runtime.download(mm,(len(pool),),np.int8)
                        if np.any(solid==1):raise AssertionError('Certified FREE source classified solid by GPU')
                        missing=np.flatnonzero((actual<0)|(solid<0));unknown+=len(missing)
                        if len(missing):actual[missing]=model.response(model.source,pool[missing])
                        columns.append(actual.astype(bool))
                    finally:runtime.free(pr);runtime.free(po)
                actual=np.array(columns).T
                if not np.array_equal(actual,expected):raise AssertionError('GPU inverse probe matrix differs from independent clipping')
                rows.append({'slot':slot,'scene_id':s['scene_id'],'family':s['family'],'proposals':len(pool),'exact_probes':len(points),'all_matrix_entries_match':True,
                             'cpu_wall_seconds':cpu_wall,'cpu_seconds':cpu_seconds,'gpu_total_wall_seconds':time.perf_counter()-t,'gpu_total_cpu_seconds':time.process_time()-c,
                             'inverse_setup_seconds':setup,'probe_loop_including_encoding_transfer_seconds':time.perf_counter()-kernel_start,'unknown_fallback_entries':unknown,
                             'runtime_timings':{k:v-before[k] for k,v in runtime.timings.items()},'known_owned_peak_vram_bytes':runtime.peak_allocated})
            finally:
                for p in buffers:runtime.free(p)
                g.close()
        atomic_json(OUT/'gpu-probe-ablation.json',{'complete':True,'cold_cuda_seconds':cold,'rows':rows,'source_screening_only':True,'full_A_B_candidate_verification_still_required':True,
                                                   'same_existing_kernel_no_numeric_changes':True,'runtime':f.report()})

if __name__=='__main__':run()
