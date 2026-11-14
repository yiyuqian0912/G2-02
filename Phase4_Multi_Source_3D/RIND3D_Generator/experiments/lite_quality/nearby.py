"""Full-window witnesses below reference separation: ambiguity research only."""
import json,time
from pathlib import Path
import numpy as np
from rind3d.geometry import load_geometry,Budget
from rind3d.rendering.context import GeometryContext
from rind3d.lite_acceptance import DRIVER_RADIUS,verify_clearance
from rind3d.pack import RINDPack,atomic_json
from rind3d.windows import Window,excludes
from rind3d.enrichment_research import SharedCandidateVerifier
from .audit import BASE,OUT

def run():
    certificates=json.loads((OUT/'inverse-outer-certificates'/'summary.json').read_text())['cases'];records=[]
    for certificate in certificates:
        path=BASE/'scenes'/certificate['scene_id'];meta=json.loads((path/'verification.json').read_text());ctx=GeometryContext(load_geometry(path/'geometry.npz'),meta['numerical_policy']);truth=np.fromfile(path/'driver.f64',dtype='<f8');r=np.fromfile(path/'window_records.bin',dtype='<u2').reshape(-1,4)[certificate['window_id']];w=Window(tuple(map(int,r[:3])),int(r[3]));v=SharedCandidateVerifier(ctx);witnesses=[];start=time.perf_counter()
        with RINDPack(path/'response.rindb',verify_file=True) as y,RINDPack(path/'solid.rindb',verify_file=True) as m:
            response=y.read_window(*w.origin,w.size);mask=m.read_window(*w.origin,w.size)
            for axis in range(3):
                for sign in (-1,1):
                    p=truth.copy();p[axis]+=sign*2**-40
                    if np.array_equal(p,truth) or not excludes(w,p,1024):raise AssertionError('Invalid nearby proposal')
                    proof=ctx.clearance(p,DRIVER_RADIUS);verify_clearance(ctx,p,proof);checks=[];match=True
                    for independent in (False,True):
                        result=v.verify(p,w,response,mask,independent=independent,budget=Budget(60));checks.append(result)
                        if result['status']!='grid_exact':match=False;break
                    witnesses.append({'position':p.tolist(),'distance_from_truth':float(np.linalg.norm(p-truth)),'independently_full_window_equivalent':match,
                                      'does_not_satisfy_reference_minimum_separation':True,'checks':checks})
        records.append({'scene_id':certificate['scene_id'],'window_id':certificate['window_id'],'family':certificate['family'],'witnesses':witnesses,'wall_seconds':time.perf_counter()-start})
    atomic_json(OUT/'nearby-ambiguity-witnesses.json',{'complete':True,'analysis_only_not_dataset_references':True,'records':records,'continuous_neighbourhood_or_inner_box_not_proven':True})

if __name__=='__main__':run()
