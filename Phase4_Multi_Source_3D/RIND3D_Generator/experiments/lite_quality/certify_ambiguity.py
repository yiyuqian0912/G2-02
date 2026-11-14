"""Independently recheck small-radius source covers against original full FME."""
import json,time
from pathlib import Path
import numpy as np
from rind3d.pack import RINDPack,atomic_json
from rind3d.geometry import load_geometry
from rind3d.rendering.context import GeometryContext
from rind3d.windows import Window,excludes
from rind3d.observation_research import transition_observations
from rind3d.inverse_research import InverseConstraints,explore
from rind3d.inverse_certificate import verify
from .audit import BASE,OUT

def run():
    records=json.loads((OUT/'ambiguity-transition'/'summary.json').read_text())['records'];dest=OUT/'inverse-outer-certificates';dest.mkdir(exist_ok=True);summaries=[]
    for r in records:
        if not any(v['no_source_at_minimum_separation_certified'] for v in r['variants']):continue
        row=r['window'];path=BASE/'scenes'/row['scene_id'];meta=json.loads((path/'verification.json').read_text());ctx=GeometryContext(load_geometry(path/'geometry.npz'),meta['numerical_policy']);truth=np.fromfile(path/'driver.f64',dtype='<f8');w=Window(tuple(row['origin']),row['side'])
        with RINDPack(path/'response.rindb',verify_file=True) as y,RINDPack(path/'solid.rindb',verify_file=True) as m:points,labels,observations=transition_observations(y,m,w,count=64,seed=17)
        constraints=InverseConstraints(ctx,points,labels);_,outer=explore(constraints,truth,w,seconds=30,max_regions=20000,max_depth=10,seed=23,separation=0,record_rejections=True)
        checked=verify(ctx,points,labels,truth,w,outer,separation=0)
        if not checked['no_source_at_distance_32_over_1024_or_more']:raise AssertionError('Independent source separation claim did not survive')
        artifact={'window':row,'true_source':truth.tolist(),'observation_points':points.tolist(),'labels':labels.tolist(),'observations':observations,'outer':outer,'independent_verification':checked}
        output=dest/(row['scene_id']+'-'+str(row['window_id'])+'.json');atomic_json(output,artifact);summaries.append({'family':row['family'],'scene_id':row['scene_id'],'window_id':row['window_id'],**checked})
    atomic_json(dest/'summary.json',{'complete':True,'cases':summaries,'continuous_uniqueness_never_claimed':True,'production_scenes_modified':False})

if __name__=='__main__':run()
