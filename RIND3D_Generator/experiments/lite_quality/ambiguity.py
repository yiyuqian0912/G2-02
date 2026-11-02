"""Certified sparse-constraint outer bounds; no continuous-uniqueness claims."""
import json,time
from fractions import Fraction as Q
from pathlib import Path
import numpy as np
from rind3d.geometry import load_geometry
from rind3d.rendering.context import GeometryContext
from rind3d.pack import RINDPack,atomic_json
from rind3d.windows import Window
from rind3d.observation_research import sparse_observations,transition_observations
from rind3d.inverse_research import InverseConstraints,explore,max_distance_sq
from .audit import BASE,OUT

def run():
    rows=json.loads((OUT/'inverse-v2'/'summary.json').read_text())['records'];folder=OUT/'ambiguity-transition';folder.mkdir(exist_ok=True);results=[]
    for record in rows:
        row=record['window']
        if row['side']!=256:continue
        target=folder/(row['scene_id']+'-'+str(row['window_id'])+'.json')
        if target.exists():results.append(json.loads(target.read_text()));continue
        path=BASE/'scenes'/row['scene_id'];meta=json.loads((path/'verification.json').read_text());ctx=GeometryContext(load_geometry(path/'geometry.npz'),meta['numerical_policy']);truth=np.fromfile(path/'driver.f64',dtype='<f8');w=Window(tuple(row['origin']),row['side']);variants=[]
        with RINDPack(path/'response.rindb',verify_file=True) as y,RINDPack(path/'solid.rindb',verify_file=True) as m:
            for strategy in ('spatial','transition'):
                t=time.perf_counter()
                if strategy=='spatial':points,labels=sparse_observations(y,m,w,count=64,seed=17);observations={}
                else:points,labels,observations=transition_observations(y,m,w,count=64,seed=17)
                inverse=InverseConstraints(ctx,points,labels);setup=time.perf_counter()-t
                if not inverse.screen(truth[None])[0]:raise AssertionError('Inverse constraints removed the verified true source')
                p,report=explore(inverse,truth,w,seconds=10,max_regions=20000,max_depth=10,seed=23,separation=0)
                cover=report['outer_cells']
                if not any(np.all(truth>=b['lower']) and np.all(truth<=b['upper']) for b in cover):raise AssertionError('Outer-cover construction lost the true source')
                lo=np.min([b['lower'] for b in cover],axis=0);hi=np.max([b['upper'] for b in cover],axis=0);diameter=sum((Q(float(b))-Q(float(a)))**2 for a,b in zip(lo,hi));distance=max(max_distance_sq(b['lower'],b['upper'],truth) for b in cover)
                variants.append({'strategy':strategy,'observations':observations,'setup_seconds':setup,'actual_constraints':len(points),'outer':report,
                                 'outer_bbox_lower':lo.tolist(),'outer_bbox_upper':hi.tolist(),'diameter_squared_upper_bound_exact':str(diameter),
                                 'truth_distance_squared_upper_bound_exact':str(distance),'no_source_at_minimum_separation_certified':distance<Q(1,32)**2,
                                 'continuous_source_uniqueness_proven':False,'full_window_certified_inner_boxes':0,'component_count_proven':False})
        result={'window':row,'variants':variants};atomic_json(target,result);results.append(result)
    atomic_json(folder/'summary.json',{'complete':True,'records':results,'bounds_are_necessary_constraint_outer_bounds':True})

if __name__=='__main__':run()
