"""Frozen hard/region controls and a selected complete old-Arb comparison."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ..models import load,record,Kernel,KernelScene,Threshold
from ..queries import FieldOracle,Limits
from ..render import small_field
from ..study import atomic_json
from ..b2_native.api import NativeOracle
from ..b2_native.controls import bridge
from .api import RegionOracle
from .build import sources


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--previous',type=Path,default=Path('outputs/gaussian-b2-native-v1'))
    p.add_argument('--input',type=Path,default=Path('outputs/gaussian-geometry-v1/distribution-v2'))
    p.add_argument('--field32',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    rows=[];root=args.previous/'adversarial-v2'
    for path in sorted(x for x in root.iterdir() if x.is_dir()):
        scene=load(path/'geometry.json');q=np.load(path/'queries.npz');source=q['source'];points=q['receivers']
        ids=np.unique(np.linspace(0,len(points)-1,min(12,len(points)),dtype=int));regions=[]
        for i in ids:
            for radius in ((0.,) if len(points)==1 else (0.,1/1024,1/64)):
                regions.append((np.maximum(0,points[i]-radius),np.minimum(1,points[i]+radius)))
        old=FieldOracle(scene,Limits(max_nodes=512,seconds=.15));samplings={}
        for ordinal,(lo,hi) in enumerate(regions):
            sample=np.vstack([lo,hi,(lo+hi)/2]);samplings[ordinal]=[old.segment(source,x) for x in sample]
        row={'fixture':path.name,'geometry_id':record(scene)['geometry_id'],'regions':len(regions),'methods':{}}
        for backend in ('primary','independent'):
            for mode in (1,2,3,4):
                prep=time.perf_counter()
                engine=RegionOracle(scene,independent=backend=='independent',mode=mode,max_nodes=512,query_seconds=.5)
                setup=time.perf_counter()-prep
                labels=Counter();work=Counter();start=time.perf_counter();disagree=0;checks=0
                for i,(lo,hi) in enumerate(regions):
                    (y,m),counts=engine.region(source,lo,hi);labels[y]+=1;work.update(counts)
                    if y>=0:
                        for d in samplings[i]:
                            if d.certified:checks+=1;disagree+=d.status!=('MISS' if y==1 else 'HIT')
                assert not disagree,(path.name,backend,mode)
                duration=time.perf_counter()-start
                row['methods'][f'{backend}_{mode}']={'wall_seconds':duration,'setup_seconds':setup,'cold_total_seconds':setup+duration,'visibility_counts':dict(labels),
                     'sampled_old_Arb_ray_checks':checks,'certified_disagreements':disagree,'work':dict(work)}
        rows.append(row);atomic_json(args.output/f'{path.name}.json',row)
        print(json.dumps({'fixture':path.name,'regions':len(regions),'disagreements':0}),flush=True)
    s=load(args.input/'gaussian_sum-007/geometry.json');source=json.loads((args.input/'gaussian_sum-007/result.json').read_text())['source_xyz_synthetic'];receiver=(np.array([23,38,45])+.5)/64
    hard=[]
    for backend,e in [('old_Arb',FieldOracle(s,Limits(max_nodes=65536,seconds=10.))),('primary',NativeOracle(s)),('independent',NativeOracle(s,independent=True))]:
        start=time.perf_counter();d=e.segment(source,receiver);hard.append({'backend':backend,'status':d.status,'nodes':d.nodes,'wall_seconds':time.perf_counter()-start});assert d.status=='MISS'
    s=load(args.input/'gaussian_sum-014/geometry.json');source=json.loads((args.input/'gaussian_sum-014/result.json').read_text())['source_xyz_synthetic']
    start=time.perf_counter();old=small_field(FieldOracle(s),source,32,90.);saved=np.load(args.field32)
    assert old['complete'] and saved['valid'].all()
    assert all(np.array_equal(old[k],saved[k]) for k in ('response','solid'))
    arb32={'complete':True,'voxels':32**3,'disagreements':0,'wall_seconds':time.perf_counter()-start,'new_field_file_SHA256':hashlib.sha256(args.field32.read_bytes()).hexdigest()}
    fusion=[];s=bridge();source=[.575,.5,.05]
    tile_lo=(np.array([586,506,960])+.5)/1024;tile_hi=(np.array([586,506,960])+8-.5)/1024
    individual=[]
    for i,kernel in enumerate(s.kernels):
        single=KernelScene('gaussian_sum',[kernel],s.threshold)
        for independent in (False,True):
            result,work=RegionOracle(single,independent=independent,mode=2).region(source,tile_lo,tile_hi)
            assert result==(1,0),(i,result)
        individual.append({'kernel':i,'whole_receiver_volume_FREE_under_individual_term':True})
    for backend in ('primary','independent'):
        e=RegionOracle(s,independent=backend=='independent');v=e.tile(source,1024,[586,506,960],8,30.)
        assert v['complete'] and not v['solid'].any() and not v['response'].any()
        fusion.append({'backend':backend,'complete':True,'FREE_shadow_voxels':512,'work':v['work'],'wall_seconds':v['wall_seconds']})
    near=KernelScene('gaussian_sum',[Kernel([.5]*3,np.eye(3)*.125)],Threshold('exp',-.5))
    near_source=[.625+2**-10,.503,.503];near_rows=[]
    for backend in ('primary','independent'):
        old=NativeOracle(near,independent=backend=='independent');assert old.source_clearance(near_source).status=='FREE'
        state=old.classify_box([.625,.5,.5],[.75,.625,.625]).status;assert state=='UNKNOWN'
        fields=[]
        for mode in (1,5):
            e=RegionOracle(near,independent=backend=='independent',mode=mode,cover_depth=3)
            v=e.field(near_source,16,30);assert v['complete'];fields.append(v)
            near_rows.append({'backend':backend,'mode':mode,'source_clearance':'FREE','coarse_source_box_state':state,
                              'wall_seconds':v['wall_seconds'],'cover_seconds':v['cover_seconds'],'work':v['work']})
        assert all(np.array_equal(fields[0][k],fields[1][k]) for k in ('response','solid'))
    atomic_json(args.output/'summary.json',{'sources':sources(),'fixed_regions':rows,'known_case7_MISS':hard,'complete_Arb32':arb32,
                 'fusion_only_ray_tile':fusion,'individual_kernel_volume_proofs':individual,
                 'legal_source_in_UNKNOWN_cover_control':near_rows,'numerical_failures_are_not_physical_rejections':True,
                 'caution':'Old-Arb point probes are sanity checks only; the sufficient analytic region proof establishes the continuous certificate.'})


if __name__=='__main__':main()
