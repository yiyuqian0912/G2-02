"""Continuous B2 fusion, nonconvexity, free passage and separation witnesses."""
import argparse
import json
from pathlib import Path

import numpy as np

from ..models import Kernel,KernelScene,Threshold,save
from ..queries import FieldOracle
from ..study import atomic_json
from .api import NativeOracle


CENTERS=np.array([[.5,.5,.5],[.65,.5,.5],[.35,.5,.5],[.5,.65,.5],
                  [.5,.35,.5],[.61,.76,.5],[.39,.76,.5]])
EDGES=[(0,1),(0,2),(0,3),(0,4),(3,5),(3,6)]


def bridge(scale=1.):
    centers=.5+scale*(CENTERS-.5)
    return KernelScene('gaussian_sum',[Kernel(c,np.eye(3)*(.06*scale)) for c in centers],Threshold('exp',-.5),
                       {'controlled_fixture':'continuous_B2_branch','scale':scale,'ready_capable':False})


def prove(scene,scale):
    a=NativeOracle(scene);b=NativeOracle(scene,independent=True);old=FieldOracle(scene);certificates=[]
    for start,stop in EDGES:
        p,q=scene.kernels[start].center,scene.kernels[stop].center
        for i in range(32):
            # Enclose exact rational-affine endpoints. A point HIT would not
            # suffice: every closed subsegment needs a whole SOLID box proof.
            from fractions import Fraction as F
            exact=[[F(float(v))+(F(float(w))-F(float(v)))*F(j,32) for v,w in zip(p,q)] for j in (i,i+1)]
            lo=np.array([np.nextafter(float(min(x,y)),-np.inf) for x,y in zip(*exact)])
            hi=np.array([np.nextafter(float(max(x,y)),np.inf) for x,y in zip(*exact)])
            assert a.classify_box(lo,hi).status==b.classify_box(lo,hi).status==old.classify_box(lo,hi).status=='SOLID'
        certificates.append({'edge':[start,stop],'closed_solid_box_cover_intervals':32})
    left=scene.kernels[5].center;right=scene.kernels[6].center;middle=(left+right)/2
    from fractions import Fraction as F
    exact_middle=[(F(float(x))+F(float(y)))/2 for x,y in zip(left,right)]
    ml=np.array([np.nextafter(float(x),-np.inf) for x in exact_middle]);mh=np.array([np.nextafter(float(x),np.inf) for x in exact_middle])
    for e in (a,b,old):
        assert e.membership(left).status==e.membership(right).status=='SOLID'
        assert e.classify_box(ml,mh).status=='FREE'
    lo=.5+scale*(np.array([.4975,.76,.4975])-.5);hi=.5+scale*(np.array([.5025,.95,.5025])-.5)
    assert all(e.classify_box(lo,hi).status=='FREE' for e in (a,b,old))
    midpoint=(scene.kernels[3].center+scene.kernels[5].center)/2;radius=scale*2**-10
    assert all(e.classify_box(np.nextafter(midpoint-radius,-np.inf),np.nextafter(midpoint+radius,np.inf)).status=='SOLID' for e in (a,b,old))
    fusion=(scene.kernels[0].center+scene.kernels[1].center)/2
    fusion_lo=np.nextafter(fusion-radius,-np.inf);fusion_hi=np.nextafter(fusion+radius,np.inf)
    assert all(e.classify_box(fusion_lo,fusion_hi).status=='SOLID' for e in (a,b,old))
    from ..queries import qbox,point
    assert all(qbox(k,point(fusion_lo),point(fusion_hi))[0]>1 for k in scene.kernels)
    return {'scale':scale,'all_seven_centers_connected':True,'connections':certificates,
            'nonconvexity_witness':{'solid_points':[left.tolist(),right.tolist()],'free_midpoint_display':middle.tolist(),
                                   'exact_midpoint_FREE_enclosure':[ml.tolist(),mh.tolist()]},
            'closed_free_passage_box':{'lo':lo.tolist(),'hi':hi.tolist(),'narrow_width_world':float(hi[0]-lo[0])},
            'positive_width_bridge_cube':{'center':midpoint.tolist(),'half_side':radius},
            'fusion_only_solid_box_outside_every_individual_level':{'lo':fusion_lo.tolist(),'hi':fusion_hi.tolist()},
            'global_topology_not_claimed':True,'three_numerical_paths_agree':True}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);rows=[]
    for scale in (1.,.125):
        scene=bridge(scale);result=prove(scene,scale);save(args.output/f'branch-{scale}.json',scene);rows.append(result)
    scene=KernelScene('gaussian_sum',[Kernel([x,.5,.5],np.eye(3)*.04) for x in (.25,.75)],Threshold('exp',-.5))
    assert FieldOracle(scene).containment(.025)['status']=='CERTIFIED'
    for e in (NativeOracle(scene),NativeOracle(scene,independent=True),FieldOracle(scene)):
        assert e.membership([.25,.5,.5]).status==e.membership([.75,.5,.5]).status=='SOLID'
        assert e.classify_box([.5,0,0],[.5,1,1]).status=='FREE'
    save(args.output/'separated.json',scene)
    atomic_json(args.output/'proofs.json',{'branching':rows,'separated':{
         'at_least_two_continuous_components':True,'proof':'solid centers on opposite sides of a wholly FREE separating plane; entire solid strictly world-contained',
         'exact_total_component_count_not_claimed':True},'production':False})


if __name__=='__main__':main()
