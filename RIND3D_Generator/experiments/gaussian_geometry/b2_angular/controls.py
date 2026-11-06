"""Decisive continuous union proofs and frozen B2 numerical controls."""
import argparse
from fractions import Fraction as Q
from itertools import product
import json
import math
from pathlib import Path
import time

import numpy as np
from ..models import load,record,KernelScene
from ..queries import FieldOracle,Limits
from ..study import atomic_json
from ..render import small_field
from ..b2_native.api import NativeOracle
from ..b2_native.controls import bridge
from ..b2_regions.api import RegionOracle
from .fixtures import SOURCE,RECEIVER,INNER_LO,INNER_HI,union_scene
from .api import AngularOracle,extension
from .build import sources


def scrub(row):
    row.pop('tiles',None)
    if not math.isfinite(row['first_hit_upper']):row['first_hit_upper']=None
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--input',type=Path,default=Path('outputs/gaussian-geometry-v1/distribution-v2'))
    p.add_argument('--field32',type=Path,help='Optional complete case-14 comparison against original Arb')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'summary.json').exists():raise ValueError('Preserve measured controls: choose a new versioned output')
    scene=union_scene();proofs=[]
    for independent in (False,True):
        native=NativeOracle(scene,independent=independent)
        assert native.source_clearance(SOURCE).status=='FREE'
        states=[native.classify_box(a,b).status for a,b in zip(INNER_LO,INNER_HI)];assert states==['SOLID','SOLID']
        previous=native.certificate(SOURCE,*RECEIVER);assert previous==(-1,0)
        previous_extra=RegionOracle(scene,independent=independent,mode=4).region(SOURCE,*RECEIVER)[0];assert previous_extra==(-1,0)
        individual=[NativeOracle(KernelScene('gaussian_sum',[k],scene.threshold),independent=independent).classify_box([0]*3,[1]*3).status for k in scene.kernels]
        assert set(individual)=={'FREE'}
        screen=[]
        for fusion in (False,True):
            start=time.perf_counter();row=extension().screen(SOURCE,*RECEIVER,INNER_LO,INNER_HI,independent,fusion=fusion)
            scrub(row);row.update(fusion=fusion,wall_seconds=time.perf_counter()-start)
            assert row['visibility']==(0 if fusion else -1);screen.append(row)
        counterexamples={}
        gap_lo=np.array(INNER_LO);gap_hi=np.array(INNER_HI);gap_hi[0,1]=.5-2**-30;gap_lo[1,1]=.5+2**-30
        for name,source,receiver,lo,hi in [
            ('tiny_gap_2^-30',SOURCE,RECEIVER,gap_lo,gap_hi),
            ('blockers_behind_receivers',SOURCE,([.25,.49,.49],[.3,.51,.51]),INNER_LO,INNER_HI),
            ('cross_chart',[.125]*3,([.75,.75,.13],[.9,.9,.15]),INNER_LO,INNER_HI),
            ('overlap_incomplete',SOURCE,RECEIVER,np.array(INNER_LO)+[0,0,.02],np.array(INNER_HI)-[0,0,.02])]:
            row=scrub(extension().screen(source,*receiver,lo,hi,independent));assert row['visibility']==-1;counterexamples[name]=row
        proofs.append({'backend':'independent' if independent else 'primary','geometry_id':record(scene)['geometry_id'],
                       'inner_boxes_continuously_SOLID':states,'each_individual_kernel_entire_unit_world_FREE':individual,
                       'old_region_certificate':previous,'old_full_sum_extra_certificate':previous_extra,
                       'single_vs_union':screen,'counterexamples':counterexamples})
    # Separate exact rational affine-extrema proof for the complete angular rectangles.
    exact=[]
    for u,rho,a,b in zip(((Q(-1,6),Q(0)),(Q(0),Q(1,6))),(Q(1,4),Q(7,16)),INNER_LO,INNER_HI):
        for du,dv in product(u,(Q(-1,24),Q(1,24))):
            x=(Q(1,8)+rho,Q(1,2)+rho*du,Q(1,2)+rho*dv)
            assert all(Q(float(l))<=v<=Q(float(h)) for l,v,h in zip(a,x,b))
        exact.append({'u_interval':list(map(str,u)),'v_interval':['-1/24','1/24'],'blocker_depth_rho':str(rho),'all_affine_extrema_inside':True})
    original=FieldOracle(scene)
    arb_checks=[]
    for point in product(*zip(*RECEIVER)):
        d=original.segment(SOURCE,point);assert d.status=='HIT';arb_checks.append(d.status)
    hard=[]
    scene7=load(args.input/'gaussian_sum-007/geometry.json');source7=json.loads((args.input/'gaussian_sum-007/result.json').read_text())['source_xyz_synthetic'];receiver7=(np.array([23,38,45])+.5)/64
    for name,engine in [('old_Arb',FieldOracle(scene7,Limits(max_nodes=65536,seconds=10))),('primary',NativeOracle(scene7)),('independent',NativeOracle(scene7,independent=True))]:
        t=time.perf_counter();d=engine.segment(source7,receiver7);assert d.status=='MISS'
        hard.append({'backend':name,'status':d.status,'nodes':d.nodes,'seconds':time.perf_counter()-t})
    fused=[];scene7kernel=bridge();source=[.575,.5,.05];origin=[586,506,960]
    for independent in (False,True):
        native=NativeOracle(scene7kernel,independent=independent);assert native.source_clearance(source).status=='FREE'
        old=RegionOracle(scene7kernel,independent=independent).tile(source,1024,origin,8,30)
        for method in (3,4):
            start=time.perf_counter();v=AngularOracle(scene7kernel,independent=independent,method=method).tile(source,1024,origin,8,30)
            assert v['complete'] and not v['solid'].any() and not v['response'].any()
            assert all(np.array_equal(v[k],old[k]) for k in ('response','solid'))
            fused.append({'backend':'independent' if independent else 'primary','method':method,'complete':True,
                          'FREE_shadowed_voxels':512,'wall_seconds':time.perf_counter()-start,'work':v['work']})
    arb32=None
    if args.field32:
        folder=args.input/'gaussian_sum-014';scene14=load(folder/'geometry.json');source14=json.loads((folder/'result.json').read_text())['source_xyz_synthetic']
        t=time.perf_counter();old=small_field(FieldOracle(scene14),source14,32,90);new=np.load(args.field32)
        assert old['complete'] and new['valid'].all()
        assert all(np.array_equal(old[k],new[k]) for k in ('response','solid'))
        arb32={'complete':True,'voxels':32**3,'bit_disagreements':0,'wall_seconds':time.perf_counter()-t}
    atomic_json(args.output/'summary.json',{'sources':sources(),'continuous_union':proofs,'independent_rational_universal_proof':exact,
                'old_Arb_corner_sanity_only':arb_checks,'known_case7_MISS':hard,'preserved_seven_kernel_fusion_tile':fused,
                'complete_old_Arb32':arb32,
                'new_READY_scenes':0,'scope':'All coordinates here are explicit synthetic controls, not private dataset source coordinates.'})
    print(json.dumps({'continuous_union':'PASS','tiny_gap_and_depth_counterexamples':'UNKNOWN','old_Arb_sanity':'PASS','case7_MISS':'PASS','seven_kernel_tile':'PASS'}),flush=True)


if __name__=='__main__':main()
