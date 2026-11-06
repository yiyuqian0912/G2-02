"""Controlled continuous branching/connectivity witness, not voxel topology."""
import argparse
import json
from pathlib import Path

from flint import ctx,fmpq
import numpy as np

from experiments.gaussian_geometry.models import Kernel,KernelScene,Threshold,dot,rational,save
from experiments.gaussian_geometry.queries import FieldOracle,qrange,ARB_LOCK
from experiments.gaussian_geometry.reference import Reference
from experiments.gaussian_geometry.study import atomic_json


def prove():
    centers=np.array([[.5,.5,.5],[.65,.5,.5],[.35,.5,.5],[.5,.65,.5],
                      [.5,.35,.5],[.61,.76,.5],[.39,.76,.5]])
    sigma=.06;kernels=[Kernel(c,np.eye(3)*sigma) for c in centers]
    union=KernelScene('ellipsoid_union',kernels,Threshold('exp',-.5))
    summed=KernelScene('gaussian_sum',kernels,Threshold('exp',-.5))
    for i,a in enumerate(centers):
        for b in centers[i+1:]:
            d=tuple(rational(y)-rational(x) for x,y in zip(a,b))
            if dot(d,d)<=(2*rational(sigma))**2:raise AssertionError('Individual spheres are not disjoint')
    oracle=FieldOracle(summed);reference=Reference(summed);rows=[]
    for start,stop in [(0,1),(0,2),(0,3),(0,4),(3,5),(3,6)]:
        a,b=centers[start],centers[stop];coefficients=[k.coefficients(a,b) for k in kernels]
        minimum=float('inf');mpfr_min=float('inf')
        for i in range(32):
            left,right=fmpq(i,32),fmpq(i+1,32)
            with ARB_LOCK,ctx.workprec(192):
                score=oracle._score_ranges([qrange(c,left,right)[:2] for c in coefficients])
                if not score>0:raise AssertionError('Continuous bridge not proved')
                minimum=min(minimum,float(score.lower()))
            from fractions import Fraction as Q
            x=[Q(float(v))+(Q(float(w))-Q(float(v)))*Q(i,32) for v,w in zip(a,b)]
            y=[Q(float(v))+(Q(float(w))-Q(float(v)))*Q(i+1,32) for v,w in zip(a,b)]
            lo,hi=reference.bounds(tuple(min(v,w) for v,w in zip(x,y)),tuple(max(v,w) for v,w in zip(x,y)))
            if lo<=0:raise AssertionError('Independent continuous bridge not proved')
            mpfr_min=min(mpfr_min,float(lo))
        rows.append({'centers':[start,stop],'closed_parameter_intervals':32,
                     'Arb_minimum_strict_lower_margin_display':minimum,'MPFR_minimum_strict_lower_margin_display':mpfr_min,
                     'all_continuous_points_on_connection_certified_solid':True})
    return union,summed,{'scope':'exact disjointness plus two independent continuous whole-segment lower enclosures',
                         'union_exact_components':7,'sum_all_seven_centers_in_one_connected_component':True,
                         'sum_total_component_count_not_claimed':True,'edges':rows,
                         'containment':oracle.containment(.025),'READY_published':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);union,summed,result=prove()
    save(args.output/'independent-levels.json',union);save(args.output/'summed-kernels.json',summed)
    atomic_json(args.output/'proof.json',result);print(json.dumps(result))


if __name__=='__main__':main()
