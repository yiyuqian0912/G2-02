"""One-factor Gaussian-sum controls; quality estimates are not topology proofs."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import qmc

from experiments.gaussian_geometry.models import load,Kernel,KernelScene,Threshold
from experiments.gaussian_geometry.queries import make_oracle
from experiments.gaussian_geometry.reference import Reference
from experiments.gaussian_geometry.study import atomic_json,quality


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();path=args.input/'gaussian_sum-007'
    base=load(path/'geometry.json');source=np.array(json.loads((path/'result.json').read_text())['source_xyz_synthetic'])
    anchors={i:np.array(c['anchor']) for i,c in enumerate(base.metadata['clusters'])}
    points=qmc.Sobol(3,scramble=True,seed=8117).random_base2(5);rows=[]
    subset_order=np.random.default_rng(128).permutation(len(base.kernels))
    for parameter,values in [('cluster_displacement_factor',[.5,1.,1.5]),('kernel_matrix_factor',[.75,1.,1.25]),
                             ('threshold_factor',[.7,1.,1.4]),('nested_kernel_fraction',[1/3,2/3,1.])]:
        for value in values:
            ks=[];chosen=set(subset_order[:max(1,round(len(base.kernels)*value))]) if parameter=='nested_kernel_fraction' else set(range(len(base.kernels)))
            for i,k in enumerate(base.kernels):
                if i not in chosen:continue
                center=anchors[k.cluster]+value*(k.center-anchors[k.cluster]) if parameter=='cluster_displacement_factor' else k.center
                matrix=k.matrix*value if parameter=='kernel_matrix_factor' else k.matrix
                ks.append(Kernel(center,matrix,k.weight,k.cluster,k.scale_class))
            tau=base.threshold.value*(value if parameter=='threshold_factor' else 1.)
            scene=KernelScene('gaussian_sum',ks,Threshold('binary64',tau));oracle=make_oracle(scene)
            row={'parameter':parameter,'value':value,'kernel_count':len(ks),'quality':quality(scene,64),
                 'containment':oracle.containment(.025),'publication':'EXPERIMENTAL_ONLY'}
            if oracle.source_clearance(source).status=='FREE':
                receivers=[v for v in points if oracle.membership(v).status=='FREE'];t=time.perf_counter()
                decisions=[oracle.segment(source,v) for v in receivers]
                row.update(physical_ray_seconds=time.perf_counter()-t,ray_count=len(receivers),
                           shadow_count=sum(d.status=='HIT' for d in decisions),unresolved=sum(not d.certified for d in decisions))
                reference=Reference(scene);disagree=unresolved=0
                for v,d in zip(receivers[:8],decisions):
                    other=reference.segment(source,v)
                    if d.certified and other.certified:disagree+=d.status!=other.status
                    else:unresolved+=1
                row.update(independent_disagreements=disagree,independent_unresolved=unresolved)
                assert not disagree
            else:row['source_status']='same fixed source not certified admissible; no replacement'
            rows.append(row)
    atomic_json(args.output,{'base_case':'gaussian_sum-007','cases':rows,
              'scope':'one fixed proposal recipe; no density recalibration, threshold repair, production acceptance or general causal claim',
              'kernel_subset_policy':'nested seeded permutation, preserving retained kernels exactly; no duplicate kernels',
              'no_production':True})


if __name__=='__main__':main()
