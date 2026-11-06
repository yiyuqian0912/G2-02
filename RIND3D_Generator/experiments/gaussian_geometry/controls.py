"""Unchanged convex-family controls; new synthetic geometries, no v0.1 reads."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from rind3d.lite_generators import generate
from rind3d.rendering.context import GeometryContext
from rind3d.lite_acceptance import verify_geometry_proof,sample_source

from .study import atomic_json,quality


class SampledControl:
    def __init__(self,context):self.context=context
    def approx_values(self,points):return self.context.contains(points).astype(float)-.5


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);cases=[]
    for family in ('rotated_boxes','corridors','multiscale_convex'):
        for ordinal in range(4):
            path=args.output/f'{family}-{ordinal}.json'
            if path.exists():cases.append(json.loads(path.read_text()));continue
            began=time.perf_counter();cpu=time.process_time();bundle,recipe=generate(20261105,ordinal,family)
            context=GeometryContext(bundle);proof=context.geometry_proof();verify_geometry_proof(bundle,proof)
            source,sreport=sample_source(context,20261105,ordinal)
            row={'family':family,'ordinal':ordinal,'geometry_id':context.geometry_id,
                 'setup_seconds':time.perf_counter()-began,'objects':len(recipe['objects']),
                 'actual_declared_planes':sum(o['declared_planes'] for o in recipe['objects']),
                 'quality':quality(SampledControl(context),64),'fields':[]}
            for n in (32,64,128):
                values=[];outputs=[]
                for independent in (False,True):
                    start=time.perf_counter();c=time.process_time();backend=context.bind(source,independent=independent)
                    before=dict(backend.work);y=np.zeros((n,)*3,dtype=bool);m=np.zeros_like(y);covered=0
                    side=min(64,n)
                    from itertools import product
                    for z,j,i in product(range(0,n,side),repeat=3):
                        yy,mm,cc=backend.render((i,j,z),side,n)
                        region=(slice(z,z+side),slice(j,j+side),slice(i,i+side))
                        y[region]=yy;m[region]=mm;covered+=cc
                    outputs.append((y,m));values.append({'backend':'B' if independent else 'A',
                        'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-c,
                        'certificate_coverage':covered/n**3,'work':{k:v-before.get(k,0) for k,v in backend.work.items()}})
                mismatch=sum(int(np.count_nonzero(a!=b)) for a,b in zip(outputs[0],outputs[1]))
                assert not mismatch
                row['fields'].append({'resolution':n,'complete':True,'mismatches':mismatch,'backends':values})
            row.update(total_seconds=time.perf_counter()-began,cpu_seconds=time.process_time()-cpu,
                       no_existing_dataset_used=True,READY_published=False)
            atomic_json(path,row);cases.append(row)
            print(json.dumps({'family':family,'ordinal':ordinal,'seconds':row['total_seconds']}),flush=True)
    atomic_json(args.output/'summary.json',{'cases':cases,'scope':'12 deterministic new synthetic control geometries; no production'})


if __name__=='__main__':main()
