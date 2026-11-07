"""Freeze ordinary FREE receiver controls, independently of residual selection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ..models import load,record
from ..study import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--input',type=Path,default=Path('outputs/gaussian-geometry-v1/distribution-v2'))
    p.add_argument('--traces',type=Path,default=Path('outputs/gaussian-b2-angular-v1'));a=p.parse_args()
    if (a.output/'frozen-rays.json').exists():raise ValueError('Keep frozen controls immutable')
    rays=[];scenes=[]
    for ordinal in (0,7,14,21):
        case=f'gaussian_sum-{ordinal:03d}';n=128 if ordinal in (7,14) else 64
        method='A_fixed7' if n==128 else 'A_fixed6';directory=a.traces/('depth7-scales-v3' if n==128 else 'four64-v1')
        archive=directory/f'{case}-{n}-{method}-independent.npz';f=np.load(archive);assert f['valid'].all()
        scene=load(a.input/case/'geometry.json');source=json.loads((a.input/case/'result.json').read_text())['source_xyz_synthetic']
        free=np.flatnonzero(~f['solid'].astype(bool).ravel());rng=np.random.default_rng(2026110600+ordinal)
        chosen=np.sort(rng.choice(free,size=128,replace=False))
        gid=record(scene)['geometry_id']
        for index in chosen:
            x=(np.array(np.unravel_index(index,(n,n,n)))[::-1]+.5)/n
            identity=hashlib.sha256(np.asarray([*source,*x],dtype='<f8').tobytes()+gid.encode()).hexdigest()
            rays.append({'ray_id':identity,'case':case,'resolution':n,'linear_ZYX_index':int(index),'source_xyz':source,'receiver_xyz':x.tolist(),
                         'expected':'MISS' if f['response'].ravel()[index] else 'HIT','strata':['ordinary_uniform_FREE_receiver'],'split':'ordinary_control'})
        scenes.append({'case':case,'geometry_id':gid,'trace_archive_SHA256':hashlib.sha256(archive.read_bytes()).hexdigest()})
    atomic_json(a.output/'frozen-rays.json',{'schema':'b2-max-ray-fixtures-v1','selection':'Uniform FREE voxel samples, seed 2026110600+ordinal, no residual-cost preference','scenes':scenes,'rays':rays})
    print(json.dumps({'ordinary_FREE_receivers':len(rays),'selection_before_control_benchmark':True}))


if __name__=='__main__':main()
