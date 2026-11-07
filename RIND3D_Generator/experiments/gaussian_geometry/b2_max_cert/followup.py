"""Fresh, cost-unselected residual-ray holdout for the endpoint-probe ablation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ..models import load,record
from ..study import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('outputs/gaussian-b2-max-cert-v1'))
    p.add_argument('--input',type=Path,default=Path('outputs/gaussian-geometry-v1/distribution-v2'));a=p.parse_args();output=a.root/'followup-fixture-v1/frozen-rays.json'
    if output.exists():raise ValueError('Followup holdout already frozen')
    excluded=[]
    for file in [a.root/'frozen-rays.json',a.root/'ordinary-fixture-v1/frozen-rays.json']:excluded+=json.loads(file.read_text())['rays']
    rays=[];scenes=[]
    for ordinal in (0,7,14,21):
        case=f'gaussian_sum-{ordinal:03d}';n=128 if ordinal in (7,14) else 64;method='A_fixed7' if n==128 else 'A_fixed6'
        trace=Path('outputs/gaussian-b2-angular-v1')/('depth7-scales-v3' if n==128 else 'four64-v1')/f'{case}-{n}-{method}-independent.npz'
        f=np.load(trace);tag=f['evaluation_path'].ravel();scene=load(a.input/case/'geometry.json');gid=record(scene)['geometry_id']
        source=json.loads((a.input/case/'result.json').read_text())['source_xyz_synthetic'];indices=[r['linear_ZYX_index'] for r in excluded if r['case']==case and r['resolution']==n]
        rng=np.random.default_rng(2026110617+ordinal)
        for state,count,label in [(6,96,'MISS'),(7,32,'HIT')]:
            pool=np.setdiff1d(np.flatnonzero(tag==state),indices);chosen=np.sort(rng.choice(pool,size=count,replace=False))
            for i in chosen:
                x=(np.array(np.unravel_index(i,(n,n,n)))[::-1]+.5)/n;identity=hashlib.sha256(np.asarray([*source,*x],dtype='<f8').tobytes()+gid.encode()).hexdigest()
                rays.append({'ray_id':identity,'case':case,'resolution':n,'linear_ZYX_index':int(i),'source_xyz':source,'receiver_xyz':x.tolist(),
                             'expected':label,'strata':['uniform_residual_'+label],'split':'followup_holdout'})
        scenes.append({'case':case,'geometry_id':gid,'trace_archive_SHA256':hashlib.sha256(trace.read_bytes()).hexdigest()})
    assert not ({r['ray_id'] for r in rays}&{r['ray_id'] for r in excluded})
    atomic_json(output,{'schema':'b2-max-ray-fixtures-v1','selection':'Uniform independent-renderer residual indices, no cost selection; seed 2026110617+ordinal; exclude both prior corpora. Frozen before E timing.',
                        'scenes':scenes,'rays':rays,'existing_geometry_holdout_not_unseen_geometry_generalization':True})
    print(json.dumps({'new_holdout_queries':len(rays),'no_previous_query_overlap':True}))


if __name__=='__main__':main()
