"""Matched outer-support/proxy ablation on frozen new synthetic geometries."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import qmc

from experiments.gaussian_geometry.models import load
from experiments.gaussian_geometry.queries import FieldOracle
from experiments.gaussian_geometry.render import small_field
from experiments.gaussian_geometry.study import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();rows=[]
    for model in ('ellipsoid_union','gaussian_sum'):
        for ordinal in (0,7,14,21):
            case=f'{model}-{ordinal:03d}';path=args.input/case
            scene=load(path/'geometry.json');source=json.loads((path/'result.json').read_text())['source_xyz_synthetic']
            points=qmc.Sobol(3,scramble=True,seed=731+ordinal).random_base2(5)
            outputs=[];row={'case':case,'rays':len(points),'backends':[]}
            for accelerated in (False,True):
                t=time.perf_counter();cpu=time.process_time()
                oracle=FieldOracle(scene,use_support=accelerated,use_proxies=accelerated)
                setup=time.perf_counter()-t;decisions=[oracle.segment(source,p) for p in points]
                row['backends'].append({'accelerated':accelerated,'setup_wall_seconds':setup,
                    'ray_wall_seconds':time.perf_counter()-t-setup,'cpu_seconds_including_setup':time.process_time()-cpu,
                    'decisions':dict(Counter(d.status for d in decisions)),'work':dict(oracle.work)})
                field=small_field(oracle,source,8,45.)
                row['backends'][-1]['field_8']={k:v for k,v in field.items() if k not in ('response','solid','valid')}
                outputs.append((decisions,field))
            common=sum(a.certified and b.certified for a,b in zip(outputs[0][0],outputs[1][0]))
            mismatches=sum(a.certified and b.certified and a.status!=b.status for a,b in zip(outputs[0][0],outputs[1][0]))
            a,b=outputs[0][1],outputs[1][1];valid=a['valid']&b['valid']
            field_mismatch=int(np.count_nonzero((a['response']!=b['response'])&valid)+np.count_nonzero((a['solid']!=b['solid'])&valid))
            assert mismatches==field_mismatch==0
            row.update(compared_resolved_rays=int(common),ray_disagreements=int(mismatches),
                       compared_field_voxels=int(valid.sum()),field_bit_disagreements=field_mismatch)
            rows.append(row)
    atomic_json(args.output,{'cases':rows,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            'claim':'identical inputs and proof budgets; outer supports retain all Gaussian tail contributions'})


if __name__=='__main__':main()
