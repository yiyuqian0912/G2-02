"""Bounded residual-only ablation of ALREADY EXISTING rigorous Taylor settings."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from ..models import load
from ..b2_native.api import NativeOracle
from ..study import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fields',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--input',type=Path,default=Path('outputs/gaussian-geometry-v1/distribution-v2'))
    p.add_argument('--query-seconds',type=float,default=.5);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for directory in args.fields:
        for archive in sorted(directory.glob('*128*independent.npz')):
            if 'A_fixed7' not in archive.name and 'F_proof_reuse' not in archive.name:continue
            case=archive.name.split('-128-')[0];scene=load(args.input/case/'geometry.json')
            source=json.loads((args.input/case/'result.json').read_text())['source_xyz_synthetic']
            arrays=np.load(archive);path=arrays['evaluation_path'];cost=arrays['ray_nanoseconds'];n=path.shape[0]
            selected=[]
            for state in (6,7):
                candidates=np.flatnonzero(path.ravel()==state);order=candidates[np.argsort(cost.ravel()[candidates],kind='stable')]
                if not len(order):continue
                selected.extend(order[-4:].tolist());selected.extend(order[np.linspace(0,len(order)-1,4,dtype=int)].tolist())
            selected=sorted(set(selected));receivers=[(np.array(np.unravel_index(i,path.shape))[::-1]+.5)/n for i in selected]
            expected=['MISS' if path.ravel()[i]==6 else 'HIT' for i in selected]
            for name,taylor,depth in [('off',False,4),('depth2',True,2),('depth4_default',True,4),('depth8',True,8)]:
                for independent in (False,True):
                    checkpoint=args.output/f'{case}-{name}-{"independent" if independent else "primary"}.json'
                    if checkpoint.exists():
                        row=json.loads(checkpoint.read_text())
                        assert row['query_seconds_limit']==args.query_seconds
                        assert row['baseline_archive_SHA256']==hashlib.sha256(archive.read_bytes()).hexdigest()
                        results.append(row);continue
                    engine=NativeOracle(scene,independent=independent,taylor=taylor,taylor_depth=depth,query_seconds=args.query_seconds,max_nodes=16384)
                    wall=time.perf_counter();cpu=time.process_time();rows=[];counts=Counter()
                    for i,x,y in zip(selected,receivers,expected):
                        t=time.perf_counter();decision=engine.segment(source,x);dt=time.perf_counter()-t
                        assert decision.status in (y,'UNKNOWN'),(case,name,decision.status,y)
                        counts[decision.status]+=1;rows.append({'linear_ZYX_index':i,'expected':y,'status':decision.status,'nodes':decision.nodes,'seconds':dt,'reason':decision.reason})
                    result={'case':case,'backend':'independent' if independent else 'primary','method':name,'queries':len(selected),
                            'wall_seconds':time.perf_counter()-wall,'CPU_seconds':time.process_time()-cpu,'status_counts':dict(counts),'queries_detail':rows,'work':dict(engine.work),
                            'baseline_archive_SHA256':hashlib.sha256(archive.read_bytes()).hexdigest(),'no_changed_physical_predicate':True}
                    result['query_seconds_limit']=args.query_seconds
                    results.append(result);atomic_json(args.output/f'{case}-{name}-{result["backend"]}.json',result)
                    print(json.dumps({k:result[k] for k in ('case','backend','method','queries','wall_seconds','status_counts')}),flush=True)
    atomic_json(args.output/'summary.json',{'results':results,'scope':'Bounded settings experiment with frozen canonical implementations. No new Taylor/affine renderer is claimed; UNKNOWN is not a physical MISS. Timings include failed query attempts.'})


if __name__=='__main__':main()
