"""Gated complete B2 fields. Independent arrays are computed, never copied."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

from ..models import load
from ..queries import FieldOracle
from ..render import small_field
from ..study import atomic_json
from .api import NativeOracle
from .build import sources


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--ordinals',nargs='+',type=int,default=[0,7,14,21])
    p.add_argument('--resolutions',nargs='+',type=int,default=[8,32]);p.add_argument('--seconds',type=float,default=120.)
    p.add_argument('--arb-case',type=int,default=14);p.add_argument('--scalar-control',action='store_true');args=p.parse_args()
    if args.resolutions[0]!=8 or any(n not in (8,32,64,128) for n in args.resolutions):raise ValueError('Start at 8 and use explicit small-field gates')
    identity={'native_source':sources(),'input_manifest_sha256':hashlib.sha256((args.input/'run-manifest.json').read_bytes()).hexdigest(),
              'ordinals':args.ordinals,'resolutions':args.resolutions,'seconds_per_field':args.seconds,
              'arb_case':args.arb_case,'scalar_control':args.scalar_control,'production':False}
    identity['driver_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    identity['api_sha256']=hashlib.sha256(Path(__file__).with_name('api.py').read_bytes()).hexdigest()
    args.output.mkdir(parents=True,exist_ok=True);manifest=args.output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=identity:raise ValueError('Field identity changed; use a new output')
    atomic_json(manifest,identity);rows=[]
    for ordinal in args.ordinals:
        case=f'gaussian_sum-{ordinal:03d}';path=args.input/case;target=args.output/case;target.mkdir(exist_ok=True)
        checkpoint=target/'result.json'
        if checkpoint.exists():rows.append(json.loads(checkpoint.read_text()));continue
        scene=load(path/'geometry.json');source=json.loads((path/'result.json').read_text())['source_xyz_synthetic']
        t=time.perf_counter();a=NativeOracle(scene);b=NativeOracle(scene,independent=True);setup=time.perf_counter()-t
        assert a.source_clearance(source).status==b.source_clearance(source).status=='FREE'
        row={'case':case,'setup_both_engines_seconds':setup,'fields':[],'ready_capable':False}
        for n in args.resolutions:
            arrays=[];frow={'resolution':n,'backends':[]}
            for name,e in [('primary',a),('independent',b)]:
                cpu=time.process_time();result=e.field(source,n,args.seconds)
                payload={k:result.pop(k) for k in ('response','solid','valid')};arrays.append(payload)
                archive=target/f'{name}-{n}.npz';np.savez_compressed(archive,**payload)
                result.update(backend=name,cpu_seconds=time.process_time()-cpu,valid_voxels=int(payload['valid'].sum()),
                              storage_bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                              process_peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
                frow['backends'].append(result)
            common=arrays[0]['valid']&arrays[1]['valid'];mismatch=sum(int(np.count_nonzero((arrays[0][k]!=arrays[1][k])&common)) for k in ('response','solid'))
            frow.update(compared_voxels=int(common.sum()),bit_disagreements=mismatch,
                        independently_complete=all(x['complete'] for x in frow['backends']))
            if mismatch:atomic_json(target/'disagreement.json',frow);raise AssertionError('Independent field mismatch')
            if ordinal==args.arb_case and n in (8,32) and frow['independently_complete']:
                t=time.perf_counter();cpu=time.process_time();old=small_field(FieldOracle(scene),source,n,args.seconds)
                common=old['valid']&arrays[0]['valid'];wrong=sum(int(np.count_nonzero((old[k]!=arrays[0][k])&common)) for k in ('response','solid'))
                frow['old_Arb_crosscheck']={'complete':old['complete'],'wall_seconds':time.perf_counter()-t,'cpu_seconds':time.process_time()-cpu,
                     'valid_voxels':int(old['valid'].sum()),'bit_disagreements':wrong,'status':old['status']}
                if wrong:raise AssertionError('Independent Arb mismatch')
            if args.scalar_control and n in (8,32) and frow['independently_complete']:
                scalar=a.field(source,n,args.seconds,scalar=True)
                wrong=sum(int(np.count_nonzero((scalar[k]!=arrays[0][k])&scalar['valid'])) for k in ('response','solid'))
                frow['primary_scalar_control']={'complete':scalar['complete'],'wall_seconds':scalar['wall_seconds'],'bit_disagreements':wrong,'work':scalar['work']}
                if wrong:raise AssertionError('Scalar/hierarchy mismatch')
            row['fields'].append(frow);atomic_json(target/'partial.json',row)
            print(json.dumps({'case':case,'n':n,'both_complete':frow['independently_complete'],
                              'times':[round(x['wall_seconds'],4) for x in frow['backends']]}),flush=True)
            if not frow['independently_complete']:break
        atomic_json(checkpoint,row);rows.append(row)
    atomic_json(args.output/'summary.json',{'cases':rows,'new_READY_scenes':0,'new_1024_fields':0})


if __name__=='__main__':main()
