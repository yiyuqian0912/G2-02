"""Cold complete-field comparisons on frozen B2 geometries; resumable per field."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

from ..models import load,record
from ..study import atomic_json
from ..b2_native.api import NativeOracle
from .api import RegionOracle
from .build import sources

MODES={'A_scalar':0,'B_old_regions':1,'C_ray_volume':2,'D_fused_shadow':3,'E_combined':4,'F_proof_reuse':5}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('outputs/gaussian-geometry-v1/distribution-v2'))
    p.add_argument('--baseline',type=Path,default=Path('outputs/gaussian-b2-native-v1/fields-v3'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--ordinals',type=int,nargs='+',default=[14])
    p.add_argument('--resolutions',type=int,nargs='+',default=[8,32]);p.add_argument('--modes',nargs='+',choices=MODES,default=list(MODES))
    p.add_argument('--backends',nargs='+',choices=['primary','independent'],default=['primary','independent'])
    p.add_argument('--seconds',type=float,default=180);p.add_argument('--cover-depth',type=int,default=6)
    p.add_argument('--cover-nodes',type=int,default=32769);p.add_argument('--shadow-boxes',type=int,default=256)
    p.add_argument('--fast-exp',action='store_true');p.add_argument('--volume-after-cache',action='store_true')
    p.add_argument('--trace',action='store_true');args=p.parse_args()
    config={k:v for k,v in vars(args).items() if not isinstance(v,Path)}
    identity={'config':config,'sources':sources(),'driver_sha256':sha(Path(__file__)),
              'api_sha256':sha(Path(__file__).with_name('api.py')),'input_manifest_sha256':sha(args.input/'run-manifest.json'),
              'old_fields_manifest_sha256':sha(args.baseline/'manifest.json'),'new_READY_scenes':0}
    args.output.mkdir(parents=True,exist_ok=True);manifest=args.output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=identity:raise ValueError('Changed experiment; use a new versioned output')
    atomic_json(manifest,identity);results=[]
    for ordinal in args.ordinals:
        name=f'gaussian_sum-{ordinal:03d}';scene=load(args.input/name/'geometry.json');original=json.loads((args.input/name/'result.json').read_text())
        source=original['source_xyz_synthetic'];geometry=record(scene)['geometry_id']
        for backend in args.backends:assert NativeOracle(scene,independent=backend=='independent').source_clearance(source).status=='FREE'
        for n in args.resolutions:
            level_complete=True
            offset=ordinal%len(args.modes);order=args.modes[offset:]+args.modes[:offset]
            for mode in order:
                for backend in args.backends:
                    label=f'{name}-{n}-{mode}-{backend}';checkpoint=args.output/f'{label}.json'
                    if checkpoint.exists():
                        row=json.loads(checkpoint.read_text());results.append(row);level_complete &= row['complete'];continue
                    cpu=time.process_time();begin=time.perf_counter()
                    engine=RegionOracle(scene,independent=backend=='independent',mode=MODES[mode],cover_depth=args.cover_depth,
                                        cover_nodes=args.cover_nodes,shadow_boxes=args.shadow_boxes,fast_exp=args.fast_exp,
                                        volume_after_cache=args.volume_after_cache,trace=args.trace)
                    setup=time.perf_counter()-begin;t=time.perf_counter();row=engine.field(source,n,args.seconds)
                    call=time.perf_counter()-t;CPU=time.process_time()-cpu;arrays={k:row.pop(k) for k in ('response','solid','valid')}
                    for k in ('evaluation_path','ray_nanoseconds','point_nanoseconds'):
                        if k in row:arrays[k]=row.pop(k)
                    t=time.perf_counter();archive=args.output/f'{label}.npz';np.savez_compressed(archive,**arrays);write=time.perf_counter()-t
                    row.update(case=name,resolution=n,mode=mode,backend=backend,geometry_id=geometry,
                               effective_settings={k:getattr(engine.settings,k) for k in ('mode','cover_depth','cover_nodes','cover_seconds','shadow_boxes','parameter_nodes','parameter_depth','fast_exp','volume_after_cache','trace')},
                               source_coordinates_sha256=hashlib.sha256(np.array(source,dtype='<f8').tobytes()).hexdigest(),
                               setup_python_seconds=setup,field_call_seconds=call,output_write_seconds=write,
                               cold_total_seconds=setup+call+write,CPU_seconds=CPU,valid_voxels=int(arrays['valid'].sum()),
                               archive_bytes=archive.stat().st_size,archive_SHA256=sha(archive),
                               physical_bits_SHA256=hashlib.sha256(np.packbits(arrays['response']).tobytes()+np.packbits(arrays['solid']).tobytes()).hexdigest(),
                               process_peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
                    saved=args.baseline/name/f'primary-{n}.npz'
                    if saved.exists():
                        old=np.load(saved);common=old['valid']&arrays['valid']
                        row['old_baseline_compared_voxels']=int(common.sum())
                        row['old_baseline_bit_disagreements']=sum(int(np.count_nonzero((old[k]!=arrays[k])&common)) for k in ('response','solid'))
                        if row['old_baseline_bit_disagreements']:atomic_json(checkpoint,row);raise AssertionError('Physical disagreement with frozen field')
                    atomic_json(checkpoint,row);results.append(row);level_complete &= row['complete']
                    print(json.dumps({k:row[k] for k in ('case','resolution','mode','backend','complete','valid_voxels','cold_total_seconds','cover_seconds','region_seconds','fallback_seconds')}),flush=True)
            if not level_complete:
                print(json.dumps({'case':name,'resolution':n,'gate':'incomplete; do not advance this sweep'}),flush=True);break
    # Independently compare each same-mode primary/reference pair.
    for case in args.ordinals:
        for n in args.resolutions:
            for mode in args.modes:
                prefix=f'gaussian_sum-{case:03d}-{n}-{mode}-';paths=[args.output/f'{prefix}{backend}.npz' for backend in ('primary','independent')]
                if all(p.exists() for p in paths):
                    a,b=(np.load(p) for p in paths);common=a['valid']&b['valid']
                    assert not any(np.any((a[k]!=b[k])&common) for k in ('response','solid')),'Independent mismatch'
    atomic_json(args.output/'summary.json',{'identity':identity,'results':results,'new_READY_scenes':0,'new_1024_fields':0})


if __name__=='__main__':main()
