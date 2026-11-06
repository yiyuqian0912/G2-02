"""Versioned complete-field ablations; one fresh, bounded process per field."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
from ..models import load,record
from ..study import atomic_json
from ..b2_native.api import NativeOracle
from ..b2_regions.api import RegionOracle
from .api import AngularOracle
from .build import sources

METHODS={'A_fixed6':None,'A_fixed7':None,'B_spatial':1,'C_angular':2,'D_union':3,'E_hybrid':4,'P_empty_prefix':3,'Q_hybrid_prefix':4,'R_point_cover':0,'S_all_inner':None}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run_one(args):
    case=f'gaussian_sum-{args.ordinals[0]:03d}';n=args.resolutions[0];method=args.methods[0];backend=args.backends[0]
    folder=args.input/case;scene=load(folder/'geometry.json');source=json.loads((folder/'result.json').read_text())['source_xyz_synthetic']
    independent=backend=='independent';label=f'{case}-{n}-{method}-{backend}'
    t=time.perf_counter();native=NativeOracle(scene,independent=independent)
    assert native.source_clearance(source).status=='FREE';clearance=time.perf_counter()-t
    cpu=time.process_time();t=time.perf_counter()
    if method.startswith('A_') or method=='S_all_inner':
        engine=RegionOracle(scene,independent=independent,mode=5,cover_depth=7 if method=='A_fixed7' else 6,
                            cover_nodes=131073 if method=='A_fixed7' else 32769,
                            shadow_boxes=100000 if method=='S_all_inner' else 256,trace=True)
    else:
        engine=AngularOracle(scene,independent=independent,method=METHODS[method],cover_depth=args.cover_depth,
                             cover_nodes=args.cover_nodes,adaptive_nodes=args.adaptive_nodes,adaptive_batch=args.adaptive_batch,
                             angular_depth=args.angular_depth,angular_nodes=args.angular_nodes,
                             angular_boxes=args.angular_boxes,
                             empty_prefix=method in ('P_empty_prefix','Q_hybrid_prefix'),prefix_builds=args.prefix_builds,trace=True)
    setup=time.perf_counter()-t;t=time.perf_counter()
    if args.tile_origin is None:row=engine.field(source,n,args.seconds)
    else:row=engine.tile(source,args.world_grid,args.tile_origin,n,args.seconds)
    call=time.perf_counter()-t;cpu_seconds=time.process_time()-cpu
    row['effective_settings']={k:getattr(engine.settings,k) for k in dir(engine.settings)
                               if not k.startswith('_') and isinstance(getattr(engine.settings,k),(bool,int,float))}
    row['effective_numerical_options']={k:getattr(engine.options,k) for k in dir(engine.options)
                                       if not k.startswith('_') and isinstance(getattr(engine.options,k),(bool,int,float))}
    arrays={k:row.pop(k) for k in ('response','solid','valid','evaluation_path','ray_nanoseconds','point_nanoseconds') if k in row}
    for k in ('visibility_evidence','angular_tiles'):
        if k in row:
            arrays[k]=np.asarray(row.pop(k))
            if k=='angular_tiles':
                tiles=arrays[k].reshape(-1,8)
                row['angular_leaf_tile_summary']={'leaves':len(tiles),'finite_H_leaves':int(np.isfinite(tiles[:,5]).sum()),
                    'nonnegative_L_leaves':int((tiles[:,6]>=0).sum()),
                    'scope':'Unique retained angular leaves, not all cached ancestor certificates or a sampled first-hit map.'}
    t=time.perf_counter();archive=args.output/f'{label}.npz';np.savez_compressed(archive,**arrays);write=time.perf_counter()-t
    row.update(case=case,resolution=n,method=method,backend=backend,geometry_id=record(scene)['geometry_id'],
               source_coordinates_SHA256=hashlib.sha256(np.asarray(source,dtype='<f8').tobytes()).hexdigest(),
               source_clearance='FREE',source_clearance_and_native_setup_seconds=clearance,
               setup_python_seconds=setup,field_call_seconds=call,output_write_seconds=write,
               cold_total_seconds=setup+call+write,cold_including_clearance_seconds=clearance+setup+call+write,
               CPU_seconds=cpu_seconds,valid_voxels=int(arrays['valid'].sum()),archive_bytes=archive.stat().st_size,
               archive_SHA256=sha(archive),physical_bits_SHA256=hashlib.sha256(np.packbits(arrays['response']).tobytes()+np.packbits(arrays['solid']).tobytes()).hexdigest(),
               process_peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    if args.tile_origin is None:
        old=args.baseline/case/f'primary-{n}.npz'
        if old.exists():
            baseline=np.load(old);common=baseline['valid']&arrays['valid']
            row['frozen_baseline_compared_voxels']=int(common.sum())
            row['frozen_baseline_bit_disagreements']=sum(int(np.count_nonzero((baseline[k]!=arrays[k])&common)) for k in ('response','solid'))
            assert row['frozen_baseline_bit_disagreements']==0
    atomic_json(args.output/f'{label}.json',row)
    print(json.dumps({k:row[k] for k in ('case','resolution','method','backend','complete','valid_voxels','cold_total_seconds','cover_seconds','fallback_seconds')}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('outputs/gaussian-geometry-v1/distribution-v2'))
    p.add_argument('--baseline',type=Path,default=Path('outputs/gaussian-b2-native-v1/fields-v3'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ordinals',type=int,nargs='+',default=[0,7,14,21]);p.add_argument('--resolutions',type=int,nargs='+',default=[32])
    p.add_argument('--methods',choices=METHODS,nargs='+',default=list(METHODS)[:6]);p.add_argument('--backends',choices=['primary','independent'],nargs='+',default=['primary','independent'])
    p.add_argument('--seconds',type=float,default=360);p.add_argument('--cover-depth',type=int,default=6);p.add_argument('--cover-nodes',type=int,default=32769)
    p.add_argument('--adaptive-nodes',type=int,default=8192);p.add_argument('--adaptive-batch',type=int,default=8)
    p.add_argument('--angular-depth',type=int,default=10);p.add_argument('--angular-nodes',type=int,default=32766)
    p.add_argument('--prefix-builds',type=int,default=512)
    p.add_argument('--angular-boxes',type=int,default=-1,help='-1 all inner proofs; 256 isolates the old inner subset')
    p.add_argument('--tile-origin',type=int,nargs=3);p.add_argument('--world-grid',type=int,default=1024)
    p.add_argument('--one',action='store_true',help=argparse.SUPPRESS);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.one:run_one(args);return
    cfg={k:v for k,v in vars(args).items() if not isinstance(v,Path) and k!='one'}
    build=Path(os.environ.get('RIND_B2_ANGULAR_BUILD','outputs/gaussian-b2-angular-v1/build'))/'build-identity.json'
    identity={'config':cfg,'native_build':json.loads(build.read_text()),'driver_SHA256':sha(Path(__file__)),
              'api_SHA256':sha(Path(__file__).with_name('api.py')),'sources':sources(),
              'input_manifest_SHA256':sha(args.input/'run-manifest.json'),
              'original_field_manifest_SHA256':sha(args.baseline/'manifest.json') if (args.baseline/'manifest.json').exists() else None,
              'new_READY_scenes':0,'cold_scope':'engine construction + continuous cover + all attempted certificates + leaf work + NPZ write; process startup separately recorded'}
    manifest=args.output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=identity:raise ValueError('Changed benchmark identity: choose a new versioned output')
    atomic_json(manifest,identity);results=[]
    for ordinal in args.ordinals:
        for n in args.resolutions:
            complete=True;offset=ordinal%len(args.methods);order=args.methods[offset:]+args.methods[:offset]
            for method in order:
                for backend in args.backends:
                    label=f'gaussian_sum-{ordinal:03d}-{n}-{method}-{backend}';checkpoint=args.output/f'{label}.json'
                    if not checkpoint.exists():
                        command=[sys.executable,'-m','experiments.gaussian_geometry.b2_angular.benchmark','--one',
                                 '--output',str(args.output),'--input',str(args.input),'--baseline',str(args.baseline),
                                 '--ordinals',str(ordinal),'--resolutions',str(n),'--methods',method,'--backends',backend,
                                 '--seconds',str(args.seconds),'--cover-depth',str(args.cover_depth),'--cover-nodes',str(args.cover_nodes),
                                 '--adaptive-nodes',str(args.adaptive_nodes),'--adaptive-batch',str(args.adaptive_batch),
                                 '--angular-depth',str(args.angular_depth),'--angular-nodes',str(args.angular_nodes),'--prefix-builds',str(args.prefix_builds),'--angular-boxes',str(args.angular_boxes)]
                        if args.tile_origin is not None:command+=['--tile-origin',*map(str,args.tile_origin),'--world-grid',str(args.world_grid)]
                        start=time.perf_counter();subprocess.run(command,check=True,timeout=args.seconds+120)
                        row=json.loads(checkpoint.read_text());row['fresh_process_end_to_end_seconds']=time.perf_counter()-start;atomic_json(checkpoint,row)
                    row=json.loads(checkpoint.read_text());archive=args.output/f'{label}.npz';assert sha(archive)==row['archive_SHA256']
                    complete &= row['complete'];results.append(row)
            for method in args.methods:
                paths=[args.output/f'gaussian_sum-{ordinal:03d}-{n}-{method}-{b}.npz' for b in ('primary','independent')]
                if all(x.exists() for x in paths):
                    a,b=[np.load(x) for x in paths];common=a['valid']&b['valid'];assert not any(np.any((a[k]!=b[k])&common) for k in ('response','solid'))
            atomic_json(args.output/'summary.json',{'identity':identity,'results':results,'all_completed_so_far':all(x['complete'] for x in results)})
            if not complete:print(json.dumps({'case':ordinal,'resolution':n,'gate':'incomplete, no larger resolution for this case'}),flush=True);break


if __name__=='__main__':main()
