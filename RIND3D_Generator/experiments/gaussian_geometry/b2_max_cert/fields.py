"""Fresh-process complete fields with only the residual primary ray replaced."""
import argparse
import hashlib
import json
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
from .api import parameters
from .field_bridge import extension,source_identity

METHODS=['A_primary','A_independent','D_hybrid','D_checked','E_lean','E_checked']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def worker(args):
    case=f'gaussian_sum-{args.ordinals[0]:03d}';n=args.resolutions[0];method=args.methods[0]
    scene=load(args.input/case/'geometry.json');source=json.loads((args.input/case/'result.json').read_text())['source_xyz_synthetic']
    start=time.perf_counter();original=NativeOracle(scene,independent=method=='A_independent');assert original.source_clearance(source).status=='FREE';clearance=time.perf_counter()-start
    start=time.perf_counter();cpu=time.process_time()
    if method.startswith('A_'):
        oracle=RegionOracle(scene,independent=method=='A_independent',mode=5,cover_depth=7,cover_nodes=131073,shadow_boxes=256,trace=True,max_nodes=16384,max_depth=40)
        engine_setup=time.perf_counter()-start;t=time.perf_counter();data=oracle.field(source,n,args.seconds)
    else:
        m=extension(args.hook);engine=m.Engine(*parameters(scene),False);settings=m.Settings();options=m.Options()
        settings.mode=5;settings.cover_depth=7;settings.cover_nodes=131073;settings.shadow_boxes=256;settings.trace=True
        settings.maximum_method=3;settings.check_miss_proof=method.endswith('checked');settings.checker_precision=256
        settings.initial_endpoint_probes=not method.startswith('E_')
        options.max_nodes=16384;options.max_depth=40
        engine_setup=time.perf_counter()-start;t=time.perf_counter();data=engine.field(source,n,args.seconds,options,settings)
    call=time.perf_counter()-t;cpu_seconds=time.process_time()-cpu
    arrays={k:np.asarray(data.pop(k)) for k in ('response','solid','valid','evaluation_path','ray_nanoseconds','point_nanoseconds')}
    valid=arrays['valid'].astype(bool);response=arrays['response'].astype(bool);solid=arrays['solid'].astype(bool)
    # The original frozen complete field is an additional immutable comparison.
    baseline=args.previous/f'{case}-{n}-A_fixed7-independent.npz'
    if not baseline.exists():baseline=args.previous/f'{case}-{n}-A_independent.npz'
    if not baseline.exists():baseline=args.output/f'{case}-{n}-A_independent.npz'
    if baseline.exists():
        metadata=json.loads(baseline.with_suffix('.json').read_text())
        assert metadata['geometry_id']==record(scene)['geometry_id'],'Independent geometry identity mismatch'
        expected_source=hashlib.sha256(np.asarray(source,dtype='<f8').tobytes()).hexdigest()
        assert metadata.get('source_SHA256',metadata.get('source_coordinates_SHA256'))==expected_source,'Independent source identity mismatch'
        saved=np.load(baseline);common=saved['valid']&valid
        assert saved['valid'].all(),'Independent baseline is incomplete'
        mismatches=sum(int(np.count_nonzero((saved[k]!=arrays[k])&common)) for k in ('response','solid'))
        assert mismatches==0
    else:
        if not method.startswith('A_'):raise ValueError('Run a complete original independent field first, or pass --previous')
        mismatches=None
    name=f'{case}-{n}-{method}';archive=args.output/(name+'.npz');t=time.perf_counter();np.savez_compressed(archive,**arrays);write=time.perf_counter()-t
    data.update(case=case,resolution=n,method=method,geometry_id=record(scene)['geometry_id'],source_SHA256=hashlib.sha256(np.array(source,dtype='<f8').tobytes()).hexdigest(),
                source_clearance='FREE',source_clearance_and_setup_seconds=clearance,engine_setup_seconds=engine_setup,field_call_seconds=call,
                CPU_seconds=cpu_seconds,output_write_seconds=write,cold_total_seconds=engine_setup+call+write,
                cold_including_clearance_seconds=clearance+engine_setup+call+write,
                process_peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                valid_voxels=int(valid.sum()),archive_SHA256=sha(archive),archive_bytes=archive.stat().st_size,
                physical_bits_SHA256=hashlib.sha256(np.packbits(response).tobytes()+np.packbits(solid).tobytes()).hexdigest(),
                frozen_independent_bit_mismatches=mismatches,
                frozen_independent_archive_SHA256=sha(baseline) if baseline.exists() else None,
                experimental_ray_policy={'local_maximum_method':3,'initial_endpoint_probes':not method.startswith('E_'),'old_fallback':True,'checker_precision':256,'proof_check':method.endswith('checked')},
                proof_transport='Native interval objects only; no per-voxel JSON serialization in field timings',
                unchanged_independent_renderer_still_required=True,new_READY_scenes=0)
    atomic_json(args.output/(name+'.json'),data)
    print(json.dumps({k:data[k] for k in ('case','resolution','method','complete','valid_voxels','cold_total_seconds')}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=Path('outputs/gaussian-geometry-v1/distribution-v2'))
    p.add_argument('--previous',type=Path,default=Path('outputs/gaussian-b2-angular-v1/depth7-scales-v3'))
    p.add_argument('--hook',type=Path,default=Path('outputs/gaussian-b2-max-cert-v1/field-hook-v3'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ordinals',type=int,nargs='+',default=[14,7]);p.add_argument('--resolutions',type=int,nargs='+',default=[32,64])
    p.add_argument('--methods',choices=METHODS,nargs='+',default=METHODS);p.add_argument('--seconds',type=float,default=480)
    p.add_argument('--one',action='store_true',help=argparse.SUPPRESS);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    assert all(n in (8,16,32,64,128) for n in a.resolutions),'No full 1024³ generation in this experiment'
    if a.one:worker(a);return
    identity={'sources':source_identity(),'driver_SHA256':sha(Path(__file__)),'hook_identity':json.loads((a.hook/'identity.json').read_text()),
              'ordinals':a.ordinals,'resolutions':a.resolutions,'methods':a.methods,'seconds':a.seconds,'source_clearance_unchanged':True}
    identity['frozen_inputs']={f'gaussian_sum-{i:03d}':{name:sha(a.input/f'gaussian_sum-{i:03d}'/name) for name in ('geometry.json','result.json')} for i in a.ordinals}
    manifest=a.output/'manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())==identity,'Use a new field-run identity'
    else:atomic_json(manifest,identity)
    results=[]
    for ordinal in a.ordinals:
        for n in a.resolutions:
            current=[]
            for method in a.methods:
                label=f'gaussian_sum-{ordinal:03d}-{n}-{method}';path=a.output/(label+'.json')
                if not path.exists():
                    cmd=[sys.executable,'-m','experiments.gaussian_geometry.b2_max_cert.fields','--one','--output',str(a.output),'--input',str(a.input),'--previous',str(a.previous),'--hook',str(a.hook),'--ordinals',str(ordinal),'--resolutions',str(n),'--methods',method,'--seconds',str(a.seconds)]
                    t=time.perf_counter();subprocess.run(cmd,check=True,timeout=a.seconds+120);row=json.loads(path.read_text());row['fresh_process_end_to_end_seconds']=time.perf_counter()-t;atomic_json(path,row)
                row=json.loads(path.read_text());assert sha(a.output/(label+'.npz'))==row['archive_SHA256'];results.append(row);current.append(row)
            assert len({r['physical_bits_SHA256'] for r in current if r['complete']})<=1,'Completed field bit mismatch'
            atomic_json(a.output/'summary.json',{'identity':identity,'results':results,'all_completed_so_far':all(r['complete'] for r in results)})
            if not all(r['complete'] for r in current):print('INCOMPLETE: no larger field for this case',flush=True);break


if __name__=='__main__':main()
