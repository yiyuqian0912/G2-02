"""Matched B2 query ablations on immutable v1 geometries and frozen receivers."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
from scipy.stats import qmc

from ..models import load
from ..queries import FieldOracle,Limits
from ..study import atomic_json
from .api import NativeOracle
from .build import sources


def quantiles(v):
    return dict(zip(('p50','p95','p99','max'),map(float,np.percentile(v,[50,95,99,100])))) if len(v) else {}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,default=Path('experiments/gaussian_geometry/configs/b2-native-v2.json'))
    p.add_argument('--ordinals',nargs='+',type=int);args=p.parse_args();cfg=json.loads(args.config.read_text())
    if args.ordinals is not None:cfg['case_ordinals']=args.ordinals
    identity={'config':cfg,'native_sources':sources(),'input_manifest_sha256':hashlib.sha256((args.input/'run-manifest.json').read_bytes()).hexdigest(),
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.mkdir(parents=True,exist_ok=True);manifest=args.output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=identity:raise ValueError('Benchmark identity changed')
    atomic_json(manifest,identity);rows=[]
    opts=dict(max_nodes=cfg['max_nodes'],max_depth=cfg['max_depth'],query_seconds=cfg['query_seconds'],precision=cfg['primary_fallback_precision'])
    modes={'native_no_cache':dict(coefficients=False,filters=False,taylor=False,exact_depth=51),
           'native_cached':dict(coefficients=True,filters=False,taylor=False,exact_depth=51),
           'native_filters':dict(coefficients=True,filters=True,taylor=False,exact_depth=51),
           'native_groups':dict(coefficients=True,filters=True,groups=True,taylor=False,exact_depth=51),
           'native_taylor':dict(coefficients=True,filters=True,taylor=True,exact_depth=20),
           'independent_plain':dict(independent=True,taylor=False,tail_bounds=False),
           'independent_tail':dict(independent=True,taylor=False,tail_bounds=True),
           'independent_taylor':dict(independent=True,taylor=True,tail_bounds=False),
           'independent':dict(independent=True,taylor=True,tail_bounds=True)}
    for ordinal in cfg['case_ordinals']:
        case=f'gaussian_sum-{ordinal:03d}';out=args.output/case;out.mkdir(exist_ok=True);checkpoint=out/'result.json'
        if checkpoint.exists():rows.append(json.loads(checkpoint.read_text()));continue
        path=args.input/case;scene=load(path/'geometry.json');original=json.loads((path/'result.json').read_text());source=original['source_xyz_synthetic']
        old=FieldOracle(scene,Limits(max_nodes=cfg['max_nodes'],max_depth=cfg['max_depth'],seconds=cfg['query_seconds']))
        candidates=qmc.Sobol(3,scramble=True,seed=cfg['master_seed']+ordinal*317).random_base2(9)
        points=candidates[:cfg['point_queries_per_case']]
        receivers=np.array([x for x in candidates if old.membership(x).status=='FREE'])[:cfg['free_receiver_rays_per_case']]
        if len(receivers)!=cfg['free_receiver_rays_per_case']:raise RuntimeError('Insufficient baseline-certified FREE query pool')
        query_path=out/'queries.npz';np.savez_compressed(query_path,source=source,receivers=receivers,points=points)
        row={'case':case,'geometry_id':original['geometry_id'],'kernel_count':len(scene.kernels),
             'query_sha256':hashlib.sha256(query_path.read_bytes()).hexdigest(),'rays':len(receivers),'modes':{},'publication':'EXPERIMENTAL_ONLY'}
        row['query_coordinates_sha256']=hashlib.sha256(np.asarray(source,dtype='<f8').tobytes()+receivers.astype('<f8').tobytes()+points.astype('<f8').tobytes()).hexdigest()
        cpu=time.process_time();begin=time.perf_counter();expected=[old.segment(source,x) for x in receivers]
        row['python_ray_seconds']=time.perf_counter()-begin;row['python_ray_CPU_seconds']=time.process_time()-cpu
        begin=time.perf_counter();point_expected=[old.membership(x) for x in points];row['python_point_seconds']=time.perf_counter()-begin
        latencies=[]
        for x in receivers[:cfg['scalar_latency_rays_per_case']]:
            t=time.perf_counter();old.segment(source,x);latencies.append(time.perf_counter()-t)
        row['python_scalar_seconds']=quantiles(latencies);row['python_statuses']=dict(Counter(d.status for d in expected))
        order=list(modes);order=order[ordinal%len(order):]+order[:ordinal%len(order)]
        for name in order:
            begin=time.perf_counter();engine=NativeOracle(scene,**opts,**modes[name]);setup=time.perf_counter()-begin
            times=[];cpu_times=[];results=None;work={}
            for _ in range(cfg['native_batch_repeats']):
                engine.work.clear();cpu=time.process_time();t=time.perf_counter();results=engine.rays(source,receivers)
                times.append(time.perf_counter()-t);cpu_times.append(time.process_time()-cpu);work=dict(engine.work)
            mismatches=sum(a.certified and b.certified and a.status!=b.status for a,b in zip(expected,results))
            unresolved=sum(not d.certified for d in results)
            if mismatches:raise AssertionError(f'Matched physical disagreement: {case} {name}')
            t=time.perf_counter();pp=engine.points(points);point_sec=time.perf_counter()-t
            if any(a.certified and b.certified and a.status!=b.status for a,b in zip(pp,point_expected)):raise AssertionError('Point disagreement')
            latencies=[]
            for x in receivers[:cfg['scalar_latency_rays_per_case']]:
                t=time.perf_counter();engine.segment(source,x);latencies.append(time.perf_counter()-t)
            row['modes'][name]={'setup_seconds':setup,'batch_wall_seconds':times,'batch_CPU_seconds':cpu_times,
                  'point_batch_seconds':point_sec,'scalar_seconds':quantiles(latencies),'status_counts':dict(Counter(d.status for d in results)),
                  'unresolved':unresolved,'disagreements':mismatches,'work_one_batch':work,
                  'reasons':dict(Counter(d.reason for d in results))}
        row['process_peak_RSS_MiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        atomic_json(checkpoint,row);rows.append(row)
        print(json.dumps({'case':case,'python_seconds':row['python_ray_seconds'],
                          'native_seconds':float(np.median(row['modes']['native_filters']['batch_wall_seconds'])),
                          'unknown':{k:v['unresolved'] for k,v in row['modes'].items()}}),flush=True)
    atomic_json(args.output/'summary.json',{'cases':rows,'identity':identity,'new_READY_scenes':0})


if __name__=='__main__':main()
