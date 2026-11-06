"""Ablate global worst-object scaling versus individual world-fitting caps.

Same Gaussian centers, primitive kinds and orientations; unchanged native
geometry acceptance, source clearance and A/B physics. No production change.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiments.gaussian_geometry.generators import layout,convex_objects,rng_for,grid_points,DEFAULT
from experiments.gaussian_geometry.models import ConvexScene,save,record
from experiments.gaussian_geometry.queries import make_oracle
from experiments.gaussian_geometry.study import atomic_json,quality,sample_source,query_benchmark,window_diagnostic,field_benchmark


def generate_capped(seed,ordinal,target,cfg=DEFAULT):
    kernels,meta=layout(seed,ordinal,cfg)
    objects=convex_objects(kernels,rng_for(seed,ordinal,3107))
    points=grid_points(cfg['calibration_resolution']);margin=cfg['world_margin']
    eligible=np.full(len(points),np.inf);caps=[]
    for o in objects:
        local=(points-o['center'])@np.linalg.inv(o['matrix']).T
        gauge=np.max(np.asarray(o['normal'])@local.T/np.asarray(o['plane_offset'])[:,None],axis=0) if 'normal' in o else np.max(np.abs(local),axis=1)
        extent=np.sum(np.abs(o['matrix']),axis=1)*o['local_radius_bound']
        cap=.94*float(np.min(np.minimum(o['center']-margin,1-margin-o['center'])/extent))
        if cap<=0:raise ValueError('No legal positive object proposal scale')
        caps.append(cap);eligible=np.minimum(eligible,np.where(gauge<=cap,gauge,np.inf))
    finite=eligible[np.isfinite(eligible)]
    if not len(finite):raise RuntimeError('No occupied calibration witness, not an emptiness proof')
    rank=min(len(eligible)-1,max(0,int(np.ceil(target*len(eligible)))-1))
    chosen=float(np.partition(eligible,rank)[rank]);limited=not np.isfinite(chosen)
    if limited:chosen=float(finite.max())
    scales=[min(chosen,cap) for cap in caps]
    for o,scale in zip(objects,scales):o['matrix']=np.asarray(o['matrix']*scale,dtype='<f8')
    meta.update(world_fit_policy='per-component-caps-v1',density_target=target,
                object_scale_caps=caps,object_scales=scales,common_density_parameter=chosen,
                approximate_max_attainable_calibration_occupancy=len(finite)/len(eligible),
                target_exceeds_calibration_capacity=limited,calibration_is_not_a_certificate=True)
    return ConvexScene(objects,meta)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();cfg=json.loads((args.input/'run-manifest.json').read_text())['config']
    identity={'input_manifest_sha256':hashlib.sha256((args.input/'run-manifest.json').read_bytes()).hexdigest(),
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'policy':'per-component-caps-v1'}
    args.output.mkdir(parents=True,exist_ok=True);manifest=args.output/'ablation-manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=identity:raise ValueError('Ablation identity changed')
    atomic_json(manifest,identity);rows=[];regimes=list(cfg['occupancy_targets'])
    for ordinal in range(cfg['scenes_per_model']):
        case=f'gaussian_convex-{ordinal:03d}';target=args.output/case;target.mkdir(exist_ok=True);path=target/'result.json'
        if path.exists():rows.append(json.loads(path.read_text()));continue
        began=time.perf_counter();cpu=time.process_time();regime=regimes[(ordinal//len(cfg['styles']))%len(regimes)]
        scene=generate_capped(cfg['master_seed'],ordinal,cfg['occupancy_targets'][regime],cfg)
        row={'case':case,'model':'gaussian_convex_capped','ordinal':ordinal,'regime':regime,'geometry_id':record(scene)['geometry_id'],
             'generation_seconds':time.perf_counter()-began,'status':'INCOMPLETE','READY_published':False}
        save(target/'geometry.json',scene);row['geometry_json_bytes']=(target/'geometry.json').stat().st_size
        row['quality']=quality(scene,cfg['quality_resolution']);oracle=make_oracle(scene)
        row['containment']=oracle.containment(cfg['world_margin']);assert row['containment']['status']=='CERTIFIED'
        old=json.loads((args.input/case/'result.json').read_text());source=np.array(old['source_xyz_synthetic'])
        same=oracle.membership(source).status=='FREE' and oracle.source_clearance(source).status=='FREE'
        if not same:source,sreport=sample_source(oracle,cfg['master_seed']+ordinal)
        else:sreport={'policy':'same certified source as baseline','proposals':0}
        if source is None:row['status']='SOURCE_BUDGET_INCOMPLETE'
        else:
            row.update(source_same_as_baseline=bool(same),source_xyz_synthetic=source.tolist(),source_sampling=sreport)
            row['queries']=query_benchmark(scene,oracle,source,ordinal,cfg)
            if ordinal in cfg['field_case_ordinals']:
                row['windows']=window_diagnostic(oracle,source,cfg['master_seed']+ordinal*31)
                row['fields']=field_benchmark(scene,oracle,source,target,dict(cfg,field_resolutions=[32,64,128]))
            row['status']='RESEARCH_QUERIES_COMPLETE'
        row.update(total_seconds=time.perf_counter()-began,cpu_seconds=time.process_time()-cpu)
        atomic_json(path,row);rows.append(row)
        print(json.dumps({'case':case,'occupancy':row['quality']['occupied_ratio_estimate'],'status':row['status']}),flush=True)
    atomic_json(args.output/'summary.json',{'cases':rows,'mass_production':False,'new_READY_scenes':0})


if __name__=='__main__':main()
