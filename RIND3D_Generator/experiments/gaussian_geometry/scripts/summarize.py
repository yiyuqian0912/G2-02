"""Aggregate measured research artifacts, without any production/private labels."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics

import numpy as np

from experiments.gaussian_geometry.study import atomic_json


def distribution(values):
    a=[float(x) for x in values if x is not None]
    return {'count':len(a),'min':min(a),'median':statistics.median(a),
            'p95':float(np.percentile(a,95)),'max':max(a)} if a else None


def models(rows):
    result={}
    for model in sorted({x['model'] for x in rows}):
        rr=[x for x in rows if x['model']==model];work=Counter();rays=Counter();membership=Counter()
        for r in rr:
            work.update(r['queries']['ray_work']);rays.update(r['queries']['ray_counts']);membership.update(r['queries']['membership_counts'])
        q=lambda key:distribution(r['quality'].get(key) for r in rr)
        result[model]={
            'scenes':len(rr),'status_counts':dict(Counter(r['status'] for r in rr)),
            'occupancy_mean_by_target_regime':{s:statistics.mean(r['quality']['occupied_ratio_estimate'] for r in rr if r['regime']==s)
                                             for s in sorted({r['regime'] for r in rr})},
            'generation_seconds':distribution(r['generation_seconds'] for r in rr),
            'primary_ms_per_ray':distribution(1000*r['queries']['ray_seconds']/r['queries']['free_receiver_rays'] for r in rr),
            'primary_ms_per_point':distribution(1000*r['queries']['point_seconds']/r['queries']['point_queries'] for r in rr),
            'very_sparse_ms_per_ray':distribution(1000*r['queries']['ray_seconds']/r['queries']['free_receiver_rays'] for r in rr if r['regime']=='very_sparse'),
            'ray_counts':dict(rays),'membership_counts':dict(membership),'work_counters':dict(work),
            'max_ray_depth':max(r['queries']['max_subdivision_depth'] for r in rr),
            'max_ray_nodes':max(r['queries']['max_ray_nodes'] for r in rr),
            'independent_agreed':sum(r['queries']['independent_agreed'] for r in rr),
            'independent_disagreed':sum(r['queries']['independent_disagreed'] for r in rr),
            'independent_unresolved_comparisons':sum(r['queries']['independent_unresolved'] for r in rr),
            'independent_seconds':sum(r['queries']['independent_seconds'] for r in rr),
            'geometry_json_bytes':distribution(r['geometry_json_bytes'] for r in rr),
            'geometry_json_bytes_total':sum(r['geometry_json_bytes'] for r in rr),
            'components_6_connected':q('components_6_connected_estimate'),
            'tiny_components_at_most_8_voxels':q('tiny_components_at_most_8_voxels'),
            'spatial_octants':q('spatial_octants_estimate'),'spatial_4cubed_coverage':q('spatial_4x4x4_coverage_estimate'),
            'pca_aspect':q('PCA_aspect_estimate'),'parameter_orientation_entropy':q('kernel_orientation_entropy_estimate'),
            'sampled_boundary_distance':q('boundary_distance_estimate'),
            'voxel_thin_fraction':q('voxel_thickness_below_0_04_fraction_estimate'),
            'surface_cubical_area':q('surface_cubical_area_estimate'),
            'parameter_axis_ratio_above_20_total':sum(r['quality']['parameter_anisotropy_above_20_count'] for r in rr),
            'source_proposals':sum(r['source_sampling']['proposals'] for r in rr),
            'source_same_as_baseline_count':sum(r.get('source_same_as_baseline',False) for r in rr),
            'case_seconds_sum':sum(r['total_seconds'] for r in rr),'case_cpu_seconds_sum':sum(r['cpu_seconds'] for r in rr),
        }
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();base=args.root
    read=lambda name:json.loads((base/name).read_text())
    original=read('distribution-v2/summary.json')['cases'];capped=read('capped-convex-v1/summary.json')['cases']
    old=read('distribution-v1/summary.json')['cases'];assert {r['case']:r['geometry_id'] for r in old}=={r['case']:r['geometry_id'] for r in original}
    out={'public_name':'RIND3D','experiment':'gaussian-geometry-2026-11-05','base_commit':'59e343ac51d2cd22dbe7dea6ffa08713e3b84e15',
         'new_READY_scenes':0,'production_1024_fields':0,'main_models':models(original),'capped_A':models(capped),
         'v1_v2_geometry_id_match_count':len(original),
         'measurement_scope':'new synthetic research inputs; voxel quality/topology estimates are not continuous certificates',
         'runtime':{'python':platform.python_version(),'platform':platform.platform(),
                    'versions':{n:importlib.metadata.version(n) for n in ('numpy','scipy','python-flint','gmpy2','scikit-image','matplotlib')},
                    'cpu_model':next(x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),
                    'OMP_NUM_THREADS':1,'OPENBLAS_NUM_THREADS':1,
                    'distribution_process_peak_RSS_MiB':max(r['process_peak_RSS_MiB'] for r in original),
                    'GPU_geometry_queries':False}}
    from rind3d import _convex
    out['runtime']['production_native_module_sha256']=hashlib.sha256(Path(_convex.__file__).read_bytes()).hexdigest()
    out['fields']=read('fields-v3/summary.json')['cases']
    out['sampled_physical_geometry']=[]
    for row in out['fields']:
        case=row['case'];data=np.load(base/'fields-v3'/case/'field-8.npz')
        lit,solid,valid=data['response'],data['solid'],data['valid'];free=valid&~solid;shadow=free&~lit
        edges=transitions=0
        for axis in range(3):
            first=[slice(None)]*3;last=first.copy();first[axis]=slice(None,-1);last[axis]=slice(1,None)
            a,b=tuple(first),tuple(last);eligible=free[a]&free[b]
            edges+=int(eligible.sum());transitions+=int(np.count_nonzero(eligible&(lit[a]!=lit[b])))
        vox=np.argwhere(shadow);octants=len(np.unique((vox>=4).astype(int)@[1,2,4])) if len(vox) else 0
        source=np.array(read(f'distribution-v2/{case}/result.json')['source_xyz_synthetic'])
        distance=np.linalg.norm((vox[:,::-1]+.5)/8-source,axis=1) if len(vox) else []
        out['sampled_physical_geometry'].append({'case':case,'resolution':8,'complete':bool(valid.all()),
              'valid_voxels':int(valid.sum()),'solid_voxels':int(solid[valid].sum()),'lit_free_voxels':int(np.count_nonzero(lit&free)),
              'shadow_free_voxels':int(shadow.sum()),'free_adjacent_pairs':edges,'opposite_response_free_adjacent_pairs':transitions,
              'shadow_spatial_octants':octants,'shadow_receiver_distance':distribution(distance),
              'scope':'exact resolved sampled voxel labels; boundary/spread estimates only at 8³, not continuous topology or first-blocker distances'})
    out['capped_fields']=[{'case':r['case'],'fields':r['fields']} for r in capped if 'fields' in r]
    out['controls']=read('convex-controls/summary.json')['cases']
    out['nonlocal']={}
    for row in read('nonlocal-v2/summary.json')['cases']:
        group=out['nonlocal'].setdefault(row['model'],{'cases':0,'sides':{}});group['cases']+=1
        for side,block in row['sides'].items():
            d=group['sides'].setdefault(side,{'counts':Counter(),'wall_seconds':0.,'soft_budget_exhausted_cases':0})
            d['counts'].update(block['counts']);d['wall_seconds']+=block['wall_seconds'];d['soft_budget_exhausted_cases']+=block['time_budget_reached']
    for group in out['nonlocal'].values():
        for d in group['sides'].values():d['counts']=dict(d['counts'])
    out['bridge_proof']=read('bridge-control/proof.json')
    out['hard_case_replays']=read('hard-cases.json')['cases']
    out['acceleration_ablation']=read('acceleration-ablation.json')['cases']
    if (base/'extended-gates/summary.json').exists():
        out['explicit_larger_budget_controls']=read('extended-gates/summary.json')['cases']
    if (base/'parameter-ablation.json').exists():
        out['one_factor_parameters']=read('parameter-ablation.json')
    files=['distribution-v1/run-manifest.json','distribution-v2/run-manifest.json','capped-convex-v1/ablation-manifest.json',
           'fields-v3/summary.json','nonlocal-v2/audit-manifest.json','hard-cases.json','acceleration-ablation.json',
           'extended-gates/summary.json','parameter-ablation.json']
    out['input_sha256']={n:hashlib.sha256((base/n).read_bytes()).hexdigest() for n in files}
    atomic_json(args.output,out)


if __name__=='__main__':main()
