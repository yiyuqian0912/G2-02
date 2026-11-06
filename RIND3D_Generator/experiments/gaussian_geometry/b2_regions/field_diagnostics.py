"""Measured leaf work versus sampled boundary neighborhoods; not distance proofs."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation
from ..study import atomic_json


def analyze(path):
    a=np.load(path);valid=a['valid'];solid=a['solid']&valid;free=~solid&valid
    lit=a['response']&free;shadow=~a['response']&free
    structure=np.ones((3,)*3,dtype=bool)
    near_solid=free&binary_dilation(solid,structure=structure)
    near_shadow=free&binary_dilation(lit,structure=structure)&binary_dilation(shadow,structure=structure)&~near_solid
    masks={'solid':solid,'FREE_adjacent_to_sampled_solid':near_solid,
           'FREE_sampled_lit_shadow_transition':near_shadow,
           'FREE_other_interior':free&~near_solid&~near_shadow,'unresolved':~valid}
    tags=a['evaluation_path'];rays=a['ray_nanoseconds'];points=a['point_nanoseconds']
    groups={}
    for name,m in masks.items():
        groups[name]={'voxels':int(m.sum()),'scalar_ray_calls':int(np.count_nonzero(np.isin(tags[m],[6,7,8]))),
                      'timed_scalar_ray_wall_seconds':float(rays[m].sum(dtype=np.float64)/1e9),
                      'timed_membership_wall_seconds':float(points[m].sum(dtype=np.float64)/1e9)}
    actual=np.isin(tags,[6,7,8]);latencies=rays[actual].astype(float)/1e9
    worst=np.unravel_index(int(np.argmax(rays)),rays.shape) if actual.any() else None
    return {'artifact':path.name,'complete':bool(valid.all()),'groups':groups,
            'path_counts':{str(i):int(np.count_nonzero(tags==i)) for i in np.unique(tags)},
            'conditional_scalar_ray_latency_seconds':dict(zip(('p50','p95','p99','max'),map(float,np.percentile(latencies,[50,95,99,100])))) if len(latencies) else None,
            'worst_receiver_voxel_xyz':list(map(int,worst[::-1])) if worst else None,
            'total_timed_ray_seconds':float(rays.sum(dtype=np.float64)/1e9),
            'boundary_definition':'26-neighbor one-voxel sampled transitions; not exact continuous distance, topology or a lower bound on unavoidable work',
            'timing_scope':'wall time inside actual native scalar calls only; excludes region/preparation/assembly and timer overhead'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=[]
    for path in sorted(a.input.glob('*.npz')):
        with np.load(path) as data:
            if 'evaluation_path' not in data:continue
        rows.append(analyze(path))
    if not rows:raise ValueError('No traced fields; run the benchmark with --trace')
    atomic_json(a.output,{'diagnostics':rows,'scientific_labels_modified':False})
    print(json.dumps({'traced_fields':len(rows)}))


if __name__=='__main__':main()
