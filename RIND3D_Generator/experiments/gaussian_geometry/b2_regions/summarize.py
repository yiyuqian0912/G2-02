"""Compact measured region results, preserving versioned workload identities."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

from ..study import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    runs={};artifacts={}
    for path in sorted(a.root.glob('*/summary.json')):
        data=json.loads(path.read_text())
        if 'results' not in data or 'identity' not in data:continue
        rows=data['results']
        for r in rows:
            assert r.get('old_baseline_bit_disagreements',0)==0,r
            work=r['work'];raykey='reference_ray_queries' if r['backend']=='independent' else 'ray_queries'
            r['derived']={'region_voxel_fraction':work.get('region_filled_voxels',0)/r['resolution']**3,
                          'scalar_ray_queries':work.get(raykey,0),
                          'query_exp_endpoints':work.get('exp_endpoints',0)+work.get('reference_exp_endpoints',0)+work.get('high_precision_exp_endpoints',0),
                          'projection_prep_seconds':work.get('reference_projection_build_nanoseconds',0)/1e9,
                          'certificate_size_counts':{k.removeprefix('certificate_side_'):v for k,v in work.items() if k.startswith('certificate_side_')}}
        runs[path.parent.name]=data;artifacts[str(path.relative_to(a.root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    row={'experiment':'RIND3D B2 whole-receiver-region certification','date':'2026-11-06','branch':'yushu/gaussian-b2-native',
         'base_commit':'bd5e0d7956d896eb1a9f4f20334fdb9ad07bc547','new_READY_scenes':0,'new_1024_complete_fields':0,
         'runs':runs,'artifacts_SHA256':artifacts,'runtime':{'Python':platform.python_version(),'platform':platform.platform(),
              'cpu_model':next(x.split(':')[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),
              'versions':{x:importlib.metadata.version(x) for x in ('numpy','python-flint','gmpy2','scipy')},'threads':1,'new_physics_GPU_used':False},
         'current_build':json.loads((a.root/'build/build-identity.json').read_text()),
         'prior_measured_v2_build':json.loads((a.root/'native-measured-v2/build-identity.json').read_text()),
         'failed_diagnostics':[{'name':'unit-v1','reason':'Three test assertions indexed the old certificate API twice; no contradictory physical labels'},
             {'name':'pre-v2 proof audit','reason':'Finite-lambda projection needed a separate closed t=0 source-contact guard; added before four-scene gates'}]}
    for key,name in [('exponential','fast-exp-v1/results.json'),('controls','controls-v1/summary.json'),('controls_inputs','controls-v1/input-provenance.json'),('leaf_work','trace-analysis-v1.json')]:
        path=a.root/name
        if path.exists():row[key]=json.loads(path.read_text());row['artifacts_SHA256'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(a.output,row)
    print(json.dumps({'runs':list(runs),'field_records':sum(len(x['results']) for x in runs.values()),'new_READY_scenes':0}))


if __name__=='__main__':main()
