"""Two explicit larger-budget controls; never overwrite the matched 45s study.

These controls test whether incomplete results are budget effects. They do not
advance to 1024 or establish complete production geometry acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.gaussian_geometry.models import load
from experiments.gaussian_geometry.queries import make_oracle,Limits
from experiments.gaussian_geometry.study import atomic_json
from experiments.gaussian_geometry.reference import Reference
from experiments.gaussian_geometry.render import small_field


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((args.input/'run-manifest.json').read_text())['config'];rows=[]
    identity={'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'input_manifest_sha256':hashlib.sha256((args.input/'run-manifest.json').read_bytes()).hexdigest(),
              'field_seconds':180,'primary_nodes':8192,'primary_seconds':8,
              'independent_nodes':8192,'independent_seconds':8,
              'note':'explicit field and per-query budget increase only; unresolved remains non-physical'}
    manifest=args.output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=identity:raise ValueError('Extended control identity changed')
    atomic_json(manifest,identity)
    for case,n in [('ellipsoid_union-014',32),('spectral_grf-007',8)]:
        path=args.output/case;path.mkdir(exist_ok=True);out=path/'result.json'
        if out.exists():rows.append(json.loads(out.read_text()));continue
        scene=load(args.input/case/'geometry.json');original=json.loads((args.input/case/'result.json').read_text())
        oracle=make_oracle(scene,limits=Limits(max_nodes=8192,seconds=8.))
        source=original['source_xyz_synthetic'];value=small_field(oracle,source,n,180.)
        primary={k:value.pop(k) for k in ('response','solid','valid')};np.savez_compressed(path/f'field-{n}.npz',**primary)
        value.update(resolution=n,storage_bytes=(path/f'field-{n}.npz').stat().st_size)
        if value['complete']:
            reference=small_field(oracle,source,n,180.,independent=Reference(scene,max_nodes=8192,seconds=8.))
            common=primary['valid']&reference['valid']
            mismatch=int(np.count_nonzero((primary['response']!=reference['response'])&common)+np.count_nonzero((primary['solid']!=reference['solid'])&common))
            value['independent_check']={k:v for k,v in reference.items() if k not in ('response','solid','valid')}
            value['independent_check'].update(compared_voxels=int(common.sum()),mismatched_response_or_solid_bits=mismatch)
            assert mismatch==0
        fields=[value]
        row={'case':case,'fields':fields,'production':False};atomic_json(out,row);rows.append(row)
    atomic_json(args.output/'summary.json',{'cases':rows,'original_incomplete_artifacts_unchanged':True})


if __name__=='__main__':main()
