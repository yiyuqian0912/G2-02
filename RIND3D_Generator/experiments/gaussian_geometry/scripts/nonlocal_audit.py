"""Corrected, per-size bounded external-occlusion audit on frozen research inputs.

No production scenes or private production sources are read. This does not
estimate full-window response ratios or fulfill any production window quota.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.gaussian_geometry.models import load
from experiments.gaussian_geometry.queries import make_oracle,ConvexOracle
from experiments.gaussian_geometry.study import atomic_json,window_diagnostic


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--capped-input',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds-per-side',type=float,default=8.)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).parents[1]
    identity={'source_sha256':{str(x.relative_to(root)):hashlib.sha256(x.read_bytes()).hexdigest()
                              for x in [Path(__file__),root/'study.py',root/'queries.py']},
              'input':str(args.input.resolve()),'capped_input':str(args.capped_input.resolve()) if args.capped_input else None,
              'seconds_per_side':args.seconds_per_side,'ordinals':[0,7,14,21],
              'seed_protocol':'master_seed + ordinal*31 + side*65537',
              'reason_for_new_audit':'proposal/probe metadata incorrectly consumed old global shadow-attribution cap; old evidence preserved'}
    manifest=args.output/'audit-manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=identity:raise ValueError('Audit identity changed; choose a new output')
    atomic_json(manifest,identity);rows=[]
    for base in [args.input]+([args.capped_input] if args.capped_input else []):
        summary=json.loads((base/'summary.json').read_text())
        cfg=json.loads((args.input/'run-manifest.json').read_text())['config']
        for original in summary['cases']:
            if original['ordinal'] not in identity['ordinals']:continue
            case=original['case'];tag=original['model'];out=args.output/f'{tag}-{original["ordinal"]:03d}.json'
            if out.exists():rows.append(json.loads(out.read_text()));continue
            geometry_path=base/case/'geometry.json';scene=load(geometry_path)
            source=np.array(original['source_xyz_synthetic']);oracle=make_oracle(scene)
            assert oracle.membership(source).status=='FREE' and oracle.source_clearance(source).status=='FREE'
            row={'case':case,'model':tag,'ordinal':original['ordinal'],
                 'geometry_sha256':hashlib.sha256(geometry_path.read_bytes()).hexdigest(),
                 'source_scope':'new synthetic research source, never a v0.1 hidden source',
                 'sides':{}}
            for side in (64,128,256):
                row['sides'][str(side)]=window_diagnostic(oracle,source,cfg['master_seed']+original['ordinal']*31+side*65537,
                                                        args.seconds_per_side,sides=(side,))
            atomic_json(out,row);rows.append(row)
            print(json.dumps({'model':tag,'ordinal':original['ordinal'],
                              'informative_by_size':{s:r['counts'].get(f'side_{s}_informative',0) for s,r in row['sides'].items()}}),flush=True)
    # Original certified-family controls, regenerated from deterministic public
    # recipes only. Reuse their unchanged canonical physics and dual proof.
    from rind3d.lite_generators import generate
    from rind3d.rendering.context import GeometryContext
    from rind3d.lite_acceptance import verify_geometry_proof,sample_source
    for family in ('rotated_boxes','corridors','multiscale_convex'):
        for ordinal in (0,1):
            out=args.output/f'control-{family}-{ordinal}.json'
            if out.exists():rows.append(json.loads(out.read_text()));continue
            bundle,_=generate(20261105,ordinal,family);context=GeometryContext(bundle)
            proof=context.geometry_proof();verify_geometry_proof(bundle,proof)
            source,_=sample_source(context,20261105,ordinal)
            oracle=ConvexOracle.__new__(ConvexOracle);oracle.context=context;oracle.work=Counter();oracle.proof=proof
            row={'case':f'{family}-{ordinal}','model':'control_'+family,'ordinal':ordinal,
                 'geometry_id':context.geometry_id,'sides':{}}
            for side in (64,128,256):
                row['sides'][str(side)]=window_diagnostic(oracle,source,20261105+ordinal*31+side*65537,
                                                        args.seconds_per_side,sides=(side,))
            atomic_json(out,row);rows.append(row)
    atomic_json(args.output/'summary.json',{'cases':rows,'READY_scenes':0,
                'claim':'proved external-blocking examples, not an unbiased estimate of external-occlusion prevalence'})


if __name__=='__main__':main()
