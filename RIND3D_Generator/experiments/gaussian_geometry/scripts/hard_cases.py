"""Reproduce unresolved small-field rays at original and increased budgets.

A resolved replay of a single query does not complete its original field.
Frozen incomplete artifacts are never changed.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiments.gaussian_geometry.models import load
from experiments.gaussian_geometry.queries import FieldOracle,Limits
from experiments.gaussian_geometry.reference import Reference
from experiments.gaussian_geometry.study import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--fields',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    rows=[]
    for record in json.loads((args.fields/'summary.json').read_text())['cases']:
        case=record['case'];original=json.loads((args.input/case/'result.json').read_text())
        scene=load(args.input/case/'geometry.json');source=original['source_xyz_synthetic']
        for field in record['fields']:
            for backend,block in [('primary',field),('reference',field.get('independent_check',{}))]:
                for example in block.get('unresolved_examples',[])[:2]:
                    receiver=(np.array(example['voxel'])+.5)/field['resolution'];trials=[]
                    for nodes,seconds in [(512 if backend=='primary' else 2048,.5 if backend=='primary' else 2.),(8192,8.)]:
                        oracle=FieldOracle(scene,Limits(max_nodes=nodes,seconds=seconds)) if backend=='primary' else Reference(scene,max_nodes=nodes,seconds=seconds)
                        start=time.perf_counter();cpu=time.process_time()
                        d=oracle.membership(receiver) if example['stage']=='membership' else oracle.segment(source,receiver)
                        trials.append({'node_budget':nodes,'seconds_budget':seconds,'decision':asdict(d),
                                       'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,
                                       'limit_diagnosis':'node_limit' if d.nodes>=nodes else 'resolved' if d.certified else 'time_or_depth_or_precision',
                                       'work':dict(getattr(oracle,'work',{}))})
                    rows.append({'case':case,'resolution':field['resolution'],'backend':backend,'original_example':example,
                                 'receiver_xyz':receiver.tolist(),'trials':trials})
    atomic_json(args.output,{'cases':rows,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            'fields_summary_sha256':hashlib.sha256((args.fields/'summary.json').read_bytes()).hexdigest(),
                            'claim':'bounded individual-query replay only; original incomplete fields remain incomplete'})


if __name__=='__main__':main()
