"""Recover only the four frozen synthetic research inputs, not dataset scenes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ..generators import generate
from ..models import record,save
from ..queries import FieldOracle,Limits
from ..study import atomic_json,sample_source


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    frozen=json.loads((Path(__file__).parents[1]/'configs/b2-angular-frozen-v1.json').read_text());cfg=frozen['original_config']
    args.output.mkdir(parents=True,exist_ok=True)
    for ordinal,expected in frozen['expected'].items():
        i=int(ordinal);name=f'gaussian_sum-{i:03d}';folder=args.output/name;folder.mkdir(exist_ok=True)
        targets=list(cfg['occupancy_targets'].values());target=targets[(i//len(cfg['styles']))%len(targets)]
        scene=generate('gaussian_sum',cfg['master_seed'],i,target,cfg)
        assert record(scene)['geometry_id']==expected['geometry_id'],'Generator/numerical environment differs from frozen inputs'
        geometry=folder/'geometry.json'
        if geometry.exists():assert hashlib.sha256(geometry.read_bytes()).hexdigest()==expected['geometry_json_SHA256']
        else:save(geometry,scene)
        assert hashlib.sha256(geometry.read_bytes()).hexdigest()==expected['geometry_json_SHA256']
        oracle=FieldOracle(scene,Limits(max_nodes=cfg['query_node_budget'],max_depth=cfg['query_depth_limit'],seconds=cfg['query_seconds']))
        source,report=sample_source(oracle,cfg['master_seed']+i);assert source is not None
        digest=hashlib.sha256(np.asarray(source,dtype='<f8').tobytes()).hexdigest();assert digest==expected['source_coordinates_SHA256']
        result={'case':name,'geometry_id':expected['geometry_id'],'source_xyz_synthetic':source.tolist(),'source_sampling':report,
                'status':'EXPERIMENTAL_INPUT_ONLY','ready_capable':False}
        path=folder/'result.json'
        if path.exists():assert json.loads(path.read_text())==result
        else:atomic_json(path,result)
    atomic_json(args.output/'run-manifest.json',{'frozen_research_recovery':frozen,'original_config':cfg})
    print(json.dumps({'frozen_research_cases':4,'geometry_and_source_hashes_identical':True,'dataset_scenes_generated':0}))


if __name__=='__main__':main()
