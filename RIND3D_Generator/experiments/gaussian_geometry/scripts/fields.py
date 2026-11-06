"""Matched small-field gates using saved, immutable synthetic scene parameters."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from experiments.gaussian_geometry.models import load,record
from experiments.gaussian_geometry.queries import make_oracle,Limits
from experiments.gaussian_geometry.study import atomic_json,field_benchmark


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ordinals',nargs='+',type=int,default=[0,7,14,21])
    parser.add_argument('--seconds',type=float,default=45.)
    parser.add_argument('--models',nargs='+',default=['gaussian_convex','ellipsoid_union','gaussian_sum','spectral_grf'])
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    config=json.loads((args.input/'run-manifest.json').read_text())['config']
    config['field_seconds']=args.seconds;config['field_resolutions']=[8,32,64,128]
    identity={'input_manifest_sha256':hashlib.sha256((args.input/'run-manifest.json').read_bytes()).hexdigest(),
              'config':config,'ordinals':args.ordinals,'models':args.models,'gate':'complete independent smaller field before next resolution',
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    identity['actual_research_source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parents[1].glob('*.py')}
    manifest=args.output/'field-manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=identity:raise ValueError('Field run identity changed')
    atomic_json(manifest,identity);results=[]
    for model in args.models:
        for ordinal in args.ordinals:
            case=f'{model}-{ordinal:03d}';target=args.output/case;target.mkdir(exist_ok=True);path=target/'result.json'
            if path.exists():results.append(json.loads(path.read_text()));continue
            scene=load(args.input/case/'geometry.json');source_record=json.loads((args.input/case/'result.json').read_text())
            if source_record.get('status')!='RESEARCH_QUERIES_COMPLETE':
                result={'case':case,'status':'INPUT_CASE_INCOMPLETE','fields':[]}
            else:
                assert record(scene)['geometry_id']==source_record['geometry_id']
                source=source_record['source_xyz_synthetic'];start=time.perf_counter()
                oracle=make_oracle(scene,limits=Limits(max_nodes=config['query_node_budget'],max_depth=config['query_depth_limit'],seconds=config['query_seconds']))
                assert oracle.source_clearance(source).status=='FREE'
                setup=time.perf_counter()-start
                case_config=dict(config,field_resolutions=[8,32,64,128])
                result={'case':case,'geometry_id':source_record['geometry_id'],'model':model,'status':'RESEARCH_FIELD_GATES_EVALUATED',
                        'initialization_seconds':setup,'fields':field_benchmark(scene,oracle,source,target,case_config),
                        'READY_published':False,'resolution_1024_attempted':False}
            atomic_json(path,result);results.append(result)
            print(json.dumps({'case':case,'fields':[(v['resolution'],v['complete'],v.get('independent_check',{}).get('complete')) for v in result['fields']]}),flush=True)
    atomic_json(args.output/'summary.json',{'cases':results,'production_scenes':0})


if __name__=='__main__':main()
