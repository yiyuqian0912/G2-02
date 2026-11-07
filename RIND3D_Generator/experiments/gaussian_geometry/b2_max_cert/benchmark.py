"""Resumable matched frozen-ray comparison; certificate costs remain explicit."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from ..models import load,record
from ..b2_native.api import NativeOracle
from ..study import atomic_json
from .api import MaximumOracle,CertificateChecker,dumps
from .build import sources


def validate_checkpoint(saved,row,method,repetition,certificate_path):
    if any(saved.get(k)!=v for k,v in {'ray_id':row['ray_id'],'case':row['case'],'expected':row['expected'],'method':method,'repetition':repetition}.items()):
        raise ValueError('Conflicting ray checkpoint identity')
    if saved.get('status') not in (row['expected'],'UNKNOWN'):raise ValueError('Invalid physical checkpoint')
    if method.endswith('_proof') and saved['status']=='MISS':
        if not saved.get('checker',{}).get('accepted'):raise ValueError('Failed proof checkpoint is not completed work')
        if not certificate_path.exists() or hashlib.sha256(certificate_path.read_bytes()).hexdigest()!=saved.get('proof_SHA256'):
            raise ValueError('Missing or changed certificate checkpoint')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('outputs/gaussian-b2-max-cert-v1'))
    p.add_argument('--input',type=Path,default=Path('outputs/gaussian-geometry-v1/distribution-v2'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['development','evaluation','all'],default='development');p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--methods',nargs='+',default=['A_primary','A_independent','B_derivative','C_stationary','D_hybrid','D_proof','E_lean','E_lean_proof'])
    p.add_argument('--seconds',type=float,default=2.);p.add_argument('--max-nodes',type=int,default=16384);p.add_argument('--precision',type=int,default=256)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);fixture=a.root/'frozen-rays.json';f=json.loads(fixture.read_text())
    selected=[r for r in f['rays'] if a.split=='all' or r['split']==a.split]
    here=Path(__file__).parent
    identity={'source_SHA256':sources(),'Python_driver_SHA256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (here/'benchmark.py',here/'api.py')},
              'fixture_SHA256':hashlib.sha256(fixture.read_bytes()).hexdigest(),'split':a.split,'repeats':a.repeats,'methods':a.methods,
              'seconds':a.seconds,'max_nodes':a.max_nodes,'max_depth':40,'fallback':False,'checker_precision':a.precision}
    manifest=a.output/'identity.json'
    if manifest.exists():assert json.loads(manifest.read_text())==identity,'Use a new output directory after a source/config change'
    else:atomic_json(manifest,identity)
    results=[];setups=[];started=time.perf_counter();cpu_start=time.process_time()
    for case in sorted({r['case'] for r in selected}):
        start=time.perf_counter();scene=load(a.input/case/'geometry.json');input_time=time.perf_counter()-start
        expected_scene=next(r for r in f['scenes'] if r['case']==case)
        assert record(scene)['geometry_id']==expected_scene['geometry_id'],'Frozen canonical geometry changed'
        if 'geometry_file_SHA256' in expected_scene:
            assert hashlib.sha256((a.input/case/'geometry.json').read_bytes()).hexdigest()==expected_scene['geometry_file_SHA256']
        engines={};setup={}
        for method in a.methods:
            t=time.perf_counter()
            if method.startswith('A_'):engine=NativeOracle(scene,independent=method=='A_independent',max_nodes=a.max_nodes,query_seconds=a.seconds)
            else:engine=MaximumOracle(scene,method={'B_derivative':1,'C_stationary':2,'D_hybrid':3,'D_proof':3,'E_lean':3,'E_lean_proof':3}[method],max_nodes=a.max_nodes,max_depth=40,seconds=a.seconds,endpoint_probes=not method.startswith('E_'))
            engines[method]=engine;setup[method]=time.perf_counter()-t
        t=time.perf_counter();checker=CertificateChecker(scene);setup['checker']=time.perf_counter()-t
        rows=[r for r in selected if r['case']==case]
        clearances={}
        for independent in (False,True):
            t=time.perf_counter();oracle=NativeOracle(scene,independent=independent);decision=oracle.source_clearance(rows[0]['source_xyz'])
            assert decision.status=='FREE';clearances['independent' if independent else 'primary']={'status':decision.status,'seconds':time.perf_counter()-t}
        setups.append({'case':case,'input_load_seconds':input_time,'engine_setup_seconds':setup,'original_source_clearance':clearances})
        for row in rows:
            for repetition in range(a.repeats):
                # Rotate order deterministically, limiting systematic warm/thermal order bias.
                methods=a.methods[repetition%len(a.methods):]+a.methods[:repetition%len(a.methods)]
                for method in methods:
                    path=a.output/f'{row["ray_id"]}-{method}-{repetition}.json'
                    proof_path=a.output/f'certificate-{row["ray_id"]}-{method}-{repetition}.json'
                    if path.exists():
                        saved=json.loads(path.read_text());validate_checkpoint(saved,row,method,repetition,proof_path)
                        results.append(saved);continue
                    engine=engines[method];t=time.perf_counter();c=time.process_time()
                    if method.startswith('A_'):
                        previous=engine.work.copy();decision=engine.segment(row['source_xyz'],row['receiver_xyz']);data={'status':decision.status,'reason':decision.reason,'nodes':decision.nodes,'depth':decision.depth,'work':dict(engine.work-previous)}
                    else:data=engine.segment(row['source_xyz'],row['receiver_xyz'],proof=method.endswith('_proof'))
                    api_time=time.perf_counter()-t;cpu=time.process_time()-c;certificate=data.pop('certificate',None)
                    assert data['status'] in (row['expected'],'UNKNOWN'),('OPPOSITE PHYSICAL LABEL',row,method,data)
                    result={'ray_id':row['ray_id'],'case':case,'split':row['split'],'expected':row['expected'],'strata':row['strata'],'method':method,'repetition':repetition,'wall_seconds':api_time,'CPU_seconds':cpu,**data}
                    if method.endswith('_proof') and data['status']=='MISS' and certificate:
                        t=time.perf_counter();encoded=dumps(certificate);serialization=time.perf_counter()-t
                        t=time.perf_counter();c=time.process_time();checked=checker.check(encoded,row['source_xyz'],row['receiver_xyz'],precision=a.precision,seconds=a.seconds)
                        result.update(checker_wall_seconds=time.perf_counter()-t,checker_CPU_seconds=time.process_time()-c,serialization_seconds=serialization,proof_bytes=len(encoded),proof_intervals=len(certificate['partition']),proof_operations=dict(Counter(r['op'] for r in certificate['partition'])),checker=checked)
                        t=time.perf_counter();proof_path.write_bytes(encoded)
                        result['proof_persistence_seconds']=time.perf_counter()-t
                        result['proof_SHA256']=hashlib.sha256(encoded).hexdigest()
                        result['proof_file']=proof_path.name
                        if not checked['accepted']:
                            atomic_json(path,result);raise AssertionError(('CHECKER REJECTED',row['ray_id'],checked))
                    atomic_json(path,result);validate_checkpoint(result,row,method,repetition,proof_path);results.append(result)
        print(json.dumps({'case':case,'rays':len(rows),'records':len(results),'elapsed':time.perf_counter()-started}),flush=True)
    atomic_json(a.output/'summary.json',{'identity':identity,'results':results,'setups':setups,'orchestration_wall_seconds':time.perf_counter()-started,'orchestration_CPU_seconds':time.process_time()-cpu_start,
                'existing_independent_renderer_retained':True,'checker_is_not_an_independent_renderer':True,'new_READY_scenes':0})


if __name__=='__main__':main()
