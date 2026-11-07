"""Compact matched metrics; raw timings, traces and certificates stay ignored."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from ..study import atomic_json


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(rows):
    by=defaultdict(list)
    for row in rows:by[row['ray_id']].append(row)
    values=[float(np.median([v['wall_seconds'] for v in r])) for r in by.values()]
    trial=defaultdict(float)
    for r in rows:trial[r['repetition']]+=r['wall_seconds']
    work=Counter()
    for r in rows:work.update(r['work'])
    repetitions=len(trial)
    result={'unique_queries':len(by),'status_counts':dict(Counter(r[0]['status'] for r in by.values())),
            'sum_per_query_median_seconds':sum(values),'P50_seconds':float(np.quantile(values,.5)),'P95_seconds':float(np.quantile(values,.95)),
            'P99_seconds':float(np.quantile(values,.99)),'max_per_query_median_seconds':max(values),'max_single_trial_seconds':max(r['wall_seconds'] for r in rows),
            'trial_total_seconds':dict(sorted(trial.items())),'CPU_seconds_mean_total':sum(r['CPU_seconds'] for r in rows)/repetitions,
            'nodes_mean_total':sum(r['nodes'] for r in rows)/repetitions,'exp_endpoints_mean_total':sum(r['work'].get('exp_endpoints',r['work'].get('reference_exp_endpoints',0))+r['work'].get('high_precision_exp_endpoints',0) for r in rows)/repetitions,
            'solver_preparation_seconds_mean_total':sum(r.get('preparation_seconds',0) for r in rows)/repetitions,
            'work_mean_total':{k:v/repetitions for k,v in sorted(work.items())},
            'fallback_queries':sum(any(v.get('fallback_used') for v in r) for r in by.values())}
    checked=[r for r in rows if 'checker' in r]
    if checked:
        cv=[float(np.median([v['checker_wall_seconds'] for v in r if 'checker' in v])) for r in by.values() if any('checker' in v for v in r)]
        result['proof']={'checked_unique_queries':len(cv),'all_accepted':all(r['checker']['accepted'] for r in checked),'checker_sum_per_query_median_seconds':sum(cv),
                         'checker_P50_seconds':float(np.quantile(cv,.5)),'checker_P95_seconds':float(np.quantile(cv,.95)),'checker_P99_seconds':float(np.quantile(cv,.99)),
                         'serialization_seconds_mean_total':sum(r['serialization_seconds'] for r in checked)/repetitions,
                         'persistence_seconds_mean_total':sum(r.get('proof_persistence_seconds',0) for r in checked)/repetitions,
                         'proof_bytes_mean_total':sum(r['proof_bytes'] for r in checked)/repetitions,
                         'proof_bytes_P50':float(np.median([r['proof_bytes'] for r in checked])),'proof_bytes_max':max(r['proof_bytes'] for r in checked),
                         'proof_intervals_P50':float(np.median([r['proof_intervals'] for r in checked])),'proof_intervals_max':max(r['proof_intervals'] for r in checked),
                         'operations_mean_total':{k:v/repetitions for k,v in sum((Counter(r['proof_operations']) for r in checked),Counter()).items()},
                         'checker_exp_endpoints_mean_total':sum(r['checker']['work'].get('checker_exp_endpoints',0) for r in checked)/repetitions,
                         'checker_seconds_mean_total':sum(r['checker_wall_seconds'] for r in checked)/repetitions,
                         'checker_CPU_seconds_mean_total':sum(r['checker_CPU_seconds'] for r in checked)/repetitions,
                         'checker_native_seconds_mean_total':sum(r['checker']['seconds'] for r in checked)/repetitions}
        result['proof']['checker_preparation_seconds_mean_total']=sum(r['checker']['preparation_seconds'] for r in checked)/repetitions
    return result


def summarize(summary):
    groups={'all':lambda r:True,'MISS':lambda r:r['expected']=='MISS','HIT':lambda r:r['expected']=='HIT',
            'expensive_MISS':lambda r:'expensive_MISS' in r['strata'],'median_MISS':lambda r:'median_MISS' in r['strata'],
            'cheap_MISS':lambda r:'cheap_MISS' in r['strata'],'near_threshold_MISS':lambda r:'near_threshold_MISS_heuristic' in r['strata']}
    groups.update({case:(lambda r,c=case:r['case']==c) for case in sorted({r['case'] for r in summary['results']})})
    result={}
    for group,predicate in groups.items():
        methods={}
        for method in summary['identity']['methods']:
            rows=[r for r in summary['results'] if r['method']==method and predicate(r)]
            if rows:methods[method]=metric(rows)
        if methods:result[group]=methods
    # Per-ray medians preserve matched identities without committing large logs.
    by=defaultdict(lambda:defaultdict(list))
    for row in summary['results']:by[row['ray_id']][row['method']].append(row)
    compact=[]
    for ray_id,methods in sorted(by.items()):
        example=next(iter(methods.values()))[0]
        item={k:example[k] for k in ('ray_id','case','split','expected','strata')};item['methods']={}
        for method,rows in methods.items():
            item['methods'][method]={'status':rows[0]['status'],'seconds_median':float(np.median([r['wall_seconds'] for r in rows])),
                                    'nodes':rows[0]['nodes'],'depth':rows[0]['depth'],'exp_endpoints':rows[0]['work'].get('exp_endpoints',rows[0]['work'].get('reference_exp_endpoints',0))}
            if 'checker' in rows[0]:item['methods'][method].update(checker_seconds_median=float(np.median([r['checker_wall_seconds'] for r in rows])),proof_bytes=rows[0]['proof_bytes'],proof_intervals=rows[0]['proof_intervals'])
        compact.append(item)
    return {'identity':summary['identity'],'groups':result,'matched_ray_medians':compact,'cold_setup_records':summary['setups'],
            'orchestration_wall_seconds':summary['orchestration_wall_seconds'],'orchestration_CPU_seconds':summary['orchestration_CPU_seconds']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='+',type=Path,required=True);p.add_argument('--fields',type=Path,nargs='*',default=[])
    p.add_argument('--adversarial',type=Path);p.add_argument('--third',type=Path);p.add_argument('--root',type=Path,default=Path('outputs/gaussian-b2-max-cert-v1'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result={'experiment':'RIND3D B2 continuous one-dimensional maximum proofs','base_commit':'d67215cce0f6dcfccb504813da5c1e52feb18547',
            'runs':{},'raw_artifact_hashes':{},'new_READY_scenes':0,'complete_1024_fields':0,'B2_ready_capable':False,'existing_independent_renderer_retained':True}
    for directory in a.runs:
        path=directory/'summary.json';data=json.loads(path.read_text());result['runs'][directory.name]=summarize(data);result['raw_artifact_hashes'][str(path)]=sha(path)
        for row in data['results']:
            if row.get('proof_SHA256'):
                cert=directory/row.get('proof_file',f'certificate-{row["ray_id"]}-{row["repetition"]}.json');assert sha(cert)==row['proof_SHA256']
    for key,path in [('baseline_characterization',a.root/'baseline-characterization.json'),('preimplementation_diagnostics',a.root/'preimplementation-diagnostic-summary.json'),('derivative_diagnostics',a.root/'derivative-diagnostic.json'),
                     ('adversarial',a.adversarial/'summary.json' if a.adversarial else None),('Arb_frozen_queries',a.third)]:
        if path:
            data=json.loads(path.read_text());result['raw_artifact_hashes'][str(path)]=sha(path)
            if key=='derivative_diagnostics':data={'summary':data['summary'],'method':data['method']}
            result[key]=data
    result['field_runs']={}
    for directory in a.fields:
        path=directory/'summary.json';data=json.loads(path.read_text());result['raw_artifact_hashes'][str(path)]=sha(path)
        decomposition=[];reference={}
        for row in data['results']:
            if row['method']=='A_primary':
                archive=directory/f'{row["case"]}-{row["resolution"]}-{row["method"]}.npz'
                reference[(row['case'],row['resolution'])]=np.load(archive)['evaluation_path']
        for row in data['results']:
            assert row['complete'] and row['valid_voxels']==row['resolution']**3
            assert row['work'].get('MISS_proof_rejections',0)==0
            assert sha(directory/f'{row["case"]}-{row["resolution"]}-{row["method"]}.npz')==row['archive_SHA256']
            arrays=np.load(directory/f'{row["case"]}-{row["resolution"]}-{row["method"]}.npz');tags=arrays['evaluation_path'];times=arrays['ray_nanoseconds']
            ref=reference[(row['case'],row['resolution'])];same=bool(np.array_equal(tags,ref))
            if row['method']!='A_independent':assert same,'Unexpected change to scalar support/traversal'
            decomposition.append({'case':row['case'],'resolution':row['resolution'],'method':row['method'],
                                  'MISS_queries':int((tags==6).sum()),'HIT_queries':int((tags==7).sum()),
                                  'MISS_scalar_seconds':float(times[tags==6].sum()/1e9),'HIT_scalar_seconds':float(times[tags==7].sum()/1e9),
                                  'same_primary_scalar_coordinate_and_status_map':same})
        result['field_runs'][directory.name]={**data,'scalar_decomposition':decomposition}
    atomic_json(a.output,result);print(json.dumps({'output':str(a.output),'bytes':a.output.stat().st_size,'ray_runs':len(a.runs)}))


if __name__=='__main__':main()
