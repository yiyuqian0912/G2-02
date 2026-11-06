"""Compact measured B2 results; excludes synthetic light coordinates/raw arrays."""
from collections import Counter
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import statistics

import numpy as np

from ..study import atomic_json


def dist(v):
    return dict(zip(('min','p50','p95','p99','max'),map(float,np.percentile(list(v),[0,50,95,99,100]))))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();base=args.root
    read=lambda name:json.loads((base/name).read_text())
    rows=read('matched-v2/summary.json')['cases'];old=sum(r['python_ray_seconds'] for r in rows)
    result={'experiment':'RIND3D B2 native foundation','dates':'2026-11-05–06','branch':'yushu/gaussian-b2-native',
       'base_commit':'bf4caf1496412c6f1df085cd084d0939f5a31026','new_READY_scenes':0,'new_1024_fields':0,
       'matched_cases':len(rows),'matched_rays':sum(r['rays'] for r in rows),'python_total_ray_seconds':old,
       'python_total_ray_CPU_seconds':sum(r['python_ray_CPU_seconds'] for r in rows),'modes':{},
       'runtime':{'python':platform.python_version(),'platform':platform.platform(),
          'versions':{x:importlib.metadata.version(x) for x in ('numpy','python-flint','gmpy2','scipy')},
          'threads':1,'new_engine_GPU_used':False,'cpu_model':next(x.split(':')[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name'))},
       'native_build':read('build/build-identity.json'),'input_identity':read('matched-v2/manifest.json')}
    for name in rows[0]['modes']:
        rr=[r['modes'][name] for r in rows];time=sum(statistics.median(r['batch_wall_seconds']) for r in rr);setup=sum(r['setup_seconds'] for r in rr);counts=Counter();work=Counter()
        for r in rr:counts.update(r['status_counts']);work.update(r['work_one_batch'])
        result['modes'][name]={'batch_wall_seconds_sum':time,
          'batch_CPU_seconds_sum':sum(statistics.median(r['batch_CPU_seconds']) for r in rr),
          'setup_seconds_sum':setup,'speedup_vs_python':old/time,
          'speedup_including_native_setup':old/(time+setup),'status_counts':dict(counts),'work_one_batch_per_case_sum':dict(work),
          'scalar_quantile_scope':'distribution across cases of each within-case quantile, not pooled ray quantiles',
          'scalar_case_median_seconds':{q:dist(r['scalar_seconds'][q] for r in rr) for q in ('p50','p95','p99','max')},
          'point_batch_seconds_sum':sum(r['point_batch_seconds'] for r in rr),'unresolved':sum(r['unresolved'] for r in rr),
          'disagreements':sum(r['disagreements'] for r in rr)}
    result['python_scalar_case_quantiles']={q:dist(r['python_scalar_seconds'][q] for r in rows) for q in ('p50','p95','p99','max')}
    result['python_point_seconds_sum']=sum(r['python_point_seconds'] for r in rows)
    result['matched_process_peak_RSS_MiB']=max(r['process_peak_RSS_MiB'] for r in rows)
    equal=0
    for row in rows:
        a=np.load(base/'matched-v1'/row['case']/'queries.npz');b=np.load(base/'matched-v2'/row['case']/'queries.npz')
        assert all(np.array_equal(a[k],b[k]) for k in ('source','receivers','points'));equal+=1
    result['prior_query_coordinate_sets_identical']=equal
    result['fields']=read('fields-v3/summary.json')['cases']
    result['field128']=read('fields-128-v1/summary.json')['cases']
    result['field_ablation']=read('field-ablation-v1.json')
    result['hard_before']=read('hard-before-v3.json');result['hard_after']=read('hard-after-v3.json')
    result['controls']=read('controlled/proofs.json')
    result['adversarial']=read('adversarial-v2/summary.json')['cases']
    # Unknown examples are synthetic, but omit coordinates from compact results.
    for row in result['adversarial']:
        for block in row['backends'].values():block.pop('unknown_examples',None)
    result['proxy_experiment']=read('proxy-experiment-v2/summary.json')['cases']
    result['generation']={}
    for version in ('v1','v2'):
        data=read(f'generation-{version}/summary.json')['cases'];status=Counter(r['status'] for r in data)
        group={'attempted':len(data),'statuses':dict(status),'regimes':{},'proposal_count':sum(len(r['proposals']) for r in data),
               'wall_seconds_sum':sum(r['total_seconds'] for r in data)}
        for target in (.001,.01,.05,.15):
            rr=[r for r in data if r['target']==target];accepted=[r for r in rr if 'quality' in r]
            group['regimes'][str(target)]={'attempted':len(rr),'accepted':len(accepted),
                 'realized_occupancy_estimate':dist(r['quality']['occupied_ratio_estimate'] for r in accepted) if accepted else None,
                 'realized_occupancy_mean_estimate':statistics.mean(r['quality']['occupied_ratio_estimate'] for r in accepted) if accepted else None,
                 'fusion_only_solid_fraction_estimate':dist(r['quality'].get('fusion_only_solid_fraction_estimate',0) for r in accepted) if accepted else None,
                 'components_6_connected_estimate':dist(r['quality']['components_6_connected_estimate'] for r in accepted) if accepted else None}
        group['cases']=[{'ordinal':r['ordinal'],'status':r['status'],'target':r['target'],'proposals':len(r['proposals']),
                         'quality':r.get('quality'),'selected_attempt':r.get('selected_attempt')} for r in data]
        result['generation'][version]=group
    result['native_instruction_profile']={}
    for mode in ('primary','reference'):
        text=(base/f'native-profile/{mode}-annotated.txt').read_text();total=re.search(r'([\d,]+) \(100\.0%\)  PROGRAM TOTALS',text)
        exp=re.search(r'([\d,]+) \(\s*([\d.]+)%\).*?:mpfr_exp \[',text)
        result['native_instruction_profile'][mode]={'instructions':int(total[1].replace(',','')),
          'mpfr_exp_inclusive_instructions':int(exp[1].replace(',','')),'mpfr_exp_inclusive_instruction_percent':float(exp[2]),
          'scope':'128 matched rays, standalone same-source O3 harness; inclusive instruction counts, not exclusive wall time'}
    result['native_instruction_profile']['identity']=read('native-profile/identity.json')
    result['baseline_profile']=read('profile/summary.json')
    for r in result['baseline_profile']['cases']:r['profile_functions']=r['profile_functions'][:24]
    result['failed_diagnostics']=[{'run':'adversarial-v1','reason':'assertion incorrectly excluded old BUDGET_EXHAUSTED, no contradictory certified label'},
       {'run':'proxy-experiment-v1','reason':'depth-5 auxiliary box cover exceeded unchanged native 64-component capability; retained as unsupported experiment'}]
    files=['matched-v2/manifest.json','fields-v3/manifest.json','fields-128-v1/manifest.json','generation-v2/manifest.json',
           'adversarial-v2/manifest.json','field-ablation-v1.json','proxy-experiment-v2/summary.json']
    result['artifact_sha256']={name:hashlib.sha256((base/name).read_bytes()).hexdigest() for name in files}
    atomic_json(args.output,result)


if __name__=='__main__':main()
