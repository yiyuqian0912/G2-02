"""Join the complete attempt ledger to the pinned 807-channel population and historical diagnostics."""
import argparse
import base64
from collections import Counter
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import evidence
from compare_canary import fl_label


def lines(p):
    return [json.loads(line) for line in p.read_text().splitlines()]


def quant(v):
    v=list(v)
    if not v: return None
    q=np.quantile(v,[0,.5,.95,.99,1])
    return dict(zip(("min","p50","p95","p99","max"),map(float,q)))


def probs(w):
    records=np.frombuffer(base64.b64decode(w,validate=True),dtype=[("c","<u4"),("p","<f8")])
    return records["c"],records["p"]


def tv(c,p,d,q):
    both,i,j=np.intersect1d(c,d,return_indices=True,assume_unique=True)
    return float(max(0,1-np.minimum(p[i],q[j]).sum(dtype=np.float64)))


def group(rs):
    yes=[r for r in rs if r["status"]=="CERTIFIED"]
    return dict(channels=len(rs),first_certified=sum(r['initial']=='CERTIFIED' for r in rs),certified=len(yes),
            total_cpu_s=sum(r["cpu_s"] for r in rs),attempt_wall_s=quant(r['wall_s'] for r in rs),
            minority_fraction=quant(r['minority_fraction'] for r in rs),support=quant(r["support"] for r in yes),
            entropy_nats=quant(r["entropy_nats"] for r in yes),tv=quant(r["bound"] for r in yes))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('root','work','output'): ap.add_argument(k,type=Path)
    ap.add_argument('manifest_sha')
    a=ap.parse_args()
    source=(a.root/'manifest.json').read_bytes()
    assert evidence.digest(source)==a.manifest_sha
    plan=json.loads(source)
    population={r['key']:r for r in lines(HERE/'sampling_cost/results/channels.jsonl')}
    hist=a.work/'xlf_cbc/v301_canary_10s'
    old={r['key']:r for r in json.loads((hist/'compare_canary.json').read_bytes())}
    runs={r['task']['key']:r for r in lines(hist/'runs.jsonl')}
    assert set(old)==set(runs)==set(plan['tasks']) and set(old).issubset(population)
    rows=[]
    attempts=[]
    digest_map={}
    for key,ident in plan["tasks"].items():
        base=population[key]
        assert base['crop']==ident['observation_xyz_sha256'] and base['family']==ident['family'] and base['L']==ident['side']
        prev=json.loads((hist/'results'/f'{key}.json').read_bytes())
        for k in ('geometry_sha256','observation_xyz_sha256','origin','side','resolution','slot','window','channel'):
            assert prev['problem'][k]==ident[k]
        task=a.root/'tasks'/key
        history=[]
        for path in sorted(task.glob('*/receipt.json')):
            rec=json.loads(path.read_bytes())
            start=json.loads((path.parent/'start.json').read_bytes())
            assert start['manifest_sha']==a.manifest_sha and start['key']==key
            at=dict(key=key,attempt=int(path.parent.name),budget=start['seconds'],cap=start['cap'],**rec)
            attempts.append(at)
            history.append(at)
        assert history
        first,last=history[0],history[-1]
        r=dict(key=key,initial=first['status'],status=last['status'],family=ident['family'],side=ident['side'],
               slot=ident['slot'],minority=base['minority'],minority_fraction=base['fraction'],
               attempts=len(history),wall_s=sum(t['wall'] for t in history),cpu_s=sum(t['cpu'] for t in history),
               reason=last['reason'],old_status=old[key]['status'],fl_status=old[key]['fl_status'],
               old_wall=old[key]['wall'],old_cpu=(runs[key]['summary'] or {}).get('cpu'),
               fl_seconds=old[key]['fl_seconds'])
        if last['status']=='CERTIFIED':
            label=path.parent/'label.json'
            issued=evidence.read_issued(label,ident,plan['build'],last['sha256'])
            digest_map[key]=last['sha256']
            c,p=probs(issued['wire'])
            r.update(support=len(p),bound=float(Q(issued['tv'])),entropy_nats=last['entropy_nats'],wire=issued['wire_sha256'])
            if old[key]['status']=='CERTIFIED':
                d,q=probs(prev['wire'])
                r['old_tv']=tv(c,p,d,q)
                r['old_wire_equal']=(prev['wire']==issued['wire'])
                r['old_pair_excess']=r['old_tv']-r['bound']-old[key]['tv']
            fast,meta=fl_label(a.work/'release/canary',ident['slot'],ident['window'],ident['channel'])
            if fast is not None:
                d=np.fromiter(fast, dtype=np.int64)
                q=np.fromiter(fast.values(),dtype=np.float64)
                r['fl_tv']=tv(c,p,d,q)
                r['fl_pair_excess']=r['fl_tv']-r['bound']-float(meta['tv_bound'])
        rows.append(r)
    def groups(field):
        return {str(k):group([r for r in rows if r[field]==k]) for k in sorted(set(r[field] for r in rows))}
    for r in rows:
        f=r['minority_fraction']
        r['minority_bin']=('one voxel' if r['minority']==1 else '<1%' if f<.01 else '1-10%' if f<.1 else '10-50%')
    flat=[]
    for b in sorted({v['budget'] for v in attempts}):
        rs=[t for t in attempts if t['budget']==b]
        flat.append(dict(seconds=b,n=len(rs),outcomes=dict(Counter(t['status'] for t in rs)),
                reasons=dict(Counter(t['reason'] for t in rs)),cpu_sum=sum(t['cpu'] for t in rs),
                wall_sum=sum(t['wall'] for t in rs),wall_s=quant(t['wall'] for t in rs),cpu_s=quant(t['cpu'] for t in rs),
                rss_kib=quant(t['rss_kib'] for t in rs)))
    sessions=[json.loads(p.read_bytes()) for p in sorted(a.root.glob('session*.json'))]
    golden=[r for r in rows if 'old_tv' in r]
    fast=[r for r in rows if 'fl_tv' in r]
    histcpu=[r['old_cpu'] for r in rows if r['old_cpu'] is not None]
    diagnostics=dict(old_matched=len(golden),old_wire_identical=sum(r['old_wire_equal'] for r in golden),
        old_tv=quant(r['old_tv'] for r in golden),old_excess_over_1e_12=[r['key'] for r in golden if r['old_pair_excess']>1e-12],
        fl_matched=len(fast),fl_tv=quant(r['fl_tv'] for r in fast),
        fl_excess_over_1e_12=[r['key'] for r in fast if r['fl_pair_excess']>1e-12],
        old_first_only=[r['key'] for r in rows if r['initial']!='CERTIFIED' and r['old_status']=='CERTIFIED'],
        new_first_only=[r['key'] for r in rows if r['initial']=='CERTIFIED' and r['old_status']!='CERTIFIED'],
        old_status=dict(Counter(r['old_status'] for r in rows)),old_process_wall=sum(r['old_wall'] for r in rows),
        old_cpu_sum_for_available= sum(histcpu),old_cpu_available=len(histcpu),
        fl_status=dict(Counter(r['fl_status'] for r in rows)),fl_seconds_sum=sum(r['fl_seconds'] or 0 for r in rows),
        fl_seconds_available=sum(r['fl_seconds'] is not None for r in rows))
    summary=dict(manifest_sha=a.manifest_sha,overall=group(rows),tiers=flat,
        attempt_wall_s=quant(t['wall'] for t in attempts),attempt_cpu_s=quant(t['cpu'] for t in attempts),
        rss_kib=quant(t['rss_kib'] for t in attempts),family=groups('family'),side=groups('side'),
        minority=groups('minority_bin'),sessions=sessions,comparison=diagnostics,
        unfinished=[r for r in rows if r['status']!='CERTIFIED'])
    a.output.mkdir(parents=True,exist_ok=True)
    for name,data in [('summary.json',summary),('issued-sha256.json',digest_map),('build.json',plan['build'])]:
        (a.output/name).write_text(json.dumps(data,sort_keys=True,indent=2)+'\n')
    for name,data in [('channels.jsonl',rows),('attempts.jsonl',attempts)]:
        with (a.output/name).open('w') as f:
            for r in data: f.write(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n')


if __name__=='__main__':
    main()
