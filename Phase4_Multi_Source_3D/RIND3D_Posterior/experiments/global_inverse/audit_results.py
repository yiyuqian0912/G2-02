"""Post-run integrity checks, matched wall times and whole-grid TV comparisons.

Reference data are read only here, after discovery. Charge external grid export
only where the solver did not already produce an explicit probability vector.
"""
import argparse
from fractions import Fraction
import gzip
from hashlib import sha256
import json
from pathlib import Path
from math import fsum
import numpy as np
from query import load_records

RUNS=('archive','holdout_random','heldout','heldout_raster','fresh_envelope','fresh_raster','fresh_support')


def archive_grid(path):
    with np.load(path,allow_pickle=False) as z:
        bins,idx=np.unique(z['cell'],return_inverse=True)
        m=np.bincount(idx,weights=z['lo'])
        if not m.sum(): m=np.bincount(idx,weights=z['hi'])
    return dict(zip(bins,m/m.sum()))


def time_summary(values):
    return dict(zip(('p50','p95','max','total'),map(float,[*np.percentile(values,[50,95,100]),sum(values)])))


def evidence_file(directory, name):
    path=directory/(name+'.jsonl')
    return path if path.exists() else path.with_suffix('.jsonl.gz')


def evaluate(folder,comparison):
    summary={}
    historic={r['key']:r for r in load_records(comparison/'tp.jsonl')}
    cpu={}
    for loc in ('reference','fresh_reference'):
        for f in sorted((folder/loc).glob('*cpu.jsonl*')):
            for r in load_records(f):
                key=r['task']['key']
                if key in cpu: raise ValueError('overlapping references')
                cpu[key]=r
    for name in RUNS:
        meshes=folder/('grids' if name=='heldout' else 'grids_'+name)
        lines=[line.split() for line in (folder/('grid_'+name+'.log')).read_text().splitlines()]
        exports={key:(float(t),digest) for key,_,_,t,digest in lines}
        out=[]
        for r in load_records(evidence_file(folder,name)):
            t=r['task']
            key=t['key']
            reference=historic[key] if name=='archive' else cpu[key]
            base=reference.get('wall_s',reference.get('wall'))
            lazy=r.get('grid_status')=='lazy_polytope_queries'
            cost=r['wall_s']+(exports[key][0] if lazy and key in exports else 0)
            row=dict(key=key,family=t['family'],side=t['L'],positive=r['positive'],
                     certified=r['certified_tv_005'],volume_solved=r['positive'] and not r['unknown_pieces'],
                     wall_s=r['wall_s'],delivered_s=cost,reference_s=base,
                     volume_tv_bound=r['tv_upper'],discovery_stop=r['discovery_stop'],
                     finish_stop=r['finish_stop'],unknown_pieces=r['unknown_pieces'])
            if r['certified_tv_005']:
                payload=gzip.decompress((meshes/(key+'.json.gz')).read_bytes())
                assert sha256(payload).hexdigest()==exports[key][1]
                g=json.loads(payload)
                weights=g['weights']
                cells=g['cells']
                assert cells==sorted(set(cells)) and min(cells)>=0 and max(cells)<128**3
                assert sum(weights)==g['denominator']>0 and min(weights)>0
                assert max(weights)*128**3<=1<<g['scale_bits']
                assert g['geometry_sha256']==r['geometry_sha256'] and g['task']==t
                bound=Fraction(g['tv_numerator'],g['tv_denominator'])
                assert 0<=bound<=Fraction(1,20)
                q=dict(zip(cells,[w/g['denominator'] for w in weights]))
                if 'probabilities' in r:
                    # Internal representation is rational, including the exact volume-solved cases.
                    exact={int(i):Fraction(v) for i,v in r['probabilities'].items()}
                    assert sum(exact.values())==1
                    assert set(q)<=set(exact)
                    q={i:float(v) for i,v in exact.items()}
                if name=='archive': ref=archive_grid(comparison/'labels'/(key+'_tp.npz'))
                else:
                    directory=folder/('fresh_reference' if name.startswith('fresh') else 'reference')
                    with np.load(directory/(key+'_cpu.npz'),allow_pickle=False) as z:
                        ref=dict(zip(z['cell'],z['q']))
                tv=.5*fsum(abs(q.get(i,0)-ref.get(i,0)) for i in set(q)|set(ref))
                row.update(cells=len(q),reference_tv=tv,export_tv_bound=float(bound),
                           reference_reported_bound=reference['tv_bound'])
            out.append(row)
        summary[name]=dict(tasks=len(out),positive=sum(r['positive'] for r in out),
                           passed=sum(r['certified'] for r in out),exact=sum(r['volume_solved'] for r in out),
                           end_to_end_s=time_summary([r['delivered_s'] for r in out]),
                           matched_reference_s=time_summary([r['reference_s'] for r in out]),
                           sequential_fallback_s=sum(r['delivered_s']+(0 if r['certified'] else r['reference_s']) for r in out),
                           results=out)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('folder','comparison','out'): p.add_argument(arg,type=Path)
    a=p.parse_args()
    with a.out.open('x') as f:
        json.dump(evaluate(a.folder,a.comparison),f,indent=1)
        f.write('\n')
