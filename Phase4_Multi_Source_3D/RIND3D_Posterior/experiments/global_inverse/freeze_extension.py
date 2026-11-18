"""Select a second disjoint geometry per family, before running the reciprocal solver."""
from pathlib import Path
import argparse,json
from research import task_manifest

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('comparison',type=Path)
    p.add_argument('root',type=Path)
    p.add_argument('first',type=Path)
    p.add_argument('out',type=Path)
    a=p.parse_args()
    old=json.loads(a.first.read_bytes())['holdout']
    new=task_manifest(a.comparison,a.root,{t['slot'] for t in old})['holdout']
    groups=sorted({r['family'] for r in new})
    new=[t for t in new if t['L']==(64,128,256)[groups.index(t['family'])%3]]
    assert len(new)==9 and len({t['slot'] for t in new})==9
    with a.out.open('x') as f: json.dump({'holdout':new},f,indent=1)
