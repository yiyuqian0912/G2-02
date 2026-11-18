"""Exact per-cell mass of the inner approximation in a frozen research record."""
import argparse
from fractions import Fraction
import gzip
import json
from pathlib import Path
from poly import Poly,within


def load_records(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt') as f:
        for line in f:
            yield json.loads(line)


def cell_mass(record, index):
    if not 0<=index<128**3: raise ValueError("outside the output grid")
    if 'inner_pieces' not in record: raise ValueError("no inner distribution")
    if 'probabilities' in record: return Fraction(record['probabilities'].get(str(index),0))
    ijk=index//16384,index//128%128,index%128
    rows=[]
    for k,i in enumerate(ijk):
        n=tuple(128*int(j==k) for j in range(3))
        rows.extend([(*n,-i-1),(*(-x for x in n),i)])
    total=mass=Fraction(0)
    for faces in record['inner_pieces']:
        p=Poly([tuple(tuple(x) for x in face) for face in faces])
        total+=p.volume
        q=within(p,rows)
        if q is not None: mass+=q.volume
    return mass/total


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('file',type=Path)
    ap.add_argument('task')
    ap.add_argument('ix',type=int)
    a=ap.parse_args()
    r=next(r for r in load_records(a.file) if r['task']['key']==a.task)
    print(cell_mass(r,a.ix))
    print('global TV upper bound (rational):',
          Fraction(r['unknown_upper'],r['outer_upper']))
