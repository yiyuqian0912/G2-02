"""Publish all 128³ probabilities of a certified case as exact integer weights.

Clip every occupied source polytope to the grid. Round each intersection volume
down by less than 2^-160 and charge the entire deficit to the global TV bound.
"""
import argparse
from collections import defaultdict
from fractions import Fraction
import gzip
import json
from pathlib import Path
from time import perf_counter
from hashlib import sha256
from poly import Poly,grid_pieces
from query import load_records


def materialize(r):
    masses=defaultdict(int)
    cuts=0
    scale=1<<r['scale_bits']
    for faces in r['inner_pieces']:
        p=Poly([tuple(tuple(v) for v in f) for f in faces])
        subtotal=0
        n=0
        for (x,y,z),piece in grid_pieces(p,128):
            v=piece.volume*scale
            m=v.numerator//v.denominator
            if m: masses[(x*128+y)*128+z]+=m
            subtotal+=m
            n+=1
        if not subtotal<=p.volume*scale<subtotal+n:
            raise ArithmeticError('grid clipping failed volume enclosure')
        cuts+=n
    total=sum(masses.values())
    if not total or total>r['outer_upper']:
        raise ArithmeticError('invalid mass interval')
    return dict(task=r['task'],geometry_sha256=r['geometry_sha256'],grid=128,scale_bits=r['scale_bits'],
                cells=sorted(masses),weights=[masses[k] for k in sorted(masses)],denominator=total,
                tv_numerator=r['outer_upper']-total,tv_denominator=r['outer_upper'],intersections=cuts)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('records',type=Path)
    p.add_argument('out',type=Path)
    a=p.parse_args()
    a.out.mkdir(exist_ok=True)
    for r in load_records(a.records):
        if not r['certified_tv_005']: continue
        start=perf_counter()
        m=materialize(r)
        payload=(json.dumps(m,separators=(',',':'))+'\n').encode()
        path=a.out/(r['task']['key']+'.json.gz')
        with path.open('xb') as f: f.write(gzip.compress(payload,mtime=0))
        print(r['task']['key'],len(m['cells']),m['intersections'],round(perf_counter()-start,3),
              sha256(payload).hexdigest(),flush=True)


if __name__=='__main__': main()
