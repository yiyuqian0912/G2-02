"""Independent exact segment checks for both orientations and for ambiguity."""
from fractions import Fraction as Q
from itertools import product
import random

import numpy as np

import inputs
from certificates import integer_row,segment_hits
from interface_chart import interface_chart
from poly import form
from solver import Observation
from cbc_search import solve
from ledger import decode


def rows(lo,hi):
    out=[]
    for i in range(3):
        n=tuple(int(k==i) for k in range(3))
        out.extend([(*(-v for v in n),lo[i]),(*n,-hi[i])])
    return tuple(out)


def test_exact_projective_products():
    rng=random.Random(8291)
    body=rows((4,-1,-10),(6,1,10))
    counters=set()
    cases=[((0,-4,-1),(1,-3,1),(10,5,-1),(11,9,1))]
    for _ in range(320):
        x,y=rng.randrange(-9,10),rng.randrange(-9,10)
        cases.append(((0,x,0),(1,x+rng.randrange(1,7),1),(10,y,0),(11,y+rng.randrange(1,7),1)))
    for a,b,c,d in cases:
        def vertices(l,h):
            return [(*v,1) for v in product(*zip(l,h))]
        s,r=vertices(a,b),vertices(c,d)
        kind,info=interface_chart([body],s,r)
        counters.add(kind)
        if kind=="unknown": continue
        if info:
            _,i,j,rev=info
            f=tuple(-v for v in body[i]) if rev else body[i]
            g=body[j]
            counters.add("reverse" if rev else "direct")
            assert interface_chart([body,body],s,r)[0]=="unknown"
        for source in [*s,(*[Q(h+l,2) for h,l in zip(a,b)],1)]:
            for recv in [*r,(*[Q(h+l,2) for h,l in zip(c,d)],1)]:
                physical=segment_hits(body,source[:3],recv[:3],1)
                if kind=="pencil":
                    assert form(f,source)>0 and form(f,recv)<0
                    expected=form(f,source)*form(g,recv)<=form(g,source)*form(f,recv)
                else:
                    expected=kind=="hit"
                assert physical==expected
    assert {"direct","reverse","miss","hit","unknown"}<=counters


def test_closed_room_cbc():
    import room
    from geometry import rows_of
    boxes=[]
    for i in range(3):
        for low,hi in [(room.OUTER.lo[i],room.CAVITY.lo[i]),(room.CAVITY.hi[i],room.OUTER.hi[i])]:
            a,b=list(room.OUTER.lo),list(room.OUTER.hi)
            a[i],b[i]=low,hi
            boxes.append(room.Box(tuple(a),tuple(b)))
    bodies=tuple(tuple(integer_row(r) for r in rows_of(box)) for box in [*boxes,room.POST])
    cls=np.frombuffer(room.make_observation(),np.uint8).reshape((64,)*3)
    obs=Observation(bodies,cls,room.ORIGIN,1024)
    wire,d=solve(obs,"room-positive",Q(1,20),30,patch_depth=1)
    assert d["status"]=="CERTIFIED", d
    assert decode(wire)=={(64*128+64)*128+64:Q(1)} and Q(d["tv"])==0
