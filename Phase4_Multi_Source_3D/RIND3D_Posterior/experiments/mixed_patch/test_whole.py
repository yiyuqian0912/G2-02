from fractions import Fraction
from itertools import product

import numpy as np

from certificates import segment_hits
from whole import legal_volume, line_endpoints, solve


def box_body(lo,hi):
    return tuple(tuple(int(i==k)*sgn for i in range(3)) +
                 ((-hi[k] if sgn==1 else lo[k]),)
                 for k in range(3) for sgn in (-1,1))


def test_class_endpoints_preserve_fractional_extrema_on_nonconvex_masks():
    rng=np.random.default_rng(19)
    for _ in range(40):
        mask=rng.random((8,8,8))<0.6
        def values(pts):
            return [Fraction(int(3*x-2*y+z-17),int(x+2*y+3*z+1)) for x,y,z in pts]
        all_values=values(np.argwhere(mask))
        keep=values(line_endpoints(mask))
        assert (min(keep),max(keep))==(min(all_values),max(all_values))
    assert len(line_endpoints(np.zeros((8,8,8),bool)))==0
    assert len(line_endpoints(np.ones((8,8,8),bool)))==8


def test_entire_mixed_window_on_legal_source_cell():
    # Unit-world denominator 64. Numerator geometry is an exact translated,
    # scaled box; no hidden coordinate is consumed by solve.
    denom=64
    body=box_body((18,16,1),(22,20,60))
    box=((10,10,24),(12,12,26))
    body=tuple((*row[:3],Fraction(row[3],denom)) for row in body)
    o=(15,17,12)
    cls=np.empty((2,2,2),np.uint8)
    for p in product(range(2),repeat=3):
        r=tuple(2*(v+i)+1 for v,i in zip(o,p))
        cls[p]=not segment_hits(body,(11,11,25),r,denom)
    assert set(cls.flat)=={0,1}
    state,frac,stats,n=solve([body],cls,o,box,denom)
    assert state=='exact' and 0<frac<1 and n==1
    assert legal_volume([body],box,denom,o,2)
    assert not legal_volume([body],((30,34,24),(32,36,26)),denom,o,2)
    assert not legal_volume([body],((18,16,24),(20,18,26)),denom,o,2)


def test_unresolved_union_never_becomes_an_exact_label():
    denom=64
    body=tuple((*r[:3],Fraction(r[3],denom)) for r in box_body((18,16,1),(22,20,60)))
    cls=np.array([0,1]*4,np.uint8).reshape(2,2,2)
    state,v,stats,_=solve([body,body],cls,(15,17,12),((10,10,24),(12,12,26)),denom,depth=0)
    assert state=='unresolved' and v is None and stats['unresolved']==1
