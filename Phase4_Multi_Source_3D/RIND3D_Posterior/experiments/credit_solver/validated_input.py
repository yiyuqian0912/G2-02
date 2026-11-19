"""Canonical finite polyhedral problem on the 1024-centre lattice."""
from fractions import Fraction
from itertools import product
from math import gcd, lcm

import numpy as np


def integer(v):
    if type(v) is int or isinstance(v, np.integer) and not isinstance(v, np.bool_):
        return int(v)
    raise ValueError("expected an exact integer")


def canonical_row(row):
    if len(row) != 4:
        raise ValueError("expected a homogeneous halfspace")
    q = [v if type(v) is Fraction else Fraction(integer(v)) for v in row]
    den = lcm(*(v.denominator for v in q))
    values = [v.numerator * (den // v.denominator) for v in q]
    if not any(values[:3]):
        raise ValueError("a facet needs a nonzero normal")
    g = gcd(*values)
    return tuple(v // g for v in values)


def _exact_interval(rows, x0, y, z, L):
    lo, hi = 0, L-1
    for a,b,c,d in rows:
        t = a*x0+b*y+c*z+d*2048
        if a > 0:
            hi = min(hi, -t//(2*a))
        elif a < 0:
            lo = max(lo, -(t//(2*a)))
        elif t > 0:
            return 1, 0
        if lo > hi:
            break
    return lo, hi


def _intervals(rows, x0, y, z, L):
    shape = (len(z), y.size)
    lo, hi = np.zeros(shape, np.int64), np.full(shape, L-1, np.int64)
    for a,b,c,d in rows:
        t = a*x0+b*y+c*z+d*2048
        if a > 0:
            np.minimum(hi, -t//(2*a), out=hi)
        elif a < 0:
            np.maximum(lo, -(t//(2*a)), out=lo)
        else:
            hi[t > 0] = -1
    return lo, hi


def verify_solid(cls, origin, bodies):
    """Intersect each x scanline with each closed body. Equal directed bounds are exact."""
    from raster import fixed
    L = cls.shape[0]
    base = tuple(2*v+1 for v in origin)
    corners = tuple(product(*[(v, v+2*(L-1)) for v in base]))
    nearby = []
    for body in bodies:
        if any(all(a*x+b*y+c*z+d*2048 > 0 for x,y,z in corners) for a,b,c,d in body):
            continue
        nearby.append((body, fixed(body, True), fixed(body, False)))
    y = base[1]+2*np.arange(L, dtype=np.int64)[None,:]
    for k in range(0,L,16):
        nz = min(16,L-k)
        z = base[2]+2*np.arange(k,k+nz,dtype=np.int64)[:,None]
        events = np.zeros((nz,L,L+1), np.int16)
        for body,inner,outer in nearby:
            lo,hi = _intervals(inner,base[0],y,z,L)
            ol,oh = _intervals(outer,base[0],y,z,L)
            for iz,iy in zip(*np.nonzero((ol<=oh)&((lo!=ol)|(hi!=oh)))):
                a,b = _exact_interval(body,base[0],int(y[0,iy]),int(z[iz,0]),L)
                if a<=b:
                    lo[iz,iy],hi[iz,iy] = a,b
                else:
                    lo[iz,iy],hi[iz,iy] = 1,0
            iz,iy = np.nonzero(lo<=hi)
            events[iz,iy,lo[iz,iy]] += 1
            events[iz,iy,hi[iz,iy]+1] -= 1
        occupied = np.cumsum(events[:,:,:L],axis=2,dtype=np.int16)>0
        if not np.array_equal(occupied, cls.T[k:k+nz]==2):
            raise ValueError("SOLID crop disagrees with canonical geometry at receiver centres")


def validate(observation):
    L = integer(observation.side)
    if integer(observation.denom)!=2048 or L not in (64,128,256):
        raise ValueError("expected a v3 window in a 1024 world")
    origin = tuple(integer(v) for v in observation.origin)
    if len(origin)!=3 or any(o<0 or o+L>1024 for o in origin):
        raise ValueError("window outside world")
    bodies = tuple(tuple(canonical_row(r) for r in b) for b in observation.bodies)
    if not 1 <= len(bodies) <= 64 or any(not 1 <= len(b) <= 64 for b in bodies):
        raise ValueError("expected 1..64 convex components and 1..64 defining rows per component")
    cls = np.array(observation.cls, copy=True, subok=False)
    if cls.shape != (L,L,L) or cls.dtype.kind not in "iu" or cls.min()<0 or cls.max()>2:
        raise ValueError("expected integer FREE-inactive, FREE-active or SOLID classes")
    cls = cls.astype(np.uint8,copy=False)
    verify_solid(cls,origin,bodies)
    cls.flags.writeable = False
    return bodies,cls,origin
