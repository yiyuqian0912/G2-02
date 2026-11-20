"""Exact half-space volumes in axis-aligned output cells and decoded dyadic wire accounting."""
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from math import factorial, prod

from kernel import COMMON, SCALE
from ledger import M, RECORD, TICKS

CELL = SCALE // 128**3


@lru_cache(maxsize=256)
def coefficients(normal):
    """Terms of inclusion-exclusion on the positive-width coordinates of a unit cell."""
    widths = tuple(abs(a) for a in normal if a)
    n = len(widths)
    if not n:
        raise ArithmeticError("zero facet normal")
    terms = [(sum(subset), (-1)**k) for k in range(n+1) for subset in combinations(widths,k)]
    return sum(widths), n, factorial(n)*prod(widths), terms


def half_cell(row, i, j, k):
    a,b,c,d = row
    extent,n,den,terms = coefficients((a,b,c))
    # Shift to the minimizing corner; reflect each negative coordinate.
    h = -(128*d+a*i+b*j+c*k+min(a,0)+min(b,0)+min(c,0))
    if h <= 0:
        return 0,0
    if h >= extent:
        return CELL,CELL
    num = sum(sign*(h-offset)**n for offset,sign in terms if h>offset)
    if not 0 < num < den:
        raise ArithmeticError("invalid cut-cell volume")
    q,r = divmod(CELL*num,den)
    return q,q+bool(r)


def intersecting(rows,box):
    """Return active facets, or None when a separating facet proves zero volume."""
    x,X,y,Y,z,Z=box
    active=[]
    for row in rows:
        a,b,c,d=row
        base=a*x+b*y+c*z+128*d
        dx,dy,dz=a*(X+1-x),b*(Y+1-y),c*(Z+1-z)
        if base+min(0,dx)+min(0,dy)+min(0,dz)>=0:
            return None
        if base+max(0,dx)+max(0,dy)+max(0,dz)>0:
            active.append(row)
    return active


def dyadic_bound(payload,lower,upper,total_live,cell_live):
    """Same lower overlap as wire_bound, specialized to the documented dyadic52 wire."""
    if not lower or not payload or len(payload)%RECORD.size or len(payload)//RECORD.size>M:
        raise ValueError("invalid payload or credit")
    total=sum(upper.values())+total_live
    previous=-1
    common=ticks=0
    for cid,value in RECORD.iter_unpack(payload):
        if not previous<cid<M or not 0<value<=1:
            raise ValueError("invalid record")
        # Exact: binary64 multiplication by 2^52 has no rounding for 0 < value <= 1.
        scaled=value*TICKS
        k=int(scaled)
        if scaled!=k:
            raise ValueError("noncanonical tick")
        previous=cid
        ticks+=k
        l=lower.get(cid,0)
        outside=total-upper.get(cid,0)-cell_live.get(cid,0)
        if outside<0:
            raise ArithmeticError("negative outer mass")
        if l:
            common+=min(k*(COMMON//TICKS),l*COMMON//(l+outside))
    if ticks!=TICKS:
        raise ValueError("invalid normalization")
    return Fraction(COMMON-common,COMMON)

