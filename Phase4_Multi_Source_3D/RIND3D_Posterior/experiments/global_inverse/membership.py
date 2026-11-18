"""Independent exact rational point membership, including zero-volume contact strata."""
from fractions import Fraction as Q
from itertools import combinations
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mixed_patch.certificates import segment_hits


def dot(a,b):
    return sum(x*y for x,y in zip(a,b))


def solve_linear(mat,rhs):
    a=[list(map(Q,row))+[Q(y)] for row,y in zip(mat,rhs)]
    for j in range(len(a)):
        piv=next((k for k in range(j,len(a)) if a[k][j]),None)
        if piv is None: return None
        a[j],a[piv]=a[piv],a[j]
        v=a[j][j]
        a[j]=[x/v for x in a[j]]
        for k in range(len(a)):
            if k!=j:
                v=a[k][j]
                a[k]=[x-v*y for x,y in zip(a[k],a[j])]
    return [r[-1] for r in a]


def clear(body,s,radius=Q(1,2**26)):
    """For a rational point compare its squared Euclidean distance to a convex body."""
    vals=[dot(r[:3],s)+r[3] for r in body]
    if max(vals)<=0: return False
    if any(v>0 and v*v>radius*radius*dot(r[:3],r[:3]) for v,r in zip(vals,body)):
        return True
    for n in range(1,4):
        for ids in combinations(range(len(body)),n):
            A=[body[i][:3] for i in ids]
            lm=solve_linear([[dot(a,b) for b in A] for a in A],[vals[i] for i in ids])
            if lm is None: continue
            delta=tuple(sum(l*a[j] for l,a in zip(lm,A)) for j in range(3))
            if dot(delta,delta)>radius**2: continue
            x=tuple(y-d for y,d in zip(s,delta))
            if all(dot(r[:3],x)+r[3]<=0 for r in body): return False
    return True


def contains(obs,s):
    """Slow reference for any rational source; no fitted region or archived label needed."""
    s=tuple(Q(val) for val in s)
    N=obs.denom//2
    if len(s)!=3 or not all(0<v<1 for v in s): return False
    if all(o<=v*N<=o+obs.side for o,v in zip(obs.origin,s)): return False
    if not all(clear(body,s) for body in obs.bodies): return False
    import numpy as np
    for p in np.argwhere(obs.cls!=2):
        r=tuple(Q(n,obs.denom) for n in obs.world(tuple(map(int,p)))[:3])
        hit=any(segment_hits(b,s,r,1) for b in obs.bodies)
        if hit==bool(obs.cls[tuple(p)]): return False
    return True
