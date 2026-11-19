"""Global inverse visibility by certified convex difference and receiver-side compression."""
from functools import lru_cache
from itertools import product
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"mixed_patch"))
from certificates import integer_row
from whole import line_endpoints
from poly import Poly, form, primitive, within, without


class Limit(Exception):
    """Only the last fully covered partition is a valid checkpoint."""


class Budget:
    def __init__(self, seconds, cap):
        self.deadline = perf_counter()+seconds
        self.cap = cap

    def check(self, n=0):
        if n > self.cap: raise Limit("fragment_cap")
        if perf_counter() >= self.deadline: raise Limit("time_cap")


def cone(body, receiver):
    """Exact source halfspaces whose segments hit the closed convex body."""
    h = [form(r,receiver) for r in body]
    front = [i for i,v in enumerate(h) if v>0]
    if not front:
        raise ValueError("FREE receiver inside a canonical obstacle")
    rows = [r for r,v in zip(body,h) if v>=0]
    for i in front:
        for j,v in enumerate(h):
            if v<0:
                r = primitive(tuple(h[i]*b-v*a for a,b in zip(body[i],body[j])))
                if any(r): rows.append(r)
    return tuple(dict.fromkeys(rows))


def subtract_many(parts, cones, budget):
    for c in cones:
        pending = []
        for p in parts:
            pending.extend(without(p,c))
            budget.check(len(pending))
        parts = pending
        if not parts: break
    return parts


def apply_pair(poly, dark, active, budget):
    """First-blocker ownership for dark, every obstacle must miss active."""
    hits = []
    for j,c in enumerate(dark):
        p = within(poly,c)
        if p is not None:
            hits.append((j,p))
    out = []
    previous = []
    for j,p in hits:
        # The same obstacle usually removes most of its own dark cone first.
        pieces = without(p,active[j])
        cuts = [c for k,c in enumerate(active) if k!=j] + previous
        out.extend(subtract_many(pieces,cuts,budget))
        budget.check(len(out))
        previous.append(dark[j])
    return out


def choose_pairs(cls, count, seed, strategy):
    """No source positions or posterior labels. Deterministic receiver inputs only."""
    rng = np.random.default_rng(seed)
    # The [x,y,z] class array is usually a transposed view of a contiguous [z,y,x] array;
    # C-order traversals of it miss cache on every element (~0.8 s at L=256). Detect the
    # interfaces on the contiguous array and rebuild exactly the ascending flat indices that
    # np.flatnonzero would return on the [x,y,z] slices, so all random choices are unchanged.
    zyx = np.ascontiguousarray(cls.T)
    L = cls.shape[0]
    pools = []
    for axis in range(3):
        a,b = [slice(None)]*3,[slice(None)]*3
        a[axis],b[axis] = slice(None,-1),slice(1,None)
        x,y = cls[tuple(a)],cls[tuple(b)]
        za,zb = [slice(None)]*3,[slice(None)]*3
        za[2-axis],zb[2-axis] = slice(None,-1),slice(1,None)
        u,v = zyx[tuple(za)],zyx[tuple(zb)]
        zz,yy,xx = np.nonzero(((u==0)&(v==1))|((u==1)&(v==0)))
        shape = x.shape
        ix = np.sort((xx.astype(np.int64)*shape[1] + yy)*shape[2] + zz)
        if len(ix):
            ix = ix[rng.choice(len(ix),min(256,len(ix)),replace=False)]
            loc = np.column_stack(np.unravel_index(ix,x.shape))
            other = loc.copy()
            other[:,axis]+=1
            for p,q in zip(loc,other):
                p,q = tuple(map(int,p)),tuple(map(int,q))
                pools.append((p,q) if cls[p]==0 else (q,p))
    if not pools: raise ValueError("no direct FREE interface")
    if strategy in ("random_classes", "classes_spread"):
        result=[]
        for k in (0,1):
            inds=np.flatnonzero(cls==k)
            n = count if strategy == "random_classes" else len(pools)
            ids=rng.choice(inds,min(n,len(inds)),replace=False)
            result.append([tuple(map(int,p)) for p in zip(*np.unravel_index(ids,cls.shape))])
        pools = list(zip(*result))
        if strategy == "random_classes":
            return pools
    elif strategy == "interface_random":
        return [pools[i] for i in rng.choice(len(pools),min(count,len(pools)),replace=False)]
    elif strategy != "interface":
        raise ValueError(strategy)
    # Maximin order on doubled midpoints (integer), with a deterministic first edge.
    m = np.array([np.add(a,b) for a,b in pools])
    picked=[]
    d=np.full(len(m),2**60,dtype=np.int64)
    i=int(rng.integers(len(m)))
    for _ in range(min(count,len(m))):
        picked.append(pools[i])
        d=np.minimum(d,((m-m[i])**2).sum(1))
        d[i]=-1
        i=int(d.argmax())
    return picked


def bounding_mask(bodies, p, receiver_vertices):
    """Resolve a whole product of a possibly oblique source polytope and receiver box.

    Values at differently scaled homogeneous vertices cannot be compared as
    magnitudes. Only their signs and signs of like-scaled determinants are used.
    """
    possible=[]
    for c,body in enumerate(bodies):
        ss = [[form(row,v) for v in p.vertices] for row in body]
        rr = [[form(row,v) for v in receiver_vertices] for row in body]
        pos = [i for i,row in enumerate(ss) if min(row)>0]
        neg = [j for j,row in enumerate(ss) if max(row)<0]
        if any(min(rr[i])>0 for i in pos): continue
        unsure = []
        missed = False
        for i in pos:
            for j in neg:
                # Fast extrema filter, then exact same-vertex determinant.
                values = [a*v-b*u for a,b in zip(ss[i],ss[j]) for u,v in zip(rr[i],rr[j])]
                if min(values)>0:
                    missed=True
                    break
                if max(values)>0: unsure.append((i,j))
            if missed: break
        if missed: continue
        fixed = len(pos)+len(neg)==len(body) and pos and all(max(rr[i])<=0 for i in pos)
        if fixed and not unsure: return "hit",None
        if fixed and len(unsure)==1 and max(rr[unsure[0][0]])<0:
            i,j=unsure[0]
            possible.append((c,i,j))
        else:
            possible.append(None)
    if not possible: return "miss",None
    if len(possible)==1 and possible[0] is not None: return "pencil",possible[0]
    return "unknown",None


class Observation:
    def __init__(self,bodies,cls,origin,n):
        self.bodies=bodies
        self.cls=cls
        self.origin=origin
        self.denom=2*n
        self.side=len(cls)
        # Per-observation caches; never share receiver proofs across identities.
        self.band=lru_cache(maxsize=2048)(self._band)
        self.block=lru_cache(maxsize=8192)(self._block)

    def world(self,p):
        return (*(2*(a+b)+1 for a,b in zip(self.origin,p)), self.denom)

    def _block(self,loc,side):
        data=self.cls[tuple(slice(i,i+side) for i in loc)]
        kinds=int((data==0).any())+2*int((data==1).any())
        rv=tuple(self.world(tuple(l+i for l,i in zip(loc,k))) for k in product((0,side-1),repeat=3))
        return kinds,rv

    def _band(self,loc,side,hinge):
        c,i,j=hinge[:3]
        front,back=self.bodies[c][i],self.bodies[c][j]
        if len(hinge)==4 and hinge[3]:
            front=tuple(-x for x in front)
        data=self.cls[tuple(slice(i,i+side) for i in loc)]
        cuts=[]
        for active in (True,False):
            best=None
            for local in line_endpoints(data==int(active)):
                r=self.world(tuple(l+int(o) for l,o in zip(loc,local)))
                n,d=-form(back,r),-form(front,r)
                if d<=0: raise ArithmeticError("unproved pencil denominator")
                if best is None or (n*best[1]>d*best[0] if active else n*best[1]<d*best[0]):
                    best=n,d
            if best is not None:
                n,d=best
                sign=1 if active else -1
                row=primitive(tuple(sign*(n*a-d*b) for a,b in zip(front,back)))
                if not any(row):
                    if active: return None
                else: cuts.append(row)
        return cuts

    def cleared(self,p):
        for body in self.bodies:
            for r in body:
                norm2=sum(a*a for a in r[:3])
                if all((v:=form(r,x))>0 and (v*2**26)**2>norm2*x[3]**2 for x in p.vertices):
                    break
            else: return False
        return True

    def reduce_patches(self,p,depth,budget,stats, reciprocal=True):
        """Apply certified receiver-product cuts. An unknown block retains all obligations."""
        from interface_chart import interface_chart, extend_rank
        basis = []
        hinges = set()
        blockers = set()
        stack=[((0,0,0),self.side,0)]
        outstanding=[]
        while stack:
            budget.check()
            loc,side,d=stack.pop()
            kinds,rv=self.block(loc,side)
            if kinds==0: continue
            if reciprocal:
                state,hinge=interface_chart(self.bodies,p.vertices,rv)
            else:
                state,hinge=bounding_mask(self.bodies,p,rv)
            stats[state]+=1
            if state=="pencil":
                distinct = hinge not in hinges
                hinges.add(hinge)
                if reciprocal and hinge[3]:
                    stats["receiver_oriented_patches"] += 1
                blockers.add(hinge[0])
                stats["hinges_in_one_piece_max"] = max(stats["hinges_in_one_piece_max"],len(hinges))
                stats["blockers_in_one_piece_max"] = max(stats["blockers_in_one_piece_max"],len(blockers))
                cuts=self.band(loc,side,hinge)
                stats["extremal_rows"] += len(cuts) if cuts is not None else 0
                if distinct and cuts and kinds == 3:
                    extend_rank(basis,cuts[0][:3])
                    stats["direction_rank_max"] = max(stats["direction_rank_max"],len(basis))
                if kinds == 3:
                    stats["mixed_patches"] += 1
                    data = self.cls[tuple(slice(v,v+side) for v in loc)]
                    stats["mixed_free_obligations"] += int(np.count_nonzero(data!=2))
                p=within(p,cuts) if cuts is not None else None
                if p is None: return None,True
            elif state=="unknown":
                if d>=depth or side==1:
                    outstanding.append((loc,side))
                else:
                    half=side//2
                    stack.extend((tuple(a+b for a,b in zip(loc,k)),half,d+1)
                                 for k in product((0,half),repeat=3))
            elif (state=="miss" and kinds&1) or (state=="hit" and kinds&2):
                return None,True
        return p,not outstanding

    def certify_piece(self,p,depth,budget,stats):
        q, covered = self.reduce_patches(p,depth,budget,stats,reciprocal=False)
        return q, covered and (q is None or self.cleared(q))

    def exclude_window(self,parts,budget):
        rows=[]
        for i,o in enumerate(self.origin):
            u=tuple(int(k==i)*(self.denom//2) for k in range(3))
            rows.extend([(*(-v for v in u),o),(*u,-o-self.side)])
        return subtract_many(parts,[rows],budget)
