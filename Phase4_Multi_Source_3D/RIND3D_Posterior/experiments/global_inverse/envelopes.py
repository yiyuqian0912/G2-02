"""Receiver convexity and exact inverse-cone sandwiches for a whole labelled block."""
from collections import Counter
from functools import lru_cache
from itertools import product

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from poly import form, primitive, within, without
from solver import cone, subtract_many
from whole import line_endpoints


def convex_generators(mask):
    """Every deletion is a convex-combination proof over the integer receiver lattice."""
    pts=line_endpoints(mask)
    if len(pts)<12:
        return pts
    if any(s>256 for s in mask.shape):
        return pts  # bounded int64 determinant proof below is for the v3 sizes
    try:
        hull=ConvexHull(pts)
    except QhullError:
        return pts
    retained=np.zeros(len(pts),bool)
    retained[hull.vertices]=True
    unknown=pts[~retained]
    anchor=pts[hull.vertices[0]]
    for indices in hull.simplices:
        if not len(unknown): break
        a,b,c=pts[indices]-anchor
        normals=np.array([np.cross(b,c),np.cross(c,a),np.cross(a,b)])
        det=int(a@normals[0])
        if det==0: continue
        # Cramer's rule: four nonnegative barycentric coordinates. All tetra
        # vertices belong to this actual class. An incomplete hull only costs work.
        weights=(unknown-anchor)@normals.T
        if det<0: weights=-weights
        inside=(weights>=0).all(axis=1)&(weights.sum(axis=1)<=abs(det))
        unknown=unknown[~inside]
    return np.concatenate((pts[retained],unknown))


def extreme_pair(h,i,j,ids,maximize):
    """Extremize h_j / h_i where h_i>0, retaining a real witness."""
    best=ids[0]
    for k in ids[1:]:
        v=h[j][k]*h[i][best]-h[j][best]*h[i][k]
        if (maximize and v>0) or (not maximize and v<0):
            best=k
    return best


def projected(body, r, i, j):
    return primitive(tuple(r[i]*b-r[j]*a for a,b in zip(body[i],body[j])))


class Envelope:
    """inner = intersection of all inverse shadows; outer contains their union."""
    def __init__(self,body,points):
        h=[[form(row,p) for p in points] for row in body]
        rows=list(zip(*h))
        self.inner=[]
        self.outer=[]
        witnesses=[]
        for i,values in enumerate(h):
            if max(values)>=0: self.inner.append(body[i])
            if min(values)>=0: self.outer.append(body[i])
            positive=[k for k,v in enumerate(values) if v>0]
            if not positive: continue
            for j,back in enumerate(h):
                ids=[k for k in positive if back[k]<0]
                if not ids: continue
                tight=extreme_pair(h,i,j,ids,True)
                self.inner.append(projected(body,rows[tight],i,j))
                if len(ids)==len(points):
                    k=extreme_pair(h,i,j,ids,False)
                    self.outer.append(projected(body,rows[k],i,j))
                    witnesses.append(k)
        self.inner=tuple(dict.fromkeys(r for r in self.inner if any(r)))
        self.outer=tuple(dict.fromkeys(r for r in self.outer if any(r)))
        # Witness choice only affects effort. Failing to cover outer requires
        # refinement, irrespective of how many witnesses have been tried.
        self.witnesses=tuple(points[k] for k,_ in Counter(witnesses).most_common(4))
        if not self.witnesses: self.witnesses=(points[0],)


class Compiler:
    def __init__(self,obs,depth):
        self.obs=obs
        self.depth=depth
        self.points=lru_cache(maxsize=1024)(self._points)
        self.envelope=lru_cache(maxsize=4096)(self._env)
        self.stats=Counter()

    def _points(self,loc,size,active):
        data=self.obs.cls[tuple(slice(x,x+size) for x in loc)]
        pts=convex_generators(data==int(active))
        return tuple(self.obs.world(tuple(a+int(b) for a,b in zip(loc,p))) for p in pts)

    def _env(self,loc,size,active,c):
        pts=self.points(loc,size,active)
        return Envelope(self.obs.bodies[c],pts)

    def dark(self,parts,loc,size,budget):
        if not self.points(loc,size,False): return parts,[]
        good,unknown=[],[]
        for p in parts:
            budget.check(len(good)+len(unknown))
            possible=[]
            for c in range(len(self.obs.bodies)):
                env=self.envelope(loc,size,False,c)
                if within(p,env.outer) is not None:
                    possible.append(env)
            if not possible:
                self.stats['dark_empty']+=1
            elif len(possible)==1:
                q=within(p,possible[0].inner)
                if q is not None: good.append(q)
                self.stats['dark_unique']+=1
            elif any(within(p,e.inner) is p for e in possible):
                good.append(p)
            elif size==1:
                # First-cover partition of a union, disjoint in volume.
                previous=[]
                for e in possible:
                    q=within(p,e.inner)
                    if q is not None: good.extend(subtract_many([q],previous,budget))
                    previous.append(e.inner)
            else: unknown.append(p)
        return good,unknown

    def light(self,parts,loc,size,budget):
        if not self.points(loc,size,True): return parts,[]
        # For each body, flag only those source fragments that still require
        # more receiver checks. An already-uncertain fragment may still shrink.
        work=[(p,False) for p in parts]
        for c,body in enumerate(self.obs.bodies):
            e=self.envelope(loc,size,True,c)
            new=[]
            for p,unresolved in work:
                budget.check(len(new))
                outer=within(p,e.outer)
                if outer is None:
                    new.append((p,unresolved))
                    continue
                # Convex coverage can be checked on vertices. This proves the
                # whole possible obstruction equals a single real obstruction on P.
                rows=[cone(body,w) for w in e.witnesses]
                if any(within(outer,shadow) is outer for shadow in rows):
                    new.extend((q,unresolved) for q in without(p,e.outer))
                    self.stats['active_exact_union']+=1
                else:
                    residual=subtract_many([p],rows,budget)
                    for q in residual:
                        new.append((q,unresolved or within(q,e.outer) is not None))
            work=new
            if not work: break
        return [p for p,u in work if not u],[p for p,u in work if u]

    def block(self,parts,loc,size,d,budget):
        budget.check(len(parts))
        if not parts: return [],[]
        self.stats['block_visits']+=1
        good,bad=self.dark(parts,loc,size,budget)
        good,bad_lit=self.light(good,loc,size,budget)
        bad=bad+bad_lit
        if not bad or d>=self.depth: return good,bad
        half=size//2
        if not half: raise ArithmeticError('singleton unresolved')
        # A piece still unresolved in one child must remain unresolved even
        # after passing all the other children.
        done=bad
        partial=[]
        for offsets in product((0,half),repeat=3):
            child=tuple(a+b for a,b in zip(loc,offsets))
            x,y=self.block(done,child,half,d+1,budget)
            u,v=self.block(partial,child,half,d+1,budget)
            done=x
            partial=y+u+v
            budget.check(len(done)+len(partial)+len(good))
        return good+done,partial

    def finish(self,parts,budget):
        yes,maybe=[],[]
        visibility_only=[]
        reason='complete_traversal'
        from solver import Limit
        for i,p in enumerate(parts):
            try:
                done,partial=self.block([p],(0,0,0),self.obs.side,0,budget)
                budget.check(len(done)+len(partial))
            except Limit as exc:
                maybe.extend(parts[i:])
                reason=str(exc)
                break
            maybe.extend(partial)
            for q in done:
                if self.obs.cleared(q): yes.append(q)
                else:
                    # Try to classify even the rounded clearance shell before
                    # treating a physical solution as unresolved.
                    try:
                        inside,shell=clearance_partition(self.obs,q,budget)
                    except Limit:
                        maybe.append(q)
                        visibility_only.append(q)
                    else:
                        yes.extend(inside)
                        maybe.extend(shell)
                        visibility_only.extend(shell)
        return yes,maybe,reason,visibility_only

    def lower_block(self,parts,loc,size,d,budget):
        """One-sided proof. A depth-budget failure loses only inner mass."""
        if not parts: return []
        budget.check(len(parts))
        # Reuse geometry-aware coarse decisions if possible. Only expend the
        # receiver subdivision budget where there are unresolved fragments.
        good,bad=self.dark(parts,loc,size,budget)
        good,litbad=self.light(good,loc,size,budget)
        uncertain=bad+litbad
        if not uncertain: return good
        if d==0 or size==1:
            if self.points(loc,size,False):
                cones=[self.envelope(loc,size,False,c).inner for c in range(len(self.obs.bodies))]
                shadow=[]
                for p in uncertain:
                    prior=[]
                    for h in cones:
                        q=within(p,h)
                        if q is not None:
                            shadow.extend(subtract_many([q],prior,budget))
                            prior.append(h)
                    budget.check(len(shadow))
                uncertain=shadow
            if self.points(loc,size,True):
                uncertain=subtract_many(uncertain,[self.envelope(loc,size,True,c).outer
                                         for c in range(len(self.obs.bodies))],budget)
            return good+uncertain
        half=size//2
        for offset in product((0,half),repeat=3):
            child=tuple(a+b for a,b in zip(loc,offset))
            uncertain=self.lower_block(uncertain,child,half,d-1,budget)
            if not uncertain: break
        return good+uncertain


def clearance_covers(obs):
    """Outer facet offsets and inner rational translates of the clearance body."""
    from math import isqrt
    expanded=[]
    for body in obs.bodies:
        h=[]
        for r in body:
            norm2=sum(a*a for a in r[:3])
            # Directed bound even for small primitive coefficients.
            norm2 <<= 80
            up=isqrt(norm2)
            up+=int(up*up<norm2)
            scale=1<<66
            h.append((*[a*scale for a in r[:3]],r[3]*scale-up))
        expanded.append(h)
    inside=[]
    for body in obs.bodies:
        inside.append(body)
        for axis,sgn in product(range(3),(-1,1)):
            inside.append(tuple((*(a*(1<<26) for a in r[:3]),(r[3]<<26)-sgn*r[axis]) for r in body))
    return expanded,inside


def clearance_partition(obs,p,budget):
    if not hasattr(obs,'clearance_covers'):
        obs.clearance_covers=clearance_covers(obs)
    enlarged,forbidden=obs.clearance_covers
    outside=[p]
    possible=[]
    for h in enlarged:
        rest=[]
        for q in outside:
            inter=within(q,h)
            if inter is None: rest.append(q)
            else:
                possible.append(inter)
                rest.extend(without(q,h))
            budget.check(len(possible)+len(rest))
        outside=rest
    return outside,subtract_many(possible,forbidden,budget)


def legal_inner(obs,p,budget):
    if obs.cleared(p): return [p]
    if not hasattr(obs,'clearance_covers'):
        obs.clearance_covers=clearance_covers(obs)
    return subtract_many([p],obs.clearance_covers[0],budget)
