"""Dual separation oracle: certify an entire observation over one source polytope."""
from collections import Counter
from itertools import chain
import numpy as np
from numba import njit

from envelopes import Envelope, clearance_partition, legal_inner
from poly import Poly, form, primitive, within
from solver import Limit, cone, subtract_many


def fixed(rows, up):
    out=[]
    for r in rows:
        m=max(map(abs,r))
        if m:
            out.append([(-((-a*(1<<45))//m) if up else (a*(1<<45))//m) for a in r])
    return out


def encoded(obs,p):
    if obs.denom>2048: raise ValueError('fixed-point kernel verified only for N<=1024')
    anchor=primitive(tuple(map(sum,zip(*p.vertices))))
    rows=[]
    offsets=[0]
    if getattr(obs,'use_support',False) and not hasattr(obs,'obstacles'):
        obs.obstacles=[within(Poly.cube(),b) for b in obs.bodies]
        if any(k is None for k in obs.obstacles): raise ValueError('expected full-dimensional body in world')
    for j,b in enumerate(obs.bodies):
        e=Envelope(b,p.vertices)
        if getattr(obs,'use_support',False):
            from penumbra import directions
            e.outer=e.outer+directions(obs.obstacles[j],p)
        # An interior candidate might be in this solid even if some P sources
        # are legal. An everywhere-shadowed anchor is a proposal hint only.
        h=cone(b,anchor) if any(form(r,anchor)>0 for r in b) else []
        for planes,up in ((e.inner,True),(e.outer,False),(h,False)):
            rows.extend(fixed(planes,up))
            offsets.append(len(rows))
    return np.array(rows,np.int64).reshape(-1,4),np.array(offsets,np.int32)


@njit(cache=True)
def scan(cls,origin,denom,rows,off,skipped):
    L=cls.shape[0]
    picks=np.full(16,-1,np.int64)
    best=np.full(16,-1,np.int64)
    diff=np.zeros((3,L+1),np.int32)
    counts=np.empty(3,np.int32)
    ambiguous=0
    anchor_wrong=0
    for z in range(L):
        wz=2*(origin[2]+z)+1
        for y in range(L):
            wy=2*(origin[1]+y)+1
            diff[:]=0
            for group in range(len(off)-1):
                lo,hi=0,L-1
                for i in range(off[group],off[group+1]):
                    a,b,c,d=rows[i]
                    v=a*(2*origin[0]+1)+b*wy+c*wz+d*denom
                    if a>0: hi=min(hi,(-v)//(2*a))
                    elif a<0: lo=max(lo,-((-v)//(-2*a)))
                    elif v>0: hi=-1
                    if lo>hi: break
                if lo<=hi:
                    diff[group%3,lo]+=1
                    diff[group%3,hi+1]-=1
            counts[:]=0
            for x in range(L):
                for j in range(3): counts[j]+=diff[j,x]
                act=cls[z,y,x]
                if act==2: continue
                if (act==1 and counts[0]>0) or (act==0 and counts[1]==0):
                    return -1,picks,anchor_wrong
                if (act==1 and counts[1]==0) or (act==0 and counts[0]>0): continue
                idx=(z*L+y)*L+x
                if len(skipped):
                    pos=np.searchsorted(skipped,idx)
                    if pos<len(skipped) and skipped[pos]==idx: continue
                ambiguous+=1
                wrong=(act==1)==(counts[2]>0)
                anchor_wrong+=int(wrong)
                boundary=((x>0 and cls[z,y,x-1]==1-act) or (x+1<L and cls[z,y,x+1]==1-act)
                          or (y>0 and cls[z,y-1,x]==1-act) or (y+1<L and cls[z,y+1,x]==1-act)
                          or (z>0 and cls[z-1,y,x]==1-act) or (z+1<L and cls[z+1,y,x]==1-act))
                k=((int(z>=L//2)*2+int(y>=L//2))*2+int(x>=L//2))*2+act
                tie=((idx*2654435761)%2147483647)
                score=(int(wrong)*2+int(boundary))*2147483648+tie
                if score>best[k]:
                    picks[k]=idx
                    best[k]=score
    return ambiguous,picks,anchor_wrong


def screen(obs,p,done):
    rows,offs=encoded(obs,p)
    return scan(obs.scancls,np.asarray(obs.origin,np.int64),obs.denom,rows,offs,
                np.array(sorted(done),np.int64))


def enforce(obs,parts,idx,budget):
    L=obs.side
    x,y,z=idx%L,idx//L%L,idx//(L*L)
    r=obs.world((x,y,z))
    h=[cone(b,r) for b in obs.bodies]
    if obs.cls[x,y,z]==1: return subtract_many(parts,h,budget)
    out=[]
    for p in parts:
        prior=[]
        for planes in h:
            q=within(p,planes)
            if q is not None:
                out.extend(subtract_many([q],prior,budget))
                prior.append(planes)
            budget.check(len(out))
    return out


def contraction(p,n,d):
    """Source-independent of any private reference; positive homothety of P."""
    a=tuple(map(sum,zip(*p.vertices)))
    def move(v):
        return primitive((*(n*v[i]*a[3]+(d-n)*a[i]*v[3] for i in range(3)),d*v[3]*a[3]))
    lookup={v:move(v) for v in p.vertices}
    return Poly([tuple(lookup[v] for v in face) for face in p.faces])


def finish_raster(obs,parts,budget):
    from collections import deque
    obs.scancls=np.ascontiguousarray(obs.cls.T)
    queue=deque((p,frozenset()) for p in parts)
    yes,maybe=[],[]
    stats=Counter()
    reason='complete_traversal'
    while queue:
        p,done=queue.popleft()
        try:
            budget.check(len(queue))
            amb,indices,wrong=screen(obs,p,done)
            stats['scans']+=1
            if amb<0:
                stats['refuted']+=1
                continue
            if amb==0:
                legal,shell=clearance_partition(obs,p,budget)
                yes.extend(legal)
                maybe.extend(shell)
                stats['vis_complete']+=1
                continue
            if len(done)>=192:
                maybe.append(p)
                stats['point_cap']+=1
                continue
            new=[p]
            used=[]
            for idx in indices:
                if idx<0: continue
                new=enforce(obs,new,int(idx),budget)
                used.append(int(idx))
                if not new: break
            budget.check(len(queue)+len(new))
            stats['analytic_cuts']+=len(used)
            dist=done.union(used)
            queue.extend((q,dist) for q in new)
        except Limit as exc:
            maybe.extend(chain([p],(q for q,_ in queue)))
            reason=str(exc)
            break
    return yes,maybe,stats,reason


def inner_raster(obs,maybe,budget):
    out=[]
    reason='complete_traversal'
    stats=Counter()
    # Recover at most one disjoint inner core for each unsolved outer piece.
    try:
        for p in maybe:
            for n,d in ((255,256),(3,4),(1,4),(1,32)):
                budget.check()
                q=contraction(p,n,d)
                amb,_,wrong=screen(obs,q,())
                stats['contractions']+=1
                if not amb:
                    out.extend(legal_inner(obs,q,budget))
                    break
                if amb<0 or wrong: break
    except Limit as exc:
        reason=str(exc)
    return out,stats,reason
