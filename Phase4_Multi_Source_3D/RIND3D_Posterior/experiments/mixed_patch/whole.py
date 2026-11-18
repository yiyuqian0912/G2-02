"""Bounded whole-observation proof for one prescribed source cell."""
import argparse
from collections import Counter
from fractions import Fraction
from itertools import product
from pathlib import Path
from time import perf_counter
import io
import json
import numpy as np
from rind3d.geometry import load_geometry
from rind3d.rendering.polyhedra import decode_convex

from certificates import corners, integer_row, mixed_chart, value
from intersection import joint_fraction
from pilot import cropz, source_box


def line_endpoints(mask):
    """Eliminate exact convex combinations of same-class points, axis by axis."""
    for axis in range(3):
        a = np.moveaxis(mask,axis,-1)
        any_ = a.any(axis=-1)
        ij = np.nonzero(any_)
        low = a.argmax(axis=-1)[ij]
        high = a.shape[-1]-1-a[...,::-1].argmax(axis=-1)[ij]
        out = np.zeros_like(a)
        out[(*ij,low)] = True
        out[(*ij,high)] = True
        mask = np.moveaxis(out,-1,axis)
    return np.argwhere(mask)


def legal_volume(bodies, box, denom, origin, L):
    lo,hi=box
    # Source box is inside the world up to a null boundary, and its interior
    # misses the entire closed observation window.
    if any(a < 0 or b > denom for a,b in zip(lo,hi)):
        return False
    if not any(hi[k] <= 2*origin[k] or lo[k] >= 2*(origin[k]+L) for k in range(3)):
        return False
    for body in bodies:
        for r in body:
            gap = min(value(r,p,denom) for p in corners(box))
            if gap>0 and (gap*2**26)**2 > denom**2*sum(a*a for a in r[:3]):
                break
        else:
            return False
    return True


def solve(bodies, cls, o, box, denom, depth=2):
    hinges = {}
    scans=Counter()
    failed=False
    stack=[((0,0,0),len(cls),0)]
    while stack and not failed:
        loc,side,d=stack.pop()
        p=cls[tuple(slice(i,i+side) for i in loc)]
        if (p==2).all():
            scans['solid']+=1
            continue
        low=tuple(2*(v+i)+1 for v,i in zip(o,loc))
        rbox=low,tuple(v+2*(side-1) for v in low)
        state,identity=mixed_chart(bodies,box,rbox,denom)
        scans[state]+=1
        if state=='unknown':
            if d==depth or side==1:
                scans['unresolved']+=1
            else:
                half=side//2
                for offset in product((0,half),repeat=3):
                    stack.append((tuple(i+j for i,j in zip(loc,offset)),half,d+1))
        elif state=='pencil':
            b,i,j=identity
            front,back=bodies[b][i],bodies[b][j]
            bounds=hinges.setdefault(identity,[None,None])
            for k,active in enumerate((True,False)):
                points=line_endpoints(p==int(active))
                best=bounds[k]
                for position in points:
                    r=tuple(a+2*int(v) for a,v in zip(low,position))
                    # Use positive denominator and cross-multiply. Normalize only the winner.
                    n=-value(back,r,denom)
                    den=-value(front,r,denom)
                    if den<=0:
                        raise AssertionError("invalid receiver chart")
                    if best is None or (n*best[1]>best[0]*den if active else n*best[1]<best[0]*den):
                        best=n,den
                bounds[k]=best
        else:
            failed=(p==int(state=='constant_hit')).any()
    if failed:
        return 'exact_empty',Fraction(0),scans,len(hinges)
    if scans['unresolved'] or len(hinges)>8:
        return 'unresolved',None,scans,len(hinges)
    if not legal_volume(bodies,box,denom,o,len(cls)):
        return 'legality_unresolved',None,scans,len(hinges)
    cuts=[]
    for (c,i,j),(lb,ub) in hinges.items():
        front,back=bodies[c][i],bodies[c][j]
        if lb is not None:
            n,h=lb
            cuts.append((tuple(n*a-h*b for a,b in zip(front,back)),True))
        if ub is not None:
            n,h=ub
            cuts.append((tuple(h*b-n*a for a,b in zip(front,back)),False))
    return 'exact',joint_fraction(box,denom,cuts),scans,len(hinges)


def baseline(P,K,cls,o,N,cell):
    from rind3d_posterior.fastlabel import engine as T
    from rind3d_posterior.fastlabel.window import LazyWindow2
    win=LazyWindow2(cls,np.array(o),N)
    codes,kls=win.top_items()
    geom=T.Geom(P,K)
    root=np.array([[7,cell//16384,(cell//128)%128,cell%128]],np.int64)
    r=T.stage2(geom,win,codes,kls,root)
    if np.any((r['cell'] != cell) & (r['hi'] > 0)):
        raise AssertionError('Tau emitted positive volume outside the common cell')
    return [float(r[k].sum())*128**3 for k in ('lo','hi')],r['stats']['undet']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('comparison',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    tasks=json.loads((a.comparison/'tasks.json').read_bytes())
    results=[]
    jit=False
    for t in tasks:
        key=t['key']
        with np.load(a.comparison/'labels'/(key+'_tp.npz'),allow_pickle=False) as lab:
            cells,inv=np.unique(lab['cell'],return_inverse=True)
            cell=int(cells[np.argmax(np.bincount(inv,weights=lab['lo']))])
        with np.load(t['container'],allow_pickle=False) as z:
            w,s=t['w'],t['s']
            win=z['windows'][w]
            N=int(json.loads(z['meta'].tobytes())['resolution'])
            o=tuple(int(win[k]) for k in ('x','y','z'))
            L=int(win['L'])
            idx={(int(i),int(j)):(int(off),int(n)) for i,j,off,n in z['obs_index']}
            blob=z['obs'].tobytes()
            cls=cropz(blob,idx,w,s,L)
            cls[cropz(blob,idx,w,-1,L).astype(bool)]=2
            P,K=z['P'],z['K']
            geom=z['geometry'].tobytes()
        t0=perf_counter()
        bodies=[tuple(integer_row(r) for r in b.rows) for b in decode_convex(load_geometry(io.BytesIO(geom)))]
        box=source_box(cell,2*N//128,2*N)
        outcome,v,stats,hinges=solve(bodies,cls,o,box,2*N)
        rec=dict(key=key,cell=cell,status=outcome,wall_s=perf_counter()-t0,hinges=hinges,stats=stats)
        if v is not None:
            rec['fraction_exact']=str(v)
            rec['fraction']=float(v)
            if not jit:
                t0=perf_counter()
                baseline(P,K,cls,o,N,cell)
                rec['first_tau_warmup_seconds']=perf_counter()-t0
                jit=True
            # Paired runs on decoded inputs. Alternate order, rebuild both geometry
            # representations each time; only imports/JIT and OS caches stay warm.
            new_times, tau_times = [], []
            for repeat in range(5):
                for method in (('patch','tau') if repeat%2==0 else ('tau','patch')):
                    t0=perf_counter()
                    if method=='tau':
                        rec['tau_interval'],rec['tau_undet']=baseline(P,K,cls,o,N,cell)
                        tau_times.append(perf_counter()-t0)
                    else:
                        rebuilt=[tuple(integer_row(r) for r in b.rows) for b in decode_convex(load_geometry(io.BytesIO(geom)))]
                        other=solve(rebuilt,cls,o,box,2*N)
                        new_times.append(perf_counter()-t0)
                        if other[:2]!=(outcome,v):
                            raise AssertionError('nondeterministic conditional volume')
            rec['tau_wall_s']=float(np.median(tau_times))
            rec['patch_warm_s']=float(np.median(new_times))
            rec['paired_runs_seconds']={'tau':tau_times,'patch':new_times}
        results.append(rec)
    a.output.write_text(json.dumps(results,separators=(',',':'))+'\n')
    print(Counter(r['status'] for r in results))


if __name__=='__main__':
    main()
