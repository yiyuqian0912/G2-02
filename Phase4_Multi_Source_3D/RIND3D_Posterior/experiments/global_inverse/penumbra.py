"""Extra valid planes for all possible shadows of a convex source region."""
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from poly import form,primitive


def cross(a,b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def directions(body_geom,source):
    k=body_geom.vertices
    p=source.vertices
    def floats(vs):
        return np.array([[v[j]/v[3] for j in range(3)] for v in vs])
    dif=(floats(k)[:,None,:]-floats(p)[None,:,:]).reshape(-1,3)
    # Projection axis (proposal only): any a with a.(k-p) > 0 for all vertex pairs. Try the
    # mean difference, the centroid direction, then minus each facet normal of K (a facet
    # of K that strictly separates P gives such an axis exactly). Validation below is exact.
    candidates=[dif.mean(axis=0), floats(k).mean(axis=0)-floats(p).mean(axis=0)]
    fk=floats(k)
    for face in body_geom.faces:
        q=floats(face[:3])
        n=np.cross(q[1]-q[0],q[2]-q[0])
        if not np.any(n): continue
        if np.dot(n,fk.mean(axis=0)-q[0])>0: n=-n   # outward normal
        candidates.append(-n)
    for axis in candidates:
        scale=dif@axis
        if np.all(scale>0) and np.all(np.isfinite(scale)): break
    else:
        return ()
    # Silhouette candidates only. No decisions use these float coordinates.
    drop=np.argmax(np.abs(axis))
    proj=np.delete(dif,drop,axis=1)/scale[:,None]
    if not np.isfinite(proj).all(): return ()
    try:
        order=ConvexHull(proj).vertices
    except QhullError:
        return ()
    rows=[]
    def ray(idx):
        u,v=divmod(int(idx),len(p))
        return tuple(k[u][j]*p[v][3]-p[v][j]*k[u][3] for j in range(3)), k[u]
    for first,second in zip(order,np.roll(order,-1)):
        d,v=ray(first)
        e,_=ray(second)
        n=cross(d,e)
        if not any(n): continue
        h=primitive((*(x*v[3] for x in n),-sum(n[j]*v[j] for j in range(3))))
        kv=[form(h,t) for t in k]
        pv=[form(h,t) for t in p]
        if max(kv)<=0<=min(pv): rows.append(h)
        elif min(kv)>=0>=max(pv): rows.append(tuple(-x for x in h))
    return tuple(dict.fromkeys(rows))
