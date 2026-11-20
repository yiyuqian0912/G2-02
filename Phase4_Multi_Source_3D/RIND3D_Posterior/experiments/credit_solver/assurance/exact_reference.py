"""Independent Cartesian H-polytope integration. Enumerate triple intersections; no solver split or fan code."""
from fractions import Fraction as Q
from itertools import combinations


def cross(a,b):
    return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])


def dot(a,b):
    return sum(x*y for x,y in zip(a,b))


def sub(a,b):
    return tuple(x-y for x,y in zip(a,b))


def volume(rows):
    """Divergence theorem on exactly ordered facets; callers supply bounded intersections."""
    verts=set()
    for a,b,c in combinations(rows,3):
        u,v,w=cross(b,c),cross(c,a),cross(a,b)
        det=dot(a[:3],u)
        if not det:
            continue
        p=tuple(Q(-a[3]*u[i]-b[3]*v[i]-c[3]*w[i],det) for i in range(3))
        if all(dot(r[:3],p)+r[3]<=0 for r in rows):
            verts.add(p)
    total=Q()
    seen=set()
    for r in rows:
        points=frozenset(p for p in verts if dot(r[:3],p)+r[3]==0)
        if len(points)<3 or points in seen:
            continue
        seen.add(points)
        axis=max(range(3),key=lambda i:abs(r[i]))
        x,y=(axis+1)%3,(axis+2)%3
        ordered=sorted(points,key=lambda p:(p[x],p[y]))
        def chain(seq):
            path=[]
            for c in seq:
                while len(path)>=2:
                    a,b=path[-2:]
                    if (b[x]-a[x])*(c[y]-a[y])>(b[y]-a[y])*(c[x]-a[x]):
                        break
                    path.pop()
                path.append(c)
            return path[:-1]
        hull=chain(ordered)+chain(reversed(ordered))
        for b,c in zip(hull[1:],hull[2:]):
            # dot(x,n) dA = -offset dA/|n|; projected signed area removes the square root.
            a=hull[0]
            twice=(b[x]-a[x])*(c[y]-a[y])-(b[y]-a[y])*(c[x]-a[x])
            total-=Q(r[3],abs(r[axis]))*twice/6
    # Fewer than four affinely independent vertices means zero three-volume.
    if len(verts)<4 or not any(dot(sub(b,a),cross(sub(c,a),sub(d,a))) for a,b,c,d in combinations(verts,4)):
        return Q()
    return total
