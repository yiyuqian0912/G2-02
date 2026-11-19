"""One remaining facet-pair inequality on a convex source × receiver product."""
from poly import form


def _oriented(start, end, ids):
    # Homogeneous vertex weights are positive. Every extremum sign of a
    # separately affine expression follows from testing all vertex pairs.
    front = [i for i in ids if min(start[i])>0]
    back = [j for j in ids if max(start[j])<0]
    if not front or len(front)+len(back)!=len(ids) or any(max(end[i])>0 for i in front):
        return "unknown", None
    varying = []
    for i in front:
        for j in back:
            values = [a*v-b*u for a,b in zip(start[i],start[j])
                      for u,v in zip(end[i],end[j])]
            if min(values)>0:
                return "miss", None
            if max(values)>0:
                varying.append((i,j))
    if not varying:
        return "hit", None
    if len(varying)==1 and max(end[varying[0][0]])<0:
        return "pencil", varying[0]
    return "unknown", None


def interface_chart(bodies, sources, receivers):
    hinge = None
    possible = 0
    for c, body in enumerate(bodies):
        s = [[form(row,v) for v in sources] for row in body]
        r = [[form(row,v) for v in receivers] for row in body]
        if any(min(a)>0 and min(b)>0 for a,b in zip(s,r)):
            continue
        # On every segment between these polytopes, these omitted rows
        # are already nonpositive by affinity in the segment parameter.
        ids = [i for i,(a,b) in enumerate(zip(s,r)) if max(a)>0 or max(b)>0]
        if not ids:
            return "hit", None
        kind, pair = _oriented(s,r,ids)
        reverse = kind=="unknown"
        if reverse:
            kind, pair = _oriented(r,s,ids)
        if kind=="hit":
            return "hit", None
        if kind!="miss":
            possible += 1
            if pair is not None:
                hinge = (c,*pair,reverse)
    if possible==0:
        return "miss", None
    if possible==1 and hinge is not None:
        return "pencil", hinge
    return "unknown", None


def extend_rank(basis, v):
    """Independent directions of distinct binding pairs; a diagnostic on one source piece."""
    if len(basis)==3 or not any(v):
        return
    if not basis:
        basis.append(v)
        return
    a = basis[0]
    b = basis[1] if len(basis)==2 else v
    cross = (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
    if (len(basis)==1 and any(cross)) or (len(basis)==2 and sum(i*j for i,j in zip(cross,v))!=0):
        basis.append(v)
