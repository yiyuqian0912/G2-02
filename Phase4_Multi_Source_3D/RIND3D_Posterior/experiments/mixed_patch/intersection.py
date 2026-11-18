"""Small exact 3D volume reference for intersections of certified mixed-patch bands."""
from fractions import Fraction
from itertools import combinations
from math import gcd, lcm


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def dot(a, b):
    return sum(x*y for x,y in zip(a,b))


def delta(a,b):
    return tuple(x-y for x,y in zip(a,b))


def face_hull(points, normal):
    """Exact monotone-chain boundary in an injective projection of the face."""
    drop = max(range(3), key=lambda d: abs(normal[d]))
    i,j = [k for k in range(3) if k != drop]
    p = sorted(points, key=lambda v:(v[i],v[j]))
    def chain(vertices):
        out = []
        for c in vertices:
            while len(out) >= 2:
                a,b = out[-2:]
                if (b[i]-a[i])*(c[j]-a[j]) > (b[j]-a[j])*(c[i]-a[i]):
                    break
                out.pop()
            out.append(c)
        return out
    return chain(p)[:-1]+chain(reversed(p))[:-1]


def joint_fraction(box, denom, cuts):
    """Fraction of box satisfying (row,strict) halfspaces, all exact rational.

    World coordinates are lattice numerators divided by denom. Boundary-only
    distinctions have zero 3D mass except when an affine form is identically zero.
    """
    lo, hi = box
    if denom <= 0 or any(a >= b for a,b in zip(lo,hi)):
        raise ValueError("positive 3D box required")
    rows = set()
    for row, strict in cuts:
        mult = lcm(*(x.denominator for x in row))
        row = tuple(int(x*mult) for x in row)
        if not any(row[:3]):
            if row[3] > 0 or (strict and row[3] == 0):
                return Fraction(0)
            continue
        row = (*row[:3], row[3]*denom)
        d = gcd(*row)
        rows.add(tuple(x//d for x in row))
    for k in range(3):
        axis = tuple(int(i == k) for i in range(3))
        rows.add((*axis,-hi[k]))
        rows.add((*(-i for i in axis),lo[k]))
    rows = sorted(rows)
    verts = set()
    for a,b,c in combinations(rows,3):
        ab,bc,ca = cross(a,b),cross(b,c),cross(c,a)
        det = dot(ab,c[:3])
        if det == 0:
            continue
        xyz = tuple(-(a[3]*bc[k]+b[3]*ca[k]+c[3]*ab[k]) for k in range(3))
        if det < 0:
            det,xyz = -det, tuple(-x for x in xyz)
        if all(dot(r[:3],xyz)+r[3]*det <= 0 for r in rows):
            verts.add(tuple(Fraction(x,det) for x in xyz))
    if len(verts) < 4:
        return Fraction(0)
    center = tuple(sum(p[k] for p in verts)/len(verts) for k in range(3))
    vol6 = Fraction(0)
    for row in rows:
        face = [p for p in verts if dot(row[:3],p)+row[3] == 0]
        if len(face) < 3:
            continue
        h = face_hull(face,row)
        for b,c in zip(h[1:-1],h[2:]):
            vol6 += abs(dot(delta(h[0],center),cross(delta(b,center),delta(c,center))))
    return vol6 / (6*(hi[0]-lo[0])*(hi[1]-lo[1])*(hi[2]-lo[2]))


def band_cuts(front, back, lower, upper):
    # lower < back(s)/front(s) <= upper, front(s) > 0 in the certified box.
    return [(tuple(lower*a-b for a,b in zip(front,back)), True),
            (tuple(b-upper*a for a,b in zip(front,back)), False)]
