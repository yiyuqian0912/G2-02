"""Bounded convex 3-polytopes with primitive homogeneous integer vertices.

Full-dimensional interiors are disjoint after a split. Lower-dimensional pieces
are omitted only for Lebesgue mass. Use the original Boolean formula for points.
"""
from fractions import Fraction
from functools import cached_property
from itertools import product
from math import gcd


def form(row, p):
    a,b,c,d = row
    x,y,z,w = p
    return a*x + b*y + c*z + d*w


def primitive(v):
    k = gcd(*v)
    return tuple(x//k for x in v) if k else tuple(v)


def cap_hull(points, plane):
    """Order cap vertices exactly in an injective coordinate projection."""
    drop = max(range(3), key=lambda k: abs(plane[k]))
    i,j = [k for k in range(3) if k != drop]
    p = sorted(points, key=lambda v:(Fraction(v[i],v[3]), Fraction(v[j],v[3])))
    def chain(items):
        out = []
        for c in items:
            while len(out) >= 2:
                a,b = out[-2:]
                # Homogeneous 2D orientation. All w are positive.
                dx,dy = b[i]*a[3]-a[i]*b[3], b[j]*a[3]-a[j]*b[3]
                if dx*(c[j]*a[3]-a[j]*c[3]) > dy*(c[i]*a[3]-a[i]*c[3]):
                    break
                out.pop()
            out.append(c)
        return out
    return tuple(chain(p)[:-1] + chain(reversed(p))[:-1])


class Poly:
    def __init__(self, faces):
        self.faces = tuple(faces)
        self.vertices = tuple(dict.fromkeys(p for face in faces for p in face))

    @classmethod
    def cube(cls):
        faces = []
        for axis,bit in product(range(3),(0,1)):
            i,j = [k for k in range(3) if k!=axis]
            face = []
            for u,v in ((0,0),(0,1),(1,1),(1,0)):
                p = [0,0,0,1]
                p[axis],p[i],p[j] = bit,u,v
                face.append(tuple(p))
            faces.append(tuple(face))
        return cls(faces)

    @cached_property
    def fvertices(self):
        # Correctly rounded int/int division; no overflow for these magnitudes.
        return tuple((p[0] / p[3], p[1] / p[3], p[2] / p[3]) for p in self.vertices)

    def side(self, row):
        """Classify both halfspaces from exact homogeneous integers."""
        neg = pos = False
        for vertex in self.vertices:
            value = form(row, vertex)
            neg |= value < 0
            pos |= value > 0
            if neg and pos:
                return 0
        return 1 if pos else -1

    def split(self, row):
        """Return closures of negative and positive volume pieces, respectively."""
        side = self.side(row)
        if side < 0:
            return self,None
        if side > 0:
            return None,self
        signs = {p:form(row,p) for p in self.vertices}
        if max(signs.values()) <= 0:
            return self,None
        if min(signs.values()) >= 0:
            return None,self
        neg,pos,cap = [],[],set()
        cut = {}  # each crossing edge is shared by two faces; its point is symmetric in the endpoints
        for face in self.faces:
            a,b = [],[]
            last = face[-1]
            for p in face:
                u,v = signs[last],signs[p]
                if u < 0 < v or v < 0 < u:
                    key = (last, p) if last < p else (p, last)
                    h = cut.get(key)
                    if h is None:
                        h = primitive(tuple(abs(v)*x + abs(u)*y for x,y in zip(last,p)))
                        cut[key] = h
                    a.append(h)
                    b.append(h)
                    cap.add(h)
                if v <= 0: a.append(p)
                if v >= 0: b.append(p)
                if v == 0: cap.add(p)
                last = p
            if len(a) >= 3: neg.append(tuple(a))
            if len(b) >= 3: pos.append(tuple(b))
        newface = cap_hull(cap,row)
        return Poly(neg+[newface]),Poly(pos+[newface])

    @cached_property
    def volume(self):
        def xyz(p):
            return tuple(Fraction(x,p[3]) for x in p[:3])
        o = xyz(self.vertices[0])
        v = {p:tuple(x-y for x,y in zip(xyz(p),o)) for p in self.vertices}
        total = 0
        for face in self.faces:
            a,b,c = v[face[0]]
            for p,q in zip(face[1:-1],face[2:]):
                x,y,z = v[p]
                h,i,j = v[q]
                total += abs(a*(y*j-z*i) + b*(z*h-x*j) + c*(x*i-y*h))
        return total/6

    def grid_bounds(self, n):
        # Positive-volume range; closed facets have zero grid mass.
        return tuple((min(p[k]*n//p[3] for p in self.vertices),
                      max((p[k]*n-1)//p[3] for p in self.vertices)) for k in range(3))


def within(poly, rows):
    for row in rows:
        poly,_ = poly.split(row)
        if poly is None: break
    return poly


def without(poly, rows):
    """Subtract one closed convex set, retaining disjoint positive-volume interiors."""
    for r in rows:
        if poly.side(r) > 0:  # all vertex values >= 0 and at least one > 0
            return [poly]
    result = []
    for r in rows:
        poly,outside = poly.split(r)
        if outside is not None: result.append(outside)
        if poly is None: break
    return result


def grid_pieces(poly, n):
    """Lazy exact integration with bisection; never allocate an n³ tensor."""
    stack = [poly]
    while stack:
        p = stack.pop()
        dims = p.grid_bounds(n)
        k = max(range(3),key=lambda axis:dims[axis][1]-dims[axis][0])
        lo,hi = dims[k]
        if hi == lo:
            yield tuple(l for l,h in dims),p
            continue
        plane = [0,0,0,-((lo+hi+1)//2)]
        plane[k] = n
        left,right = p.split(plane)
        if left is None or right is None:
            raise ArithmeticError("inconsistent grid extent")
        stack.extend((left,right))
