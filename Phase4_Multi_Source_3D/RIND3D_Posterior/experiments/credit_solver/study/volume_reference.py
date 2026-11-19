"""Independent exact Cartesian-coordinate differential for the homogeneous fan integral."""
from fractions import Fraction as Q


def cartesian_volume(poly):
    anchor = poly.vertices[0]
    offsets = {p: tuple(Q(x,p[3])-Q(y,anchor[3]) for x,y in zip(p[:3],anchor[:3])) for p in poly.vertices}
    total = Q()
    for face in poly.faces:
        a,b,c = offsets[face[0]]
        for p,q in zip(face[1:-1], face[2:]):
            x,y,z = offsets[p]
            h,i,j = offsets[q]
            total += abs(a*(y*j-z*i)+b*(z*h-x*j)+c*(x*i-y*h))
    return total/6
