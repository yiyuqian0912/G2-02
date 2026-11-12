"""Controlled reference cases (declared in code)."""
import sys, numpy as np
sys.path.insert(0, 'code')
from rpl.synth import box_rows, geometry_from_rows, render_window
from rpl.predicates import render_points

def find_transition(geom, s, x, z, ylo=0.2, yhi=0.8, n=4000):
    ys = np.linspace(ylo, yhi, n)
    R = np.stack([np.full(n, x), ys, np.full(n, z)], 1)
    y = render_points(np.asarray(s, float), R, geom[0], geom[1])
    k = np.flatnonzero(np.diff(y.astype(int)) != 0)
    return ys[k]

def case(name):
    if name == 'T1_edge_slab':
        bodies = [box_rows([0.40, 0.40, 0.20], [0.50, 0.50, 0.80])]
        s = np.array([0.15, 0.32, 0.47]); x, z = 0.72, 0.47
    elif name == 'T2_split_slab':
        bodies = [box_rows([0.40, 0.40, 0.20], [0.50, 0.50, 0.80]),
                  box_rows([0.20, 0.05, 0.62], [0.30, 0.95, 0.70])]     # a plank that hides part of the slab
        s = np.array([0.15, 0.32, 0.47]); x, z = 0.72, 0.47
    elif name == 'T3_corner_tube':
        bodies = [box_rows([0.40, 0.40, 0.40], [0.50, 0.50, 0.50])]
        s = np.array([0.15, 0.32, 0.30]); x, z = 0.72, 0.62
    elif name == 'T4_tangent_face':
        # far x-face exactly on the receiver-centre plane x = 511.5/1024: receivers coplanar with a face
        bodies = [box_rows([0.40, 0.40, 0.20], [511.5 / 1024, 0.50, 0.80])]
        s = np.array([0.15, 0.32, 0.47]); x, z = 0.515, 0.47
    else:
        raise KeyError(name)
    geom = geometry_from_rows(bodies)
    ys = find_transition(geom, s, x, z)
    yc = ys[0]
    L = 16
    if name == 'T3_corner_tube':
        # shadow top boundary of the far top edge (x = 0.5, z = 0.5) at this x
        zc = s[2] + (0.5 - s[2]) * (x - s[0]) / (0.5 - s[0])
        ys = find_transition(geom, s, x, zc - 0.004)
        yc = ys[0]
        o = (int(x * 1024) - L // 2, int(yc * 1024) - L // 2, int(zc * 1024) - L // 2)
    elif name == 'T4_tangent_face':
        o = (504, int(yc * 1024) - L // 2, int(z * 1024) - L // 2)
    else:
        o = (int(x * 1024) - L // 2, int(yc * 1024) - L // 2, int(z * 1024) - L // 2)
    cls = render_window(geom, s, o, L)
    return dict(name=name, geom=geom, truth=s, origin=o, L=L, cls=cls)

if __name__ == '__main__':
    for nm in ['T1_edge_slab', 'T2_split_slab', 'T3_corner_tube']:
        c = case(nm); cl = c['cls']
        print(nm, c['origin'], 'lit', int((cl == 1).sum()), 'dark', int((cl == 0).sum()), 'solid', int((cl == 2).sum()))
