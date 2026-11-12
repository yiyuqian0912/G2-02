"""Controlled reference cases at physical resolution N (declared in code).  T1-T4 are the first study's cases
(rpl synth_cases.py, N = 1024); the same constructions are available at N = 128 for Stage A."""
import numpy as np
from rpl.synth import box_rows
from rpl.predicates import render_points
from tpl.geom import Geom
from tpl.window import Window


def render_window(geom, s, origin, L, N):
    g = np.stack(np.meshgrid(*[np.arange(L)] * 3, indexing='ij'), -1).reshape(-1, 3)
    X = (np.asarray(origin) + g + 0.5) / N
    P, K = geom.P, geom.K
    solid = np.zeros(len(X), bool)
    for c in range(P.shape[0]):
        solid |= np.all(X @ P[c, :K[c], :3].T + P[c, :K[c], 3] <= 0, axis=1)
    y = render_points(np.asarray(s, float), X, P, K)
    if np.any(y[~solid] < 0):
        raise ValueError('undecided receiver')
    return np.where(solid, 2, np.where(y == 1, 1, 0)).astype(np.uint8).reshape(L, L, L)


def find_transition(geom, s, x, z, ylo=0.2, yhi=0.8, n=4000):
    ys = np.linspace(ylo, yhi, n)
    R = np.stack([np.full(n, x), ys, np.full(n, z)], 1)
    y = render_points(np.asarray(s, float), R, geom.P, geom.K)
    k = np.flatnonzero(np.diff(y.astype(int)) != 0)
    return ys[k]


def case(name, N=1024, L=16):
    if name == 'T1_edge_slab':
        bodies = [box_rows([0.40, 0.40, 0.20], [0.50, 0.50, 0.80])]
        s = np.array([0.15, 0.32, 0.47]); x, z = 0.72, 0.47
    elif name == 'T2_split_slab':
        bodies = [box_rows([0.40, 0.40, 0.20], [0.50, 0.50, 0.80]),
                  box_rows([0.20, 0.05, 0.62], [0.30, 0.95, 0.70])]
        s = np.array([0.15, 0.32, 0.47]); x, z = 0.72, 0.47
    elif name == 'T3_corner_tube':
        bodies = [box_rows([0.40, 0.40, 0.40], [0.50, 0.50, 0.50])]
        s = np.array([0.15, 0.32, 0.30]); x, z = 0.72, 0.62
    elif name == 'T4_tangent_face':
        bodies = [box_rows([0.40, 0.40, 0.20], [(N // 2 - 0.5) / N, 0.50, 0.80])]
        s = np.array([0.15, 0.32, 0.47]); x, z = (N // 2 + 3) / N, 0.47
    elif name == 'T5_rot_cluster':
        # three rotated boxes: overlapping shadows (multi-candidate dark receivers), silhouette vertices
        from scipy.spatial.transform import Rotation as Rot
        R1 = Rot.from_euler('xyz', [20, 35, 10], degrees=True).as_matrix()
        R2 = Rot.from_euler('xyz', [-15, 10, 40], degrees=True).as_matrix()
        R3 = Rot.from_euler('xyz', [5, -30, 25], degrees=True).as_matrix()
        bodies = [box_rows([0.40, 0.38, 0.40], [0.47, 0.50, 0.52], R1),
                  box_rows([0.44, 0.47, 0.36], [0.52, 0.55, 0.45], R2),
                  box_rows([0.36, 0.50, 0.47], [0.43, 0.58, 0.56], R3)]
        s = np.array([0.18, 0.41, 0.43]); x, z = 0.70, 0.47
    else:
        raise KeyError(name)
    geom = Geom.from_bodies(bodies)
    ys = find_transition(geom, s, x, z)
    yc = ys[0]
    if name == 'T3_corner_tube':
        zc = s[2] + (0.5 - s[2]) * (x - s[0]) / (0.5 - s[0])
        yc = find_transition(geom, s, x, zc - 4.0 / N)[0]
        o = (int(x * N) - L // 2, int(yc * N) - L // 2, int(zc * N) - L // 2)
    elif name == 'T4_tangent_face':
        o = (N // 2 - 8 * N // 1024, int(yc * N) - L // 2, int(z * N) - L // 2)
    else:
        o = (int(x * N) - L // 2, int(yc * N) - L // 2, int(z * N) - L // 2)
    cls = render_window(geom, s, o, L, N)
    return dict(name=name, geom=geom, truth=s, win=Window(cls, o, N))
