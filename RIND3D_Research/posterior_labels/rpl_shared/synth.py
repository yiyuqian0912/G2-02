"""Controlled reference problems: axis-aligned / rotated convex boxes, windows rendered with the float renderer
(cases are constructed so that no receiver is undecided), and a brute-force feasibility reference that tests every
window receiver for every sample point (independent of the octree, the shadow rows and the polytope code)."""
import numpy as np
from numba import njit, prange

from .data import Scene
from .predicates import render_points, segment_hits
from .topology import adjacency


def box_rows(lo, hi, R=None):
    """H-rows of the box [lo,hi] (optionally rotated by R about its centre); rows normalised by max |coef|."""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    c = (lo + hi) / 2; h = (hi - lo) / 2
    R = np.eye(3) if R is None else np.asarray(R, float)
    rows = []
    for k in range(3):
        for sgn in (1, -1):
            n = sgn * R[:, k]
            d = -(n @ c) - h[k]
            row = np.r_[n, d]
            rows.append(row / np.max(np.abs(row[:3])))
    return np.array(rows)


def geometry_from_rows(bodies):
    m = len(bodies); kmax = max(len(b) for b in bodies)
    P = np.zeros((m, kmax, 4)); K = np.zeros(m, np.int32)
    for c, b in enumerate(bodies):
        P[c, :len(b)] = b; K[c] = len(b)
    clo, chi = Scene.component_aabbs(P, K)
    adj, aoff = adjacency(P, K)
    return P, K, clo, chi, adj, aoff


def inside_solid(p, P, K):
    for c in range(P.shape[0]):
        if np.all(P[c, :K[c], :3] @ p + P[c, :K[c], 3] <= 0):
            return True
    return False


def render_window(geom, s, origin, L):
    P, K = geom[0], geom[1]
    g = np.stack(np.meshgrid(*[np.arange(L)] * 3, indexing='ij'), -1).reshape(-1, 3)
    X = (np.asarray(origin) + g + 0.5) / 1024
    solid = np.array([inside_solid(x, P, K) for x in X]) if L <= 32 else _solid_mask(X, P, K)
    y = render_points(np.asarray(s, float), X, P, K)
    if np.any(y[~solid] < 0):
        raise ValueError('undecided receiver in a synthetic case')
    cls = np.where(solid, 2, np.where(y == 1, 1, 0)).astype(np.uint8)
    return cls.reshape(L, L, L)


@njit(cache=True)
def _solid_mask(X, P, K):
    out = np.zeros(X.shape[0], np.bool_)
    for q in range(X.shape[0]):
        for c in range(P.shape[0]):
            ins = True
            for i in range(K[c]):
                if P[c, i, 0] * X[q, 0] + P[c, i, 1] * X[q, 1] + P[c, i, 2] * X[q, 2] + P[c, i, 3] > 0:
                    ins = False
                    break
            if ins:
                out[q] = True
                break
    return out


@njit(parallel=True, cache=True)
def feasible_points(S, X, obs, P, K, wlo, whi):
    """Brute force: S (n,3) candidate sources, X (m,3) FREE receivers with observations obs (1 lit / 0 dark).
    Returns int8: 1 feasible, 0 infeasible, -1 undecided (some receiver too close to call)."""
    n = S.shape[0]
    out = np.empty(n, np.int8)
    for t in prange(n):
        s = S[t]
        inw = True
        for k in range(3):
            if s[k] < wlo[k] or s[k] >= whi[k]:
                inw = False
        if inw:
            out[t] = 0
            continue
        ok = 1
        for q in range(X.shape[0]):
            y = 1
            und = False
            for c in range(P.shape[0]):
                h = segment_hits(s, X[q], P, K, c)
                if h == 1:
                    y = 0
                    und = False
                    break
                if h == -1:
                    und = True
            if und:
                ok = -1
                continue
            if y != obs[q]:
                ok = 0
                break
        out[t] = ok
    return out


def window_receivers(cls, origin):
    L = cls.shape[0]
    g = np.stack(np.meshgrid(*[np.arange(L)] * 3, indexing='ij'), -1).reshape(-1, 3)
    c = cls.reshape(-1)
    free = c != 2
    X = (np.asarray(origin) + g[free] + 0.5) / 1024
    return np.ascontiguousarray(X), (c[free] == 1).astype(np.int8)
