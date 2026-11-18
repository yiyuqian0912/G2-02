"""Exact inputs of the verifier: canonical geometry rows, component vertex/edge enclosures, the window observation.

Geometry: the canonical bundle is decoded by the generator's `decode_convex` (binary64 parameters read as exact
rationals; rows a.x + d <= 0 normalised by max |coefficient|). This decoder and the stored observation bits are
inputs of the posterior definition, not of any labelling engine. Every float derived here is an enclosure:
  RM[c, i] (binary64, round-to-nearest of the exact coefficient), RR[c, i] >= |exact - RM| (componentwise),
  FB[c, i] >= |f_i(x) - fl(RM.x)| for every binary64 point x with |x_k| <= 2 (row radius + evaluation rounding),
  aabb_lo/hi: outward-rounded box of the exact vertices of each component."""
import itertools
from fractions import Fraction as Q

import numpy as np

from .ar import U, INFL, TINY, gamma

G4 = gamma(4) * (1.0 + 2.0 ** -40)


def _enclose(q):
    """(mid, rad): mid = binary64 nearest to the rational q, rad >= |q - mid| (exact check)."""
    m = float(q)
    d = abs(Q(m) - q)
    if d == 0:
        return m, 0.0
    r = float(d)
    if Q(r) < d:
        r = float(np.nextafter(r, np.inf))
    return m, r


def _enclose_dd(q):
    """(hi, lo, rad): hi + lo is a double-double approximation of the rational q, rad >= |q - (hi + lo)| (exact)."""
    hi = float(q)
    lo = float(q - Q(hi))
    d = abs(q - Q(hi) - Q(lo))
    if d == 0:
        return hi, lo, 0.0
    r = float(d)
    if Q(r) < d:
        r = float(np.nextafter(r, np.inf))
    return hi, lo, r


def _dn(q):
    m = float(q)
    return m if Q(m) <= q else float(np.nextafter(m, -np.inf))


def _up(q):
    m = float(q)
    return m if Q(m) >= q else float(np.nextafter(m, np.inf))


def exact_vertices(rows):
    """Exact vertices (Fractions) of {x : a.x + d <= 0} and, for every facet pair, the vertices on both facets."""
    n = len(rows)
    pts = {}
    for i, j, k in itertools.combinations(range(n), 3):
        A = [rows[t][:3] for t in (i, j, k)]; b = [-rows[t][3] for t in (i, j, k)]
        det = (A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
               + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]))
        if det == 0:
            continue
        x = []
        for c in range(3):
            M = [list(r) for r in A]
            for r in range(3):
                M[r][c] = b[r]
            dc = (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                  + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
            x.append(dc / det)
        x = tuple(x)
        if x in pts:
            continue
        if all(r[0] * x[0] + r[1] * x[1] + r[2] * x[2] + r[3] <= 0 for r in rows):
            pts[x] = frozenset(t for t, r in enumerate(rows) if r[0] * x[0] + r[1] * x[1] + r[2] * x[2] + r[3] == 0)
    return pts


def bounded(rows):
    """Exact test that {x : a.x + d <= 0} is bounded: its recession cone {u : a_i.u <= 0 for all i} is {0}. A
    nonzero recession cone has an extreme ray on the intersection of two of its planes (or, if all normals are
    coplanar, along their common normal, which is again some a_i x a_j), so it suffices to test u = +-(a_i x a_j)."""
    N = [r[:3] for r in rows]
    for a, b in itertools.combinations(N, 2):
        u = (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
        if not any(u):
            continue
        for sg in (1, -1):
            if all(sg * (n[0] * u[0] + n[1] * u[1] + n[2] * u[2]) <= 0 for n in N):
                return False
    return True


class ExactScene:
    """Canonical geometry of one scene for the verifier."""

    def __init__(self, components, P_engine=None, K_engine=None):
        self.rows = [tuple(tuple(Q(v) for v in r) for r in c.rows) for c in components]
        C = len(self.rows)
        if not 1 <= C <= 62:
            raise ValueError("verifier supports 1..62 components")
        kmax = max(len(r) for r in self.rows)
        self.C, self.K = C, np.array([len(r) for r in self.rows], np.int64)
        self.RM = np.zeros((C, kmax, 4)); self.RR = np.zeros((C, kmax, 4)); self.FB = np.zeros((C, kmax))
        self.RL = np.zeros((C, kmax, 4)); self.RR2 = np.zeros((C, kmax, 4))     # double-double rows, residual radius
        self.aabb_lo = np.zeros((C, 3)); self.aabb_hi = np.zeros((C, 3))
        self.edges = []          # (c, i, j, A (exact), B (exact)) for every facet pair sharing >= 2 vertices
        for c, rows in enumerate(self.rows):
            for i, r in enumerate(rows):
                for t in range(4):
                    self.RM[c, i, t], self.RR[c, i, t] = _enclose(r[t])
                    h, l_, rr = _enclose_dd(r[t])
                    assert h == self.RM[c, i, t]
                    self.RL[c, i, t] = l_; self.RR2[c, i, t] = rr
                ab = float(np.sum(np.abs(self.RM[c, i, :3])) * 2 + abs(self.RM[c, i, 3]))
                self.FB[c, i] = (float(np.sum(self.RR[c, i, :3])) * 2 + self.RR[c, i, 3] + G4 * ab) * INFL + TINY
            V = exact_vertices(rows)
            if len(V) < 4 or not bounded(rows):
                raise ValueError(f"component {c} is not a bounded 3D polytope")
            for t in range(3):
                self.aabb_lo[c, t] = _dn(min(v[t] for v in V)); self.aabb_hi[c, t] = _up(max(v[t] for v in V))
            K = len(rows)
            for i in range(K):
                for j in range(i + 1, K):
                    on = [v for v, s in V.items() if i in s and j in s]
                    if len(on) >= 2:
                        # endpoints: lexicographic extremes along the segment direction
                        d = [on[1][t] - on[0][t] for t in range(3)]
                        key = lambda p: sum(p[t] * d[t] for t in range(3))
                        self.edges.append((c, i, j, min(on, key=key), max(on, key=key)))
        # occluder edge enclosures (exact endpoints rounded to mid +- radius) for clearance-shell arguments
        nE = len(self.edges)
        self.E_c = np.zeros(max(nE, 1), np.int64); self.E_A = np.zeros((max(nE, 1), 3)); self.E_B = np.zeros((max(nE, 1), 3))
        self.E_r = np.zeros(max(nE, 1)); self.E_i = np.zeros(max(nE, 1), np.int64); self.E_j = np.zeros(max(nE, 1), np.int64)
        for t, (c, i, j, A, B) in enumerate(self.edges):
            self.E_c[t] = c; self.E_i[t] = i; self.E_j[t] = j
            r = 0.0
            for k in range(3):
                self.E_A[t, k], ra = _enclose(A[k]); self.E_B[t, k], rb = _enclose(B[k]); r = max(r, ra, rb)
            self.E_r[t] = r
        self.nE = nE
        # exact facet adjacency (facet pairs sharing an edge of the component): with it, the shadow cone of a convex
        # polytope from an external point is bounded exactly by its silhouette-edge planes (check.py)
        self.ADJ = np.zeros((C, kmax, kmax), np.bool_)
        for (c, i, j, A, B) in self.edges:
            self.ADJ[c, i, j] = True; self.ADJ[c, j, i] = True
        if P_engine is not None:
            self.check_engine_rows(P_engine, K_engine)

    def check_engine_rows(self, P, K):
        """The engine's face indices must name the same facets: engine row = positive multiple of the exact row up
        to rounding (only the event recipes use the engine's indexing; geometry always comes from the exact rows)."""
        for c in range(self.C):
            if int(K[c]) != int(self.K[c]):
                raise ValueError(f"component {c}: engine has {K[c]} rows, canonical geometry {self.K[c]}")
            for i in range(self.K[c]):
                a = np.asarray(P[c, i], float); b = self.RM[c, i]
                s = float(a @ b) / float(b @ b)
                if not (s > 0 and np.max(np.abs(a - s * b)) <= 1e-9 * max(1.0, np.max(np.abs(a)))):
                    raise ValueError(f"engine row ({c}, {i}) does not match the canonical facet")


def scene_from_container(container):
    """ExactScene of a RIND3D-100K scene container (canonical bundle decoded by rind3d.rendering.polyhedra)."""
    import io, json
    from rind3d.geometry import load_geometry
    from rind3d.rendering.polyhedra import decode_convex
    with np.load(container, allow_pickle=False) as z:
        geo = z["geometry"].tobytes(); P = z["P"]; K = z["K"]
    comps = decode_convex(load_geometry(io.BytesIO(geo)))
    return ExactScene(comps, P, K)


def window_from_container(container, w, s):
    """(cls (L, L, L) uint8: 0 response-inactive FREE, 1 response-active FREE, 2 SOLID; origin (3,) int64; N).
    Decoded independently of the labelling code: packed bits [Z, Y, X], big-endian bit order, zstd frames."""
    import json
    import zstandard
    with np.load(container, allow_pickle=False) as z:
        meta = json.loads(z["meta"].tobytes().decode())
        W = z["windows"]; idx = z["obs_index"]; blob = z["obs"].tobytes()
    pos = {(int(a), int(b)): (int(o), int(n)) for a, b, o, n in idx}
    r = W[int(w)]; L = int(r["L"]); origin = np.array([int(r["x"]), int(r["y"]), int(r["z"])], np.int64)
    dz = zstandard.ZstdDecompressor()

    def bits(j):
        o, n = pos[(int(w), j)]
        b = np.frombuffer(dz.decompress(blob[o:o + n]), np.uint8)
        return np.unpackbits(b, bitorder="big")[:L ** 3].reshape(L, L, L).transpose(2, 1, 0).astype(bool)   # -> [x, y, z]

    lit = bits(int(s)); solid = bits(-1)
    cls = np.where(solid, 2, np.where(lit, 1, 0)).astype(np.uint8)
    return cls, origin, int(meta["resolution"])
