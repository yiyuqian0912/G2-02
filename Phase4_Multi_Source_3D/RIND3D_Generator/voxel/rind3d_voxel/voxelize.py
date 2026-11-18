"""Deterministic voxelization of convex polytopes and closed balls on the N^3 grid.

Cells: C_ijk = [i/N, (i+1)/N] x [j/N, (j+1)/N] x [k/N, (k+1)/N] (closed), i, j, k in {0, ..., N-1}.
Rules (one per experiment, never mixed):
* "supercover": C is occupied iff the closed cell meets the closed body (touching a face, edge or vertex counts).
* "center":     C is occupied iff the cell centre lies in the closed body (the current SOLID sampling, promoted to physics).
The union over bodies is a set union, so the result is independent of body and evaluation order.

Exactness. Bodies are exact rationals: polytope rows a.x + d <= 0 from the canonical decoder (`context.components`),
balls with binary64 centre and radius. Each cell decision is computed in float64, and any decision whose separating
gap is within `MARGIN` of zero is recomputed exactly with rationals:
* supercover polytope test: separating-axis test over the polytope's face normals, the three cell axes and the cross
  products of every pair of face normals (a superset of the edge directions) with the cell axes;
* supercover ball test: the squared distance from the centre to the cell;
* centre tests: membership.
Two levels are used: 8^3 blocks are classified first (fully inside / separated / boundary), then the boundary blocks'
cells. A block inside the body has every cell occupied under both rules; a block separated from it has none."""
import itertools
from fractions import Fraction

import numpy as np

MARGIN = 1e-10                 # in units of the cell size: far above float64 error, far below any geometric scale


# ------------------------------------------------------------------ bodies
class Polytope:
    def __init__(self, rows_exact):
        self.rows_q = [tuple(Fraction(v) for v in r) for r in rows_exact]
        R = np.array([[float(v) for v in r] for r in self.rows_q])
        self.A, self.d = R[:, :3], R[:, 3]
        V = []
        for i, j, k in itertools.combinations(range(len(R)), 3):
            M = self.A[[i, j, k]]
            if abs(np.linalg.det(M)) < 1e-14:
                continue
            x = np.linalg.solve(M, -self.d[[i, j, k]])
            if np.all(self.A @ x + self.d <= 1e-9):
                V.append(x)
        self.V = np.unique(np.round(np.array(V), 13), axis=0)
        n = self.A / np.linalg.norm(self.A, axis=1)[:, None]
        cr = [np.cross(n[i], n[j]) for i, j in itertools.combinations(range(len(n)), 2)]
        cr = [c / np.linalg.norm(c) for c in cr if np.linalg.norm(c) > 1e-12]
        self.edges = np.unique(np.round(np.array(cr), 12), axis=0) if cr else np.zeros((0, 3))
        self.lo = self.V.min(0); self.hi = self.V.max(0)

    def exact_vertices(self):
        """Exact rational vertices (for the exact fallback)."""
        out = set()
        for i, j, k in itertools.combinations(range(len(self.rows_q)), 3):
            M = [list(self.rows_q[t][:3]) for t in (i, j, k)]; b = [-self.rows_q[t][3] for t in (i, j, k)]
            x = _solve3(M, b)
            if x is not None and all(r[0] * x[0] + r[1] * x[1] + r[2] * x[2] + r[3] <= 0 for r in self.rows_q):
                out.add(tuple(x))
        return list(out)


class Ball:
    def __init__(self, c, r):
        self.c = np.asarray(c, float); self.r = float(r)
        self.lo = self.c - self.r; self.hi = self.c + self.r


def _solve3(M, b):
    M = [row[:] + [bb] for row, bb in zip(M, b)]
    for p in range(3):
        q = next((r for r in range(p, 3) if M[r][p] != 0), None)
        if q is None:
            return None
        M[p], M[q] = M[q], M[p]
        for r in range(3):
            if r != p and M[r][p] != 0:
                f = M[r][p] / M[p][p]; M[r] = [a - f * c for a, c in zip(M[r], M[p])]
    return [M[i][3] / M[i][i] for i in range(3)]


# ------------------------------------------------------------------ cell / block tests (vectorized over boxes)
def _boxes(lo_idx, hi_idx, N):
    """Boxes [lo, hi] (world units) for integer index ranges (k, 3) inclusive-lo exclusive-hi."""
    return lo_idx / N, hi_idx / N


def poly_box_class(P, blo, bhi):
    """For boxes [blo, bhi] (k, 3): 1 = box inside P, 0 = separated, 2 = meets P but not inside, -1 = undecided."""
    c = 0.5 * (blo + bhi); e = 0.5 * (bhi - blo)
    sc = np.max(e)                                              # length scale for the margin
    # face normals: separated if min over box of a.x + d > 0; inside if max over box <= 0 for every face
    s = c @ P.A.T + P.d[None]; w = e @ np.abs(P.A).T
    mn = s - w; mx = s + w
    sep = np.any(mn > MARGIN * sc * np.abs(P.A).sum(1)[None], axis=1)
    inside = np.all(mx <= -MARGIN * sc * np.abs(P.A).sum(1)[None], axis=1)
    amb = np.any(np.abs(mn) <= MARGIN * sc * np.abs(P.A).sum(1)[None], axis=1) & ~sep
    # box axes
    for k in range(3):
        sk = (P.hi[k] < blo[:, k] - MARGIN * sc) | (P.lo[k] > bhi[:, k] + MARGIN * sc)
        ak = (np.abs(P.hi[k] - blo[:, k]) <= MARGIN * sc) | (np.abs(P.lo[k] - bhi[:, k]) <= MARGIN * sc)
        sep |= sk; amb |= ak & ~sk
    # cross axes (pairs of face normals x cell axes)
    for u in P.edges:
        for k in range(3):
            n = np.cross(u, np.eye(3)[k])
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n = n / nn
            pv = P.V @ n; pmin, pmax = pv.min(), pv.max()
            bc = c @ n; bw = e @ np.abs(n)
            g1 = (bc - bw) - pmax; g2 = pmin - (bc + bw)
            sk = (g1 > MARGIN * sc) | (g2 > MARGIN * sc)
            ak = (np.abs(g1) <= MARGIN * sc) | (np.abs(g2) <= MARGIN * sc)
            sep |= sk; amb |= ak & ~sk
    out = np.where(sep, 0, np.where(inside, 1, 2))
    out[amb & ~sep & ~inside] = -1
    return out


def poly_box_meets_exact(P, blo, bhi):
    """Exact: does the closed box meet the closed polytope? (rational separating-axis test)."""
    lo = [Fraction(float(v)) for v in blo]; hi = [Fraction(float(v)) for v in bhi]
    V = P.exact_vertices()
    corners = list(itertools.product(*zip(lo, hi)))
    def disjoint(n):
        pv = [sum(n[t] * v[t] for t in range(3)) for v in V]; bv = [sum(n[t] * q[t] for t in range(3)) for q in corners]
        return max(pv) < min(bv) or max(bv) < min(pv)
    axes = [r[:3] for r in P.rows_q] + [tuple(Fraction(int(i == k)) for i in range(3)) for k in range(3)]
    nrm = [r[:3] for r in P.rows_q]
    for a, b in itertools.combinations(nrm, 2):
        u = (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
        if any(u):
            for k in range(3):
                ek = [0, 0, 0]; ek[k] = 1
                n = (u[1] * ek[2] - u[2] * ek[1], u[2] * ek[0] - u[0] * ek[2], u[0] * ek[1] - u[1] * ek[0])
                if any(n):
                    axes.append(n)
    return not any(disjoint(n) for n in axes)


def poly_contains_exact(P, x):
    xq = [Fraction(float(v)) for v in x]
    return all(r[0] * xq[0] + r[1] * xq[1] + r[2] * xq[2] + r[3] <= 0 for r in P.rows_q)


def ball_box_class(B, blo, bhi):
    q = np.clip(B.c[None], blo, bhi); d2 = ((q - B.c[None]) ** 2).sum(1)
    far = np.max(np.maximum(np.abs(blo - B.c[None]), np.abs(bhi - B.c[None])) ** 2 * 3, axis=1)   # not used for 'inside'
    corners_far = np.max([((np.where(np.array(bits)[None] == 1, bhi, blo) - B.c[None]) ** 2).sum(1)
                          for bits in itertools.product((0, 1), repeat=3)], axis=0)
    r2 = B.r * B.r; sc = np.max(bhi - blo)
    tol = MARGIN * sc * max(B.r, sc)
    out = np.where(d2 > r2 + tol, 0, np.where(corners_far <= r2 - tol, 1, 2))
    amb = (np.abs(d2 - r2) <= tol) | ((np.abs(corners_far - r2) <= tol) & (d2 <= r2 + tol))
    out[amb] = -1
    return out


def ball_box_meets_exact(B, blo, bhi):
    c = [Fraction(float(v)) for v in B.c]; r = Fraction(float(B.r))
    d2 = sum((min(max(c[k], Fraction(float(blo[k]))), Fraction(float(bhi[k]))) - c[k]) ** 2 for k in range(3))
    return d2 <= r * r


def ball_contains_exact(B, x):
    c = [Fraction(float(v)) for v in B.c]; xq = [Fraction(float(v)) for v in x]; r = Fraction(float(B.r))
    return sum((xq[k] - c[k]) ** 2 for k in range(3)) <= r * r


# ------------------------------------------------------------------ voxelization
def voxelize_body(body, N=1024, rule="supercover", block=8, stats=None):
    """Occupied cell indices (n, 3) int64 [i, j, k] of one body under `rule`."""
    if rule not in ("supercover", "center"):
        raise ValueError(rule)
    st = stats if stats is not None else {}
    lo = np.clip(np.floor(body.lo * N).astype(np.int64) - 1, 0, N - 1); hi = np.clip(np.ceil(body.hi * N).astype(np.int64) + 1, 0, N)
    # blocks
    bi = [np.arange(lo[a] - lo[a] % block, hi[a], block) for a in range(3)]
    B = np.stack(np.meshgrid(*bi, indexing="ij"), -1).reshape(-1, 3)
    blo = B / N; bhi = np.minimum(B + block, N) / N
    cls = (poly_box_class(body, blo, bhi) if isinstance(body, Polytope) else ball_box_class(body, blo, bhi))
    full = B[cls == 1]; bnd = B[cls != 0]
    bnd = bnd[(cls[cls != 0] != 1)]
    out = []
    if len(full):
        g = np.stack(np.meshgrid(*[np.arange(block)] * 3, indexing="ij"), -1).reshape(-1, 3)
        F = (full[:, None, :] + g[None]).reshape(-1, 3); out.append(F[np.all(F < N, axis=1)])
    st["blocks_full"] = st.get("blocks_full", 0) + len(full); st["blocks_boundary"] = st.get("blocks_boundary", 0) + len(bnd)
    if len(bnd):
        g = np.stack(np.meshgrid(*[np.arange(block)] * 3, indexing="ij"), -1).reshape(-1, 3)
        C = (bnd[:, None, :] + g[None]).reshape(-1, 3); C = C[np.all(C < N, axis=1)]
        if rule == "supercover":
            cc = poly_box_class(body, C / N, (C + 1) / N) if isinstance(body, Polytope) else ball_box_class(body, C / N, (C + 1) / N)
            occ = cc >= 1
            amb = np.flatnonzero(cc == -1)
            for i in amb:
                occ[i] = (poly_box_meets_exact(body, C[i] / N, (C[i] + 1) / N) if isinstance(body, Polytope)
                          else ball_box_meets_exact(body, C[i] / N, (C[i] + 1) / N))
        else:
            X = (C + 0.5) / N
            if isinstance(body, Polytope):
                g_ = X @ body.A.T + body.d[None]
                occ = np.all(g_ <= 0, axis=1); amb = np.flatnonzero(np.any(np.abs(g_) <= MARGIN / N, axis=1))
                for i in amb:
                    occ[i] = poly_contains_exact(body, X[i])
            else:
                g_ = ((X - body.c[None]) ** 2).sum(1) - body.r ** 2
                occ = g_ <= 0; amb = np.flatnonzero(np.abs(g_) <= MARGIN / N)
                for i in amb:
                    occ[i] = ball_contains_exact(body, X[i])
        st["exact_fallbacks"] = st.get("exact_fallbacks", 0) + len(amb)
        out.append(C[occ])
    return np.concatenate(out) if out else np.zeros((0, 3), np.int64)
