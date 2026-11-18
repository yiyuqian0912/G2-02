"""Verification of Local-first root restrictions (Proposition A', docs/LOCALFIRST_E123.md section 2), independent of
the labeller.

Proposition A'. Let the solid be a finite union of compact convex polytopes, (l, k) a direct response interface edge
(6-adjacent FREE receivers, l response-active, k response-inactive) whose closed segment misses the solid (H1). Then
every feasible source s has an occluder edge e (of some body, admissible: its two facets do not face p the same way
for some p in [l, k]) that meets conv{s, l, k}.
Consequences used here, for a root box Q of the stage-1 octree:
* DROP: if no admissible edge meets conv(Q u [l, k]) (the union of the triangles conv{s, l, k}, s in Q), then
  F n Q = {} and Q is OUT.
* CLIP: if exactly one edge e = facets (i, j) of body c can meet conv(Q u [l, k]), every feasible s in Q lies in the
  sliver {s : e meets conv{s, l, k}}, on which (wedge lemma) the pencil function through line(e) and p in {l, k},
      g_p(s) = f_i(p) f_j(s) - f_j(p) f_i(s),
  satisfies g_p(s) = -t mu g_p(q) (q the other endpoint, s = c + t (c - p'), c in e, p' in [l, k], t >= 0). So
  F n Q is contained in {sign(g_p(q)) g_p(s) <= 0}; the verifier keeps the OUTER binary64 plane of that half-space
  (shifted by its rigorous coefficient error), a superset.
Every predicate is decided rigorously: face values at l, k in double-double (check.face_at_point), the separations
by explicit axes with error-bounded projections of exact points (box corners, receivers) and enclosed edge endpoints.
An undecided test rejects the proof (the clip is not applied, the drop is not accepted)."""
import numpy as np
from numba import njit

from .ar import U, INFL, TINY
from .check import face_at_point
from .tree import _pencil_coef, _shift_planes

MAXK = 64


@njit(cache=True)
def _proj(u0, u1, u2, p0, p1, p2, r):
    """(lo, hi) enclosure of u . p for p enclosed within +-r per coordinate (u binary64 exact)."""
    m = u0 * p0 + u1 * p1 + u2 * p2
    e = (abs(u0 * p0) + abs(u1 * p1) + abs(u2 * p2)) * 4.0 * U * INFL + (abs(u0) + abs(u1) + abs(u2)) * r * INFL + TINY
    return m - e - 2.0 * U * abs(m), m + e + 2.0 * U * abs(m)


@njit(cache=True)
def _sep_axis(u, A, B, er, pts):
    """True if axis u strictly separates segment [A, B] (endpoints within +-er) from the exact points pts."""
    a1, a2 = _proj(u[0], u[1], u[2], A[0], A[1], A[2], er)
    b1, b2 = _proj(u[0], u[1], u[2], B[0], B[1], B[2], er)
    slo = min(a1, b1); shi = max(a2, b2)
    plo = np.inf; phi = -np.inf
    for t in range(pts.shape[0]):
        q1, q2 = _proj(u[0], u[1], u[2], pts[t, 0], pts[t, 1], pts[t, 2], 0.0)
        plo = min(plo, q1); phi = max(phi, q2)
    return shi < plo or slo > phi


@njit(cache=True)
def seg_misses_hull(A, B, er, pts):
    """Rigorous sufficient test that segment [A, B] misses conv(pts): an axis from a fixed candidate set separates.
    Candidates: coordinate axes; d x e_k; d x (p_m - p_n) for consecutive hull points; d = B - A."""
    u = np.empty(3)
    lo = np.empty(3); hi = np.empty(3)
    for k in range(3):
        lo[k] = pts[:, k].min(); hi[k] = pts[:, k].max()
        if max(A[k], B[k]) + er < lo[k] or min(A[k], B[k]) - er > hi[k]:
            return True
    d = np.empty(3)
    for k in range(3):
        d[k] = B[k] - A[k]
    n = pts.shape[0]
    for k in range(3):
        w = np.zeros(3); w[k] = 1.0
        u[0] = d[1] * w[2] - d[2] * w[1]; u[1] = d[2] * w[0] - d[0] * w[2]; u[2] = d[0] * w[1] - d[1] * w[0]
        if abs(u[0]) + abs(u[1]) + abs(u[2]) > 0.0 and _sep_axis(u, A, B, er, pts):
            return True
    for m in range(n):
        for q in range(m + 1, n):
            w0 = pts[q, 0] - pts[m, 0]; w1 = pts[q, 1] - pts[m, 1]; w2 = pts[q, 2] - pts[m, 2]
            u[0] = d[1] * w2 - d[2] * w1; u[1] = d[2] * w0 - d[0] * w2; u[2] = d[0] * w1 - d[1] * w0
            if abs(u[0]) + abs(u[1]) + abs(u[2]) > 0.0 and _sep_axis(u, A, B, er, pts):
                return True
            # face normals of triangles (A, p_m, p_q) are also separating candidates for the hull side
            v0 = pts[m, 0] - A[0]; v1 = pts[m, 1] - A[1]; v2 = pts[m, 2] - A[2]
            u[0] = v1 * w2 - v2 * w1; u[1] = v2 * w0 - v0 * w2; u[2] = v0 * w1 - v1 * w0
            if abs(u[0]) + abs(u[1]) + abs(u[2]) > 0.0 and _sep_axis(u, A, B, er, pts):
                return True
    return False


@njit(cache=True)
def seg_free(l, k, C, K, RM, RL, RR2, alo, ahi):
    """H1: the closed segment [l, k] misses every component (a facet with both endpoints strictly outside, or AABB)."""
    for c in range(C):
        sep = False
        for t in range(3):
            if max(l[t], k[t]) < alo[c, t] or min(l[t], k[t]) > ahi[c, t]:
                sep = True
        if sep:
            continue
        for i in range(K[c]):
            m1, r1 = face_at_point(c, i, l[0], l[1], l[2], RM, RL, RR2)
            m2, r2 = face_at_point(c, i, k[0], k[1], k[2], RM, RL, RR2)
            if m1 - r1 > 0.0 and m2 - r2 > 0.0:
                sep = True
                break
        if not sep:
            return False
    return True


@njit(cache=True)
def explaining_edges(qlo, qhi, l, k, EC, EI, EJ, EA, EB, ER, nE, RM, RL, RR2, out):
    """Edges that are admissible for [l, k] and NOT proved to miss conv(Q u [l, k]); written to out, count returned
    (stops at len(out) + 1 -> returns len(out) + 1 meaning 'too many')."""
    pts = np.empty((10, 3))
    for cr in range(8):
        pts[cr, 0] = qhi[0] if cr & 1 else qlo[0]
        pts[cr, 1] = qhi[1] if cr & 2 else qlo[1]
        pts[cr, 2] = qhi[2] if cr & 4 else qlo[2]
    for t in range(3):
        pts[8, t] = l[t]; pts[9, t] = k[t]
    n = 0
    for e in range(nE):
        c = EC[e]; i = EI[e]; j = EJ[e]
        # admissibility (Prop A' (iii)): drop only if both facets have the same strict sign at l and at k
        a1, ar1 = face_at_point(c, i, l[0], l[1], l[2], RM, RL, RR2)
        a2, ar2 = face_at_point(c, i, k[0], k[1], k[2], RM, RL, RR2)
        b1, br1 = face_at_point(c, j, l[0], l[1], l[2], RM, RL, RR2)
        b2, br2 = face_at_point(c, j, k[0], k[1], k[2], RM, RL, RR2)
        front = a1 - ar1 > 0.0 and a2 - ar2 > 0.0 and b1 - br1 > 0.0 and b2 - br2 > 0.0
        back = a1 + ar1 < 0.0 and a2 + ar2 < 0.0 and b1 + br1 < 0.0 and b2 + br2 < 0.0
        if front or back:
            continue
        if seg_misses_hull(EA[e], EB[e], ER[e], pts):
            continue
        if n < out.shape[0]:
            out[n] = e
        n += 1
        if n > out.shape[0]:
            return n
    return n


def edge_index(scene):
    return {(int(c), int(i), int(j)): t for t, (c, i, j, A, B) in enumerate(scene.edges)}


def verify_root_plan(scene, cls, o, N, root, plan, PLn):
    """Verify the Local-first plan of one root. plan: dict(drop=(l, k) or None, clips=[(c, i, j, l, k, which), ...])
    with l, k flat window voxel indices. Returns (dropped_ok, [signed plane refs or -1 per clip])."""
    L = cls.shape[0]; flat = cls.reshape(-1)
    d, ix, iy, iz = (int(v) for v in root)
    h = 1.0 / (1 << d)
    qlo = np.array([ix * h, iy * h, iz * h]); qhi = qlo + h
    eidx = edge_index(scene)
    out = np.zeros(4, np.int64)

    def rx(xf):
        v = np.array([xf // (L * L), (xf // L) % L, xf % L]); return (o + v + 0.5) / N

    def pair_ok(l, k):
        lv = np.unravel_index(l, (L, L, L)); kv = np.unravel_index(k, (L, L, L))
        if flat[l] != 1 or flat[k] != 0 or sum(abs(a - b) for a, b in zip(lv, kv)) != 1:
            return False
        return seg_free(rx(l), rx(k), scene.C, scene.K, scene.RM, scene.RL, scene.RR2, scene.aabb_lo, scene.aabb_hi)

    def expl(l, k):
        n = explaining_edges(qlo, qhi, rx(l), rx(k), scene.E_c, scene.E_i, scene.E_j, scene.E_A, scene.E_B, scene.E_r,
                             int(scene.nE), scene.RM, scene.RL, scene.RR2, out)
        return n, out[:min(n, len(out))].tolist()

    if plan.get("drop") is not None:
        l, k = plan["drop"]
        if not pair_ok(l, k):
            return False, []
        n, _ = expl(l, k)
        return n == 0, []
    refs = []
    gm = np.zeros(4); gr = np.zeros(4)
    for (c, i, j, l, k, which) in plan.get("clips", []):
        ref = -1
        e = eidx.get((int(c), int(min(i, j)), int(max(i, j))))
        if e is not None and pair_ok(l, k):
            n, lst = expl(l, k)
            if n == 1 and lst[0] == e or n == 0:
                p = rx(l if which == 0 else k); q = rx(k if which == 0 else l)
                _pencil_coef(int(c), int(i), int(j), p[0], p[1], p[2], scene.RM, scene.RR, scene.RL, scene.RR2, gm, gr)
                gq = gm[0] * q[0] + gm[1] * q[1] + gm[2] * q[2] + gm[3]
                eq = abs(gm[0] * q[0]) + abs(gm[1] * q[1]) + abs(gm[2] * q[2]) + abs(gm[3])
                rq = (gr[0] + gr[1] + gr[2] + gr[3]) * INFL + 8 * U * eq + TINY
                if abs(gq) > rq:
                    sg = 1.0 if gq > 0 else -1.0
                    rA, rB = _shift_planes(PLn.PL, PLn.npl, gm, gr, sg, 0.0)
                    if rB >= 0:
                        ref = int(rB ^ 1)         # outer kept side {sg g~ - E <= 0}, superset of {sg g_p <= 0}
        refs.append(ref)
    return True, refs
