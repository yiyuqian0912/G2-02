"""Certified source regions from free response-interface edges (Proposition A', docs/LOCALFIRST_E123.md §2).

Proposition A'. S is a finite union of compact convex polytopes (dimension >= 1). (l, k) is a direct interface edge of
the observation X (l response-active, k response-inactive) whose closed segment misses S (H1, checked exactly). Then
for EVERY s' in F(G, X) there are p in [l, k] and an edge e of some body C with
  (i)   e meets [s', p];
  (ii)  [s', p] meets S but not its interior;
  (iii) the plane through p and e supports C (e is a silhouette edge of C as seen from p).
Hence F(G, X) is contained in Z(l, k) = union over admissible edges e of { s : e meets conv{s, l, k} }, and a source box
Q can meet F only if e meets conv(Q u [l, k]) for some admissible e (the union of the triangles conv{s, l, k}, s in Q,
is conv(Q u [l, k])). R_I = intersection over any set I of H1-free interface edges contains F: no exceptional set.

Everything that can exclude source volume is decided conservatively:
* admissibility (iii) is evaluated on float facet rows with a margin; an edge is dropped only if p is strictly in
  front of both adjacent facets, or strictly behind both, at both ends of [l, k] by more than the margin;
* a box is excluded for an interface edge only if EVERY admissible occluder edge is separated from conv(Q u [l, k]) by
  some axis with a gap larger than the margin (the axis need not be exact: any direction with a verified gap is a
  separation certificate; the candidate axis set is complete for nondegenerate configurations, so it is also tight);
* undecided tests keep the box; capacity limits stop refinement and keep the coarser box.
The world geometry is consulted only through the occluder edge list (vertices and adjacent facet planes) and the
exact free-edge test of H1. No visibility computation is involved."""
import numpy as np
from numba import njit

from ..rpl.topology import vertices, TOL

MARGIN = 1e-10          # world units (coordinates in [0, 1]); float errors of the predicates below are ~1e-15


# ---------------------------------------------------------------------------------------------------- occluder edges
def world_edges(P_raw, K):
    """All edges of all bodies: (A (n, 3), B (n, 3), F (n, 2, 4) unit adjacent facet rows, comp (n,)).
    An edge is recorded for every pair of facet planes sharing >= 2 distinct enumerated vertices (rpl.topology rule:
    true edges are never missed; spurious pairs are segments on the boundary and only add explanations)."""
    A, B, F, C = [], [], [], []
    P_raw = np.asarray(P_raw, float)
    for c in range(P_raw.shape[0]):
        rows = P_raw[c, :K[c]]
        un = rows / np.linalg.norm(rows[:, :3], axis=1)[:, None]
        V = vertices(rows)
        if not len(V):
            continue
        on = np.abs(V @ rows[:, :3].T + rows[:, 3]) <= TOL
        for i in range(K[c]):
            for j in range(i + 1, K[c]):
                pts = V[on[:, i] & on[:, j]]
                if len(pts) < 2:
                    continue
                d = pts - pts[0]; far = int(np.argmax(np.linalg.norm(d, axis=1)))
                if np.linalg.norm(d[far]) <= 0:
                    continue
                u = d[far] / np.linalg.norm(d[far]); t = d @ u
                A.append(pts[int(np.argmin(t))]); B.append(pts[int(np.argmax(t))]); F.append(np.stack([un[i], un[j]])); C.append(c)
    return (np.array(A, float).reshape(-1, 3), np.array(B, float).reshape(-1, 3), np.array(F, float).reshape(-1, 2, 4),
            np.array(C, np.int32))


def admissible_edges(Fr, Lp, Kp, margin=1e-9, chunk=4096):
    """CSR (off, idx): for each interface edge i, the occluder edges that can be silhouette edges as seen from some
    p in [l_i, k_i] (Proposition A' (iii)). Conservative: an edge is dropped only when both adjacent facet values have
    the same strict sign (beyond margin) at both l_i and k_i (the values are linear in p)."""
    nI = len(Lp); out_idx = []; off = np.zeros(nI + 1, np.int64)
    n1, d1 = Fr[:, 0, :3], Fr[:, 0, 3]; n2, d2 = Fr[:, 1, :3], Fr[:, 1, 3]
    for s in range(0, nI, chunk):
        l = Lp[s:s + chunk]; k = Kp[s:s + chunk]
        f1l = l @ n1.T + d1; f1k = k @ n1.T + d1; f2l = l @ n2.T + d2; f2k = k @ n2.T + d2
        front = (f1l > margin) & (f1k > margin) & (f2l > margin) & (f2k > margin)
        back = (f1l < -margin) & (f1k < -margin) & (f2l < -margin) & (f2k < -margin)
        keep = ~(front | back)
        for r in range(len(l)):
            ii = np.flatnonzero(keep[r]).astype(np.int32)
            out_idx.append(ii); off[s + r + 1] = off[s + r] + len(ii)
    return off, (np.concatenate(out_idx) if out_idx else np.zeros(0, np.int32)).astype(np.int32)


# ---------------------------------------------------------------------------------------------------- predicates
@njit(cache=True)
def _cross(u, v, out):
    out[0] = u[1] * v[2] - u[2] * v[1]; out[1] = u[2] * v[0] - u[0] * v[2]; out[2] = u[0] * v[1] - u[1] * v[0]


@njit(cache=True)
def _sep(u, segA, segB, H, nh, m):
    """True if direction u separates the segment from the points H[:nh] with a gap > m |u|_1."""
    n1 = abs(u[0]) + abs(u[1]) + abs(u[2])
    if n1 < 1e-300:
        return False
    a = u[0] * segA[0] + u[1] * segA[1] + u[2] * segA[2]
    b = u[0] * segB[0] + u[1] * segB[1] + u[2] * segB[2]
    smin = min(a, b); smax = max(a, b)
    hmin = 1e300; hmax = -1e300
    for j in range(nh):
        v = u[0] * H[j, 0] + u[1] * H[j, 1] + u[2] * H[j, 2]
        if v < hmin:
            hmin = v
        if v > hmax:
            hmax = v
    g = m * n1
    return smax < hmin - g or smin > hmax + g


@njit(cache=True)
def seg_misses_hull(a, b, l, k, lo, hi, m):
    """True only if the segment [a, b] certainly misses conv(box [lo, hi] u {l, k})."""
    # axis-aligned bounding boxes first
    for t in range(3):
        hl = min(lo[t], l[t], k[t]); hh = max(hi[t], l[t], k[t])
        if max(a[t], b[t]) < hl - m or min(a[t], b[t]) > hh + m:
            return True
    H = np.empty((10, 3)); D = np.empty((20, 3)); u = np.empty(3); e = np.empty(3)
    n = 0
    for c in range(8):
        H[n, 0] = hi[0] if c & 1 else lo[0]; H[n, 1] = hi[1] if c & 2 else lo[1]; H[n, 2] = hi[2] if c & 4 else lo[2]; n += 1
    # wedge prefilter (sound): if e meets conv{s, l, k} then the plane through line(e) and s weakly separates l and k,
    # i.e. g_l(s) g_k(s) <= 0 with g_x(s) = (s - a) . ((x - a) x (b - a)), linear in s. The box misses the wedge if
    # g_l and g_k have the same strict sign (beyond the margin) at all 8 corners.
    ba = np.empty(3); la = np.empty(3); ka = np.empty(3); nl = np.empty(3); nk = np.empty(3)
    for t in range(3):
        ba[t] = b[t] - a[t]; la[t] = l[t] - a[t]; ka[t] = k[t] - a[t]
    _cross(la, ba, nl); _cross(ka, ba, nk)
    gl_min = 1e300; gl_max = -1e300; gk_min = 1e300; gk_max = -1e300
    for c in range(8):
        vl = nl[0] * (H[c, 0] - a[0]) + nl[1] * (H[c, 1] - a[1]) + nl[2] * (H[c, 2] - a[2])
        vk = nk[0] * (H[c, 0] - a[0]) + nk[1] * (H[c, 1] - a[1]) + nk[2] * (H[c, 2] - a[2])
        gl_min = min(gl_min, vl); gl_max = max(gl_max, vl); gk_min = min(gk_min, vk); gk_max = max(gk_max, vk)
    gm_l = m * (abs(nl[0]) + abs(nl[1]) + abs(nl[2])); gm_k = m * (abs(nk[0]) + abs(nk[1]) + abs(nk[2]))
    if (gl_min > gm_l and gk_min > gm_k) or (gl_max < -gm_l and gk_max < -gm_k):
        return True
    for t in range(3):
        H[8, t] = l[t]; H[9, t] = k[t]
    # hull edge directions: axes, k - l, corner - l, corner - k
    nd = 0
    for t in range(3):
        D[nd, 0] = 0.0; D[nd, 1] = 0.0; D[nd, 2] = 0.0; D[nd, t] = 1.0; nd += 1
    for t in range(3):
        D[nd, t] = k[t] - l[t]
    nd += 1
    for c in range(8):
        for t in range(3):
            D[nd, t] = H[c, t] - l[t]
        nd += 1
    for c in range(8):
        for t in range(3):
            D[nd, t] = H[c, t] - k[t]
        nd += 1
    for t in range(3):
        e[t] = b[t] - a[t]
    # (1) box face normals
    for t in range(3):
        if _sep(D[t], a, b, H, 10, m):
            return True
    # (2) faces through l or k and a box edge; (3) faces through [l, k] and a corner; (4) [l, k] with a box axis
    for t in range(3):
        for j in range(4, 20):
            _cross(D[t], D[j], u)
            if _sep(u, a, b, H, 10, m):
                return True
        _cross(D[3], D[t], u)
        if _sep(u, a, b, H, 10, m):
            return True
    for j in range(4, 12):
        _cross(D[3], D[j], u)
        if _sep(u, a, b, H, 10, m):
            return True
    # (5) the occluder edge direction with every hull edge direction
    for j in range(20):
        _cross(e, D[j], u)
        if _sep(u, a, b, H, 10, m):
            return True
    return False


@njit(cache=True)
def seg_misses_triangle(a, b, s, l, k, m):
    """True only if the segment [a, b] certainly misses the triangle conv{s, l, k} (complete SAT: triangle normal and
    the edge direction crossed with the three triangle edges)."""
    T = np.empty((3, 3)); u = np.empty(3); e = np.empty(3); d1 = np.empty(3); d2 = np.empty(3); d3 = np.empty(3)
    for t in range(3):
        T[0, t] = s[t]; T[1, t] = l[t]; T[2, t] = k[t]
        e[t] = b[t] - a[t]; d1[t] = l[t] - s[t]; d2[t] = k[t] - s[t]; d3[t] = k[t] - l[t]
    _cross(d1, d2, u)
    if _sep(u, a, b, T, 3, m):
        return True
    _cross(e, d1, u)
    if _sep(u, a, b, T, 3, m):
        return True
    _cross(e, d2, u)
    if _sep(u, a, b, T, 3, m):
        return True
    _cross(e, d3, u)
    if _sep(u, a, b, T, 3, m):
        return True
    return False


@njit(cache=True)
def in_region(s, EA, EB, Ll, Lk, off, idx, m):
    """Liberal membership of a point in R_I (measurement only): False only if some interface edge has every admissible
    occluder edge certainly separated from conv{s, l, k}."""
    for i in range(Ll.shape[0]):
        found = False
        for t in range(off[i], off[i + 1]):
            e = idx[t]
            if not seg_misses_triangle(EA[e], EB[e], s, Ll[i], Lk[i], m):
                found = True
                break
        if not found:
            return False
    return True


@njit(cache=True)
def in_region_batch(S, EA, EB, Ll, Lk, off, idx, m):
    """in_region for many points. Same decisions; per interface edge, the candidate that explained the previous point
    is tried first (nearby points share explanations), which only changes the order of the existential search."""
    out = np.zeros(S.shape[0], np.bool_)
    nI = Ll.shape[0]
    last = np.full(nI, -1, np.int64)
    for j in range(S.shape[0]):
        s = S[j]; ok = True
        for i in range(nI):
            found = False
            t0 = last[i]
            if t0 >= 0:
                e = idx[t0]
                if not seg_misses_triangle(EA[e], EB[e], s, Ll[i], Lk[i], m):
                    found = True
            if not found:
                for t in range(off[i], off[i + 1]):
                    if t == t0:
                        continue
                    e = idx[t]
                    if not seg_misses_triangle(EA[e], EB[e], s, Ll[i], Lk[i], m):
                        found = True; last[i] = t
                        break
            if not found:
                ok = False
                break
        out[j] = ok
    return out


# ---------------------------------------------------------------------------------------------------- octree
@njit(cache=True)
def region_octree(EA, EB, Ll, Lk, off, idx, roots_lo, roots_dep, D, wlo, whi, m, leaf_cap, out_lo, out_dep, stats,
                  max_tests, pyr, pyr_off, use_pyr, want_expl, expl_out):
    """Certified cover of R_I: boxes (lo, depth) of side 2^-depth, refined to depth D. A box is dropped only if it lies
    inside the closed window box (sources in B(W) are excluded by definition) or some interface edge has no admissible
    occluder edge meeting conv(box u [l, k]). Candidate lists shrink monotonically down the tree.
    stats: [boxes, excluded_window, excluded_edge, tests, leaves, capacity_stops, budget_stops, skipped_untargeted].
    Budget: once `max_tests` box tests are spent, boxes are emitted as leaves without further filtering (kept: sound).
    Targets (use_pyr): `pyr` is an OR-pyramid of a depth-7 cell mask (levels 0..7, level d at pyr_off[d], index
    (i*2^d + j)*2^d + k); boxes containing no target cell are skipped (not reported, not excluded).
    Returns the number of leaves."""
    nI = Ll.shape[0]
    R0 = off[nI]
    pool = np.empty((D + 3) * max(R0, 1), np.int32)
    offp = np.empty((D + 3) * (nI + 1), np.int64)
    for t in range(R0):
        pool[t] = idx[t]
    for i in range(nI + 1):
        offp[i] = off[i]
    SMAX = 8 * (D + 2) + 8
    st_lo = np.empty((SMAX, 3)); st_dep = np.empty(SMAX, np.int64)
    st_pb = np.empty(SMAX, np.int64); st_ob = np.empty(SMAX, np.int64); st_pe = np.empty(SMAX, np.int64); st_oe = np.empty(SMAX, np.int64)
    lo = np.empty(3); hi = np.empty(3)
    nleaf = 0
    for r in range(roots_lo.shape[0]):
        sp = 0
        st_lo[0, 0] = roots_lo[r, 0]; st_lo[0, 1] = roots_lo[r, 1]; st_lo[0, 2] = roots_lo[r, 2]; st_dep[0] = roots_dep[r]
        st_pb[0] = 0; st_ob[0] = 0; st_pe[0] = R0; st_oe[0] = nI + 1
        sp = 1
        while sp > 0:
            sp -= 1
            dep = st_dep[sp]; pb = st_pb[sp]; ob = st_ob[sp]; ptop = st_pe[sp]; otop = st_oe[sp]
            h = 1.0 / (1 << dep)
            for t in range(3):
                lo[t] = st_lo[sp, t]; hi[t] = lo[t] + h
            if use_pyr:
                dd = min(dep, 7); n_ = 1 << dd
                ii = int(lo[0] * n_ + 1e-9); jj = int(lo[1] * n_ + 1e-9); kk = int(lo[2] * n_ + 1e-9)
                if not pyr[pyr_off[dd] + (ii * n_ + jj) * n_ + kk]:
                    stats[7] += 1
                    continue
            stats[0] += 1
            if stats[3] >= max_tests:
                stats[6] += 1
                if nleaf < leaf_cap:
                    out_lo[nleaf, 0] = lo[0]; out_lo[nleaf, 1] = lo[1]; out_lo[nleaf, 2] = lo[2]; out_dep[nleaf] = dep
                    nleaf += 1
                stats[4] += 1
                continue
            inside = True
            for t in range(3):
                if lo[t] < wlo[t] - 1e-12 or hi[t] > whi[t] + 1e-12:
                    inside = False
            if inside:
                stats[1] += 1
                continue
            npb = ptop; nob = otop; cnt = 0; ok = True
            for i in range(nI):
                offp[nob + i] = cnt
                l = Ll[i]; k = Lk[i]
                for t in range(offp[ob + i], offp[ob + i + 1]):
                    e = pool[pb + t]
                    stats[3] += 1
                    if not seg_misses_hull(EA[e], EB[e], l, k, lo, hi, m):
                        pool[npb + cnt] = e
                        cnt += 1
                if cnt == offp[nob + i]:
                    ok = False
                    break
            if not ok:
                stats[2] += 1
                continue
            offp[nob + nI] = cnt
            if dep >= D or nleaf + 8 * (sp + 8) >= leaf_cap:
                if dep < D:
                    stats[5] += 1
                if nleaf < leaf_cap:
                    out_lo[nleaf, 0] = lo[0]; out_lo[nleaf, 1] = lo[1]; out_lo[nleaf, 2] = lo[2]; out_dep[nleaf] = dep
                    if want_expl:
                        # explanation summary of this leaf: the unique admissible edge, or -2 for several
                        for i in range(nI):
                            c0 = offp[nob + i]; c1 = offp[nob + i + 1]
                            expl_out[nleaf, i] = pool[npb + c0] if c1 - c0 == 1 else -2
                    nleaf += 1
                stats[4] += 1
                continue
            h2 = h * 0.5
            for c8 in range(8):
                st_lo[sp, 0] = lo[0] + (h2 if c8 & 1 else 0.0)
                st_lo[sp, 1] = lo[1] + (h2 if c8 & 2 else 0.0)
                st_lo[sp, 2] = lo[2] + (h2 if c8 & 4 else 0.0)
                st_dep[sp] = dep + 1; st_pb[sp] = npb; st_ob[sp] = nob; st_pe[sp] = npb + cnt; st_oe[sp] = nob + nI + 1
                sp += 1
    return nleaf


def _pyramid(cells, G=128):
    m = np.zeros((G, G, G), bool)
    if len(cells):
        m.reshape(-1)[cells] = True
    levels = [m]
    while levels[-1].shape[0] > 1:
        a = levels[-1]; n = a.shape[0] // 2
        levels.append(a.reshape(n, 2, n, 2, n, 2).any(axis=(1, 3, 5)))
    levels = levels[::-1]                                         # level d has side 2^d
    off = np.cumsum([0] + [x.size for x in levels]).astype(np.int64)
    return np.concatenate([x.reshape(-1) for x in levels]), off


def build_region(EA, EB, Lp, Kp, off, idx, window_lo, window_hi, D=7, roots_lo=None, roots_dep=None, leaf_cap=4_000_000,
                 m=MARGIN, target_cells=None, max_tests=1 << 62, want_expl=False, target_boxes=None):
    """Python wrapper: returns dict(lo (n, 3), dep (n,), stats). target_cells: refine only boxes containing one of these
    128^3 cells (shared filtering along the paths); max_tests: box-test budget (beyond it boxes are kept)."""
    if roots_lo is None:
        roots_lo = np.zeros((1, 3)); roots_dep = np.zeros(1, np.int64)
    out_lo = np.empty((leaf_cap, 3)); out_dep = np.empty(leaf_cap, np.int64); stats = np.zeros(8, np.int64)
    if target_boxes is not None:
        # target octree boxes (depth, i, j, k) with depth <= 7: mark all their 128^3 cells
        cl = []
        for d, i, j, k in np.asarray(target_boxes, np.int64):
            n = 1 << (7 - d); g = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).reshape(-1, 3)
            ijk = np.array([i, j, k]) * n + g; cl.append((ijk[:, 0] * 128 + ijk[:, 1]) * 128 + ijk[:, 2])
        target_cells = np.unique(np.concatenate(cl)) if cl else np.zeros(0, np.int64)
    if target_cells is not None:
        pyr, poff = _pyramid(np.asarray(target_cells, np.int64)); use = True
    else:
        pyr, poff = np.zeros(1, np.bool_), np.zeros(8, np.int64); use = False
    expl = np.empty((leaf_cap if want_expl else 1, max(Lp.shape[0], 1)), np.int64)
    n = region_octree(EA, EB, np.ascontiguousarray(Lp, float), np.ascontiguousarray(Kp, float), np.asarray(off, np.int64),
                      np.asarray(idx, np.int32), np.ascontiguousarray(roots_lo, float), np.asarray(roots_dep, np.int64), int(D),
                      np.asarray(window_lo, float), np.asarray(window_hi, float), float(m), int(leaf_cap), out_lo, out_dep, stats,
                      np.int64(min(max_tests, 1 << 62)), pyr, poff, use, bool(want_expl), expl)
    names = ["boxes", "excluded_window", "excluded_edge", "tests", "leaves", "capacity_stops", "budget_stops", "skipped_untargeted"]
    out = dict(lo=out_lo[:n].copy(), dep=out_dep[:n].copy(), stats=dict(zip(names, stats.tolist())))
    if want_expl:
        out["expl"] = expl[:n].copy()
    return out


def explanations_for_roots(EA, EB, Lp, Kp, off, idx, roots, window_lo, window_hi, m=MARGIN):
    """Explanation summaries for stage-1 roots (rows (depth, i, j, k), all of one depth) through ONE targeted octree
    descent (candidate filtering shared along the paths). Returns (expl (n_roots, nI): edge / -2 several / -1 none,
    seconds). A root absent from the leaves was excluded (some interface edge has no explanation): all -1."""
    import time
    t = time.perf_counter(); roots = np.asarray(roots, np.int64); d = int(roots[0, 0])
    r = build_region(EA, EB, Lp, Kp, off, idx, window_lo, window_hi, D=d, target_boxes=roots, want_expl=True,
                     leaf_cap=max(4 * len(roots), 1024), m=m)
    h = 1.0 / (1 << d); key = {tuple(np.floor(lo / h + 1e-9).astype(np.int64)): q for q, lo in enumerate(r["lo"]) if r["dep"][q] == d}
    expl = np.full((len(roots), len(Lp)), -1, np.int64)
    for q, (_, i, j, k) in enumerate(roots):
        z = key.get((i, j, k))
        if z is not None:
            expl[q] = r["expl"][z]
    # a leaf coarser than d (capacity stop) is not a root summary: such roots keep -2 (no clip, not dropped)
    for q, (_, i, j, k) in enumerate(roots):
        if (i, j, k) not in key:
            lo = np.array([i, j, k]) * h
            for z in np.flatnonzero(r["dep"] < d):
                hz = 1.0 / (1 << r["dep"][z])
                if np.all(lo >= r["lo"][z] - 1e-12) and np.all(lo + h <= r["lo"][z] + hz + 1e-12):
                    expl[q] = -2; break
    return expl, time.perf_counter() - t


def leaf_cells(lo, dep, G=128):
    """128^3 output cells met by the leaves (cell id (ix*G+iy)*G+iz); leaves coarser than a cell cover all its cells."""
    cells = []
    for d in np.unique(dep):
        L = lo[dep == d]
        if d >= 7:
            ijk = np.floor(L * G + 1e-9).astype(np.int64)
            cells.append((ijk[:, 0] * G + ijk[:, 1]) * G + ijk[:, 2])
        else:
            n = 1 << (7 - d)
            g = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).reshape(-1, 3)
            base = np.floor(L * G + 1e-9).astype(np.int64)
            ijk = (base[:, None, :] + g[None]).reshape(-1, 3)
            cells.append((ijk[:, 0] * G + ijk[:, 1]) * G + ijk[:, 2])
    return np.unique(np.concatenate(cells)) if cells else np.zeros(0, np.int64)


# ---------------------------------------------------------------------------------------------------- root clips (E2)
@njit(cache=True)
def box_explanations(EA, EB, Ll, Lk, off, idx, boxes_lo, boxes_h, m):
    """For every box and interface edge: -1 if no admissible occluder edge meets conv(box u [l, k]) (the box carries
    no feasible source), the edge index if exactly one does, -2 if several do."""
    nb = boxes_lo.shape[0]; nI = Ll.shape[0]
    out = np.empty((nb, nI), np.int64)
    lo = np.empty(3); hi = np.empty(3)
    for q in range(nb):
        for t in range(3):
            lo[t] = boxes_lo[q, t]; hi[t] = lo[t] + boxes_h[q]
        for i in range(nI):
            found = -1
            for u in range(off[i], off[i + 1]):
                e = idx[u]
                if not seg_misses_hull(EA[e], EB[e], Ll[i], Lk[i], lo, hi, m):
                    if found == -1:
                        found = e
                    else:
                        found = -2
                        break
            out[q, i] = found
    return out


def wedge_rows(a, b, l, k, m=MARGIN):
    """The two certified half-spaces (rows (n, d), keep n.x + d <= 0) of the wedge through line(e = [a, b]) that contains
    the sliver {s : e meets conv{s, l, k}}: with n_x = (x - a) x (b - a), g_x(s) = n_x . (s - a) and G = g_l(k), every
    s = c + t (c - p) (c in e, p = l + mu (k - l), t >= 0) has g_l(s) = -t mu G and g_k(s) = t (1 - mu) G, hence
    sign(G) g_l(s) <= 0 <= sign(G) g_k(s). Offsets are relaxed outward by m |n|_1. None if G = 0 (degenerate)."""
    nl = np.cross(l - a, b - a); nk = np.cross(k - a, b - a); G = float((k - a) @ nl)
    if abs(G) <= 1e-300 or abs(G) < 1e-14 * np.linalg.norm(nl) * np.linalg.norm(k - l):
        return None
    sg = 1.0 if G > 0 else -1.0
    r1 = np.r_[sg * nl, -sg * (nl @ a) - m * np.abs(nl).sum()]
    r2 = np.r_[-sg * nk, sg * (nk @ a) - m * np.abs(nk).sum()]
    return r1, r2


def root_clips(EA, EB, Lp, Kp, expl, max_edges=12):
    """Per box: certified clip rows from the interface edges with a unique explanation on that box. For each explaining
    occluder edge keep the two extreme planes of its pencil (any subset of valid half-spaces is valid). Returns
    (off (nb+1,), rows (n, 4), empty (nb,) bool: some interface edge has no explanation on the box)."""
    nb = expl.shape[0]; rows = []; off = [0]; empty = np.zeros(nb, bool)
    for q in range(nb):
        r_q = []
        if np.any(expl[q] == -1):
            empty[q] = True
        else:
            u = expl[q]; es, cnt = np.unique(u[u >= 0], return_counts=True)
            for e in es[np.argsort(-cnt)][:max_edges]:
                a, b = EA[e], EB[e]; d = (b - a) / np.linalg.norm(b - a)
                w1 = np.cross(d, [1.0, 0, 0] if abs(d[0]) < 0.9 else [0, 1.0, 0]); w1 /= np.linalg.norm(w1); w2 = np.cross(d, w1)
                cand = []
                for i in np.flatnonzero(u == e):
                    wr = wedge_rows(a, b, Lp[i], Kp[i])
                    if wr is not None:
                        cand += list(wr)
                if not cand:
                    continue
                N_ = np.array([c[:3] for c in cand]); ang = np.arctan2(N_ @ w2, N_ @ w1)
                ref = np.angle(np.mean(np.exp(1j * ang))); rel = np.angle(np.exp(1j * (ang - ref)))
                for j in {int(np.argmin(rel)), int(np.argmax(rel))}:
                    r_q.append(cand[j])
        rows += r_q; off.append(off[-1] + len(r_q))
    return np.array(off, np.int64), np.array(rows, float).reshape(-1, 4), empty
