"""Independent verifier of certified posterior labels (driver).

verify_certificate(scene, cls, origin, N, cert) -> report with rigorous per-cell volume enclosures [lo_c, hi_c] of
    v_c = Vol(F(G, X) n c),   F(G, X) = {s in D(G) : R_G(s) = X on every FREE receiver of the window},
a pooled upper volume `pool` of thin undetermined slabs, and the certified TV bound of a stored label (tvb.py).

Inputs it trusts: the canonical geometry decoder (rind3d.rendering.polyhedra.decode_convex), the stored observation
bits, IEEE-754 binary64 arithmetic (checked by ar.self_test), and this package. Inputs it does NOT trust: anything
the labeller computed. The certificate (`cert`) is only a proof tree: stage-1 octree records, root order, root
clip recipes and the event stream; every claim in it is re-proved here, and a claim that cannot be proved only turns
certified volume into undetermined volume (it can never create certified volume)."""
import time
from fractions import Fraction as Q

import numpy as np

from . import ar, poly, tree
from .tree import NSTATS
from .scene import ExactScene

_SELF_TESTED = False
STAT_NAMES = ["nodes", "in_ok", "in_fail", "out_ok", "out_fail", "slabs", "undet", "empty_exact", "empty_sliver",
              "overflow", "evals", "self_out", "clip_out_ok", "clip_out_fail", "arena", "pieces", "shell", "planes", "apex",
              "apex_fail"]


# ------------------------------------------------------------------------------------------------ window pyramid
def pure_pyramid(cls):
    """Codes per level (0 DARK, 1 LIT, 2 SOLID, 3 MIXED) and the maximal pure non-SOLID blocks (top items), with
    the engines' block encoding (level << 48) | (bx << 32) | (by << 16) | bz. Independent implementation."""
    L = cls.shape[0]; q = int(round(np.log2(L)))
    if (1 << q) != L or cls.shape != (L, L, L):
        raise ValueError("window side must be a power of two")
    levels = []
    d = cls == 0; li = cls == 1; so = cls == 2
    for l in range(q + 1):
        code = np.full(d.shape, 3, np.uint8); code[d] = 0; code[li] = 1; code[so] = 2
        levels.append(code)
        if l < q:
            n = d.shape[0] // 2
            d = d.reshape(n, 2, n, 2, n, 2).all(axis=(1, 3, 5)); li = li.reshape(n, 2, n, 2, n, 2).all(axis=(1, 3, 5))
            so = so.reshape(n, 2, n, 2, n, 2).all(axis=(1, 3, 5))
    out_c, out_k = [], []
    stack = [(q, 0, 0, 0)]
    while stack:
        l, bx, by, bz = stack.pop()
        k = int(levels[l][bx, by, bz])
        if k == 2:
            continue
        if k == 3:
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        stack.append((l - 1, 2 * bx + dx, 2 * by + dy, 2 * bz + dz))
            continue
        out_c.append((l << 48) | (bx << 32) | (by << 16) | bz); out_k.append(k)
    return np.array(out_c, np.int64), np.array(out_k, np.int8)


# ------------------------------------------------------------------------------------------------ plane table
class Planes:
    def __init__(self, cap):
        self.PL = np.zeros((cap, 4)); self.npl = np.zeros(1, np.int64); self.axis = {}

    def axis_plane(self, k, v):
        key = (int(k), float(v))
        if key not in self.axis:
            n = int(self.npl[0]); e = [0.0, 0.0, 0.0]; e[k] = 1.0
            self.PL[n] = (*e, -float(v))
            assert Q(-self.PL[n, 3]) == Q(float(v))
            self.npl[0] = n + 1; self.axis[key] = n
        return self.axis[key]


class Slots:
    def __init__(self, S):
        d = poly.new_slots(S)
        self.a = (d["Fp"], d["Fn"], d["Fv"], d["Fe"], d["Vp"], d["Vm"], d["Vr"], d["nV"], d["nF"])


def _octree_partition_ok(recs):
    """Terminal stage-1 boxes (depth, ix, iy, iz) tile the unit cube: exact total volume 1 and no box inside another."""
    keys = {(int(d), int(i), int(j), int(k)) for d, i, j, k in recs}
    if len(keys) != len(recs):
        return False
    if sum(Q(1, 8 ** d) for d, _, _, _ in keys) != 1:
        return False
    for d, i, j, k in keys:
        for e in range(d):
            sh = d - e
            if (e, i >> sh, j >> sh, k >> sh) in keys:
                return False
    return True


def verify_certificate(scene: ExactScene, cls, origin, N, cert, shell_eps=2.0 ** -26, arena=1 << 23, plane_cap=1 << 22,
                       max_records=1 << 22, max_depth=420):
    """Verify one certificate (see _verify_once); buffers that overflow are enlarged and the whole verification is
    repeated (it is deterministic, so the result is the one an unbounded buffer would give)."""
    while True:
        try:
            return _verify_once(scene, cls, origin, N, cert, shell_eps, arena, plane_cap, max_records, max_depth)
        except _Overflow as ex:
            if ex.what == "arena":
                arena *= 4
            elif ex.what == "records":
                max_records *= 4
            else:
                plane_cap *= 4
            if arena > 1 << 28 or max_records > 1 << 28 or plane_cap > 1 << 27:
                raise RuntimeError(f"verifier buffers exhausted ({ex.what})")


class _Overflow(Exception):
    def __init__(self, what):
        super().__init__(what); self.what = what


def _verify_once(scene, cls, origin, N, cert, shell_eps, arena, plane_cap, max_records, max_depth):
    """Verify one certificate. cert keys: s1 (m, 6) stage-1 terminal boxes (depth, ix, iy, iz, kind, witness flat
    voxel), ev (n, 8) int64, evf (n,) float64, root_order (r, 4) roots in stream order; optional Local-first tables
    lf_clip (t, 6) = (body, facet i, facet j, l, k, endpoint) with per-root offsets lf_clip_off, and lf_drop (m, 6) =
    (depth, ix, iy, iz, l, k), all re-proved by certify.propa. Returns a dict report."""
    global _SELF_TESTED
    if not _SELF_TESTED:
        ar.self_test(); _SELF_TESTED = True
    t0 = time.perf_counter()
    cls = np.ascontiguousarray(cls, np.uint8); L = cls.shape[0]; o = np.asarray(origin, np.int64); N = int(N)
    cls_flat = cls.reshape(-1)
    items_code, items_cls = pure_pyramid(cls)
    if not np.any(cls == 1):
        raise ValueError("no response-active FREE receiver: the posterior support is not certified to avoid the solid")
    Wlo = (o + 0.5) / N; Whi = (o + L - 0.5) / N
    # ---- planes: grid, window (deduplicated), labeller axis splits
    PLn = Planes(plane_cap)
    gid = np.zeros((3, 129), np.int64)
    for k in range(3):
        for j in range(129):
            gid[k, j] = PLn.axis_plane(k, j / 128)
    wid = np.zeros((3, 2), np.int64)
    for k in range(3):
        wid[k, 0] = PLn.axis_plane(k, o[k] / N); wid[k, 1] = PLn.axis_plane(k, (o[k] + L) / N)
    ev = np.array(cert["ev"], np.int64, copy=True); evf = np.asarray(cert["evf"], float)
    m = ev[:, 0] == tree.EV_SPLIT_AXIS
    for e in np.flatnonzero(m):
        ev[e, 2] = PLn.axis_plane(int(ev[e, 1]), float(evf[e]))
    # ---- Local-first root restrictions (Proposition A'): every clip row and every dropped root is re-proved here;
    # rows that fail are not applied (sound: the root stays larger), dropped roots that fail are pooled as UNDET
    from . import propa
    roots_all = np.asarray(cert["root_order"], np.int64)
    lf_clip = np.asarray(cert.get("lf_clip", np.zeros((0, 6), np.int64)), np.int64).reshape(-1, 6)
    lf_off = np.asarray(cert.get("lf_clip_off", np.zeros(len(roots_all) + 1, np.int64)), np.int64)
    rclip_ref = np.full(max(len(lf_clip), 1), -1, np.int64)
    lf_stats = dict(clips=int(len(lf_clip)), clips_ok=0, drops=0, drops_ok=0)
    for r in range(len(roots_all)):
        a_, b_ = int(lf_off[r]), int(lf_off[r + 1])
        if b_ > a_:
            _, refs = propa.verify_root_plan(scene, cls, o, N, roots_all[r], dict(clips=[tuple(x) for x in lf_clip[a_:b_]]), PLn)
            rclip_ref[a_:b_] = refs
            lf_stats["clips_ok"] += int(sum(1 for x in refs if x >= 0))
    lf_drop = np.asarray(cert.get("lf_drop", np.zeros((0, 6), np.int64)), np.int64).reshape(-1, 6)
    dropped_ok = set(); dropped_fail = []
    for row in lf_drop:
        lf_stats["drops"] += 1
        ok_, _ = propa.verify_root_plan(scene, cls, o, N, row[:4], dict(drop=(int(row[4]), int(row[5]))), PLn)
        if ok_:
            dropped_ok.add(tuple(int(v) for v in row[:4])); lf_stats["drops_ok"] += 1
        else:
            dropped_fail.append(tuple(int(v) for v in row[:4]))
    # ---- geometry arrays
    C = scene.C; K = scene.K; RM = scene.RM; RR = scene.RR; FB = scene.FB; alo = scene.aabb_lo; ahi = scene.aabb_hi
    RL = scene.RL; RR2 = scene.RR2
    kmax = RM.shape[1]
    S = max_depth + 140
    sl = Slots(S)
    fvm = np.zeros((C, kmax, poly.MV)); fvr = np.zeros((C, kmax, poly.MV))
    fRlo = np.zeros((C, kmax)); fRhi = np.zeros((C, kmax)); asp = np.zeros((C, kmax), np.int64)
    Tlo = np.zeros((C, kmax, kmax)); Thi = np.zeros((C, kmax, kmax))
    fqm = np.zeros((kmax, 8)); fqr = np.zeros((kmax, 8)); fql = np.zeros(kmax); fqh = np.zeros(kmax)
    stats = np.zeros(NSTATS, np.int64)
    wit = np.zeros(2, np.int64)
    cflat = np.zeros(8, np.int64)
    geo = (C, K, RM, RR, FB, RL, RR2, alo, ahi)
    win = (Wlo, Whi, o, float(N), L, cls_flat)
    scr = (fvm, fvr, fRlo, fRhi, asp, Tlo, Thi, fqm, fqr, fql, fqh, cflat)
    NF_ = 1 << 14
    FST = (np.zeros(NF_, np.int64), np.zeros(NF_, np.int64), np.zeros(NF_, np.int64), np.zeros(NF_, np.int64),
           np.zeros(NF_, np.int64), np.zeros((NF_, 2)), np.zeros((NF_, 2)))
    AOST = (np.zeros(NF_, np.int64), np.zeros(NF_, np.int64), np.zeros(NF_, np.int64))
    nofa = (FST[0], FST[1], FST[2], FST[3], FST[4], FST[5], FST[6], 0); noao = (AOST[0], AOST[1], AOST[2], 0)
    # ---- stage 1: partition and refutations
    s1 = np.asarray(cert["s1"], np.int64)
    report = dict(stage1=dict(boxes=int(len(s1)), partition=_octree_partition_ok(s1[:, :4]), refuted_ok=0, refuted_fail=0,
                              window_ok=0, window_fail=0, roots=int(np.sum(s1[:, 4] == 2))))
    undet_boxes = []
    while True:
        AC = np.zeros(arena, np.int64); AK = np.zeros(arena, np.int8); AM = np.zeros(arena, np.int64)
        WC = np.zeros(arena // 2, np.int64); WK = np.zeros(arena // 2, np.int8); WM = np.zeros(arena // 2, np.int64)
        break
    wlo_q = [Q(int(o[k]), N) for k in range(3)]; whi_q = [Q(int(o[k] + L), N) for k in range(3)]
    for d, ix, iy, iz, kind, wx in s1:
        if kind == 2:
            continue
        h = Q(1, 1 << int(d)); lo = [Q(int(v)) * h for v in (ix, iy, iz)]
        if kind == 0:
            ok = all(lo[k] >= wlo_q[k] and lo[k] + h <= whi_q[k] for k in range(3))
            report["stage1"]["window_ok" if ok else "window_fail"] += 1
            if not ok:
                undet_boxes.append((d, ix, iy, iz))
            continue
        ids = np.zeros((3, 2), np.int64)
        for k, v in enumerate((ix, iy, iz)):
            ids[k, 0] = PLn.axis_plane(k, float(lo[k])); ids[k, 1] = PLn.axis_plane(k, float(lo[k] + h))
        poly.set_box(PLn.PL, 0, ids, *sl.a)
        wx = int(wx); code = np.int64(((wx // (L * L)) << 32) | (((wx // L) % L) << 16) | (wx % L))
        ok = tree.eval_one(0, code, np.int8(cls_flat[wx]), C, AC, AK, AM, WC, WK, WM, geo, win, nofa, noao,
                           (sl.a[5], sl.a[6], sl.a[7]), scr, stats, wit)
        report["stage1"]["refuted_ok" if ok else "refuted_fail"] += 1
        if not ok:
            undet_boxes.append((d, ix, iy, iz))
    # ---- roots
    rec_cell = np.zeros(max_records, np.int64); rec_lo = np.zeros(max_records); rec_hi = np.zeros(max_records)
    rec_kind = np.zeros(max_records, np.int8)
    pool = np.zeros(8)
    SPL = np.zeros(1 << 14, np.int64)
    dbgV = np.zeros((64, 256, 3)); dbgI = np.zeros(1 + 4 * 64, np.int64)
    dbgF = np.zeros((64, 256, 5), np.int64); dbgA = np.zeros((64, 256, 3), np.int64)
    dbgINV = np.zeros((256, 256, 3)); dbgIN = np.zeros(257, np.int64)
    nrec = 0
    pos = 0
    roots = np.asarray(cert["root_order"], np.int64)
    root_status = []
    s1roots = {tuple(int(v) for v in r[:4]) for r in s1 if r[4] == 2}
    report["local_first"] = lf_stats
    for r in roots:
        if tuple(int(v) for v in r) not in s1roots:
            raise ValueError("root not in the stage-1 partition")
        pos, nrec, code = tree.verify_root(PLn.PL, PLn.npl, gid, wid, ev, pos, rclip_ref, int(r[0]), int(r[1]), int(r[2]),
                                           int(r[3]), AC, AK, AM, items_code, items_cls, WC, WK, WM, geo, win, shell_eps,
                                           sl.a, scr, rec_cell, rec_lo, rec_hi, rec_kind, nrec, pool, stats, SPL, FST, AOST,
                                           (dbgV, dbgI, dbgF, dbgA, dbgINV, dbgIN), (scene.E_c, scene.E_i, scene.E_j, scene.E_A, scene.E_B, scene.E_r, int(scene.nE)))
        root_status.append(code)
        if code == 3:
            raise _Overflow("arena")
        if code != 0:
            raise RuntimeError(f"verifier: root {r.tolist()} returned code {code} (malformed stream)")
        if int(PLn.npl[0]) >= PLn.PL.shape[0] - 64:
            raise _Overflow("planes")
    if pos != len(ev):
        raise ValueError(f"event stream not fully consumed ({pos} of {len(ev)})")
    covered = set(map(tuple, roots.tolist())) | dropped_ok | set(dropped_fail)
    if covered != s1roots or len(roots) + len(lf_drop) != len(s1roots):
        report["roots_without_tree"] = sorted(s1roots - covered)
    for b_ in undet_boxes:           # an unproved stage-1 refutation: the whole box is undetermined mass
        pool[0] = float(np.nextafter(pool[0] + 2.0 ** (-3 * int(b_[0])), np.inf))
    for b_ in dropped_fail:          # an unproved drop: the whole root box is undetermined mass
        pool[0] = float(np.nextafter(pool[0] + 2.0 ** (-3 * b_[0]), np.inf))
    if nrec > max_records:
        raise _Overflow("records")
    stats[tree.ST_NPL] = int(PLn.npl[0])
    report["stats"] = dict(zip(STAT_NAMES, stats.tolist()))
    report["records"] = dict(cell=rec_cell[:nrec].copy(), lo=rec_lo[:nrec].copy(), hi=rec_hi[:nrec].copy(),
                             kind=rec_kind[:nrec].copy())
    report["pool"] = float(pool[0])
    report["debug_in_pieces"] = [dbgINV[k, :dbgIN[1 + k]].copy() for k in range(int(dbgIN[0]))]
    report["debug_out_fail"] = [dict(pos=int(dbgI[1 + 4 * k]), V=dbgV[k, :dbgI[2 + 4 * k]].copy(), nfa=int(dbgI[3 + 4 * k]),
                                     nao=int(dbgI[4 + 4 * k]), F=dbgF[k, :min(int(dbgI[3 + 4 * k]), 256)].copy(),
                                     A=dbgA[k, :min(int(dbgI[4 + 4 * k]), 256)].copy()) for k in range(int(dbgI[0]))]

    report["pool_parts"] = dict(clip_slab=float(pool[1]), face_slab=float(pool[2]), pencil_slab=float(pool[3]), emit_fail=float(pool[4]))
    report["undet_boxes"] = undet_boxes
    report["seconds"] = time.perf_counter() - t0
    return report


def cell_enclosures(rep):
    """Per-cell rigorous [lo, hi]: exact rational sums of the binary64 record bounds, rounded outward."""
    r = rep["records"]
    acc = {}
    for c, lo, hi in zip(r["cell"].tolist(), r["lo"].tolist(), r["hi"].tolist()):
        a = acc.setdefault(c, [Q(0), Q(0)])
        a[0] += Q(lo); a[1] += Q(hi)
    cells = np.array(sorted(acc), np.int64)
    lo = np.array([_dnq(acc[c][0]) for c in cells.tolist()]); hi = np.array([_upq(acc[c][1]) for c in cells.tolist()])
    return cells, lo, hi


def _dnq(q):
    m = float(q)
    return m if Q(m) <= q else float(np.nextafter(m, -np.inf))


def _upq(q):
    m = float(q)
    return m if Q(m) >= q else float(np.nextafter(m, np.inf))
