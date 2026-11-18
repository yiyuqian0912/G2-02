"""XLF solver: exact BSP of the source domain driven by local inverse geometry (research prototype).

Every node is an exact plane-based polytope (certify.poly: planes are binary64 rows read as exact rationals; children
{h <= 0}, {h >= 0} tile their parent exactly). A node becomes a leaf only with a proven claim:
  OUT   some receiver's observation is contradicted for every source in the node (xlf.pred lemmas D/L), or the node
        lies in the closed window box (outside the source domain D);
  IN    every FREE receiver of the window is proven consistent for every source of the node and the node is proven
        inside D (disjoint from the window box, clearance > 2^-26 from the solid): Vol(F n node) = Vol(node);
  UNDET neither (thin uncertainty slabs, depth limits): its volume is an upper bound only.
Search (untrusted, cannot affect correctness):
  coarse  bisection at dyadic cell boundaries with collective row classification (pred.scan_rows); children re-scan only
          the rows that held an ambiguous receiver;
  fine    (node inside one posterior cell, few ambiguous receivers) local inverse geometry: every ambiguous receiver
          whose observation is a convex constraint on the node (pred.recv_test, lemma S) contributes halfspaces; the
          halfspaces of one occluder edge form a pencil, of which only the angular extremes bind; the node is cut by
          those planes (each rounded to binary64 and widened to a slab of half-width delta so that both sides are
          provable); outer pieces are OUT by their receiver, slabs UNDET, the inner piece is re-classified globally.
          Receivers with non-convex constraints split the node by one of their planes."""
import time

import numpy as np

from ..certify import poly as PY
from . import pred as PR

EPS_CLEAR = 2.0 ** -26
NCELL = 128


class Problem:
    def __init__(self, scene, cls, o, N):
        self.scene = scene; self.cls = np.ascontiguousarray(cls, np.uint8); self.o = np.asarray(o, np.int64)
        self.N = int(N); self.L = int(cls.shape[0])
        self.RM, self.RR, self.K, self.ADJ = scene.RM, scene.RR, scene.K, scene.ADJ
        self.AL, self.AH, self.C = scene.aabb_lo, scene.aabb_hi, int(scene.C)
        L = self.L
        X, Y = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
        rows = np.stack([X.ravel(), Y.ravel()], 1).astype(np.int64)
        # interleaved order (bit reversal of the row index): early rows sample the whole window
        q = int(round(np.log2(L * L)))
        rev = np.array([int(format(i, f"0{q}b")[::-1], 2) for i in range(L * L)], np.int64)
        self.rows = np.ascontiguousarray(rows[np.argsort(rev)])
        self.wlo = self.o / self.N; self.whi = (self.o + L) / self.N
        self.rlo = (self.o + 0.5) / self.N; self.rhi = (self.o + L - 0.5) / self.N
        self.nrm = np.array([[np.sqrt(np.sum((np.abs(scene.RM[c, i, :3]) + scene.RR[c, i, :3]) ** 2)) * (1 + 2 ** -40)
                              for i in range(scene.RM.shape[1])] for c in range(self.C)])


class Planes:
    def __init__(self, cap=1 << 20):
        self.PL = np.zeros((cap, 4)); self.n = 0; self.axis = {}

    def add(self, row):
        if self.n >= len(self.PL):
            P = np.zeros((2 * len(self.PL), 4)); P[:self.n] = self.PL[:self.n]; self.PL = P
        self.PL[self.n] = row; self.n += 1
        return self.n - 1

    def axis_plane(self, k, val):
        key = (k, float(val))
        if key not in self.axis:
            r = np.zeros(4); r[k] = 1.0; r[3] = -float(val)
            self.axis[key] = self.add(r)
        return self.axis[key]


class Slots:
    def __init__(self, S=1 << 14):
        self.S = S; self.a = PY.new_slots(S); self.free = list(range(S - 1, -1, -1))

    def get(self):
        if not self.free:
            raise MemoryError("slot arena exhausted")
        return self.free.pop()

    def put(self, s):
        self.a["nV"][s] = 0; self.a["nF"][s] = 0; self.free.append(s)

    def args(self):
        a = self.a
        return a["Fp"], a["Fn"], a["Fv"], a["Fe"], a["Vp"], a["Vm"], a["Vr"], a["nV"], a["nF"]


class Node:
    __slots__ = ("slot", "rows", "depth", "box", "fine", "nid", "cell")

    def __init__(self, slot, rows, depth, box, fine, nid, cell=-1):
        self.slot, self.rows, self.depth, self.box, self.fine, self.nid = slot, rows, depth, box, fine, nid
        self.cell = cell
        if box is not None and cell < 0:
            a = np.floor(box[0] * NCELL).astype(np.int64); b = np.ceil(box[1] * NCELL).astype(np.int64) - 1
            if np.all(a == b):
                self.cell = int((a[0] * NCELL + a[1]) * NCELL + a[2])


def box_cells(box):
    """Cell index ranges [a, b] (inclusive) of an exact box."""
    a = np.floor(box[0] * NCELL).astype(np.int64); b = np.ceil(box[1] * NCELL).astype(np.int64) - 1
    return np.clip(a, 0, NCELL - 1), np.clip(b, 0, NCELL - 1)


class Solver:
    def __init__(self, prob, cap_fine=4096, cap_amb=1 << 16, max_depth=160, max_fine=48, delta_floor=2.0 ** -49, rcap=256):
        self.p = prob; self.rcap = rcap; self.cap_fine = cap_fine; self.cap_amb = cap_amb; self.max_depth = max_depth
        self.max_fine = max_fine; self.delta_floor = delta_floor
        self.PLs = Planes(); self.sl = Slots()
        self.stats = dict(nodes=0, out=0, out_window=0, in_=0, undet=0, empty=0, bisect=0, fine=0, oneshot=0,
                          pencil_split=0, clips=0, slabs=0, out_confirm_fail=0, scan_s=0.0, recv_s=0.0, rows_scanned=0,
                          in_box=0, depth_max=0, clear_fail=0)
        self.cell_lo = {}; self.cell_hi = {}; self.pool = 0.0
        # certificate: BSP nodes (parent, plane ref of the child's side) and leaves
        self.bsp_parent = []; self.bsp_ref = []; self.leaves = []
        self.comps = np.zeros(max(prob.C, 1), np.int64)
        self.amb = np.zeros((cap_amb, 4), np.int64); self.wit = np.zeros(4, np.int64)
        self.hs_m = np.zeros((PR.MH, 4)); self.hs_r = np.zeros((PR.MH, 4)); self.hs_key = np.zeros((PR.MH, 4), np.int64)
        self.hs_s = np.zeros(PR.MH, np.int64)

    # ------------------------------------------------------------------ BSP bookkeeping
    def _new_nid(self, parent, ref):
        self.bsp_parent.append(parent); self.bsp_ref.append(ref)
        return len(self.bsp_parent) - 1

    def _leaf(self, node, kind, data=None):
        self.leaves.append((node.nid, kind, data))
        self.sl.put(node.slot)

    def _split(self, node, pid, box_lo=None, box_hi=None, rows=None):
        """Children {h <= 0} (ref 2 pid) and {h >= 0} (ref 2 pid + 1); returns the non-empty children nodes."""
        a = self.sl.args(); out = []
        rows = node.rows if rows is None else rows
        for side in (0, 1):
            s = self.sl.get()
            PY.copy_slot(node.slot, s, *a)
            st = PY.clip(self.PLs.PL, s, 2 * pid + side, *a)
            if st == PY.OVERFLOW:
                raise RuntimeError("polytope overflow")
            nid = self._new_nid(node.nid, 2 * pid + side)
            if st == PY.EMPTY or a[7][s] == 0:
                self.leaves.append((nid, "EMPTY", None)); self.sl.put(s); self.stats["empty"] += 1
                continue
            out.append(Node(s, rows, node.depth + 1, (box_lo[side], box_hi[side]) if box_lo is not None else None,
                            node.fine, nid, node.cell))
        self.sl.put(node.slot)
        return out

    # ------------------------------------------------------------------ geometry helpers
    def _verts(self, s):
        a = self.sl.a; nv = int(a["nV"][s])
        return a["Vm"][s, :nv], a["Vr"][s, :nv], nv

    def _bbox(self, s):
        Vm, Vr, nv = self._verts(s)
        return np.min(Vm - Vr, 0), np.max(Vm + Vr, 0)

    # ------------------------------------------------------------------ main loop
    def run(self, time_limit=None):
        p = self.p; t0 = time.perf_counter()
        PLs = self.PLs
        s = self.sl.get()
        ax = np.array([[PLs.axis_plane(k, 0.0), PLs.axis_plane(k, 1.0)] for k in range(3)], np.int64)
        PY.set_box(PLs.PL, s, ax, *self.sl.args())
        root = Node(s, p.rows, 0, (np.zeros(3), np.ones(3)), 0, self._new_nid(-1, -1))
        stack = [root]
        while stack:
            if time_limit is not None and time.perf_counter() - t0 > time_limit:
                raise TimeoutError
            node = stack.pop()
            self.stats["nodes"] += 1
            if node.depth > self.stats["depth_max"]:
                self.stats["depth_max"] = node.depth
            stack.extend(self._process(node))
        self.seconds = time.perf_counter() - t0
        return self

    def _process(self, node):
        p = self.p; s = node.slot
        Vm, Vr, nv = self._verts(s)
        if nv == 0:
            self._leaf(node, "EMPTY"); self.stats["empty"] += 1
            return []
        lo, hi = np.min(Vm - Vr, 0), np.max(Vm + Vr, 0)
        # inside the closed window box: outside D
        if np.all(lo >= p.wlo) and np.all(hi <= p.whi):
            self._leaf(node, "OUT_WINDOW"); self.stats["out_window"] += 1
            return []
        nc = PR.relevant_comps(Vm, Vr, nv, p.rlo, p.rhi, p.AL, p.AH, p.C, self.comps)
        comps = self.comps[:nc].copy()
        rows = node.rows
        rowflag = np.zeros(len(rows), np.int64)
        t0 = time.perf_counter()
        code, namb = PR.scan_rows(Vm, Vr, nv, rows, len(rows), p.cls, p.o, p.N, p.L, comps, nc, p.RM, p.RR, p.K, p.ADJ,
                                  self.amb, self.cap_amb, rowflag, self.wit, self.rcap)
        self.stats["scan_s"] += time.perf_counter() - t0; self.stats["rows_scanned"] += len(rows)
        if code == PR.SC_BUG:
            raise AssertionError(f"inconsistent sure classification at receiver {self.wit.tolist()}")
        if code == PR.SC_OUT:
            self._leaf(node, "OUT", tuple(int(v) for v in self.wit)); self.stats["out"] += 1
            return []
        nrows = rows[rowflag == 1]
        if namb == 0:
            return self._in_or_split(node, lo, hi)
        if node.depth >= self.max_depth or node.fine >= self.max_fine:
            return self._undet(node)
        if namb > self.cap_fine:
            return self._bisect(node, lo, hi, nrows)
        return self._fine(node, nrows, namb, comps)

    # ------------------------------------------------------------------ leaves
    def _undet(self, node):
        vlo, vhi = PY.volume(node.slot, self.sl.a["Fn"], self.sl.a["Fv"], self.sl.a["Vm"], self.sl.a["Vr"], self.sl.a["nF"])
        if node.cell >= 0:
            self.cell_hi[node.cell] = self.cell_hi.get(node.cell, 0.0) + vhi
        else:
            self._cell_volumes(node.slot, False)
        self._leaf(node, "UNDET", vhi); self.stats["undet"] += 1
        return []

    def _clear(self, Vm, Vr, nv, lo, hi):
        """Proof that the node has clearance > 2^-26 from every component (AABB gap or a facet with f_i > eps |n_i|)."""
        p = self.p
        for c in range(p.C):
            if np.any(lo - p.AH[c] > EPS_CLEAR * 2) or np.any(p.AL[c] - hi > EPS_CLEAR * 2):
                continue
            ok = False
            for i in range(int(p.K[c])):
                l, _ = PR.range_on(p.RM[c, i], p.RR[c, i], Vm, Vr, nv)
                if l > EPS_CLEAR * p.nrm[c, i] * (1 + 2 ** -30):
                    ok = True
                    break
            if not ok:
                return False
        return True

    def _in_or_split(self, node, lo, hi):
        p = self.p
        # disjoint from the closed window box? else split by one of its planes
        disj = np.any(hi <= p.wlo) or np.any(lo >= p.whi)
        if not disj:
            for k in range(3):
                for val in (p.wlo[k], p.whi[k]):
                    if lo[k] < val < hi[k]:
                        pid = self.PLs.axis_plane(k, val)
                        bl = bh = None
                        if node.box is not None:
                            b0, b1 = node.box
                            l0 = b0.copy(); h0 = b1.copy(); h0[k] = val
                            l1 = b0.copy(); h1 = b1.copy(); l1[k] = val
                            bl, bh = (l0, l1), (h0, h1)
                        return self._split(node, pid, bl, bh)
            return self._undet(node)
        Vm, Vr, nv = self._verts(node.slot)
        if not self._clear(Vm, Vr, nv, lo, hi):
            self.stats["clear_fail"] += 1
            if node.depth >= self.max_depth or (hi - lo).max() < 1e-9:
                return self._undet(node)
            return self._bisect(node, lo, hi, node.rows)
        a = self.sl.a
        if node.cell >= 0:
            vlo, vhi = PY.volume(node.slot, a["Fn"], a["Fv"], a["Vm"], a["Vr"], a["nF"])
            cid = node.cell
            self.cell_lo[cid] = self.cell_lo.get(cid, 0.0) + vlo; self.cell_hi[cid] = self.cell_hi.get(cid, 0.0) + vhi
            self._leaf(node, "IN", (vlo, vhi)); self.stats["in_"] += 1
            return []
        if node.box is None:
            self._cell_volumes(node.slot, True)
            self._leaf(node, "IN", None); self.stats["in_"] += 1
            return []
        # an IN box over several cells: exact box, per-cell volumes from the exact corner coordinates
        b0, b1 = node.box
        ca, cb = box_cells(node.box)
        for ix in range(ca[0], cb[0] + 1):
            for iy in range(ca[1], cb[1] + 1):
                for iz in range(ca[2], cb[2] + 1):
                    c0 = np.array([ix, iy, iz]) / NCELL; c1 = (np.array([ix, iy, iz]) + 1) / NCELL
                    e = np.minimum(b1, c1) - np.maximum(b0, c0)
                    if np.any(e <= 0):
                        continue
                    v = float(e[0]) * float(e[1]) * float(e[2])
                    vlo = np.nextafter(np.nextafter(v, 0.0) * (1 - 2 ** -50), 0.0); vhi = np.nextafter(v * (1 + 2 ** -50), np.inf)
                    cid = int((ix * NCELL + iy) * NCELL + iz)
                    self.cell_lo[cid] = self.cell_lo.get(cid, 0.0) + vlo; self.cell_hi[cid] = self.cell_hi.get(cid, 0.0) + vhi
        self._leaf(node, "IN_BOX", (b0.tolist(), b1.tolist())); self.stats["in_"] += 1; self.stats["in_box"] += 1
        return []

    # ------------------------------------------------------------------ coarse: bisection
    def _bisect(self, node, lo, hi, nrows):
        self.stats["bisect"] += 1
        if node.box is not None:
            lo, hi = node.box
            ca, cb = box_cells(node.box)
        else:
            ca = cb = np.zeros(3, np.int64)
        ext = hi - lo
        span = cb - ca
        if np.any(span > 0):
            k = int(np.argmax(np.where(span > 0, ext, -1)))
            mid = 0.5 * (lo[k] + hi[k])
            val = np.clip(np.round(mid * NCELL), ca[k] + 1, cb[k]) / NCELL
        else:
            k = int(np.argmax(ext)); val = 0.5 * (lo[k] + hi[k])
        pid = self.PLs.axis_plane(k, val)
        bl = bh = None
        if node.box is not None:
            b0, b1 = node.box
            l0 = b0.copy(); h0 = b1.copy(); h0[k] = val
            l1 = b0.copy(); h1 = b1.copy(); l1[k] = val
            bl, bh = (l0, l1), (h0, h1)
        return self._split(node, pid, bl, bh, rows=nrows)

    # ------------------------------------------------------------------ fine: local inverse geometry
    def _recv(self, s, X, Y, t, comps):
        p = self.p; Vm, Vr, nv = self._verts(s)
        rx = (p.o[0] + X + 0.5) / p.N; ry = (p.o[1] + Y + 0.5) / p.N; rz = (p.o[2] + t + 0.5) / p.N
        obs = int(p.cls[X, Y, t])
        st, nh, w = PR.recv_test(Vm, Vr, nv, rx, ry, rz, obs, comps, len(comps), p.RM, p.RR, p.K, p.ADJ,
                                 self.hs_m, self.hs_r, self.hs_key, self.hs_s)
        return st, nh, obs

    def _contradiction(self, st, obs):
        return (st == PR.R_DARK and obs == PR.LIT) or (st == PR.R_LIT and obs == PR.DARK)

    def _plane_pair(self, cm, cr):
        """Binary64 planes A = h + delta, B = h - delta around g = cm +- cr (h = cm / max|cm|) with
        {A <= 0} subset {g < 0} and {B >= 0} subset {g > 0} on [0, 1]^3 (delta > |g/s - h| there)."""
        sc = float(np.max(np.abs(cm)))
        h = cm / sc
        E = float(np.sum(np.abs(cm - sc * h) + cr)) * (1 + 2 ** -40) / sc
        d = max(8.0 * E, self.delta_floor)
        A = h.copy(); A[3] = h[3] + d
        B = h.copy(); B[3] = h[3] - d
        return self.PLs.add(A), self.PLs.add(B)

    def _fine(self, node, nrows, namb, comps):
        self.stats["fine"] += 1
        p = self.p; s = node.slot
        Vm, Vr, nv = self._verts(s)
        t0 = time.perf_counter()
        cap = 4 * PR.MH * max(namb, 1)
        HM = np.empty((cap, 4)); HR = np.empty((cap, 4)); HK = np.empty((cap, 4), np.int64); HS = np.empty(cap, np.int64)
        HPHI = np.empty(cap); HREC = np.empty(cap, np.int64)
        code, n, nconv, fnc = PR.fine_collect(Vm, Vr, nv, self.amb, min(namb, self.cap_amb), p.cls, p.o, p.N, comps,
                                              len(comps), p.RM, p.RR, p.K, p.ADJ, HM, HR, HK, HS, HPHI, HREC, cap, self.wit)
        self.stats["recv_s"] += time.perf_counter() - t0
        if code == 1:
            self._leaf(node, "OUT", tuple(int(v) for v in self.wit)); self.stats["out"] += 1
            return []
        if n == 0:
            if fnc >= 0:
                X, Y, t = (int(v) for v in self.amb[fnc, :3])
                st, nh, obs = self._recv(s, X, Y, t, comps)
                return self._pencil_split(node, (X, Y, t, self.hs_m[:nh].copy(), self.hs_r[:nh].copy()), nrows)
            return self._undet(node)
        # binding halfspaces: per (pencil, sense) group the angular extremes
        keys = np.concatenate([HK[:n], HS[:n, None]], 1)
        order = np.lexsort((HPHI[:n], keys[:, 4], keys[:, 3], keys[:, 2], keys[:, 1], keys[:, 0]))
        ks = keys[order]
        brk = np.flatnonzero(np.any(ks[1:] != ks[:-1], 1)) + 1
        starts = np.concatenate([[0], brk]); ends = np.concatenate([brk, [n]]) - 1
        pick = np.unique(np.concatenate([order[starts], order[ends]]))
        hs = []
        for h in pick:
            a = int(HREC[h])
            hs.append((int(HS[h]), HM[h].copy(), HR[h].copy(), tuple(int(v) for v in self.amb[a, :3])))
        self.stats["oneshot"] += 1
        return self._clip_sequence(node, hs, nrows, comps)

    def _pencil_split(self, node, nc, nrows):
        self.stats["pencil_split"] += 1
        X, Y, t, hm, hr = nc
        Vm, Vr, nv = self._verts(node.slot)
        best = None
        for h in range(len(hm)):
            lo, hi = PR.range_on(hm[h], hr[h], Vm, Vr, nv)
            sc = min(-lo, hi)
            if best is None or sc > best[0]:
                best = (sc, h)
        A, B = self._plane_pair(hm[best[1]], hr[best[1]])
        out = []
        for ch in self._split(node, A, rows=nrows):
            ch.fine = node.fine + 1
            out.append(ch)
        res = []
        for ch in out:
            if self.bsp_ref[ch.nid] == 2 * A + 1:          # {A >= 0}: split again by B
                for g in self._split(ch, B, rows=nrows):
                    g.fine = ch.fine
                    res.append(g)
            else:
                res.append(ch)
        return res

    def _clip_sequence(self, node, hs, nrows, comps):
        """Cut the node by the binding halfspaces; returns the nodes left to process (inner piece + unconfirmed)."""
        a = self.sl.args()
        pieces = [(node, False)]          # (node, is_slab)
        rest = []
        for sense, cm, cr, rcv in hs:
            A, B = self._plane_pair(cm, cr)
            inner_ref = 2 * A if sense == 1 else 2 * B + 1
            first = A if sense == 1 else B
            second = B if sense == 1 else A
            new = []
            for pc, slab in pieces:
                kids = self._split(pc, first, rows=nrows)
                for ch in kids:
                    ch.fine = node.fine + 1
                    if self.bsp_ref[ch.nid] == inner_ref:
                        new.append((ch, slab))
                        continue
                    # rest: split by the second plane into the outer piece and the slab
                    for g in self._split(ch, second, rows=nrows):
                        g.fine = node.fine + 1
                        is_outer = (self.bsp_ref[g.nid] == (2 * B + 1 if sense == 1 else 2 * A))
                        if is_outer:
                            st, nh, obs = self._recv(g.slot, rcv[0], rcv[1], rcv[2], comps)
                            if self._contradiction(st, obs):
                                self._leaf(g, "OUT", (rcv[0], rcv[1], rcv[2], 3)); self.stats["out"] += 1
                            else:
                                self.stats["out_confirm_fail"] += 1
                                rest.append(g)
                        else:
                            self.stats["slabs"] += 1
                            new.append((g, True))
                self.stats["clips"] += 1
            pieces = new
        for pc, slab in pieces:
            if slab:
                self._undet(pc)
            else:
                rest.append(pc)
        return rest

    def _cell_volumes(self, s0, is_in):
        """Per-cell volume enclosures of the polytope in slot s0 by recursive clipping at cell boundaries (exact
        partition); IN adds [lo, hi], UNDET adds hi. Slot s0 is left untouched."""
        a = self.sl.args(); A = self.sl.a
        stack = []
        s = self.sl.get(); PY.copy_slot(s0, s, *a); stack.append(s)
        while stack:
            s = stack.pop()
            if A["nV"][s] == 0:
                self.sl.put(s); continue
            lo, hi = self._bbox(s)
            ca = np.clip(np.floor(lo * NCELL).astype(np.int64), 0, NCELL - 1)
            cb = np.clip(np.ceil(hi * NCELL).astype(np.int64) - 1, 0, NCELL - 1)
            span = cb - ca
            if np.all(span == 0):
                vlo, vhi = PY.volume(s, A["Fn"], A["Fv"], A["Vm"], A["Vr"], A["nF"])
                cid = int((ca[0] * NCELL + ca[1]) * NCELL + ca[2])
                if is_in:
                    self.cell_lo[cid] = self.cell_lo.get(cid, 0.0) + vlo
                self.cell_hi[cid] = self.cell_hi.get(cid, 0.0) + vhi
                self.sl.put(s); continue
            k = int(np.argmax(span)); val = (ca[k] + (span[k] + 1) // 2) / NCELL
            pid = self.PLs.axis_plane(k, val)
            for side in (0, 1):
                t = self.sl.get(); PY.copy_slot(s, t, *a)
                st = PY.clip(self.PLs.PL, t, 2 * pid + side, *a)
                if st == PY.OVERFLOW:
                    raise RuntimeError("polytope overflow")
                if st == PY.EMPTY:
                    self.sl.put(t)
                else:
                    stack.append(t)
            self.sl.put(s)

    # ------------------------------------------------------------------ result
    def result(self):
        cells = sorted(set(self.cell_lo) | set(self.cell_hi))
        lo = np.array([self.cell_lo.get(c, 0.0) for c in cells]); hi = np.array([self.cell_hi.get(c, 0.0) for c in cells])
        return np.array(cells, np.int64), lo, hi, float(self.pool)
