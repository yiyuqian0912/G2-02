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
        self.pyr, self.poff, self.q = PR.pyramid(self.cls)
        self.lb = min(3, self.q)                       # inheritance granularity of the block scan (8^3 receivers)
        self.root_starts = np.array([[self.q, 0, 0, 0]], np.int64)
        # static sentinels (deterministic): interface receivers and a few interior ones of each class, tested before
        # the full scan of coarse nodes (any contradicted receiver is a valid OUT witness)
        lit = self.cls == PR.LIT; dark = self.cls == PR.DARK
        bd = np.zeros_like(lit)
        for ax in range(3):
            a = [slice(None)] * 3; b = [slice(None)] * 3; a[ax] = slice(0, L - 1); b[ax] = slice(1, L)
            x = (lit[tuple(a)] & dark[tuple(b)]) | (dark[tuple(a)] & lit[tuple(b)]); bd[tuple(a)] |= x; bd[tuple(b)] |= x
        rng = np.random.default_rng(20261118)
        picks = []
        for mask, n in ((bd, 384), (lit & ~bd, 64), (dark & ~bd, 64)):
            idx = np.flatnonzero(mask.reshape(-1))
            if len(idx):
                picks.append(rng.choice(idx, min(n, len(idx)), replace=False))
        idx = np.sort(np.concatenate(picks)) if picks else np.zeros(0, np.int64)
        v = np.stack(np.unravel_index(idx, self.cls.shape), 1)
        self.sentinels = np.concatenate([np.zeros((len(v), 1), np.int64), v.astype(np.int64)], 1)
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
    __slots__ = ("slot", "rows", "depth", "box", "fine", "nid", "cell", "nbm", "npar", "stuck", "used", "inside")

    def __init__(self, slot, rows, depth, box, fine, nid, cell=-1):
        self.slot, self.rows, self.depth, self.box, self.fine, self.nid = slot, rows, depth, box, fine, nid
        self.cell = cell; self.nbm = float("inf"); self.npar = float("inf"); self.stuck = False
        self.used = frozenset()          # facets (c, i) this node was already split by (never split by them again)
        self.inside = {}                 # (c, i) -> plane id A with node subset {A <= 0} subset {f_i < 0} (by construction)
        if box is not None and cell < 0:
            a = np.floor(box[0] * NCELL).astype(np.int64); b = np.ceil(box[1] * NCELL).astype(np.int64) - 1
            if np.all(a == b):
                self.cell = int((a[0] * NCELL + a[1]) * NCELL + a[2])


def box_cells(box):
    """Cell index ranges [a, b] (inclusive) of an exact box."""
    a = np.floor(box[0] * NCELL).astype(np.int64); b = np.ceil(box[1] * NCELL).astype(np.int64) - 1
    return np.clip(a, 0, NCELL - 1), np.clip(b, 0, NCELL - 1)


class Solver:
    def __init__(self, prob, cap_fine=16384, cap_amb=1 << 16, max_depth=160, max_fine=48, delta_floor=2.0 ** -49, rcap=256,
                 mode="tree"):
        self.mode = mode; self.prepass = 3; self.fine_sample = 1 << 30; self.sentinel_min = 0; self.multires = True
        self.stuck_splits = True         # median pencil splits of stalled nodes (no enumeration abort)
        if self.multires:
            self.prepass = 0
        self.p = prob; self.rcap = rcap; self.progress = 0; self.cap_fine = cap_fine; self.cap_amb = cap_amb; self.max_depth = max_depth
        self.max_fine = max_fine; self.delta_floor = delta_floor
        self.PLs = Planes(); self.sl = Slots()
        self.stats = dict(nodes=0, out=0, out_window=0, in_=0, undet=0, empty=0, bisect=0, fine=0, oneshot=0,
                          pencil_split=0, clips=0, slabs=0, out_confirm_fail=0, scan_s=0.0, recv_s=0.0, rows_scanned=0,
                          in_box=0, depth_max=0, clear_fail=0)
        self.cell_lo = {}; self.cell_hi = {}; self.pool = 0.0; self.undet_vol = {}
        # certificate: BSP nodes (parent, plane ref of the child's side) and leaves
        self.bsp_parent = []; self.bsp_ref = []; self.leaves = []
        self.comps = np.zeros(max(prob.C, 1), np.int64)
        self.hint = np.zeros((16, 2), np.int64); self.nhint = 0
        self.thint = np.zeros((16, 4), np.int64); self.ntint = 0
        self.stats.update(rows_hint=0, hint_out=0)
        self.amb = np.zeros((cap_amb, 4), np.int64); self._bm0 = np.zeros(prob.L ** 3, np.int64); self.cnt = np.zeros(4, np.int64); self.wit = np.zeros(4, np.int64)
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
            ch = Node(s, rows, node.depth + 1, (box_lo[side], box_hi[side]) if box_lo is not None else None,
                      node.fine, nid, node.cell)
            ch.used = node.used; ch.inside = node.inside; ch.stuck = node.stuck
            out.append(ch)
        self.sl.put(node.slot)
        return out

    def _scan_tree_node(self, node, Vm, Vr, nv, lo, hi, comps, nc, prep, t0):
        p = self.p
        if self.ntint:
            bm = np.zeros(1 << (3 * (p.q - 0)), np.int64) if False else self._bm0
            code, _ = PR.scan_tree(Vm, Vr, nv, self.thint[:self.ntint], self.ntint, 0, p.pyr, p.poff, p.q, p.cls, p.o, p.N,
                                   p.L, comps, nc, p.RM, p.RR, p.K, p.ADJ, self.amb, self.cap_amb, bm, self.wit, 0, *prep)
            if code == PR.SC_OUT:
                self.stats["scan_s"] += time.perf_counter() - t0; self.stats["hint_out"] += 1
                self._leaf(node, "OUT", tuple(int(v) for v in self.wit)); self.stats["out"] += 1
                return []
        if self.prepass:
            code, _ = PR.scan_tree(Vm, Vr, nv, node.rows, len(node.rows), p.lb, p.pyr, p.poff, p.q, p.cls, p.o, p.N, p.L,
                                   comps, nc, p.RM, p.RR, p.K, p.ADJ, self.amb, self.cap_amb, self._bm0, self.wit, 0, *prep,
                                   self.prepass)
            if code == PR.SC_OUT:
                self.stats["scan_s"] += time.perf_counter() - t0; self.stats["prepass_out"] = self.stats.get("prepass_out", 0) + 1
                self._add_thint(int(self.wit[0]), int(self.wit[1]), int(self.wit[2]))
                self._leaf(node, "OUT", tuple(int(v) for v in self.wit)); self.stats["out"] += 1
                return []
        if len(node.rows) > self.sentinel_min and len(p.sentinels):
            code, _ = PR.scan_tree(Vm, Vr, nv, p.sentinels, len(p.sentinels), 0, p.pyr, p.poff, p.q, p.cls, p.o, p.N, p.L,
                                   comps, nc, p.RM, p.RR, p.K, p.ADJ, self.amb, self.cap_amb, self._bm0, self.wit, 0, *prep)
            if code == PR.SC_OUT:
                self.stats["scan_s"] += time.perf_counter() - t0; self.stats["sentinel_out"] = self.stats.get("sentinel_out", 0) + 1
                self._add_thint(int(self.wit[0]), int(self.wit[1]), int(self.wit[2]))
                self._leaf(node, "OUT", tuple(int(v) for v in self.wit)); self.stats["out"] += 1
                return []
        nbk = p.L >> p.lb
        bmark = np.zeros(nbk ** 3, np.int64)
        node_nbm = float("inf")
        if self.multires and p.lb > 0:
            # block resolution first: refutation + undecided level-lb blocks; full receiver resolution only when few
            code, nbm = PR.scan_tree(Vm, Vr, nv, node.rows, len(node.rows), p.lb, p.pyr, p.poff, p.q, p.cls, p.o, p.N,
                                     p.L, comps, nc, p.RM, p.RR, p.K, p.ADJ, self.amb, self.cap_amb, bmark, self.wit, 0,
                                     *prep, p.lb, self.cnt)
            if code == PR.SC_BUG:
                raise AssertionError(f"inconsistent sure classification at receiver block {self.wit.tolist()}")
            if code == PR.SC_OUT:
                self.stats["scan_s"] += time.perf_counter() - t0; self.stats["block_out"] = self.stats.get("block_out", 0) + 1
                self._add_thint(int(self.wit[0]), int(self.wit[1]), int(self.wit[2]))
                self._leaf(node, "OUT", tuple(int(v) for v in self.wit)); self.stats["out"] += 1
                return []
            idx = np.flatnonzero(bmark)
            nst = np.empty((len(idx), 4), np.int64); nst[:, 0] = p.lb
            nst[:, 1] = idx // (nbk * nbk); nst[:, 2] = (idx // nbk) % nbk; nst[:, 3] = idx % nbk
            if len(idx) == 0:
                self.stats["scan_s"] += time.perf_counter() - t0
                return self._in_or_split(node, lo, hi)
            # coarse while the block-scale ambiguity is large AND still shrinking (pure blocks can stay undecided at
            # block resolution even for tiny nodes: a constraint plane may cross a block's hull without separating
            # its receivers), else full receiver resolution
            if nbm > self.cap_fine // 512 and nbm < 0.9 * node.nbm:
                self.stats["scan_s"] += time.perf_counter() - t0; self.stats["block_bisect"] = self.stats.get("block_bisect", 0) + 1
                if node.depth >= self.max_depth:
                    return self._undet(node, "tree_depth_block")
                sp = self._solid_split(node, lo, hi, nst)
                if sp is not None:
                    return sp
                kids = self._bisect(node, lo, hi, nst)
                for ch in kids:
                    ch.nbm = nbm
                return kids
            node.rows = nst
            node_nbm = nbm
            bmark = np.zeros(nbk ** 3, np.int64)
        code, namb = PR.scan_tree(Vm, Vr, nv, node.rows, len(node.rows), p.lb, p.pyr, p.poff, p.q, p.cls, p.o, p.N, p.L,
                                  comps, nc, p.RM, p.RR, p.K, p.ADJ, self.amb, self.cap_amb, bmark, self.wit, self.rcap,
                                  *prep, 0, self.cnt, 1 << 62)
        self.stats["starts"] = self.stats.get("starts", 0) + len(node.rows)
        if code == PR.SC_MANY:
            # stalled node: the full ambiguous set is not needed to split it again by a pencil plane
            self.stats["scan_s"] += time.perf_counter() - t0; self.stats["many"] = self.stats.get("many", 0) + 1
            if node.depth >= self.max_depth:
                return self._undet(node, "depth_many")
            kids = self._stuck_split(node, node.rows, comps, namb)
            if kids is not None:
                return kids
            return self._bisect(node, lo, hi, node.rows)
        if code == PR.SC_MANY:
            self.stats["scan_s"] += time.perf_counter() - t0; self.stats["many"] = self.stats.get("many", 0) + 1
            if node.depth >= self.max_depth:
                return self._undet(node, "depth_many")
            return self._bisect(node, lo, hi, node.rows)
        self.stats["scan_s"] += time.perf_counter() - t0; self.stats["rows_scanned"] += len(node.rows)
        if code == PR.SC_BUG:
            raise AssertionError(f"inconsistent sure classification at receiver block {self.wit.tolist()}")
        if code == PR.SC_OUT:
            self._add_thint(int(self.wit[0]), int(self.wit[1]), int(self.wit[2]))
            self._leaf(node, "OUT", tuple(int(v) for v in self.wit)); self.stats["out"] += 1
            return []
        idx = np.flatnonzero(bmark)
        nst = np.empty((len(idx), 4), np.int64); nst[:, 0] = p.lb
        nst[:, 1] = idx // (nbk * nbk); nst[:, 2] = (idx // nbk) % nbk; nst[:, 3] = idx % nbk
        if namb == 0:
            return self._in_or_split(node, lo, hi)
        if node.depth >= self.max_depth or node.fine >= self.max_fine:
            return self._undet(node, "depth_guard")
        if namb > self.cap_fine:
            sp = self._solid_split(node, lo, hi, nst)
            if sp is not None:
                return sp
            if self.stuck_splits and node.depth > 12 and namb > 0.7 * node.npar:
                # bisection no longer shrinks the ambiguity (node close to an occluder edge or vertex, where the
                # relevant coordinate is the pencil angle about the edge): split by a central pencil plane of an
                # ambiguous receiver instead
                kids = self._stuck_split(node, nst, comps, namb)
                if kids is not None:
                    return kids
            kids = self._bisect(node, lo, hi, nst)
            for ch in kids:
                ch.npar = namb
            if self.multires and p.lb > 0:
                for ch in kids:
                    ch.nbm = node_nbm
            return kids
        return self._fine(node, nst, namb, comps)

    def _add_thint(self, X, Y, t):
        h = self.thint
        for k in range(self.ntint):
            if h[k, 1] == X and h[k, 2] == Y and h[k, 3] == t:
                return
        if self.ntint < len(h):
            self.ntint += 1
        h[1:self.ntint] = h[:self.ntint - 1].copy(); h[0] = (0, X, Y, t)

    def _add_hint(self, X, Y):
        h = self.hint
        for k in range(self.nhint):
            if h[k, 0] == X and h[k, 1] == Y:
                return
        if self.nhint < len(h):
            self.nhint += 1
        h[1:self.nhint] = h[:self.nhint - 1].copy(); h[0, 0] = X; h[0, 1] = Y

    # ------------------------------------------------------------------ geometry helpers
    def _side(self, s, pid):
        """Exact position of the polytope in slot s w.r.t. plane pid (SoS signs of its vertices, certify.poly):
        -1 inside {h <= 0}, +1 inside {h >= 0}, 0 straddling."""
        A = self.sl.a; nv = int(A["nV"][s]); neg = pos = 0
        for v in range(nv):
            p, q, r = (int(x) for x in A["Vp"][s, v])
            if pid in (p, q, r):
                continue
            g = PY.sos_sign(self.PLs.PL, pid, p, q, r, A["Vm"][s, v], A["Vr"][s, v])
            if g > 0:
                pos += 1
            else:
                neg += 1
        if pos == 0:
            return -1
        if neg == 0:
            return 1
        return 0


    def _verts(self, s):
        a = self.sl.a; nv = int(a["nV"][s])
        return a["Vm"][s, :nv], a["Vr"][s, :nv], nv

    def _bbox(self, s):
        """Rigorous bounding box of the exact polytope (vertex enclosures, outward rounding)."""
        Vm, Vr, nv = self._verts(s)
        return np.nextafter(np.min(Vm - Vr, 0), -np.inf), np.nextafter(np.max(Vm + Vr, 0), np.inf)

    # ------------------------------------------------------------------ main loop
    def run(self, time_limit=None):
        p = self.p; t0 = time.perf_counter()
        PLs = self.PLs
        s = self.sl.get()
        ax = np.array([[PLs.axis_plane(k, 0.0), PLs.axis_plane(k, 1.0)] for k in range(3)], np.int64)
        PY.set_box(PLs.PL, s, ax, *self.sl.args())
        self.root_axes = ax
        root = Node(s, p.root_starts if self.mode == "tree" else p.rows, 0, (np.zeros(3), np.ones(3)), 0, self._new_nid(-1, -1))
        stack = [root]
        while stack:
            if time_limit is not None and time.perf_counter() - t0 > time_limit:
                raise TimeoutError
            node = stack.pop()
            self.stats["nodes"] += 1
            if self.progress and self.stats["nodes"] % self.progress == 0:
                import sys
                print(f"[xlf] t={time.perf_counter() - t0:.1f}s nodes={self.stats['nodes']} stack={len(stack)} "
                      f"out={self.stats['out']} in={self.stats['in_']} undet={self.stats['undet']} fine={self.stats['fine']} "
                      f"scan={self.stats['scan_s']:.1f}s recv={self.stats['recv_s']:.1f}s", file=sys.stderr, flush=True)
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
        if node.depth > self.max_depth:
            self.stats["depth_guard"] = self.stats.get("depth_guard", 0) + 1
            return self._undet(node, "tree_depth_fine")
        lo, hi = np.nextafter(np.min(Vm - Vr, 0), -np.inf), np.nextafter(np.max(Vm + Vr, 0), np.inf)
        # inside the closed window box: outside D
        if np.all(lo >= p.wlo) and np.all(hi <= p.whi):
            self._leaf(node, "OUT_WINDOW"); self.stats["out_window"] += 1
            return []
        nc = PR.relevant_comps(Vm, Vr, nv, p.rlo, p.rhi, p.AL, p.AH, p.C, self.comps)
        comps = self.comps[:nc].copy()
        rows = node.rows
        t0 = time.perf_counter()
        # witness hints first: receivers that refuted recent nodes often refute this one (any contradicted receiver is
        # a valid OUT witness, whatever rows the node inherited)
        prep = PR.prep_all(Vm, Vr, nv, comps, nc, p.RM, p.RR, p.K, p.ADJ)
        self.stats["prep_s"] = self.stats.get("prep_s", 0.0) + time.perf_counter() - t0
        self.stats["ncomp_sum"] = self.stats.get("ncomp_sum", 0) + nc; self.stats["nv_sum"] = self.stats.get("nv_sum", 0) + nv
        self.stats["nd_sum"] = self.stats.get("nd_sum", 0) + int(np.sum(np.maximum(prep[0], 0))); self.stats["nl_sum"] = self.stats.get("nl_sum", 0) + int(np.sum(prep[1]))
        if self.mode == "tree":
            return self._scan_tree_node(node, Vm, Vr, nv, lo, hi, comps, nc, prep, t0)
        if self.nhint:
            hf = np.zeros(self.nhint, np.int64)
            code, _ = PR.scan_core(Vm, Vr, nv, self.hint[:self.nhint], self.nhint, p.cls, p.o, p.N, p.L, comps, nc, p.RM,
                                   p.RR, p.K, p.ADJ, self.amb, self.cap_amb, hf, self.wit, 0, *prep)
            self.stats["rows_hint"] += self.nhint
            if code == PR.SC_OUT:
                self.stats["scan_s"] += time.perf_counter() - t0; self.stats["hint_out"] += 1
                self._leaf(node, "OUT", tuple(int(v) for v in self.wit)); self.stats["out"] += 1
                return []
        rowflag = np.zeros(len(rows), np.int64)
        code, namb = PR.scan_core(Vm, Vr, nv, rows, len(rows), p.cls, p.o, p.N, p.L, comps, nc, p.RM, p.RR, p.K, p.ADJ,
                                  self.amb, self.cap_amb, rowflag, self.wit, self.rcap, *prep)
        self.stats["scan_s"] += time.perf_counter() - t0; self.stats["rows_scanned"] += len(rows)
        if code == PR.SC_BUG:
            raise AssertionError(f"inconsistent sure classification at receiver {self.wit.tolist()}")
        if code == PR.SC_OUT:
            self._add_hint(int(self.wit[0]), int(self.wit[1]))
            self._leaf(node, "OUT", tuple(int(v) for v in self.wit)); self.stats["out"] += 1
            return []
        nrows = rows[rowflag == 1]
        if namb == 0:
            return self._in_or_split(node, lo, hi)
        if node.depth >= self.max_depth or node.fine >= self.max_fine:
            return self._undet(node, "rows_depth_fine")
        if namb > self.cap_fine:
            sp = self._solid_split(node, lo, hi, nrows)
            if sp is not None:
                return sp
            return self._bisect(node, lo, hi, nrows)
        return self._fine(node, nrows, namb, comps)

    # ------------------------------------------------------------------ leaves
    def _undet(self, node, why="?"):
        vlo, vhi = PY.volume(node.slot, self.sl.a["Fn"], self.sl.a["Fv"], self.sl.a["Vm"], self.sl.a["Vr"], self.sl.a["nF"])
        self.undet_vol[why] = self.undet_vol.get(why, 0.0) + vhi
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
        # disjoint from the open window box (exact side tests on its six face planes)? else split by a straddling face
        sides = []
        for k in range(3):
            for val, outside in ((p.wlo[k], -1), (p.whi[k], 1)):
                pid = self.PLs.axis_plane(k, val)
                sides.append((k, val, pid, outside, self._side(node.slot, pid)))
        disj = any(sd == outside for (k, val, pid, outside, sd) in sides)
        if not disj:
            if all(sd == -outside for (k, val, pid, outside, sd) in sides):
                self._leaf(node, "OUT_WINDOW"); self.stats["out_window"] += 1
                return []
            for (k, val, pid, outside, sd) in sides:
                if sd == 0:
                    bl = bh = None
                    if node.box is not None:
                        b0, b1 = node.box
                        l0 = b0.copy(); h0 = b1.copy(); h0[k] = val
                        l1 = b0.copy(); h1 = b1.copy(); l1[k] = val
                        bl, bh = (l0, l1), (h0, h1)
                    return self._split(node, pid, bl, bh)
            return self._undet(node, "window_unsplittable")
        Vm, Vr, nv = self._verts(node.slot)
        if not self._clear(Vm, Vr, nv, lo, hi):
            self.stats["clear_fail"] += 1
            if node.depth >= self.max_depth or (hi - lo).max() < 1e-9:
                return self._undet(node, "clear_fail")
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

    def _solid_split(self, node, lo, hi, nrows):
        """A node straddling an occluder's boundary keeps every receiver ambiguous at every scale (part of it is inside
        the solid). OUT_SOLID if proven inside a component: every facet is either <= 0 at every vertex (rigorous range)
        or holds by construction (the node lies in {A <= 0} for a split plane A with {A <= 0} subset {f_i < 0} on the
        unit cube, see _plane_pair). Otherwise split by the most central straddling facet plane not used before on
        this path (binary64 rounding + delta slab; the slab is processed as a node). None if nothing to do."""
        p = self.p
        Vm, Vr, nv = self._verts(node.slot)
        best = None
        for c in range(p.C):
            if np.any(p.AL[c] > hi) or np.any(p.AH[c] < lo):
                continue
            outside = False; inside = True; cand = None
            for i in range(int(p.K[c])):
                if (c, i) in node.inside:
                    continue
                l, h = PR.range_on(p.RM[c, i], p.RR[c, i], Vm, Vr, nv)
                if l > 0.0:
                    outside = True
                    break
                if h > 0.0:
                    inside = False
                    if l < 0.0 and (c, i) not in node.used:
                        sc = min(h, -l)
                        if cand is None or sc > cand[0]:
                            cand = (sc, c, i)
            if outside:
                continue
            if inside:
                by = [(int(i), int(a)) for (cc, i), a in node.inside.items() if cc == c]
                self._leaf(node, "OUT_SOLID", (c, by)); self.stats["out_solid"] = self.stats.get("out_solid", 0) + 1
                return []
            if cand is not None and (best is None or cand[0] > best[0]):
                best = cand
        if best is None:
            return None
        _, c, i = best
        self.stats["solid_split"] = self.stats.get("solid_split", 0) + 1
        A, B = self._plane_pair(p.RM[c, i].copy(), p.RR[c, i].copy())
        used = node.used | {(c, i)}
        out = []
        for ch in self._split(node, A, rows=nrows):
            ch.used = used
            if self.bsp_ref[ch.nid] == 2 * A:                  # {A <= 0}: inside facet i by construction
                ch.inside = dict(node.inside); ch.inside[(c, i)] = A
                out.append(ch)
                continue
            for g in self._split(ch, B, rows=nrows):
                g.used = used
                out.append(g)                                  # outer piece and slab are processed as nodes
        return out

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
        ns = min(namb, self.cap_amb, self.fine_sample)
        cap = 4 * PR.MH * max(ns, 1)
        HM = np.empty((cap, 4)); HR = np.empty((cap, 4)); HK = np.empty((cap, 4), np.int64); HS = np.empty(cap, np.int64)
        HPHI = np.empty(cap); HREC = np.empty(cap, np.int64)
        code, n, nconv, fnc = PR.fine_collect(Vm, Vr, nv, self.amb, ns, p.cls, p.o, p.N, comps,
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
            # every receiver left ambiguous by the collective scan is consistent under the exact source-form test (and
            # all others were consistent in the scan): the node is fully consistent
            if namb <= ns:
                lo, hi = self._bbox(s)
                return self._in_or_split(node, lo, hi)
            lo, hi = self._bbox(s)
            return self._bisect(node, lo, hi, nrows)
            return self._undet(node, "fine_nothing")
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

    def _stuck_split(self, node, nrows, comps, namb):
        """Median pencil split. Near an occluder edge e the receivers' constraints are planes through line(e) and the
        relevant coordinate is the pencil angle about e: a node touching line(e) keeps a wide angular range however
        small it is in x, y, z. Sample the ambiguous receivers, take the edge carrying most straddling pair constraints
        and split by the constraint plane of median angle (halves the ambiguity of that edge per split)."""
        p = self.p
        m = min(namb, self.cap_amb)
        idx = np.unique(np.linspace(0, m - 1, min(m, 256)).astype(np.int64))
        groups = {}
        for a in idx:
            X, Y, t = (int(v) for v in self.amb[a, :3])
            st, nh, obs = self._recv(node.slot, X, Y, t, comps)
            if st not in (PR.R_CONV, PR.R_NONCONV):
                continue
            rx = (p.o + np.array([X, Y, t]) + 0.5) / p.N
            for h in range(nh):
                c, i, j, kind = (int(v) for v in self.hs_key[h])
                if kind != 1:
                    continue
                fi = float(p.RM[c, i, :3] @ rx + p.RM[c, i, 3]); fj = float(p.RM[c, j, :3] @ rx + p.RM[c, j, 3])
                # plane = -f_j(r) f_i + f_i(r) f_j: angle of (alpha, beta) = (-f_j(r), f_i(r)) modulo pi (key order free)
                a_, b_ = (-fj, fi) if i < j else (fi, -fj)
                psi = float(np.arctan2(b_, a_)) % np.pi
                groups.setdefault((c, min(i, j), max(i, j)), []).append((psi, self.hs_m[h].copy(), self.hs_r[h].copy()))
        if not groups:
            return None
        key = max(groups, key=lambda k: len(groups[k]))
        lst = sorted(groups[key], key=lambda e: e[0])
        psi, hm, hr = lst[len(lst) // 2]
        self.stats["stuck_split"] = self.stats.get("stuck_split", 0) + 1
        A, B = self._plane_pair(hm, hr)
        out = []
        for ch in self._split(node, A, rows=nrows):
            ch.npar = namb; ch.stuck = True; ch.fine = node.fine
            if self.bsp_ref[ch.nid] == 2 * A + 1:
                for g in self._split(ch, B, rows=nrows):
                    g.npar = namb; g.stuck = True; g.fine = node.fine
                    out.append(g)
            else:
                out.append(ch)
        return out

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
                self._undet(pc, "slab")
            else:
                rest.append(pc)
        return rest

    def _cell_volumes(self, s0, is_in):
        """Per-cell volume enclosures of the polytope in slot s0 by recursive clipping at cell boundaries (exact
        partition); IN adds [lo, hi], UNDET adds hi. The cell range of every piece is tracked explicitly: it starts as
        the range of the rigorous bounding box and every split halves it, so a piece reaching a single-cell range lies
        in that closed cell (bounding box + the split half-spaces on its path). Slot s0 is left untouched."""
        a = self.sl.args(); A = self.sl.a
        lo, hi = self._bbox(s0)
        ca = np.clip(np.floor(lo * NCELL).astype(np.int64), 0, NCELL - 1)
        cb = np.clip(np.floor(hi * NCELL).astype(np.int64), 0, NCELL - 1)
        s = self.sl.get(); PY.copy_slot(s0, s, *a)
        stack = [(s, ca, cb)]
        while stack:
            s, ca, cb = stack.pop()
            if A["nV"][s] == 0:
                self.sl.put(s); continue
            span = cb - ca
            if np.all(span == 0):
                vlo, vhi = PY.volume(s, A["Fn"], A["Fv"], A["Vm"], A["Vr"], A["nF"])
                cid = int((ca[0] * NCELL + ca[1]) * NCELL + ca[2])
                if is_in:
                    self.cell_lo[cid] = self.cell_lo.get(cid, 0.0) + vlo
                self.cell_hi[cid] = self.cell_hi.get(cid, 0.0) + vhi
                self.sl.put(s); continue
            k = int(np.argmax(span)); m = int(ca[k] + (span[k] + 1) // 2); val = m / NCELL
            pid = self.PLs.axis_plane(k, val)
            for side in (0, 1):
                t = self.sl.get(); PY.copy_slot(s, t, *a)
                st = PY.clip(self.PLs.PL, t, 2 * pid + side, *a)
                if st == PY.OVERFLOW:
                    raise RuntimeError("polytope overflow")
                if st == PY.EMPTY:
                    self.sl.put(t); continue
                na, nb = ca.copy(), cb.copy()
                if side == 0:
                    nb[k] = m - 1            # {x_k <= m / NCELL}
                else:
                    na[k] = m                # {x_k >= m / NCELL}
                stack.append((t, na, nb))
            self.sl.put(s)

    def certificate(self):
        """Partition certificate for xlf.check: plane table, BSP (parent, signed plane ref), leaf claims."""
        kinds = []; nodes = []; wits = []
        for nid, kind, data in self.leaves:
            kinds.append(kind); nodes.append(nid)
            wits.append(list(data)[:4] if kind == "OUT" else ([int(data[0]), 0, 0, 0] if kind == "OUT_SOLID" else [0, 0, 0, 0]))
        solid_by = {int(nid): data[1] for nid, kind, data in self.leaves if kind == "OUT_SOLID"}
        return dict(PL=self.PLs.PL[:self.PLs.n].copy(), parent=np.array(self.bsp_parent, np.int64),
                    ref=np.array(self.bsp_ref, np.int64), leaf_kind=np.array(kinds), leaf_node=np.array(nodes, np.int64),
                    leaf_wit=np.array(wits, np.int64).reshape(-1, 4), root_axes=self.root_axes, solid_by=solid_by)

    # ------------------------------------------------------------------ result
    def result(self):
        cells = sorted(set(self.cell_lo) | set(self.cell_hi))
        lo = np.array([self.cell_lo.get(c, 0.0) for c in cells]); hi = np.array([self.cell_hi.get(c, 0.0) for c in cells])
        return np.array(cells, np.int64), lo, hi, float(self.pool)
