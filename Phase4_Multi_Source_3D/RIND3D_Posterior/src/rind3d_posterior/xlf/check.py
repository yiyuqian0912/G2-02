"""XLF replay checker: re-proves an XLF partition certificate without the solver's search.

Certificate (from Solver.certificate()): the plane table PL (binary64 rows, exact rationals), the BSP (parent id and
signed plane reference of every node; node 0 = the box [0, 1]^3), and one claim per leaf:
  OUT (receiver X, Y, t)   the receiver's observation is contradicted for every source of the leaf;
  OUT_WINDOW               the leaf lies in the closed window box (outside D);
  IN                       every FREE receiver is consistent for every source of the leaf, and the leaf is inside D;
  UNDET / EMPTY            no claim (volume upper bound / nothing).
The checker rebuilds every node polytope from its parent by one exact clip (certify.poly), re-proves each claim
from scratch with the trusted predicates:
  * OUT: scan_tree started from the single receiver (no inheritance);
  * IN: scan_tree over the WHOLE window from its top block (no inheritance from ancestors), with the exact source-form
    re-test of every receiver the collective test leaves ambiguous; window side tests; clearance;
then integrates the IN and UNDET leaves per cell and bounds TV exactly (certify.tvb). The checker shares only the
trusted kernel and predicates with the solver; plane choice, inheritance, hints and caps are not used."""
import time

import numpy as np

from ..certify import poly as PY, tvb
from . import pred as PR
from .solve import Slots, NCELL, EPS_CLEAR


def check(prob, cert, rcap=1 << 20):
    p = prob; par = cert["parent"]; ref = cert["ref"]
    # plane table, extended (never modified) with the checker's own cell planes: SoS orders planes by id, so appending
    # planes changes no sign involving the certificate's planes
    PL = np.zeros((len(cert["PL"]) + 4096, 4)); PL[:len(cert["PL"])] = cert["PL"]
    PLn = [len(cert["PL"])]
    kind = cert["leaf_kind"]; lnode = cert["leaf_node"]; lwit = cert["leaf_wit"]
    t0 = time.perf_counter()
    nN = len(par)
    children = [[] for _ in range(nN)]
    for i in range(1, nN):
        children[par[i]].append(i)
    leaf_of = {int(n): k for k, n in enumerate(lnode)}
    sl = Slots(1 << 12); a = sl.args(); A = sl.a
    # root box from the axis planes x_k = 0, x_k = 1 (stored first)
    ax = cert["root_axes"]
    s = sl.get(); PY.set_box(PL, s, ax, *a)
    comps = np.zeros(max(p.C, 1), np.int64)
    amb = np.zeros((1 << 16, 4), np.int64); wit = np.zeros(4, np.int64)
    bm = np.zeros(p.L ** 3, np.int64)
    top = np.array([[p.q, 0, 0, 0]], np.int64)
    cell_lo = {}; cell_hi = {}; pool = 0.0
    bad = []; nchk = dict(OUT=0, IN=0, OUT_WINDOW=0, OUT_SOLID=0, UNDET=0, EMPTY=0)
    stack = [(0, s)]
    while stack:
        nid, s = stack.pop()
        if nid in leaf_of:
            k = leaf_of[nid]; kd = str(kind[k])
            nchk[kd if kd in nchk else "UNDET"] = nchk.get(kd, 0) + 1
            nv = int(A["nV"][s]); Vm = A["Vm"][s, :nv]; Vr = A["Vr"][s, :nv]
            if kd == "EMPTY":
                if nv != 0:
                    bad.append((nid, "EMPTY leaf is not empty"))
            elif nv == 0:
                pass
            elif kd == "OUT":
                nc = PR.relevant_comps(Vm, Vr, nv, p.rlo, p.rhi, p.AL, p.AH, p.C, comps)
                cc = comps[:nc].copy(); prep = PR.prep_all(Vm, Vr, nv, cc, nc, p.RM, p.RR, p.K, p.ADJ)
                st = np.array([[0, *lwit[k][:3]]], np.int64)
                code, _ = PR.scan_tree(Vm, Vr, nv, st, 1, 0, p.pyr, p.poff, p.q, p.cls, p.o, p.N, p.L, cc, nc, p.RM, p.RR,
                                       p.K, p.ADJ, amb, len(amb), bm, wit, 1, *prep)
                if code != PR.SC_OUT:
                    bad.append((nid, f"OUT witness {lwit[k][:3].tolist()} not re-proven"))
            elif kd == "OUT_SOLID":
                c = int(lwit[k][0]); by = dict(cert.get("solid_by", {}).get(nid, []))
                ok = True
                for i in range(int(p.K[c])):
                    if PR.range_on(p.RM[c, i], p.RR[c, i], Vm, Vr, nv)[1] <= 0.0:
                        continue
                    a = by.get(i)                     # by construction: node in {A <= 0} and {A <= 0} in {f_i < 0}
                    if a is None or not _inner_plane_ok(p, PL, A, s, a, c, i):
                        ok = False
                        break
                if not ok:
                    bad.append((nid, "OUT_SOLID leaf not inside its component"))
            elif kd == "OUT_WINDOW":
                if not _inside_window(p, PL, A, s):
                    bad.append((nid, "OUT_WINDOW leaf not inside the window box"))
            elif kd in ("IN", "IN_BOX"):
                nc = PR.relevant_comps(Vm, Vr, nv, p.rlo, p.rhi, p.AL, p.AH, p.C, comps)
                cc = comps[:nc].copy(); prep = PR.prep_all(Vm, Vr, nv, cc, nc, p.RM, p.RR, p.K, p.ADJ)
                bmark = np.zeros((p.L >> p.lb) ** 3, np.int64)
                code, namb = PR.scan_tree(Vm, Vr, nv, top, 1, p.lb, p.pyr, p.poff, p.q, p.cls, p.o, p.N, p.L, cc, nc,
                                          p.RM, p.RR, p.K, p.ADJ, amb, len(amb), bmark, wit, rcap, *prep)
                ok = code == PR.SC_OK and namb == 0 and _outside_window(p, PL, A, s) and _clear(p, Vm, Vr, nv)
                if not ok:
                    bad.append((nid, f"IN not re-proven (code {code}, ambiguous {namb})"))
                else:
                    _cells(PL, PLn, sl, s, True, cell_lo, cell_hi)
            else:                                            # UNDET
                pool_add = _cells(PL, PLn, sl, s, False, cell_lo, cell_hi)
                pool += pool_add
            sl.put(s)
            continue
        kids = children[nid]
        for ch in kids:
            t = sl.get(); PY.copy_slot(s, t, *a)
            st = PY.clip(PL, t, int(ref[ch]), *a)
            if st == PY.OVERFLOW:
                bad.append((ch, "overflow")); sl.put(t); continue
            stack.append((ch, t))
        if not kids:
            bad.append((nid, "internal node without children and without a leaf claim"))
        sl.put(s)
    cells = np.array(sorted(set(cell_lo) | set(cell_hi)), np.int64)
    lo = np.array([cell_lo.get(int(c), 0.0) for c in cells]); hi = np.array([cell_hi.get(int(c), 0.0) for c in cells])
    b, Lt = tvb.tv_bound_exact(cells, lo, hi, pool)
    return dict(ok=not bad, failures=bad[:20], n_fail=len(bad), checked=nchk, tv=float(b) if b is not None else None,
                Lt=float(np.sum(lo)), Ht=float(np.sum(hi) + pool), cells=int(np.sum(lo > 0)), seconds=time.perf_counter() - t0,
                cert_cells=cells, cert_lo=lo, cert_hi=hi, pool=pool)


def _inner_plane_ok(p, PL, A, s, a, c, i):
    """Plane a has {a <= 0} subset {f_i < 0} on the unit cube (a = f_i / sc shifted by more than the coefficient
    error E, as built by Solver._plane_pair), and the polytope in slot s lies in {a <= 0} (exact side test)."""
    cm = p.RM[c, i]; cr = p.RR[c, i]; sc = float(np.max(np.abs(cm)))
    h = cm / sc
    if not np.all(PL[a, :3] == h[:3]):
        return False
    E = float(np.sum(np.abs(cm - sc * h) + cr)) * (1 + 2 ** -40) / sc
    if not (PL[a, 3] - h[3] > E * (1 + 2 ** -30)):
        return False
    return _side(PL, A, s, a) == -1


def _side(PL, A, s, pid):
    nv = int(A["nV"][s]); neg = pos = 0
    for v in range(nv):
        pp, qq, rr = (int(x) for x in A["Vp"][s, v])
        if pid in (pp, qq, rr):
            continue
        g = PY.sos_sign(PL, pid, pp, qq, rr, A["Vm"][s, v], A["Vr"][s, v])
        if g > 0:
            pos += 1
        else:
            neg += 1
    return -1 if pos == 0 else (1 if neg == 0 else 0)


def _axis_id(PL, k, val):
    """Plane id of x_k = val in the certificate's plane table (axis planes are stored by the solver)."""
    m = (PL[:, k] == 1.0) & (PL[:, 3] == -val) & (np.sum(PL[:, :3] != 0, 1) == 1)
    i = np.flatnonzero(m)
    return int(i[0]) if len(i) else -1


def _outside_window(p, PL, A, s):
    Vm = A["Vm"][s, :int(A["nV"][s])]; Vr = A["Vr"][s, :int(A["nV"][s])]
    lo = np.nextafter(np.min(Vm - Vr, 0), -np.inf); hi = np.nextafter(np.max(Vm + Vr, 0), np.inf)
    if np.any(hi <= p.wlo) or np.any(lo >= p.whi):
        return True
    for k in range(3):
        for val, outside in ((p.wlo[k], -1), (p.whi[k], 1)):
            pid = _axis_id(PL, k, val)
            if pid >= 0 and _side(PL, A, s, pid) == outside:
                return True
    return False


def _inside_window(p, PL, A, s):
    Vm = A["Vm"][s, :int(A["nV"][s])]; Vr = A["Vr"][s, :int(A["nV"][s])]
    lo = np.nextafter(np.min(Vm - Vr, 0), -np.inf); hi = np.nextafter(np.max(Vm + Vr, 0), np.inf)
    if np.all(lo >= p.wlo) and np.all(hi <= p.whi):
        return True
    for k in range(3):
        for val, outside in ((p.wlo[k], -1), (p.whi[k], 1)):
            pid = _axis_id(PL, k, val)
            if pid < 0 or _side(PL, A, s, pid) != -outside:
                return False
    return True


def _clear(p, Vm, Vr, nv):
    lo = np.nextafter(np.min(Vm - Vr, 0), -np.inf); hi = np.nextafter(np.max(Vm + Vr, 0), np.inf)
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


def _cells(PL, PLn, sl, s0, is_in, cell_lo, cell_hi):
    """Per-cell volume enclosures by recursive clipping at cell boundaries (explicit cell ranges). Returns 0 (all mass
    attributed to cells)."""
    a = sl.args(); A = sl.a
    Vm = A["Vm"][s0, :int(A["nV"][s0])]; Vr = A["Vr"][s0, :int(A["nV"][s0])]
    lo = np.nextafter(np.min(Vm - Vr, 0), -np.inf); hi = np.nextafter(np.max(Vm + Vr, 0), np.inf)
    ca = np.clip(np.floor(lo * NCELL).astype(np.int64), 0, NCELL - 1)
    cb = np.clip(np.floor(hi * NCELL).astype(np.int64), 0, NCELL - 1)
    s = sl.get(); PY.copy_slot(s0, s, *a)
    stack = [(s, ca, cb)]
    while stack:
        s, ca, cb = stack.pop()
        if A["nV"][s] == 0:
            sl.put(s); continue
        span = cb - ca
        if np.all(span == 0):
            vlo, vhi = PY.volume(s, A["Fn"], A["Fv"], A["Vm"], A["Vr"], A["nF"])
            cid = int((ca[0] * NCELL + ca[1]) * NCELL + ca[2])
            if is_in:
                cell_lo[cid] = cell_lo.get(cid, 0.0) + vlo
            cell_hi[cid] = cell_hi.get(cid, 0.0) + vhi
            sl.put(s); continue
        k = int(np.argmax(span)); m = int(ca[k] + (span[k] + 1) // 2); val = m / NCELL
        pid = _axis_id(PL[:PLn[0]], k, val)
        if pid < 0:
            if PLn[0] >= len(PL):
                raise ValueError("checker plane table full")
            PL[PLn[0]] = 0.0; PL[PLn[0], k] = 1.0; PL[PLn[0], 3] = -val; pid = PLn[0]; PLn[0] += 1
        for side in (0, 1):
            t = sl.get(); PY.copy_slot(s, t, *a)
            st = PY.clip(PL, t, 2 * pid + side, *a)
            if st == PY.EMPTY:
                sl.put(t); continue
            na, nb = ca.copy(), cb.copy()
            if side == 0:
                nb[k] = m - 1
            else:
                na[k] = m
            stack.append((t, na, nb))
        sl.put(s)
    return 0.0
