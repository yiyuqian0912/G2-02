"""Trusted correct-by-construction kernel, version 2 (Claude), evolved from Astra's kernel.py.

The kernel owns one problem (canonical geometry + one FREE-channel observation) and a
set of live source polytopes ("pieces"). Only kernel methods evaluate physics, and every
method is one atomic transition on an immutable State. The untrusted search can choose
handles, receivers and cut planes; it cannot supply a classification, a volume or a
credit.

Invariant (frontier accounting), for the true feasible set F (up to Lebesgue-null sets):
  (I1) F is contained in  U{IN pieces}  union  U{live pieces};
  (I2) every IN piece is contained in F; IN pieces have pairwise disjoint interiors;
  (I3) lower[c] <= vol(F & cell c)   and   upper_in[c] >= vol(IN & cell c)   (units 2^-160);
  (I4) for every live piece, every FREE receiver NOT in its pending obligation set is
       reproduced by every source of the piece (proved on an ancestor; inherited).
Each transition below states why it preserves (I1)-(I4). The certificate is computed
from (I1)-(I3) only, on the decoded output bytes (kernel.wire_bound).

Transitions
  exclude_window()        drop the closed window box and closed obstacles (illegal)  (I1)
  condition_pair(d, a)    keep  piece & U_c C_c(d) \\ U_c C_c(a)  (exact cones)      (I1)
  split(h, row)           replace a piece by its two closed halves                   (I1),(I4) by inclusion
  screen(h)               classify the piece's pending receivers:
                            contradiction            -> drop the piece             (I1)
                            none pending, legal part -> IN credit (exact grid)     (I2),(I3)
                            some pending             -> pending := ambiguous ones  (I4)
  enforce(h, receivers)   exact cone cuts; children's pending excludes them          (I1),(I4)
  snapshot()              payload + exact upper TV bound on the decoded payload
  release(tolerance)      the kernel's own CERTIFIED / INCOMPLETE / ERROR decision

Release discipline: the kernel records every (payload, bound) it computes and only
release() turns the best of them into a CERTIFIED result; the search never decides a
status. Any ArithmeticError inside a transition (a failed exactness guard, a structural
check, an overflow) taints the kernel: no further transitions, release() returns ERROR.
Arrays stored in pieces are read-only and the search receives read-only views and
polytope copies, so a (buggy, non-malicious) search cannot alter kernel evidence.
"""
import copy
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import wraps
from time import perf_counter
import weakref

import numpy as np

import inputs  # noqa: F401  (installs pinned research paths)
from envelopes import clearance_partition
from ledger import decode, encode
from poly import Poly, form, grid_pieces, primitive
from raster import encoded
from screen2 import check_ranges, scan_collect, scan_ids
from poly import within
from solver import Budget, Limit, apply_pair, cone, subtract_many
from kernel import SCALE, ceil_mass, cell_id, single_cell, wire_bound
import polycheck

COLLECT_CAP = 1 << 21  # largest residual obligation list stored explicitly

try:  # optional exact GMP helper (native/build.sh); CBC_NATIVE=0 forces the Python path
    import os as _os
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent / "native"))
    NATIVE = None if _os.environ.get("CBC_NATIVE") == "0" else __import__("cbc_cells")
except ImportError:
    NATIVE = None
CELL_UNITS = SCALE // (128 ** 3)  # exact volume of one output cell in units of 2^-160
CHECK_LIVE = True  # structural certificate (polycheck) for every live piece charged in a snapshot


def _readonly(a, dtype):
    """Kernel-owned copy that nobody can write through."""
    out = np.array(a, dtype=dtype, copy=True)
    out.flags.writeable = False
    return out


def _view(a):
    if a is None:
        return None
    v = a.view()
    v.flags.writeable = False
    return v


@dataclass(frozen=True)
class Release:
    status: str          # CERTIFIED, INCOMPLETE or ERROR
    payload: object      # bytes iff CERTIFIED
    bound: object        # Fraction: exact upper TV bound of payload (best recorded), or None
    tolerance: Fraction
    reason: str


def _transition(fn):
    """Refuse work after a failed guard; taint the kernel on any ArithmeticError."""
    @wraps(fn)
    def run(self, *args, **kwargs):
        if self._tainted is not None:
            raise RuntimeError("kernel tainted by a failed guard: " + self._tainted)
        try:
            return fn(self, *args, **kwargs)
        except ArithmeticError as exc:
            self._tainted = f"{fn.__name__}: {type(exc).__name__}: {exc}"
            raise
    return run


def cell_box_rows(i, j, k, n=128):
    rows = []
    for axis, v in enumerate((i, j, k)):
        e = [0, 0, 0]
        e[axis] = n
        rows.append((-e[0], -e[1], -e[2], v))
        rows.append((e[0], e[1], e[2], -(v + 1)))
    return rows


def encode_int(lower):
    """Integer-only equivalent of ledger.encode (same bytes; property-tested).

    Quotas TICKS*m/total share one denominator, so the fractional parts compare as the
    integer remainders of divmod(TICKS*m, total); ties break by cell id as in ledger.encode.
    """
    from ledger import RECORD, TICKS
    total = sum(lower.values())
    if total <= 0 or any(v <= 0 for v in lower.values()):
        raise ValueError("positive lower mass required")
    qr = {c: divmod(TICKS * m, total) for c, m in lower.items()}
    ticks = {c: q for c, (q, r) in qr.items()}
    missing = TICKS - sum(ticks.values())
    for c in sorted(qr, key=lambda c: (-qr[c][1], c))[:missing]:
        ticks[c] += 1
    return b"".join(RECORD.pack(c, n / TICKS) for c, n in sorted(ticks.items()) if n)


def remap_masks(masks, rel):
    """Local bit c (position in rel) -> global body bit rel[c]."""
    out = np.zeros(len(masks), np.uint64)
    for c, g in enumerate(rel):
        out |= ((masks >> np.uint64(c)) & np.uint64(1)) << np.uint64(g)
    return out


def encoded_subset(obs, p, bodies):
    """raster.encoded restricted to the listed bodies (same rows, same directed rounding)."""
    from raster import fixed
    from envelopes import Envelope
    from penumbra import directions
    if getattr(obs, "centroid_anchor", False):
        # Hint only (never a decision): the binary64 vertex centroid snapped to 2^-60.
        fv = p.fvertices
        anchor = primitive(tuple(int(round(sum(v[i] for v in fv) / len(fv) * 2.0 ** 60)) for i in range(3))
                           + (1 << 60,))
    else:
        anchor = primitive(tuple(map(sum, zip(*p.vertices))))
    rows, offsets = [], [0]
    if getattr(obs, "use_support", False) and not hasattr(obs, "obstacles"):
        obs.obstacles = [within(Poly.cube(), b) for b in obs.bodies]
        if any(k is None for k in obs.obstacles):
            raise ValueError("expected full-dimensional body in world")
    for j in bodies:
        b = obs.bodies[j]
        e = Envelope(b, p.vertices)
        if getattr(obs, "use_support", False):
            e.outer = e.outer + directions(obs.obstacles[j], p)
        hint = cone(b, anchor) if any(form(r, anchor) > 0 for r in b) else []
        for planes, up in ((e.inner, True), (e.outer, False), (hint, False)):
            rows.extend(fixed(planes, up))
            offsets.append(len(rows))
    return np.array(rows, np.int64).reshape(-1, 4), np.array(offsets, np.int32)


def facet_rows(p):
    """Outward facet rows (p on the <= 0 side) from the face cycles, validated exactly.

    Each row is checked against every vertex (all <= 0) and must vanish on its own face;
    a failure raises instead of producing a possibly wrong inside test.
    """
    out = []
    for face in p.faces:
        a = face[0]
        def d(b):
            return tuple(b[i] * a[3] - a[i] * b[3] for i in range(3))
        u = d(face[1])
        n = None
        for b in face[2:]:
            v = d(b)
            n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
            if any(n):
                break
        if n is None or not any(n):
            continue  # degenerate (collinear) face cycle carries no plane
        row = primitive((*(x * a[3] for x in n), -sum(n[i] * a[i] for i in range(3))))
        vals = [form(row, q) for q in p.vertices]
        if max(vals) > 0:
            row = tuple(-x for x in row)
            vals = [-x for x in vals]
        if max(vals) > 0 or any(form(row, q) != 0 for q in face):
            raise ArithmeticError("inconsistent face structure")
        out.append(row)
    if len(out) < 4:
        raise ArithmeticError("polytope with fewer than four facets")
    return out


@dataclass(frozen=True)
class Piece:
    poly: Poly
    pending: object = None        # None: all FREE receivers except `done`; else sorted int64 ids
    done: frozenset = frozenset()  # receivers enforced exactly while pending is None
    deferred: bool = False         # visibility proved; clearance shell only (never rescreened)
    masks: object = None           # uint64 per pending id: bodies whose possible shadow contains it


@dataclass(frozen=True)
class State:
    live: dict
    lower: Counter
    upper_in: Counter
    next_id: int
    outside_window: bool


class Kernel:
    def __init__(self, observation, budget):
        self.obs = observation
        self.budget = budget
        if observation.denom != 2048 or observation.side not in (64, 128, 256):
            raise ValueError("kernel verified for N=1024 and v3 window sizes only")
        if any(o < 0 or o + observation.side > 1024 for o in observation.origin):
            raise ValueError("window outside the world")
        self._state = State({0: Piece(Poly.cube())}, Counter(), Counter(), 1, False)
        self.stats = Counter()
        self._cones = {}
        self.obs.scancls = np.ascontiguousarray(self.obs.cls.T)
        # Exactly validated separating support planes tighten every possible shadow
        # (penumbra.directions: float proposals, exact integer validation; see design doc).
        self.obs.use_support = True
        self._origin = np.asarray(self.obs.origin, np.int64)
        self._tainted = None
        self._best = None  # (bound, payload) with the smallest bound computed so far
        self._checked = weakref.WeakSet()  # polytopes that passed polycheck.check

    # ---------------------------------------------------------------- read-only views
    def handles(self):
        return tuple(self._state.live)

    def is_open(self, h):
        p = self._state.live.get(h)
        return p is not None and not p.deferred

    def open_handles(self):
        return [h for h, p in self._state.live.items() if not p.deferred]

    def volume(self, h):
        return self._state.live[h].poly.volume

    def pending_count(self, h):
        p = self._state.live[h]
        return None if p.pending is None else len(p.pending)

    def poly(self, h):
        """A copy of the polytope, for proposal heuristics only (the kernel keeps its own)."""
        return copy.copy(self._state.live[h].poly)

    def contact(self, h):
        """Hint: does a vertex of the piece lie in a closed obstacle? (search hint, untrusted use)"""
        p = self._state.live[h].poly
        return any(all(form(r, v) <= 0 for r in body) for body in self.obs.bodies for v in p.vertices)

    @property
    def positive(self):
        return bool(self._state.lower)

    def lower_total(self):
        return Fraction(sum(self._state.lower.values()), SCALE)

    # ---------------------------------------------------------------- commit
    def _commit(self, remove, children, lower=None, upper=None, exclude=False):
        old = self._state
        self.budget.check(len(old.live) - len(remove) + len(children))
        live = old.live.copy()
        for key in remove:
            del live[key]  # a stale or duplicate handle raises KeyError before any change
        new_ids = list(range(old.next_id, old.next_id + len(children)))
        live.update(zip(new_ids, children))
        lo = old.lower if lower is None else old.lower + lower
        hi = old.upper_in if upper is None else old.upper_in + upper
        self._state = State(live, lo, hi, old.next_id + len(children), old.outside_window or exclude)
        self.stats["transitions"] += 1
        self.stats["peak_live"] = max(self.stats["peak_live"], len(live))
        return new_ids

    def _receiver(self, pos, label=None):
        if len(pos) != 3 or any(type(i) is not int or not 0 <= i < self.obs.side for i in pos):
            raise ValueError("invalid receiver coordinate")
        c = int(self.obs.cls[pos])
        if c not in (0, 1) or (label is not None and c != label):
            raise ValueError("receiver evidence contradicts observation")
        return self.obs.world(pos)

    def _idx_pos(self, idx):
        L = self.obs.side
        if type(idx) is not int or not 0 <= idx < L ** 3:
            raise ValueError("invalid receiver id")
        z, yx = divmod(idx, L * L)
        y, x = divmod(yx, L)
        return (x, y, z)

    # ---------------------------------------------------------------- transitions
    @_transition
    def exclude_window(self):
        """Drop the closed window box and every closed obstacle from all live pieces.

        Sound: a legal source is outside the closed window and at distance > 2^-26 from
        every obstacle, so F misses both sets; exact polytope differences keep the rest.
        """
        if self._state.outside_window:
            return
        children = []
        solids = [tuple(b) for b in self.obs.bodies]
        for piece in self._state.live.values():
            parts = self.obs.exclude_window([piece.poly], self.budget)
            parts = subtract_many(parts, solids, self.budget)
            children.extend(Piece(q, piece.pending, piece.done, piece.deferred, piece.masks) for q in parts)
            self.budget.check(len(children))
        self._commit(self.handles(), children, exclude=True)

    @_transition
    def condition_pair(self, dark, lit):
        t_pair = perf_counter()
        d, a = self._receiver(dark, 0), self._receiver(lit, 1)
        cones_d = [cone(b, d) for b in self.obs.bodies]
        cones_a = [cone(b, a) for b in self.obs.bodies]
        children = []
        for piece in self._state.live.values():
            children.extend(Piece(q, piece.pending, piece.done, piece.deferred, piece.masks)
                            for q in apply_pair(piece.poly, cones_d, cones_a, self.budget))
            self.budget.check(len(children))
        self._commit(self.handles(), children)
        self.stats["pairs"] += 1
        self.stats["pair_seconds"] += perf_counter() - t_pair

    @_transition
    def split(self, h, row):
        """Any integer affine row. Both closed halves stay live; null overlap on the plane."""
        if len(row) != 4 or not all(type(v) is int for v in row) or not any(row[:3]):
            raise ValueError("invalid split plane")
        piece = self._state.live[h]
        neg, pos = piece.poly.split(tuple(row))
        if neg is None or pos is None:
            return [h]  # the plane misses the interior: no transition
        if neg.volume + pos.volume != piece.poly.volume:
            raise ArithmeticError("split failed exact volume conservation")  # face-structure guard
        self.stats["splits"] += 1
        return self._commit((h,), [Piece(neg, piece.pending, piece.done, piece.deferred, piece.masks),
                                    Piece(pos, piece.pending, piece.done, piece.deferred, piece.masks)])

    @_transition
    def screen(self, h):
        """Classify pending receivers on the piece. Returns a dict; the piece may be consumed."""
        if not self._state.outside_window:
            raise ValueError("closed window has not been excluded")
        piece = self._state.live[h]
        if piece.deferred:
            raise ValueError("deferred clearance shell")
        p = piece.poly
        t0 = perf_counter()
        nb = len(self.obs.bodies)
        rel = None
        if piece.pending is not None and piece.masks is not None and nb <= 64 and len(piece.pending):
            # Only bodies in some residual receiver's mask can affect a residual decision:
            # an excluded body's possible shadow over an ancestor (hence over this subset)
            # misses the receiver, so it can neither block nor guarantee a block there.
            union = int(np.bitwise_or.reduce(piece.masks))
            rel = [c for c in range(nb) if union >> c & 1]
        t_enc = perf_counter()
        rows, offs = encoded_subset(self.obs, p, range(nb) if rel is None else rel)
        self.stats["envelope_seconds"] += perf_counter() - t_enc
        check_ranges(rows, self.obs.origin, self.obs.side, self.obs.denom)
        if piece.pending is None:
            skipped = np.array(sorted(piece.done), np.int64)
            n, ids, wrong, bnd, masks = scan_collect(self.obs.scancls, self._origin, self.obs.denom, rows, offs,
                                                     skipped, COLLECT_CAP)
            self.stats["full_scans"] += 1
        else:
            n, ids, wrong, bnd, masks = scan_ids(self.obs.scancls, piece.pending, self._origin, self.obs.denom,
                                                 rows, offs)
            self.stats["list_scans"] += 1
            self.stats["list_receivers"] += len(piece.pending)
        self.stats["screen_seconds"] += perf_counter() - t0
        if n in (-2, -3):
            raise ArithmeticError("corrupt obligation list (SOLID, unsorted or out of range)")
        if n < 0:
            self._commit((h,), [])
            self.stats["whole_out"] += 1
            return dict(outcome="out")
        if n == 0:
            return self._settle(h, piece)
        if n > len(ids):  # overflow: keep the full-window obligation
            return dict(outcome="open", handle=h, ambiguous=n, ids=None, wrong=None, boundary=None)
        if rel is not None:
            masks = remap_masks(masks, rel)
        ids, masks = _readonly(ids, np.int64), _readonly(masks, np.uint64)
        new = self._commit((h,), [Piece(p, ids, frozenset(), False, masks)])[0]
        return dict(outcome="open", handle=new, ambiguous=n, ids=_view(ids), wrong=wrong, boundary=bnd,
                    masks=_view(masks))  # read-only views: hints for the search

    def _settle(self, h, piece):
        p = piece.poly
        if self.obs.cleared(p):
            legal, shell = [p], []
        else:
            legal, shell = clearance_partition(self.obs, p, self.budget)
        empty = _readonly(np.zeros(0), np.int64)
        if not legal and len(shell) == 1 and shell[0] is p:
            new = self._commit((h,), [Piece(p, empty, frozenset(), True)])
            return dict(outcome="shell", handle=new[0])
        for q in legal:
            self._certify_structure(q)  # the credit below is exact only for a valid face structure
        lo, hi = self._grid_credit(legal)
        self._commit((h,), [Piece(q, empty, frozenset(), True) for q in shell], lo, hi)
        self.stats["whole_in"] += len(legal)
        return dict(outcome="in", legal=len(legal), shell=len(shell))

    @_transition
    def enforce(self, h, receiver_ids):
        """Exact inverse-cone cuts for the given FREE receivers (any order, any subset)."""
        piece = self._state.live[h]
        if piece.deferred:
            raise ValueError("deferred clearance shell")
        sel = list(dict.fromkeys(int(i) for i in receiver_ids))
        if not sel:
            raise ValueError("no receivers")
        for i in sel:
            self._receiver(self._idx_pos(i))  # FREE and inside the window
        t_enf = perf_counter()
        cand = self._candidates(piece, sel)
        parts = [piece.poly]
        for idx, bodies in zip(sel, cand):
            parts = self._exact_cut(parts, idx, bodies)
            if not parts:
                break
        same = len(parts) == 1 and parts[0] is piece.poly
        if not same and sum((q.volume for q in parts), Fraction(0)) > piece.poly.volume:
            raise ArithmeticError("exact cut created volume")  # disjoint subsets cannot exceed the parent
        if piece.pending is None:
            children = [Piece(q, None, piece.done.union(sel), False) for q in parts]
        else:
            drop = np.unique(np.array(sel, np.int64))
            pos = np.searchsorted(piece.pending, drop)
            ok = pos < len(piece.pending)
            pos, drop = pos[ok], drop[ok]
            pos = pos[piece.pending[pos] == drop]
            rest = _readonly(np.delete(piece.pending, pos), np.int64)
            rmask = None if piece.masks is None else _readonly(np.delete(piece.masks, pos), np.uint64)
            children = [Piece(q, rest, frozenset(), False, rmask) for q in parts]
        self.stats["receiver_cuts"] += len(sel)
        self.stats["cuts_no_change" if same else "cuts_changed"] += 1
        self.stats["enforce_seconds"] += perf_counter() - t_enf
        return self._commit((h,), children), same

    def _candidates(self, piece, sel):
        """Bodies that can block each receiver on this piece (kernel's own screen masks).

        A body outside the mask has a possible shadow (over the piece, hence over every
        subset) that misses the receiver, so its inverse cone misses the piece a.e.
        Receivers without a stored mask use every body.
        """
        nb = len(self.obs.bodies)
        every = tuple(range(nb))
        if piece.masks is None or piece.pending is None or nb > 64:
            return [every] * len(sel)
        pos = np.searchsorted(piece.pending, np.array(sel, np.int64))
        out = []
        for i, q in zip(sel, pos):
            if q < len(piece.pending) and piece.pending[q] == i:
                m = int(piece.masks[q])
                out.append(tuple(c for c in range(nb) if m >> c & 1))
            else:
                out.append(every)
        return out

    def _cone(self, c, idx):
        key = (c, idx)
        hit = self._cones.get(key)
        if hit is None:
            hit = cone(self.obs.bodies[c], self.obs.world(self._idx_pos(idx)))
            if len(self._cones) > 200000:
                self._cones.clear()
            self._cones[key] = hit
        return hit

    def _exact_cut(self, parts, idx, bodies):
        """Exact inverse-cone enforcement of one receiver over the candidate bodies."""
        x, y, z = self._idx_pos(idx)
        cones = [self._cone(c, idx) for c in bodies]
        if int(self.obs.cls[x, y, z]) == 1:
            return subtract_many(parts, cones, self.budget)
        out = []
        for p in parts:
            prior = []
            for hcone in cones:
                q = within(p, hcone)
                if q is not None:
                    out.extend(subtract_many([q], prior, self.budget))
                    prior.append(hcone)
                self.budget.check(len(out))
        return out

    # ---------------------------------------------------------------- accounting
    def _grid_credit(self, parts):
        if NATIVE is not None:
            return self._grid_credit_native(parts)
        return self._grid_credit_python(parts)

    def _grid_credit_native(self, parts):
        """GMP octree (native/cells.cpp), same algorithm as _grid_credit_python and exactly
        equal per-cell credits (differential test). Conservation is re-checked here against
        the independent Python rational volume of each piece."""
        started = perf_counter()
        lo, hi = Counter(), Counter()
        try:
            for p in parts:
                self.budget.check()
                credits, boundary, interior = NATIVE.grid_credit(p.faces, 128, 160)
                subtotal = 0
                for cid, low, high in credits:
                    if low:
                        lo[cid] += low
                    hi[cid] += high
                    subtotal += low
                total = p.volume * SCALE
                ok = (subtotal <= total < subtotal + boundary) if boundary else (subtotal == total)
                if not ok:
                    raise ArithmeticError("native grid integration failed volume conservation")
                self.stats["grid_intersections"] += boundary
                self.stats["cells_interior"] += interior
            return lo, hi
        finally:
            self.stats["integration_seconds"] += perf_counter() - started

    def _grid_credit_python(self, parts):
        """Exact per-cell credits of IN pieces (units 2^-160): floor for lower, ceiling for upper.

        Adaptive octree over the piece's cell range. A block whose 8 corners satisfy every
        (validated) facet row lies inside the convex piece and gets the exact cell volume
        per cell; a block on the nonnegative side of one facet row misses the interior; a
        boundary cell is clipped exactly. Conservation: floors of the clipped cells plus
        the exact interior cells must lie in [vol*2^160 - count, vol*2^160].
        """
        started = perf_counter()
        lo, hi = Counter(), Counter()
        try:
            for p in parts:
                rows = facet_rows(p)
                subtotal = count = 0
                (i0, i1), (j0, j1), (k0, k1) = p.grid_bounds(128)
                # Each stack entry carries q = p clipped exactly to its cell block.
                stack = [((i0, i1, j0, j1, k0, k1), p)]
                while stack:
                    self.budget.check()
                    (a0, a1, b0, b1, c0, c1), q = stack.pop()
                    corners = [(x, y, z, 128) for x in (a0, a1 + 1) for y in (b0, b1 + 1) for z in (c0, c1 + 1)]
                    if all(form(r, cq) <= 0 for r in rows for cq in corners):
                        for x in range(a0, a1 + 1):
                            for y in range(b0, b1 + 1):
                                for z in range(c0, c1 + 1):
                                    cid = cell_id((x, y, z))
                                    lo[cid] += CELL_UNITS
                                    hi[cid] += CELL_UNITS
                                    subtotal += CELL_UNITS
                        self.stats["cells_interior"] += (a1 - a0 + 1) * (b1 - b0 + 1) * (c1 - c0 + 1)
                        continue
                    if a0 == a1 and b0 == b1 and c0 == c1:
                        mass = q.volume * SCALE
                        low = mass.numerator // mass.denominator
                        cid = cell_id((a0, b0, c0))
                        if low:
                            lo[cid] += low
                        hi[cid] += -(-mass.numerator // mass.denominator)
                        subtotal += low
                        count += 1
                        continue
                    ext = (a1 - a0, b1 - b0, c1 - c0)
                    ax = max(range(3), key=lambda t: ext[t])
                    box = [a0, a1, b0, b1, c0, c1]
                    mid = (box[2 * ax] + box[2 * ax + 1]) // 2
                    plane = [0, 0, 0, -(mid + 1)]
                    plane[ax] = 128
                    left_q, right_q = q.split(tuple(plane))
                    left, right = list(box), list(box)
                    left[2 * ax + 1] = mid
                    right[2 * ax] = mid + 1
                    if left_q is not None:
                        stack.append((tuple(left), left_q))
                    if right_q is not None:
                        stack.append((tuple(right), right_q))
                total = p.volume * SCALE
                ok = (subtotal <= total < subtotal + count) if count else (subtotal == total)
                if not ok:
                    raise ArithmeticError("grid integration failed volume conservation")
                self.stats["grid_intersections"] += count
            return lo, hi
        finally:
            self.stats["integration_seconds"] += perf_counter() - started

    @_transition
    def _certify_structure(self, q):
        if q not in self._checked:
            t = perf_counter()
            polycheck.check(q)  # raises ArithmeticError (taints the kernel) on failure
            self._checked.add(q)
            self.stats["structure_checks"] += 1
            self.stats["structure_seconds"] += perf_counter() - t

    def snapshot(self):
        state = self._state
        if not state.lower:
            return None, None
        unknown_total = 0
        unknown_in = Counter()
        for piece in state.live.values():
            if CHECK_LIVE:
                self._certify_structure(piece.poly)  # its charge ceil(vol) is exact only then
            amount = ceil_mass(piece.poly)
            unknown_total += amount
            cid = single_cell(piece.poly)
            if cid is not None:
                unknown_in[cid] += amount
        payload = encode_int(state.lower)
        bound = wire_bound(payload, state.lower, state.upper_in, unknown_total, unknown_in)
        if self._best is None or bound < self._best[0]:
            self._best = (bound, payload)
        return payload, bound

    def release(self, tolerance):
        """The only source of a CERTIFIED status: the best bound the kernel itself computed."""
        if type(tolerance) is not Fraction or not 0 <= tolerance <= 1:
            raise ValueError("exact tolerance in [0, 1] required")
        if self._tainted is not None:
            return Release("ERROR", None, None, tolerance, self._tainted)
        try:
            self.snapshot()  # the final state is a valid checkpoint too
        except ArithmeticError:
            return Release("ERROR", None, None, tolerance, self._tainted)
        if self._best is None:
            return Release("INCOMPLETE", None, None, tolerance, "no positive feasible mass")
        bound, payload = self._best
        if bound <= tolerance:
            return Release("CERTIFIED", payload, bound, tolerance, "tv_target")
        return Release("INCOMPLETE", None, bound, tolerance, "bound above tolerance")

    def unknown_volume(self):
        """Search hint: total live volume (binary64; exact Fraction sums over hundreds of
        pieces with large denominators cost ~1 s)."""
        return sum(float(p.poly.volume) for p in self._state.live.values())

    def lower_total_float(self):
        return sum(self._state.lower.values()) / float(SCALE)

    def unknown_fraction(self):
        """Search hint: live volume relative to proved IN volume."""
        st = self._state
        unk = sum((p.poly.volume for p in st.live.values()), Fraction(0))
        lo = Fraction(sum(st.lower.values()), SCALE)
        return unk, lo


def certify(obs, tolerance, seconds, cap, search):
    """Trusted driver: owns the budget and the kernel, runs the untrusted search, releases.

    search(kernel, budget) may return a diagnostics dict, run out of time (Limit) or fail
    with any exception: all three are abstentions of the search. Only kernel.release()
    decides the status, payload and bound of the result; diagnostics cannot override them.
    """
    if type(tolerance) is not Fraction or not 0 <= tolerance <= 1 or seconds <= 0 or cap < 1:
        raise ValueError("invalid budget or tolerance")
    budget = Budget(seconds, cap)
    kernel = Kernel(obs, budget)
    try:
        details = dict(search(kernel, budget) or {})
    except Limit as exc:
        details = dict(search_reason=str(exc))
    except Exception as exc:  # noqa: BLE001  (a failed search abstains; the kernel state stays valid)
        details = dict(search_reason="search_error", search_error=f"{type(exc).__name__}: {exc}")
    rel = kernel.release(tolerance)
    details.update(status=rel.status, tv=None if rel.bound is None else str(rel.bound), reason=rel.reason,
                   live=len(kernel.handles()), positive=kernel.positive, stats=dict(kernel.stats))
    return rel.payload, details

