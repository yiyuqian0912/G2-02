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
"""
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from time import perf_counter

import numpy as np

import inputs  # noqa: F401  (installs pinned research paths)
from envelopes import clearance_partition
from ledger import decode, encode
from poly import Poly, form, grid_pieces
from raster import encoded
from screen2 import check_ranges, scan_collect, scan_ids
from poly import within
from solver import apply_pair, cone, subtract_many
from kernel import SCALE, ceil_mass, cell_id, single_cell, wire_bound

COLLECT_CAP = 1 << 21  # largest residual obligation list stored explicitly


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

    # ---------------------------------------------------------------- read-only views
    def handles(self):
        return tuple(self._state.live)

    def open_handles(self):
        return [h for h, p in self._state.live.items() if not p.deferred]

    def volume(self, h):
        return self._state.live[h].poly.volume

    def pending_count(self, h):
        p = self._state.live[h]
        return None if p.pending is None else len(p.pending)

    def poly(self, h):
        """The polytope (immutable). Exposed for proposal heuristics only."""
        return self._state.live[h].poly

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

    def condition_pair(self, dark, lit):
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

    def split(self, h, row):
        """Any integer affine row. Both closed halves stay live; null overlap on the plane."""
        if len(row) != 4 or not all(type(v) is int for v in row) or not any(row[:3]):
            raise ValueError("invalid split plane")
        piece = self._state.live[h]
        neg, pos = piece.poly.split(tuple(row))
        if neg is None or pos is None:
            return [h]  # the plane misses the interior: no transition
        self.stats["splits"] += 1
        return self._commit((h,), [Piece(neg, piece.pending, piece.done, piece.deferred, piece.masks),
                                    Piece(pos, piece.pending, piece.done, piece.deferred, piece.masks)])

    def screen(self, h):
        """Classify pending receivers on the piece. Returns a dict; the piece may be consumed."""
        if not self._state.outside_window:
            raise ValueError("closed window has not been excluded")
        piece = self._state.live[h]
        if piece.deferred:
            raise ValueError("deferred clearance shell")
        p = piece.poly
        t0 = perf_counter()
        rows, offs = encoded(self.obs, p)
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
        new = self._commit((h,), [Piece(p, np.ascontiguousarray(ids), frozenset(), False,
                                        np.ascontiguousarray(masks))])[0]
        return dict(outcome="open", handle=new, ambiguous=n, ids=ids, wrong=wrong, boundary=bnd)

    def _settle(self, h, piece):
        p = piece.poly
        if self.obs.cleared(p):
            legal, shell = [p], []
        else:
            legal, shell = clearance_partition(self.obs, p, self.budget)
        if not legal and len(shell) == 1 and shell[0] is p:
            new = self._commit((h,), [Piece(p, np.zeros(0, np.int64), frozenset(), True)])
            return dict(outcome="shell", handle=new[0])
        lo, hi = self._grid_credit(legal)
        empty = np.zeros(0, np.int64)
        self._commit((h,), [Piece(q, empty, frozenset(), True) for q in shell], lo, hi)
        self.stats["whole_in"] += len(legal)
        return dict(outcome="in", legal=len(legal), shell=len(shell))

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
        cand = self._candidates(piece, sel)
        parts = [piece.poly]
        for idx, bodies in zip(sel, cand):
            parts = self._exact_cut(parts, idx, bodies)
            if not parts:
                break
        same = len(parts) == 1 and parts[0] is piece.poly
        if piece.pending is None:
            children = [Piece(q, None, piece.done.union(sel), False) for q in parts]
        else:
            drop = np.unique(np.array(sel, np.int64))
            pos = np.searchsorted(piece.pending, drop)
            ok = pos < len(piece.pending)
            pos, drop = pos[ok], drop[ok]
            pos = pos[piece.pending[pos] == drop]
            rest = np.delete(piece.pending, pos)
            rmask = None if piece.masks is None else np.delete(piece.masks, pos)
            children = [Piece(q, rest, frozenset(), False, rmask) for q in parts]
        self.stats["receiver_cuts"] += len(sel)
        self.stats["cuts_no_change" if same else "cuts_changed"] += 1
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
        started = perf_counter()
        lo, hi = Counter(), Counter()
        try:
            for p in parts:
                subtotal = count = 0
                for xyz, small in grid_pieces(p, 128):
                    self.budget.check()
                    mass = small.volume * SCALE
                    low = mass.numerator // mass.denominator
                    c = cell_id(xyz)
                    if low:
                        lo[c] += low
                    hi[c] += -(-mass.numerator // mass.denominator)
                    subtotal += low
                    count += 1
                if not subtotal <= p.volume * SCALE < subtotal + count:
                    raise ArithmeticError("grid integration failed volume conservation")
                self.stats["grid_intersections"] += count
            return lo, hi
        finally:
            self.stats["integration_seconds"] += perf_counter() - started

    def snapshot(self):
        state = self._state
        if not state.lower:
            return None, None
        unknown_total = 0
        unknown_in = Counter()
        for piece in state.live.values():
            amount = ceil_mass(piece.poly)
            unknown_total += amount
            cid = single_cell(piece.poly)
            if cid is not None:
                unknown_in[cid] += amount
        payload = encode(state.lower)
        return payload, wire_bound(payload, state.lower, state.upper_in, unknown_total, unknown_in)

    def unknown_fraction(self):
        """Search hint: live volume relative to proved IN volume."""
        st = self._state
        unk = sum((p.poly.volume for p in st.live.values()), Fraction(0))
        lo = Fraction(sum(st.lower.values()), SCALE)
        return unk, lo
