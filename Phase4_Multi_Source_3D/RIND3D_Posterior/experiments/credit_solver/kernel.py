"""Research kernel. A single owner issues/consumes source-region handles.

Only methods evaluating the problem's own observations can earn mass or discard
possible sources. Each expensive transition prepares scratch results before
replacing the committed state. See SPEC.md for the physical and numerical TCB.
"""
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from time import perf_counter

import numpy as np

import inputs  # installs the pinned research dependency paths
from envelopes import clearance_partition
from ledger import decode, encode
from poly import Poly, grid_pieces
from raster import enforce, screen
from solver import Limit, apply_pair, cone

SCALE = 1 << 160
COMMON = 1 << 96


def ceil_mass(p):
    volume = p.volume * SCALE
    return -(-volume.numerator // volume.denominator)


def cell_id(ijk):
    return (ijk[0] * 128 + ijk[1]) * 128 + ijk[2]


def single_cell(p):
    dims = p.grid_bounds(128)
    return cell_id([a for a,b in dims]) if all(a == b for a,b in dims) else None


@dataclass(frozen=True)
class Piece:
    poly: Poly
    enforced: frozenset
    deferred: bool = False


@dataclass(frozen=True)
class State:
    live: dict
    lower: Counter
    upper_in: Counter
    next_id: int
    outside_window: bool


def wire_bound(payload, lower, upper_in, unknown_total, unknown_in):
    """Integer-tick enclosure on the decoded wire simplex. No geometry replay."""
    if not any(lower.values()):
        raise ValueError("no positive feasible mass")
    total = sum(upper_in.values()) + unknown_total
    common = 0
    for c, p in decode(payload).items():
        l = lower.get(c,0)
        outside = total - upper_in.get(c,0) - unknown_in.get(c,0)
        if outside < 0:
            raise ArithmeticError("negative outer mass")
        if l:
            # Floor common mass; this makes the released upper TV round outward.
            common += min(int(p*COMMON), l*COMMON//(l+outside))
    return Fraction(COMMON-common,COMMON)


class Kernel:
    def __init__(self, observation, budget):
        self.obs = observation
        self.budget = budget
        self._state = State({0:Piece(Poly.cube(),frozenset())},Counter(),Counter(),1,False)
        self.stats = Counter()
        self.obs.scancls = np.ascontiguousarray(self.obs.cls.T)

    def handles(self):
        return tuple(self._state.live)

    def priority(self, handle):
        return self._state.live[handle].poly.volume

    def work_handles(self):
        return [h for h,p in self._state.live.items() if not p.deferred]

    def _commit(self, remove, children, lower=None, upper=None, exclude=False):
        old = self._state
        self.budget.check(len(old.live)-len(remove)+len(children))
        live = old.live.copy()
        for key in remove:
            del live[key]  # stale or duplicate consumption fails before assignment
        live.update((old.next_id+i,ch) for i,ch in enumerate(children))
        lo = old.lower if lower is None else old.lower+lower
        hi = old.upper_in if upper is None else old.upper_in+upper
        self._state = State(live,lo,hi,old.next_id+len(children),old.outside_window or exclude)
        self.stats["transitions"] += 1
        self.stats["peak_live"] = max(self.stats["peak_live"],len(live))

    def _receiver(self, pos, label):
        if len(pos)!=3 or any(type(i) is not int or not 0<=i<self.obs.side for i in pos):
            raise ValueError("invalid receiver coordinate")
        if int(self.obs.cls[pos]) != label:
            raise ValueError("receiver evidence contradicts observation")
        return self.obs.world(pos)

    def condition_pair(self, dark, lit):
        d,a = self._receiver(dark,0), self._receiver(lit,1)
        cones_d = [cone(b,d) for b in self.obs.bodies]
        cones_a = [cone(b,a) for b in self.obs.bodies]
        children = []
        for piece in self._state.live.values():
            parts = apply_pair(piece.poly,cones_d,cones_a,self.budget)
            children.extend(Piece(p,piece.enforced) for p in parts)
            self.budget.check(len(children))
        self._commit(self.handles(),children)
        self.stats["pairs"] += 1

    def exclude_window(self):
        if self._state.outside_window:
            return
        children = []
        for piece in self._state.live.values():
            parts = self.obs.exclude_window([piece.poly],self.budget)
            children.extend(Piece(p,piece.enforced) for p in parts)
            self.budget.check(len(children))
        self._commit(self.handles(),children,exclude=True)

    def _grid_credit(self, parts):
        started = perf_counter()
        lo,hi = Counter(),Counter()
        try:
            for p in parts:
                subtotal = count = 0
                for xyz,small in grid_pieces(p,128):
                    self.budget.check()
                    mass = small.volume*SCALE
                    low = mass.numerator//mass.denominator
                    c = cell_id(xyz)
                    if low:
                        lo[c] += low
                    hi[c] += -(-mass.numerator//mass.denominator)
                    subtotal += low
                    count += 1
                if not subtotal <= p.volume*SCALE < subtotal+count:
                    raise ArithmeticError("grid integration failed volume conservation")
                self.stats["grid_intersections"] += count
            return lo,hi
        finally:
            self.stats["integration_seconds"] += perf_counter()-started

    def advance(self, handle):
        if not self._state.outside_window:
            raise ValueError("closed window has not been excluded")
        piece = self._state.live[handle]
        p, done = piece.poly, piece.enforced
        started = perf_counter()
        ambiguous, ids, _ = screen(self.obs,p,done)
        self.stats["collective_seconds"] += perf_counter()-started
        self.stats["full_scans"] += 1
        if ambiguous < 0:
            self._commit((handle,),[])
            self.stats["whole_out"] += 1
        elif ambiguous == 0:
            if self.obs.cleared(p):
                legal,uncertain = [p],[]
            else:
                legal,uncertain = clearance_partition(self.obs,p,self.budget)
            if not legal and len(uncertain)==1 and uncertain[0] is p:
                return False
            lo,hi = self._grid_credit(legal)
            self._commit((handle,),[Piece(q,done,True) for q in uncertain],lo,hi)
            self.stats["whole_in"] += len(legal)
            # A shell can remain even after a successful visibility proof.
            # Returning these handles to the scheduler is safe, even if it stalls.
        else:
            selected = tuple(dict.fromkeys(int(i) for i in ids if i>=0))
            if not selected or len(done)>=192:
                return False
            parts = [p]
            for idx in selected:
                z,yx = divmod(idx,self.obs.side**2)
                y,x = divmod(yx,self.obs.side)
                if z>=self.obs.side or self.obs.cls[x,y,z] not in (0,1) or idx in done:
                    raise ArithmeticError("invalid inherited obligation")
                parts = enforce(self.obs,parts,idx,self.budget)
                if not parts:
                    break
            self._commit((handle,),[Piece(q,done.union(selected)) for q in parts])
            self.stats["receiver_cuts"] += len(selected)
        return True

    def snapshot(self):
        state = self._state
        if not state.lower:
            return None,None
        unknown_total = 0
        unknown_in = Counter()
        for piece in state.live.values():
            amount = ceil_mass(piece.poly)
            unknown_total += amount
            cid = single_cell(piece.poly)
            if cid is not None:
                unknown_in[cid] += amount
        payload = encode(state.lower)
        return payload,wire_bound(payload,state.lower,state.upper_in,unknown_total,unknown_in)

    @property
    def positive(self):
        return bool(self._state.lower)
