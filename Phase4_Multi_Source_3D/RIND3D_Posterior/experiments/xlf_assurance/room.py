"""Exact synthetic 64^3 window. A sealed room lies inside one posterior cell."""
from fractions import Fraction as Q
from itertools import product
from time import perf_counter
import hashlib

from geometry import affine, inverse_rows, rows_of, segment_hit
from ledger import Box, Ledger, decode

N = 1024
SIDE = 64
ORIGIN = (516, 512, 512)


def unit(values):
    return tuple(Q(x) / N for x in values)


def cube(a, b):
    return Box(unit((a,)*3), unit((b,)*3))


# The solid is outer minus the OPEN cavity, plus the axis-aligned post inside.
# This shell has an exact six-closed-box union representation.
OUTER = cube(513, 519)
CAVITY = cube(514, 518)
POST = Box(unit((Q(4127,8), Q(4121,8), Q(4121,8))),
           unit((Q(4135,8), Q(4127,8), Q(4127,8))))
W = Box(unit(ORIGIN), unit(tuple(v+SIDE for v in ORIGIN)))
EPS = Q(1, 2**26)


def room_samples():
    for x,y,z in product(range(516,519), range(513,519), range(513,519)):
        r = unit((Q(2*x+1,2), Q(2*y+1,2), Q(2*z+1,2)))
        i = ((x-516)*SIDE + y-512)*SIDE + z-512
        interior = all(a < v < b for a,v,b in zip(CAVITY.lo,r,CAVITY.hi))
        in_post = all(a <= v <= b for a,v,b in zip(POST.lo,r,POST.hi))
        yield i,r,interior and not in_post


def make_observation():
    # This source is local to observation generation. The solver takes only bytes.
    source = unit(tuple(515+Q(f,256) for f in (119,147,131)))
    rows = rows_of(POST)
    obs = bytearray(SIDE**3)  # free exterior receivers are all blocked by the shell
    for i,r,free_in_room in room_samples():
        obs[i] = int(not segment_hit(rows,source,r)) if free_in_room else 2
    return bytes(obs)


def solve(obs):
    if len(obs) != SIDE**3:
        raise ValueError("incomplete window")
    rows = rows_of(POST)
    free, shadows = [], []
    accounted_nonzero = 0
    for i,r,interior in room_samples():
        value = obs[i]
        if (interior and value not in (0,1)) or (not interior and value != 2):
            raise ValueError("invalid receiver class")
        if interior:
            free.append((r,value))
            shadows.append((inverse_rows(rows,r),value))
        accounted_nonzero += int(value != 0)
    if len(obs) - obs.count(0) != accounted_nonzero or not any(y for _,y in free):
        raise ValueError("exterior is not all dark or no interior active witness")

    def assess(box):
        if not box.overlap(CAVITY):
            return "OUT"  # any segment from here to the active receiver crosses the shell
        if box.overlap(W) or not all(a+EPS < c and d < b-EPS for a,b,c,d in
                                     zip(CAVITY.lo,CAVITY.hi,box.lo,box.hi)):
            return "R"
        if not any(a-d > EPS or c-b > EPS for a,b,c,d in zip(box.lo,box.hi,POST.lo,POST.hi)):
            return "R"
        vertices = tuple(box.vertices())
        for shadow,y in shadows:
            ranges = [tuple(affine(row,v) for v in vertices) for row in shadow]
            certified = any(min(q)>0 for q in ranges) if y else all(max(q)<=0 for q in ranges)
            if not certified:
                return "R"
        return "IN"

    ledger = Ledger(assess)
    key = 1
    # Use six complement pieces to certify zero possible mass outside cell (64,64,64).
    for ax in range(3):
        low,key = ledger.split(key,ax,Q(512,N))
        key,high = ledger.split(key,ax,Q(520,N))
        for outer in (low,high):
            if ledger.assess(outer) != "OUT":
                raise AssertionError("outer proof failed")

    # Untrusted point proposals in lexicographic order. Verify a box, never just its centre.
    delta = Q(1,256*N)
    inner_post = Box(tuple(a+2*delta for a in POST.lo),tuple(b-2*delta for b in POST.hi))
    inner_rows = rows_of(inner_post)
    outer_rows = rows_of(Box(tuple(a-2*delta for a in POST.lo),tuple(b+2*delta for b in POST.hi)))
    offsets = tuple(product((0,-1,1), repeat=3))
    centers = (unit(tuple(v+Q(1,2)+Q(h,16) for v,h in zip(ijk,shift)))
               for ijk in product(range(514,516),range(514,518),range(514,518))
               for shift in offsets)
    count = 0
    p = None
    for center in centers:
        count += 1
        if any(int(not segment_hit(rows,center,r)) != label for r,label in free):
            continue
        for shift in offsets:
            point = tuple(v+Q(d,64*N) for v,d in zip(center,shift))
            count += 1
            if all((not segment_hit(outer_rows,point,r)) if label else segment_hit(inner_rows,point,r)
                   for r,label in free):
                p = point
                break
        if p is not None:
            break
    if p is None:
        return dict(status="ABSTAIN", reason="no matching point proposal", point_proposals=count)
    for ax in range(3):
        _,key = ledger.split(key,ax,p[ax]-delta)
        key,_ = ledger.split(key,ax,p[ax]+delta)
    proof = ledger.assess(key)
    state,payload,bound = ledger.certify(0)
    # Count only direct interfaces (avoid flattened-axis wrap).
    local_free = {r:y for r,y in free}
    edges = 0
    for r,y in free:
        for ax in range(3):
            v = list(r)
            v[ax] += Q(1,N)
            if local_free.get(tuple(v),y) != y:
                edges += 1
    return dict(status=state, inner_region=proof, upper=None if bound is None else str(bound),
                decoded=None if payload is None else {str(c):str(v) for c,v in decode(payload).items()},
                positive_lower=str(sum(ledger.credits()[0].values())), point_proposals=count,
                receivers=len(obs), active=obs.count(1), inactive_free=obs.count(0), solid=obs.count(2),
                direct_free_interface_edges_inside_room=edges, observation_sha256=hashlib.sha256(obs).hexdigest())


if __name__ == "__main__":
    import json
    observation = make_observation()
    start = perf_counter()
    result = solve(observation)
    result["solve_seconds"] = perf_counter()-start
    print(json.dumps(result,indent=2))
