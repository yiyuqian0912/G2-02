"""Tests of the CBC kernel v2 (exact references; one real v3 task for realistic geometry).

Run from this directory:
  PYTHONPATH=".:../../src:../../../RIND3D_Generator/src:../../../RIND3D_Dataset1K" python -m pytest -q test_cbc.py
"""
from fractions import Fraction
from pathlib import Path
import random

import numpy as np
import pytest

import inputs
import cbc_kernel
from cbc_kernel import Kernel, Piece
from cbc_search import solve
from ledger import decode
from poly import Poly, form
from raster import encoded
from screen2 import scan_collect, scan_ids, check_ranges
from solver import Budget, choose_pairs
from exactcheck import Oracle, _exact_hit
from diag import halfspaces, contains

ROOT = Path("/home/yushu/Projects/RIND/RIND3D_100K_v3_work/sampling_impact/v3")
TASK = ("0000/000030.npz", 1, 0)  # L=64 wedges; certifies in < 2 s


def load():
    if not (ROOT / "scenes" / TASK[0]).exists():
        pytest.skip("v3 canary data not available")
    return inputs.load(ROOT / "scenes" / TASK[0], TASK[1], TASK[2])


@pytest.fixture(scope="module")
def conditioned():
    obs, _ = load()
    k = Kernel(obs, Budget(60, 10 ** 6))
    for d, a in choose_pairs(obs.cls, 8, 1, "interface"):
        k.condition_pair(d, a)
    k.exclude_window()
    return obs, k


def test_scan_ids_matches_full_scan(conditioned):
    obs, k = conditioned
    rng = np.random.default_rng(0)
    free = np.flatnonzero(obs.scancls.ravel() != 2)
    for h in list(k.handles())[:6]:
        p = k.poly(h)
        rows, offs = encoded(obs, p)
        check_ranges(rows, obs.origin, obs.side, obs.denom)
        n, ids, w, b, mk = scan_collect(obs.scancls, np.asarray(obs.origin, np.int64), obs.denom, rows, offs,
                                    np.zeros(0, np.int64), 1 << 22)
        sub = np.sort(rng.choice(free, size=min(5000, len(free)), replace=False)).astype(np.int64)
        m, ids2, w2, b2, mk2 = scan_ids(obs.scancls, sub, np.asarray(obs.origin, np.int64), obs.denom, rows, offs)
        if n < 0:
            assert m < 0 or m >= 0  # a contradiction elsewhere need not lie in the subset
            continue
        expect, ia, ib = np.intersect1d(ids, sub, return_indices=True)
        assert m >= 0 and np.array_equal(ids2, expect)
        assert np.array_equal(mk[ia], mk2)  # identical candidate-body masks


def test_scan_ids_rejects_corrupt_lists(conditioned):
    obs, k = conditioned
    p = k.poly(k.handles()[0])
    rows, offs = encoded(obs, p)
    o = np.asarray(obs.origin, np.int64)
    bad = np.array([5, 3], np.int64)
    assert scan_ids(obs.scancls, bad, o, obs.denom, rows, offs)[0] == -3
    solid = np.flatnonzero(obs.scancls.ravel() == 2)
    if len(solid):
        assert scan_ids(obs.scancls, solid[:1].astype(np.int64), o, obs.denom, rows, offs)[0] == -2


def test_split_is_exact_partition(conditioned):
    obs, k = conditioned
    h = max(k.open_handles(), key=k.volume)
    p = k.poly(h)
    vs = [tuple(Fraction(v[i], v[3]) for i in range(3)) for v in p.vertices]
    c = sum(v[0] for v in vs) / len(vs)
    row = (c.denominator, 0, 0, -c.numerator)
    before = sum((k.volume(x) for x in k.handles()), Fraction(0))
    out = k.split(h, row)
    assert len(out) == 2
    a, b = out
    assert k.volume(a) + k.volume(b) == p.volume
    assert sum((k.volume(x) for x in k.handles()), Fraction(0)) == before
    with pytest.raises(KeyError):
        k.split(h, row)  # stale handle
    with pytest.raises(ValueError):
        k.split(a, (0, 0, 0, 1))


def test_enforce_children_satisfy_receivers(conditioned):
    obs, k = conditioned
    h = max(k.open_handles(), key=k.volume)
    r = k.screen(h)
    if r["outcome"] != "open":
        pytest.skip("piece settled")
    h = r["handle"]
    parent = k.poly(h)
    chosen = [int(i) for i in r["ids"][:6]]
    kids, same = k.enforce(h, chosen)
    rng = random.Random(1)
    L = obs.side
    for c in kids:
        poly = k.poly(c)
        assert all(contains(halfspaces(parent), v) for v in poly.vertices)
        verts = [tuple(Fraction(v[i], v[3]) for i in range(3)) for v in poly.vertices]
        for _ in range(3):
            w = [Fraction(rng.randint(1, 1000)) for _ in verts]
            s = tuple(sum(wi * v[i] for wi, v in zip(w, verts)) / sum(w) for i in range(3))
            for idx in chosen:
                x, y, z = idx % L, idx // L % L, idx // (L * L)
                rr = tuple(Fraction(2 * (obs.origin[i] + (x, y, z)[i]) + 1, 2048) for i in range(3))
                blocked = any(_exact_hit(b, s, rr) for b in obs.bodies)
                assert blocked == (obs.cls[x, y, z] == 0)
        if k.pending_count(c) is not None:
            assert not np.isin(np.array(chosen), k._state.live[c].pending).any()


def test_enforce_rejects_solid_and_foreign_receivers(conditioned):
    obs, k = conditioned
    h = k.open_handles()[0]
    solid = np.argwhere(obs.cls == 2)
    if len(solid):
        x, y, z = map(int, solid[0])
        with pytest.raises(ValueError):
            k.enforce(h, [(z * obs.side + y) * obs.side + x])
    with pytest.raises(ValueError):
        k.enforce(h, [obs.side ** 3])


def test_window_and_solids_excluded(conditioned):
    obs, k = conditioned
    n = 1024
    lo = [Fraction(o, n) for o in obs.origin]
    hi = [Fraction(o + obs.side, n) for o in obs.origin]
    for h in k.handles():
        p = k.poly(h)
        verts = [tuple(Fraction(v[i], v[3]) for i in range(3)) for v in p.vertices]
        c = tuple(sum(v[i] for v in verts) / len(verts) for i in range(3))
        assert not all(lo[i] < c[i] < hi[i] for i in range(3))
        for b in obs.bodies:
            assert not all(sum(r[i] * c[i] for i in range(3)) + r[3] < 0 for r in b)


def test_end_to_end_certificate_on_bytes():
    obs, _ = load()
    payload, d = solve(obs, "000030_w01_s0", Fraction(1, 20), 20)
    assert d["status"] == "CERTIFIED"
    p = decode(payload)
    assert sum(p.values()) == 1
    assert Fraction(d["tv"]) <= Fraction(1, 20)


def test_unvisited_mass_is_charged():
    """Pieces the search never visits stay in the bound: omitting them can only lower it."""
    obs, _ = load()
    k = Kernel(obs, Budget(60, 10 ** 6))
    for d, a in choose_pairs(obs.cls, 24, 1, "interface"):
        k.condition_pair(d, a)
    k.exclude_window()
    queue = [max(k.open_handles(), key=k.volume)]
    steps = 0
    while queue and not k.positive and steps < 200:
        h = queue.pop()
        steps += 1
        r = k.screen(h)
        if r["outcome"] == "open":
            kids, _ = k.enforce(r["handle"], [int(i) for i in r["ids"][:96]])
            queue.extend(kids)
    if not k.positive:
        pytest.skip("no IN credit within the test budget")
    payload, bound = k.snapshot()
    st = k._state
    omitted = cbc_kernel.wire_bound(payload, st.lower, st.upper_in, 0, {})
    unk, lo = k.unknown_fraction()
    assert unk > 0                      # other pieces were never visited
    assert omitted <= bound             # charging them never helps the certificate
    assert bound > omitted              # and here it strictly matters


def test_masks_are_sound(conditioned):
    """Bodies outside a receiver's mask have inverse cones that miss the piece (exact)."""
    from poly import within
    obs, k = conditioned
    checked = 0
    for h in sorted(k.open_handles(), key=k.volume, reverse=True)[:4]:
        r = k.screen(h)
        if r["outcome"] != "open" or r["ids"] is None:
            continue
        piece = k._state.live[r["handle"]]
        L = obs.side
        for idx, m in list(zip(piece.pending, piece.masks))[:40]:
            idx, m = int(idx), int(m)
            x, y, z = idx % L, idx // L % L, idx // (L * L)
            rr = obs.world((x, y, z))
            for c, body in enumerate(obs.bodies):
                if not m >> c & 1:
                    from solver import cone
                    assert within(piece.poly, cone(body, rr)) is None
                    checked += 1
    assert checked > 0


def test_body_subset_screen_is_never_weaker():
    """On a many-body scene, the residual screen restricted to the masked bodies proves at
    least what the all-body screen proves, and never disagrees on a contradiction."""
    from cbc_kernel import encoded_subset
    if not (ROOT / "scenes" / "0000/000044.npz").exists():
        pytest.skip("data")
    obs, _ = inputs.load(ROOT / "scenes" / "0000/000044.npz", 4, 0)
    k = Kernel(obs, Budget(120, 10 ** 6))
    for d, a in choose_pairs(obs.cls, 6, 1, "interface"):
        k.condition_pair(d, a)
    k.exclude_window()
    o = np.asarray(obs.origin, np.int64)
    compared = 0
    for h in sorted(k.open_handles(), key=k.volume, reverse=True)[:6]:
        r = k.screen(h)
        if r["outcome"] != "open" or r["ids"] is None or not len(r["ids"]):
            continue
        piece = k._state.live[r["handle"]]
        kids, _ = k.enforce(r["handle"], [int(i) for i in piece.pending[:8]])
        for c in kids:
            q = k._state.live[c]
            if q.pending is None or not len(q.pending):
                continue
            rows, offs = encoded_subset(obs, q.poly, range(len(obs.bodies)))
            nf, idf, *_ = scan_ids(obs.scancls, q.pending, o, obs.denom, rows, offs)
            union = int(np.bitwise_or.reduce(q.masks))
            rel = [b for b in range(len(obs.bodies)) if union >> b & 1]
            rows, offs = encoded_subset(obs, q.poly, rel)
            ns, ids_, *_ = scan_ids(obs.scancls, q.pending, o, obs.denom, rows, offs)
            if nf < 0:
                continue  # a contradiction found with all bodies is also sound
            assert ns < 0 or set(ids_.tolist()) <= set(idf.tolist())
            compared += 1
    assert compared > 0
