"""Adversarial tests of the unified kernel: no search behaviour can create a false certificate.

Covers: unauthorized certificates, search failures, double settlement and duplicate handles,
unprocessed (missing) branches, invalid or degenerate cutting planes, invalid receivers,
writes through returned evidence, mutation of the shared input after construction,
invalid polytope structures at credit time, timeouts inside transitions, inexact
tolerances, and a static check that untrusted modules never touch kernel-private state.

Run: PYTHONPATH=... python -m pytest -q test_adversarial.py
"""
import ast
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import cbc_kernel
import inputs
import polycheck
from cbc_kernel import Kernel, certify
from kernel import ceil_mass
from ledger import decode
from solver import Budget, Limit, choose_pairs

HERE = Path(__file__).resolve().parent
ROOT = Path("/home/yushu/Projects/RIND/RIND3D_100K_v3_work/sampling_impact/v3")
TASK = ("0000/000030.npz", 1, 0)


def load():
    if not (ROOT / "scenes" / TASK[0]).exists():
        pytest.skip("v3 canary data not available")
    return inputs.load(ROOT / "scenes" / TASK[0], TASK[1], TASK[2])[0]


def prepared(seconds=120):
    obs = load()
    k = Kernel(obs, Budget(seconds, 10 ** 6))
    for d, a in choose_pairs(obs.cls, 8, 1, "interface"):
        k.condition_pair(d, a)
    k.exclude_window()
    return obs, k


def drive_until_positive(k, limit=400):
    """Greedy driver until some IN credit exists; returns the handles it left open."""
    for _ in range(limit):
        if k.positive:
            return
        opened = k.open_handles()
        if not opened:
            return
        h = max(opened, key=k.volume)
        r = k.screen(h)
        if r["outcome"] == "open" and r["ids"] is not None:
            k.enforce(r["handle"], [int(i) for i in r["ids"][:96]])
    raise AssertionError("no positive mass reached")


# ---------------------------------------------------------------- certification authority
def test_search_cannot_issue_a_certificate():
    obs = load()

    def forger(kernel, budget):
        return dict(status="CERTIFIED", tv="0", reason="tv_target", payload=b"forged")
    payload, d = certify(obs, Fraction(1, 20), 10, 4096, forger)
    assert payload is None and d["status"] == "INCOMPLETE" and d["tv"] is None


def test_search_failure_is_an_abstention():
    obs = load()

    def broken(kernel, budget):
        kernel.exclude_window()
        raise RuntimeError("search bug")
    payload, d = certify(obs, Fraction(1, 20), 10, 4096, broken)
    assert payload is None and d["status"] == "INCOMPLETE" and "search bug" in d["search_error"]


def test_release_requires_an_exact_tolerance():
    obs, k = prepared()
    with pytest.raises(ValueError):
        k.release(0.05)
    with pytest.raises(ValueError):
        k.release(Fraction(3, 2))


# ---------------------------------------------------------------- double counting
def test_a_piece_cannot_be_settled_twice():
    obs, k = prepared()
    drive_until_positive(k)
    before = sum(k._state.lower.values())
    settled = [h for h in k.handles() if not k.is_open(h)]
    # any handle consumed by a transition is stale afterwards
    h = max(k.open_handles(), key=k.volume)
    r = k.screen(h)
    with pytest.raises(KeyError):
        k.screen(h)
    with pytest.raises(KeyError):
        k.enforce(h, [0])
    with pytest.raises(KeyError):
        k.split(h, (1, 0, 0, -1))
    assert sum(k._state.lower.values()) >= before
    assert settled is not None and r is not None


def test_duplicate_handle_in_one_commit_is_rejected():
    obs, k = prepared()
    h = k.handles()[0]
    state = k._state
    with pytest.raises(KeyError):
        k._commit((h, h), [])
    assert k._state is state


# ---------------------------------------------------------------- missing branches
def test_unprocessed_pieces_stay_charged():
    obs, k = prepared()
    drive_until_positive(k)
    for h in list(k.open_handles())[:5]:  # create more branches and never process them
        p = k.poly(h)
        x = sum(Fraction(v[0], v[3]) for v in p.vertices) / len(p.vertices)
        k.split(h, (x.denominator, 0, 0, -x.numerator))
    payload, bound = k.snapshot()
    # recompute the charge of every live piece independently and rebuild the bound
    from kernel import single_cell, wire_bound
    from collections import Counter
    unknown, inside = 0, Counter()
    for piece in k._state.live.values():
        amount = ceil_mass(piece.poly)
        unknown += amount
        c = single_cell(piece.poly)
        if c is not None:
            inside[c] += amount
    assert bound == wire_bound(payload, k._state.lower, k._state.upper_in, unknown, inside)
    assert unknown > 0 and bound > 0


# ---------------------------------------------------------------- invalid cuts and receivers
def test_invalid_or_degenerate_split_planes():
    obs, k = prepared()
    h = max(k.open_handles(), key=k.volume)
    state = k._state
    for row in ((0, 0, 0, 1), (1.0, 0, 0, -1), (1, 0, 0), (1 << 4097, 0, 0, -1)):
        with pytest.raises(ValueError):
            k.split(h, row)
        assert k._state is state
    assert k.split(h, (1, 0, 0, 1)) == [h] and k._state is state  # plane misses the piece
    p = k.poly(h)
    a, b, c = p.vertices[:3]
    d1 = [b[i] * a[3] - a[i] * b[3] for i in range(3)]
    d2 = [c[i] * a[3] - a[i] * c[3] for i in range(3)]
    n = (d1[1] * d2[2] - d1[2] * d2[1], d1[2] * d2[0] - d1[0] * d2[2], d1[0] * d2[1] - d1[1] * d2[0])
    row = (n[0] * a[3], n[1] * a[3], n[2] * a[3], -(n[0] * a[0] + n[1] * a[1] + n[2] * a[2]))
    out = k.split(h, row) if any(n) else [h]  # plane through three vertices
    assert sum((k.volume(x) for x in out), Fraction(0)) == p.volume


def test_invalid_receivers_are_rejected():
    obs, k = prepared()
    h = max(k.open_handles(), key=k.volume)
    state = k._state
    L = obs.side
    solid = np.flatnonzero(k.receiver_classes().ravel() == 2)
    bad = [[-1], [L ** 3], [1.5]] + ([[int(solid[0])]] if len(solid) else [])
    for ids in bad:
        with pytest.raises((ValueError, TypeError)):
            k.enforce(h, ids)
        assert k._state is state
    with pytest.raises(ValueError):
        k.enforce(h, [])


# ---------------------------------------------------------------- evidence isolation
def test_returned_evidence_is_read_only():
    obs, k = prepared()
    for _ in range(20):
        h = max(k.open_handles(), key=k.volume)
        r = k.screen(h)
        if r["outcome"] == "open" and r["ids"] is not None and len(r["ids"]):
            break
    else:
        pytest.skip("no open screen")
    with pytest.raises(ValueError):
        r["ids"][0] = 0
    with pytest.raises(ValueError):
        r["masks"][0] = 0
    with pytest.raises(ValueError):
        k.receiver_classes()[0, 0, 0] = 1
    piece = k._state.live[r["handle"]]
    with pytest.raises(ValueError):
        piece.pending[0] = 0
    q = k.poly(r["handle"])
    q.volume = Fraction(0)  # a mutated copy must not affect the kernel's polytope
    assert k.volume(r["handle"]) > 0


def test_mutating_the_shared_input_after_construction_has_no_effect():
    obs = load()
    reference = Kernel(obs, Budget(120, 10 ** 6))
    victim = Kernel(obs, Budget(120, 10 ** 6))
    obs.bodies = ()                       # a buggy search rebinding the shared input
    obs.obstacles = []
    obs.clearance_covers = ([], [])
    obs.cls = np.zeros_like(obs.cls)
    for kk in (reference, victim):
        for d, a in choose_pairs(reference._obs.cls, 6, 1, "interface"):
            kk.condition_pair(d, a)
        kk.exclude_window()
    outs = []
    for kk in (reference, victim):
        seq = []
        for _ in range(8):
            opened = kk.open_handles()
            if not opened:
                break
            h = max(opened, key=kk.volume)
            r = kk.screen(h)
            seq.append((r["outcome"], r.get("ambiguous")))
        outs.append(seq)
    assert outs[0] == outs[1]


# ---------------------------------------------------------------- invalid structures, timeouts
def test_invalid_structure_at_credit_time_taints(monkeypatch):
    obs, k = prepared()

    def reject(p):
        raise ArithmeticError("face structure: injected failure")
    monkeypatch.setattr(polycheck, "check", reject)
    with pytest.raises(ArithmeticError):
        for _ in range(400):
            h = max(k.open_handles(), key=k.volume)
            r = k.screen(h)
            if r["outcome"] == "open" and r["ids"] is not None:
                k.enforce(r["handle"], [int(i) for i in r["ids"][:96]])
    rel = k.release(Fraction(1, 20))
    assert rel.status == "ERROR" and rel.payload is None
    with pytest.raises(RuntimeError):
        k.screen(k.handles()[0])


def test_timeout_inside_a_transition_leaves_the_state_unchanged():
    obs = load()
    k = Kernel(obs, Budget(120, 10 ** 6))
    for d, a in choose_pairs(obs.cls, 4, 1, "interface"):
        k.condition_pair(d, a)
    state = k._state
    k.budget = Budget(0, 10 ** 6)  # every check inside the next transition expires
    with pytest.raises(Limit):
        for d, a in choose_pairs(obs.cls, 8, 2, "interface"):
            k.condition_pair(d, a)
    assert k._state is state
    rel = k.release(Fraction(1, 20))
    assert rel.status in ("INCOMPLETE", "CERTIFIED")


def test_certified_bytes_decode_and_carry_the_bound():
    obs = load()
    from cbc_search import solve
    payload, d = solve(obs, "000030_w01_s0", Fraction(1, 20), 20)
    assert d["status"] == "CERTIFIED"
    p = decode(payload)
    assert sum(p.values()) == 1 and Fraction(d["tv"]) <= Fraction(1, 20)


# ---------------------------------------------------------------- static architecture check
UNTRUSTED = ("cbc_search.py", "hull_proposals.py", "shadow_facets.py")


@pytest.mark.parametrize("name", UNTRUSTED)
def test_untrusted_modules_never_touch_kernel_private_state(name):
    """No attribute access to a single-underscore name and no attribute assignment on the kernel."""
    tree = ast.parse((HERE / name).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr.startswith("_") and not node.attr.startswith("__"):
            raise AssertionError(f"{name}:{node.lineno}: private attribute access .{node.attr}")
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "kernel":
                    raise AssertionError(f"{name}:{node.lineno}: assignment to kernel.{t.attr}")
        if isinstance(node, ast.ImportFrom) and node.module == "cbc_kernel":
            allowed = {"certify"}
            assert {a.name for a in node.names} <= allowed, f"{name}: imports {[a.name for a in node.names]}"
