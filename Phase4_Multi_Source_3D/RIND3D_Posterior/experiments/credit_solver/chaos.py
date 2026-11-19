"""Chaos search: soundness of the kernel must not depend on the search policy.

Drives the real trusted kernel with random legal API sequences (degenerate split planes
through vertices, voxel and cell planes, random receiver subsets, random pair conditioning,
any order), interleaved with the real greedy policy so that the runs reach meaningful
bounds. At every snapshot the released payload is compared with an independent
exact-certified reference label q_ref (bound b_ref):

    TV(payload, q_ref) <= bound + b_ref + 1e-12        (else: soundness violation)

It also attempts misuse that the kernel must reject without changing its state: writes
through returned arrays, stale handles, constant split rows, SOLID or out-of-window
receivers, screening before the window exclusion. Research instrument (not trusted).

  python chaos.py ROOT KEY --seeds 4 --seconds 20 --chaos 0.5
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np

import inputs
from cbc_kernel import Kernel
from compare import certified_reference, tv
from ledger import decode
from poly import primitive
from solver import Budget, Limit, choose_pairs


def through_vertices(poly, rng):
    """Plane through three vertices of the piece (maximally degenerate cut)."""
    vs = list(poly.vertices)
    for _ in range(20):
        a, b, c = rng.sample(vs, 3)
        d1 = [b[i] * a[3] - a[i] * b[3] for i in range(3)]
        d2 = [c[i] * a[3] - a[i] * c[3] for i in range(3)]
        n = (d1[1] * d2[2] - d1[2] * d2[1], d1[2] * d2[0] - d1[0] * d2[2], d1[0] * d2[1] - d1[1] * d2[0])
        if any(n):
            return primitive((n[0] * a[3], n[1] * a[3], n[2] * a[3], -(n[0] * a[0] + n[1] * a[1] + n[2] * a[2])))
    return None


def random_plane(poly, rng):
    kind = rng.choice(("vertices", "voxel", "cell", "small", "dyadic"))
    vs = poly.vertices
    if kind == "vertices":
        return through_vertices(poly, rng), kind
    k = rng.randrange(3)
    lo = min(Fraction(v[k], v[3]) for v in vs)
    hi = max(Fraction(v[k], v[3]) for v in vs)
    row = [0, 0, 0, 0]
    if kind in ("voxel", "cell"):
        n = 1024 if kind == "voxel" else 128
        a, b = int(lo * n), int(hi * n) + 1
        m = rng.randint(a, b)  # may miss the piece or touch a vertex exactly
        row[k], row[3] = n, -m
        return tuple(row), kind
    if kind == "dyadic":
        t = lo + (hi - lo) * Fraction(rng.randrange(1, 64), 64)
        row[k], row[3] = t.denominator, -t.numerator
        return tuple(row), kind
    # small integer normal through a random convex combination of two vertices
    a, b = rng.sample(list(vs), 2)
    x = [Fraction(a[i], a[3]) / 2 + Fraction(b[i], b[3]) / 2 for i in range(3)]
    nrm = [rng.randint(-3, 3) for _ in range(3)]
    if not any(nrm):
        nrm[0] = 1
    d = -sum(n * xi for n, xi in zip(nrm, x))
    return primitive(tuple(n * d.denominator for n in nrm) + (d.numerator,)), kind


def misuse(kernel, obs, rng, last_ids, counts):
    """Each attempt must raise and leave the state object unchanged."""
    before = kernel._state
    L = obs.side
    attempts = []
    if last_ids is not None and len(last_ids):
        def write():
            last_ids[0] = last_ids[0] + 1
        attempts.append(("write_view", write, ValueError))
    attempts.append(("constant_split", lambda: kernel.split(next(iter(kernel.handles())), (0, 0, 0, 1)), ValueError))
    attempts.append(("stale_handle", lambda: kernel.screen(-12345), KeyError))
    solid = np.flatnonzero(obs.scancls.ravel() == 2)
    opened = kernel.open_handles()
    if len(solid) and opened:
        attempts.append(("solid_receiver", lambda: kernel.enforce(opened[0], [int(solid[0])]), ValueError))
    if opened:
        attempts.append(("bad_receiver", lambda: kernel.enforce(opened[0], [L ** 3]), ValueError))
    name, fn, exc = rng.choice(attempts)
    try:
        fn()
    except exc:
        counts[name] = counts.get(name, 0) + 1
    else:
        raise AssertionError(f"misuse accepted: {name}")
    if kernel._state is not before:
        raise AssertionError(f"misuse changed the state: {name}")


def greedy_step(kernel, obs, h, r):
    ids, wrong, bnd = r["ids"], r["wrong"], r["boundary"]
    score = wrong.astype(np.int64) * 2 + bnd.astype(np.int64)
    order = np.argsort(-score, kind="stable")
    chosen = [int(ids[i]) for i in order[:96]]
    kernel.enforce(h, chosen)


def run(obs, ref, seed, seconds, chaos):
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    kernel = Kernel(obs, Budget(seconds, 20000))
    checks, worst, viol = 0, None, []
    counts, ops = {}, {}
    free = np.flatnonzero(obs.scancls.ravel() != 2)
    dark = [tuple(int(t) for t in np.unravel_index(i, obs.cls.shape)) for i in np.flatnonzero(obs.cls.ravel() == 0)]
    lit = [tuple(int(t) for t in np.unravel_index(i, obs.cls.shape)) for i in np.flatnonzero(obs.cls.ravel() == 1)]
    last = {}

    def check():
        nonlocal checks, worst
        payload, bound = kernel.snapshot()
        if payload is None:
            return
        d = tv(decode(payload), ref["q"])
        checks += 1
        slack = float(bound) + ref["bound"] + 1e-12 - d
        if worst is None or slack < worst[0]:
            worst = (slack, float(bound), d)
        if slack < 0:
            viol.append(dict(bound=float(bound), tv_ref=d))
    try:
        try:
            kernel.screen(0)
        except ValueError:
            counts["screen_before_exclusion"] = 1
        pairs = choose_pairs(obs.cls, 24, seed, "interface")
        for d_, a_ in pairs[:rng.randint(0, 24)]:
            kernel.condition_pair(d_, a_)
        for _ in range(rng.randint(0, 3)):  # random (non-interface) pairs
            if dark and lit:
                kernel.condition_pair(rng.choice(dark), rng.choice(lit))
        kernel.exclude_window()
        while True:
            opened = kernel.open_handles()
            if not opened:
                break
            if rng.random() < 0.03:
                misuse(kernel, obs, rng, last.get("ids"), counts)
            if rng.random() < chaos:
                h = rng.choice(opened)
                act = rng.choice(("screen", "split", "split", "enforce_random", "enforce_pending", "snapshot"))
            else:
                h = max(opened, key=lambda x: float(kernel.volume(x)))
                act = "greedy"
            ops[act] = ops.get(act, 0) + 1
            if act == "snapshot":
                check()
            elif act == "split":
                row, kind = random_plane(kernel.poly(h), rng)
                if row is not None and any(row[:3]):
                    try:
                        kernel.split(h, row)
                    except ValueError:  # oversize proposal (> 2^4096): rejected, state unchanged
                        counts["oversize_split"] = counts.get("oversize_split", 0) + 1
            elif act == "enforce_random":
                sel = nrng.choice(free, size=min(len(free), rng.randint(1, 40)), replace=True)
                kernel.enforce(h, [int(i) for i in sel])
            elif act == "enforce_pending":
                ids = last.get(h)
                if ids is not None and len(ids):
                    k = rng.randint(1, min(len(ids), 120))
                    kernel.enforce(h, [int(i) for i in nrng.choice(ids, size=k, replace=False)])
            else:  # screen, then (greedy) enforce
                r = kernel.screen(h)
                if r["outcome"] == "open" and r["ids"] is not None:
                    last[r["handle"]] = r["ids"]
                    last["ids"] = r["ids"]
                    if act == "greedy":
                        greedy_step(kernel, obs, r["handle"], r)
                elif r["outcome"] == "open" and act == "greedy":
                    row, _ = random_plane(kernel.poly(r["handle"]), rng)
                    if row is not None and any(row[:3]):
                        try:
                            kernel.split(r["handle"], row)
                        except ValueError:
                            counts["oversize_split"] = counts.get("oversize_split", 0) + 1
                if r["outcome"] == "in":
                    check()
    except Limit:
        pass
    check()
    rel = kernel.release(Fraction(1))
    if kernel.positive and rel.status != "CERTIFIED":
        viol.append(dict(release=rel.status, reason=rel.reason))
    if rel.status == "CERTIFIED":
        d = tv(decode(rel.payload), ref["q"])
        if d > float(rel.bound) + ref["bound"] + 1e-12:
            viol.append(dict(release_bound=float(rel.bound), tv_ref=d))
    return dict(seed=seed, checks=checks, worst_slack=worst, violations=viol, misuse=counts, ops=ops,
                final_bound=None if rel.bound is None else float(rel.bound), stats=dict(kernel.stats),
                tainted=kernel._tainted)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("key")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--chaos", type=float, default=0.5)
    args = ap.parse_args()
    slot, w, s = int(args.key[:6]), int(args.key[8:10]), int(args.key[12])
    obs, ident = inputs.load(args.root / "scenes" / f"{slot // 1000:04d}" / f"{slot:06d}.npz", w, s)
    ref = certified_reference(args.key)
    if ref is None:
        raise SystemExit("no exact-certified reference")
    for seed in range(args.seeds):
        t = perf_counter()
        obs, _ = inputs.load(args.root / "scenes" / f"{slot // 1000:04d}" / f"{slot:06d}.npz", w, s)
        res = run(obs, ref, seed, args.seconds, args.chaos)
        res.update(key=args.key, L=ident["side"], ref_bound=ref["bound"], wall=perf_counter() - t)
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
