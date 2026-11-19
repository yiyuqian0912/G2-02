"""Soundness hunt: find a discarded source that reproduces the observation, then locate the transition.

Research instrument. 1) Replay the run with diag.Recorder. 2) Collect discarded sample points
in the target cells, rendered with the native renderer. 3) Hill-climb on the mismatch count
by small exact rational moves. 4) For a zero-mismatch point, report every transition whose
parent contained it while no child or IN piece did, and evaluate the receivers that the
scan or cut would have used, with exact rational segment clipping.
"""
import argparse
from fractions import Fraction
import io
from itertools import product
import json
from pathlib import Path
import random

import numpy as np

import diag
from diag import Recorder, halfspaces, contains, cell_rows, sample
import inputs
import run as runmod
from run import search
from poly import within


def renderer(container, obs):
    from rind3d.geometry import load_geometry
    from rind3d.rendering.context import GeometryContext
    from rind3d_posterior.fastlabel.inputs import POLICY, legal_source
    with np.load(container, allow_pickle=False) as z:
        geom = load_geometry(io.BytesIO(z["geometry"].tobytes()))
    ctx = GeometryContext(geom, POLICY)
    n = obs.denom // 2

    def mism(pt, where=False):
        x = np.array([pt[i] / pt[3] for i in range(3)], dtype="<f8")
        if legal_source(ctx, x) is None:
            return None
        model = ctx.bind(x)
        errs = 0
        locs = []
        for loc in product(range(0, obs.side, 64), repeat=3):
            o = tuple(a + b for a, b in zip(obs.origin, loc))
            y, m, _ = model.render(o, 64, n)
            ob = obs.cls[tuple(slice(a, a + 64) for a in loc)].T
            bad = (y != (ob == 1)) & (ob != 2)
            errs += int(np.count_nonzero(bad)) + int(np.count_nonzero(m != (ob == 2)))
            if where:
                for zyx in np.argwhere(bad):
                    locs.append(tuple(int(loc[k] + zyx[2 - k]) for k in range(3)))
        return (errs, locs) if where else errs
    return mism


def exact_hit(body, s, r):
    """Closed segment [s,r] meets {f<=0}: exact parametric clipping (s,r homogeneous ints)."""
    lo, hi = Fraction(0), Fraction(1)
    for row in body:
        fs = Fraction(sum(a * x for a, x in zip(row[:3], s[:3])) + row[3] * s[3], s[3])
        fr = Fraction(sum(a * x for a, x in zip(row[:3], r[:3])) + row[3] * r[3], r[3])
        d = fr - fs
        if d == 0:
            if fs > 0:
                return False
        elif d > 0:
            hi = min(hi, -fs / d)
        else:
            lo = max(lo, -fs / d)
        if lo > hi:
            return False
    return lo <= hi


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("container", type=Path)
    ap.add_argument("window", type=int)
    ap.add_argument("channel", type=int)
    ap.add_argument("--cells", type=int, nargs="+", required=True)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--starts", type=int, default=40)
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    obs, ident = inputs.load(args.container, args.window, args.channel)
    key = f"{ident['slot']:06d}_w{args.window:02d}_s{args.channel}"
    runmod.Kernel = Recorder
    search(obs, key, Fraction(1, 20), 8, 192, 24, "interface", "largest")
    mism = renderer(args.container, obs)
    rng = random.Random(3)
    in_rows = [halfspaces(p) for p in Recorder.in_parts]
    logs = [(t, tr, [halfspaces(c) for c in tr["children"]], [halfspaces(p) for p in tr["parents"]])
            for t, tr in enumerate(Recorder.log)]
    cand = []
    for t, tr, crow, prow in logs:
        for parent in tr["parents"]:
            for cid in args.cells:
                q = within(parent, cell_rows(cid))
                if q is None:
                    continue
                for pt in sample(q, rng, args.samples):
                    if not any(contains(r, pt) for r in crow) and not any(contains(r, pt) for r in in_rows):
                        cand.append(pt)
    rng.shuffle(cand)
    scored = []
    for pt in cand[:args.starts]:
        e = mism(pt)
        if e is not None:
            scored.append((e, pt))
    scored.sort(key=lambda x: x[0])
    print("start mismatches", [e for e, _ in scored[:10]])
    best = None
    for e0, pt in scored[:5]:
        cur, ce = pt, e0
        step = Fraction(1, 1 << 30)
        for it in range(args.steps):
            if ce == 0:
                break
            d = tuple(int(rng.choice((-1, 0, 1)) * rng.randint(1, 8)) for _ in range(3))
            x = tuple(Fraction(cur[i], cur[3]) + d[i] * step for i in range(3))
            from math import lcm
            den = lcm(*(c.denominator for c in x))
            nxt = tuple(int(c * den) for c in x) + (den,)
            e = mism(nxt)
            if e is not None and e <= ce:
                cur, ce = nxt, e
            if it % 50 == 49:
                step *= 2 if ce > 0 else 1
        print("climb", e0, "->", ce)
        if ce == 0:
            best = cur
            break
    if best is None:
        print("no feasible discarded point found")
        return
    x = [float(Fraction(best[i], best[3])) for i in range(3)]
    print("FEASIBLE (native render) point", x)
    owner = [t for t, tr, crow, prow in logs if any(contains(r, best) for r in prow)
             and not any(contains(r, best) for r in crow)]
    print("parents containing it, no child:", [(t, Recorder.log[t]["kind"]) for t in owner])
    print("in IN pieces:", any(contains(r, best) for r in in_rows))
    json.dump(dict(point=[str(Fraction(best[i], best[3])) for i in range(3)], owner=owner),
              open(f"/tmp/claude-1000/-home-yushu-Projects-RIND/eeba4904-c162-43fd-88e4-90e670b669b8/scratchpad/hunt_{key}.json", "w"))


if __name__ == "__main__":
    main()
