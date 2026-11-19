"""Diagnostic replay of one credit-solver run: record every transition, then test discarded volume.

Research instrument (not part of any certificate). Runs the same search as run.py
with a recording kernel. For given output cells, it samples exact rational points of
each consumed parent inside the cell, keeps the points not covered by any child or
IN credit (i.e. volume the transition declared infeasible), and re-renders each such
point with the native renderer of the generator to test whether it reproduces the
complete observation. A discarded point that reproduces the observation and passes
native clearance is a soundness counterexample.
"""
import argparse
from fractions import Fraction
import io
from itertools import product
import json
from pathlib import Path
import random
import sys

import numpy as np

import inputs
from kernel import Kernel, Piece, cell_id
from poly import Poly, form, primitive, within
from run import search
import run as runmod


def halfspaces(p):
    rows = []
    for face in p.faces:
        a = face[0]
        def d(b):
            return tuple(b[i] * a[3] - a[i] * b[3] for i in range(3))
        u = d(face[1])
        for b in face[2:]:
            v = d(b)
            n = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
            if any(n):
                break
        else:
            continue
        row = primitive((*(x * a[3] for x in n), -sum(n[i] * a[i] for i in range(3))))
        # orient so that the polytope is on the nonpositive side
        if any(form(row, q) > 0 for q in p.vertices):
            row = tuple(-x for x in row)
        rows.append(row)
    return rows


def contains(rows, pt):
    return all(form(r, pt) <= 0 for r in rows)


def cell_rows(cid, n=128):
    i, j, k = cid // n**2, cid // n % n, cid % n
    rows = []
    for axis, v in enumerate((i, j, k)):
        e = [0, 0, 0]
        e[axis] = n
        rows.append((*[-x for x in e], v))
        rows.append((*e, -(v + 1)))
    return rows


def sample(p, rng, count):
    """Exact rational points: random convex combinations of vertices (not uniform)."""
    verts = [tuple(Fraction(v[i], v[3]) for i in range(3)) for v in p.vertices]
    out = []
    for _ in range(count):
        w = [Fraction(rng.randint(1, 1 << 20)) for _ in verts]
        if rng.random() < 0.5:  # bias toward a few vertices to reach corners
            for i in range(len(w)):
                if rng.random() < 0.7:
                    w[i] = Fraction(1)
        t = sum(w)
        x = tuple(sum(wi * v[k] for wi, v in zip(w, verts)) / t for k in range(3))
        from math import lcm
        den = lcm(*(c.denominator for c in x))
        out.append(tuple(int(c * den) for c in x) + (den,))
    return out


class Recorder(Kernel):
    log = []

    def _commit(self, remove, children, lower=None, upper=None, exclude=False):
        parents = [self._state.live[k].poly for k in remove]
        super()._commit(remove, children, lower, upper, exclude)
        Recorder.log.append(dict(parents=parents, children=[c.poly for c in children],
                                 kind=Recorder.kind))

    def condition_pair(self, dark, lit):
        Recorder.kind = "pair"
        return super().condition_pair(dark, lit)

    def exclude_window(self):
        Recorder.kind = "window"
        return super().exclude_window()

    def advance(self, handle):
        Recorder.kind = "advance"
        return super().advance(handle)

    def _grid_credit(self, parts):
        Recorder.in_parts.extend(parts)
        return super()._grid_credit(parts)


Recorder.kind = None
Recorder.in_parts = []


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("container", type=Path)
    ap.add_argument("window", type=int)
    ap.add_argument("channel", type=int)
    ap.add_argument("--cells", type=int, nargs="+", required=True)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--seconds", type=float, default=8)
    ap.add_argument("--render", type=int, default=20, help="max native renders")
    args = ap.parse_args()
    obs, ident = inputs.load(args.container, args.window, args.channel)
    key = f"{ident['slot']:06d}_w{args.window:02d}_s{args.channel}"
    runmod.Kernel = Recorder
    payload, details = search(obs, key, Fraction(1, 20), args.seconds, 192, 24, "interface", "largest")
    print("run", details["status"], details["tv"] and float(Fraction(details["tv"])), "transitions", len(Recorder.log))
    rng = random.Random(1)
    in_rows = [halfspaces(p) for p in Recorder.in_parts]
    dropped = []
    for t, tr in enumerate(Recorder.log):
        child_rows = [halfspaces(c) for c in tr["children"]]
        for parent in tr["parents"]:
            for cid in args.cells:
                q = within(parent, cell_rows(cid))
                if q is None:
                    continue
                for pt in sample(q, rng, args.samples):
                    if any(contains(r, pt) for r in child_rows) or any(contains(r, pt) for r in in_rows):
                        continue
                    dropped.append((t, tr["kind"], cid, pt))
    print("dropped samples in target cells:", len(dropped))
    from collections import Counter
    print(Counter((k, c) for _, k, c, _ in dropped))
    if not dropped:
        return
    # Native forward check (independent parametric clipping renderer).
    from rind3d.geometry import load_geometry
    from rind3d.rendering.context import GeometryContext
    from rind3d_posterior.fastlabel.inputs import POLICY, legal_source
    with np.load(args.container, allow_pickle=False) as z:
        geom = load_geometry(io.BytesIO(z["geometry"].tobytes()))
    ctx = GeometryContext(geom, POLICY)
    n = obs.denom // 2
    rng.shuffle(dropped)
    found = 0
    for t, kind, cid, pt in dropped[:args.render]:
        x = np.array([pt[i] / pt[3] for i in range(3)], dtype="<f8")
        if legal_source(ctx, x) is None:
            print("  illegal (clearance)", t, kind, cid)
            continue
        model = ctx.bind(x)
        errs = 0
        for loc in product(range(0, obs.side, 64), repeat=3):
            o = tuple(a + b for a, b in zip(obs.origin, loc))
            y, m, _ = model.render(o, 64, n)
            ob = obs.cls[tuple(slice(a, a + 64) for a in loc)].T
            errs += int(np.count_nonzero(y != (ob == 1))) + int(np.count_nonzero(m != (ob == 2)))
        print("  transition", t, kind, "cell", cid, "render mismatches", errs)
        found += errs == 0
    print("feasible discarded points:", found)


if __name__ == "__main__":
    main()


def check_in(container, window, channel, per_piece=5, seconds=8):
    """Render random exact interior points of every IN piece; all must reproduce the observation."""
    obs, ident = inputs.load(container, window, channel)
    key = f"{ident['slot']:06d}_w{window:02d}_s{channel}"
    runmod.Kernel = Recorder
    Recorder.in_parts.clear()
    payload, details = search(obs, key, Fraction(1, 20), seconds, 192, 24, "interface", "largest")
    from rind3d.geometry import load_geometry
    from rind3d.rendering.context import GeometryContext
    from rind3d_posterior.fastlabel.inputs import POLICY, legal_source
    with np.load(container, allow_pickle=False) as z:
        geom = load_geometry(io.BytesIO(z["geometry"].tobytes()))
    ctx = GeometryContext(geom, POLICY)
    n = obs.denom // 2
    rng = random.Random(2)
    res = []
    for i, p in enumerate(Recorder.in_parts):
        for pt in sample(p, rng, per_piece):
            x = np.array([pt[k] / pt[3] for k in range(3)], dtype="<f8")
            legal = legal_source(ctx, x) is not None
            model = ctx.bind(x)
            errs = 0
            for loc in product(range(0, obs.side, 64), repeat=3):
                o = tuple(a + b for a, b in zip(obs.origin, loc))
                y, m, _ = model.render(o, 64, n)
                ob = obs.cls[tuple(slice(a, a + 64) for a in loc)].T
                errs += int(np.count_nonzero(y != (ob == 1))) + int(np.count_nonzero(m != (ob == 2)))
            res.append((i, float(p.volume), legal, errs))
    return details, res
