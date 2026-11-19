"""Exact differential soundness audit of the CBC v2 kernel on real tasks (research instrument).

For one task: run the v2 search with a recording kernel, then
  * IN soundness:   sample exact rational points in every credited IN piece; each must be
                    legal and reproduce every FREE receiver (exactcheck.Oracle);
  * drop soundness: in every transition, sample exact points of the consumed parents that
                    lie in no child, IN piece or deferred shell; none may be feasible.
Prints one JSON line. A violation is a soundness counterexample of the kernel.
"""
import argparse
from fractions import Fraction
import json
from math import lcm
from pathlib import Path
import random
import sys
from time import perf_counter

import inputs
import cbc_kernel
import cbc_search
from diag import halfspaces, contains
from exactcheck import Oracle


class Recorder(cbc_kernel.Kernel):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.log, self.in_parts, self.kind = [], [], None

    def _commit(self, remove, children, lower=None, upper=None, exclude=False):
        parents = [self._state.live[k].poly for k in remove]
        out = super()._commit(remove, children, lower, upper, exclude)
        self.log.append((self.kind, parents, [c.poly for c in children]))
        return out

    def _grid_credit(self, parts):
        self.in_parts.extend(parts)
        return super()._grid_credit(parts)


for _name in ("condition_pair", "exclude_window", "split", "screen", "enforce"):
    def _wrap(name):
        base = getattr(cbc_kernel.Kernel, name)

        def method(self, *a, **k):
            self.kind = name
            return base(self, *a, **k)
        return method
    setattr(Recorder, _name, _wrap(_name))


def points(poly, rng, k, rows=None):
    """Exact sample points of poly, snapped to a dyadic grid (2^-64 .. 2^-160) while staying inside."""
    verts = [tuple(Fraction(v[i], v[3]) for i in range(3)) for v in poly.vertices]
    rows = rows if rows is not None else halfspaces(poly)
    out = []
    for j in range(k):
        w = [Fraction(rng.randint(1, 1 << 16)) for _ in verts]
        if j % 3 == 1:  # concentrate near one vertex
            w[rng.randrange(len(w))] *= 1 << 12
        t = sum(w)
        x = tuple(sum(wi * v[i] for wi, v in zip(w, verts)) / t for i in range(3))
        for bits in (64, 96, 128, 160):
            sc = 1 << bits
            y = tuple(Fraction(round(c * sc), sc) for c in x)
            if contains(rows, hom(y)):
                out.append(y)
                break
    return out


def hom(p):
    den = lcm(*(c.denominator for c in p))
    return tuple(int(c * den) for c in p) + (den,)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("container", type=Path)
    ap.add_argument("window", type=int)
    ap.add_argument("channel", type=int)
    ap.add_argument("output", type=Path, nargs="?", default=None)
    ap.add_argument("--seconds", type=float, default=8)
    ap.add_argument("--in-samples", type=int, default=6)
    ap.add_argument("--drop-samples", type=int, default=6)
    ap.add_argument("--max-drop-checks", type=int, default=150)
    args = ap.parse_args()
    obs, ident = inputs.load(args.container, args.window, args.channel)
    key = f"{ident['slot']:06d}_w{args.window:02d}_s{args.channel}"
    holder = {}
    orig = cbc_kernel.Kernel

    class K(Recorder):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            holder["k"] = self
    cbc_search.Kernel = K
    t0 = perf_counter()
    payload, det = cbc_search.solve(obs, key, Fraction(1, 20), args.seconds)
    cbc_search.Kernel = orig
    kern = holder["k"]
    oracle = Oracle(obs.bodies, obs.cls, obs.origin)
    rng = random.Random(7)
    res = dict(key=key, status=det["status"], tv=det["tv"] and float(Fraction(det["tv"])), in_pieces=len(kern.in_parts),
               in_checked=0, in_illegal=0, in_mismatch=0, drop_checked=0, drop_feasible=0, drop_by_kind={},
               rechecks=0, violations=[])
    for i, p in enumerate(kern.in_parts):
        for s in points(p, rng, args.in_samples):
            ok, mism, rc = oracle.feasible(s)
            res["in_checked"] += 1
            res["rechecks"] += rc
            if not ok:
                res["in_illegal"] += 1
                res["violations"].append(dict(kind="IN_illegal", piece=i))
            elif mism:
                res["in_mismatch"] += 1
                res["violations"].append(dict(kind="IN_mismatch", piece=i, mismatches=mism))
    in_rows = [halfspaces(p) for p in kern.in_parts]
    pairs = [(t, j) for t, (_, parents, _) in enumerate(kern.log) for j in range(len(parents))]
    rng.shuffle(pairs)
    kinds = sorted({kern.log[t][0] for t, _ in pairs})
    quota = {k: max(4, args.max_drop_checks // max(1, len(kinds))) for k in kinds}
    got = {k: 0 for k in kinds}
    cand, tries, rows_cache = [], 0, {}
    for t, j in pairs:
        if len(cand) >= args.max_drop_checks or tries > 40 * args.max_drop_checks:
            break
        if got[kern.log[t][0]] >= quota[kern.log[t][0]]:
            continue
        kind, parents, children = kern.log[t]
        if t not in rows_cache:
            rows_cache = {t: [halfspaces(c) for c in children]}
        crow = rows_cache[t]
        for s in points(parents[j], rng, args.drop_samples):
            tries += 1
            h = hom(s)
            if any(contains(r, h) for r in crow) or any(contains(r, h) for r in in_rows):
                continue
            cand.append((t, kind, s))
            got[kind] += 1
    res["drop_candidates_tried"] = tries
    for t, kind, s in cand[:args.max_drop_checks]:
        ok, mism, rc = oracle.feasible(s)
        res["drop_checked"] += 1
        res["rechecks"] += rc
        res["drop_by_kind"][kind] = res["drop_by_kind"].get(kind, 0) + 1
        if ok and mism == 0:
            res["drop_feasible"] += 1
            res["violations"].append(dict(kind="DROP_feasible", transition=t, op=kind,
                                          point=[str(x) for x in s]))
    res["seconds"] = perf_counter() - t0
    if args.output is not None:
        args.output.write_text(json.dumps(res))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
