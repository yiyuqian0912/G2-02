"""Diagnose the stored-lower-volume > cell-capacity records of the tau-pencil engine (ASTRA_MIXED_INTERFACE.md).

usage: diag_volume.py KEY [--cell ID]
Reruns the CPU tau-pencil label of one task (same call as the archived 'tp' labels), then for every record whose
lower volume exceeds the 128^3 cell capacity (and for the --cell record):
* the exact rational volume of the convex hull of the stored float vertices (Fractions, scipy hull combinatorics),
* whether the stored vertices are the exact cell corners,
* the float volume formula evaluated on those vertices, compared with the record.
Also: per cell, the exact dyadic sum of the record lower volumes against the capacity."""
import argparse, json, sys
from fractions import Fraction
from pathlib import Path

import numpy as np

CAP = Fraction(1, 128 ** 3)


def exact_hull_volume(V):
    from scipy.spatial import ConvexHull
    h = ConvexHull(V)
    P = [tuple(Fraction(float(x)) for x in v) for v in V]
    c = P[0]; vol = Fraction(0)
    for s in h.simplices:
        a, b, d = (P[i] for i in s)
        u = [a[k] - c[k] for k in range(3)]; v = [b[k] - c[k] for k in range(3)]; w = [d[k] - c[k] for k in range(3)]
        vol += abs(u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0]) + u[2] * (v[0] * w[1] - v[1] * w[0]))
    return vol / 6


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("key"); ap.add_argument("--cell", type=int, default=None)
    ap.add_argument("--tasks", default="/home/yushu/Projects/RIND/RIND3D_100K_v3_work/cmp_v3/tasks.json")
    a = ap.parse_args()
    t = next(x for x in json.loads(Path(a.tasks).read_text()) if x["key"] == a.key)
    from rind3d_posterior.fastlabel import window_from_100k
    from rind3d_posterior.fastlabel.window import window_classes
    from rind3d_posterior import label_window
    task = window_from_100k(t["container"], t["w"], t["s"])
    r = label_window(task.P, task.K, window_classes(task), task.origin, task.N)
    cell, lo, hi, kind, voff, vn, vpool = (r[k] for k in ("cell", "lo", "hi", "kind", "voff", "vn", "vpool"))
    print(f"records {len(cell)}  tv_bound {r['tv_bound']:.3e}  stats nodes={r['stats']['nodes']}")
    G = 128
    over = np.flatnonzero(lo > float(CAP))
    sel = list(over[:12]) + ([int(np.flatnonzero(cell == a.cell)[0])] if a.cell is not None and np.any(cell == a.cell) else [])
    print(f"single records with lo > capacity: {len(over)}")
    for i in sel:
        V = vpool[voff[i]:voff[i] + vn[i]]
        ijk = np.array([cell[i] // (G * G), (cell[i] // G) % G, cell[i] % G])
        corners = np.array([[(ijk[0] + dx) / G, (ijk[1] + dy) / G, (ijk[2] + dz) / G] for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)])
        on_corner = all(np.any(np.all(corners == v, axis=1)) for v in V)
        dev = max(float(np.min(np.abs(V[:, k] - np.round(V[:, k] * G) / G))) for k in range(3))
        ev = exact_hull_volume(V) if len(V) >= 4 else Fraction(0)
        print(json.dumps(dict(record=int(i), cell=int(cell[i]), kind=int(kind[i]), lo=float(lo[i]).hex(), hi=float(hi[i]).hex(),
                              n_vertices=int(vn[i]), vertices_are_cell_corners=bool(on_corner),
                              exact_hull_volume_over_cap=str(ev / CAP), lo_over_cap_minus_1=float(Fraction(float(lo[i])) / CAP - 1),
                              max_vertex_offgrid=dev)))
    # per-cell exact sums
    from collections import defaultdict
    sums = defaultdict(Fraction)
    for c, v in zip(cell.tolist(), lo.tolist()):
        sums[c] += Fraction(v)
    bad = [(c, float(s / CAP - 1)) for c, s in sums.items() if s > CAP]
    print(f"cells with exact sum of lower volumes > capacity: {len(bad)} of {len(sums)}; max excess (fraction of a cell) "
          f"{max((b for _, b in bad), default=0):.3e}")
    multi = sum(1 for c, _ in bad if np.sum(cell == c) > 1)
    print(f"  of which cells with more than one record: {multi}")


if __name__ == "__main__":
    main()
