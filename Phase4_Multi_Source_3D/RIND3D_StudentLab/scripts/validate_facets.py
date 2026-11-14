"""Facet law check (diagnostic; uses the privileged true source only to score): for every extracted facet, the distance
of the true source to the facet plane, by facet size and side. usage: validate_facets.py [split] [--workers 12]
Prints a JSON summary; the facet law predicts ~0 distance for every facet that is a piece of one shadow boundary."""
import json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from build_facets import crops, load_rows, ROOT      # noqa: E402
from slab.facets import facets_of                    # noqa: E402

ROLE = {"train": 0, "dev": 1, "confirm": 2}
rows = load_rows()


def job(a):
    (ds, sc), idxs = a; get = crops(ROOT[ds] / f"{sc:06d}"); out = []
    for i in idxs:
        r = rows[i]; F, _ = facets_of(get(int(r["window"]), -1), get(int(r["window"]), int(r["source"])))
        s = np.array([r["sx"], r["sy"], r["sz"]]) * 1024 - np.array([r["ox"], r["oy"], r["oz"]])
        for f in F:
            out.append((int(r["Lp"]), f[6], abs(float((s - f[:3]) @ f[3:6]))))
    return out


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    sel = np.flatnonzero((rows["role"] == ROLE[split]) & (rows["ds"] == 1) & (rows["type"] == 0))
    g = {}
    for i in sel:
        g.setdefault((int(rows["ds"][i]), int(rows["scene"][i])), []).append(i)
    t0 = time.time()
    with Pool(12) as p:
        res = np.array([x for part in p.imap_unordered(job, list(g.items())) for x in part])
    out = dict(split=split, rows=int(len(sel)), facets=int(len(res)), wall=round(time.time() - t0))
    for L in (64, 128, 256):
        for lo, hi in ((24, 100), (100, 1000), (1000, 1e12)):
            m = (res[:, 0] == L) & (res[:, 1] >= lo) & (res[:, 1] < hi); d = res[m, 2]
            out[f"L{L}_inl{lo}"] = dict(n=int(m.sum()), median=float(np.median(d)) if m.any() else None,
                                       within2=float(np.mean(d < 2)) if m.any() else None, within8=float(np.mean(d < 8)) if m.any() else None)
    print(json.dumps(out, indent=1))
