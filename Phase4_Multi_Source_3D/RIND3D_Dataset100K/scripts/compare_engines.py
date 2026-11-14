#!/usr/bin/env python3
"""FastLabel vs exact tau-pencil on labelled tasks of a RIND3D-100K root (accuracy gate of the canary).

  compare_engines.py <root> [--n 200] [--seed 0] [--max-L 128] [--workers 8] [--out ENGINES.json]

Samples consolidated FastLabel labels stratified by (side, status), recomputes each task with the exact tau-pencil
CPU engine (`rind3d_posterior.label_window`, the reference) and reports TV(q_FastLabel, q_exact) with q_exact the
interval midpoint. A task passes iff TV <= tv_bound(FastLabel) + tv_bound(exact). The 1K reference: 447 tasks, max TV
0.0164. L256 exact labels can take minutes each; --max-L limits the sides (default 128). CPU only: run it on the
canary root after the labels are consolidated."""
import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parent
sys.path[:0] = [str(HERE), str(REPO / "RIND3D_Dataset1K"), str(REPO / "RIND3D_Posterior" / "src")]
import numpy as np  # noqa: E402


def _one(args):
    root, slot, w, j = args
    from rind3d_100k.loader import Scene
    from rind3d_posterior.fastlabel import window_from_100k
    from rind3d_posterior.fastlabel.window import window_classes
    from rind3d_100k.storage import scene_path
    import rind3d_posterior as RP
    S = Scene(root, slot); lab = S.posterior(w, j)
    t = window_from_100k(scene_path(root, slot), w, j)
    t0 = time.perf_counter()
    ex = RP.label_window(np.asarray(t.P, float), np.asarray(t.K), window_classes(t), np.asarray(t.origin), t.N, engine="cpu")
    u, inv = np.unique(np.asarray(ex["cell"], np.int64), return_inverse=True)
    mid = np.bincount(inv, weights=0.5 * (np.asarray(ex["lo"]) + np.asarray(ex["hi"])))
    qa = dict(zip(lab["cells"].tolist(), lab["q"].tolist())); qb = dict(zip(u.tolist(), (mid / mid.sum()).tolist()))
    tv = 0.5 * sum(abs(qa.get(c, 0.0) - qb.get(c, 0.0)) for c in set(qa) | set(qb))
    bound = float(lab["tv_bound"]) + float(ex["tv_bound"])
    return dict(slot=slot, window=w, source=j, L=int(S.S.W[w]["Lp"]), status=lab["status"], tv=tv, bound=bound,
                ok=tv <= bound, exact_tv_bound=float(ex["tv_bound"]), exact_s=time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max-L", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--out"); a = ap.parse_args()
    from rind3d_100k.loader import RIND3D100K, Scene
    ds = RIND3D100K(a.root); strata = defaultdict(list)
    for slot in ds.slots:
        S = Scene(a.root, slot)
        if S.lab is None:
            continue
        for m in S.lab["metas"]:
            if m.get("engine") == "fastlabel" and int(m["L"]) <= a.max_L and m["status"] in ("certified", "certified_stat"):
                strata[(int(m["L"]), m["status"])].append((str(a.root), slot, int(m["window"]), int(m["source"])))
    rng = np.random.default_rng(a.seed); jobs = []
    per = max(1, a.n // max(len(strata), 1))
    for key in sorted(strata):
        v = strata[key]; jobs += [v[i] for i in sorted(rng.choice(len(v), min(per, len(v)), replace=False))]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        res = pool.map(_one, jobs, chunksize=1)
    tv = np.array([r["tv"] for r in res]) if res else np.zeros(0)
    out = dict(root=str(a.root), n=len(res), strata={f"L{k[0]}:{k[1]}": len(v) for k, v in sorted(strata.items())},
               all_within_bounds=all(r["ok"] for r in res), max_tv=float(tv.max()) if len(tv) else None,
               quantiles={q: float(np.quantile(tv, q)) for q in (0.5, 0.9, 0.99)} if len(tv) else None,
               failures=[r for r in res if not r["ok"]][:20], results=res)
    s = json.dumps(out, indent=1, default=float)
    Path(a.out or Path(a.root) / "ENGINES.json").write_text(s)
    print(json.dumps({k: out[k] for k in ("n", "strata", "all_within_bounds", "max_tv", "quantiles")}, default=float))
    sys.exit(0 if out["all_within_bounds"] else 1)


if __name__ == "__main__":
    main()
