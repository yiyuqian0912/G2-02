"""Construction cost of the E2b root clips: brute-force box_explanations versus the shared-octree explanations_for_roots,
on the depth-4 stage-1 roots of every task (same interface edges as e2_clip.py; equality of the summaries checked).

usage: e2_clip_cost.py --tasks tasks.json --out OUT [--n-edges 256]"""
import argparse, json, sys, time, zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve(); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[3] / "RIND3D_Dataset100K"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tasks", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-edges", type=int, default=256); a = ap.parse_args()
    from rind3d.rendering.backends import BackendFactory
    from rind3d_1k.physical import POLICY
    from rind3d_100k.storage import SceneFile
    from rind3d_posterior.fastlabel import engine as T, window_from_100k
    from rind3d_posterior.fastlabel.window import window_classes, LazyWindow2
    from rind3d_posterior.localfirst.local import interface_edges, plane_components, edge_free_mask
    from rind3d_posterior.localfirst.pipeline import DEFAULTS as CFG
    from rind3d_posterior.localfirst import certregion as CR
    from e12 import select_edges
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); fp = out / "e2_clip_cost.jsonl"
    done = {json.loads(l)["key"] for l in fp.read_text().splitlines() if l.strip()} if fp.exists() else set()
    fac = BackendFactory(cuda_allowed=False)
    for t in json.loads(Path(a.tasks).read_text()):
        if t["key"] in done:
            continue
        task = window_from_100k(t["container"], t["w"], t["s"]); S = SceneFile(t["container"])
        rows = [c.rows for c in fac.context(S.geometry(), POLICY).components]
        geom = T.Geom(np.asarray(task.P, float), np.asarray(task.K))
        cls = window_classes(task); o = np.asarray(task.origin, np.int64); N = int(task.N); L = int(task.L)
        win = LazyWindow2(cls, o, N)
        lit, dark = interface_edges(cls); comps = plane_components(lit, dark, seed=CFG["seed"])
        roots, _ = T.stage1(geom, win, d1=CFG["s1_d1"], n_boundary=CFG["s1_nb"], n_class=CFG["s1_nc"], n_learn=CFG["s1_learn"], max_tests=CFG["s1_tests"])
        free = edge_free_mask(rows, o, N, lit, dark)
        rng = np.random.default_rng([20261118, zlib.crc32(t["key"].encode()), zlib.crc32(b"independent"), a.n_edges])
        I = select_edges(lit, dark, free, comps, a.n_edges, "independent", rng)
        rec = dict(key=t["key"], family=t["family"], L=t["L"], roots=int(len(roots)), used_edges=int(len(I)))
        if len(I) == 0:
            rec["empty"] = True
        else:
            Lp = (o + lit[I] + .5) / N; Kp = (o + dark[I] + .5) / N
            EA, EB, Fr, _ = CR.world_edges(np.asarray(task.P, float), np.asarray(task.K)); off, idx = CR.admissible_edges(Fr, Lp, Kp)
            h = 1.0 / (1 << roots[:, 0]); lo = roots[:, 1:] * h[:, None]
            CR.box_explanations(EA, EB, Lp[:1], Kp[:1], off[:2], idx, lo[:1], h[:1], CR.MARGIN)       # JIT warm-up
            CR.explanations_for_roots(EA, EB, Lp[:1], Kp[:1], off[:2], idx, roots[:1], o / N, (o + L) / N)
            tt = time.perf_counter(); bf = CR.box_explanations(EA, EB, Lp, Kp, off, idx, lo, h, CR.MARGIN); rec["t_brute"] = time.perf_counter() - tt
            ex, rec["t_octree"] = CR.explanations_for_roots(EA, EB, Lp, Kp, off, idx, roots, o / N, (o + L) / N)
            ok = True
            for q in range(len(roots)):
                if np.all(ex[q] == -1):
                    ok &= bool(np.any(bf[q] == -1))
                elif not np.array_equal(ex[q], bf[q]):
                    ok &= bool(np.all(ex[q] == -2))                               # capacity-stopped root: no clip, sound
            rec["equal"] = bool(ok)
            tt = time.perf_counter(); CR.root_clips(EA, EB, Lp, Kp, ex); rec["t_rows"] = time.perf_counter() - tt
        print(json.dumps(rec), flush=True)
        with open(fp, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
