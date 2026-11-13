"""E12: independent test of outer-bound soundness (F ⊆ support of the label), the assumption A1 that both the
tau-pencil labels and the FastLabel statistical certificate rest on and that the certificate itself cannot test.

For each selected task: draw M points from the PRIOR (uniform in the cube, not from B), decide exact membership in F
with the certified GPU point verifier (rind3d_1k.fastlabel.gpuverify.WindowVerifier: legality, closed box, every FREE
voxel), and check that every feasible point's 128^3 cell carries label mass. Also compares vol(F) estimated from the
prior sample with the label's volume (exact paths: sum lo; statistical: vol(B) >= vol(F)).
Labels: FastLabel v3.4 computed in memory (never written), or the stored release label (scenes 000000/000001).
Usage: python e12_soundness_prior_mc.py <tasks_file> <M> > out/e12_soundness.jsonl
tasks_file lines: scene_dir window source source_of_label(fastlabel|release)
"""
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/yushu/Projects/RIND/G2-02-3d/RIND3D_Dataset1K"); sys.path.insert(0, "/home/yushu/Projects/RIND/G2-02-3d/RIND3D_Posterior/src")
from rind3d_1k.fastlabel import label_task
from rind3d_1k.fastlabel import pipeline as PL, gpuverify as GV
from rind3d_1k.fastlabel.window import fast_cls3
G = 128


def cell_of(P):
    i = np.minimum((P * G).astype(np.int64), G - 1)
    return (i[:, 0] * G + i[:, 1]) * G + i[:, 2]


def main(tf, M):
    rng = np.random.default_rng(1212)
    for line in open(tf):
        sc, w, s, src = line.split(); w, s = int(w), int(s)
        t0 = time.time()
        try:
            if src == "fastlabel":
                lab = label_task(sc, w, s); cell = np.asarray(lab["cell"], np.int64); lo = np.asarray(lab["lo"]); hi = np.asarray(lab["hi"])
                status = lab["status"]
            else:
                with np.load(Path(sc) / "labels" / "tasks" / f"w{w:03d}_s{s}.npz") as z:
                    cell = z["cell"].astype(np.int64); lo = z["lo"]; hi = z["hi"]; status = "release"
            cells, inv = np.unique(cell, return_inverse=True)
            hic = np.bincount(inv, weights=hi); loc = np.bincount(inv, weights=lo)
            supp = cells[hic > 0]
            scn = PL.scene_cache(sc); S = scn["S"]
            V = GV.WindowVerifier(PL.process_factory(), scn["ctx"], S, w, s, fast_cls3(S, w, s))
            pts = rng.random((M, 3)); res = V.test_batch(pts); V.close()
            ok = np.array([r[0] for r in res]); feas = pts[ok]
            outside = int((~np.isin(cell_of(feas), supp)).sum()) if len(feas) else 0
            p = ok.mean()
            out = dict(scene=Path(sc).name, window=w, source=s, label=src, status=status, M=M, n_feasible=int(ok.sum()),
                       feasible_outside_support=outside, vol_F_prior_mc=float(p), vol_F_se=float(np.sqrt(p * (1 - p) / M)),
                       label_vol_lo=float(loc.sum()), label_vol_hi=float(hic.sum()), n_cells=int(len(supp)), wall=time.time() - t0)
        except Exception as e:
            out = dict(scene=Path(sc).name, window=w, source=s, label=src, error=repr(e)[:300], wall=time.time() - t0)
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
