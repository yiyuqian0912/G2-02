"""Independent empirical validation of S6 labels (not part of the certificate).
usage: validate_s6.py <scene_dir> <w> <s> [n_b=4096] [n_prior=40000]
1. Label with the S6 prototype (base_seed 777, as in run_s6.py).
2. Fresh RNG: n_b uniform samples of B (rejecting those inside closed solids, as in the test) -> exact membership in F
   -> p_hat = vol(F)/vol(B') with a 99% Clopper-Pearson interval (the certificate claims p >= 0.95).
3. n_prior uniform points of the cube -> exact membership in F (exact GPU verifier) -> every feasible point must lie
   in a cell with positive label mass (soundness of F ⊆ B'); also the MC volume of F vs. sum(label) * p_hat."""
import json, sys, time
sys.path.insert(0, "/home/yushu/Projects/RIND/G2-02-3d/RIND3D_StudentResearch/s6")
import numpy as np
from scipy.stats import beta
from fastlabel_s6 import label_task_s6
from rind3d_1k.fastlabel.pipeline import scene_cache, process_factory
from rind3d_1k.fastlabel import gpuverify as GV
from rind3d_1k.fastlabel.window import fast_cls3
from rind3d_1k.fastlabel.sampler import BSampler

sd, w, s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
n_b = int(sys.argv[4]) if len(sys.argv) > 4 else 4096; n_pr = int(sys.argv[5]) if len(sys.argv) > 5 else 40000
t0 = time.perf_counter()
lab = label_task_s6(sd, w, s, base_seed=777)
m = lab["meta"]
assert m["path"] == "stat_certified_s6", m["path"]
scn = scene_cache(sd); S = scn["S"]; ctx = scn["ctx"]
V = GV.WindowVerifier(process_factory(), ctx, S, w, s, fast_cls3(S, w, s))
rng = np.random.default_rng([4242, w, s])
cells = np.asarray(lab["cell"], np.int64); mass = np.asarray(lab["hi"], float)
order = np.argsort(cells); cells, mass = cells[order], mass[order]


def cell_of(P):
    i = np.minimum((P * 128).astype(np.int64), 127)
    return (i[:, 0] * 128 + i[:, 1]) * 128 + i[:, 2]


# (2) uniform on B' via the label's cells is NOT what the certificate tests; we test vol(F)/vol(B') with points uniform on
# the label measure q_B': pick a cell by mass, then a uniform point in the cell, keep it only if outside solids and in a
# B' piece. The pieces are not stored in the label, so instead test the label measure directly: p_label = q_B'(F-cells)
# would be cell-level; we use the exact statement available from the label: points uniform on supp-weighted cells.
# Simplest exact check of the claim: sample cells by mass, a uniform point in the cell, accept if not inside a solid;
# for fully-inside-B' cells this is uniform on B' ∩ cell; partial cells (B' boundary) are rare and make p_hat conservative.
pc = rng.choice(len(cells), size=n_b * 2, p=mass / mass.sum())
ix, iy, iz = cells[pc] // 16384, (cells[pc] // 128) % 128, cells[pc] % 128
P = (np.stack([ix, iy, iz], 1) + rng.random((len(pc), 3))) / 128
inside = ctx.contains(np.ascontiguousarray(P, "<f8"))
P = P[~np.asarray(inside, bool)][:n_b]
okF = np.array([g[0] for g in V.test_batch(P)])
k = int(okF.sum()); n = len(okF)
ci = [float(beta.ppf(0.005, k, n - k + 1)) if k else 0.0, float(beta.ppf(0.995, k + 1, n - k)) if k < n else 1.0]
# (3) prior points
Q = rng.random((n_pr, 3))
fe = np.array([g[0] for g in V.test_batch(Q)])
fc = cell_of(Q[fe]); pos = np.searchsorted(cells, fc)
outside = int(np.sum(~((pos < len(cells)) & (cells[np.minimum(pos, len(cells) - 1)] == fc))))
V.close()
print(json.dumps(dict(scene=sd[-6:], window=w, source=s, label_volume=float(mass.sum()), cells=int(len(cells)),
                      cellmeasure_p_hat=k / n, cellmeasure_ci99=ci, n_b=n,
                      prior_points=n_pr, feasible=int(fe.sum()), feasible_outside_support=outside,
                      mc_volF=float(fe.mean()), mc_volF_se=float(np.sqrt(fe.mean() * (1 - fe.mean()) / n_pr)),
                      certificate_p_lower=m["p_lower"], wall=time.perf_counter() - t0)))
