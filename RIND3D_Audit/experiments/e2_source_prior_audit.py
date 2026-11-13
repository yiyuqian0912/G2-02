"""E2 + E3: source spatial prior audit of release_1k_v1 relative to each scene's legal FREE volume, and the source
distribution induced by accepted windows, informative channels and training weights.

Per scene: M uniform Monte-Carlo points in (0,1)^3 classified by the float64 halfspace rows (geometry_rows.npz;
inside body c iff all rows a.x + d <= 0). For every source s and statistic T (x, y, z, r = |s - 1/2|_inf, local solid
fraction) the PIT value u = P_{y ~ U(D(G))}(T(y) < T(s)) is estimated from the FREE MC points. Under the declared prior
(s i.i.d. Uniform(D(G))) u ~ U(0,1) up to MC error O(M^-1/2).

E3: each source j gets the multiplicities n_win(j) = #accepted windows with channel j informative and
w(j) = sum over those windows of omega (rate). The PIT histograms of sources weighted by these show what the training
population does to the marginal source distribution (expected to tilt; not an error by itself).
Run: nice -n 19 python e2_source_prior_audit.py <release_root> > out/e2_source_prior_audit.json
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import kstest, chisquare

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/yushu/Projects/RIND/RIND3D_1K_data/release_1k_v1")
M = 6000; RB = 0.06; MB = 400
rng = np.random.default_rng(20261113)


def solid_mask(P, K, X):
    s = np.zeros(len(X), bool)
    for c in range(len(K)):
        R = P[c, :K[c]]
        s |= np.all(X @ R[:, :3].T + R[:, 3] <= 0, axis=1)
    return s


def local_solid(P, K, s):
    Y = s + rng.uniform(-RB, RB, (MB, 3))
    Y = Y[np.all((Y > 0) & (Y < 1), 1)]
    return float(solid_mask(P, K, Y).mean()) if len(Y) else 0.0


recs = []
for d in sorted((ROOT / "scenes").iterdir()):
    meta = json.loads((d / "scene.json").read_text())
    g = np.load(d / "privileged" / "geometry_rows.npz"); P, K = g["P"], g["K"]
    src = np.array([[float.fromhex(v) for v in s] for s in json.loads((d / "privileged" / "sources.json").read_text())["sources_hex"]])
    W = np.load(d / "public" / "windows.npy")
    X = rng.random((M, 3)); free = ~solid_mask(P, K, X); Xf = X[free]
    vfree = float(free.mean())
    # local solid fraction of FREE reference points (subsample for cost); ties broken by a randomised PIT
    ref_loc = np.array([local_solid(P, K, y) for y in Xf[:150]])
    assert not solid_mask(P, K, src).any(), d.name
    lam = {int(k): float(v) for k, v in meta["Lambda_hat"].items()}
    for j, s in enumerate(src):
        rel = np.abs(Xf - 0.5).max(1); rs = float(np.abs(s - 0.5).max())
        loc = local_solid(P, K, s)
        inf = np.array([(int(w["mask"]) >> j) & 1 for w in W], bool)
        nwin = int(inf.sum()); wsum = float(sum(lam[int(w["Lp"])] for w, f in zip(W, inf) if f))
        nwin_side = {Lp: int(sum(f for w, f in zip(W, inf) if int(w["Lp"]) == Lp)) for Lp in (64, 128, 256)}
        recs.append(dict(scene=d.name, family=meta["family"], k=int(meta["k"]), vfree=vfree,
                         ux=float(np.mean(Xf[:, 0] < s[0])), uy=float(np.mean(Xf[:, 1] < s[1])), uz=float(np.mean(Xf[:, 2] < s[2])),
                         ur=float(np.mean(rel < rs)), r=rs, uloc=float((np.sum(ref_loc < loc) + rng.random() * (np.sum(ref_loc == loc) + 1)) / (len(ref_loc) + 1)),
                         rbin_expect=np.histogram(rel, bins=[0, .1, .2, .3, .4, .5])[0] / len(rel),
                         loc=loc, nwin=nwin, wsum=wsum, nwin_side=nwin_side))

out = dict(config=dict(M=M, local_half_width=RB, local_points=MB, n_sources=len(recs)))


def pit_tests(rs, key):
    u = np.array([r[key] for r in rs])
    return dict(n=len(u), ks_p=float(kstest(u, "uniform").pvalue), mean=float(u.mean()),
                decile_counts=np.histogram(u, bins=10, range=(0, 1))[0].tolist())


out["pit_all"] = {key: pit_tests(recs, key) for key in ("ux", "uy", "uz", "ur", "uloc")}
fams = sorted({r["family"] for r in recs})
out["pit_by_family_ks_p"] = {f: {key: pit_tests([r for r in recs if r["family"] == f], key)["ks_p"] for key in ("ux", "uy", "uz", "ur")} for f in fams}
out["pit_by_k_ks_p"] = {str(k): {key: pit_tests([r for r in recs if r["k"] == k], key)["ks_p"] for key in ("ux", "uy", "uz", "ur")} for k in range(1, 5)}
# aggregate radial histogram: observed vs expected under per-scene Uniform(D(G)) vs Uniform(cube)
edges = np.array([0, .1, .2, .3, .4, .5])
obs = np.histogram([r["r"] for r in recs], bins=edges)[0]
cube = np.diff((2 * edges) ** 3) * len(recs)        # P(|U - 1/2|_inf < t) = (2t)^3
exp_D = np.sum([r["rbin_expect"] for r in recs], 0)
out["radial_histogram"] = dict(edges=edges.tolist(), observed=obs.tolist(), expected_uniform_cube=cube.round(1).tolist(),
                               expected_per_scene_uniform_D=exp_D.round(1).tolist(),
                               chi2_p_vs_per_scene_uniform_D=float(chisquare(obs, exp_D * obs.sum() / exp_D.sum()).pvalue),
                               chi2_p_vs_uniform_cube=float(chisquare(obs, cube).pvalue),
                               note="per-scene uniformity is tested by the PIT values (ur), not by this aggregate histogram")


# E3: weighted PIT histograms (training population tilt)
def wpit(key, wkey):
    u = np.array([r[key] for r in recs]); w = np.array([r[wkey] for r in recs], float)
    h = np.histogram(u, bins=10, range=(0, 1), weights=w)[0]
    return dict(weighted_decile_share=(h / h.sum()).round(4).tolist(), weighted_mean_u=float((u * w).sum() / w.sum()),
                kish_ess=float(w.sum() ** 2 / (w ** 2).sum()))


out["training_tilt"] = {wk: {key: wpit(key, wk) for key in ("ur", "uloc")} for wk in ("nwin", "wsum")}
loc = np.array([r["loc"] for r in recs]); nw = np.array([r["nwin"] for r in recs]); ws = np.array([r["wsum"] for r in recs])
from scipy.stats import spearmanr
out["multiplicity_vs_local_solid"] = dict(spearman_nwin=float(spearmanr(loc, nw).correlation), spearman_wsum=float(spearmanr(loc, ws).correlation),
                                         share_sources_with_zero_informative_windows=float(np.mean(nw == 0)),
                                         nwin_quantiles={str(q): float(np.quantile(nw, q)) for q in (.1, .25, .5, .75, .9)},
                                         nwin_mean_by_k={str(k): float(np.mean([r["nwin"] for r in recs if r["k"] == k])) for k in range(1, 5)})
json.dump(out, sys.stdout, indent=1)
