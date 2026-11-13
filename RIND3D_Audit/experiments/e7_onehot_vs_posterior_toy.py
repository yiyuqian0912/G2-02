"""E7: one-hot vs privileged-posterior (Rao-Blackwellised) supervision with finite data, exact 2D toy world of E5.

Population: natural single-source law (G ~ pi, s ~ U(D(G)), four fixed windows, informative windows outside which s
lies are examples). The Bayes target P(s | X) is computed exactly. A tabular "Student" (one categorical per X) is fit
by minimising the empirical cross-entropy with Dirichlet smoothing alpha toward p0 (uniform over cells outside the
window); its minimiser is p_hat(.|X) = (sum_i target_i + alpha p0) / (n_X + alpha). Arms (same data):
  onehot     target = delta_{s_i}
  posterior  target = q_{G_i} = Uniform(F)                          (exact label)
  post_eps   target = (1 - eps) q_F + eps Uniform(near-miss cells)  (certified-label worst case: TV = eps = 0.05,
                                                                     all extra mass on infeasible Hamming-1 cells)
  hybrid     target = (delta + q) / 2
Metric: excess risk E_{X~P}[KL(P(.|X) || p_hat(.|X))] in nats (the reducible part of the log-loss), mean and s.e. over
replicates; alpha chosen per arm and n by an oracle (best of a grid), so the comparison is not a smoothing artefact.
Second study: fixed number of sources, split into (scenes x sources per scene) = 1200x1, 300x4, 100x12.
Run: nice -n 19 python e7_onehot_vs_posterior_toy.py > out/e7_onehot_vs_posterior_toy.json
"""
import json, sys, time
import numpy as np
from toyworld import Toy, kl

EPS = 0.05
ALPHAS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


def build(T):
    nW, S, R = len(T.windows), T.S, T.w * T.w
    full = (1 << R) - 1
    geo = []; probs = []
    xid = {}
    for b, pr, solid in T.geometries():
        legal = ~solid
        lit = T.responses(b); code = T.codes(lit)
        inf = (code != 0) & (code != full) & legal[:, None] & ~T.in_win.T
        geo.append((np.flatnonzero(legal), code, inf, lit)); probs.append(pr)
        for wi in range(nW):
            for c in np.unique(code[inf[:, wi], wi]):
                xid.setdefault((wi, int(c)), len(xid))
    return geo, np.array(probs), xid


def main():
    t0 = time.time()
    T = Toy(); S = T.S; nW = len(T.windows)
    geo, probs, xid = build(T)
    nX = len(xid)
    # exact Bayes target P(s | X) (natural single source)
    J = np.zeros((nX, S))
    for (Ls, code, inf, lit), pr in zip(geo, probs):
        for wi in range(nW):
            idx = np.flatnonzero(inf[:, wi])
            for s in idx:
                J[xid[(wi, int(code[s, wi]))], s] += pr / len(Ls)
    PX = J.sum(1); P = J / PX[:, None]; PX /= PX.sum()
    p0 = np.zeros((nX, S))
    for (wi, c), i in xid.items():
        p0[i] = ~T.in_win[wi]; p0[i] /= p0[i].sum()
    # per-geometry label tables, built lazily
    cache = {}

    def labels(g, wi, c):
        key = (g, wi, c)
        if key not in cache:
            Ls, code, inf, lit = geo[g]
            F = inf[:, wi] & (code[:, wi] == c)
            q = F / F.sum()
            # near-miss cells: legal, outside window, Hamming distance 1 from the observed pattern
            bits = lit[:, wi, :].astype(int); obs = np.array([(c >> r) & 1 for r in range(bits.shape[1])])
            d = np.abs(bits - obs).sum(1)
            Nm = (d == 1) & np.isin(np.arange(S), Ls) & ~T.in_win[wi]
            if not Nm.any():
                Nm = np.isin(np.arange(S), Ls) & ~T.in_win[wi] & ~F
            qe = (1 - EPS) * q + EPS * Nm / Nm.sum()
            cache[key] = (q, qe)
        return cache[key]

    rng = np.random.default_rng(2026)
    cum = np.cumsum(probs)

    def dataset(n_scenes, per_scene):
        sums = {a: np.zeros((nX, S)) for a in ("onehot", "posterior", "post_eps", "hybrid")}
        cnt = np.zeros(nX)
        for _ in range(n_scenes):
            g = int(np.searchsorted(cum, rng.random() * cum[-1]))
            Ls, code, inf, lit = geo[g]
            for s in rng.choice(Ls, per_scene, replace=True):
                for wi in range(nW):
                    if not inf[s, wi]:
                        continue
                    c = int(code[s, wi]); i = xid[(wi, c)]
                    q, qe = labels(g, wi, c)
                    cnt[i] += 1
                    sums["onehot"][i, s] += 1; sums["posterior"][i] += q; sums["post_eps"][i] += qe
                    sums["hybrid"][i] += 0.5 * q; sums["hybrid"][i, s] += 0.5
        return sums, cnt

    def risk(sm, cnt, alpha):
        ph = (sm + alpha * p0) / (cnt[:, None] + alpha)
        return float(PX @ kl(P, ph))

    out = dict(config=dict(eps=EPS, alphas=ALPHAS, n_X=nX, reps=None), curve={}, split={})
    reps = 24; out["config"]["reps"] = reps
    for n in (30, 100, 300, 1000, 3000):
        res = {a: {al: [] for al in ALPHAS} for a in ("onehot", "posterior", "post_eps", "hybrid")}
        for _ in range(reps):
            sm, cnt = dataset(n, 1)
            for a in res:
                for al in ALPHAS:
                    res[a][al].append(risk(sm[a], cnt, al))
        row = {}
        for a in res:
            best = min(ALPHAS, key=lambda al: np.mean(res[a][al]))
            v = np.array(res[a][best])
            row[a] = dict(alpha=best, excess_kl=float(v.mean()), se=float(v.std(ddof=1) / np.sqrt(reps)))
        out["curve"][str(n)] = row
    # bias floor of the eps-label arm: infinite data limit
    big = {a: np.zeros((nX, S)) for a in ("posterior", "post_eps")}
    for g, ((Ls, code, inf, lit), pr) in enumerate(zip(geo, probs)):
        for wi in range(nW):
            for s in np.flatnonzero(inf[:, wi]):
                q, qe = labels(g, wi, int(code[s, wi])); i = xid[(wi, int(code[s, wi]))]
                big["posterior"][i] += pr / len(Ls) * q; big["post_eps"][i] += pr / len(Ls) * qe
    out["infinite_data_excess_kl"] = {a: float(PX @ kl(P, big[a] / big[a].sum(1, keepdims=True))) for a in big}
    # scenes vs sources per scene at a fixed source budget
    for ns, ps in ((1200, 1), (300, 4), (100, 12)):
        res = {a: [] for a in ("onehot", "posterior")}
        for _ in range(reps):
            sm, cnt = dataset(ns, ps)
            for a in res:
                res[a].append(min(risk(sm[a], cnt, al) for al in ALPHAS))
        out["split"][f"{ns}x{ps}"] = {a: dict(excess_kl=float(np.mean(v)), se=float(np.std(v, ddof=1) / np.sqrt(reps))) for a, v in res.items()}
    out["seconds"] = time.time() - t0
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
