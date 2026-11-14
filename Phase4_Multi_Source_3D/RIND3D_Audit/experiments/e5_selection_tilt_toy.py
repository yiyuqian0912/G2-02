"""E5: exact toy computation of what the 1K selection law does to (i) the per-channel label given G and (ii) the
Student's population optimum given only X (single-channel, flattened).

Populations of flattened informative examples (G, W, s_j, X = (W, r_j)):
  N  natural single source:     pi(G) u_D(s) p_nat(W) 1[s not in W]                       (reference target)
  B  1K law, weighted by omega: pi(G) kappa(k) prod u_D(s_i) p_nat(W) A(W,S) c(m,L,k)     (contract training law)
  C  1K law, weighted by omega / c(m):  removes the balancing factor exactly
  B0 1K law, UNweighted (k = 2 only, enumerated over source pairs): label-level bias of the quota normalisation
Label for every population: Uniform(F_W^j(G)) (the tau-pencil target).

Run: nice -n 19 python e5_selection_tilt_toy.py > out/e5_selection_tilt_toy.json
"""
import json, sys, time
from math import comb
import numpy as np
from toyworld import Toy, tv, kl

KAPPA = np.array([154, 135, 158, 553]) / 1000.0                    # realised 1K k histogram
C1K = {1: [1.0], 2: [0.01, 1.0], 3: [0.01, 0.2241, 1.0], 4: [0.01, 0.0965, 0.4483, 1.0]}   # 1K L64 median c(m | k)


def H_balanced(a, b, kappa=KAPPA, ctab=C1K):
    """Selection factor of an informative channel after integrating the other k-1 i.i.d. sources:
    sum_k kappa_k * k * sum_{m'} C(k-1, m') a^m' b^(k-1-m') c(1 + m', k)."""
    h = 0.0
    for k in range(1, 5):
        s = 0.0
        for mp in range(k):
            s += comb(k - 1, mp) * a ** mp * b ** (k - 1 - mp) * ctab[k][mp]
        h += kappa[k - 1] * k * s
    return h


def H_debalanced(a, b, kappa=KAPPA):
    return sum(kappa[k - 1] * k * (a + b) ** (k - 1) for k in range(1, 5))


def main():
    t0 = time.time()
    T = Toy()
    nW, S, R = len(T.windows), T.S, T.w * T.w
    full = (1 << R) - 1
    pops = ("N", "B", "C")
    J = {p: np.zeros((nW, 1 << R, S)) for p in pops}
    Hs = {"B": [], "C": []}
    # label-level, k = 2, unweighted vs weighted (enumerate source pairs)
    lab = dict(B0_tv_sum=0.0, B0_mass=0.0, Bw_tv_sum=0.0, Bw_mass=0.0, B0_tv_max=0.0, Bw_tv_max=0.0, pacc_lt1_mass=0.0)
    c2 = np.array([0.0, 0.01, 1.0])               # c(m) for k = 2, index m (m = 0 -> rejected by A)
    for b, pr, solid in T.geometries():
        legal = ~solid; nD = legal.sum(); u = legal / nD
        lit = T.responses(b); code = T.codes(lit)                 # (S, nW)
        inf = (code != 0) & (code != full) & legal[:, None]
        out = legal[:, None] & ~T.in_win.T                         # (S, nW) legal and outside the window
        for wi in range(nW):
            sel = inf[:, wi] & out[:, wi]
            a = sel.sum() / nD; bb = (out[:, wi] & ~inf[:, wi]).sum() / nD
            hB, hC = H_balanced(a, bb), H_debalanced(a, bb)
            Hs["B"].append((pr * sel.sum() / nD, hB)); Hs["C"].append((pr * sel.sum() / nD, hC))
            idx = np.flatnonzero(sel); cc = code[idx, wi]
            for p, h in (("N", 1.0), ("B", hB), ("C", hC)):
                np.add.at(J[p][wi], (cc, idx), pr * h / nD)
        # ---- label level, k = 2: g(s1, w) = sum_s2 u(s2) A_w c_w / Lambda   (B0)   or  sum_s2 u(s2) A_w c_w  (Bw)
        Ls = np.flatnonzero(legal)
        I = inf[Ls]; O = out[Ls]                                  # (n, nW)
        m = I[:, None, :].astype(int) + I[None, :, :].astype(int)  # (n, n, nW)
        A = O[:, None, :] & O[None, :, :] & (m >= 1)
        acw = A * c2[m]                                           # A_w c_w
        Lam = acw.mean(-1)                                        # Lambda(s1, s2), proposals uniform over windows
        np.fill_diagonal(Lam, 0.0)                                # anti-duplicate (distinct cells)
        ok = Lam > 0                                              # scene acceptable (P_acc = 1[Lambda > 0], infinite cap)
        uu = 1.0 / len(Ls)
        lab["pacc_lt1_mass"] += pr * float((~ok).sum()) / (len(Ls) ** 2 - len(Ls)) if len(Ls) > 1 else 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            invL = np.where(ok, 1.0 / Lam, 0.0)
        for wi in range(nW):
            g0 = (uu * acw[:, :, wi] * invL).sum(1)               # unweighted: expected accepted-count density factor
            gw = (uu * acw[:, :, wi] * ok).sum(1)                 # omega-weighted (with P_acc conditioning)
            codes_w = code[Ls, wi]; infw = I[:, wi] & O[:, wi]
            for cval in np.unique(codes_w[infw]):
                F = infw & (codes_w == cval)
                qF = F / F.sum()
                for g, key in ((g0, "B0"), (gw, "Bw")):
                    m_ = (g * F)
                    if m_.sum() <= 0:
                        continue
                    t = float(tv(m_ / m_.sum(), qF))
                    mass = pr * uu * float(m_.sum())               # example mass of this (G, w, X) in that population
                    lab[key + "_tv_sum"] += mass * t; lab[key + "_mass"] += mass
                    lab[key + "_tv_max"] = max(lab[key + "_tv_max"], t)
    res = dict(config=dict(n=T.n, w=T.w, windows=T.windows, n_obst=T.no, p_obst=T.p_obst, obst=T.obst,
                           kappa=KAPPA.tolist(), c_table=C1K, geometries=2 ** T.no))
    # ---- Student level
    P = {p: J[p].reshape(-1, S) for p in pops}
    mX = {p: P[p].sum(1) for p in pops}
    keep = mX["N"] > 0
    post = {p: P[p][keep] / mX[p][keep][:, None] for p in pops}
    wN = mX["N"][keep] / mX["N"][keep].sum(); wB = mX["B"][keep] / mX["B"][keep].sum(); wC = mX["C"][keep] / mX["C"][keep].sum()
    tvBN = tv(post["B"], post["N"]); tvCN = tv(post["C"], post["N"])
    klNB = kl(post["N"], post["B"]); klNC = kl(post["N"], post["C"])
    res["student_level"] = dict(
        n_X=int(keep.sum()),
        TV_B_vs_N=dict(mean_under_B=float(wB @ tvBN), mean_under_N=float(wN @ tvBN), p90_under_B=float(np.quantile(tvBN, .9)), max=float(tvBN.max())),
        TV_C_vs_N=dict(mean_under_C=float(wC @ tvCN), mean_under_N=float(wN @ tvCN), max=float(tvCN.max())),
        excess_logloss_nats_on_N=dict(B_optimum=float(wN @ klNB), C_optimum=float(wN @ klNC)),
        TV_of_X_marginals=dict(B_vs_N=float(0.5 * np.abs(wB - wN).sum()), C_vs_N=float(0.5 * np.abs(wC - wN).sum())),
        selection_factor_H_B=dict(min=float(min(h for _, h in Hs["B"])), max=float(max(h for _, h in Hs["B"]))),
    )
    res["label_level_k2"] = dict(
        unweighted_mean_TV_to_UniformF=lab["B0_tv_sum"] / lab["B0_mass"], unweighted_max_TV=lab["B0_tv_max"],
        omega_weighted_mean_TV_to_UniformF=lab["Bw_tv_sum"] / lab["Bw_mass"], omega_weighted_max_TV=lab["Bw_tv_max"],
        prior_mass_of_unacceptable_source_pairs=lab["pacc_lt1_mass"],
        note="with an infinite proposal cap, scene acceptance (Lambda > 0) is implied by the example's own acceptance, so P_acc does not tilt the label; the omega-weighted label equals Uniform(F) to rounding. With finite caps P_acc(G,S) is smooth in Lambda and does tilt it (bounded by the rejection rate, SELECTION_LAW_1K section 3)")
    res["seconds"] = time.time() - t0
    json.dump(res, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
