"""E6: hard geometric posterior vs soft response-matching posteriors in an exact 2D toy world (fixed known G).

Truth: s* uniform over legal sources outside the window; X = R_G(s*)|W exactly (noise-free) or with i.i.d. receiver
flips (rate eps_true). Candidate posteriors over the same legal source set:
  hard        q ∝ 1[R(s) = X]                               (Uniform(F); exact Bayes posterior when noise-free)
  flip(eps)   q ∝ (eps/(1-eps))^{d_H(R(s), X)}              (exact Bayes posterior under i.i.d. flips with rate eps)
  nham(beta)  q ∝ exp(-beta * d_H / n)                      (normalised-Hamming soft matching; implied flip rate
                                                              eps_eff = 1 / (1 + exp(beta / n)) depends on n)
Metrics averaged over s*: log score E[-log q(s*)] (nats), mass on the exact feasible set q(F_X), and the share of
mass on candidate sources whose response is ALL-LIT on W, split by the dark fraction of X (Case A: 0 < dark < 10 %; all-lit X reported separately;
Case B: 30-70 % dark).
Run: nice -n 19 python e6_hard_vs_soft_toy.py > out/e6_hard_vs_soft_toy.json
"""
import json, sys
import numpy as np
from toyworld import Toy

OBST = [(10, 8), (10, 9), (10, 10), (10, 11), (16, 16), (17, 16), (5, 18), (19, 5), (14, 3), (13, 13), (20, 20), (6, 3)]


def run_world(w, origin, n=24, eps_true=(0.0, 0.01, 0.03), seed=3):
    T = Toy(n=n, w=w, windows=(origin,), obst_cells=OBST, p_obst=1.0)
    b = np.ones(T.no, bool); solid = np.zeros(T.S, bool); solid[T.obst_idx] = True
    legal = ~solid & ~T.in_win[0]
    lit = T.responses(b)[:, 0, :][legal].astype(np.int8)           # (Ns, R)
    Ns, R = lit.shape
    dark_frac = 1 - lit.mean(1)
    alllit = lit.all(1)
    rng = np.random.default_rng(seed)
    fams = {"hard": None, "flip_0.01": 0.01, "flip_0.03": 0.03, "flip_0.10": 0.10,
            "nham_beta5": ("b", 5.0), "nham_beta20": ("b", 20.0), "nham_beta100": ("b", 100.0)}
    res = {}
    for et in eps_true:
        acc = {f: dict(logscore=[], zero=[], massF=[], mass_alllit=[], case=[]) for f in fams}
        for t in range(Ns):
            X = lit[t].copy()
            if et > 0:
                X ^= (rng.random(R) < et).astype(np.int8)
            d = np.abs(lit - X[None]).sum(1)                       # Hamming distance of every candidate
            F = d == 0
            df = 1 - X.mean()
            case = ("alllit_X" if df == 0 else "A_lt10pct_dark" if df < 0.10 else
                    "B_30to70pct_dark" if 0.3 <= df <= 0.7 else "other")
            for f, par in fams.items():
                if par is None:
                    lw = np.where(F, 0.0, -np.inf)
                elif isinstance(par, tuple):
                    lw = -par[1] * d / R
                else:
                    lw = d * np.log(par / (1 - par))
                if np.all(np.isinf(lw)):
                    q = None
                else:
                    lw = lw - lw.max(); q = np.exp(lw); q /= q.sum()
                a = acc[f]
                if q is None or q[t] == 0:
                    a["zero"].append(1); a["logscore"].append(np.inf)
                else:
                    a["zero"].append(0); a["logscore"].append(float(-np.log(q[t])))
                a["massF"].append(float(q[F].sum()) if q is not None else 0.0)
                a["mass_alllit"].append(float(q[alllit].sum()) if q is not None else 0.0)
                a["case"].append(case)
        out = {}
        for f, a in acc.items():
            ls = np.array(a["logscore"]); z = np.array(a["zero"]); cs = np.array(a["case"])
            row = dict(share_true_source_zero_mass=float(z.mean()),
                       mean_logscore_finite=float(ls[np.isfinite(ls)].mean()) if np.isfinite(ls).any() else None,
                       mean_mass_on_exact_F=float(np.mean(a["massF"])))
            for c in ("alllit_X", "A_lt10pct_dark", "B_30to70pct_dark"):
                m = cs == c
                if m.any():
                    row[c] = dict(n=int(m.sum()), mean_logscore=float(ls[m].mean()) if np.isfinite(ls[m]).all() else "inf",
                                  mass_on_alllit_candidates=float(np.mean(np.array(a["mass_alllit"])[m])),
                                  mass_on_exact_F=float(np.mean(np.array(a["massF"])[m])))
            out[f] = row
        res[f"eps_true_{et}"] = out
    res["world"] = dict(n=n, w=w, receivers=R, legal_sources=int(Ns), share_alllit_sources=float(alllit.mean()),
                        share_case_A=float(np.mean(~alllit & (dark_frac < .1))),
                        implied_flip_rate_of_nham={str(bb): float(1 / (1 + np.exp(bb / R))) for bb in (5, 20, 100)})
    return res


if __name__ == "__main__":
    out = {"w4": run_world(4, (2, 8)), "w8": run_world(8, (1, 7))}
    json.dump(out, sys.stdout, indent=1)
