"""Exact (rational) check of the FastLabel v3.4 sequential test at the v3 production budget (FASTLABEL_V34_CERTIFICATION.md §5).

For every round r = 1..R the production design (pipeline.design2, scipy floats at eps = 0.05) gives thresholds
(a1, a2) at delta_r = delta * 2^-r. Here we recompute, in exact rational arithmetic at the TRUE boundary eps = 1/20:
  alpha_r = P_{theta = 1/20}( K_{n1} <= a1  or  K_{n2} <= a2 ),   K_n = failures among n i.i.d. points,
and check alpha_r <= delta_r (delta_r as the float the code uses, read as an exact rational) and sum_r alpha_r <= delta.
The certificate event is increasing in the pass indicators, so alpha_r bounds P(certify at r | B_r) whenever
theta_r = 1 - |F|/|B_r| >= 1/20. We also report the power P_theta(certify in one round) and E[points tested] for several
theta (the production chunk schedule 128 / 512 / 1024 with early stop on K > a2), and the designs for other budgets.

usage: python design_check.py OUT.json"""
import json
import sys
from fractions import Fraction as Q
from math import comb

sys.dont_write_bytecode = True
from rind3d_posterior.fastlabel.pipeline import DEFAULTS, design2   # noqa: E402

E = Q(1, 20)


def pmf(n, th):
    return [comb(n, k) * th ** k * (1 - th) ** (n - k) for k in range(n + 1)]


def alpha(n1, n2, a1, a2, th):
    """Exact P(K1 <= a1 or K2 <= a2) with K2 = K1 + K', K' ~ Bin(n2 - n1, th) independent."""
    p1 = pmf(n1, th); p2 = pmf(n2 - n1, th)
    c2 = [Q(0)] * (n2 - n1 + 1); s = Q(0)
    for k, v in enumerate(p2):
        s += v; c2[k] = s
    tot = sum(p1[:a1 + 1], Q(0)) if a1 >= 0 else Q(0)
    for k1 in range(max(a1 + 1, 0), min(a2, n1) + 1):
        tot += p1[k1] * c2[a2 - k1]
    return tot


def round_profile(n1, n2, a1, a2, th, chunks=(128, 384, 512)):
    """(P(certify), E[points tested]) of one production round at failure probability th (float, Monte-Carlo free:
    exact recursion over the chunk checkpoints)."""
    import numpy as np
    from scipy.stats import binom
    b = list(np.cumsum(chunks)); b = [int(x) for x in b if x < n2] + [n2]
    # distribution of K at each checkpoint, conditional on continuing
    dist = np.zeros(n2 + 1); dist[0] = 1.0; prev = 0; cert = 0.0; tested = 0.0
    for hi in b:
        m = hi - prev
        inc = binom.pmf(np.arange(m + 1), m, th)
        dist = np.convolve(dist, inc)[:n2 + 1]
        alive = dist.sum(); tested += alive * m
        dist[a2 + 1:] = 0.0                                 # K > a2: stop, no certificate
        if hi == n1 and n1 < n2:
            cert += dist[:a1 + 1].sum(); dist[:a1 + 1] = 0.0  # stage-1 certificate
        if hi == n2:
            cert += dist.sum()
        prev = hi
    return float(cert), float(tested)


def main(out):
    n2 = int(DEFAULTS["n_cert"]); n1 = int(DEFAULTS["n_stage1"]); R = int(DEFAULTS["R"]); eps = float(DEFAULTS["eps"])
    budgets = {"v3_100k": 0.01 / 4800000, "1k": 0.01 / 159802, "per_label_1e-2": 0.01}
    res = dict(n1=n1, n2=n2, R=R, eps=eps, budgets={})
    for name, delta in budgets.items():
        rows = []; tot = Q(0)
        for r in range(1, R + 1):
            d_r = delta * 0.5 ** r
            a1, a2, al_float = design2(n1, n2, eps, d_r)
            al = alpha(n1, n2, a1, a2, E)
            # maximality: one more failure allowed in stage 2 would exceed delta_r (exact)
            al_next = alpha(n1, n2, a1, a2 + 1, E)
            tot += al
            row = dict(r=r, delta_r=d_r, a1=a1, a2=a2, alpha_exact=float(al), alpha_scipy=al_float,
                       ok=bool(al <= Q(d_r)), a2_maximal=bool(al_next > Q(d_r)), margin=float(1 - al / Q(d_r)))
            if name == "v3_100k":
                # robustness to the numerical terms: the same thresholds stay valid at the boundary 1/20 - eta
                for eta in (Q(1, 10 ** 8), Q(1, 10 ** 6)):
                    a_eta = alpha(n1, n2, a1, a2, E - eta)
                    row[f"ok_at_eps_minus_{float(eta):.0e}"] = bool(a_eta <= Q(d_r))
                    row[f"alpha_at_eps_minus_{float(eta):.0e}"] = float(a_eta)
            if r <= 3:
                row["profile"] = {str(th): round_profile(n1, n2, a1, a2, th) for th in (0.0, 0.001, 0.002, 0.005, 0.01, 0.02,
                                                                                         0.03, 0.04, 0.05)}
            rows.append(row)
            print(name, json.dumps({k: v for k, v in row.items() if k != "profile"}), flush=True)
        res["budgets"][name] = dict(delta=delta, rounds=rows, sum_alpha_exact=float(tot), sum_ok=bool(tot <= Q(delta)))
        print(name, "sum alpha", float(tot), "<= delta", delta, tot <= Q(delta), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
