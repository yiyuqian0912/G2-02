"""Rigorous TV bound for a label built from per-cell volume intervals [L_c, U_c] (docs/METHOD.md section 5).

Label qhat = L / Lt.  For true masses m = L + e, 0 <= e <= x = U - L, E = sum(e):
    |Lt e_c - E L_c| <= e_c (Lt - L_c) + L_c (E - e_c)   =>   TV(q, qhat) <= sum_c e_c w_c / (Lt + E),
with w_c = 1 - L_c / Lt in [0, 1].  The right-hand side is a linear-fractional function of e; its maximum over the
box 0 <= e <= x is attained by adding cells in decreasing w_c while w_c exceeds the current value (exact greedy).
The bound is 0 when every uncertain cell is the one cell holding all certified mass (certified one-hot) and reduces
to U/(Lt+U) when certified mass is spread thinly."""
import numpy as np


def tv_bound(L, U):
    L = np.asarray(L, float); U = np.asarray(U, float)
    Lt = L.sum()
    if Lt <= 0:
        return 1.0
    x = np.maximum(U - L, 0.0)
    w = 1.0 - L / Lt
    m = x > 0
    x = x[m]; w = w[m]
    order = np.argsort(-w, kind='stable')
    num = 0.0; den = Lt; best = 0.0
    for i in order:
        if w[i] <= best:
            break
        num += x[i] * w[i]; den += x[i]
        best = max(best, num / den)
    return float(min(1.0, best))


def tv_sup_bruteforce(L, U, n_vertices=None):
    """Exact sup of TV over the box, by enumerating all vertices m in {L_c, U_c} (small n only)."""
    import itertools
    L = np.asarray(L, float); U = np.asarray(U, float); qh = L / L.sum(); best = 0.0
    for bits in itertools.product([0, 1], repeat=len(L)):
        m = np.where(np.array(bits, bool), U, L)
        best = max(best, 0.5 * np.abs(m / m.sum() - qh).sum())
    return best
