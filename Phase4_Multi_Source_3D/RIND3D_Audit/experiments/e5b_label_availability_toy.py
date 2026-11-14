"""E5b: Student-level bias from keeping only some labels, in the exact toy world of E5 (natural single-source law).

The loader default flatten(certified_only=True, accept_statistical=False) keeps a label only if its status is
"certified" (FastLabel: single-cell support or exact path). Which path a label takes depends on (G, W, r_j), i.e. on
privileged geometry. Here: keep an example iff |F(G, W, X)| <= t cells (t = 1 mimics 'single-cell support only'),
or drop the broadest x % of labels (mimics dropping hard / uncertified labels). Report the TV between the Student's
population optimum under the filtered population and the unfiltered one, and the share of X that disappear.
Run: nice -n 19 python e5b_label_availability_toy.py > out/e5b_label_availability_toy.json"""
import json, sys
import numpy as np
from toyworld import Toy, tv

T = Toy(); nW, S, R = len(T.windows), T.S, T.w * T.w; full = (1 << R) - 1
recs = []   # (wi, code, s, weight, |F|)
for b, pr, solid in T.geometries():
    legal = ~solid; nD = legal.sum(); code = T.codes(T.responses(b))
    inf = (code != 0) & (code != full) & legal[:, None] & ~T.in_win.T
    for wi in range(nW):
        idx = np.flatnonzero(inf[:, wi]); cc = code[idx, wi]
        sizes = {c: int((cc == c).sum()) for c in np.unique(cc)}
        for s, c in zip(idx, cc):
            recs.append((wi, int(c), int(s), pr / nD, sizes[int(c)]))
recs = np.array(recs, dtype=float)
keyX = recs[:, 0] * (1 << R) + recs[:, 1]
Xs, xi = np.unique(keyX, return_inverse=True)


def post(mask):
    J = np.zeros((len(Xs), S)); np.add.at(J, (xi[mask], recs[mask, 2].astype(int)), recs[mask, 3])
    m = J.sum(1); return J, m


Jall, mall = post(np.ones(len(recs), bool)); Pall = Jall / mall[:, None]; wX = mall / mall.sum()
out = dict(n_X=int(len(Xs)), label_size_quantiles={str(q): float(np.quantile(recs[:, 4], q)) for q in (.1, .5, .9, .99)})
size = recs[:, 4]
for name, mask in [("single_cell_only", size <= 1), ("size_le_3", size <= 3),
                   ("drop_broadest_5pct", size <= np.quantile(size, .95)), ("drop_broadest_1pct", size <= np.quantile(size, .99))]:
    J, m = post(mask); keep = m > 0
    Pf = J[keep] / m[keep][:, None]
    t = tv(Pf, Pall[keep])
    wk = m[keep] / m[keep].sum()
    out[name] = dict(example_mass_kept=float(recs[mask, 3].sum() / recs[:, 3].sum()), share_X_mass_lost=float(wX[~keep].sum()),
                     mean_TV_vs_unfiltered=float(wk @ t), max_TV=float(t.max()))
json.dump(out, sys.stdout, indent=1)
