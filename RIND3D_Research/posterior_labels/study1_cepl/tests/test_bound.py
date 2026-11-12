import sys, numpy as np
sys.path.insert(0, 'code')
from rpl.bound import tv_bound, tv_sup_bruteforce
rng = np.random.default_rng(0); gaps = []
for t in range(3000):
    n = int(rng.integers(1, 10))
    L = rng.random(n) * (rng.random(n) < 0.7); U = L + rng.random(n) * (rng.random(n) < 0.6) * rng.choice([1e-3, 1, 10])
    if L.sum() == 0: continue
    b, s = tv_bound(L, U), tv_sup_bruteforce(L, U)
    assert b >= s - 1e-12, (L, U, b, s)          # the bound is valid
    gaps.append(b - s)
print('bound - exact sup: median %.3g, max %.3g' % (np.median(gaps), np.max(gaps)))
assert tv_bound([1e-16, 0, 0], [1e-9, 0, 0]) == 0.0
L = np.r_[1e-16, 0, 0]; U = np.r_[1e-9, 1e-12, 0]; print('one-hot with leak', tv_bound(L, U), tv_sup_bruteforce(L, U))
print('ok')
