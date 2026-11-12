import sys, time, numpy as np
sys.path.insert(0, 'code')
from rpl.data import Scene, scene_ids, splits, LIT, DARK, SOLID, N
from rpl.predicates import render_points
sp = splits(); ids = [s for s in scene_ids() if sp.get(s) == 'train'][:3]
for sid in ids:
    sc = Scene(sid); P, K, clo, chi = sc.geometry_arrays()
    for wid in (0, 100, 200):
        cls, o, L = sc.window_classes(wid)
        if L > 128: continue
        g = np.stack(np.meshgrid(*[np.arange(L)] * 3, indexing='ij'), -1).reshape(-1, 3)
        R = (np.asarray(o) + g + 0.5) / N
        t = time.time(); y = render_points(sc.truth, R, P, K); dt = time.time() - t
        c = cls.reshape(-1); free = c != SOLID
        mism = np.sum((y[free] != (c[free] == LIT)) & (y[free] >= 0)); und = np.sum(y[free] < 0)
        print(sid[:8], wid, L, 'components', len(K), 'free', free.sum(), 'lit', (c == LIT).sum(), 'mismatch', mism, 'undecided', und, f'{dt:.2f}s')
