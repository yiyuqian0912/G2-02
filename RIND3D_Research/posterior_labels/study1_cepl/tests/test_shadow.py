"""Shadow-polyhedron rows (front faces + silhouette edges) vs full Fourier-Motzkin rows vs the exact
canonical segment oracle, on real v0.1 geometry (train scenes)."""
import sys, numpy as np
sys.path.insert(0, 'code')
from rpl.data import Scene, scene_ids, splits
from rpl.engine2 import shadow_rows, scene_geometry
from rpl.predicates import shadow_planes

def run(n_scenes=6, n_rec=40, n_pts=4000, seed=0):
    rng = np.random.default_rng(seed)
    sp = splits(); ids = [s for s in scene_ids() if sp.get(s) == 'train']
    tot = dict(points=0, red_vs_full=0, red_vs_exact=0, near=0)
    for sid in rng.choice(ids, n_scenes, replace=False):
        sc = Scene(sid); P, K, clo, chi, adj, aoff = scene_geometry(sc)
        alpha = np.empty(P.shape[1]); rows = np.empty((4096, 4)); full = np.empty((4096, 4))
        for c in range(len(K)):
            ctr = (clo[c] + chi[c]) / 2; ext = (chi[c] - clo[c])
            for _ in range(n_rec):
                r = rng.random(3)
                if sc.ctx.contains(r[None])[0]:
                    continue
                n1, f1 = shadow_rows(r, P, K, c, adj, aoff, alpha, rows)
                n2, f2 = shadow_planes(r, P, K, c, full)
                if f1 != 0 or f2 != 0:
                    continue
                # points concentrated around the component so both outcomes occur
                S = ctr + ext * (rng.random((n_pts, 3)) * 3 - 1.5)
                S = np.clip(S, 1e-6, 1 - 1e-6)
                g1 = S @ rows[:n1, :3].T + rows[:n1, 3]
                g2 = S @ full[:n2, :3].T + full[:n2, 3]
                in1 = np.all(g1 <= 0, 1); in2 = np.all(g2 <= 0, 1)
                near = (np.min(np.abs(g1), 1) < 1e-9) | (np.min(np.abs(g2), 1) < 1e-9)
                # exact oracle: bind the receiver as an endpoint; response 0 means [r, s] meets the solid
                m = sc.ctx.bind(r)
                y, solid, _ = m.native.points(np.ascontiguousarray(S), False)
                # component-specific truth: segment meets C_c  <=>  shadowed when only C_c is present -- use the
                # union oracle only where the other components are irrelevant: compare union membership
                tot['points'] += int((~near).sum()); tot['near'] += int(near.sum())
                tot['red_vs_full'] += int((in1 != in2)[~near].sum())
                # union over components from reduced rows
                union = np.zeros(len(S), bool)
                for cc in range(len(K)):
                    nn, ff = shadow_rows(r, P, K, cc, adj, aoff, alpha, rows)
                    if ff == 1:
                        union[:] = True; continue
                    if ff == 2:
                        continue
                    union |= np.all(S @ rows[:nn, :3].T + rows[:nn, 3] <= 0, 1)
                shadowed = (y == 0)
                tot['red_vs_exact'] += int((union != shadowed)[~near].sum())
    print(tot)
    assert tot['red_vs_full'] == 0 and tot['red_vs_exact'] == 0

if __name__ == '__main__':
    run()
