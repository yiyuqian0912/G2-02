"""Numerical check of Lemma 2 (source-side pencil form) and Lemma 2W (window-side pencil form) against the
float Cyrus-Beck segment predicate, on random convex polyhedra (rotated boxes and real RIND3D components)."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))
from rpl.synth import box_rows
from rpl.predicates import segment_hits
from tpl.geom import Geom


def pencil_dark(P, K, adj, aoff, c, s, x, side):
    """x in S_c(s) by Lemma 2 (side='S': aspect of the source) or Lemma 2W (side='W': aspect of the receiver).
    Returns None when the point pair is too close to a face plane to decide (|f| < 1e-9)."""
    f_s = P[c, :K[c], :3] @ s + P[c, :K[c], 3]
    f_x = P[c, :K[c], :3] @ x + P[c, :K[c], 3]
    if np.min(np.abs(f_s)) < 1e-9 or np.min(np.abs(f_x)) < 1e-9:
        return None
    if side == 'S':
        F = f_s > 0
        if not F.any():
            return True                       # s inside C
        if np.any(f_x[F] > 0):
            return False                      # x not gated
        a, b = s, x
    else:
        F = f_x > 0
        if np.any(f_s[F] > 0):
            return False                      # s outside G
        a, b = x, s
    fa = P[c, :K[c], :3] @ a + P[c, :K[c], 3]
    fb = P[c, :K[c], :3] @ b + P[c, :K[c], 3]
    for t in range(aoff[c], aoff[c + 1]):
        i, j = adj[t, 1], adj[t, 2]
        if F[i] and not F[j]:
            f, k = i, j
        elif F[j] and not F[i]:
            f, k = j, i
        else:
            continue
        # tau_e(a) <= tau_e(b) with tau = f_k / f_f  (S: f_f(s) > 0 > f_f(x); W: f_f(x) > 0 > f_f(s))
        if not (fa[k] / fa[f] <= fb[k] / fb[f]):
            return False
    return True


def run(n_pairs=200000, seed=0):
    rng = np.random.default_rng(seed)
    from scipy.spatial.transform import Rotation as Rot
    bodies = []
    for _ in range(12):
        c = rng.uniform(0.3, 0.7, 3); h = rng.uniform(0.02, 0.12, 3)
        bodies.append(box_rows(c - h, c + h, Rot.random(random_state=rng.integers(1 << 30)).as_matrix()))
    geoms = {'rotated_boxes': Geom.from_bodies(bodies)}
    try:
        from rpl.data import Scene, scene_ids, splits
        sp = splits()
        train = [s for s in scene_ids() if sp.get(s) == 'train']   # never the test split
        for sid in train[:3]:
            geoms['scene_' + sid[:8]] = Geom.from_scene(Scene(sid))
    except Exception as e:  # noqa: BLE001
        print('real scenes unavailable:', e)
    out = {}
    for name, g in geoms.items():
        tot = dict(pairs=0, decided_S=0, mismatch_S=0, decided_W=0, mismatch_W=0, undecided_ref=0, dark=0)
        for _ in range(n_pairs):
            c = int(rng.integers(g.C))
            s = rng.random(3); x = rng.random(3)
            ref = segment_hits(s, x, g.P, g.K, c)
            if ref < 0:
                tot['undecided_ref'] += 1
                continue
            tot['pairs'] += 1
            tot['dark'] += ref == 1
            for side in ('S', 'W'):
                v = pencil_dark(g.P, g.K, g.adj, g.aoff, c, s, x, side)
                if v is None:
                    continue
                tot['decided_' + side] += 1
                tot['mismatch_' + side] += int(v != (ref == 1))
        out[name] = tot
        print(name, tot)
    return out


if __name__ == '__main__':
    r = run()
    json.dump(r, open(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'TEST_TAU_LEMMAS.json'), 'w'), indent=1)
    assert all(v['mismatch_S'] == 0 and v['mismatch_W'] == 0 for v in r.values())
    print('OK: 0 mismatches')
