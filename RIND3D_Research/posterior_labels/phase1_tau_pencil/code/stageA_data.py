"""Stage A platform: the same v0.1 scenes (validation split), the same true sources and the same physical window
boxes, observed on a 128^3 receiver grid instead of 1024^3 (window sides 64/128/256 -> 8/16/32 voxels).
Same ray-visibility semantics (closed segment meets the solid), same FREE/SOLID convention, float renderer
(windows with an undecided receiver are skipped and counted).  Declared sample: seed below."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from rpl.data import Scene, splits
from rpl.predicates import render_points
from tpl.geom import Geom

N = 128


def render(geom, s, origin, L):
    g = np.stack(np.meshgrid(*[np.arange(L)] * 3, indexing='ij'), -1).reshape(-1, 3)
    X = (np.asarray(origin) + g + 0.5) / N
    P, K = geom.P, geom.K
    solid = np.zeros(len(X), bool)
    for c in range(P.shape[0]):
        solid |= np.all(X @ P[c, :K[c], :3].T + P[c, :K[c], 3] <= 0, axis=1)
    y = render_points(np.asarray(s, float), X, P, K)
    und = int((y[~solid] < 0).sum())
    cls = np.where(solid, 2, np.where(y == 1, 1, 0)).astype(np.uint8).reshape(L, L, L)
    return cls, und


def main(n_scenes=20, per_scene=6, seed=20261112, out='outputs/stageA_windows'):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed)
    sp = splits()
    val = sorted(s for s, v in sp.items() if v == 'val')
    pick = rng.choice(val, n_scenes, replace=False)
    meta = []
    skipped = dict(undecided=0, uninformative=0, contains_source=0)
    for sid in pick:
        sc = Scene(sid)
        geom = Geom.from_scene(sc)
        W = sc.windows
        order = rng.permutation(len(W))
        got = 0
        for wid in order:
            if got >= per_scene:
                break
            x, y, z, L = (int(v) for v in W[wid])
            L8 = L // 8
            o = (x // 8, y // 8, z // 8)
            if any(o[k] + L8 > N for k in range(3)):
                continue
            lo = np.array(o) / N; hi = (np.array(o) + L8) / N
            if np.all((sc.truth >= lo) & (sc.truth < hi)):
                skipped['contains_source'] += 1
                continue
            cls, und = render(geom, sc.truth, o, L8)
            if und:
                skipped['undecided'] += 1
                continue
            if not ((cls == 0).any() and (cls == 1).any()):
                skipped['uninformative'] += 1
                continue
            name = f'{sid[:8]}_{wid}'
            np.savez_compressed(os.path.join(out, name + '.npz'), cls=cls, origin=np.array(o), N=N, scene=sid,
                                wid=wid, truth=sc.truth)
            meta.append(dict(name=name, scene=sid, wid=int(wid), side=L8, origin=list(o), dark=int((cls == 0).sum()),
                             lit=int((cls == 1).sum()), solid=int((cls == 2).sum())))
            got += 1
    json.dump(dict(seed=seed, n_scenes=n_scenes, per_scene=per_scene, N=N, skipped=skipped, windows=meta),
              open(os.path.join(out, 'INDEX.json'), 'w'), indent=1)
    print(len(meta), skipped)


if __name__ == '__main__':
    main()
