"""Scene-level sharing opportunity: multiplicity of window voxels and of boundary voxels across the 256 selected
windows of a scene (validation scenes)."""
import sys, json, numpy as np
sys.path.insert(0, 'code')
from rpl.data import Scene, scene_ids, splits
from rpl.engine2 import boundary_mask
sp = splits(); ids = sorted(s for s in scene_ids() if sp.get(s) == 'val')
rng = np.random.default_rng(7)
out = []
for sid in rng.choice(ids, 3, replace=False):
    sc = Scene(sid)
    keys = []; per_w = []
    for wid in range(len(sc.windows)):
        cls, o, L = sc.window_classes(wid)
        bd = boundary_mask(cls)
        idx = np.argwhere(bd) + np.asarray(o)
        k = (idx[:, 0].astype(np.int64) << 20) | (idx[:, 1].astype(np.int64) << 10) | idx[:, 2]
        keys.append(k); per_w.append((L, len(k)))
    allk = np.concatenate(keys)
    u, cnt = np.unique(allk, return_counts=True)
    mult = dict(zip(u.tolist()[:0], []))
    # fraction of each window's boundary voxels that occur in at least one other window
    lookup = dict()
    cnt_of = np.zeros(0)
    shared_frac = []
    pos = np.searchsorted(u, allk)
    c_all = cnt[pos]
    start = 0
    for (L, n) in per_w:
        c = c_all[start:start + n]; start += n
        shared_frac.append(float((c > 1).mean()) if n else 0.0)
    vox = sum(L ** 3 for L, _ in per_w)
    r = dict(scene=sid[:8], boundary_voxel_memberships=int(len(allk)), distinct_boundary_voxels=int(len(u)),
             boundary_multiplicity=float(len(allk) / len(u)), window_voxel_memberships=int(vox),
             mean_boundary_voxels_by_side={str(s): float(np.mean([n for L, n in per_w if L == s])) for s in (64, 128, 256)},
             mean_shared_fraction_by_side={str(s): float(np.mean([f for (L, _), f in zip(per_w, shared_frac) if L == s])) for s in (64, 128, 256)})
    print(json.dumps(r), flush=True); out.append(r)
json.dump(out, open('outputs/overlap_val3.json', 'w'), indent=1)
