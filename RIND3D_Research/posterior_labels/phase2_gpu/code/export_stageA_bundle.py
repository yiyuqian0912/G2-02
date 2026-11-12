"""Self-contained bundle of Phase-1 Stage-A windows (128^3 receiver grid, validation scenes): geometry rows + classes."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from rpl.data import Scene
src = '/home/yushu/Projects/RIND/RIND3D_PosteriorLabels2/outputs/stageA_windows'
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
idx = json.load(open(os.path.join(src, 'INDEX.json')))
geo = {}
for w in idx['windows']:
    z = np.load(os.path.join(src, w['name'] + '.npz'))
    sid = str(z['scene'])
    if sid not in geo:
        P, K, _, _ = Scene(sid).geometry_arrays(); geo[sid] = (P, K)
    P, K = geo[sid]
    np.savez_compressed(os.path.join(out, w['name'] + '.npz'), P=P, K=K, cls=z['cls'], origin=z['origin'], N=int(z['N']), truth=z['truth'])
json.dump(dict(source=src, rule=idx.get('seed'), windows=idx['windows']), open(os.path.join(out, 'INDEX.json'), 'w'), indent=1)
print(len(idx['windows']))
