"""Export a self-contained benchmark bundle (geometry rows + window class grids) so Colab needs only numpy/numba/scipy.
Declared selection: from Phase-1's random validation sample (engine b3 records), per side the windows at the 10/40/70/95%
quantiles of measured CPU-seconds (comparable with Phase-1 numbers).  Validation split only."""
import sys, os, json, glob
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from rpl.data import Scene

P1 = '/home/yushu/Projects/RIND/RIND3D_PosteriorLabels2/outputs/real_val_random90_b3'
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
R = [json.load(open(f)) for f in glob.glob(os.path.join(P1, '*.json')) if not os.path.basename(f).startswith('SAMPLE')]
sel = []
for side in (64, 128, 256):
    rr = sorted([r for r in R if r['side'] == side and r['tv_bound'] == 0], key=lambda r: r['cpu_seconds'])
    for q in (0.10, 0.40, 0.70, 0.95):
        r = rr[min(len(rr) - 1, int(q * len(rr)))]
        sel.append((r['scene'], r['window'], side, q, r['cpu_seconds'], r['n_cells_certified'], r['L_total']))
index = []
for sid, wid, side, q, cpu, cells, vol in sel:
    sc = Scene(sid)
    P, K, _, _ = sc.geometry_arrays()
    cls, o, L = sc.window_classes(wid)
    name = f'{sid[:8]}_{wid}'
    np.savez_compressed(os.path.join(out, name + '.npz'), P=P, K=K, cls=cls, origin=np.array(o), N=1024, truth=sc.truth)
    index.append(dict(name=name, scene=sid, window=int(wid), side=int(L), quantile=q, phase1_cpu_seconds=cpu,
                      phase1_cells=cells, phase1_volume=vol))
json.dump(dict(rule=__doc__, windows=index), open(os.path.join(out, 'INDEX.json'), 'w'), indent=1)
print(json.dumps(index, indent=1))
