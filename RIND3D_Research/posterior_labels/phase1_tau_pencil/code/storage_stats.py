"""Label statistics and storage over all real 1024^3 labels (engine b3 batches): connected modes, effective cells,
sparse storage in format v2 and its exact quantisation TV.  Writes outputs/LABEL_STORAGE.json + LABEL_STATS.json."""
import sys, glob, json, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from tpl.labels import encode, storage_tv
from label_stats import stats

dirs = ['outputs/real_val_random90_b3', 'outputs/real_val64_prev24_b3', 'outputs/real_paired30_b3']
rows = {}
for d in dirs:
    for f in sorted(glob.glob(os.path.join(d, '*.npz'))):
        z = np.load(f); m = z['L'] > 0
        if not m.any():
            continue
        cells, L = z['cells'][m], z['L'][m]
        e = encode(cells, L); e99 = encode(cells, L, truncate_tv=0.01)
        s = stats(cells, L)
        s.update(bytes=len(e['blob']), storage_tv=storage_tv(cells, L, e), bytes99=len(e99['blob']))
        rows[os.path.relpath(f)] = s
json.dump(rows, open('outputs/LABEL_STATS.json', 'w'), indent=1)
v = list(rows.values())
if v:
    a = lambda k: np.array([x[k] for x in v])
    out = dict(n=len(v), cells_median=float(np.median(a('cells'))), cells_mean=float(a('cells').mean()), cells_max=int(a('cells').max()),
               bytes_median=float(np.median(a('bytes'))), bytes_mean=float(a('bytes').mean()), bytes_max=int(a('bytes').max()),
               storage_tv_max=float(a('storage_tv').max()), bytes99_median=float(np.median(a('bytes99'))),
               modes_median=float(np.median(a('modes'))), multimodal=int((a('modes') > 1).sum()),
               eff_cells_median=float(np.median(a('eff_cells'))), one_cell=int((a('cells') == 1).sum()))
    json.dump(out, open('outputs/LABEL_STORAGE.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))
