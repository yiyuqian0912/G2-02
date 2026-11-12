"""Per-window label statistics from saved cell masses (outputs/*/<win>.npz): support size, number of connected
modes on the 128^3 grid (26-connectivity of cells with positive mass), effective number of cells (exp entropy),
cells needed for 99% / 95% of the mass (truncation TV <= 0.01 / 0.05), and sparse storage sizes."""
import sys, os, glob, json
import numpy as np
from scipy import ndimage

G = 128


def stats(cells, L):
    m = L > 0
    cells = cells[m]; q = L[m] / L[m].sum()
    xyz = np.stack(np.unravel_index(cells, (G, G, G)), 1)
    lo = xyz.min(0); hi = xyz.max(0)
    box = np.zeros(tuple(hi - lo + 3), bool)
    box[tuple((xyz - lo + 1).T)] = True
    lab, nmode = ndimage.label(box, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(lab[tuple((xyz - lo + 1).T)], weights=q)[1:]
    o = np.sort(q)[::-1]
    c = np.cumsum(o)
    k99 = int(np.searchsorted(c, 0.99) + 1); k95 = int(np.searchsorted(c, 0.95) + 1)
    H = float(-(q * np.log(q)).sum())
    return dict(cells=int(len(q)), modes=int(nmode), mode_mass_sorted=sorted(sizes.tolist(), reverse=True)[:5],
                eff_cells=float(np.exp(H)), cells_99=k99, cells_95=k95, extent_cells=(hi - lo + 1).tolist(),
                bytes_sparse_full=int(len(q) * 6), bytes_sparse_99=int(k99 * 6))


if __name__ == '__main__':
    out = {}
    for d in sys.argv[1:]:
        for f in sorted(glob.glob(os.path.join(d, '*.npz'))):
            if f.endswith('_tpe.npz'):
                z = np.load(f); cells, L = z['cells'], z['q']
            else:
                z = np.load(f); cells, L = z['cells'], z['L']
            if L.sum() <= 0:
                continue
            out[os.path.relpath(f)] = stats(cells, L)
    json.dump(out, open('outputs/LABEL_STATS.json', 'w'), indent=1)
    v = list(out.values())
    for k in ('cells', 'modes', 'eff_cells', 'cells_99'):
        a = np.array([x[k] for x in v]); print(k, 'median', np.median(a), 'mean %.1f' % a.mean(), 'max', a.max())
    print('windows', len(v), 'multimodal (>1 mode)', sum(x['modes'] > 1 for x in v))
