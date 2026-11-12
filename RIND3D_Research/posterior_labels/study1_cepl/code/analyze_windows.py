"""Per-window certified quantities from saved per-cell intervals [L_c, U_c] (batch outputs).
  tv_bound_tight : rigorous TV bound of the label L/sum(L)  (rpl.bound.tv_bound)
  offtruth_lo/hi : certified interval for 1 - q(c*), the posterior mass outside the true source's 128^3 cell
                   (0 means the exact label equals the one-hot NLL target at 128^3)
  same at 16^3 for the coarse cells."""
import sys, json, glob, numpy as np
sys.path.insert(0, 'code')
from rpl.bound import tv_bound

def coarse_of(cells):
    ix, rem = np.divmod(cells, 128 * 128); iy, iz = np.divmod(rem, 128)
    return ((ix // 8) * 16 + iy // 8) * 16 + iz // 8

def offtruth(cells, lo, hi, tc):
    m = cells == tc
    Lt, Ut = lo[m].sum(), hi[m].sum(); Lo, Uo = lo[~m].sum(), hi[~m].sum()
    lo_b = Lo / (Ut + Lo) if Ut + Lo > 0 else 0.0
    hi_b = 1.0 - (Lt / (Lt + Uo) if Lt + Uo > 0 else 0.0)
    return float(lo_b), float(hi_b)

def analyze(d):
    out = []
    for f in sorted(glob.glob(f'{d}/*.json')):
        if f.endswith(('SAMPLE.json', 'ANALYSIS.json', '.err.json', '.terminated.json')):
            continue
        s = json.load(open(f)); z = np.load(f[:-5] + '.npz')
        cells, lo, hi = z['cells'], z['lo'], z['hi']
        tc = s.get('truth_cell')
        if tc is None:
            import importlib
        out.append(dict(scene=s['scene'][:8], window=s['window'], side=s['side'], seconds=s['seconds'],
                        complete=s['complete'], nodes=s['stats']['nodes'], L=s['L'], U=s['U'], lit=s['lit'],
                        dark=s['dark'], minority=min(s['lit'], s['dark']), components=s['components'],
                        n_cells_certified=int((lo > 0).sum()), n_cells_possible=int((hi > 0).sum()),
                        tv_bound_tight=tv_bound(lo, hi), cells=cells, lo=lo, hi=hi))
    return out

if __name__ == '__main__':
    import os
    from rpl.data import Scene
    rows = analyze(sys.argv[1])
    res = []
    for r in rows:
        sid = [p for p in os.listdir(os.path.dirname(__file__) + '/../' + sys.argv[1]) if p.startswith(r['scene'])]
        full = json.load(open(f"{sys.argv[1]}/{r['scene']}_{r['window']}.json"))['scene']
        t = np.fromfile(os.path.join(Scene.__init__.__globals__['rindenv'].SCENES, full, 'driver.f64'), '<f8')
        tc = (int(t[0] * 128) * 128 + int(t[1] * 128)) * 128 + int(t[2] * 128)
        tck = (int(t[0] * 16) * 16 + int(t[1] * 16)) * 16 + int(t[2] * 16)
        fo = offtruth(r['cells'], r['lo'], r['hi'], tc)
        co = offtruth(coarse_of(r['cells']), r['lo'], r['hi'], tck)
        rr = {k: v for k, v in r.items() if k not in ('cells', 'lo', 'hi')}
        rr.update(offtruth_fine=fo, offtruth_coarse=co, truth_cell_certified=bool(r['lo'][r['cells'] == tc].sum() > 0))
        res.append(rr)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rr.items()}))
    json.dump(res, open(sys.argv[1] + '/ANALYSIS.json', 'w'), indent=1)
