import sys, time, json, numpy as np
sys.path.insert(0, 'code')
from synth_cases import case
from tpl.run import run_window, cell_of
names = sys.argv[1].split(',') if len(sys.argv) > 1 else ['T1_edge_slab']
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
for nm in names:
    c = case(nm, N=N)
    w = c['win']
    t = time.perf_counter()
    r = run_window(c['geom'], w)
    print(nm, 'N', N, 'origin', tuple(w.origin), w.counts(), 'items', r['n_items'])
    print('  stage1', r['stage1'], 'sec', {k: round(v, 3) for k, v in r['seconds'].items()})
    print('  stats', {k: v for k, v in r['stats'].items() if v})
    print('  L %.6e U %.6e undet %.3e cells_cert %d possible %d TVbound %.5f truth_cell_in %s' % (
        r['L_total'], r['U_total'], r['undet_total'], r['n_cells_certified'], r['n_cells_possible'], r['tv_bound'],
        cell_of(c['truth']) in set(r['cell'][r['lo'] > 0].tolist())))
