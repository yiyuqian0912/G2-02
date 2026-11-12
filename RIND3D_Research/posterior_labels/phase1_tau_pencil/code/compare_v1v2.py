import sys, time, numpy as np
sys.path.insert(0, 'code')
from synth_cases import case
from tpl import run_v1, run2, run3
from tpl.run import cell_intervals
for nm in sys.argv[1].split(','):
    c = case(nm)
    out = []
    for mod, kw in ((run_v1, {}), (run2, dict(use_wmode=True)), (run3, {})):
        r = mod.run_window(c['geom'], c['win'], d1=4, **kw)
        r = mod.run_window(c['geom'], c['win'], d1=4, **kw)  # warm
        cells, L, U = cell_intervals(r)
        out.append((cells, L, U, r['seconds']['total'], r['stats']['nodes'], r['stats']['entry_evals']))
    ref = dict(zip(out[0][0].tolist(), out[0][1].tolist()))
    for k, (cells, L, U, t, nd, ev) in enumerate(out):
        d = dict(zip(cells.tolist(), L.tolist()))
        tv = 0.5 * sum(abs(d.get(q, 0) / sum(d.values()) - ref.get(q, 0) / sum(ref.values())) for q in set(d) | set(ref))
        print(nm, ['v1', 'v2 W1', 'v3 classes'][k], 'L %.9e undet %.2e cells %d TV-vs-v1 %.2e sec %.3f nodes %d evals %d' % (
            L.sum(), (U - L).sum(), (L > 0).sum(), tv, t, nd, ev))
