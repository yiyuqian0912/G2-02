import sys, json, numpy as np
sys.path.insert(0, 'code')
from synth_cases import case
from tpl.run import run_window
from tpl.verify import verify
for nm in sys.argv[1].split(','):
    c = case(nm)
    r = run_window(c['geom'], c['win'], d1=4)
    v = verify(r, c['geom'], c['win'], inflate=0.5)
    print(nm, 'L %.6e TV %.4g sec %.2f' % (r['L_total'], r['tv_bound'], r['seconds']['total']), json.dumps(v))
