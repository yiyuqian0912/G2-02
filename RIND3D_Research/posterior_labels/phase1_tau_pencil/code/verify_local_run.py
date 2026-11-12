import sys, json
sys.path.insert(0, 'code')
from synth_cases import case
from tpl.run_b3 import run_window
from tpl.verify import verify_local
out = {}
for nm in ['T1_edge_slab', 'T2_split_slab', 'T3_corner_tube', 'T4_tangent_face', 'T5_rot_cluster']:
    c = case(nm)
    r = run_window(c['geom'], c['win'], d1=4)
    out[nm] = verify_local(r, c['geom'], c['win'], n_polys=600, per=100)
    print(nm, out[nm])
json.dump(out, open('outputs/VERIFY_LOCAL_SYNTH.json', 'w'), indent=1)
