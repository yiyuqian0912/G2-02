"""Controlled cases T1-T5 with the frozen engine b3: exact result, timing, brute-force verification -> JSON."""
import sys, json, time
import numpy as np
sys.path.insert(0, 'code')
from synth_cases import case
from tpl.run_b3 import run_window
from tpl.verify import verify
out = []
for nm in ['T1_edge_slab', 'T2_split_slab', 'T3_corner_tube', 'T4_tangent_face', 'T5_rot_cluster']:
    c = case(nm)
    run_window(c['geom'], c['win'], d1=4)                        # warm (JIT cache)
    t = time.process_time()
    r = run_window(c['geom'], c['win'], d1=4)
    cpu = time.process_time() - t
    v = verify(r, c['geom'], c['win'], n_sound=20000, n_box=400000, n_world=400000, inflate=0.5, seed=3)
    out.append(dict(case=nm, cpu_seconds=cpu, L=r['L_total'], undet=r['undet_total'], tv_bound=r['tv_bound'],
                    cells=r['n_cells_certified'], stats=r['stats'], verify=v,
                    first_study=dict(T1=dict(seconds=100, tv=0.0024, L=3.1369e-5), T2=dict(seconds=582, tv=0.0064, L=2.719e-5),
                                     T3=dict(seconds=69, tv=0.64, L=2.27e-8), T4=dict(seconds=29, tv=1.7e-5, L=1.5594e-3)).get(nm[:2])))
    print(nm, 'cpu %.3f L %.6e TV %.3g cells %d' % (cpu, r['L_total'], r['tv_bound'], r['n_cells_certified']),
          {k: v[k] for k in ('sound_points', 'sound_infeasible', 'box_feasible', 'box_feasible_outside_polytopes', 'world_feasible', 'world_feasible_outside', 'mc_volume', 'mc_volume_se')})
json.dump(out, open('outputs/SYNTH_B3.json', 'w'), indent=1)
