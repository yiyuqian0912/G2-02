"""Print the headline numbers used in the report text (all read from outputs/)."""
import json, glob, os
import numpy as np


def jl(d):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, '*.json')))
            if not os.path.basename(f).startswith('SAMPLE') and os.path.basename(f) != 'SUMMARY.json']


R = jl('outputs/real_val_random90_b3')
print('random sample: windows', len(R), 'of 90; TV==0', sum(r['tv_bound'] == 0 for r in R),
      'TV<=0.05', sum(r['tv_bound'] <= 0.05 for r in R))
for side in (64, 128, 256):
    rr = [r for r in R if r['side'] == side]
    if not rr:
        continue
    c = np.array([r['cpu_seconds'] for r in rr]); tv = np.array([r['tv_bound'] for r in rr])
    tm = np.array([r['truth_cell_L'] / r['L_total'] for r in rr if r['L_total'] > 0])
    ce = np.array([r['n_cells_certified'] for r in rr])
    mem = [r['maxrss_mb'] for r in rr if r.get('maxrss_mb')]
    print(f'  side {side}: n {len(rr)} exact {(tv == 0).sum()} cpu median {np.median(c):.1f} mean {c.mean():.1f} p90 {np.percentile(c, 90):.1f} max {c.max():.1f}'
          f' | <=1s {(c <= 1).mean():.2f} <=5s {(c <= 5).mean():.2f} <=60s {(c <= 60).mean():.2f} <=600s {(c <= 600).mean():.2f}'
          f' | cells median {np.median(ce):.0f} one-cell {(ce == 1).sum()} | truth mass median {np.median(tm):.4f} | rss median {np.median(mem) if mem else 0:.0f} MB')
c = np.array([r['cpu_seconds'] for r in R])
mix = sum(q * np.mean([r['cpu_seconds'] for r in R if r['side'] == s]) for s, q in ((64, 85), (128, 85), (256, 86)) if any(r['side'] == s for r in R)) / 256
print(f'  mix mean cpu per window {mix:.1f} s -> 25.6M windows {25.6e6 * mix / 3600:.3g} core-hours ({25.6e6 * mix / 3600 / 8766:.3g} core-years)')
vs = [r['verify'] for r in R if 'verify' in r]
print('  verification: sound pts', sum(v.get('sound_points', 0) for v in vs), 'infeasible', sum(v.get('sound_infeasible', 0) for v in vs),
      'feasible found', sum(v.get('box_feasible', 0) + v.get('world_feasible', 0) for v in vs), 'outside', sum(v.get('box_feasible_outside_polytopes', 0) + v.get('world_feasible_outside', 0) for v in vs))
P = jl('outputs/real_paired30_b3')
print('paired: windows', len(P), 'exact', sum(r['tv_bound'] == 0 for r in P))
A = [json.load(open(f)) for f in glob.glob('outputs/stageA_compare30/*.json') if 'ARGS' not in f]
print('stageA compare: windows', len(A))
for k in sorted({k for r in A for k in r['results']}):
    rr = [r['results'][k] for r in A if k in r['results']]
    print(f'  {k}: TV<=0.05 {sum(x["tv_bound"] <= 0.05 for x in rr)}/{len(rr)} median cpu {np.median([x["cpu_seconds"] for x in rr]):.2f}')
