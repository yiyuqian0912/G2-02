"""Scaling assessment from measured per-window costs (measurements and extrapolations kept separate).

Inputs (all measured in this study, see outputs/):
  * batch ANALYSIS.json files (seconds, nodes, completion per window, single core of an i7 laptop CPU)
  * synthetic reference results (outputs/synth_*.json)
  * boundary-voxel counts per window side and their cross-window multiplicity (outputs/overlap_val3.json)
Extrapolation rules (declared):
  * per-node cost scales with the number of boundary receivers of the window (measured ~ side^2:
    4.5k / 16k / 70k for side 64 / 128 / 256);
  * node counts at side 128/256 are assumed no smaller than at side 64 (optimistic);
  * incomplete windows are charged their budget (a lower bound on their true cost).
"""
import json, glob, sys, numpy as np

def load(d):
    a = json.load(open(f'{d}/ANALYSIS.json'))
    return a

def main(dirs, out):
    rows = sum((load(d) for d in dirs), [])
    by = {}
    for r in rows:
        by.setdefault(r['side'], []).append(r)
    ov = json.load(open('outputs/overlap_val3.json'))
    bvox = {s: float(np.mean([o['mean_boundary_voxels_by_side'][str(s)] for o in ov])) for s in (64, 128, 256)}
    mult = float(np.mean([o['boundary_multiplicity'] for o in ov]))
    meas = {}
    for s, rs in by.items():
        sec = np.array([r['seconds'] for r in rs]); comp = np.array([r['complete'] for r in rs])
        meas[str(s)] = dict(windows=len(rs), complete=int(comp.sum()), seconds_mean=float(sec.mean()),
                            seconds_median=float(np.median(sec)), seconds_max=float(sec.max()),
                            seconds_mean_complete=float(sec[comp].mean()) if comp.any() else None,
                            note='incomplete windows charged their budget: a lower bound on true cost')
    s64 = meas.get('64')
    proj = {}
    if s64:
        base = s64['seconds_mean']
        per_side = {s: base * bvox[s] / bvox[64] for s in (64, 128, 256)}
        scen = {'S256_25.6M': dict(windows=25.6e6, mix={64: 85 / 256, 128: 85 / 256, 256: 86 / 256}),
                'S102_10.16M': dict(windows=10.16e6, mix={64: 1 / 3, 128: 1 / 3, 256: 1 / 3})}
        for k, v in scen.items():
            cpu_s = v['windows'] * sum(f * per_side[s] for s, f in v['mix'].items())
            proj[k] = dict(cpu_core_hours=cpu_s / 3600, cpu_core_years=cpu_s / 3600 / 8766,
                           with_ideal_receiver_cache_lower=cpu_s / 3600 / mult,
                           assumption='per-window cost = measured side-64 mean x boundary-voxel ratio; no failures '
                                      'retried; ideal cross-window receiver cache divides cost by the measured '
                                      f'boundary multiplicity {mult:.2f}')
        proj['per_window_seconds_extrapolated'] = per_side
    res = dict(measured=meas, boundary_voxels_per_window=bvox, boundary_multiplicity=mult, projection=proj,
               nll_reference='true-source NLL needs one 24-byte source record per scene and no physics at label time')
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps(res, indent=1))

if __name__ == '__main__':
    main(sys.argv[1:-1], sys.argv[-1])
