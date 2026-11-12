"""Summarize a real-window batch directory: per-window table + coverage/cost frontier (machine-readable)."""
import sys, json, glob, os
import numpy as np


def load(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, '*.json'))):
        if os.path.basename(f) in ('SAMPLE.json', 'SUMMARY.json'):
            continue
        rows.append(json.load(open(f)))
    return rows


def summarize(d):
    rows = load(d)
    out = dict(directory=d, n=len(rows), windows=[])
    for r in rows:
        v = r.get('verify', {})
        out['windows'].append(dict(scene=r['scene'][:8], window=r['window'], side=r['side'], components=r['components'],
                                   lit=r['counts']['lit'], dark=r['counts']['dark'], items=r['n_items'],
                                   cpu_seconds=r['cpu_seconds'], wall_seconds=r['wall_seconds'], tv_bound=r['tv_bound'],
                                   volume=r['L_total'], undet=r['undet_total'], cells=r['n_cells_certified'],
                                   truth_cell_mass=r['truth_cell_L'] / r['L_total'] if r['L_total'] > 0 else None,
                                   sound_infeasible=v.get('sound_infeasible'), sound_points=v.get('sound_points'),
                                   miss_box=v.get('box_feasible_outside_polytopes'), box_feasible=v.get('box_feasible'),
                                   miss_world=v.get('world_feasible_outside'), world_feasible=v.get('world_feasible')))
    W = out['windows']
    if not W:
        return out
    cpu = np.array([w['cpu_seconds'] for w in W]); tvb = np.array([w['tv_bound'] for w in W])
    out['frontier'] = {}
    for side in sorted(set(w['side'] for w in W)) + ['all']:
        m = np.array([side == 'all' or w['side'] == side for w in W])
        c = cpu[m]; t = tvb[m]
        out['frontier'][str(side)] = dict(
            n=int(m.sum()), cert_tv_le_0_05=float((t <= 0.05).mean()), cert_tv_le_0_10=float((t <= 0.10).mean()),
            cert_tv_eq_0=float((t == 0).mean()), cpu_median=float(np.median(c)), cpu_mean=float(c.mean()),
            cpu_p90=float(np.percentile(c, 90)), cpu_max=float(c.max()),
            within_1s=float((c <= 1).mean()), within_5s=float((c <= 5).mean()), within_60s=float((c <= 60).mean()),
            cells_median=float(np.median([w['cells'] for w, k in zip(W, m) if k])),
            truth_cell_mass_median=float(np.median([w['truth_cell_mass'] for w, k in zip(W, m) if k and w['truth_cell_mass'] is not None])))
    vs = [w for w in W if w['sound_points'] is not None]
    out['verification'] = dict(windows=len(vs), sound_points=int(sum(w['sound_points'] for w in vs)),
                               sound_infeasible=int(sum(w['sound_infeasible'] for w in vs)),
                               box_feasible=int(sum(w['box_feasible'] for w in vs)), miss_box=int(sum(w['miss_box'] for w in vs)),
                               world_feasible=int(sum(w['world_feasible'] for w in vs)), miss_world=int(sum(w['miss_world'] for w in vs)))
    json.dump(out, open(os.path.join(d, 'SUMMARY.json'), 'w'), indent=1)
    return out


if __name__ == '__main__':
    o = summarize(sys.argv[1])
    for w in sorted(o['windows'], key=lambda w: (w['side'], w['cpu_seconds'])):
        print('%s/%-4d side %3d comps %2d lit %8d dark %8d items %6d cpu %7.1fs TV %.4f vol %.3e cells %6d truth %.4f sound %s/%s miss %s/%s' % (
            w['scene'], w['window'], w['side'], w['components'], w['lit'], w['dark'], w['items'], w['cpu_seconds'], w['tv_bound'],
            w['volume'], w['cells'], w['truth_cell_mass'] or -1, w['sound_infeasible'], w['sound_points'], w['miss_box'], w['miss_world']))
    print(json.dumps(o.get('frontier'), indent=1)); print(json.dumps(o.get('verification')))
