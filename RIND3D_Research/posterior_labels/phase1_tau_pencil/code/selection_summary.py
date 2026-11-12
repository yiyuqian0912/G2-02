"""Summarize the selection-law study: per window, the model-estimated TV(q_nat, q_v01) for the two pure model
components, and the noise floor that the CRN sampling error of the frequency ratios alone would produce."""
import sys, json, glob, os
import numpy as np

rows = []
rng = np.random.default_rng(0)
for f in sorted(glob.glob('outputs/selection_study/*.json')):
    r = json.load(open(f))
    src = r['sources'][1:]
    out = dict(window=f"{r['scene'][:8]}/{r['window']}", side=r['side'], cells=r['n_cells'], bin=r['bin'],
               extent=[round(x, 4) for x in r['support_extent']])
    for comp, key, se in (('valid', 'ratio_valid', 'se_ratio_valid'), ('bin', 'ratio_b', 'se_ratio_b')):
        rat = np.array([s[key] for s in src]); ses = np.array([s[se] for s in src])
        w = 1 / np.maximum(rat, 1e-12)
        tv = 0.5 * np.mean(np.abs(w / w.mean() - 1))
        # noise floor: ratios that are exactly 1 plus the measured CRN standard errors
        floor = []
        for _ in range(200):
            rr = 1 + rng.standard_normal(len(ses)) * ses
            ww = 1 / np.maximum(rr, 1e-6)
            floor.append(0.5 * np.mean(np.abs(ww / ww.mean() - 1)))
        out[f'tv_{comp}'] = float(tv); out[f'noise_floor_{comp}'] = float(np.mean(floor))
        out[f'ratio_range_{comp}'] = [float(rat.min()), float(rat.max())]
    out['W_reproduced_all'] = all(r['W_reproduced'])
    rows.append(out)
json.dump(rows, open('outputs/SELECTION_SUMMARY.json', 'w'), indent=1)
for o in rows:
    print('%-14s side %3d cells %6d extent %s  TV valid %.3f (noise %.3f) range %.2f-%.2f | TV bin %.3f (noise %.3f) range %.2f-%.2f | W ok %s' % (
        o['window'], o['side'], o['cells'], o['extent'], o['tv_valid'], o['noise_floor_valid'], *o['ratio_range_valid'],
        o['tv_bin'], o['noise_floor_bin'], *o['ratio_range_bin'], o['W_reproduced_all']))
