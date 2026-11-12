"""Generate docs/RESULTS_TABLES.md from the saved JSON evidence (no numbers are typed by hand)."""
import json, glob, os, numpy as np

OUT = 'docs/RESULTS_TABLES.md'
L = []
def p(s=''):
    L.append(s)

# synthetic reference cases
syn = []
for f in sorted(glob.glob('outputs/synth_*.json')):
    if os.path.exists(f):
        for r in json.load(open(f)):
            syn.append((os.path.basename(f), r))
p('## Controlled reference cases (engine vs brute force over every window receiver)')
p()
p('| Case | Engine | Engine s | Certified volume L | Brute-force volume (stratified) | Cells > 0 | TV bound (cert.) | TV engine vs MC | MC noise TV | Uniform feasible pts outside engine cells |')
p('|---|---|---|---|---|---|---|---|---|---|')
for src, r in syn:
    eng = 'v3' if 'engine3' in src else 'v2'
    p(f"| {r['case']} | {eng} | {r['engine_seconds']:.0f} | {r['L_engine']:.4e} | {r['mc_volume']:.4e} | {r['cells_positive']} | "
      f"{r['tv_bound']:.4f} | {r['tv_engine_vs_mc']:.4f} | {r['mc_expected_tv_noise']:.4f} | "
      f"{r['uniform_feasible_outside_engine_cells']} of {r['uniform_feasible']} |")
p()

# real windows
for d in sorted(glob.glob('outputs/batch_*')):
    fa = os.path.join(d, 'ANALYSIS.json')
    if not os.path.exists(fa):
        continue
    rows = json.load(open(fa))
    p(f'## Real windows: {os.path.basename(d)} ({len(rows)} windows, validation split)')
    p()
    p('| Scene | W | Side | Lit / dark FREE | Comp. | Nodes | s | Complete | Certified L | Cells cert. / possible | TV bound | Off-truth mass 128³ [lo, hi] | Off-truth mass 16³ [lo, hi] |')
    p('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for r in sorted(rows, key=lambda r: (r['minority'])):
        fo = r['offtruth_fine']; co = r['offtruth_coarse']
        p(f"| {r['scene']} | {r['window']} | {r['side']} | {r['lit']} / {r['dark']} | {r['components']} | {r['nodes']} | "
          f"{r['seconds']:.0f} | {'yes' if r['complete'] else 'no'} | {r['L']:.2e} | {r['n_cells_certified']} / {r['n_cells_possible']} | "
          f"{r['tv_bound_tight']:.3f} | [{fo[0]:.3f}, {fo[1]:.3f}] | [{co[0]:.3f}, {co[1]:.3f}] |")
    p()
    sec = np.array([r['seconds'] for r in rows]); comp = np.array([r['complete'] for r in rows])
    tvb = np.array([r['tv_bound_tight'] for r in rows])
    fo_lo = np.array([r['offtruth_fine'][0] for r in rows]); fo_hi = np.array([r['offtruth_fine'][1] for r in rows])
    p(f'* Complete within budget: {comp.sum()} / {len(rows)}; certified (TV bound <= 0.05): {(tvb <= 0.05).sum()}; '
      f'<= 0.01: {(tvb <= 0.01).sum()}.')
    p(f'* Seconds per window (one core): median {np.median(sec):.0f}, mean {sec.mean():.0f}, max {sec.max():.0f}.')
    p(f'* Certified to differ from the one-hot NLL target (off-truth mass lower bound >= 0.5 at 128³): '
      f'{(fo_lo >= 0.5).sum()} / {len(rows)}; certified one-hot (upper bound <= 0.01): {(fo_hi <= 0.01).sum()} / {len(rows)}.')
    p()

# bank comparison
banks = sorted(glob.glob('outputs/bank_*.json'))
if banks:
    p('## Exact posterior given a receiver bank vs the full window (same geometry, same window)')
    p()
    p('| Window | Receivers used | s | Certified volume | Cells | Mass in truth cell | TV bound |')
    p('|---|---|---|---|---|---|---|')
    for f in banks:
        r = json.load(open(f))
        p(f"| {r['scene']}/{r['window']} | {r['bank']} | {r['seconds']:.0f} | {r['L']:.3e} | {r['cells_certified']} | {r['bank_mass_in_truth_cell']:.3f} | {r['tv_bound']:.3f} |")
    p()

# sharing
if os.path.exists('outputs/overlap_val3.json'):
    p('## Scene-level sharing opportunity (3 validation scenes, 256 windows each)')
    p()
    p('| Scene | Boundary voxels per window (64 / 128 / 256) | Boundary-voxel multiplicity | Share of a window\'s boundary voxels seen in another window (64 / 128 / 256) |')
    p('|---|---|---|---|')
    for r in json.load(open('outputs/overlap_val3.json')):
        b = r['mean_boundary_voxels_by_side']; s = r['mean_shared_fraction_by_side']
        p(f"| {r['scene']} | {b['64']:.0f} / {b['128']:.0f} / {b['256']:.0f} | {r['boundary_multiplicity']:.2f} | {s['64']:.2f} / {s['128']:.2f} / {s['256']:.2f} |")
    p()

if os.path.exists('outputs/minority_distribution_val.json'):
    m = json.load(open('outputs/minority_distribution_val.json'))
    p(f"## Minority-class size of v0.1 validation windows ({m['windows']} windows, {m['val_scenes']} scenes)")
    p()
    p('| Side | minority <= 16 | <= 64 | <= 256 | <= 1024 | <= 4096 | median minority |')
    p('|---|---|---|---|---|---|---|')
    for s in ('64', '128', '256'):
        r = m[s]
        p(f"| {s} | {r['minority<=16']:.3f} | {r['minority<=64']:.3f} | {r['minority<=256']:.3f} | {r['minority<=1024']:.3f} | {r['minority<=4096']:.3f} | {r['median_minority']:.0f} |")
    p()

open(OUT, 'w').write('\n'.join(L))
print('\n'.join(L))
