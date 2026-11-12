"""Generate docs/RESULTS_TABLES.md from saved JSON evidence (no number typed by hand)."""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
L = []


def p(s=''):
    L.append(s)


def jl(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        b = os.path.basename(f)
        if b.startswith('SAMPLE') or b in ('SUMMARY.json', 'ARGS.json', 'INDEX.json'):
            continue
        out.append(json.load(open(f)))
    return out


def fmt_s(x):
    return f'{x:.2f}' if x < 10 else f'{x:.0f}'


# ------------------------------------------------------------------ controlled cases
if os.path.exists('outputs/SYNTH_B3.json'):
    S = json.load(open('outputs/SYNTH_B3.json'))
    loc = json.load(open('outputs/VERIFY_LOCAL_SYNTH.json')) if os.path.exists('outputs/VERIFY_LOCAL_SYNTH.json') else {}
    p('## 1. Controlled cases at 1024³ (engine b3), brute-force verified')
    p()
    p('| Case | CPU s | Exact volume | Cells | TV bound | MC volume ± SE (box) | Sound pts / infeasible | Box-local feasible / outside | Scaled ×2: inside / infeasible, outside / feasible | First study: s, TV bound |')
    p('|---|---|---|---|---|---|---|---|---|---|')
    for r in S:
        v = r['verify']; lv = loc.get(r['case'], {})
        fs = r.get('first_study') or {}
        p(f"| {r['case']} | {r['cpu_seconds']:.2f} | {r['L']:.6e} | {r['cells']} | {r['tv_bound']:.3g} | "
          f"{v['mc_volume']:.3e} ± {v['mc_volume_se']:.1e} | {v['sound_points']} / {v['sound_infeasible']} | "
          f"{lv.get('local_feasible', '–')} / {lv.get('local_feasible_outside', '–')} | "
          f"{lv.get('inside', '–')} / {lv.get('inside_infeasible', '–')}, {lv.get('outside', '–')} / {lv.get('outside_feasible', '–')} | "
          f"{fs.get('seconds', '–')} s, {fs.get('tv', '–')} |")
    p()

# ------------------------------------------------------------------ lemma tests
if os.path.exists('outputs/TEST_TAU_LEMMAS.json'):
    T = json.load(open('outputs/TEST_TAU_LEMMAS.json'))
    p('## 2. Pencil lemmas vs the segment predicate (tests/test_tau_lemmas.py; train scenes only)')
    p()
    p('| Geometry | Random (s, x) pairs | dark | Lemma 2 (source-side) mismatches | Lemma 2W (window-side) mismatches |')
    p('|---|---|---|---|---|')
    for k, v in T.items():
        p(f"| {k} | {v['pairs']} | {v['dark']} | {v['mismatch_S']} | {v['mismatch_W']} |")
    p()

# ------------------------------------------------------------------ stage A
A = jl('outputs/stageA_tpe120/*.json')
if A:
    p('## 3. Stage A: 128³ physical resolution, tau-pencil engine on all 120 windows (validation scenes)')
    p()
    p('| Window side (voxels at 128³) | Windows | TV bound = 0 (exact) | TV ≤ 0.05 | CPU s median / p90 / max | Cells with mass: median / max | Truth-cell mass: median |')
    p('|---|---|---|---|---|---|---|')
    for side in (8, 16, 32):
        rr = [r for r in A if r['side'] == side]
        t = np.array([r['results']['tpe']['cpu_seconds'] for r in rr]); tv = np.array([r['results']['tpe']['tv_bound'] for r in rr])
        ce = np.array([r['results']['tpe']['cells'] for r in rr]); tm = np.array([r['results']['tpe']['truth_cell_mass'] for r in rr])
        p(f"| {side} | {len(rr)} | {(tv == 0).sum()} | {(tv <= 0.05).sum()} | {np.median(t):.2f} / {np.percentile(t, 90):.2f} / {t.max():.2f} | "
          f"{int(np.median(ce))} / {ce.max()} | {np.median(tm):.3f} |")
    p()
C = jl('outputs/stageA_compare30/*.json')
if C:
    p(f'## 4. Stage A method comparison ({len(C)} windows: 10 per side, declared subset)')
    p()
    p('Guarantee: deterministic for TPE and CEPL; statistical (≥ 99%, per window) for the two Monte Carlo methods.')
    p()
    p('| Method | Budget | Windows | TV ≤ 0.05 | TV ≤ 0.10 | Median reported TV bound | CPU s median / max | Median TV to the exact label (when TPE exact) |')
    p('|---|---|---|---|---|---|---|---|')
    keys = sorted({k for r in C for k in r['results']}, key=lambda k: (k.split('_')[0], len(k), k))
    for k in keys:
        rr = [r['results'][k] for r in C if k in r['results']]
        tv = np.array([x['tv_bound'] for x in rr]); t = np.array([x['cpu_seconds'] for x in rr])
        te = [x['tv_vs_exact'] for x in rr if 'tv_vs_exact' in x]
        meth = k.rsplit('_', 1)[0] if k != 'tpe' else 'tpe'
        bud = k.rsplit('_', 1)[1] if k != 'tpe' else 'complete'
        p(f"| {meth} | {bud} | {len(rr)} | {(tv <= 0.05).sum()} | {(tv <= 0.10).sum()} | {np.median(tv):.3f} | {np.median(t):.2f} / {t.max():.1f} | "
          f"{(f'{np.median(te):.3f} (n={len(te)})') if te else '–'} |")
    p()


# ------------------------------------------------------------------ stage B
def real_table(d, title):
    R = jl(os.path.join(d, '*.json'))
    if not R:
        return R
    p(f'## {title}')
    p()
    p('| Side | Windows | TV bound = 0 | TV ≤ 0.05 | TV ≤ 0.10 | CPU s median / mean / p90 / max | ≤ 1 s | ≤ 5 s | ≤ 60 s | Cells median / max | Truth-cell mass median |')
    p('|---|---|---|---|---|---|---|---|---|---|---|')
    for side in sorted({r['side'] for r in R}) + ['all']:
        rr = [r for r in R if side == 'all' or r['side'] == side]
        c = np.array([r['cpu_seconds'] for r in rr]); tv = np.array([r['tv_bound'] for r in rr])
        ce = np.array([r['n_cells_certified'] for r in rr])
        tm = np.array([r['truth_cell_L'] / r['L_total'] for r in rr if r['L_total'] > 0])
        p(f"| {side} | {len(rr)} | {(tv == 0).sum()} | {(tv <= 0.05).sum()} | {(tv <= 0.10).sum()} | "
          f"{fmt_s(np.median(c))} / {fmt_s(c.mean())} / {fmt_s(np.percentile(c, 90))} / {fmt_s(c.max())} | "
          f"{(c <= 1).sum()} | {(c <= 5).sum()} | {(c <= 60).sum()} | {int(np.median(ce))} / {ce.max()} | {np.median(tm):.4f} |")
    p()
    vs = [r['verify'] for r in R if 'verify' in r]
    if vs:
        p(f"Brute-force verification over these windows: {sum(v.get('sound_points', 0) for v in vs)} points sampled inside "
          f"emitted IN polytopes, {sum(v.get('sound_infeasible', 0) for v in vs)} infeasible; "
          f"{sum(v.get('box_feasible', 0) for v in vs) + sum(v.get('world_feasible', 0) for v in vs)} feasible points found by "
          f"uniform sampling (support box and world), {sum(v.get('box_feasible_outside_polytopes', 0) for v in vs) + sum(v.get('world_feasible_outside', 0) for v in vs)} outside the support.")
        p()
    p('<details><summary>Per-window rows</summary>')
    p()
    p('| Window | Side | Comps | Lit / dark FREE | CPU s | Peak RSS MB | TV bound | Volume | Cells | Truth-cell mass |')
    p('|---|---|---|---|---|---|---|---|---|---|')
    for r in sorted(R, key=lambda r: (r['side'], r['cpu_seconds'])):
        p(f"| {r['scene'][:8]}/{r['window']} | {r['side']} | {r['components']} | {r['counts']['lit']} / {r['counts']['dark']} | "
          f"{fmt_s(r['cpu_seconds'])} | {('%.0f' % r['maxrss_mb']) if r.get('maxrss_mb') else '–'} | {r['tv_bound']:.3g} | {r['L_total']:.3e} | {r['n_cells_certified']} | "
          f"{(r['truth_cell_L'] / r['L_total']) if r['L_total'] > 0 else 0:.4f} |")
    p()
    p('</details>')
    p()
    return R


real_table('outputs/real_val_random90_b3', '5. Stage B: 1024³, declared random validation sample (30 scenes × sides 64/128/256), engine b3')
real_table('outputs/real_val64_prev24_b3', '6. Stage B: the first study\'s 24 validation side-64 windows, engine b3')
real_table('outputs/real_val64_prev24', '7. Stage B: the first study\'s 24 windows, engine v1 (19 finished; stopped in verification)')
real_table('outputs/real_b4_reruns', '7b. Engine b4 reruns of the random-sample windows that overflowed the b3 work queue (deferred expansion)')
real_table('outputs/real_paired30_b3', '8. Stage B: the 30 Stage-A windows at their original 1024³ resolution (transfer gap), engine b3')

# ------------------------------------------------------------------ paired transfer gap
if glob.glob('outputs/real_paired30_b3/*.json') and C:
    P = {f"{r['scene'][:8]}_{r['window']}": r for r in jl('outputs/real_paired30_b3/*.json')}
    p('## 9. Transfer gap: the same physical windows at 128³ and 1024³')
    p()
    p('| Window | Side 128³ / 1024³ | CPU s 128³ | CPU s 1024³ | Ratio | Cells 128³ | Cells 1024³ | Exact at both |')
    p('|---|---|---|---|---|---|---|---|')
    ratios = []
    for r in C:
        k = r['name']
        if k not in P:
            continue
        a = r['results']['tpe']; b = P[k]
        ratio = b['cpu_seconds'] / max(a['cpu_seconds'], 1e-3)
        ratios.append(ratio)
        p(f"| {k} | {r['side']} / {b['side']} | {a['cpu_seconds']:.2f} | {fmt_s(b['cpu_seconds'])} | {ratio:.0f} | {a['cells']} | {b['n_cells_certified']} | "
          f"{'yes' if a['tv_bound'] == 0 and b['tv_bound'] == 0 else 'no'} |")
    if ratios:
        p()
        p(f'Median CPU ratio 1024³ / 128³: {np.median(ratios):.0f} (range {min(ratios):.0f}–{max(ratios):.0f}).')
    p()

# ------------------------------------------------------------------ selection law
if os.path.exists('outputs/SELECTION_SUMMARY.json'):
    Ssel = json.load(open('outputs/SELECTION_SUMMARY.json'))
    p('## 10. v0.1 selection factor: model-estimated TV(q_nat, q_v0.1) over each exact posterior (32 exact posterior samples)')
    p()
    p('Model [E]: λ_W ∝ k_nat/P_valid + k_b/P_b. Columns give the two pure components. Not a certificate. The noise floor is the TV that the CRN sampling error of the ratios alone would produce.')
    p()
    p('| Window | Cells | Support extent (world units) | TV, natural component (noise) | P_valid ratio range | TV, bin component (noise) | P_b ratio range | Window reproduced by all samples |')
    p('|---|---|---|---|---|---|---|---|')
    for o in Ssel:
        p(f"| {o['window']} | {o['cells']} | {o['extent']} | {o['tv_valid']:.3f} ({o['noise_floor_valid']:.3f}) | {o['ratio_range_valid'][0]:.2f}–{o['ratio_range_valid'][1]:.2f} | "
          f"{o['tv_bin']:.3f} ({o['noise_floor_bin']:.3f}) | {o['ratio_range_bin'][0]:.2f}–{o['ratio_range_bin'][1]:.2f} | {o['W_reproduced_all']} |")
    p()

# ------------------------------------------------------------------ labels / storage / cost
if os.path.exists('outputs/LABEL_STORAGE.json'):
    st = json.load(open('outputs/LABEL_STORAGE.json'))
    p('## 11. Label storage (rind3d-posterior-labels-v2: delta-varint cells + float16 masses, zlib)')
    p()
    p('| Labels | Cells median / mean / max | Bytes median / mean / max | Max storage TV | Truncated to TV 0.01: bytes median |')
    p('|---|---|---|---|---|')
    p(f"| {st['n']} | {st['cells_median']:.0f} / {st['cells_mean']:.0f} / {st['cells_max']} | {st['bytes_median']:.0f} / {st['bytes_mean']:.0f} / {st['bytes_max']} | "
      f"{st['storage_tv_max']:.1e} | {st['bytes99_median']:.0f} |")
    p()
if os.path.exists('outputs/COST_MODEL.json'):
    cm = json.load(open('outputs/COST_MODEL.json'))
    p('## 12. Production cost projection (measured per-window CPU s; outputs/COST_MODEL.json)')
    p()
    for k, v in cm.items():
        p(f"**{k}** ({v['sample']}): mean CPU per window at the v0.1 side mix = {v['cpu_seconds_per_window_mix']:.1f} s "
          f"(gap to the 1 s target: {v['25.6M windows']['gap_vs_1s_target']:.0f}×; to the 5 s ceiling: {v['25.6M windows']['gap_vs_5s_ceiling']:.1f}×).")
        p()
        p('| Side | Windows | TV = 0 | CPU s mean / median / p90 / max | Peak RSS MB median |')
        p('|---|---|---|---|---|')
        for side, s in v['per_side'].items():
            p(f"| {side} | {s['n']} | {s['certified_tv0']} | {fmt_s(s['cpu_mean'])} / {fmt_s(s['cpu_median'])} / {fmt_s(s['cpu_p90'])} / {fmt_s(s['cpu_max'])} | {s['maxrss_mb_median'] or 0:.0f} |")
        p()
        p(f"Total: 25.6M windows → {v['25.6M windows']['core_hours']:.3g} CPU-core-hours ({v['25.6M windows']['core_years']:.3g} core-years); "
          f"10.16M windows → {v['10.16M windows']['core_hours']:.3g} core-hours.")
        p()


# ------------------------------------------------------------------ boundary-local verification on real windows
if os.path.exists('outputs/VERIFY_LOCAL_REAL12.json'):
    V = json.load(open('outputs/VERIFY_LOCAL_REAL12.json'))
    p('## 13. Boundary-local brute-force verification on real 1024³ windows (declared subset: first 4 scenes of the random sample)')
    p()
    p('Scaled mode: points uniform in sampled IN polytopes scaled ×2 about their centroid (inside points test soundness, '
      'outside points test completeness right at the boundary). Box mode (side 64 only): points in the polytope bounding box grown 50%. '
      'Every FREE receiver of the window is tested by brute force.')
    p()
    p('| Window | Side | Scaled: points | inside / infeasible | outside / feasible | Box: points | feasible / outside support |')
    p('|---|---|---|---|---|---|---|')
    for v in V:
        p(f"| {v['window']} | {v['side']} | {v.get('scaled_points', 0)} | {v.get('inside', 0)} / {v.get('inside_infeasible', 0)} | "
          f"{v.get('outside', 0)} / {v.get('outside_feasible', 0)} | {v.get('local_points', '–')} | {v.get('local_feasible', '–')} / {v.get('local_feasible_outside', '–')} |")
    p()
    p(f"Total (scaled): {sum(v.get('scaled_points', 0) for v in V)} points; {sum(v.get('inside', 0) for v in V)} inside the emitted support, "
      f"{sum(v.get('inside_infeasible', 0) for v in V)} of them infeasible; {sum(v.get('outside', 0) for v in V)} outside, "
      f"{sum(v.get('outside_feasible', 0) for v in V)} of them feasible.")
    p()
# ------------------------------------------------------------------ clearance shell
for f in sorted(glob.glob('outputs/SHELL_CHECK_*.json')):
    if 'conservative' in f:
        continue
    Sh = json.load(open(f))
    p(f'## 14. Clearance shell (2⁻²⁶) — exact upper bound per IN polytope ({os.path.basename(f)})')
    p()
    p('| Window | Polytopes | Touching the shell | Shell fraction of the volume | TV bound without / with the shell |')
    p('|---|---|---|---|---|')
    for r in Sh:
        p(f"| {r['window']} | {r['polytopes']} | {r['polytopes_touching_shell']} | {r['shell_fraction']:.2e} | {r['tv_bound_without_shell']:.2g} / {r['tv_bound_with_shell']:.2e} |")
    p()
    p(f"Maximum TV bound including the shell: {max(r['tv_bound_with_shell'] for r in Sh):.2e}.")
    p()

os.makedirs('docs', exist_ok=True)
open('docs/RESULTS_TABLES.md', 'w').write('# Results tables (generated by code/make_tables.py from outputs/)\n\n' + '\n'.join(L))
print('\n'.join(L))
