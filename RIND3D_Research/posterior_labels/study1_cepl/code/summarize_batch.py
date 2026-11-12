import sys, json, glob, numpy as np
d = sys.argv[1]
rows = [json.load(open(f)) for f in sorted(glob.glob(f'{d}/*.json')) if not f.endswith('SAMPLE.json') and not f.endswith('.err.json')]
errs = glob.glob(f'{d}/*.err.json')
print('windows', len(rows), 'errors', len(errs))
if not rows: sys.exit()
comp = [r for r in rows if r['complete']]
def q(a, ps=(0, 25, 50, 75, 100)):
    a = np.asarray(a, float); return [round(float(np.percentile(a, p)), 4) for p in ps] if len(a) else []
print('complete', len(comp), '/', len(rows))
print('seconds all  (min,q1,med,q3,max)', q([r['seconds'] for r in rows]))
print('seconds complete', q([r['seconds'] for r in comp]))
print('nodes complete', q([r['stats']['nodes'] for r in comp]))
tvb = [r['tv_bound'] for r in comp if r['tv_bound'] is not None]
print('tv_bound complete', q(tvb), 'frac<=0.01', np.mean(np.array(tvb) <= 0.01) if tvb else None, 'frac<=0.05', np.mean(np.array(tvb) <= 0.05) if tvb else None)
print('feasible volume L', q([r['L'] for r in comp]))
print('n fine cells', q([r['n_fine_cells'] for r in comp]), 'frac one cell', np.mean([r['n_fine_cells'] == 1 for r in comp]))
print('n coarse cells', q([r['n_coarse_cells'] for r in comp]))
print('truth fine mass', q([r['truth_fine_mass'] for r in comp]), 'truth coarse mass', q([r['truth_coarse_mass'] for r in comp]))
print('fine entropy', q([r['fine_entropy'] for r in comp]))
for r in rows:
    print(r['scene'][:8], r['window'], r['side'], round(r['seconds'], 1), r['complete'], r['stats']['nodes'],
          '%.2e' % r['L'], None if r['tv_bound'] is None else round(r['tv_bound'], 4), r['n_fine_cells'], r['truth_fine_mass'] and round(r['truth_fine_mass'], 3), r['components'], r['lit'], r['dark'])
