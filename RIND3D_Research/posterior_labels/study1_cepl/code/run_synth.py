"""Engine vs brute-force reference on the controlled cases.  Brute force tests every window receiver for every
sample point (rpl.synth.feasible_points): uniform points over the whole cube (completeness) and stratified points
in every fine cell the engine reports as possibly feasible, dilated by one cell (accuracy)."""
import sys, time, json, numpy as np
sys.path.insert(0, 'code')
from synth_cases import case
import importlib, os
run_window = importlib.import_module(os.environ.get('RPL_ENGINE', 'rpl.engine3')).run_window
from rpl.synth import feasible_points, window_receivers

def cell_lo(cid):
    s = 128; ix, rem = divmod(cid, s * s); iy, iz = divmod(rem, s)
    return np.array([ix, iy, iz]) / s

def main(name, n_uniform=2_000_000, per_cell=4000, seed=0, kw=None):
    rng = np.random.default_rng(seed)
    c = case(name); geom = c['geom']; o = c['origin']; L = c['L']
    t = time.time(); res = run_window(geom, c['cls'], o, **(kw or {})); dt = time.time() - t
    cells, inv = np.unique(res['cell'], return_inverse=True)
    lo = np.bincount(inv, res['lo']); hi = np.bincount(inv, res['hi'])
    Lm = lo.sum(); U = (hi - lo).sum()
    X, obs = window_receivers(c['cls'], o)
    wlo = np.asarray(o) / 1024; whi = (np.asarray(o) + L) / 1024
    # truth must be feasible
    tr = feasible_points(c['truth'][None], X, obs, geom[0], geom[1], wlo, whi)[0]
    # completeness: uniform points
    S = rng.random((n_uniform, 3))
    f = feasible_points(S, X, obs, geom[0], geom[1], wlo, whi)
    fe = S[f == 1]
    cid = (np.floor(fe * 128).astype(int) @ np.array([128 * 128, 128, 1]))
    pos = set(cells[hi > 0].tolist())
    missed = int(sum(1 for q in cid if q not in pos))
    # accuracy: stratified in dilated positive cells
    poscells = cells[hi > 0]
    nb = set()
    for q in poscells:
        b = (cell_lo(q) * 128).astype(int)
        for d in np.ndindex(3, 3, 3):
            e = b + np.array(d) - 1
            if np.all(e >= 0) and np.all(e < 128):
                nb.add(int(e @ np.array([128 * 128, 128, 1])))
    nb = sorted(nb)
    est = np.zeros(len(nb)); und = 0
    for i, q in enumerate(nb):
        P0 = cell_lo(q) + rng.random((per_cell, 3)) / 128
        ff = feasible_points(P0, X, obs, geom[0], geom[1], wlo, whi)
        und += int((ff < 0).sum())
        est[i] = (ff == 1).mean() / 128 ** 3
    eng = np.zeros(len(nb))
    m = {int(q): k for k, q in enumerate(cells)}
    for i, q in enumerate(nb):
        if q in m:
            eng[i] = lo[m[q]]
    tv = 0.5 * np.abs(eng / eng.sum() - est / est.sum()).sum() if eng.sum() > 0 and est.sum() > 0 else None
    # MC noise floor of the same estimator: TV between two independent half-samples is ~ sqrt(2) x noise
    p = est * 128 ** 3
    sd = np.sqrt(np.maximum(p * (1 - p), 1e-12) / per_cell) / 128 ** 3
    noise_tv = 0.5 * np.sum(sd) / est.sum() * np.sqrt(2 / np.pi) if est.sum() > 0 else None
    out = dict(case=name, origin=list(o), L=L, engine_seconds=dt, stats=res['stats'], L_engine=Lm, U_engine=U,
               tv_bound=U / (Lm + U) if Lm + U > 0 else None, cells_positive=int((lo > 0).sum()),
               truth_feasible=int(tr), uniform_points=n_uniform, uniform_feasible=int(len(fe)),
               uniform_volume_estimate=len(fe) / n_uniform, uniform_feasible_outside_engine_cells=missed,
               dilated_cells=len(nb), per_cell=per_cell, undecided_points=und,
               mc_volume=float(est.sum()), engine_volume=float(eng.sum()),
               tv_engine_vs_mc=tv, mc_expected_tv_noise=noise_tv)
    print(json.dumps(out))
    return out

if __name__ == '__main__':
    names = sys.argv[1:] or ['T1_edge_slab', 'T2_split_slab', 'T3_corner_tube']
    kw = json.loads(os.environ.get('RPL_KW', '{}'))
    allr = [main(n, kw=kw) for n in names]
    json.dump(allr, open(os.environ.get('RPL_OUT', 'outputs/synth_results.json'), 'w'), indent=1)
