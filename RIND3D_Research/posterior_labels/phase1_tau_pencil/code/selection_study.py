"""Selection-law sensitivity on real windows (Deliverable A).  For a window W with exact posterior q_W = Unif(F_W):
draw K sources from q_W (exact: polytope chosen by volume, uniform inside), render each source's full 1024^3 field,
and estimate with common random origins (CRN) the two global frequencies that drive the v0.1 selection factor
(tpl/selection.py):  P_valid(sigma, s) and P_b(sigma, s) for W's side sigma and W's lit-ratio bin b.
Reported: ratios to the true source's values, and the implied model estimate of eta (max/min - 1 of the model
lambda over the sample).  Model-based and sample-based: an estimate, not a bound."""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from rpl.data import Scene, scene_ids
from tpl.geom import Geom
from tpl.window import Window
from tpl.run_b3 import run_window
from tpl.verify import polytopes, sample_in_hull
from tpl.selection import Field, valid_and_bin


def study(pref, wid, K=32, M=16384, seed=7):
    sid = [s for s in scene_ids() if s.startswith(pref)][0]
    sc = Scene(sid)
    geom = Geom.from_scene(sc)
    cls, o, L = sc.window_classes(wid)
    w = Window(cls, o, 1024)
    r = run_window(geom, w, d1=4)
    polys = polytopes(r, 0)
    vols = np.array([p[2] for p in polys])
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(polys), K, p=vols / vols.sum())
    S = np.array([sample_in_hull(polys[i][1], 1, rng)[0] for i in pick])
    lit = int((cls == 1).sum()); free = int((cls != 2).sum())
    b = min(9, (10 * lit) // free)
    org = np.random.default_rng(seed + 1).integers(0, 1024 - L + 1, size=(M, 3))
    rows = []
    for k, s in enumerate(np.vstack([sc.truth, S])):
        t = time.perf_counter()
        F = Field(geom, s)
        cnt = F.counts(org, L)
        valid, bins = valid_and_bin(cnt, org, L, s)
        # the window W itself must be reproduced by every s in F_W (consistency check of the renderer)
        cW = F.counts(np.array([o]), L)[0]
        rows.append(dict(source=s.tolist(), is_truth=k == 0, valid=valid.astype(np.int8), inbin=(valid & (bins == b)).astype(np.int8),
                         W_counts=cW.tolist(), seconds=time.perf_counter() - t))
    Vt = rows[0]['valid'].astype(float); Bt = rows[0]['inbin'].astype(float)
    out = []
    for rr in rows:
        v = rr.pop('valid').astype(float); bb = rr.pop('inbin').astype(float)
        rr['P_valid'] = v.mean(); rr['P_b'] = bb.mean()
        dv = v - Vt; db = bb - Bt
        rr['ratio_valid'] = v.mean() / Vt.mean(); rr['ratio_b'] = bb.mean() / max(Bt.mean(), 1e-12)
        rr['se_ratio_valid'] = dv.std() / np.sqrt(M) / Vt.mean(); rr['se_ratio_b'] = db.std() / np.sqrt(M) / max(Bt.mean(), 1e-12)
        out.append(rr)
    rv = np.array([x['ratio_valid'] for x in out]); rb = np.array([x['ratio_b'] for x in out])
    # TV(Unif(F_W), lambda-reweighted) estimated from the K posterior samples (truth excluded), for the two pure
    # model components: lambda ~ 1/P_valid (natural picks) and lambda ~ 1/P_b (bin-balanced picks)
    def tv_reweight(ratio):
        w = 1.0 / np.maximum(ratio[1:], 1e-12)
        return float(0.5 * np.mean(np.abs(w / w.mean() - 1.0)))
    # model lambda ~ k_nat / P_valid + k_b / P_b; eta bracket over the two extreme mixtures
    eta_valid = rv.max() / rv.min() - 1; eta_b = rb.max() / rb.min() - 1
    allv = np.concatenate([p[1] for p in polys])
    res = dict(scene=sid, window=int(wid), side=int(L), bin=int(b), lit=lit, free=free, K=K, M=M,
               n_cells=int(r['n_cells_certified']), volume=r['L_total'],
               support_extent=(allv.max(0) - allv.min(0)).tolist(),
               W_reproduced=[x['W_counts'] == [lit, free - lit, int((cls == 2).sum())] for x in out],
               eta_model_valid_only=float(eta_valid), eta_model_bin_only=float(eta_b),
               eta_model=float(max(eta_valid, eta_b)), tv_bias_bound_model=float(max(eta_valid, eta_b) / 2),
               tv_reweight_valid=tv_reweight(rv), tv_reweight_bin=tv_reweight(rb), sources=out)
    return res


if __name__ == '__main__':
    tasks = json.loads(sys.argv[1])
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    for pref, wid in tasks:
        f = os.path.join(out, f'{pref[:8]}_{wid}.json')
        if os.path.exists(f):
            continue
        t = time.perf_counter()
        res = study(pref, wid)
        res['seconds'] = time.perf_counter() - t
        json.dump(res, open(f, 'w'), indent=1)
        print(pref[:8], wid, 'side', res['side'], 'cells', res['n_cells'], 'eta_model %.4f (valid %.4f, bin %.4f)' % (
            res['eta_model'], res['eta_model_valid_only'], res['eta_model_bin_only']),
            'TV_reweight valid %.4f bin %.4f' % (res['tv_reweight_valid'], res['tv_reweight_bin']), 'W reproduced', all(res['W_reproduced']),
            '%.0fs' % res['seconds'], flush=True)
