"""Exact (rational) comparison of FastLabel v3.4 labels with independent references (FASTLABEL_V34_CERTIFICATION.md §7).

A FastLabel label is a vector of float64 per-cell volumes b_c (statistical labels: b_c = vol(B ∩ c); deterministic
labels: per-cell lower bounds). The label is its EXACT normalisation q_c = b_c / sum b (every float64 is a rational).

A reference gives rationals lo_c <= Vol(F ∩ c) <= hi_c + p_c with sum p_c <= pool (pool mass may sit in any cell).
Let Lt = sum lo, qt = lo / Lt and t_ref = the exact greedy bound sup_v TV(qt, v / |v|) over the enclosure
(the tvb.tv_bound_exact algorithm). For every q_F consistent with the reference:
    TV(q, q_F) <= TV(q, qt) + t_ref                       (worst case over the reference enclosure; rigorous)
    TV(q, q_F) >= TV(q, qt) - t_ref.
Containment check (F ⊆ B is necessary for the statistical certificate): for every cell, lo_c <= b_c must hold up to the
float error of b_c. excess = sum_c (lo_c - b_c)^+ is a lower bound of vol(F minus B) (modulo that error).
Pre-projection TV: if F ⊆ B, TV(U_B, U_F) = 1 - |F|/|B|, enclosed by [1 - (sum hi + pool)/|B|, 1 - Lt/|B|].

References:
  tp_engine : cmp_v3/labels/{key}_tp.npz, tau-pencil engine records (same float engine as FastLabel's B; NOT
              independent of assumption A1, but uses ALL receivers and no outer-bound filter);
  tp_cert   : xlf/ref/labels and perf/{key}_tp_label.npz, tau-pencil labels re-proved by the exact certify verifier;
  astra     : Astra's exact rational inner pieces (grids_*/{key}.json.gz: weights / 2^scale_bits are proved inner
              volumes, outer_upper / 2^scale_bits a proved upper bound of |F|).

usage: python compare_refs.py OUT.jsonl  (paths below; read-only inputs)"""
import glob
import gzip
import json
import os
import sys
from fractions import Fraction as Q

import numpy as np

WORK = "/home/yushu/Projects/RIND/RIND3D_100K_v3_work"
ASTRA = ("/home/yushu/Projects/RIND/G2-02-astra-inverse/Phase4_Multi_Source_3D/RIND3D_Posterior/experiments/"
         "global_inverse/records")
EPS = Q(1, 20)


def per_cell(cell, lo, hi):
    d = {}
    for c, a, b in zip(np.asarray(cell, np.int64).tolist(), np.asarray(lo, float).tolist(), np.asarray(hi, float).tolist()):
        x = d.setdefault(c, [Q(0), Q(0)]); x[0] += Q(a); x[1] += Q(b)
    return d


def tv_bound_exact(lo, hi, pool):
    """Exact greedy sup of TV(lo/Lt, v/|v|) over lo <= v <= hi (+ pool anywhere); same algorithm as certify.tvb."""
    Lt = sum(lo.values(), Q(0))
    if Lt <= 0:
        return None, Lt
    items = [(1 - lo[c] / Lt, hi[c] - lo[c]) for c in lo if hi[c] > lo[c]]
    if pool > 0:
        items.append((Q(1), pool))
    items.sort(key=lambda t: -t[0])
    num, den, best = Q(0), Lt, Q(0)
    for w, x in items:
        if w <= best:
            break
        num += x * w; den += x
        best = max(best, num / den)
    return min(best, Q(1)), Lt


def compare(fl, ref, pool):
    """fl: {cell: b_c}; ref: {cell: (lo, hi)}."""
    VB = sum(fl.values(), Q(0))
    lo = {c: v[0] for c, v in ref.items()}; hi = {c: v[1] for c, v in ref.items()}
    t_ref, Lt = tv_bound_exact(lo, hi, pool)
    Ht = sum(hi.values(), Q(0)) + pool
    q = {c: v / VB for c, v in fl.items()}
    qt = {c: v / Lt for c, v in lo.items() if v > 0}
    cells = set(q) | set(qt)
    d = sum((abs(q.get(c, Q(0)) - qt.get(c, Q(0))) for c in cells), Q(0)) / 2
    exc = [(c, lo[c] - fl.get(c, Q(0))) for c in lo if lo[c] > fl.get(c, Q(0))]
    exc_tot = sum((e for _, e in exc), Q(0))
    rel = [e / lo[c] for c, e in exc]
    out_mass = sum((q[c] for c in q if hi.get(c, Q(0)) == 0), Q(0))      # FL mass in cells with no F mass (pool aside)
    miss = sum((qt[c] for c in qt if c not in q), Q(0))                       # reference mass in cells absent from FL
    # per-cell ratio q / qt on the reference support (weighted by qt)
    r = np.array([float(q.get(c, Q(0)) / qt[c]) for c in qt]); w = np.array([float(qt[c]) for c in qt])
    o = np.argsort(r); cw = np.cumsum(w[o])
    wq = lambda a: float(r[o][min(np.searchsorted(cw, a * cw[-1]), len(r) - 1)])
    return dict(n_fl=len(q), n_ref=len(qt), tv_q_qt=float(d), t_ref=float(t_ref),
                tv_upper=float(d + t_ref), tv_lower=float(max(d - t_ref, Q(0))), exceeds_eps=bool(d - t_ref > EPS),
                ok_eps=bool(d + t_ref <= EPS),
                one_minus_p=[float(1 - Ht / VB), float(1 - Lt / VB)], vol_B=float(VB), vol_F=[float(Lt), float(Ht)],
                containment_excess_cells=len(exc), containment_excess_rel_total=float(exc_tot / Lt),
                containment_excess_rel_max_cell=float(max(rel)) if rel else 0.0,
                mass_outside_ref_support=float(out_mass), ref_mass_missing_in_fl=float(miss),
                ratio_q_over_qF=dict(wmean=float(np.sum(r * w) / np.sum(w)), wq05=wq(0.05), wq50=wq(0.5), wq95=wq(0.95),
                                     frac_cells_ratio_gt_1=float(np.mean(r > 1 + 1e-9))))


def load_fl(path):
    with np.load(path) as z:
        d = per_cell(z["cell"], z["lo"], z["hi"])
    return {c: v[0] for c, v in d.items()}


def refs_for(key):
    out = []
    p = f"{WORK}/cmp_v3/labels/{key}_tp.npz"
    if os.path.exists(p):
        with np.load(p) as z:
            out.append(("tp_engine", per_cell(z["cell"], z["lo"], z["hi"]), Q(0)))
    for p in (f"{WORK}/xlf/ref/labels/{key}_tp_label.npz", f"{WORK}/perf/{key}_tp_label.npz"):
        if os.path.exists(p):
            with np.load(p) as z:
                assert json.loads(z["meta"].tobytes())["status"] == "certified"
                out.append(("tp_cert", per_cell(z["cells"], z["lo"], z["hi"]), Q(float(z["pool"]))))
    for p in sorted(glob.glob(f"{ASTRA}/grids*/{key}.json.gz")):
        out.append(("astra:" + (p.split("/")[-2][6:] or "heldout"),) + astra_ref(p))
    return out


def astra_ref(p):
    g = json.load(gzip.open(p, "rt"))
    s = Q(1, 2 ** int(g["scale_bits"]))
    ref = {int(c): (Q(int(m)) * s, Q(int(m)) * s) for c, m in zip(g["cells"], g["weights"])}
    tot = int(g["denominator"]); outer = int(g["tv_denominator"])          # tv = (outer_upper - total) / outer_upper
    assert int(g["tv_numerator"]) == outer - tot and tot == sum(int(m) for m in g["weights"])
    return ref, Q(outer - tot) * s


def main(out_path, fl_dirs):
    keys = sorted({os.path.basename(p)[:13] for d in fl_dirs for p in glob.glob(f"{d}/*_fl.npz")})
    rows = []
    with open(out_path, "w") as f:
        for key in keys:
            fp = next(f"{d}/{key}_fl.npz" for d in fl_dirs if os.path.exists(f"{d}/{key}_fl.npz"))
            fl = load_fl(fp)
            for name, ref, pool in refs_for(key):
                rec = dict(key=key, ref=name, fl_file=fp, **compare(fl, ref, pool))
                rows.append(rec); f.write(json.dumps(rec) + "\n"); f.flush()
                print(f"{key} {name:22s} n_fl={rec['n_fl']:6d} n_ref={rec['n_ref']:6d} TV<= {rec['tv_upper']:.3e} "
                      f"(TV(q,qt)={rec['tv_q_qt']:.3e}, t_ref={rec['t_ref']:.1e}) 1-p in [{rec['one_minus_p'][0]:.2e},"
                      f"{rec['one_minus_p'][1]:.2e}] excess_rel={rec['containment_excess_rel_total']:.1e} "
                      f"(max cell {rec['containment_excess_rel_max_cell']:.1e})", flush=True)
    return rows


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or [f"{WORK}/cmp_v3/labels"])
