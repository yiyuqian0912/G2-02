"""E4: calibration / coverage audit of the 333 stored tau-pencil labels (scenes 000000-000003) against the TRUE sources.

For a correct label q of slot (G, W, j), the true source s_j satisfies, under the weighted 1K population and given
(G, W, r_j), cell(s_j) ~ q (SELECTION_LAW_1K Theorem). Hence, for every label individually:
  * cell(s_j) must lie in the label's upper support (hi > 0): a hard check (no tolerance beyond the clearance shell);
  * the randomised HPD-rank u = q(cells with q > q_true) + V * q(cells with q = q_true), V ~ U(0,1), is U(0,1).
This is a simulation-based-calibration style test (Talts et al. 2018) that needs no reference posterior. Selection of
WHICH labels exist (computed first, engine failures) depends on (G, W, r_j) only, so it does not bias this test.
Power check: the same statistics with the x and z axes of the source swapped (a plausible ZYX/XYZ bug).
Run: nice -n 19 python e4_label_coverage.py <release_root> > out/e4_label_coverage.json
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import kstest

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/yushu/Projects/RIND/RIND3D_1K_data/release_1k_v1")
G = 128
rng = np.random.default_rng(5)


def cell_of(s):
    i = np.minimum((np.asarray(s) * G).astype(np.int64), G - 1)
    return int((i[0] * G + i[1]) * G + i[2])


rows = []
for sc in ("000000", "000001", "000002", "000003"):
    d = ROOT / "scenes" / sc
    meta = json.loads((d / "scene.json").read_text())
    src = np.array([[float.fromhex(v) for v in s] for s in json.loads((d / "privileged" / "sources.json").read_text())["sources_hex"]])
    W = np.load(d / "public" / "windows.npy")
    lab = json.loads((d / "labels" / "LABELS.json").read_text())["labels"]
    for x in lab:
        if x.get("status") != "certified":
            continue
        w, j = int(x["window"]), int(x["source"])
        with np.load(d / "labels" / "tasks" / f"w{w:03d}_s{j}.npz") as z:
            cell, lo, hi, tvb = z["cell"].astype(np.int64), z["lo"], z["hi"], float(z["tv_bound"])
        cells, inv = np.unique(cell, return_inverse=True)
        Lc = np.bincount(inv, weights=lo, minlength=len(cells)); Uc = np.bincount(inv, weights=hi, minlength=len(cells))
        q = Lc / Lc.sum()
        out = {}
        for tag, s in (("true", src[j]), ("swap_xz", src[j][::-1]), ("other_source", src[(j + 1) % len(src)] if len(src) > 1 else None)):
            if s is None:
                continue
            c = cell_of(s); pos = np.searchsorted(cells, c)
            inU = pos < len(cells) and cells[pos] == c
            qt = q[pos] if inU else 0.0
            u = float(q[q > qt].sum() + rng.random() * q[q == qt].sum()) if inU else 1.0
            out[tag] = dict(in_upper_support=bool(inU and Uc[pos] > 0), in_lower_support=bool(inU and Lc[pos] > 0), u=u)
        r = W[w]
        rows.append(dict(scene=sc, window=w, source=j, Lp=int(r["Lp"]), m=int(r["m"]), tv_bound=tvb, n_cells=int(len(cells)),
                         omega=float(meta["Lambda_hat"][str(int(r["Lp"]))]),
                         entropy_bits=float(-(q[q > 0] * np.log2(q[q > 0])).sum()), **{k: v for k, v in out.items()}))

res = dict(n_labels=len(rows), by_scene={sc: sum(r["scene"] == sc for r in rows) for sc in ("000000", "000001", "000002", "000003")})
for tag in ("true", "swap_xz", "other_source"):
    rr = [r for r in rows if tag in r]
    u = np.array([r[tag]["u"] for r in rr]); om = np.array([r["omega"] for r in rr])
    res[tag] = dict(n=len(rr), share_in_upper_support=float(np.mean([r[tag]["in_upper_support"] for r in rr])),
                    share_in_lower_support=float(np.mean([r[tag]["in_lower_support"] for r in rr])),
                    hpd_u_ks_p=float(kstest(u, "uniform").pvalue), hpd_u_mean=float(u.mean()),
                    hpd_u_quintiles=np.histogram(u, bins=5, range=(0, 1))[0].tolist(),
                    weighted_hpd_u_mean=float((u * om).sum() / om.sum()),
                    coverage_at={str(a): float(np.mean(u <= a)) for a in (0.5, 0.8, 0.9, 0.95)})
res["tv_bound"] = dict(max=float(max(r["tv_bound"] for r in rows)), median=float(np.median([r["tv_bound"] for r in rows])))
res["n_cells"] = dict(median=float(np.median([r["n_cells"] for r in rows])), max=int(max(r["n_cells"] for r in rows)),
                      min=int(min(r["n_cells"] for r in rows)))
res["entropy_bits"] = dict(quantiles={str(qq): float(np.quantile([r["entropy_bits"] for r in rows], qq)) for qq in (0, .1, .5, .9, 1)})
res["by_side"] = {str(L): dict(n=sum(r["Lp"] == L for r in rows),
                               hpd_u_mean=float(np.mean([r["true"]["u"] for r in rows if r["Lp"] == L])) if any(r["Lp"] == L for r in rows) else None)
                  for L in (64, 128, 256)}
res["rows"] = [{k: v for k, v in r.items()} for r in rows]
json.dump(res, sys.stdout, indent=1)
