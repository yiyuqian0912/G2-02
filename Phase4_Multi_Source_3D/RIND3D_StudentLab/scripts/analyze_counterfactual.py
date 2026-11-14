"""Analysis of the counterfactual pairs (scripts/counterfactual.py): how much does the full-world posterior depend on
hidden geometry that leaves X unchanged, and where do Students sit relative to the pairwise floor?

usage: analyze_counterfactual.py <cf_dir> <out.json> [name=run_dir ...]
       analyze_counterfactual.py --depth <cfd_dir> <out.json> [name=run_dir ...]   (depth sets; analyze_depth)
Per pair (q = stored label of G, q' = label of the edited world G'; both certified, TV <= 0.05 each):
  tv, jsd (nats), src_in_q' (construction check: the true source cell is in supp q'), H(q), H(q')
Per Student p (identical for both worlds, because X is identical): KL(q || p), KL(q' || p), their mean, and the
floor JSD(q, q') = min_p of that mean (p = (q + q') / 2)."""
import json, sys
from pathlib import Path
import numpy as np
import torch

LAB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB))
from slab.data import LabTable, collate, to_dev, cf_of_cell   # noqa: E402
from slab.evaluate import lab_model                           # noqa: E402

G = 128


def dist(cells, w):
    u, inv = np.unique(np.asarray(cells, np.int64), return_inverse=True)
    q = np.bincount(inv, weights=np.asarray(w, float)); return u, q / q.sum()


def main():
    cf = Path(sys.argv[1]); outp = sys.argv[2]; runs = dict(a.split("=", 1) for a in sys.argv[3:])
    T = LabTable(); rows = json.loads((cf / "cf_rows.json").read_text())
    index = {(r["scene"], r["window"], r["source"]): r for r in map(json.loads, open(cf / "cf_index.jsonl"))}
    models = {n: lab_model(d, "cuda") for n, d in runs.items()}
    recs = []
    for i in rows:
        r = T.rows[i]; key = (int(r["scene"]), int(r["window"]), int(r["source"]))
        meta = index.get(key)
        if meta is None or meta.get("status") not in ("certified", "certified_stat"):
            recs.append(dict(i=int(i), status=None if meta is None else meta.get("status"))); continue
        cells, q, _ = T.label(i)
        z = np.load(cf / f"cf_{key[0]:06d}_w{key[1]:03d}_s{key[2]}.npz")
        c2, q2 = dist(z["cell"], z["lo"])
        allc = np.union1d(cells, c2)
        a = np.zeros(len(allc)); b = np.zeros(len(allc))
        a[np.searchsorted(allc, cells)] = q; b[np.searchsorted(allc, c2)] = q2
        m = 0.5 * (a + b)
        kl = lambda x, y: float(np.sum(np.where(x > 0, x * (np.log(np.where(x > 0, x, 1)) - np.log(np.where(y > 0, y, 1))), 0)))
        src = T.source_cell(i)
        rec = dict(i=int(i), Lp=int(r["Lp"]), family=meta["family"], n_removed=len(meta["removed"]), n_bodies=meta["n_bodies"],
                   tv=float(0.5 * np.abs(a - b).sum()), jsd=0.5 * kl(a, m) + 0.5 * kl(b, m), src_in_q2=bool(np.isin(src, c2)),
                   hq=float(-(q * np.log(q)).sum()), hq2=float(-(q2 * np.log(q2)).sum()), n_cells=int(len(cells)), n_cells2=int(len(c2)),
                   status=meta["status"])
        B = to_dev(collate(T, np.array([i]), "onehot"), "cuda")
        cc, ff = cf_of_cell(allc)
        for n, f in models.items():
            with torch.no_grad():
                lp = f(B)[0][torch.from_numpy(cc).cuda(), torch.from_numpy(ff).cuda()].double().cpu().numpy()
            rec[f"{n}/kl_q"] = float(np.sum(a[a > 0] * (np.log(a[a > 0]) - lp[a > 0])))
            rec[f"{n}/kl_q2"] = float(np.sum(b[b > 0] * (np.log(b[b > 0]) - lp[b > 0])))
            rec[f"{n}/pair"] = 0.5 * (rec[f"{n}/kl_q"] + rec[f"{n}/kl_q2"])
        recs.append(rec)
    ok = [x for x in recs if "tv" in x]
    summ = dict(pairs=len(recs), labelled=len(ok), src_in_q2=float(np.mean([x["src_in_q2"] for x in ok])),
                identical=float(np.mean([x["tv"] < 0.05 for x in ok])))
    for L in (64, 128, 256, None):
        s = [x for x in ok if L is None or x["Lp"] == L]
        d = dict(n=len(s), tv_median=float(np.median([x["tv"] for x in s])), tv_mean=float(np.mean([x["tv"] for x in s])),
                 jsd_mean=float(np.mean([x["jsd"] for x in s])), frac_tv_gt_0_5=float(np.mean([x["tv"] > 0.5 for x in s])),
                 dH_mean=float(np.mean([x["hq2"] - x["hq"] for x in s])))
        for n in runs:
            d[n] = dict(kl_q=float(np.mean([x[f"{n}/kl_q"] for x in s])), kl_q2=float(np.mean([x[f"{n}/kl_q2"] for x in s])),
                        pair=float(np.mean([x[f"{n}/pair"] for x in s])))
        summ[f"L{L}" if L else "all"] = d
    json.dump(dict(summary=summ, pairs=recs), open(outp, "w"), indent=1)
    print(json.dumps(summ, indent=1))




def analyze_depth(cf, outp, runs):
    """Depth counterfactual sets: per row, worlds {G (λ = 1)} ∪ {G'_λ accepted}. With equal weights, the generalised JSD
    H(m) - mean_i H(q_i), m = mean_i q_i, is the floor of mean_i KL(q_i || p) over X-only predictors p, and
    mean_i KL(q_i || p) = JSD + KL(m || p) exactly."""
    T = LabTable(); rows = json.loads((Path(cf) / "cfd_rows.json").read_text())
    recs = [json.loads(l) for l in open(Path(cf) / "cfd_index.jsonl")]
    by = {}
    for r in recs:
        by.setdefault((r["scene"], r["window"], r["source"]), []).append(r)
    models = {n: lab_model(d, "cuda") for n, d in runs.items()}
    out = []; status = {}
    for r in recs:
        status[r["status"]] = status.get(r["status"], 0) + 1
    for i in rows:
        rr = T.rows[i]; key = (int(rr["scene"]), int(rr["window"]), int(rr["source"]))
        acc = [x for x in by.get(key, []) if x["status"] in ("certified", "certified_stat")]
        if not acc:
            continue
        cells, q, _ = T.label(i)
        dists = [(1.0, cells, q)]
        for x in acc:
            z = np.load(Path(cf) / f"cfd_{key[0]:06d}_w{key[1]:03d}_s{key[2]}_l{x['lam']:g}.npz")
            c2, q2 = dist(z["cell"], z["lo"]); dists.append((x["lam"], c2, q2))
        allc = np.unique(np.concatenate([d[1] for d in dists]))
        Q = np.zeros((len(dists), len(allc)))
        for k, (_, c, qq) in enumerate(dists):
            Q[k, np.searchsorted(allc, c)] = qq
        m = Q.mean(0)
        H = lambda x: float(-np.sum(x[x > 0] * np.log(x[x > 0])))
        src = T.source_cell(i)
        rec = dict(i=int(i), Lp=int(rr["Lp"]), n_worlds=len(dists), lams=[d[0] for d in dists], jsd=H(m) - np.mean([H(x) for x in Q]),
                   H=[H(x) for x in Q], ncell=[int((x > 0).sum()) for x in Q], src_in_all=bool(all(np.isin(src, d[1]) for d in dists)))
        B = to_dev(collate(T, np.array([i]), "onehot"), "cuda"); cc, ff = cf_of_cell(allc)
        for n, f in models.items():
            with torch.no_grad():
                lp = f(B)[0][torch.from_numpy(cc).cuda(), torch.from_numpy(ff).cuda()].double().cpu().numpy()
            kls = [float(np.sum(x[x > 0] * (np.log(x[x > 0]) - lp[x > 0]))) for x in Q]
            rec[f"{n}/kl_mean"] = float(np.mean(kls)); rec[f"{n}/kl_orig"] = kls[0]
            rec[f"{n}/kl_mix"] = float(np.sum(m[m > 0] * (np.log(m[m > 0]) - lp[m > 0])))
        out.append(rec)
    summ = dict(attempts=len(recs), status=status, rows_with_sets=len(out), src_in_all=float(np.mean([x["src_in_all"] for x in out])))
    for L in (64, 128, 256, None):
        s = [x for x in out if L is None or x["Lp"] == L]
        if not s:
            continue
        d = dict(n=len(s), worlds_mean=float(np.mean([x["n_worlds"] for x in s])), jsd_mean=float(np.mean([x["jsd"] for x in s])),
                 jsd_median=float(np.median([x["jsd"] for x in s])),
                 dH_range_mean=float(np.mean([max(x["H"]) - min(x["H"]) for x in s])))
        for n in runs:
            d[n] = {k: float(np.mean([x[f"{n}/{k}"] for x in s])) for k in ("kl_orig", "kl_mean", "kl_mix")}
        summ[f"L{L}" if L else "all"] = d
    json.dump(dict(summary=summ, rows=out), open(outp, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "--depth":
        analyze_depth(sys.argv[2], sys.argv[3], dict(a.split("=", 1) for a in sys.argv[4:]))
    else:
        main()
