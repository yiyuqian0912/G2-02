#!/usr/bin/env python3
"""Campaign report for a RIND3D-100K root: report.py <root> [--out REPORT.json]

* **Physical:** attempts, accepted, rejected by reason, failed; rejection rate r and the selection-law bound
  r/(1 - r) with a one-sided 99% Clopper-Pearson upper limit; k and family histograms.
* **Layer balance:** m histogram raw and omega-weighted (omega = rate-stream Lambda_hat of the window's side); m by side
  and by k.
* **Cost:** physical wall per scene (p50, p95, max) and its breakdown (render tiles, probes, hull certificates).
* **Storage:** scene and label bytes.
* **Labels:** slot coverage by status; per-side CPU seconds (mean, p50, p95, p99, max) and wall seconds;
  certification rates; failures by kind.
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parent
sys.path[:0] = [str(HERE), str(REPO / "RIND3D_Dataset1K"), str(REPO / "RIND3D_Posterior" / "src")]


def cp_upper(k, n, conf=0.99):
    from scipy.stats import beta
    return 1.0 if k >= n else float(beta.ppf(conf, k + 1, n - k))


def pct(a):
    a = np.asarray(a, float)
    if not len(a):
        return None
    return dict(n=int(len(a)), mean=float(a.mean()), p50=float(np.percentile(a, 50)), p95=float(np.percentile(a, 95)),
                p99=float(np.percentile(a, 99)), max=float(a.max()), sum=float(a.sum()))


def main():
    root = Path(sys.argv[1])
    from rind3d_100k.storage import journal_index, read_record, decode_payload
    recs = [json.loads(l) for l in (root / "manifest" / "physical.jsonl").read_text().splitlines() if l.strip()]
    acc = {r["slot"]: r for r in recs if r["status"] == "accepted"}
    attempts = len(recs); rej = collections.Counter(r.get("reason") for r in recs if r["status"] == "rejected")
    failed = collections.Counter(r.get("kind") or (r.get("error") or "")[:60] for r in recs if r["status"] == "failed")
    n_rej = sum(rej.values()); n_att_scene = sum(1 for r in recs if r["status"] in ("accepted", "rejected"))
    r_hat = n_rej / max(n_att_scene, 1); r_up = cp_upper(n_rej, max(n_att_scene, 1))
    m = np.zeros(4); mw = np.zeros(4); by_side = {L: np.zeros(4) for L in ("64", "128", "256")}; by_k = collections.defaultdict(lambda: np.zeros(4))
    kh = collections.Counter(); fam = collections.Counter(); walls = []; tiles = []; probes = []; hull = []; byts = []
    for r in acc.values():
        kh[r["k"]] += 1; fam[r["family"]] += 1; walls.append(r.get("wall", r["timing"]["total"]))
        tiles.append(r["work"]["tiles"]); probes.append(r["work"]["probes"]); byts.append(r.get("bytes", 0))
        hull.append(r["work"].get("hull_certified", 0) / max(r["work"].get("hull_calls", 1), 1))
        lam = r["Lambda_hat"]
        for (L, mask, mm) in r["windows"]:
            Lp = str(L)
            m[mm - 1] += 1; mw[mm - 1] += lam[Lp]; by_side[Lp][mm - 1] += 1; by_k[r["k"]][mm - 1] += 1
    idx = journal_index(root); costs = collections.defaultdict(list); walls_l = collections.defaultdict(list)
    status = collections.Counter(); fallbacks = collections.Counter()
    for key, loc in idx.items():
        meta = decode_payload(read_record(loc)[1])["meta"]
        costs[str(meta["L"])].append(meta["cpu"]); walls_l[str(meta["L"])].append(meta["wall"]); status[meta["status"]] += 1
        if meta.get("fallback"):
            fallbacks[meta["fallback"][:40]] += 1
    fails = collections.Counter()
    fp = root / "labels" / "failures.jsonl"
    if fp.exists():
        for line in fp.read_text().splitlines():
            fails[json.loads(line)["kind"]] += 1
    n_inf = sum(bin(int(mask)).count("1") for r in acc.values() for (_L, mask, _m) in r["windows"])
    n_slots = sum(int(r["k"]) * len(r["windows"]) for r in acc.values())
    lab_bytes = sum(p.stat().st_size for p in (root / "labels" / "journal").glob("*.r3lj")) if (root / "labels" / "journal").exists() else 0
    rep = dict(root=str(root), attempts=attempts, accepted=len(acc), rejected=dict(rej), failed=dict(failed),
               rejection_rate=r_hat, rejection_rate_upper99=r_up, selection_bound=r_up / (1 - r_up) if r_up < 1 else 1.0,
               k_hist={str(k): kh[k] for k in sorted(kh)}, families=dict(fam),
               windows=int(m.sum()), m_hist=m.tolist(), m_share=(m / max(m.sum(), 1)).round(4).tolist(),
               m_share_weighted=(mw / max(mw.sum(), 1e-300)).round(4).tolist(),
               m_by_side={k: v.tolist() for k, v in by_side.items()}, m_by_k={str(k): v.tolist() for k, v in sorted(by_k.items())},
               informative_channels=n_inf, source_slots=n_slots,
               physical_wall_s=pct(walls), tiles_per_scene=pct(tiles), probes_per_scene=pct(probes),
               hull_certified_fraction=pct(hull), scene_bytes=pct(byts),
               labels=dict(journal_records=len(idx), status=dict(status), failures=dict(fails), gpu_fallbacks=dict(fallbacks),
                           cpu_s_by_side={L: pct(v) for L, v in costs.items()}, wall_s_by_side={L: pct(v) for L, v in walls_l.items()},
                           journal_bytes=lab_bytes, bytes_per_label=lab_bytes / max(len(idx), 1)))
    s = json.dumps(rep, indent=1, default=str)
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else root / "REPORT.json"
    Path(out).write_text(s)
    print(s)


if __name__ == "__main__":
    main()
