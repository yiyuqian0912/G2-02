"""Tables for docs/SAMPLING_IMPACT_V3.md from sampling_impact.py records.

usage: sampling_impact_report.py --dir DIR --arms v2,v3rule_v2seed[,...] [--pair v2:v3rule_v2seed] [--json out.json]
Per arm: scene acceptance and attempts, wall time, natural-stream proposals per accepted window, P_A per side and
family (rate stream), the accepted m distribution, rare strata, coverage fallbacks. Per pair (identical worlds and
streams): the paired P_A difference and, inside the v3-rule arm, the v2 rule on the same rate proposals."""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SIDES = (64, 128, 256)


def load(d, arm):
    p = Path(d) / arm / "records.jsonl"
    recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    rej = Path(d) / arm / "rejected_attempts.jsonl"
    rejs = [json.loads(l) for l in rej.read_text().splitlines() if l.strip()] if rej.exists() else []
    return recs, rejs


def q(x, ps=(50, 95, 100)):
    x = np.asarray(x, float)
    return [round(float(np.percentile(x, p)), 2) for p in ps] if len(x) else None


def arm_summary(recs, rejs):
    acc = [r for r in recs if r["status"] == "accepted"]
    out = dict(scenes=len(recs), accepted=len(acc), final_rejected=Counter(r.get("reason") for r in recs if r["status"] != "accepted"),
               rejected_attempts=Counter(r.get("reason") for r in rejs), wall_s=q([r["wall"] for r in acc]),
               k=Counter(int(r["k"]) for r in acc))
    side = {}
    for Lp in SIDES:
        S = [s for r in acc for s in r["sides"] if int(s["Lp"]) == Lp]
        nat = [st for s in S for st in s["streams"] if st["name"] == "natural"]
        prop = sum(int(st["proposals"]) for st in nat); a = sum(int(st["accepted"]) for st in nat)
        rate_n = sum(int(s["rate_n"]) for s in S); rate_A = sum(int(s["rate_A"]) for s in S)
        reasons = Counter(); [reasons.update(s.get("rate_reasons", {})) for s in S]
        side[Lp] = dict(P_A=round(rate_A / rate_n, 4) if rate_n else None, proposals_per_window=round(prop / a, 2) if a else None,
                        rate_reasons=dict(reasons), fallback=sum(int(st.get("fallback", 0)) for st in nat),
                        rate_m=np.sum([s["rate_m"] for s in S], axis=0).tolist() if S else None)
        if S and "rate_A_mixed" in S[0]:
            ram = sum(int(s["rate_A_mixed"]) for s in S)
            side[Lp].update(P_A_mixed=round(ram / rate_n, 4), excluded_share=round(1 - rate_A / ram, 4) if ram else None,
                            rate_m_mixed=np.sum([s["rate_m_mixed"] for s in S], axis=0).tolist())
    out["sides"] = side
    m_all = Counter(); fam = defaultdict(Counter); rare = Counter()
    for r in acc:
        for L, mask, m in r["windows"]:
            m_all[int(m)] += 1; fam[r["family"]][int(m)] += 1
            if int(r["k"]) == 4 and int(m) == 4:
                rare[f"k4_m4_L{L}"] += 1
    out["m_accepted"] = dict(sorted(m_all.items())); out["rare"] = dict(rare)
    pa_f = {}
    for f in sorted({r["family"] for r in acc}):
        S = [s for r in acc if r["family"] == f for s in r["sides"]]
        pa_f[f] = round(sum(int(s["rate_A"]) for s in S) / sum(int(s["rate_n"]) for s in S), 4)
        if S and "rate_A_mixed" in S[0]:
            ram = sum(int(s["rate_A_mixed"]) for s in S)
            pa_f[f] = dict(P_A=pa_f[f], excluded_share=round(1 - sum(int(s["rate_A"]) for s in S) / ram, 4) if ram else None)
    out["P_A_by_family"] = pa_f
    return out


def paired(ra, rb):
    """Same slots: per-side P_A difference, consistency of rate_A_mixed (b) with rate_A (a), window identity."""
    A = {r["slot"]: r for r in ra if r["status"] == "accepted"}; B = {r["slot"]: r for r in rb if r["status"] == "accepted"}
    common = sorted(set(A) & set(B))
    res = {}
    for Lp in SIDES:
        da, db, cons = [], [], 0
        for s in common:
            sa = [x for x in A[s]["sides"] if int(x["Lp"]) == Lp][0]; sb = [x for x in B[s]["sides"] if int(x["Lp"]) == Lp][0]
            da.append(int(sa["rate_A"])); db.append(int(sb["rate_A"]))
            if "rate_A_mixed" in sb and int(sb["rate_A_mixed"]) == int(sa["rate_A"]):
                cons += 1
        n = sum(int([x for x in A[s]["sides"] if int(x["Lp"]) == Lp][0]["rate_n"]) for s in common)
        res[Lp] = dict(scenes=len(common), P_A_a=round(sum(da) / n, 4), P_A_b=round(sum(db) / n, 4), rate_mixed_consistent=cons)
    same = sum(A[s]["windows"] == B[s]["windows"] for s in common)
    res["identical_window_lists"] = f"{same}/{len(common)}"
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--arms", required=True)
    ap.add_argument("--pair", action="append", default=[]); ap.add_argument("--json")
    a = ap.parse_args()
    data = {arm: load(a.dir, arm) for arm in a.arms.split(",")}
    out = {arm: arm_summary(*v) for arm, v in data.items()}
    for p in a.pair:
        x, y = p.split(":")
        out[f"pair {p}"] = paired(data[x][0], data[y][0])
    s = json.dumps(out, indent=1, default=str)
    print(s)
    if a.json:
        Path(a.json).write_text(s)


if __name__ == "__main__":
    main()
