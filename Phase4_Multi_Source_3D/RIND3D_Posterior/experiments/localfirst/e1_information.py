"""Log-volume information of the certified interface region (docs/LOCALFIRST_E123.md §4).

usage: e1_information.py --e12 DIR
Information of a source region Z relative to the legal domain D: I(Z) = ln(Vol(D)/Vol(Z)) nats. For each task:
* I(F): the exact posterior support (Vol(F) from the exact phase), the full information of the observation;
* I(B): the receiver-subset outer bound (global visibility of 384 receivers);
* I(R): the certified interface region alone (interface edges + occluder edge list, no visibility); Vol(R) from the
  stratified Monte Carlo inside the depth-10 cover, or its 95% rule-of-three bound when no point hit (I(R) is then a
  95% lower bound);
* I(B n R): B plus the interface region; Vol(B n R) = share x Vol(B), with the rule-of-three bound when no sample hit.
Fractions I(.)/I(F) say how much of the observation's localisation each construction captures."""
import argparse, json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--e12", required=True); a = ap.parse_args()
    R = [json.loads(l) for l in (Path(a.e12) / "e12.jsonl").read_text().splitlines() if l.strip()]
    R = [r for r in R if "E1" in r and r.get("vol_F", 0) > 0]
    rows = []
    for r in R:
        d = {(x["mode"], x["n"]): x for x in r["E1"]}
        D = r["vol_domain"]; IF = np.log(D / r["vol_F"]); IB = np.log(D / r["vol_B"])
        out = dict(key=r["key"], IF=IF, IB=IB)
        for k in (("independent", 128), ("independent", 512), ("random", 512), ("all", 2048)):
            x = d.get(k)
            if not x or x.get("cover10_volume_est") is None:
                continue
            # Vol(R): the stratified Monte Carlo estimate inside the depth-10 cover; with no hit among MC_R points, the
            # 95% rule-of-three upper bound 3/MC_R of the cover volume (I(R) is then a 95% lower bound)
            v10 = x["cover10_volume_est"]; est = x.get("vol_R_strat") or 0.0
            vR = max(est if est > 0 else 3.0 / 20000 * v10, r["vol_F"])
            share = x.get("B_share_in_R")
            if share is None:
                continue
            share = share if share > 0 else 3.0 / x.get("B_samples", 20000)
            out[f"IR_{k[0]}_{k[1]}"] = np.log(D / vR)
            out[f"IBR_{k[0]}_{k[1]}"] = np.log(D / max(share * r["vol_B"], r["vol_F"]))
        rows.append(out)
    q = lambda x: "%.3g / %.3g / %.3g" % tuple(np.percentile(x, [10, 50, 90]))
    print(f"tasks with Vol(F) > 0: {len(rows)}")
    print("I(F) nats p10/p50/p90:", q([r["IF"] for r in rows]), "| I(B):", q([r["IB"] for r in rows]), "| I(B)/I(F):", q([r["IB"] / r["IF"] for r in rows]))
    for k in ("independent_128", "independent_512", "random_512", "all_2048"):
        g = [r for r in rows if f"IR_{k}" in r]
        print(f"{k}: I(R) (lower bound) {q([r[f'IR_{k}'] for r in g])}, I(R)/I(F) {q([r[f'IR_{k}'] / r['IF'] for r in g])} | "
              f"I(B n R) {q([r[f'IBR_{k}'] for r in g])}, I(B n R)/I(F) {q([r[f'IBR_{k}'] / r['IF'] for r in g])}, gain over B "
              f"{q([r[f'IBR_{k}'] - r['IB'] for r in g])} nats")


if __name__ == "__main__":
    main()
