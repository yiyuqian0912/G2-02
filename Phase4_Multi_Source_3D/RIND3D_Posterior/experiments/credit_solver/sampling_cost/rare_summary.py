"""Paired marginal costs at affected window slots. Other scene work is identical in both arms."""
import argparse
from itertools import product
import hashlib
import json
from pathlib import Path


def reserve_slot(slot,w):
    # Slot and window position are assigned before any source coordinate is drawn.
    return hashlib.sha256(f"rare/v2/12119/{slot}/{w}".encode()).digest()[0]<128


def summary(rows,policy,coins=None):
    costs=[]
    attempts=certified=switches=channels=focal_success=0
    for i,v in enumerate(rows):
        try_second=policy!="natural"
        if policy=="reserve":
            try_second=not reserve_slot(v["slot"],v["w"])
        elif policy=="legacy_reserve":
            tag=f"rare/v1/12119/{v['scene']}/{v['w']}"
            try_second=hashlib.sha256(tag.encode()).digest()[0]>=128
        elif coins is not None:
            try_second=coins[i]
        choose=try_second and v["take"]
        kept=v["arms"]["swap" if choose else "base"]
        switches+=choose
        attempts+=try_second*v["attempts"]
        channels+=len(kept)
        certified+=sum(q["status"]=="CERTIFIED" for q in kept)
        focal_success+=sum(q["key"].endswith("_s"+str(v["focal"])) and q["status"]=="CERTIFIED" for q in kept)
        costs.append(sum(q["seconds"] for q in kept)+(v["setup"]+v["screen"] if try_second else 0))
    return dict(windows=len(rows),swapped=switches,extra_proposals=attempts,eligible_channels=channels,
                certified=certified,focal_certified=focal_success,incremental_plus_label_s=sum(costs),
                marginal_s_per_certified=sum(costs)/certified)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("input",type=Path)
    a=p.parse_args()
    rs=list(map(json.loads,a.input.read_text().splitlines()))
    result={}
    for name,subset in [("all",rs),("held_out",[r for r in rs if r["slot"]>=18])]:
        result[name]={p:summary(subset,p) for p in ("natural","always","reserve","legacy_reserve")}
        # Exact enumeration of independent fair holdback choices on this measured finite panel.
        trials=[summary(subset,"coin",mask) for mask in product((False,True),repeat=len(subset))]
        base=result[name]["natural"]["incremental_plus_label_s"]
        result[name]["fair_coin"]={k:sum(v[k] for v in trials)/len(trials) for k in
            ("incremental_plus_label_s","certified","eligible_channels","focal_certified")}
        result[name]["fair_coin"]["fraction_strictly_cheaper"]=sum(v["incremental_plus_label_s"]<base for v in trials)/len(trials)
    print(json.dumps(result,indent=2))
