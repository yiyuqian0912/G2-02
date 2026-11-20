"""Exact incremental outer volumes after two distinct adjacent FREE interface edges."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from time import perf_counter

import inputs
from poly import Poly, within
from solver import Budget, Limit, apply_pair, choose_pairs, cone, subtract_many
from generate_pool import groups


def selected(rows):
    base={}
    pairs={(s,L) for s,L in groups()}
    for r in rows:
        if r["slot"]>=36:
            continue
        if r["w"]%4==0:
            base.setdefault((r["slot"],r["L"]),r["key"])
    return set(base.values())|{r["key"] for r in rows if (r["slot"],r["L"]) in pairs and r["w"]%4<2}


def measure(obs, key, seconds=2):
    t=perf_counter()
    bud=Budget(seconds,4096)
    seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],"big")
    pairs=choose_pairs(obs.cls,2,seed,"interface")
    pieces=[Poly.cube()]
    steps=[]
    explanations=None
    try:
        for i,(d,a) in enumerate(pairs):
            bud.check()
            dark=[cone(b,obs.world(d)) for b in obs.bodies]
            lit=[cone(b,obs.world(a)) for b in obs.bodies]
            if i==0:
                explanations=0
                for c in dark:
                    h=within(pieces[0],c)
                    if h is not None and subtract_many([h],lit,bud):
                        explanations+=1
            parts=[]
            for h in pieces:
                parts.extend(apply_pair(h,dark,lit,bud))
                bud.check(len(parts))
            pieces=parts
            vol=sum((q.volume for q in pieces),Fraction(0))
            steps.append(dict(vol=float(vol),pieces=len(pieces)))
        return dict(status="ok",steps=steps,explanations=explanations,wall=perf_counter()-t)
    except Limit as err:
        return dict(status=str(err),steps=steps,wall=perf_counter()-t)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("work",type=Path)
    p.add_argument("rows",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    rows=list(map(json.loads,a.rows.read_text().splitlines()))
    assert sum(r["slot"]==35 for r in rows)>0
    with a.output.open("x") as out:
        for key in sorted(selected(rows)):
            s,w,j=key.split("_")
            t=perf_counter()
            obs,_=inputs.load(a.work/f"sampling_impact/v3/scenes/0000/{s}.npz",int(w[1:]),int(j[1:]))
            result=measure(obs,key)
            print(json.dumps(dict(key=key,e2e=perf_counter()-t,**result)),file=out,flush=True)

