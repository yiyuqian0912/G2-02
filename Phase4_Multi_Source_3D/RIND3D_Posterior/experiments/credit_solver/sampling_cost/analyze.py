"""Scene-held-out prediction, selection ablations, and matched measured generation plus labeling."""
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import rankdata, spearmanr
from generate_pool import groups
from selection import CostModel,select,wants_alternative


def read(path):
    return list(map(json.loads,path.read_text().splitlines()))


def quant(x):
    return [float(v) for v in np.quantile(x,[.5,.95,1])] if len(x) else []


def rho(x,y):
    if len(set(x))<2 or len(set(y))<2:
        return None
    return float(spearmanr(x,y).statistic)


def auc(y,p):
    pos=np.asarray(y,dtype=bool)
    n=int(pos.sum())
    if n in (0,len(y)):
        return None
    return float((rankdata(p)[pos].sum()-n*(n+1)/2)/(n*(len(y)-n)))


def model_score(records,modes):
    tr=[r for r in records if r["slot"]<18]
    te=[r for r in records if r["slot"]>=18]
    out={}
    for mode in modes:
        m=CostModel.fit(tr,mode)
        p=[m.predict(r) for r in te]
        y=[r["wall"] for r in te]
        out[mode]=dict(n=[len(tr),len(te)],rho=rho(y,p), log_mae=float(np.abs(p-np.log(y)).mean()),
                       slow_auc=auc(y>np.quantile([r["wall"] for r in tr],.9),p),
                       incomplete_auc=auc([r["status"]!="CERTIFIED" for r in te],p))
    return out


def distribution(rs):
    success=[r for r in rs if r.get("status")=="CERTIFIED"]
    out=dict(n=len(rs),single=sum(r["minority"]==1 for r in rs),
             at_most_ten=sum(r["minority"]<=10 for r in rs),
             minority_fraction=quant([r["fraction"] for r in rs]),median_edges=quant([r["edges"] for r in rs])[0],
             geometry_families=len(set(r["family"] for r in rs)))
    if success and all("wall" in r for r in rs):
        out.update(certified=len(success),wall=quant([r["wall"] for r in rs]),support=quant([r["cells"] for r in success]),
                   cheap_but_sparse=min((r["wall"],r["key"]) for r in success if r["fraction"]<.01)
                   if any(r["fraction"]<.01 for r in success) else None)
    return out


def evaluate(pool,policies,model,fresh=None,gen=None):
    result={}
    groups_chosen=defaultdict(dict)
    for policy in policies:
        picked=[]
        score_t=gen_t=label_t=setup_t=screen_t=0.
        changes=0
        for g,pair in sorted(pool.items()):
            t=perf_counter()
            k=select(pair,policy,model)
            score_t+=perf_counter()-t
            changes+=k
            chosen=pair[k]
            picked+=chosen
            prefix=2 if wants_alternative(pair[0],policy) else 1
            # The natural holdback is decidable before generating the second window.
            gt=gen[g,prefix]["wall"] if gen else 0.
            st=gen[g,prefix]["setup"] if gen else 0.
            feature=sum(r["spatial_s"] for w in pair for r in w) if policy=="interface" else 0.
            if policy=="geometry":
                feature=sum(r["geo"]["e2e"] for w in pair for r in w)
            screen_t+=feature
            gen_t+=gt
            setup_t+=st
            rows=([fresh[r["key"]] for r in chosen] if fresh else chosen)
            lt=sum(r["wall"] for r in rows)
            label_t+=lt
            groups_chosen[g][policy]=[st+gt+feature+lt,sum(r["status"]=="CERTIFIED" for r in rows)]
        stats=distribution(picked)
        if fresh:
            r=[fresh[v["key"]] for v in picked]
            stats.update(fresh_certified=sum(v["status"]=="CERTIFIED" for v in r),fresh_label_wall=quant([v["wall"] for v in r]),
                         fresh_support=quant([v["cells"] for v in r if v["status"]=="CERTIFIED"]))
        n=stats.get("fresh_certified",stats["certified"])
        stats.update(groups=len(pool),changes=changes,setup=setup_t,generation=gen_t,features=screen_t,
                     scoring=score_t,labeling=label_t,per_success=(setup_t+gen_t+screen_t+score_t+label_t)/n,
                     rare=sum(r["fraction"]<=.001 for r in picked),broad=sum((r["cells"] or 0)>=10000 for r in picked))
        result[policy]=stats
    return result,groups_chosen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("results",type=Path)
    p.add_argument("work",type=Path)
    args=p.parse_args()
    rd=lambda name:read(args.results/(name+".jsonl"))
    rows=rd("channels")
    geo={v["key"]:v for v in rd("two_pairs")}
    arch=[r for r in rows if "wall"in r]
    for r in arch:
        r["geo"]=geo.get(r["key"])
    first={}
    for r in arch:
        if r["w"]%4==0:
            first.setdefault((r["slot"],r["L"]),r)
    subset=[r for r in first.values() if r["geo"]["status"]=="ok" and len(r["geo"]["steps"])==2]
    models=model_score(arch,["size","balance","local"])
    geometric_models=model_score(subset,["size","balance","local","geometric"])
    wgroup=defaultdict(list)
    for r in arch:
        wgroup[r["slot"],r["w"]].append(r)
    pool={(s,L):[wgroup[s,w],wgroup[s,w+1]] for s in range(36) for w,L in [(0,64),(4,128),(8,256)]}
    matched={k:v for k,v in pool.items() if k[0]>=18}
    model=CostModel.fit([r for r in arch if r["slot"]<18],"balance")
    policies=["natural","balance","interface","cost","mixture"]
    old,unused=evaluate(matched,policies,model)
    pilot={g:pool[g] for g in groups()}
    gen={((v["slot"],v["L"]),v["prefix"]):v for v in rd("prefix")}
    fresh={v["key"]:v for v in rd("pilot_labels")}
    new,bygroup=evaluate(pilot,policies+["geometry"],model,fresh,gen)
    # Nine independently selected confirmation geometries. Omit the known tail sentinel from this interval.
    keys=sorted(k for k in bygroup if k[0]>=18)
    rng=np.random.default_rng(12119)
    ix=rng.integers(0,len(keys),(4000,len(keys)))
    def ratios(pol):
        data=np.array([bygroup[k][pol] for k in keys])
        costs,ns=data[ix].sum(axis=1).T
        return costs/ns
    intervals={}
    for pol in new:
        intervals[pol]=list(np.quantile(ratios(pol)/ratios("natural"),[.025,.5,.975]))
    success=[r for r in arch if r["status"]=="CERTIFIED"]
    metrics={}
    for f in ("fraction","minority","edges","orientation","bins","clusters"):
        metrics[f]=dict(time=rho([r[f] for r in arch],[r["wall"] for r in arch]),
                        support=rho([r[f] for r in success],[r["cells"] for r in success]),
                        effective=rho([r[f] for r in success],[r["effective_cells"] for r in success]),
                        by_size={str(L):rho([r[f] for r in arch if r["L"]==L],[r["wall"] for r in arch if r["L"]==L])
                                 for L in (64,128,256)})
    gp={name:{field:rho([getter(r["geo"]) for r in subset],[r[field] for r in subset])
              for field in ("wall","cells","lower")}
        for name,getter in (("two_edge_volume",lambda g:g["steps"][1]["vol"]),
                            ("volume_ratio",lambda g:g["steps"][1]["vol"]/g["steps"][0]["vol"]),
                            ("pieces",lambda g:g["steps"][1]["pieces"]),("explanations",lambda g:g["explanations"]))}
    records=read(args.work/"sampling_impact/v3/records.jsonl")
    prop={}
    for L in (64,128,256):
        sides=[s for rec in records for s in rec["sides"] if s["L"]==L]
        prop[L]=dict(quota_proposals=sum(s["proposals"] for s in sides),quota_accepted=sum(s["quota"] for s in sides),
                     rate_proposals=sum(s["rate_n"] for s in sides),rate_accepts=sum(s["rate_A"] for s in sides),
                     by_family={family:[sum(s[k] for r in records if r["family"]==family for s in r["sides"] if s["L"]==L)
                                for k in ("rate_A","rate_n")] for family in sorted({r["family"] for r in records})})
    def geometry_stat(fn):
        return quant([fn(r["geo"]) for r in subset])
    out=dict(population=distribution(rows),by_size={L:distribution([r for r in rows if r["L"]==L]) for L in (64,128,256)},
             by_family={f:distribution([r for r in arch if r["family"]==f]) for f in sorted({r["family"] for r in arch})},
             proposals=prop,archive=distribution(arch),failures=[{k:r[k] for k in ("key","L","family","minority","fraction","edges","lower","live")} for r in arch if r["status"]!="CERTIFIED"],
             associations=metrics,models=models,geometry_models=geometric_models,
             geometry=dict(n=len(subset),rank_correlations=gp,time=geometry_stat(lambda g:g["e2e"]),second_volume=geometry_stat(lambda g:g["steps"][1]["vol"]),
               incremental_ratio=geometry_stat(lambda g:g["steps"][1]["vol"]/g["steps"][0]["vol"]),
               pieces=geometry_stat(lambda g:g["steps"][1]["pieces"]),explanations=geometry_stat(lambda g:g["explanations"])),
             historical=old,measured_pilot=new,confirmatory_cluster_bootstrap_ratio=intervals,
             bin_outcomes=[dict(bin=[a,b],n=len(v),failed=sum(r["status"]!="CERTIFIED" for r in v),
                      median_s=quant([r["wall"] for r in v])[0] if v else None)
                      for a,b in [(0,1e-4),(1e-4,.001),(.001,.01),(.01,.1),(.1,.5)]
                      for v in [[r for r in arch if a<=r["fraction"]<b or b==.5 and r["fraction"]==.5]]],
             paired_by_group={str(k):v for k,v in bygroup.items()})
    print(json.dumps(out,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
