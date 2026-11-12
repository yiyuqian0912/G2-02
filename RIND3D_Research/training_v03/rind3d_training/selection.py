"""Selection law of obs-index-v0.3: inclusion design, rule probabilities and population weights.

Design. In every (scene, side L) group, subset s contributes n_s records drawn from
    q_s(w) = p_nat(w) * A_s(w) / P(A_s),      p_nat = uniform on origins {0..1024-L}^3,  A_nat = 1,
where A_s is the declared acceptance rule (A_informative = mixed and no source in the closed window box,
A_extreme = A_informative and minority fraction <= 0.01, A_rare = A_informative and minority count <= 64).
The group's records are therefore a sample of the mixture q(w) = sum_s (n_s/n) q_s(w).

Population targets (contract task profiles):
  global        p_nat(w)                    weight  1 / (n_nat/n + sum_s (n_s/n) A_s(w)/P(A_s))
  src-excluded  p_nat(w | no source in w)   weight  1 / (n'_nat/n' + sum_s (n_s/n') A_s(w) P(not cs)/P(A_s))
                                            (natural records containing a source are not samples of this task)
Weights are self-normalised to mean 1 inside each (scene, L) group, so every scene keeps equal mass per side.
The nonlocal subset is NOT part of any training mixture: its rule depends on the hidden blocker position (geometry
and source beyond the observation), so it would bias even the conditional law; it is an evaluation stratum.
Because A_informative/A_extreme/A_rare are functions of the observation and of the declared condition (no source in
the window), training the src-excluded task on them changes only the input marginal (covariate shift), not
p(source | observation, window, no source in window). For the global task they also condition on the hidden
source position, so they must be weighted (or the natural-only profile used).

P(A_s) estimation: fixed-design estimate from the rate stream (RATE_M uniform proposals) plus the natural stream,
combined by inverse variance with the unbiased inverse-binomial estimate (r-1)/(n-1) of the subset's own stream
(which stops at its r-th acceptance). All weights are privileged: they depend on the source-containment flag.
"""
import json
from pathlib import Path

import numpy as np

from .env import OUT
from .index import CODES, SIDES

RULE_STREAM = {"A_informative": "informative", "A_extreme": "extreme", "A_rare": "rare"}
TRAIN_SUBSETS = ("natural", "informative", "extreme", "rare")


def load_index(release, profile="P256"):
    d = OUT/"privileged"/"index"/release/profile
    scenes = json.loads((d/"scenes.json").read_text())
    recs, reports, rates = [], [], []
    for s in scenes:
        sd = d/f"{s['ordinal']:05d}"
        recs.append(np.load(sd/"records.npy"))
        reports.append(json.loads((sd/"report.json").read_text()))
        rates.append(np.load(sd/"rates.npy"))
    return scenes, np.concatenate(recs), reports, rates


def rule_probabilities(report, L, natural_records):
    """P(A_s) per rule for one (scene, L) and the containment probability P(cs)."""
    rh = report["rule_hits"][str(L)]["rate_stream"]
    nat = natural_records
    n_fix = rh["proposals"] + len(nat)
    out = {}
    for rule, stream in RULE_STREAM.items():
        h = rh[rule] + int(nat[rule].sum())
        p_fix = h/n_fix
        rp = report["report"][f"{stream}/{L}"]
        r, n = rp["accepted"], rp["proposals"]
        if rp["shortfall"] == 0 and r >= 2:
            p_inv, kind = (r-1)/(n-1), "inverse-binomial"
        else:
            p_inv, kind = (r/n if n else 0.0), "fixed-cap"
        p_ref = max(p_fix, p_inv, 1e-12)
        v_fix = p_ref*(1-p_ref)/n_fix
        v_inv = p_ref**2*(1-p_ref)/max(r-2, 1) if kind == "inverse-binomial" else p_ref*(1-p_ref)/max(n, 1)
        if h == 0:
            p = p_inv
        else:
            p = (p_fix/v_fix + p_inv/v_inv)/(1/v_fix + 1/v_inv) if v_fix > 0 and v_inv > 0 else p_fix
        out[rule] = {"P": float(p), "fixed": float(p_fix), "fixed_hits": int(h), "fixed_n": int(n_fix),
                     "own_stream": float(p_inv), "own_kind": kind, "own_r": int(r), "own_n": int(n)}
    hc = rh["contains_source"] + int(nat["contains_source"].sum())
    out["contains_source"] = {"P": hc/n_fix, "hits": int(hc), "n": int(n_fix)}
    return out


def weights(records, reports, task="global", subsets=TRAIN_SUBSETS):
    """Population importance weights for the records of a training profile (records outside the profile get 0).
    task: 'global' (sb-global / ma-additive) or 'src-excluded'. Returns (w, used_mask, diagnostics)."""
    w = np.zeros(len(records))
    used = np.isin(records["subset"], [CODES[s] for s in subsets])
    if task == "src-excluded":
        used &= ~records["contains_source"]
    diag = []
    by_scene = {}
    for i, s in enumerate(records["scene"]):
        by_scene.setdefault(int(s), []).append(i)
    for scene, idx in by_scene.items():
        idx = np.asarray(idx)
        rep = reports[scene] if isinstance(reports, list) else reports[scene]
        for L in SIDES:
            g = idx[(records["L"][idx] == L) & used[idx]]
            if not len(g):
                continue
            r = records[g]
            nat_all = records[idx[(records["L"][idx] == L) & (records["subset"][idx] == CODES["natural"])]]
            P = rule_probabilities(rep, L, nat_all)
            n = len(g)
            m = np.full(n, float((r["subset"] == CODES["natural"]).sum())/n)
            pncs = 1.0 - P["contains_source"]["P"] if task == "src-excluded" else 1.0
            for s in subsets[1:]:
                ns = float((r["subset"] == CODES[s]).sum())
                rule = "A_" + s
                if ns == 0:
                    continue
                Ps = max(P[rule]["P"], 1e-12)
                m += (ns/n)*r[rule].astype(float)*pncs/Ps
            ww = 1.0/m
            ww *= n/ww.sum()
            w[g] = ww
            diag.append({"scene": scene, "L": L, "n": n, "ess": float(ww.sum()**2/(ww**2).sum()),
                         "max_weight": float(ww.max()), "min_weight": float(ww.min()),
                         "P": {k: v["P"] for k, v in P.items()},
                         "P_fixed_vs_own_rel": {k: (v["own_stream"]-v["fixed"])/max(v["fixed"], 1e-12)
                                                for k, v in P.items() if k.startswith("A_") and v["fixed_hits"] >= 5}})
    return w, used, diag


def check_weights(records, reports, rates, task="global", subsets=TRAIN_SUBSETS):
    """Validation: weighted mixture statistics reproduce the natural (rate-stream) population statistics."""
    w, used, _ = weights(records, reports, task, subsets)
    out = {}
    for L in SIDES:
        sel = used & (records["L"] == L)
        rr = np.concatenate([x[x["L"] == L] for x in rates])
        if task == "src-excluded":
            rr = rr[~rr["contains_source"]]
        stats = {}
        for name, f in {"mixed": lambda a: a["category"] == 3, "A_informative": lambda a: a["A_informative"],
                        "A_extreme": lambda a: a["A_extreme"], "A_rare": lambda a: a["A_rare"],
                        "no_free": lambda a: a["category"] == 0,
                        "contains_source": lambda a: a["contains_source"]}.items():
            x = f(records[sel]).astype(float)
            stats[name] = {"unweighted": float(x.mean()), "weighted": float((w[sel]*x).sum()/w[sel].sum()),
                           "population_rate_stream": float(f(rr).astype(float).mean())}
        out[str(L)] = stats
    return out
