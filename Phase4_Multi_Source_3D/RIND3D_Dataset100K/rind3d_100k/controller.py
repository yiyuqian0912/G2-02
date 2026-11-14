"""Epoch-frozen effective-layer balancing controller for RIND3D-100K.

The 1K rule is unchanged (`rind3d_1k.controller`):
* equal target counts per m = 1..4;
* thinning c(m, L, k) proportional to deficit(m) / p_hat(m | L, k), clipped to [c_min, 1];
* k minimises the expected imbalance, with a keyed forced floor per k.

It is generalised in two ways:
* **Quotas** are parameters (4/4/4).
* **Planning is per epoch, so that 16 GPUs can work in parallel.**
  - The plan for every slot of epoch e is computed in slot order before epoch e starts.
  - Its inputs are the prior plus all accepted scenes of epochs <= e - lag (lag >= 1).
  - Within the epoch, the controller adds each planned slot's *expected* composition (deterministic, independent of
    any outcome) so that consecutive slots diversify.
  - The plan is therefore reproducible regardless of processing order.
  - It never depends on the sources of the slot being planned, which is the premise of the per-channel posterior
    theorem (docs/SELECTION_LAW_100K.md).
"""
import json
from pathlib import Path

import numpy as np

from rind3d.geometry import keyed_rng

SIDES = (64, 128, 256)
K_STREAM = 56000


class Controller100K:
    def __init__(self, quota, c_min=0.01, pseudo=2.0):
        self.quota = {int(L): int(q) for L, q in quota.items()}
        self.c_min = float(c_min)
        self.nA = {L: {k: np.full(k, pseudo, float) for k in range(1, 5)} for L in SIDES}   # natural m counts of A-windows
        self.rate = {L: {k: [1.0, 2.0] for k in range(1, 5)} for L in SIDES}               # (A count, proposals)
        self.N = np.zeros(4)

    # -------------------------------------------------------------- statistics
    def add_natural(self, L, k, rate_m, nA, n):
        self.nA[int(L)][int(k)] += np.asarray(rate_m[:k], float)
        self.rate[int(L)][int(k)][0] += float(nA); self.rate[int(L)][int(k)][1] += float(n)

    def update(self, rec):
        """An accepted 100K scene record (m_hist, sides with rate_m / rate_exact_A / rate_exact)."""
        k = int(rec["k"])
        self.N += np.asarray(rec["m_hist"], float)
        for s in rec["sides"]:
            self.add_natural(s["Lp"], k, s["rate_m"], s["rate_exact_A"], s["rate_exact"])

    def prior_from_1k_manifest(self, path, weight=1.0):
        """Natural statistics of an independent campaign (1K release rate streams, 1024 proposals per side)."""
        n = 0
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            if r.get("status") != "accepted":
                continue
            for s in r["sides"]:
                self.add_natural(s["Lp"], r["k"], weight * np.asarray(s["rate_m"][:r["k"]], float),
                                 weight * s["rate_A"], weight * 1024)
            n += 1
        return n

    def p_hat(self, L, k):
        v = self.nA[L][k]
        return v / v.sum()

    # -------------------------------------------------------------- planning
    def plan_k(self, k, eps=0.5):
        tot = self.N.sum() + 64
        deficit = np.maximum(eps, tot / 4 - self.N)[:k]
        c, comp, props = {}, np.zeros(4), 0.0
        for L in SIDES:
            p = self.p_hat(L, k)
            d = deficit / deficit.sum()
            cc = d / np.maximum(p, 1e-9)
            cc = np.clip(cc / cc.max(), self.c_min, 1.0)
            pi = p * cc / np.sum(p * cc)
            comp[:k] += self.quota[L] * pi
            a, n = self.rate[L][k]
            props += self.quota[L] / max((a / n) * np.sum(p * cc), 1e-9)
            c[str(L)] = [float(x) for x in cc] + [1.0] * (4 - k)
        return c, comp, props

    def choose(self, u, floor=0.15):
        if u < 4 * floor:
            k = 1 + int(u // floor)
            c, comp, props = self.plan_k(k)
            return k, c, dict(expected_composition=comp.tolist(), expected_proposals=props, forced=True)
        best = None
        for k in range(1, 5):
            c, comp, props = self.plan_k(k)
            after = self.N + comp
            score = float(np.sum((after / after.sum() - 0.25) ** 2))
            key = (round(score, 9), props)
            if best is None or key < best[0]:
                best = (key, k, c, comp, props)
        _, k, c, comp, props = best
        return k, c, dict(expected_composition=comp.tolist(), expected_proposals=props, forced=False)

    def state(self):
        return dict(N=self.N.tolist(), nA={str(L): {str(k): v.tolist() for k, v in d.items()} for L, d in self.nA.items()},
                    rate={str(L): {str(k): v for k, v in d.items()} for L, d in self.rate.items()})


def epoch_of(slot, cfg):
    from .sampling import epoch_size
    return int(slot) // epoch_size(cfg)


def plan_epoch(cfg, epoch, accepted_records, prior_manifest=None):
    """Deterministic plan {slot: {"k", "c_table", "plan"}} for every slot of `epoch`.
    Contract v2 (declared kappa, `sampling.plan_epoch_declared`): independent of every outcome; the other arguments are
    ignored. Contract v1 (adaptive controller): accepted_records of all epochs <= epoch - lag (anything later is
    ignored)."""
    from .sampling import declared, plan_epoch_declared
    if declared(cfg):
        return plan_epoch_declared(cfg, epoch)
    cc = cfg["controller"]; E = int(cc["epoch_scenes"]); lag = int(cc.get("plan_lag_epochs", 1))
    ctl = Controller100K(cfg["quota"], cc.get("c_min", 0.01))
    if prior_manifest and Path(prior_manifest).exists():
        ctl.prior_from_1k_manifest(prior_manifest, weight=float(cc.get("prior_weight", 0.25)))
    used = 0
    for r in sorted(accepted_records, key=lambda r: r["slot"]):
        if r.get("status") == "accepted" and epoch_of(r["slot"], cfg) <= epoch - lag:
            ctl.update(r); used += 1
    plans = {}
    for slot in range(epoch * E, (epoch + 1) * E):
        if slot >= int(cfg["n_scenes"]):
            break
        u = float(keyed_rng(int(cfg["seed"]), slot, K_STREAM).random())     # = rind3d_1k _rng(seed, slot, 56000, 0)
        k, c, plan = ctl.choose(u, float(cc.get("k_floor", 0.15)))
        ctl.N += np.asarray(plan["expected_composition"])          # provisional, outcome-independent
        plans[slot] = dict(k=k, c_table=c, plan=dict(plan, u=u, epoch=epoch, records_used=used))
    return plans
