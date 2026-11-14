"""Dataset-level effective-layer balancing controller (docs/SELECTION_LAW_1K.md).

Before scene t, the controller chooses the source count k and the thinning table c(m, L, k) from two inputs only:
* the cumulative histogram of accepted windows by m;
* pooled natural estimates p_hat(m | L, k) and P_hat(A | L, k) from the rate streams of PREVIOUS scenes.

It never looks at the sources of scene t, so the per-channel posterior stays Uniform(F_W^j) (Theorem, section 2).
Rule:
* the target is equal counts per m;
* for each candidate k, the desired within-side composition is d(m) proportional to max(eps, deficit(m)) over
  m <= k;
* c(m) is proportional to d(m) / p_hat(m), scaled so that max c = 1 and clipped to [c_min, 1];
* k is chosen to minimise the expected distance from uniform after the scene, with the expected proposal count
  as a tie-breaker.
"""
import json
import numpy as np

SIDES = (64, 128, 256)
QUOTA = {64: 16, 128: 24, 256: 24}
C_MIN = 0.01


class Controller:
    def __init__(self, prior_counts=None, pseudo=2.0):
        # pooled natural counts n[L][k][m] of A-windows (rate streams) and A-rates; start from a weak prior
        self.nA = {L: {k: np.full(k, pseudo, float) for k in range(1, 5)} for L in SIDES}
        self.rate = {L: {k: [1.0, 2.0] for k in range(1, 5)} for L in SIDES}   # (accepted A, proposals) pseudo
        self.N = np.zeros(4)
        if prior_counts:
            for L, d in prior_counts.items():
                for k, v in d.items():
                    self.nA[int(L)][int(k)] += np.asarray(v, float)

    def p_hat(self, L, k):
        v = self.nA[L][k]
        return v / v.sum()

    def plan_k(self, k, eps=0.5):
        tot = self.N.sum() + 64
        deficit = np.maximum(eps, tot / 4 - self.N)[:k]
        c, comp, props = {}, np.zeros(4), 0.0
        for L in SIDES:
            p = self.p_hat(L, k)
            d = deficit / deficit.sum()
            cc = d / np.maximum(p, 1e-9)
            cc = np.clip(cc / cc.max(), C_MIN, 1.0)
            pi = p * cc / np.sum(p * cc)
            comp[:k] += QUOTA[L] * pi
            a, n = self.rate[L][k]
            PA = a / n
            props += QUOTA[L] / max(PA * np.sum(p * cc), 1e-9)
            c[str(L)] = [float(x) for x in cc] + [1.0] * (4 - k)
        return c, comp, props

    def choose(self, u=None, floor=0.10):
        """u: a keyed Uniform(0,1) of the slot (independent of the sources).  With probability 4*floor, k is forced
        to 1 + floor(u / floor) (source-count diversity); otherwise k minimises the expected imbalance."""
        if u is not None and u < 4 * floor:
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

    def update(self, scene_rec):
        """scene_rec: the physical-stage record (accepted scenes) with m_hist and per-side rate-stream counts."""
        k = scene_rec["k"]
        self.N += np.asarray(scene_rec["m_hist"], float)
        for s in scene_rec["sides"]:
            L = int(s["Lp"])
            self.nA[L][k] += np.asarray(s["rate_m"][:k], float)
            self.rate[L][k][0] += s["rate_A"]
            self.rate[L][k][1] += 1024

    def state(self):
        return dict(N=self.N.tolist(), nA={str(L): {str(k): v.tolist() for k, v in d.items()} for L, d in self.nA.items()},
                    rate={str(L): {str(k): v for k, v in d.items()} for L, d in self.rate.items()})
