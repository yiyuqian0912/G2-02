"""Declared sampling law of contract v2 (target population T*; RIND3D_StudentResearch/A_TASK_DEFINITION.md §4,
J_COMPLEXITY_COVERAGE.md §3-5).

* **Source count.** K ~ kappa, declared in the contract, drawn per slot from a keyed uniform u (stream 56000),
  independent of the geometry, the sources and the attempt. No controller and no dependence on other scenes.
* **Natural stream.** Every admissible proposal is accepted (c = 1). No m-dependent thinning.
* **Coverage stream (optional).** Only at the listed sides and for the listed K: `slots` of the side's quota come from
  a separate keyed stream that accepts an admissible proposal with probability c(m). If its cap is reached, the slot
  is filled from the natural stream and recorded as `natural_fallback`.
* **Rate stream.** R natural proposals per side with exact m, so that P_A = P(admissible) and the coverage acceptance
  probability Lambda_cov = sum_m c(m) P(admissible, m) are estimable (rate_A / R, sum_m c(m) rate_m / R).
* **T* weight** of a stored window (balance heuristic, per natural proposal): 1 / sum_s n_s c_s(m) / Lambda_s.

Contract v1 (adaptive controller, m-balanced thinning) is still readable and runnable for regression
(`controller.plan_epoch`)."""
import numpy as np

from rind3d.geometry import keyed_rng

SIDES = (64, 128, 256)
K_STREAM = 56000                                  # same key as the v1 controller's forced-k draw
COV_STREAM = {"quota": 58000, "thin": 59000}      # coverage origins and U (rind3d_1k.physical.STREAM uses 51001-55000)
STREAM_ID = {"natural": 0, "coverage": 1, "natural_fallback": 2}


def declared(cfg):
    return (cfg.get("source_count") or {}).get("law") == "declared"


ELIGIBILITY = {"mixed": "mixed", "direct_interface": "direct_interface"}


def eligibility(cfg):
    """Window eligibility rule of the contract: "mixed" (v1/v2: an active and an inactive FREE receiver) or
    "direct_interface" (v3: two 6-adjacent FREE receivers with different responses; rind3d_100k.interface)."""
    r = (cfg.get("eligibility") or {}).get("rule", "mixed")
    if r not in ELIGIBILITY:
        raise ValueError(f"unknown eligibility rule {r!r}")
    return r


def epoch_size(cfg):
    return int((cfg.get("planning") or cfg.get("controller"))["epoch_scenes"])


def kappa(cfg):
    kp = np.asarray(cfg["source_count"]["kappa"], float)
    if kp.shape != (4,) or np.any(kp < 0) or abs(kp.sum() - 1.0) > 1e-9:
        raise ValueError(f"kappa must be 4 probabilities summing to 1, got {kp.tolist()}")
    return kp


def kappa_draw(cfg, slot):
    """(k, u) of a slot: k = 1 + #{i < 3 : u >= cumsum(kappa)[i]}."""
    u = float(keyed_rng(int(cfg["seed"]), int(slot), K_STREAM).random())
    cum = np.cumsum(kappa(cfg))[:3]
    return 1 + int(np.sum(u >= cum)), u


def side_streams(cfg, Lp, k):
    """Window streams of one side of a k-source scene: [dict(name, n, c[4], cap)]."""
    q = int(cfg["quota"][str(int(Lp))]); cap = int(cfg["cap"][str(int(Lp))])
    cov = (cfg.get("selection") or {}).get("coverage") or {}
    nat = dict(name="natural", n=q, c=[1.0, 1.0, 1.0, 1.0], cap=cap)
    if cov.get("enabled") and int(Lp) in [int(x) for x in cov["sides"]] and int(k) in [int(x) for x in cov["k"]]:
        n_cov = int(cov.get("slots", 1))
        c = [float(x) for x in cov["c_m"]]
        if len(c) != 4 or not all(0 < x <= 1 for x in c):
            raise ValueError("coverage c_m must be 4 probabilities in (0, 1]")
        if not 0 < n_cov < q:
            raise ValueError("coverage slots must leave at least one natural slot")
        nat["n"] = q - n_cov
        return [nat, dict(name="coverage", n=n_cov, c=c, cap=int(cov.get("cap", 2000)))]
    return [nat]


def plan_epoch_declared(cfg, epoch):
    """{slot: {"k", "c_table": None, "plan"}} for every slot of the epoch (independent of every outcome)."""
    E = epoch_size(cfg); plans = {}
    for slot in range(epoch * E, min((epoch + 1) * E, int(cfg["n_scenes"]))):
        k, u = kappa_draw(cfg, slot)
        plans[slot] = dict(k=k, c_table=None, plan=dict(rule="declared kappa", u=u, epoch=epoch,
                                                        kappa=[float(x) for x in kappa(cfg)],
                                                        streams={str(L): side_streams(cfg, L, k) for L in SIDES}))
    return plans


# ------------------------------------------------------------------ weights (used by the loader and the tests)
def stream_lambda(stream, side, estimator="combined_rb", alpha=None):
    """Acceptance probability per natural proposal of one stream of one scene side.
    natural: P_A by "rate" (rate_A / rate_n), "quota" ((r - 1) / (n - 1) of the natural stream), "combined" /
    "combined_rb" ((1 - alpha) rate + alpha quota; strictly positive when r >= 2). coverage: sum_m c(m) rate_m / rate_n
    (Rao-Blackwellised; the only estimator, since a coverage stream accepts a single window)."""
    rate_n = int(side["rate_n"])
    rate_m = np.asarray(side["rate_m"], float)
    c = np.asarray(stream["c"], float)
    rb = float(np.sum(c * rate_m) / rate_n)
    if stream["name"] == "coverage":
        return rb
    rate = float(side["rate_A"]) / rate_n
    if estimator in ("rate", "rb"):
        return rate
    r = int(stream["accepted"]); n = int(stream["proposals"]) - int(stream.get("reasons", {}).get("repeat", 0))
    quota = (r - 1) / (n - 1) if (n > 1 and r > 1) else None
    if estimator == "quota":
        if quota is None:
            raise ValueError("quota estimator undefined (fewer than 2 natural accepts)")
        return quota
    if estimator in ("combined", "combined_rb"):
        if quota is None:
            return rate
        a = float(alpha if alpha is not None else 0.1)
        return (1 - a) * rate + a * quota
    raise ValueError(f"unknown estimator {estimator!r}")


def tstar_weight(m, side, estimator="combined_rb", alpha=None):
    """Balance-heuristic T* weight (per natural proposal) of a window with mixed count m >= 1 of this scene side."""
    den = 0.0
    for st in side["streams"]:
        n = int(st["accepted"])
        if n == 0:
            continue
        lam = stream_lambda(st, side, estimator, alpha)
        cm = float(st["c"][int(m) - 1])
        if cm == 0:
            continue
        if lam <= 0:
            return 0.0                      # an estimated zero acceptance probability: weight undefined -> 0
        den += n * cm / lam
    return 1.0 / den if den > 0 else 0.0


def natural_weight(side, estimator="combined_rb", alpha=None):
    """T* weight of a natural-stream window when only the natural stream is used (population Tstar_natural)."""
    nat = [st for st in side["streams"] if st["name"] == "natural"][0]
    n = int(nat["accepted"])
    return stream_lambda(nat, side, estimator, alpha) / n if n else 0.0
