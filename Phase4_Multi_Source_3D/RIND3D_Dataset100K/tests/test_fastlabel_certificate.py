"""FastLabel v3.4 certificate rules of the 100K campaign (RIND3D_Posterior/docs/FASTLABEL_V34_CERTIFICATION.md §6, §8).
No GPU.

* contract pinning: run options cannot change the FastLabel streams or budget of a contract (no re-roll);
* validate.check_stat_certificate accepts a real v3 production certificate and rejects every single mutation that
  would break the theorem's premises (stream, budget, stage decision, design, q_B arrays);
* the v3 design: the exact (rational) false-certificate probability of every round at the contract budget is <= its
  share delta * 2^-r, at the true boundary eps = 1/20;
* FastLabel refuses curved-occluder containers."""
import copy
import io
import json
from fractions import Fraction
from math import comb
from pathlib import Path

import numpy as np
import pytest

from rind3d_100k.coordinator import fastlabel_config
from rind3d_100k.validate import check_stat_certificate

V3 = json.loads((Path(__file__).resolve().parent.parent / "contract" / "RIND3D_100K_CONTRACT_v3.json").read_text())
FL = V3["labels"]["fastlabel"]

# certificate of a real FastLabel v3.4 run at the v3 production configuration (sampling_impact/v3 scene 000001, window 0,
# channel 0; experiments/fastlabel_v34_validation/results/fresh_astra/fl_runs.jsonl)
META = dict(key="000001:00:0", slot=1, window=0, source=0, status="certified_stat", tv_bound=0.05, engine="fastlabel",
            engine_version="fastlabel-v3.4",
            certificate=dict(kind="statistical", eps=0.05, delta_label=0.01 / 4800000, delta_round=0.01 / 4800000 * 0.5 ** 2,
                             round=2, stage=1, n=512, K=0, a1=1, a2=14, n1=512, n_max=1024, n_tasks=4800000,
                             seed=[20261118, 1, 0, 0]))
LO = np.array([1e-9, 2e-9, 3.5e-10])


def test_fastlabel_options_cannot_override_contract():
    cfg = dict(labels=dict(fastlabel=dict(FL)))
    got = fastlabel_config(cfg, {})
    assert got == {k: FL[k] for k in ("n_tasks", "base_seed", "family_delta", "eps")}
    assert fastlabel_config(cfg, dict(fastlabel=dict(base_seed=FL["base_seed"], n_tasks=None))) == got
    for o in (dict(base_seed=FL["base_seed"] + 1), dict(n_tasks=1000), dict(eps=0.1), dict(family_delta=0.05)):
        with pytest.raises(SystemExit):
            fastlabel_config(cfg, dict(fastlabel=o))
    # a contract without the field (legacy) may take it from the options
    assert fastlabel_config(dict(labels=dict(fastlabel=dict(eps=0.05))), dict(fastlabel=dict(n_tasks=7)))["n_tasks"] == 7


def test_check_stat_certificate_accepts_production_certificate():
    assert check_stat_certificate(META, LO, LO.copy(), FL) == []


def _mutants():
    def m(f):
        x = copy.deepcopy(META); f(x); return x
    c = "certificate"
    yield "base_seed", m(lambda x: x[c]["seed"].__setitem__(0, 20261112)), LO, LO
    yield "seed key", m(lambda x: x[c]["seed"].__setitem__(2, 3)), LO, LO
    yield "delta_label", m(lambda x: x[c].update(delta_label=0.01 / 159802, delta_round=0.01 / 159802 * 0.25)), LO, LO
    yield "n_tasks", m(lambda x: x[c].update(n_tasks=159802)), LO, LO
    yield "delta_round", m(lambda x: x[c].update(delta_round=META[c]["delta_label"] * 0.5)), LO, LO
    yield "stage 1 with K > a1", m(lambda x: x[c].update(K=2)), LO, LO
    yield "stage 2 with n < n_max", m(lambda x: x[c].update(stage=2, K=3)), LO, LO
    yield "design", m(lambda x: x[c].update(a2=15)), LO, LO
    yield "eps", m(lambda x: x[c].update(eps=0.1)), LO, LO
    yield "tv_bound", m(lambda x: x.update(tv_bound=0.0)), LO, LO
    yield "kind", m(lambda x: x[c].update(kind="deterministic")), LO, LO
    yield "no certificate", m(lambda x: x.pop(c)), LO, LO
    yield "lo != hi", META, LO, LO * np.array([1, 2, 1])
    yield "zero mass", META, LO * 0, LO * 0
    yield "nan", META, LO * np.array([1, np.nan, 1]), LO * np.array([1, np.nan, 1])
    yield "engine", m(lambda x: x.update(engine_version="fastlabel-v3.3")), LO, LO


@pytest.mark.parametrize("name,meta,lo,hi", list(_mutants()), ids=[t[0] for t in _mutants()])
def test_check_stat_certificate_rejects_mutations(name, meta, lo, hi):
    assert check_stat_certificate(meta, lo, hi, FL), name


def _alpha(n1, n2, a1, a2, th):
    p1 = [comb(n1, k) * th ** k * (1 - th) ** (n1 - k) for k in range(n1 + 1)]
    m = n2 - n1; c2 = []; s = Fraction(0)
    for k in range(m + 1):
        s += comb(m, k) * th ** k * (1 - th) ** (m - k); c2.append(s)
    tot = sum(p1[:a1 + 1], Fraction(0)) if a1 >= 0 else Fraction(0)
    for k1 in range(a1 + 1, min(a2, n1) + 1):
        tot += p1[k1] * c2[a2 - k1]
    return tot


def test_v3_design_exact_budget():
    from rind3d_posterior.fastlabel.pipeline import DEFAULTS, design2
    delta = float(FL["family_delta"]) / int(FL["n_tasks"])
    n1, n2 = int(DEFAULTS["n_stage1"]), int(DEFAULTS["n_cert"])
    tot = Fraction(0)
    for r in (1, 2, 3, 6, 12):
        d_r = delta * 0.5 ** r
        a1, a2, _ = design2(n1, n2, float(FL["eps"]), d_r)
        al = _alpha(n1, n2, a1, a2, Fraction(1, 20))
        assert al <= Fraction(d_r), r
        tot += al
    assert tot < Fraction(delta)


def test_window_from_100k_refuses_curved_containers(tmp_path):
    from rind3d_posterior.fastlabel import window_from_100k
    p = tmp_path / "curved.npz"
    buf = io.BytesIO()
    np.savez(buf, meta=np.frombuffer(json.dumps(dict(resolution=1024, slot=0)).encode(), np.uint8),
             balls=np.array([[0.5, 0.5, 0.5, 0.1]]), P=np.zeros((0, 1, 4)), K=np.zeros(0, np.int32))
    p.write_bytes(buf.getvalue())
    with pytest.raises(ValueError, match="curved"):
        window_from_100k(p, 0, 0)
