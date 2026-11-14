"""Contract v2 declared sampling law: kappa draw, streams, plans, balance-heuristic weights (no GPU)."""
import json
from pathlib import Path

import numpy as np
import pytest

from rind3d_100k import sampling as SM
from rind3d_100k.controller import plan_epoch

HERE = Path(__file__).resolve().parent.parent
V2 = json.loads((HERE / "contract" / "RIND3D_100K_CONTRACT_v2.json").read_text())
V1 = json.loads((HERE / "contract" / "RIND3D_100K_CONTRACT_v1.json").read_text())


def test_contract_v2_declares_the_recommended_design():
    assert SM.declared(V2) and not SM.declared(V1)
    assert V2["source_count"]["kappa"] == [0.15, 0.15, 0.15, 0.55]
    cov = V2["selection"]["coverage"]
    assert cov["sides"] == [64] and cov["k"] == [4] and cov["slots"] == 1 and cov["c_m"] == [0.125, 0.25, 0.5, 1.0]
    assert V2["labels"]["engine"] == "fastlabel" and V2["labels"]["fastlabel"]["n_tasks"] == 100000 * 12 * 4
    assert "provisional" in V2["status"]


def test_kappa_draw_frequencies_and_determinism():
    ks = np.array([SM.kappa_draw(V2, s)[0] for s in range(20000)])
    f = np.bincount(ks, minlength=5)[1:] / len(ks)
    assert np.all(np.abs(f - np.array(V2["source_count"]["kappa"])) < 0.012)
    assert all(SM.kappa_draw(V2, s) == SM.kappa_draw(V2, s) for s in range(50))


def test_kappa_validation():
    bad = json.loads(json.dumps(V2)); bad["source_count"]["kappa"] = [0.5, 0.5, 0.5, 0.0]
    with pytest.raises(ValueError):
        SM.kappa(bad)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_side_streams(k):
    for L in (64, 128, 256):
        st = SM.side_streams(V2, L, k)
        assert sum(x["n"] for x in st) == 4
        assert st[0]["name"] == "natural" and st[0]["c"] == [1.0] * 4
        if L == 64 and k == 4:
            assert [x["name"] for x in st] == ["natural", "coverage"] and st[0]["n"] == 3 and st[1]["n"] == 1
        else:
            assert len(st) == 1


def test_plan_is_outcome_independent():
    a = plan_epoch(V2, 3, []); b = plan_epoch(V2, 3, [dict(slot=1, status="accepted", k=1, m_hist=[4, 0, 0, 0], sides=[])])
    assert a == b and min(a) == 3000 and max(a) == 3999
    assert all(p["k"] == SM.kappa_draw(V2, s)[0] and p["c_table"] is None for s, p in a.items())


def test_v1_controller_still_plans():
    p = plan_epoch(V1, 0, [])
    assert len(p) == 1000 and all(set(x["c_table"]) == {"64", "128", "256"} for x in p.values())


def _side(rate_m, rate_n, nat_acc=4, nat_prop=10, cov=None):
    sd = dict(rate_n=rate_n, rate_A=int(sum(rate_m)), rate_m=list(rate_m),
              streams=[dict(name="natural", c=[1.0] * 4, accepted=nat_acc, proposals=nat_prop, reasons={})])
    if cov is not None:
        sd["streams"].append(dict(name="coverage", c=[0.125, 0.25, 0.5, 1.0], accepted=cov, proposals=5, reasons={}))
    return sd


def test_weights_natural_only_reduce_to_p_a_over_n():
    sd = _side([60, 30, 20, 10], 256)
    pa = 120 / 256
    for m in (1, 2, 3, 4):
        assert SM.tstar_weight(m, sd, "rate") == pytest.approx(pa / 4)
    assert SM.natural_weight(sd, "rate") == pytest.approx(pa / 4)


def test_weights_balance_heuristic_with_coverage():
    sd = _side([60, 30, 20, 10], 256, nat_acc=3, cov=1)
    pa = 120 / 256; lc = (60 * 0.125 + 30 * 0.25 + 20 * 0.5 + 10 * 1.0) / 256
    for m, c in zip((1, 2, 3, 4), (0.125, 0.25, 0.5, 1.0)):
        assert SM.tstar_weight(m, sd, "rate") == pytest.approx(1.0 / (3 / pa + c / lc))
    # bounded by the pure-natural weight of the 3 natural slots
    assert max(SM.tstar_weight(m, sd, "rate") for m in (1, 2, 3, 4)) <= pa / 3 + 1e-15


def test_unbiased_mixture_expectation():
    """E over the two streams of the weighted indicator equals the natural window law (exact, per m)."""
    p = np.array([60, 30, 20, 10]) / 256.0          # P(admissible, m) per natural proposal
    pa = p.sum(); c = np.array([0.125, 0.25, 0.5, 1.0]); lc = float(np.sum(c * p))
    sd = _side((p * 256).round().astype(int).tolist(), 256, nat_acc=3, cov=1)
    w = np.array([SM.tstar_weight(m, sd, "rate") for m in (1, 2, 3, 4)])
    dens = 3 * p / pa + 1 * c * p / lc               # expected number of stored windows with mixed count m
    assert np.allclose(dens * w, p)                  # = the T* law of m per natural proposal


def test_estimators_positive_and_validated():
    sd = _side([0, 0, 0, 0], 256, nat_acc=4, nat_prop=50)      # rate stream saw nothing admissible
    assert SM.stream_lambda(sd["streams"][0], sd, "rate") == 0.0
    assert SM.stream_lambda(sd["streams"][0], sd, "combined_rb", alpha=0.12) > 0
    with pytest.raises(ValueError):
        SM.stream_lambda(sd["streams"][0], sd, "nope")
