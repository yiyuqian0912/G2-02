"""The certified production profile admits only polyhedral families with a certified labeller (Task 3 boundary)."""
import json
from pathlib import Path

import pytest

from rind3d_100k.profile import check_certified_profile, UnsupportedProfile, CERTIFIED_FAMILIES

C = Path(__file__).resolve().parent.parent / "contract"


def test_v3_production_contract_is_certified_profile():
    cfg = json.loads((C / "RIND3D_100K_CONTRACT_v3.json").read_text())
    assert check_certified_profile(cfg) == "fastlabel"
    assert set(cfg["families"]) == set(CERTIFIED_FAMILIES)


@pytest.mark.parametrize("name", ["RIND3D_100K_CONTRACT_v1.json", "RIND3D_100K_CONTRACT_v2.json"])
def test_regression_contracts_still_admitted(name):
    cfg = json.loads((C / name).read_text())
    assert check_certified_profile(cfg) in ("fastlabel", "tpe")


def test_curved_research_contract_refused():
    cfg = json.loads((C / "RIND3D_100K_CURVED_RESEARCH_v0.json").read_text())
    with pytest.raises(UnsupportedProfile):
        check_certified_profile(cfg)


def test_curved_family_injected_into_production_contract_refused():
    cfg = json.loads((C / "RIND3D_100K_CONTRACT_v3.json").read_text())
    cfg["families"] = cfg["families"] + ["spheres"]
    with pytest.raises(UnsupportedProfile):
        check_certified_profile(cfg)
    with pytest.raises(UnsupportedProfile):
        check_certified_profile(dict(cfg, families=CERTIFIED_FAMILIES), label_engine="xlf")
