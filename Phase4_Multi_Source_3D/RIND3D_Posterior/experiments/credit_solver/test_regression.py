"""Regression fixtures with exact-certified references (slow: about 1 minute in total).

Each fixture is labelled end to end through the trusted driver; the released bytes are
decoded and compared with the independent exact-kernel label (yushu/3d-v3-staging verifier):
TV(payload, q_ref) <= bound + b_ref must hold.

  000004_w00_s0  archived 214-cell posterior (L = 64)
  000038_w06_s2  first confirmed legacy Tau-pencil defect (legacy label off by TV 0.019)
  000007_w08_s2  second confirmed legacy Tau-pencil defect (legacy off by TV 0.069; L = 256)
  000007_w09_s2  L = 256 penumbra-dense task certified only with the hull step (v2.9c)
  000030_w01_s0  small wedges task used by the kernel tests

Run: PYTHONPATH=... python -m pytest -q test_regression.py
"""
from fractions import Fraction
from pathlib import Path

import pytest

import inputs
from cbc_search import solve
from compare import certified_reference, tv
from ledger import decode

ROOT = Path("/home/yushu/Projects/RIND/RIND3D_100K_v3_work/sampling_impact/v3")
FIXTURES = [("000004_w00_s0", 20), ("000038_w06_s2", 30), ("000007_w08_s2", 40), ("000007_w09_s2", 40),
            ("000030_w01_s0", 20)]


@pytest.mark.parametrize("key,seconds", FIXTURES)
def test_fixture_certifies_consistently(key, seconds):
    slot, w, s = int(key[:6]), int(key[8:10]), int(key[12])
    path = ROOT / "scenes" / f"{slot // 1000:04d}" / f"{slot:06d}.npz"
    ref = certified_reference(key)
    if not path.exists() or ref is None:
        pytest.skip("fixture data not available")
    obs, _ = inputs.load(path, w, s)
    payload, d = solve(obs, key, Fraction(1, 20), seconds)
    assert d["status"] == "CERTIFIED", d.get("reason")
    bound = Fraction(d["tv"])
    assert 0 <= bound <= Fraction(1, 20)
    p = decode(payload)
    assert tv(p, ref["q"]) <= float(bound) + ref["bound"] + 1e-12
    if key == "000004_w00_s0":
        assert len(ref["q"]) == 214 and len(p) >= 200  # CBC credits almost every reference cell
