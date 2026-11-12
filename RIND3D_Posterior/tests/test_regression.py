"""Exactness regression: the packaged engines reproduce the Phase-2 records bit for bit (sha256 of cell, lo, hi,
kind) on three Stage-A windows, and the CPU reference and the GPU pass agree."""
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import rind3d_posterior as RP

CASES = json.loads((Path(__file__).parent / "regression_windows.json").read_text())


def _digest(r):
    h = hashlib.sha256()
    for k in ("cell", "lo", "hi", "kind"):
        h.update(np.ascontiguousarray(r[k]).tobytes())
    return h.hexdigest()[:16]


def _window(c):
    cls = np.frombuffer(base64.b64decode(c["cls_b64"]), np.uint8).reshape(c["shape"])
    return np.array(c["P"]), np.array(c["K"]), cls, np.array(c["origin"]), c["N"]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_cpu_reference_digest(case):
    r = RP.label_window(*_window(case), engine="cpu")
    assert _digest(r) == case["digest"]
    assert r["tv_bound"] <= 0.05


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_gpu_pass_matches_reference(case):
    pytest.importorskip("cupy")
    r = RP.label_window(*_window(case), engine="gpu", gpu_min=64, gpu_prefix=8)
    assert _digest(r) == case["digest"]


def test_point_estimate_normalised():
    r = RP.label_window(*_window(CASES[0]), engine="cpu")
    cells, q, tv = RP.point_estimate(r)
    assert abs(q.sum() - 1) < 1e-12 and tv <= 0.05
