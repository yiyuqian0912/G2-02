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


@pytest.mark.parametrize("engine", ["cpu", "gpu"])
def test_record_buffer_overflow_reruns_exactly(engine):
    """A record buffer far too small must never be overrun: the engine reports the exact need and the driver reruns,
    giving the bit-identical result (v3.1 guard; v3 wrote past the buffer, as in 1K release scene 000001)."""
    if engine == "gpu":
        pytest.importorskip("cupy")
    case = max(CASES, key=lambda c: c.get("n_records", 0)) if any("n_records" in c for c in CASES) else CASES[1]  # 8678 records
    kw = dict(gpu_min=64, gpu_prefix=8) if engine == "gpu" else {}
    r = RP.label_window(*_window(case), engine=engine, max_records=16, **kw)
    assert _digest(r) == case["digest"]
    assert r["stats"]["record_reruns"] >= 1
    assert r["record_reruns"][0]["max_records"] == 16 and r["record_reruns"][-1]["required"] == len(r["cell"])
