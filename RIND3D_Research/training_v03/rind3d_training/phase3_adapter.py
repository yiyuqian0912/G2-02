"""Lightweight adapter: v0.3 dataset windows -> Phase 3 Teacher objects (Phase 3 code imported read-only).

  scene_for_phase3(scene_ref)                 -> rind_phase3.data.TeacherScene (privileged geometry handle)
  observation_for_phase3(scene_ref, o, L, id) -> Phase 3 observation dict {scene_id, window_id, origin, side,
                                                 observed_response, local_solid_mask} (PackedWindow, ZYX packed X)
  oracle_for_phase3(scene_ref)                -> rind_phase3.oracle.PhysicalOracle
  phase3_full_cost(oracle, obs, xyz)          -> Phase 3 full-window cost, guarded for no-FREE windows
  phase3_sampled_cost(oracle, obs, xyz, k, seed) -> Phase 3 SparsePhysicalCost with its probe policy reported
  additive_response(oracle, sources, strengths, receivers) -> MA response from Phase 3 per-source visibility
The adapter never writes into the Phase 3 tree (PYTHONDONTWRITEBYTECODE=1 in run.sh, no file outputs). The
observation's scene_id field carries the opaque sample id, not the real scene id: Phase 3 student samples must
not carry label-committing identifiers (v0.2 scene ids are SHA256 over the true sources).
SB only for the Phase 3 window cost (Phase 3 rerender_window accepts exactly one source).
"""
import sys

import numpy as np

from .env import PHASE3


def _phase3():
    src = str(PHASE3/"src")
    if src not in sys.path:
        sys.path.insert(0, src)
    import rind_phase3.coordinates as C
    import rind_phase3.data as D
    import rind_phase3.oracle as O
    import rind_phase3.physics as P
    return C, D, O, P


def scene_for_phase3(sc):
    _, D, _, _ = _phase3()
    return D.TeacherScene(sc.scene_id, sc.geometry_id, sc.split, sc.geometry)


def observation_for_phase3(sc, origin, L, sample_id):
    from rind3d.pack import RINDPack
    if sc.K != 1:
        raise ValueError("Phase 3 observations are single-source binary (SB); MA uses additive_response")
    with RINDPack(sc.responses[0]) as y, RINDPack(sc.solid[0]) as m:
        resp = y.read_window(*origin, L, packed=True)
        solid = m.read_window(*origin, L, packed=True)
    return {"scene_id": f"opaque-{int(sample_id):016x}", "window_id": 0, "origin": np.array(origin, np.int64),
            "side": int(L), "observed_response": resp, "local_solid_mask": solid}


def oracle_for_phase3(sc, **kw):
    _, _, O, _ = _phase3()
    return O.PhysicalOracle(scene_for_phase3(sc), **kw)


def phase3_window(obs):
    C, _, _, _ = _phase3()
    return C.Window(tuple(int(v) for v in obs["origin"]), int(obs["side"]))


def phase3_full_cost(oracle, obs, xyz):
    _, _, _, P = _phase3()
    if not (~obs["local_solid_mask"].unpack()).any():
        return {"status": "no_free", "cost": float("nan"),
                "note": "Phase 3 full_window_cost divides by the FREE count: on a no-FREE window it silently returns NaN (numpy RuntimeWarning, no status)"}
    r = P.full_window_cost(oracle, obs, xyz)
    return {"status": "ok", **r}


def phase3_sampled_cost(oracle, obs, xyz, count=256, seed=0):
    _, _, _, P = _phase3()
    if not (~obs["local_solid_mask"].unpack()).any():
        return {"status": "no_free", "cost": float("nan")}
    probes = P.sample_probes(obs, count=count, seed=seed)
    costs, res = P.SparsePhysicalCost(oracle, probes).evaluate(xyz)
    return {"status": "ok", "cost": costs, "probes": len(probes.observed), "probe_policy": probes.policy,
            "uniform_design": "fallback" not in probes.policy}


def additive_response(oracle, sources, strengths, receivers):
    """MA semantics on top of the Phase 3 single-source oracle: sum_k a_k lit_k (binary64, order given)."""
    r = oracle.evaluate_points(np.asarray(sources, np.float64).reshape(-1, 3), receivers)
    if not r.validity.valid.all():
        raise ValueError(f"illegal source(s): {r.validity.reasons}")
    total = np.zeros(r.response.shape[1])
    for k, a in enumerate(np.asarray(strengths, np.float64)):
        total = total + np.where(r.response[k] == 1, float(a), 0.0)
    return total, r
