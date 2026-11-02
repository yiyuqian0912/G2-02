"""Explicit sampled mismatch versus complete-window mismatch; no source labels."""

from dataclasses import dataclass
import numpy as np
from .coordinates import Window


@dataclass(frozen=True)
class Probes:
    world_xyz: np.ndarray
    observed: np.ndarray
    window: Window
    policy: str


def sample_probes(observation, *, count=64, seed=0):
    if count < 1:
        raise ValueError("Positive probe count required")
    window = Window(tuple(observation["origin"]), observation["side"])
    L = window.side
    solid = observation["local_solid_mask"].data
    response = observation["observed_response"].data
    rng = np.random.default_rng(seed)
    positions = {}
    attempts = 0

    def accept(flat):
        z, j, x = np.unravel_index(int(flat), (L, L, L))
        if not solid[z, j, x // 8] & (1 << (x % 8)):
            positions[int(flat)] = (int(x), int(j), int(z))

    while len(positions) < count and attempts < max(1024, count * 64):
        accept(rng.integers(L**3))
        attempts += 1
    policy = "uniform FREE-voxel rejection without replacement"
    if len(positions) < count:
        policy += "; bounded rare-free fallback: deterministic packed-byte scan"
        for byte in np.flatnonzero((~solid).ravel()):
            z, j, bx = np.unravel_index(int(byte), solid.shape)
            for bit in range(8):
                if len(positions) >= count:
                    break
                accept((z * L + j) * L + bx * 8 + bit)
            if len(positions) >= count:
                break
    local = np.asarray(list(positions.values()), dtype=np.int64)
    if not len(local):
        raise ValueError("Observation has no FREE receiver")
    x, j, z = local.T
    labels = ((response[z, j, x // 8] >> (x % 8)) & 1).astype(np.int8)
    return Probes(window.centres(local), labels, window, policy)


class SparsePhysicalCost:
    def __init__(self, oracle, probes):
        self.oracle = oracle
        self.probes = probes

    def evaluate(self, candidate_xyz):
        result = self.oracle.evaluate_points(
            candidate_xyz, self.probes.world_xyz, window=self.probes.window
        )
        costs = np.full(len(result.validity.valid), np.nan)
        valid = result.validity.valid
        costs[valid] = np.mean(result.response[valid] != self.probes.observed, axis=1)
        return costs, result


def full_window_cost(oracle, observation, candidate_xyz):
    w = Window(tuple(observation["origin"]), observation["side"])
    r = oracle.rerender_window(candidate_xyz, w)
    expected = observation["observed_response"].unpack()
    solid = observation["local_solid_mask"].unpack()
    if not np.array_equal(r.solid, solid):
        raise ValueError("Observation and oracle geometry/voxel conventions disagree")
    free = ~solid
    return {
        "cost": float(
            np.count_nonzero((r.response != expected) & free) / np.count_nonzero(free)
        ),
        "scope": "full-window FREE-voxel disagreement; one certified backend",
        "equivalent_source_certification": False,
        "response": r,
    }
