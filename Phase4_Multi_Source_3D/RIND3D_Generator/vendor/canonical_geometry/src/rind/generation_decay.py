"""Independent deterministic physics variants for an accepted binary scene."""

from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction

import numpy as np

from .contracts import Scene, make_physics, make_scene
from .decay import LogPchipLaw
from .generation import MASTER_SEED


PROFILE_ID_DECAY = "rind-decay-v3-impl1"


def stream_decay(dimension: int, slot: int, attempt: int, variant: int,
                 name: str, index: int = 0) -> np.random.Generator:
    if dimension not in (2, 3) or slot < 0 or attempt < 0 or variant not in (1, 2):
        raise ValueError("invalid decay stream coordinates")
    if name not in ("strength", "decay", "query"):
        raise ValueError("invalid decay stream name")
    preimage = ["rind-stream-v3", MASTER_SEED, PROFILE_ID_DECAY, dimension,
                slot, attempt, variant, name, index]
    raw = json.dumps(preimage, separators=(",", ":"), ensure_ascii=True).encode()
    seed = int.from_bytes(hashlib.sha256(raw).digest()[:16], "little")
    return np.random.Generator(np.random.PCG64(seed))


def _pchip_physics(dimension: int, slot: int, attempt: int):
    diameter = math.nextafter(math.sqrt(dimension), math.inf)
    exact_diameter = Fraction.from_float(diameter)
    if exact_diameter ** 2 < dimension:
        raise AssertionError("PCHIP diameter does not cover the cube")
    knots = np.asarray([float(exact_diameter * j / 8) for j in range(9)],
                       dtype=np.float64)
    knots[-1] = diameter
    rng = stream_decay(dimension, slot, attempt, 2, "decay")
    strength = .5 + 7.5 * float(rng.random())
    latent = np.empty(8, dtype=np.float64)
    latent[0] = float(rng.normal())
    for i in range(1, 8):
        latent[i] = .8 * latent[i - 1] + .6 * float(rng.normal())
    weights = np.exp(.4 * latent)
    drops = strength * weights / float(np.sum(weights))
    logs = np.concatenate(([0.], -np.cumsum(drops))).astype(np.float64)
    if (not np.all(np.isfinite(logs)) or
            not np.all(logs[1:] < logs[:-1])):
        raise ValueError("sampled PCHIP log drops are not finite and strict")
    physics = make_physics(2, knots=knots, log_values=logs)
    LogPchipLaw.from_physics(physics, dimension)
    return physics


def make_decay_variant(binary_scene: Scene, slot: int, attempt: int,
                       variant: int) -> Scene:
    """Reuse S,U verbatim; variant 1 is exponential, 2 is log-PCHIP."""
    if binary_scene.physics.codec_id != 0 or variant not in (1, 2):
        raise ValueError("decay variants require a binary source scene")
    positions = binary_scene.driver_position.as_array()
    strengths = np.asarray([
        math.exp(math.log(.5) + math.log(4.) *
                 float(stream_decay(binary_scene.dimension, slot, attempt,
                                    variant, "strength", index).random()))
        for index in range(len(positions))], dtype=np.float64)
    if not np.all(np.isfinite(strengths)) or np.any(strengths < .5) or np.any(strengths > 2):
        raise ValueError("sampled driver strength outside log-uniform support")
    physics = (make_physics(1, tau=.25) if variant == 1 else
               _pchip_physics(binary_scene.dimension, slot, attempt))
    return make_scene(binary_scene.geometry_id, binary_scene.dimension,
                      physics, positions, strengths)
