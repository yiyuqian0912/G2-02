"""Deterministic A/E leaf proposals for the design-only expanded profile.

This module does not create READY candidates. B/C/D proposal integration,
reload gates and certified acceptance still have to be wired up.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import product
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from ._exact2d import convex_hull_indices
from .contracts import (CSGNode, CSGOpcode, CodecID, Geometry, Leaf,
                        RootKind, make_leaf, make_physics, make_scene)
from .generation import (CandidateInvalid, CandidateUnresolved, _exact_det, _leaf_box_or_ball,
                         _ring_valid, _transform_2d)
from .generation3d import (_box_or_ball, _exact_det3, _hull_leaf,
                           _transform_3d)
from .geometry import Clearance, Membership, QueryStatus
from .oracle import PrivateOracle
from .storage import CanonicalBundle, _decode_bundle, pack_arrays


PROFILE_PATH = Path(__file__).resolve().parents[2] / "profiles" / "rind-expanded-v3-impl1.json"


@lru_cache(maxsize=1)
def _profile() -> dict:
    profile = json.loads(PROFILE_PATH.read_text())
    if (profile["profile_id"] != "rind-expanded-v3-impl1" or
            profile["rng_engine"] != "NumPy-PCG64" or
            profile["stream_coordinates"] != [
                "domain", "master_seed", "profile_id", "dimension", "slot",
                "attempt", "variant", "object_index", "stream_name",
                "feature_index"]):
        raise ValueError("expanded generator/profile stream contract differs")
    return profile


def expanded_stream(dimension: int, slot: int, attempt: int, object_index: int,
                    name: str, feature_index: int = 0) -> np.random.Generator:
    if (dimension not in (2, 3) or min(slot, attempt, object_index,
                                         feature_index) < 0):
        raise ValueError("invalid expanded stream coordinate")
    profile = _profile()
    preimage = [profile["stream_preimage"], profile["master_seed"],
                profile["profile_id"], dimension, slot, attempt, 0,
                object_index, name, feature_index]
    raw = json.dumps(preimage, separators=(",", ":"), ensure_ascii=True).encode()
    seed = int.from_bytes(hashlib.sha256(raw).digest()[:16], "little")
    return np.random.Generator(np.random.PCG64(seed))


def sample_family(dimension: int, slot: int, attempt: int,
                  object_index: int) -> str:
    """Choose A/B/C/D/E using exact normalized stored binary64 weights."""
    profile = _profile()
    weights = [(name, Fraction.from_float(float.fromhex(
        profile["family_probabilities_hex"][name]))) for name in "ABCDE"]
    draw = Fraction.from_float(float(expanded_stream(
        dimension, slot, attempt, object_index, "family").random()))
    target = draw * sum((weight for _, weight in weights), Fraction(0))
    total = Fraction(0)
    for name, weight in weights:
        total += weight
        if target < total:
            return name
    raise AssertionError("expanded family weights do not cover the draw")


def _uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return low + (high - low) * float(rng.random())


def _direction(dimension: int, slot: int, attempt: int,
               object_index: int, name: str, feature_index: int) -> np.ndarray:
    normal = expanded_stream(dimension, slot, attempt, object_index,
                             name, feature_index).normal(size=dimension)
    length = math.sqrt(sum(float(value)**2 for value in normal))
    if length == 0 or not math.isfinite(length):
        raise ValueError("zero or nonfinite expanded frequency direction")
    return np.asarray(normal / length, dtype=np.float64)


def _affine(dimension: int, slot: int, attempt: int,
            object_index: int) -> tuple[np.ndarray, np.ndarray]:
    rng = expanded_stream(dimension, slot, attempt, object_index,
                          "object_transform")
    center, matrix = (_transform_2d(rng) if dimension == 2
                      else _transform_3d(rng))
    determinant = (_exact_det(matrix) if dimension == 2
                   else _exact_det3(matrix))
    if determinant == 0:
        raise CandidateInvalid("SINGULAR_TRANSFORM")
    return center, matrix


def sample_a_leaf(dimension: int, slot: int, attempt: int,
                  object_index: int) -> Leaf:
    """Sample one radial Fourier family-A leaf; store actual binary64 values."""
    profile = _profile()
    center, matrix = _affine(dimension, slot, attempt, object_index)
    if dimension == 2:
        recipe = profile["family_A_2d"]
    else:
        recipe = profile["family_A_3d"]
    sigma = _uniform(expanded_stream(dimension, slot, attempt, object_index,
                                     "A_sigma"),
                     *(float.fromhex(value) for value in recipe["sigma_uniform_hex"]))
    if dimension == 2:
        ell = _uniform(expanded_stream(2, slot, attempt, object_index, "A_ell"),
                       *(float.fromhex(value) for value in recipe["ell_uniform_hex"]))
        harmonics = int(recipe["harmonics"])
        coeff_a = []
        coeff_b = []
        for index in range(1, harmonics + 1):
            scale = sigma * math.exp(-.5 * (index * ell)**2)
            coeff_a.append(float(expanded_stream(
                2, slot, attempt, object_index, "A_coeff_a", index).normal()) * scale)
            coeff_b.append(float(expanded_stream(
                2, slot, attempt, object_index, "A_coeff_b", index).normal()) * scale)
        return make_leaf(CodecID.RADIAL_FOURIER_2D, 2, center=center,
                         matrix=matrix, coeff_a=coeff_a, coeff_b=coeff_b)
    features = int(recipe["features"])
    low, high = (float.fromhex(value) for value in
                 recipe["frequency_magnitude_loguniform_hex"])
    frequency = []
    coeff_a = []
    coeff_b = []
    for index in range(features):
        direction = _direction(3, slot, attempt, object_index,
                               "A_direction", index)
        magnitude = math.exp(_uniform(expanded_stream(
            3, slot, attempt, object_index, "A_magnitude", index),
            math.log(low), math.log(high)))
        frequency.append(np.asarray(direction * magnitude, dtype=np.float64))
        coeff_a.append(float(expanded_stream(
            3, slot, attempt, object_index, "A_coeff_a", index).normal()) *
            sigma / math.sqrt(features))
        coeff_b.append(float(expanded_stream(
            3, slot, attempt, object_index, "A_coeff_b", index).normal()) *
            sigma / math.sqrt(features))
    return make_leaf(CodecID.RADIAL_FOURIER_SPHERE, 3, center=center,
                     matrix=matrix, frequency=np.asarray(frequency),
                     coeff_a=coeff_a, coeff_b=coeff_b)


def sample_e_leaf(dimension: int, slot: int, attempt: int,
                  object_index: int) -> Leaf:
    """Sample an ambient cosine field independent of occupancy or response."""
    profile = _profile()
    recipe = profile["family_E"]
    features = int(recipe["features"])
    low, high = (float.fromhex(value) for value in
                 recipe["frequency_magnitude_loguniform_hex"])
    phase_low, phase_high = (float.fromhex(value) for value in
                             recipe["phase_uniform_hex"])
    threshold_low, threshold_high = (float.fromhex(value) for value in
                                     recipe["threshold_uniform_hex"])
    frequency = []
    amplitude = []
    phase = []
    for index in range(features):
        direction = _direction(dimension, slot, attempt, object_index,
                               "E_direction", index)
        magnitude = math.exp(_uniform(expanded_stream(
            dimension, slot, attempt, object_index, "E_magnitude", index),
            math.log(low), math.log(high)))
        frequency.append(np.asarray(direction * magnitude, dtype=np.float64))
        amplitude.append(float(expanded_stream(
            dimension, slot, attempt, object_index, "E_amplitude", index).normal()) /
            math.sqrt(features))
        phase.append(_uniform(expanded_stream(
            dimension, slot, attempt, object_index, "E_phase", index),
            phase_low, phase_high))
    threshold = _uniform(expanded_stream(dimension, slot, attempt,
                                         object_index, "E_threshold"),
                         threshold_low, threshold_high)
    return make_leaf(CodecID.COSINE_FIELD, dimension,
                     frequency=np.asarray(frequency), amplitude=amplitude,
                     phase=phase, threshold=threshold)


def _sample_bcd_object(bundle: CanonicalBundle, family: str, dimension: int,
                       slot: int, attempt: int, object_index: int) -> bytes:
    center, matrix = _affine(dimension, slot, attempt, object_index)
    rng = expanded_stream(dimension, slot, attempt, object_index, "BCD_recipe")
    if family == "B":
        box = float(rng.random()) < .5
        return (_leaf_box_or_ball(bundle, center, matrix, box)
                if dimension == 2 else
                _box_or_ball(bundle, center, matrix, box=box))
    if family == "C":
        if dimension == 3:
            return _hull_leaf(bundle, center, matrix, rng)
        hull = float(rng.random()) < .5
        count = int(rng.integers(5, 13))
        vertices = []
        if hull:
            for _ in range(count):
                theta = 2 * math.pi * float(rng.random())
                radius = math.sqrt(float(rng.random()))
                vertices.append((radius * math.cos(theta),
                                 radius * math.sin(theta)))
            try:
                indices = convex_hull_indices(vertices)
            except ValueError as error:
                raise CandidateInvalid("DEGENERATE_RING") from error
            vertices = [vertices[index] for index in indices]
        else:
            for index in range(count):
                theta = 2 * math.pi * (index + float(rng.random())) / count
                radius = .35 + .65 * float(rng.random())
                vertices.append((radius * math.cos(theta),
                                 radius * math.sin(theta)))
        vertices_array = np.asarray(vertices, dtype=np.float64)
        _ring_valid(vertices_array)
        leaf = make_leaf(
            CodecID.POLYGON_RINGS_2D, 2, center=center, matrix=matrix,
            vertices=vertices_array,
            ring_vertex_offsets=np.asarray([0, len(vertices)], dtype=np.uint64),
            ring_role=np.asarray([0], dtype=np.uint8),
            ring_owner=np.asarray([2**32 - 1], dtype=np.uint32),
        )
        bundle.add_leaf(leaf)
        return leaf.content_hash
    if family != "D":
        raise ValueError("expected B, C or D family")
    outer = (_leaf_box_or_ball(bundle, center, matrix, True)
             if dimension == 2 else
             _box_or_ball(bundle, center, matrix, box=True))
    if float(rng.random()) < .5:
        scales = np.asarray([_uniform(rng, .4, .9)
                             for _ in range(dimension)])
        offset = np.asarray([_uniform(rng, -.8, .8)
                             for _ in range(dimension)])
        opcode = CSGOpcode.UNION
    else:
        if float(rng.random()) < .5:
            scales = np.asarray([_uniform(rng, .2, .7)
                                 for _ in range(dimension)])
            offset = np.asarray([_uniform(rng, -.1, .1)
                                 for _ in range(dimension)])
        else:
            axis = int(rng.integers(0, dimension))
            sign = -1 if int(rng.integers(0, 2)) == 0 else 1
            scales = np.asarray([_uniform(rng, .2, .6)
                                 for _ in range(dimension)])
            offset = np.asarray([_uniform(rng, -.1, .1)
                                 for _ in range(dimension)])
            scales[axis] = 1
            offset[axis] = sign * .6
        opcode = CSGOpcode.DIFFERENCE
    secondary_center = np.asarray(center + matrix @ offset, dtype=np.float64)
    secondary_matrix = np.asarray(matrix @ np.diag(scales), dtype=np.float64)
    secondary = (_leaf_box_or_ball(bundle, secondary_center,
                                   secondary_matrix, True)
                 if dimension == 2 else
                 _box_or_ball(bundle, secondary_center,
                              secondary_matrix, box=True))
    node = CSGNode(dimension, opcode, (outer, secondary))
    bundle.add_node(node)
    return node.content_hash


def propose_expanded_geometry(dimension: int, slot: int,
                              attempt: int) -> tuple[CanonicalBundle, Geometry]:
    """Sample the full A/B/C/D/E mixture; no READY claim is made here."""
    profile = _profile()
    if dimension not in (2, 3) or slot < 0 or attempt < 0:
        raise ValueError("invalid expanded candidate coordinate")
    count = int(expanded_stream(dimension, slot, attempt, 0,
                                "structure_object_count").integers(
        profile["object_count_inclusive"][0],
        profile["object_count_inclusive"][1] + 1))
    bundle = CanonicalBundle(dimension)
    roots = []
    for object_index in range(count):
        family = sample_family(dimension, slot, attempt, object_index)
        if family == "A":
            leaf = sample_a_leaf(dimension, slot, attempt, object_index)
            bundle.add_leaf(leaf)
            root = leaf.content_hash
        elif family == "E":
            leaf = sample_e_leaf(dimension, slot, attempt, object_index)
            bundle.add_leaf(leaf)
            root = leaf.content_hash
        else:
            root = _sample_bcd_object(bundle, family, dimension,
                                      slot, attempt, object_index)
        roots.append(root)
    if not roots:
        geometry = Geometry.empty(dimension)
    else:
        if len(roots) == 1:
            root = roots[0]
        else:
            node = CSGNode(dimension, CSGOpcode.UNION, tuple(roots))
            bundle.add_node(node)
            root = node.content_hash
        geometry = Geometry(dimension, RootKind.REF, root)
    bundle.add_geometry(geometry)
    reloaded = _decode_bundle(pack_arrays(bundle), dimension)
    if reloaded.geometries[0].content_hash != geometry.content_hash:
        raise AssertionError("expanded proposal changed during typed SoA reconstruction")
    return reloaded, reloaded.geometries[0]


@dataclass(frozen=True, slots=True)
class ExpandedDriverSample:
    positions: np.ndarray
    proposals_per_driver: tuple[int, ...]


def sample_expanded_drivers(view, slot: int, attempt: int,
                            proposal_limit: int | None = None) -> ExpandedDriverSample:
    """Uniform conditional driver proposals with no uncertainty-as-rejection."""
    profile = _profile()
    dimension = view.dimension
    if dimension not in (2, 3) or slot < 0 or attempt < 0:
        raise ValueError("invalid expanded driver coordinate")
    limit = (int(profile["driver_proposal_limit"]) if proposal_limit is None
             else int(proposal_limit))
    if limit < 0:
        raise ValueError("negative driver proposal limit")
    low, high = profile["driver_count_inclusive"]
    count = int(expanded_stream(dimension, slot, attempt, 0,
                                "driver_count").integers(low, high + 1))
    if view.is_full():
        raise CandidateInvalid("NO_ALLOWED_DRIVER_PROVED")
    positions = []
    counts = []
    clearance_radius = float.fromhex(profile["driver_clearance_hex"])
    for driver_index in range(count):
        rng = expanded_stream(dimension, slot, attempt, 0,
                              "driver_position", driver_index)
        for proposal_index in range(limit):
            point = rng.random(dimension).astype(np.float64)
            classification = view.classify_points(point)
            if classification.status[0] in (QueryStatus.UNSUPPORTED,
                                            QueryStatus.UNRESOLVED):
                raise CandidateUnresolved("DRIVER_MEMBERSHIP_UNPROVED")
            if classification.value[0] != Membership.FREE:
                continue
            clearance = view.clearance_decide(point, clearance_radius)
            if clearance.status[0] != QueryStatus.OK:
                raise CandidateUnresolved("DRIVER_CLEARANCE_UNPROVED")
            if clearance.value[0] == Clearance.GT:
                positions.append(point)
                counts.append(proposal_index + 1)
                break
            if clearance.value[0] == Clearance.LE:
                witness = clearance.evidence_ref[0]
                if witness is None or not view.verify_clearance_witness(
                        point, clearance_radius, witness):
                    raise AssertionError("expanded LE driver rejection lacked witness")
                continue
            raise CandidateUnresolved("DRIVER_CLEARANCE_UNPROVED")
        else:
            raise CandidateUnresolved("DRIVER_PROPOSAL_LIMIT")
    return ExpandedDriverSample(np.asarray(positions, dtype=np.float64),
                                tuple(counts))


def make_expanded_binary_scene(bundle: CanonicalBundle, geometry: Geometry,
                               positions: np.ndarray):
    scene = make_scene(geometry.content_hash, geometry.dimension,
                       make_physics(0), positions)
    bundle.add_scene(scene)
    return scene


def probe_expanded_candidate(scene, view, slot: int,
                             attempt: int) -> dict:
    """Run prototype acceptance probes; this does not confer READY status."""
    dimension = view.dimension
    if scene.dimension != dimension or scene.geometry_id != view.geometry_id:
        raise ValueError("expanded scene/geometry identity mismatch")
    fixed = np.asarray(list(product((0, 1), repeat=dimension)) +
                       [(.5,)*dimension], dtype=np.float64)
    random_points = expanded_stream(dimension, slot, attempt, 0,
                                    "acceptance").random((16, dimension))
    points = np.concatenate((fixed, random_points))
    classified = view.classify_points(points)
    if not np.all(classified.valid):
        raise CandidateUnresolved("ACCEPTANCE_POINT_UNPROVED")
    starts = np.concatenate((random_points[0::2], random_points[1::2],
                             random_points[:4]), axis=0)
    ends = np.concatenate((random_points[1::2], random_points[0::2],
                           random_points[:4]), axis=0)
    segments = view.segment_intersect(starts, ends)
    if not np.all(segments.valid):
        raise CandidateUnresolved("ACCEPTANCE_SEGMENT_UNPROVED")
    if not np.array_equal(segments.value[:8], segments.value[8:16]):
        raise AssertionError("expanded segment reversal changed HIT/MISS")
    lo = np.zeros(dimension, dtype=np.float64)
    hi = np.ones(dimension, dtype=np.float64)
    half_left = hi.copy()
    half_left[0] = .5
    half_right = lo.copy()
    half_right[0] = .5
    boxes = np.asarray([[lo, hi], [lo, half_left], [half_right, hi]])
    classified_boxes = view.classify_box(boxes)
    if not np.all(classified_boxes.valid):
        raise CandidateUnresolved("ACCEPTANCE_BOX_UNPROVED")
    drivers = scene.driver_position.as_array()
    driver_points = view.classify_points(drivers)
    driver_clearance = view.clearance_decide(
        drivers, float.fromhex(_profile()["driver_clearance_hex"]))
    if (not np.all(driver_points.valid) or
            not np.all(driver_points.value == Membership.FREE) or
            not np.all(driver_clearance.valid) or
            not np.all(driver_clearance.value == Clearance.GT)):
        raise AssertionError("expanded driver failed free/clearance probe")
    response = PrivateOracle(scene, view).query_response(random_points)
    for index, membership in enumerate(classified.value[len(fixed):]):
        if membership == Membership.FREE:
            if response.status[index] != QueryStatus.OK or not response.valid[index]:
                raise CandidateUnresolved("ACCEPTANCE_RESPONSE_UNPROVED")
        elif response.valid[index] or response.value[index] != 255:
            raise AssertionError("expanded structure point exposed response")
    return {
        "protocol": "expanded-prototype-probe-v1",
        "geometry_id": view.geometry_id.hex(),
        "scene_id": scene.content_hash.hex(),
        "backend": view.capability(),
        "point_status": classified.status.tolist(),
        "segment_decisions": segments.value.tolist(),
        "box_classes": classified_boxes.value.tolist(),
        "driver_count": len(drivers),
        "response_status": response.status.tolist(),
        "response_value": response.value.tolist(),
        "max_precision_bits_observed": max(
            (work["precision_bits"] for work in
             (*classified.work, *segments.work, *classified_boxes.work,
              *driver_points.work, *driver_clearance.work)), default=0),
    }
