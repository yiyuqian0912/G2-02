"""Deterministic implementation-1 candidate proposals for the 2D core pilot.

The generator creates truth records only. It does not synthesize a model input
or modify proposals based on response labels or apparent difficulty.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

import numpy as np

from ._exact2d import convex_hull_indices
from .contracts import (CSGNode, CSGOpcode, Geometry, Leaf, RootKind,
                        make_leaf, make_physics, make_scene)
from .geometry import Clearance, GeometryView, Membership, QueryStatus
from .oracle import PrivateOracle
from .storage import CanonicalBundle, _decode_bundle, pack_arrays


PROFILE_ID = "rind-core-pilot-v3-impl1"
MASTER_SEED = 20260921
MAX_ATTEMPTS = 4
DRIVER_PROPOSAL_LIMIT = 1_000_000
DRIVER_CLEARANCE = 2**-26


class CandidateInvalid(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class CandidateUnresolved(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def stream_seed(dimension: int, slot: int, attempt: int, variant: int,
                name: str, index: int = 0) -> int:
    preimage = ["rind-stream-v3", MASTER_SEED, PROFILE_ID, dimension, slot,
                attempt, variant, name, index]
    raw = json.dumps(preimage, separators=(",", ":"), ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:16], "little")


def stream(dimension: int, slot: int, attempt: int, variant: int,
           name: str, index: int = 0) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(
        stream_seed(dimension, slot, attempt, variant, name, index)))


def _uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return low + (high - low) * float(rng.random())


def _transform_2d(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    center = rng.random(2).astype(np.float64)
    scale = math.exp(_uniform(rng, math.log(.02), math.log(.15)))
    aspect = math.exp(_uniform(rng, math.log(.125), math.log(8)))
    axes = np.array([scale * math.sqrt(aspect), scale / math.sqrt(aspect)])
    theta = 2 * math.pi * float(rng.random())
    cosine, sine = math.cos(theta), math.sin(theta)
    matrix = np.array([[cosine * axes[0], -sine * axes[1]],
                       [sine * axes[0], cosine * axes[1]]], dtype=np.float64)
    return center, matrix


def _exact_det(matrix: np.ndarray) -> Fraction:
    a, b, c, d = (Fraction.from_float(float(x)) for x in matrix.ravel())
    return a * d - b * c


def _orientation(a, b, c) -> Fraction:
    ax, ay = (Fraction.from_float(float(x)) for x in a)
    bx, by = (Fraction.from_float(float(x)) for x in b)
    cx, cy = (Fraction.from_float(float(x)) for x in c)
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _ring_valid(vertices: np.ndarray) -> None:
    count = len(vertices)
    if count < 3 or len({tuple(p) for p in vertices}) != count:
        raise CandidateInvalid("DEGENERATE_RING")
    area = sum(_orientation((0, 0), vertices[i], vertices[(i + 1) % count])
               for i in range(count))
    if area <= 0:
        raise CandidateInvalid("DEGENERATE_RING")

    def on_segment(a, b, p) -> bool:
        return (_orientation(a, b, p) == 0 and
                all(min(a[j], b[j]) <= p[j] <= max(a[j], b[j]) for j in range(2)))

    def intersects(a, b, c, d) -> bool:
        p, q = _orientation(a, b, c), _orientation(a, b, d)
        r, s = _orientation(c, d, a), _orientation(c, d, b)
        return ((p == 0 and on_segment(a, b, c)) or
                (q == 0 and on_segment(a, b, d)) or
                (r == 0 and on_segment(c, d, a)) or
                (s == 0 and on_segment(c, d, b)) or
                ((p > 0) != (q > 0) and (r > 0) != (s > 0)))

    for i in range(count):
        for j in range(i + 1, count):
            if j == i + 1 or (i == 0 and j == count - 1):
                continue
            if intersects(vertices[i], vertices[(i + 1) % count],
                          vertices[j], vertices[(j + 1) % count]):
                raise CandidateInvalid("SELF_INTERSECTING_RING")


def _leaf_box_or_ball(bundle: CanonicalBundle, center: np.ndarray,
                      matrix: np.ndarray, box: bool) -> bytes:
    if _exact_det(matrix) == 0:
        raise CandidateInvalid("SINGULAR_TRANSFORM")
    leaf = make_leaf("affine_box_v1" if box else "affine_ball_v1", 2,
                     center=center, matrix=matrix)
    bundle.add_leaf(leaf)
    return leaf.content_hash


def _object_2d(bundle: CanonicalBundle, slot: int, attempt: int,
               object_index: int) -> bytes:
    family_rng = stream(2, slot, attempt, 0, "family", object_index)
    parameter_rng = stream(2, slot, attempt, 0, "object_parameters", object_index)
    choice = float(family_rng.random())
    center, matrix = _transform_2d(parameter_rng)
    if _exact_det(matrix) == 0:
        raise CandidateInvalid("SINGULAR_TRANSFORM")
    if choice < .5:
        return _leaf_box_or_ball(bundle, center, matrix,
                                 float(parameter_rng.random()) < .5)
    if choice < .75:
        hull = float(parameter_rng.random()) < .5
        n = int(parameter_rng.integers(5, 13))
        vertices: list[tuple[float, float]] = []
        if hull:
            for _ in range(n):
                theta = 2 * math.pi * float(parameter_rng.random())
                radius = math.sqrt(float(parameter_rng.random()))
                vertices.append((radius * math.cos(theta), radius * math.sin(theta)))
            try:
                indices = convex_hull_indices(vertices)
            except ValueError as exc:
                raise CandidateInvalid("DEGENERATE_RING") from exc
            vertices = [vertices[index] for index in indices]
        else:
            for i in range(n):
                theta = 2 * math.pi * (i + float(parameter_rng.random())) / n
                radius = .35 + .65 * float(parameter_rng.random())
                vertices.append((radius * math.cos(theta), radius * math.sin(theta)))
        vertices_array = np.asarray(vertices, dtype=np.float64)
        _ring_valid(vertices_array)
        leaf = make_leaf(
            "polygon_rings_2d_v1", 2, center=center, matrix=matrix,
            vertices=vertices_array, ring_vertex_offsets=np.array([0, len(vertices)], dtype=np.uint64),
            ring_role=np.array([0], dtype=np.uint8),
            ring_owner=np.array([2**32 - 1], dtype=np.uint32),
        )
        bundle.add_leaf(leaf)
        return leaf.content_hash

    outer = _leaf_box_or_ball(bundle, center, matrix, True)
    recipe = float(parameter_rng.random())
    if recipe < .5:
        scales = np.array([_uniform(parameter_rng, .4, .9) for _ in range(2)])
        offset = np.array([_uniform(parameter_rng, -.8, .8) for _ in range(2)])
        opcode = CSGOpcode.UNION
    else:
        subtype = float(parameter_rng.random())
        if subtype < .5:
            scales = np.array([_uniform(parameter_rng, .2, .7) for _ in range(2)])
            offset = np.array([_uniform(parameter_rng, -.1, .1) for _ in range(2)])
        else:
            axis = int(parameter_rng.integers(0, 2))
            sign = -1 if int(parameter_rng.integers(0, 2)) == 0 else 1
            scales = np.array([_uniform(parameter_rng, .2, .6) for _ in range(2)])
            offset = np.array([_uniform(parameter_rng, -.1, .1) for _ in range(2)])
            scales[axis] = 1
            offset[axis] = sign * .6
        opcode = CSGOpcode.DIFFERENCE
    secondary_center = np.asarray(center + matrix @ offset, dtype=np.float64)
    secondary_matrix = np.asarray(matrix @ np.diag(scales), dtype=np.float64)
    secondary = _leaf_box_or_ball(bundle, secondary_center, secondary_matrix, True)
    node = CSGNode(2, opcode, (outer, secondary))
    bundle.add_node(node)
    return node.content_hash


def propose_geometry_2d(slot: int, attempt: int) -> tuple[CanonicalBundle, Geometry]:
    if slot < 0 or attempt < 0:
        raise ValueError("slot and attempt must be nonnegative")
    bundle = CanonicalBundle(2)
    count_rng = stream(2, slot, attempt, 0, "structure_object_count")
    count = int(count_rng.integers(0, 17))
    roots = [_object_2d(bundle, slot, attempt, index) for index in range(count)]
    if not roots:
        geometry = Geometry.empty(2)
    else:
        if len(roots) == 1:
            root = roots[0]
        else:
            node = CSGNode(2, CSGOpcode.UNION, tuple(roots))
            bundle.add_node(node)
            root = node.content_hash
        geometry = Geometry(2, RootKind.REF, root)
    bundle.add_geometry(geometry)
    # A typed SoA reconstruction here catches identity, offset and canonical
    # ordering errors even before the later persistent Zarr staging pass.
    checked = _decode_bundle(pack_arrays(bundle), 2)
    if checked.geometries[0].content_hash != geometry.content_hash:
        raise AssertionError("geometry changed during typed SoA reconstruction")
    return checked, checked.geometries[0]


@dataclass(frozen=True, slots=True)
class DriverSample:
    positions: np.ndarray
    proposals_per_driver: tuple[int, ...]


def sample_drivers_2d(view: GeometryView, slot: int, attempt: int,
                      variant: int = 0,
                      proposal_limit: int = DRIVER_PROPOSAL_LIMIT) -> DriverSample:
    if view.dimension != 2:
        raise ValueError("2D driver sampler requires 2D geometry")
    count_rng = stream(2, slot, attempt, variant, "driver_count")
    count = int(count_rng.integers(1, 5))
    if view.is_full():
        raise CandidateInvalid("NO_ALLOWED_DRIVER_PROVED")
    positions = []
    counts = []
    for driver_index in range(count):
        rng = stream(2, slot, attempt, variant, "driver_position", driver_index)
        for proposal_index in range(proposal_limit):
            point = rng.random(2).astype(np.float64)
            classification = view.classify_points(point)
            if classification.status[0] in (QueryStatus.UNSUPPORTED, QueryStatus.UNRESOLVED):
                raise CandidateUnresolved("DRIVER_MEMBERSHIP_UNPROVED")
            if classification.value[0] != Membership.FREE:
                continue
            clearance = view.clearance_decide(point, DRIVER_CLEARANCE)
            if clearance.status[0] != QueryStatus.OK:
                raise CandidateUnresolved("DRIVER_CLEARANCE_UNPROVED")
            if clearance.value[0] == Clearance.GT:
                positions.append(point)
                counts.append(proposal_index + 1)
                break
            if clearance.value[0] == Clearance.LE:
                witness = clearance.evidence_ref[0]
                if witness is None or not view.verify_clearance_witness(point, DRIVER_CLEARANCE, witness):
                    raise AssertionError("LE driver rejection lacked a verified witness")
                continue
            raise CandidateUnresolved("DRIVER_CLEARANCE_UNPROVED")
        else:
            raise CandidateUnresolved("DRIVER_PROPOSAL_LIMIT")
    return DriverSample(np.asarray(positions, dtype=np.float64), tuple(counts))


def make_binary_scene(bundle: CanonicalBundle, geometry: Geometry,
                      positions: np.ndarray):
    scene = make_scene(geometry.content_hash, 2, make_physics(0), positions)
    bundle.add_scene(scene)
    return scene


def validate_candidate_2d(scene, view: GeometryView, slot: int,
                          attempt: int) -> dict:
    """Run the mandatory geometry/driver/response probes without label filtering."""
    if scene.dimension != 2 or scene.geometry_id != view.geometry_id:
        raise ValueError("candidate scene and geometry mismatch")
    fixed = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1], [.5, .5]], dtype=np.float64)
    random_points = stream(2, slot, attempt, 0, "acceptance").random((16, 2))
    all_points = np.concatenate((fixed, random_points), axis=0)
    classified = view.classify_points(all_points)
    if not np.all(classified.valid):
        raise CandidateUnresolved("ACCEPTANCE_POINT_UNPROVED")
    segment_starts = np.concatenate((random_points[0::2], random_points[1::2],
                                     random_points[:4]), axis=0)
    segment_ends = np.concatenate((random_points[1::2], random_points[0::2],
                                   random_points[:4]), axis=0)
    segments = view.segment_intersect(segment_starts, segment_ends)
    if not np.all(segments.valid):
        raise CandidateUnresolved("ACCEPTANCE_SEGMENT_UNPROVED")
    if not np.array_equal(segments.value[:8], segments.value[8:16]):
        raise AssertionError("segment reversal changed certified HIT/MISS")
    boxes = np.asarray([[[0, 0], [1, 1]],
                        [[0, 0], [.5, 1]],
                        [[.5, 0], [1, 1]]], dtype=np.float64)
    box_classes = view.classify_box(boxes)
    if not np.all(box_classes.valid):
        raise CandidateUnresolved("ACCEPTANCE_BOX_UNPROVED")
    driver_points = scene.driver_position.as_array()
    driver_classes = view.classify_points(driver_points)
    driver_clearance = view.clearance_decide(driver_points, DRIVER_CLEARANCE)
    if (not np.all(driver_classes.valid) or
            not np.all(driver_classes.value == Membership.FREE) or
            not np.all(driver_clearance.valid) or
            not np.all(driver_clearance.value == Clearance.GT)):
        raise AssertionError("accepted driver failed mandatory free/clearance probe")
    oracle = PrivateOracle(scene, view)
    response = oracle.query_response(random_points)
    for index, membership in enumerate(classified.value[5:]):
        if membership == Membership.FREE:
            if response.status[index] != QueryStatus.OK or not response.valid[index]:
                raise CandidateUnresolved("ACCEPTANCE_RESPONSE_UNPROVED")
        elif response.valid[index] or response.value[index] != 255:
            raise AssertionError("structure query exposed a binary response")
    return {
        "protocol": "acceptance-core-v1",
        "geometry_id": view.geometry_id.hex(),
        "scene_id": scene.content_hash.hex(),
        "backend": view.capability(),
        "point_inputs": all_points.tolist(),
        "point_status": classified.status.tolist(),
        "point_membership": classified.value.tolist(),
        "segment_starts": segment_starts.tolist(),
        "segment_ends": segment_ends.tolist(),
        "segment_decisions": segments.value.tolist(),
        "box_bounds": boxes.tolist(),
        "box_classes": box_classes.value.tolist(),
        "driver_positions": driver_points.tolist(),
        "driver_membership": driver_classes.value.tolist(),
        "driver_clearance": driver_clearance.value.tolist(),
        "response_status": response.status.tolist(),
        "response_valid": response.valid.tolist(),
        "response_value": response.value.tolist(),
    }
