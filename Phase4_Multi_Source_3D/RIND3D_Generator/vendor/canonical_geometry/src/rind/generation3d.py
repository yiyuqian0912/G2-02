"""Deterministic 3D proposals for the unaltered B/C/D core-pilot mixture."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math

import numpy as np

from ._exact3d import convex_hull_mesh
from .contracts import (CSGNode, CSGOpcode, Geometry, RootKind, make_leaf,
                        make_physics, make_scene)
from .generation import (CandidateInvalid, CandidateUnresolved,
                         DRIVER_CLEARANCE, DRIVER_PROPOSAL_LIMIT,
                         MASTER_SEED, _uniform)
from .geometry import Clearance, GeometryView, Membership, QueryStatus
from .oracle import PrivateOracle
from .storage import CanonicalBundle, _decode_bundle, pack_arrays


PROFILE_ID_3D = "rind-core-pilot-v3-impl2"


def stream3(slot: int, attempt: int, variant: int,
            name: str, index: int = 0) -> np.random.Generator:
    preimage = ["rind-stream-v3", MASTER_SEED, PROFILE_ID_3D, 3,
                slot, attempt, variant, name, index]
    raw = json.dumps(preimage, separators=(",", ":"), ensure_ascii=True).encode()
    seed = int.from_bytes(hashlib.sha256(raw).digest()[:16], "little")
    return np.random.Generator(np.random.PCG64(seed))


def _exact_det3(matrix: np.ndarray) -> Fraction:
    a = [Fraction.from_float(float(value)) for value in matrix.ravel()]
    return (a[0] * (a[4] * a[8] - a[5] * a[7])
            - a[1] * (a[3] * a[8] - a[5] * a[6])
            + a[2] * (a[3] * a[7] - a[4] * a[6]))


def _transform_3d(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    center = rng.random(3).astype(np.float64)
    scale = math.exp(_uniform(rng, math.log(.02), math.log(.15)))
    spread = 1.3862943611198906
    deviations = np.asarray([_uniform(rng, -spread, spread) for _ in range(3)])
    deviations -= float(np.mean(deviations))
    axes = np.asarray([scale * math.exp(float(value)) for value in deviations])
    u1, u2, u3 = (float(value) for value in rng.random(3))
    x = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    y = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    z = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    w = math.sqrt(u1) * math.cos(2 * math.pi * u3)
    rotation = np.asarray([
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)
    matrix = np.asarray(rotation @ np.diag(axes), dtype=np.float64)
    return center, matrix


def _box_or_ball(bundle: CanonicalBundle, center: np.ndarray,
                 matrix: np.ndarray, *, box: bool) -> bytes:
    if _exact_det3(matrix) == 0:
        raise CandidateInvalid("SINGULAR_TRANSFORM")
    leaf = make_leaf("affine_box_v1" if box else "affine_ball_v1", 3,
                     center=center, matrix=matrix)
    bundle.add_leaf(leaf)
    return leaf.content_hash


def _hull_leaf(bundle: CanonicalBundle, center: np.ndarray,
               matrix: np.ndarray, rng: np.random.Generator) -> bytes:
    proposed: list[tuple[float, float, float]] = []
    for _ in range(24):
        normal = [float(value) for value in rng.normal(size=3)]
        norm = math.sqrt(sum(value * value for value in normal))
        if norm == 0 or not math.isfinite(norm):
            raise CandidateInvalid("DEGENERATE_HULL_PROPOSAL")
        radius = float(rng.random()) ** (1.0 / 3.0)
        proposed.append(tuple(radius * value / norm for value in normal))
    proposed.extend([(1., 0., 0.), (-1., 0., 0.),
                     (0., 1., 0.), (0., -1., 0.),
                     (0., 0., 1.), (0., 0., -1.)])
    try:
        vertices, triangles = convex_hull_mesh(proposed)
    except ValueError as exc:
        raise CandidateInvalid("INVALID_MESH_TOPOLOGY") from exc
    leaf = make_leaf(
        "oriented_mesh_3d_v1", 3, center=center, matrix=matrix,
        vertices=np.asarray(vertices, dtype=np.float64),
        triangles=np.asarray(triangles, dtype=np.uint32),
        shell_triangle_offsets=np.asarray([0, len(triangles)], dtype=np.uint64),
        shell_role=np.asarray([0], dtype=np.uint8),
        shell_owner=np.asarray([2**32 - 1], dtype=np.uint32),
    )
    bundle.add_leaf(leaf)
    return leaf.content_hash


def _object_3d(bundle: CanonicalBundle, slot: int, attempt: int,
               object_index: int) -> bytes:
    family_rng = stream3(slot, attempt, 0, "family", object_index)
    rng = stream3(slot, attempt, 0, "object_parameters", object_index)
    family = float(family_rng.random())
    center, matrix = _transform_3d(rng)
    if _exact_det3(matrix) == 0:
        raise CandidateInvalid("SINGULAR_TRANSFORM")
    if family < .5:
        return _box_or_ball(bundle, center, matrix,
                            box=float(rng.random()) < .5)
    if family < .75:
        return _hull_leaf(bundle, center, matrix, rng)

    outer = _box_or_ball(bundle, center, matrix, box=True)
    if float(rng.random()) < .5:
        scales = np.asarray([_uniform(rng, .4, .9) for _ in range(3)])
        offset = np.asarray([_uniform(rng, -.8, .8) for _ in range(3)])
        opcode = CSGOpcode.UNION
    else:
        if float(rng.random()) < .5:
            scales = np.asarray([_uniform(rng, .2, .7) for _ in range(3)])
            offset = np.asarray([_uniform(rng, -.1, .1) for _ in range(3)])
        else:
            axis = int(rng.integers(0, 3))
            sign = -1 if int(rng.integers(0, 2)) == 0 else 1
            scales = np.asarray([_uniform(rng, .2, .6) for _ in range(3)])
            offset = np.asarray([_uniform(rng, -.1, .1) for _ in range(3)])
            scales[axis] = 1
            offset[axis] = sign * .6
        opcode = CSGOpcode.DIFFERENCE
    secondary_center = np.asarray(center + matrix @ offset, dtype=np.float64)
    secondary_matrix = np.asarray(matrix @ np.diag(scales), dtype=np.float64)
    secondary = _box_or_ball(bundle, secondary_center, secondary_matrix, box=True)
    node = CSGNode(3, opcode, (outer, secondary))
    bundle.add_node(node)
    return node.content_hash


def propose_geometry_3d(slot: int, attempt: int) -> tuple[CanonicalBundle, Geometry]:
    if slot < 0 or attempt < 0:
        raise ValueError("slot and attempt must be nonnegative")
    bundle = CanonicalBundle(3)
    count = int(stream3(slot, attempt, 0,
                       "structure_object_count").integers(0, 17))
    roots = [_object_3d(bundle, slot, attempt, index) for index in range(count)]
    if not roots:
        geometry = Geometry.empty(3)
    else:
        if len(roots) == 1:
            root = roots[0]
        else:
            node = CSGNode(3, CSGOpcode.UNION, tuple(roots))
            bundle.add_node(node)
            root = node.content_hash
        geometry = Geometry(3, RootKind.REF, root)
    bundle.add_geometry(geometry)
    checked = _decode_bundle(pack_arrays(bundle), 3)
    if checked.geometries[0].content_hash != geometry.content_hash:
        raise AssertionError("3D geometry changed during typed SoA reconstruction")
    return checked, checked.geometries[0]


@dataclass(frozen=True, slots=True)
class DriverSample3D:
    positions: np.ndarray
    proposals_per_driver: tuple[int, ...]


def sample_drivers_3d(view: GeometryView, slot: int, attempt: int,
                      variant: int = 0,
                      proposal_limit: int = DRIVER_PROPOSAL_LIMIT) -> DriverSample3D:
    if view.dimension != 3:
        raise ValueError("3D driver sampler requires 3D geometry")
    count = int(stream3(slot, attempt, variant, "driver_count").integers(1, 5))
    if view.is_full():
        raise CandidateInvalid("NO_ALLOWED_DRIVER_PROVED")
    positions = []
    counts = []
    for driver_index in range(count):
        rng = stream3(slot, attempt, variant, "driver_position", driver_index)
        for proposal_index in range(proposal_limit):
            point = rng.random(3).astype(np.float64)
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
                if witness is None or not view.verify_clearance_witness(
                        point, DRIVER_CLEARANCE, witness):
                    raise AssertionError("3D LE driver rejection lacked witness")
                continue
            raise CandidateUnresolved("DRIVER_CLEARANCE_UNPROVED")
        else:
            raise CandidateUnresolved("DRIVER_PROPOSAL_LIMIT")
    return DriverSample3D(np.asarray(positions, dtype=np.float64), tuple(counts))


def make_binary_scene_3d(bundle: CanonicalBundle, geometry: Geometry,
                         positions: np.ndarray):
    scene = make_scene(geometry.content_hash, 3, make_physics(0), positions)
    bundle.add_scene(scene)
    return scene


def validate_candidate_3d(scene, view: GeometryView, slot: int,
                          attempt: int) -> dict:
    if scene.dimension != 3 or scene.geometry_id != view.geometry_id:
        raise ValueError("3D candidate scene and geometry mismatch")
    corners = np.asarray(list(__import__("itertools").product((0, 1), repeat=3)),
                         dtype=np.float64)
    fixed = np.concatenate((corners, [[.5, .5, .5]]), axis=0)
    random_points = stream3(slot, attempt, 0, "acceptance").random((16, 3))
    all_points = np.concatenate((fixed, random_points), axis=0)
    classified = view.classify_points(all_points)
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
        raise AssertionError("3D segment reversal changed HIT/MISS")
    boxes = np.asarray([[[0, 0, 0], [1, 1, 1]],
                        [[0, 0, 0], [.5, 1, 1]],
                        [[.5, 0, 0], [1, 1, 1]]], dtype=np.float64)
    box_classes = view.classify_box(boxes)
    if not np.all(box_classes.valid):
        raise CandidateUnresolved("ACCEPTANCE_BOX_UNPROVED")
    drivers = scene.driver_position.as_array()
    driver_classes = view.classify_points(drivers)
    driver_clearance = view.clearance_decide(drivers, DRIVER_CLEARANCE)
    if (not np.all(driver_classes.valid) or
            not np.all(driver_classes.value == Membership.FREE) or
            not np.all(driver_clearance.valid) or
            not np.all(driver_clearance.value == Clearance.GT)):
        raise AssertionError("3D accepted driver failed free/clearance probe")
    response = PrivateOracle(scene, view).query_response(random_points)
    for index, membership in enumerate(classified.value[len(fixed):]):
        if membership == Membership.FREE:
            if response.status[index] != QueryStatus.OK or not response.valid[index]:
                raise CandidateUnresolved("ACCEPTANCE_RESPONSE_UNPROVED")
        elif response.valid[index] or response.value[index] != 255:
            raise AssertionError("3D structure point exposed binary response")
    return {
        "protocol": "acceptance-core-v1", "geometry_id": view.geometry_id.hex(),
        "scene_id": scene.content_hash.hex(), "backend": view.capability(),
        "point_inputs": all_points.tolist(),
        "point_status": classified.status.tolist(),
        "point_membership": classified.value.tolist(),
        "segment_starts": starts.tolist(), "segment_ends": ends.tolist(),
        "segment_decisions": segments.value.tolist(),
        "box_bounds": boxes.tolist(), "box_classes": box_classes.value.tolist(),
        "driver_positions": drivers.tolist(),
        "driver_membership": driver_classes.value.tolist(),
        "driver_clearance": driver_clearance.value.tolist(),
        "response_status": response.status.tolist(),
        "response_valid": response.valid.tolist(),
        "response_value": response.value.tolist(),
    }
