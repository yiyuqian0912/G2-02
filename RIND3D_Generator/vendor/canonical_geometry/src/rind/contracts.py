"""Immutable canonical records and codec-level structural schemas."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping

import numpy as np

from .identity import DTypeTag as T
from .identity import Field, digest, frame, hash_bytes_field


SOLID_SEMANTICS = "regular-closed-omega-1"
CSG_SEMANTICS = "regular-csg-1"
QUERY_SEMANTICS = "rind-query-relative-omega-1"


class CodecID(IntEnum):
    AFFINE_BOX = 1
    AFFINE_BALL = 2
    POLYGON_RINGS_2D = 3
    CONVEX_HALFSPACES = 4
    ORIENTED_MESH_3D = 5
    RADIAL_FOURIER_2D = 6
    RADIAL_FOURIER_SPHERE = 7
    COSINE_FIELD = 8


CODEC_NAMES = {
    CodecID.AFFINE_BOX: "affine_box_v1",
    CodecID.AFFINE_BALL: "affine_ball_v1",
    CodecID.POLYGON_RINGS_2D: "polygon_rings_2d_v1",
    CodecID.CONVEX_HALFSPACES: "convex_halfspaces_v1",
    CodecID.ORIENTED_MESH_3D: "oriented_mesh_3d_v1",
    CodecID.RADIAL_FOURIER_2D: "radial_fourier_2d_v1",
    CodecID.RADIAL_FOURIER_SPHERE: "radial_fourier_sphere_v1",
    CodecID.COSINE_FIELD: "cosine_field_v1",
}
NAME_TO_CODEC = {name: codec for codec, name in CODEC_NAMES.items()}


class CSGOpcode(IntEnum):
    UNION = 0
    INTERSECTION = 1
    DIFFERENCE = 2


class RootKind(IntEnum):
    EMPTY = 0
    REF = 1


PHYSICS_CODECS = {0: "binary_v1", 1: "exponential_v1", 2: "log_pchip_v1"}


def _check_dimension(dimension: int) -> None:
    if dimension not in (2, 3):
        raise ValueError("scene dimension must be 2 or 3")


def _field_map(fields: tuple[Field, ...]) -> dict[str, Field]:
    result = {field.name: field for field in fields}
    if len(result) != len(fields):
        raise ValueError("duplicate leaf field")
    return result


def _shape(field: Field, expected: tuple[int, ...], tag: T) -> None:
    if field.shape != expected or field.tag != tag:
        raise ValueError(
            f"{field.name}: expected {tag.name}{expected}, got {field.tag.name}{field.shape}"
        )


def _offsets(field: Field, rows: int, end: int) -> None:
    _shape(field, (rows + 1,), T.UINT64)
    offsets = field.as_array()
    if int(offsets[0]) != 0 or int(offsets[-1]) != end:
        raise ValueError(f"{field.name}: invalid first/last offset")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError(f"{field.name}: offsets must be nondecreasing")


@dataclass(frozen=True, slots=True)
class Leaf:
    codec: CodecID
    dimension: int
    fields: tuple[Field, ...]

    def __post_init__(self) -> None:
        _check_dimension(self.dimension)
        values = _field_map(self.fields)
        d = self.dimension
        common = {"center", "matrix"}
        if self.codec in (CodecID.AFFINE_BOX, CodecID.AFFINE_BALL):
            expected = common
        elif self.codec == CodecID.POLYGON_RINGS_2D:
            if d != 2:
                raise ValueError("polygon rings require dimension 2")
            expected = common | {
                "vertices", "ring_vertex_offsets", "ring_role", "ring_owner"
            }
        elif self.codec == CodecID.CONVEX_HALFSPACES:
            expected = common | {"normal", "plane_offset"}
        elif self.codec == CodecID.ORIENTED_MESH_3D:
            if d != 3:
                raise ValueError("oriented mesh requires dimension 3")
            expected = common | {
                "vertices", "triangles", "shell_triangle_offsets",
                "shell_role", "shell_owner",
            }
        elif self.codec == CodecID.RADIAL_FOURIER_2D:
            if d != 2:
                raise ValueError("2D radial Fourier requires dimension 2")
            expected = common | {"coeff_a", "coeff_b"}
        elif self.codec == CodecID.RADIAL_FOURIER_SPHERE:
            if d != 3:
                raise ValueError("radial Fourier sphere requires dimension 3")
            expected = common | {"frequency", "coeff_a", "coeff_b"}
        elif self.codec == CodecID.COSINE_FIELD:
            expected = {"amplitude", "frequency", "phase", "threshold"}
        else:
            raise ValueError(f"unknown codec ID {self.codec}")
        if set(values) != expected:
            raise ValueError(f"{CODEC_NAMES[self.codec]} fields differ: {set(values) ^ expected}")

        if "center" in values:
            _shape(values["center"], (d,), T.FLOAT64)
            _shape(values["matrix"], (d, d), T.FLOAT64)
        if self.codec == CodecID.POLYGON_RINGS_2D:
            vertices = values["vertices"]
            if vertices.tag != T.FLOAT64 or len(vertices.shape) != 2 or vertices.shape[1] != 2:
                raise ValueError("polygon vertices must be float64[V,2]")
            rings = values["ring_role"].shape[0]
            _shape(values["ring_role"], (rings,), T.UINT8)
            _shape(values["ring_owner"], (rings,), T.UINT32)
            _offsets(values["ring_vertex_offsets"], rings, vertices.shape[0])
            roles = values["ring_role"].as_array()
            owners = values["ring_owner"].as_array()
            if np.any(roles > 1):
                raise ValueError("invalid ring role")
            for index, (role, owner) in enumerate(zip(roles, owners, strict=True)):
                if role == 0 and int(owner) != 2**32 - 1:
                    raise ValueError(f"outer ring {index} must have no owner")
                if role == 1 and (int(owner) >= rings or roles[int(owner)] != 0):
                    raise ValueError(f"hole ring {index} has invalid owner")
        elif self.codec == CodecID.CONVEX_HALFSPACES:
            normal = values["normal"]
            if normal.tag != T.FLOAT64 or len(normal.shape) != 2 or normal.shape[1] != d:
                raise ValueError("normals must be float64[H,d]")
            _shape(values["plane_offset"], (normal.shape[0],), T.FLOAT64)
        elif self.codec == CodecID.ORIENTED_MESH_3D:
            vertices = values["vertices"]
            triangles = values["triangles"]
            if vertices.tag != T.FLOAT64 or len(vertices.shape) != 2 or vertices.shape[1] != 3:
                raise ValueError("mesh vertices must be float64[V,3]")
            if triangles.tag != T.UINT32 or len(triangles.shape) != 2 or triangles.shape[1] != 3:
                raise ValueError("triangles must be uint32[F,3]")
            shells = values["shell_role"].shape[0]
            _shape(values["shell_role"], (shells,), T.UINT8)
            _shape(values["shell_owner"], (shells,), T.UINT32)
            _offsets(values["shell_triangle_offsets"], shells, triangles.shape[0])
            if np.any(triangles.as_array() >= vertices.shape[0]):
                raise ValueError("mesh triangle index is out of bounds")
            roles = values["shell_role"].as_array()
            owners = values["shell_owner"].as_array()
            if np.any(roles > 1):
                raise ValueError("invalid shell role")
            for index, (role, owner) in enumerate(zip(roles, owners, strict=True)):
                if role == 0 and int(owner) != 2**32 - 1:
                    raise ValueError(f"outer shell {index} must have no owner")
                if role == 1 and (int(owner) >= shells or roles[int(owner)] != 0):
                    raise ValueError(f"cavity shell {index} has invalid owner")
        elif self.codec == CodecID.RADIAL_FOURIER_2D:
            n = values["coeff_a"].shape[0]
            _shape(values["coeff_a"], (n,), T.FLOAT64)
            _shape(values["coeff_b"], (n,), T.FLOAT64)
        elif self.codec == CodecID.RADIAL_FOURIER_SPHERE:
            n = values["coeff_a"].shape[0]
            _shape(values["frequency"], (n, 3), T.FLOAT64)
            _shape(values["coeff_a"], (n,), T.FLOAT64)
            _shape(values["coeff_b"], (n,), T.FLOAT64)
        elif self.codec == CodecID.COSINE_FIELD:
            n = values["amplitude"].shape[0]
            _shape(values["amplitude"], (n,), T.FLOAT64)
            _shape(values["frequency"], (n, d), T.FLOAT64)
            _shape(values["phase"], (n,), T.FLOAT64)
            _shape(values["threshold"], (), T.FLOAT64)

    def field(self, name: str) -> Field:
        for field in self.fields:
            if field.name == name:
                return field
        raise KeyError(name)

    @property
    def semantic_fields(self) -> tuple[Field, ...]:
        return (
            Field.text("codec_name", CODEC_NAMES[self.codec]),
            Field.numeric("dimension", T.UINT8, self.dimension),
            *self.fields,
        )

    @property
    def content_frame(self) -> bytes:
        return frame("leaf", self.semantic_fields)

    @property
    def content_hash(self) -> bytes:
        return digest("leaf", self.semantic_fields)


def make_leaf(codec: CodecID | int | str, dimension: int, **values: object) -> Leaf:
    if isinstance(codec, str):
        codec = NAME_TO_CODEC[codec]
    codec = CodecID(codec)
    _check_dimension(dimension)
    variable_types = {
        "ring_vertex_offsets": T.UINT64,
        "ring_role": T.UINT8,
        "ring_owner": T.UINT32,
        "triangles": T.UINT32,
        "shell_triangle_offsets": T.UINT64,
        "shell_role": T.UINT8,
        "shell_owner": T.UINT32,
    }
    fields = tuple(
        Field.numeric(name, variable_types.get(name, T.FLOAT64), value)
        for name, value in values.items()
    )
    return Leaf(codec, dimension, fields)


@dataclass(frozen=True, slots=True)
class CSGNode:
    dimension: int
    opcode: CSGOpcode
    children: tuple[bytes, ...]

    def __post_init__(self) -> None:
        _check_dimension(self.dimension)
        if self.opcode == CSGOpcode.DIFFERENCE:
            if len(self.children) != 2:
                raise ValueError("DIFFERENCE requires exactly two children")
        elif self.opcode in (CSGOpcode.UNION, CSGOpcode.INTERSECTION):
            if len(self.children) < 2:
                raise ValueError("UNION/INTERSECTION require at least two children")
        else:
            raise ValueError("unknown CSG opcode")
        for child in self.children:
            if len(child) != 32:
                raise ValueError("child hash must be 32 bytes")

    @property
    def semantic_fields(self) -> tuple[Field, ...]:
        hashes = np.frombuffer(b"".join(self.children), dtype=np.uint8).reshape(-1, 32)
        return (
            Field.text("csg_semantics", CSG_SEMANTICS),
            Field.numeric("dimension", T.UINT8, self.dimension),
            Field.numeric("opcode", T.UINT8, int(self.opcode)),
            Field.numeric("child_hashes", T.UINT8, hashes),
        )

    @property
    def content_frame(self) -> bytes:
        return frame("csg_node", self.semantic_fields)

    @property
    def content_hash(self) -> bytes:
        return digest("csg_node", self.semantic_fields)


def empty_root_hash(dimension: int) -> bytes:
    _check_dimension(dimension)
    return digest("empty_root", (Field.numeric("dimension", T.UINT8, dimension),))


@dataclass(frozen=True, slots=True)
class Geometry:
    dimension: int
    root_kind: RootKind
    root_hash: bytes

    def __post_init__(self) -> None:
        _check_dimension(self.dimension)
        if len(self.root_hash) != 32:
            raise ValueError("root hash must be 32 bytes")
        if self.root_kind == RootKind.EMPTY and self.root_hash != empty_root_hash(self.dimension):
            raise ValueError("EMPTY geometry requires its registered empty root hash")

    @classmethod
    def empty(cls, dimension: int) -> Geometry:
        return cls(dimension, RootKind.EMPTY, empty_root_hash(dimension))

    @property
    def semantic_fields(self) -> tuple[Field, ...]:
        return (
            Field.numeric("dimension", T.UINT8, self.dimension),
            Field.text("solid_semantics", SOLID_SEMANTICS),
            Field.numeric("root_kind", T.UINT8, int(self.root_kind)),
            hash_bytes_field("root_hash", self.root_hash),
        )

    @property
    def content_frame(self) -> bytes:
        return frame("geometry", self.semantic_fields)

    @property
    def content_hash(self) -> bytes:
        return digest("geometry", self.semantic_fields)


@dataclass(frozen=True, slots=True)
class Physics:
    codec_id: int
    fields: tuple[Field, ...] = ()

    def __post_init__(self) -> None:
        if self.codec_id not in PHYSICS_CODECS:
            raise ValueError("unknown physics codec")
        values = _field_map(self.fields)
        expected = {0: set(), 1: {"tau"}, 2: {"knots", "log_values"}}[self.codec_id]
        if set(values) != expected:
            raise ValueError("incorrect physics payload fields")
        if self.codec_id == 1:
            _shape(values["tau"], (), T.FLOAT64)
            if float(values["tau"].as_array()) <= 0:
                raise ValueError("exponential tau must be positive")
        elif self.codec_id == 2:
            _shape(values["knots"], (9,), T.FLOAT64)
            _shape(values["log_values"], (9,), T.FLOAT64)
            knots = values["knots"].as_array()
            log_values = values["log_values"].as_array()
            if knots[0] != 0 or np.any(knots[1:] <= knots[:-1]):
                raise ValueError("PCHIP knots must start at zero and increase")
            if log_values[0] != 0 or np.any(log_values[1:] >= log_values[:-1]):
                raise ValueError("PCHIP log values must start at zero and decrease")

    @property
    def semantic_fields(self) -> tuple[Field, ...]:
        return (Field.text("physics_codec", PHYSICS_CODECS[self.codec_id]), *self.fields)

    @property
    def content_hash(self) -> bytes:
        return digest("physics", self.semantic_fields)


@dataclass(frozen=True, slots=True)
class Scene:
    geometry_id: bytes
    dimension: int
    physics: Physics
    driver_position: Field
    driver_strength: Field | None = None

    def __post_init__(self) -> None:
        _check_dimension(self.dimension)
        if len(self.geometry_id) != 32:
            raise ValueError("geometry ID must be 32 bytes")
        if self.driver_position.name != "driver_position":
            raise ValueError("driver position field has wrong name")
        if self.driver_position.tag != T.FLOAT64:
            raise ValueError("driver positions must be float64")
        if len(self.driver_position.shape) != 2 or self.driver_position.shape[1] != self.dimension:
            raise ValueError("driver positions must have shape [K,d]")
        if self.physics.codec_id == 0:
            if self.driver_strength is not None:
                raise ValueError("binary scene cannot have a strength field")
        else:
            if self.driver_strength is None:
                raise ValueError("additive scene requires strengths")
            _shape(self.driver_strength, (self.driver_position.shape[0],), T.FLOAT64)
            if self.driver_strength.name != "driver_strength":
                raise ValueError("strength field has wrong name")
            if np.any(self.driver_strength.as_array() <= 0):
                raise ValueError("driver strengths must be positive")
        positions = self.driver_position.as_array()
        strengths = self.driver_strength.as_array() if self.driver_strength is not None else None
        expected_order = sorted(
            range(len(positions)),
            key=lambda i: (*positions[i].tolist(), *(strengths[i:i+1].tolist() if strengths is not None else [])),
        )
        if expected_order != list(range(len(positions))):
            raise ValueError("drivers are not in canonical numeric order")

    @property
    def semantic_fields(self) -> tuple[Field, ...]:
        fields = [
            hash_bytes_field("geometry_id", self.geometry_id),
            Field.text("physics_codec", PHYSICS_CODECS[self.physics.codec_id]),
            hash_bytes_field("physics_hash", self.physics.content_hash),
            self.driver_position,
        ]
        if self.driver_strength is not None:
            fields.append(self.driver_strength)
        return tuple(fields)

    @property
    def content_frame(self) -> bytes:
        return frame("scene", self.semantic_fields)

    @property
    def content_hash(self) -> bytes:
        return digest("scene", self.semantic_fields)


def make_scene(
    geometry_id: bytes,
    dimension: int,
    physics: Physics,
    positions: object,
    strengths: object | None = None,
) -> Scene:
    _check_dimension(dimension)
    points = Field.numeric("driver_position", T.FLOAT64, positions)
    if len(points.shape) == 1 and points.shape == (0,):
        points = Field.numeric("driver_position", T.FLOAT64, np.empty((0, dimension)))
    if len(points.shape) != 2 or points.shape[1] != dimension:
        raise ValueError("positions must be [K,d]")
    position_array = points.as_array()
    if physics.codec_id == 0:
        if strengths is not None:
            raise ValueError("binary scene cannot carry strengths")
        order = sorted(range(len(position_array)), key=lambda i: tuple(position_array[i]))
        strength_field = None
    else:
        if strengths is None:
            raise ValueError("additive scene requires strengths")
        raw_strengths = Field.numeric("driver_strength", T.FLOAT64, strengths)
        _shape(raw_strengths, (len(position_array),), T.FLOAT64)
        strength_array = raw_strengths.as_array()
        order = sorted(
            range(len(position_array)),
            key=lambda i: (*position_array[i], strength_array[i]),
        )
        strength_field = Field.numeric("driver_strength", T.FLOAT64, strength_array[order])
    position_field = Field.numeric("driver_position", T.FLOAT64, position_array[order])
    return Scene(geometry_id, dimension, physics, position_field, strength_field)


def make_physics(codec_id: int, **values: object) -> Physics:
    return Physics(codec_id, tuple(Field.numeric(name, T.FLOAT64, value) for name, value in values.items()))

