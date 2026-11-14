"""Typed Zarr v3 Structure-of-Arrays round trips for canonical scene records.

This module provides the M1 storage layer. A bundle written here is not a
published READY dataset until the later geometry and candidate gates run.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import zarr
from zarr.codecs import BloscCodec

from .contracts import (
    CODEC_NAMES, CSGNode, CSGOpcode, CodecID, Geometry, Leaf, Physics,
    RootKind, Scene, make_leaf, make_physics, make_scene,
)


UINT64_MAX = 2**64 - 1


def _little_dtype(dtype: str) -> np.dtype:
    return np.dtype(dtype)


def _array(values: object, dtype: str, tail: tuple[int, ...] = ()) -> np.ndarray:
    result = np.asarray(values, dtype=_little_dtype(dtype))
    if result.size == 0:
        return np.empty((0, *tail), dtype=_little_dtype(dtype))
    if result.shape[1:] != tail:
        raise ValueError(f"expected trailing shape {tail}, got {result.shape[1:]}")
    return np.ascontiguousarray(result)


def _stack(fields: Iterable[np.ndarray], dtype: str, tail: tuple[int, ...]) -> np.ndarray:
    items = list(fields)
    if not items:
        return np.empty((0, *tail), dtype=_little_dtype(dtype))
    return np.ascontiguousarray(np.stack(items, axis=0), dtype=_little_dtype(dtype))


def _cat(fields: Iterable[np.ndarray], dtype: str, tail: tuple[int, ...] = ()) -> np.ndarray:
    items = list(fields)
    if not items:
        return np.empty((0, *tail), dtype=_little_dtype(dtype))
    return np.ascontiguousarray(np.concatenate(items, axis=0), dtype=_little_dtype(dtype))


def _hash_matrix(hashes: Iterable[bytes]) -> np.ndarray:
    hashes = list(hashes)
    if not hashes:
        return np.empty((0, 32), dtype=np.uint8)
    if any(len(hash_value) != 32 for hash_value in hashes):
        raise ValueError("content hash must have 32 bytes")
    return np.frombuffer(b"".join(hashes), dtype=np.uint8).reshape(-1, 32).copy()


def _hash_at(matrix: np.ndarray, index: int) -> bytes:
    return np.asarray(matrix[index], dtype=np.uint8).tobytes()


def _pack_variable(leaves: list[Leaf], field_name: str, dtype: str,
                   tail: tuple[int, ...] = ()) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    content: list[np.ndarray] = []
    for leaf in leaves:
        data = leaf.field(field_name).as_array()
        content.append(data)
        offsets.append(offsets[-1] + len(data))
    return _array(offsets, "<u8"), _cat(content, dtype, tail)


def _pack_nested(leaves: list[Leaf], local_offset_name: str,
                 flat_field_name: str, flat_dtype: str,
                 tail: tuple[int, ...] = ()) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    object_offsets = [0]
    nested_offsets = [0]
    content: list[np.ndarray] = []
    for leaf in leaves:
        local_offsets = leaf.field(local_offset_name).as_array()
        values = leaf.field(flat_field_name).as_array()
        object_offsets.append(object_offsets[-1] + len(local_offsets) - 1)
        base = nested_offsets[-1]
        nested_offsets.extend(base + int(local) for local in local_offsets[1:])
        content.append(values)
    return (_array(object_offsets, "<u8"), _array(nested_offsets, "<u8"),
            _cat(content, flat_dtype, tail))


@dataclass(slots=True)
class CanonicalBundle:
    """Insertion-ordered content with collision checking and explicit refs."""

    dimension: int
    references: list[Leaf | CSGNode] = field(default_factory=list)
    geometries: list[Geometry] = field(default_factory=list)
    scenes: list[Scene] = field(default_factory=list)
    _refs_by_hash: dict[bytes, tuple[bytes, int]] = field(default_factory=dict)
    _geometry_by_hash: dict[bytes, tuple[bytes, int]] = field(default_factory=dict)
    _scene_by_hash: dict[bytes, tuple[bytes, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dimension not in (2, 3):
            raise ValueError("bundle dimension must be 2 or 3")

    def add_leaf(self, leaf: Leaf) -> int:
        if leaf.dimension != self.dimension:
            raise ValueError("leaf dimension differs from bundle dimension")
        return self._add_ref(leaf)

    def add_node(self, node: CSGNode) -> int:
        if node.dimension != self.dimension:
            raise ValueError("CSG dimension differs from bundle dimension")
        for child in node.children:
            if child not in self._refs_by_hash:
                raise ValueError("CSG children must precede their parent")
        return self._add_ref(node)

    def _add_ref(self, item: Leaf | CSGNode) -> int:
        content_hash = item.content_hash
        content_frame = item.content_frame
        previous = self._refs_by_hash.get(content_hash)
        if previous is not None:
            if previous[0] != content_frame:
                raise ValueError("content hash collision: reference bytes differ")
            return previous[1]
        row = len(self.references)
        self.references.append(item)
        self._refs_by_hash[content_hash] = (content_frame, row)
        return row

    def add_geometry(self, geometry: Geometry) -> int:
        if geometry.dimension != self.dimension:
            raise ValueError("geometry dimension differs from bundle dimension")
        if geometry.root_kind == RootKind.REF and geometry.root_hash not in self._refs_by_hash:
            raise ValueError("unknown geometry root reference")
        content_hash = geometry.content_hash
        content_frame = geometry.content_frame
        previous = self._geometry_by_hash.get(content_hash)
        if previous is not None:
            if previous[0] != content_frame:
                raise ValueError("geometry hash collision: framed bytes differ")
            return previous[1]
        row = len(self.geometries)
        self.geometries.append(geometry)
        self._geometry_by_hash[content_hash] = (content_frame, row)
        return row

    def add_scene(self, scene: Scene) -> int:
        if scene.dimension != self.dimension:
            raise ValueError("scene dimension differs from bundle dimension")
        if scene.geometry_id not in self._geometry_by_hash:
            raise ValueError("unknown geometry ID in scene")
        content_hash = scene.content_hash
        content_frame = scene.content_frame
        previous = self._scene_by_hash.get(content_hash)
        if previous is not None:
            if previous[0] != content_frame:
                raise ValueError("scene hash collision: framed bytes differ")
            return previous[1]
        row = len(self.scenes)
        self.scenes.append(scene)
        self._scene_by_hash[content_hash] = (content_frame, row)
        return row


def pack_arrays(bundle: CanonicalBundle) -> dict[str, np.ndarray]:
    """Pack semantic records into the complete spec §16 SoA logical schema."""
    d = bundle.dimension
    arrays: dict[str, np.ndarray] = {}
    leaf_groups = {codec: [] for codec in CodecID}
    csg_nodes: list[CSGNode] = []
    ref_kind: list[int] = []
    codec_id: list[int] = []
    codec_row: list[int] = []
    ref_index: dict[bytes, int] = {}
    for index, item in enumerate(bundle.references):
        ref_index[item.content_hash] = index
        if isinstance(item, Leaf):
            codec = item.codec
            ref_kind.append(0)
            codec_id.append(int(codec))
            codec_row.append(len(leaf_groups[codec]))
            leaf_groups[codec].append(item)
        else:
            ref_kind.append(1)
            codec_id.append(0)
            codec_row.append(len(csg_nodes))
            csg_nodes.append(item)
    arrays["geometry_ref/ref_hash"] = _hash_matrix(
        item.content_hash for item in bundle.references
    )
    arrays["geometry_ref/ref_kind"] = _array(ref_kind, "u1")
    arrays["geometry_ref/codec_id"] = _array(codec_id, "<u2")
    arrays["geometry_ref/codec_row"] = _array(codec_row, "<u8")

    arrays["geometry/geometry_id"] = _hash_matrix(
        geometry.content_hash for geometry in bundle.geometries
    )
    arrays["geometry/dimension"] = _array([d] * len(bundle.geometries), "u1")
    arrays["geometry/root_kind"] = _array(
        [int(geometry.root_kind) for geometry in bundle.geometries], "u1"
    )
    arrays["geometry/root_ref"] = _array([
        UINT64_MAX if geometry.root_kind == RootKind.EMPTY
        else ref_index[geometry.root_hash]
        for geometry in bundle.geometries
    ], "<u8")

    for codec, prefix in ((CodecID.AFFINE_BOX, "affine_box"),
                          (CodecID.AFFINE_BALL, "affine_ball")):
        leaves = leaf_groups[codec]
        base = f"geometry_codecs/{prefix}/"
        arrays[base + "center"] = _stack(
            (leaf.field("center").as_array() for leaf in leaves), "<f8", (d,)
        )
        arrays[base + "matrix"] = _stack(
            (leaf.field("matrix").as_array() for leaf in leaves), "<f8", (d, d)
        )
        arrays[base + "leaf_hash"] = _hash_matrix(leaf.content_hash for leaf in leaves)

    leaves = leaf_groups[CodecID.POLYGON_RINGS_2D]
    base = "geometry_codecs/polygon_rings_2d/"
    arrays[base + "center"] = _stack((leaf.field("center").as_array() for leaf in leaves), "<f8", (2,))
    arrays[base + "matrix"] = _stack((leaf.field("matrix").as_array() for leaf in leaves), "<f8", (2, 2))
    obj, nested, vertices = _pack_nested(leaves, "ring_vertex_offsets", "vertices", "<f8", (2,))
    arrays[base + "object_ring_offsets"] = obj
    arrays[base + "ring_vertex_offsets"] = nested
    arrays[base + "vertices"] = vertices
    arrays[base + "ring_role"] = _cat((leaf.field("ring_role").as_array() for leaf in leaves), "u1")
    arrays[base + "ring_owner"] = _cat((leaf.field("ring_owner").as_array() for leaf in leaves), "<u4")
    arrays[base + "leaf_hash"] = _hash_matrix(leaf.content_hash for leaf in leaves)

    leaves = leaf_groups[CodecID.CONVEX_HALFSPACES]
    base = "geometry_codecs/convex_halfspaces/"
    arrays[base + "center"] = _stack((leaf.field("center").as_array() for leaf in leaves), "<f8", (d,))
    arrays[base + "matrix"] = _stack((leaf.field("matrix").as_array() for leaf in leaves), "<f8", (d, d))
    plane_offsets, normal = _pack_variable(leaves, "normal", "<f8", (d,))
    arrays[base + "object_plane_offsets"] = plane_offsets
    arrays[base + "normal"] = normal
    arrays[base + "plane_offset"] = _cat((leaf.field("plane_offset").as_array() for leaf in leaves), "<f8")
    arrays[base + "leaf_hash"] = _hash_matrix(leaf.content_hash for leaf in leaves)

    leaves = leaf_groups[CodecID.ORIENTED_MESH_3D]
    base = "geometry_codecs/oriented_mesh_3d/"
    arrays[base + "center"] = _stack((leaf.field("center").as_array() for leaf in leaves), "<f8", (3,))
    arrays[base + "matrix"] = _stack((leaf.field("matrix").as_array() for leaf in leaves), "<f8", (3, 3))
    vertex_offsets, vertices = _pack_variable(leaves, "vertices", "<f8", (3,))
    arrays[base + "object_vertex_offsets"] = vertex_offsets
    arrays[base + "vertices"] = vertices
    obj, shell, triangles = _pack_nested(leaves, "shell_triangle_offsets", "triangles", "<u4", (3,))
    arrays[base + "object_shell_offsets"] = obj
    arrays[base + "shell_triangle_offsets"] = shell
    arrays[base + "triangles"] = triangles
    arrays[base + "shell_role"] = _cat((leaf.field("shell_role").as_array() for leaf in leaves), "u1")
    arrays[base + "shell_owner"] = _cat((leaf.field("shell_owner").as_array() for leaf in leaves), "<u4")
    arrays[base + "leaf_hash"] = _hash_matrix(leaf.content_hash for leaf in leaves)

    leaves = leaf_groups[CodecID.RADIAL_FOURIER_2D]
    base = "geometry_codecs/radial_fourier_2d/"
    arrays[base + "center"] = _stack((leaf.field("center").as_array() for leaf in leaves), "<f8", (2,))
    arrays[base + "matrix"] = _stack((leaf.field("matrix").as_array() for leaf in leaves), "<f8", (2, 2))
    offsets, a = _pack_variable(leaves, "coeff_a", "<f8")
    arrays[base + "coefficient_offsets"] = offsets
    arrays[base + "coeff_a"] = a
    arrays[base + "coeff_b"] = _cat((leaf.field("coeff_b").as_array() for leaf in leaves), "<f8")
    arrays[base + "leaf_hash"] = _hash_matrix(leaf.content_hash for leaf in leaves)

    leaves = leaf_groups[CodecID.RADIAL_FOURIER_SPHERE]
    base = "geometry_codecs/radial_fourier_sphere/"
    offsets, freq = _pack_variable(leaves, "frequency", "<f8", (3,))
    arrays[base + "center"] = _stack((leaf.field("center").as_array() for leaf in leaves), "<f8", (3,))
    arrays[base + "matrix"] = _stack((leaf.field("matrix").as_array() for leaf in leaves), "<f8", (3, 3))
    arrays[base + "feature_offsets"] = offsets
    arrays[base + "frequency"] = freq
    arrays[base + "coeff_a"] = _cat((leaf.field("coeff_a").as_array() for leaf in leaves), "<f8")
    arrays[base + "coeff_b"] = _cat((leaf.field("coeff_b").as_array() for leaf in leaves), "<f8")
    arrays[base + "leaf_hash"] = _hash_matrix(leaf.content_hash for leaf in leaves)

    leaves = leaf_groups[CodecID.COSINE_FIELD]
    base = "geometry_codecs/cosine_field/"
    offsets, freq = _pack_variable(leaves, "frequency", "<f8", (d,))
    arrays[base + "feature_offsets"] = offsets
    arrays[base + "frequency"] = freq
    arrays[base + "amplitude"] = _cat((leaf.field("amplitude").as_array() for leaf in leaves), "<f8")
    arrays[base + "phase"] = _cat((leaf.field("phase").as_array() for leaf in leaves), "<f8")
    arrays[base + "threshold"] = _stack((leaf.field("threshold").as_array() for leaf in leaves), "<f8", ())
    arrays[base + "leaf_hash"] = _hash_matrix(leaf.content_hash for leaf in leaves)

    child_offsets = [0]
    child_refs: list[int] = []
    for node in csg_nodes:
        child_refs.extend(ref_index[child] for child in node.children)
        child_offsets.append(len(child_refs))
    arrays["geometry_csg/opcode"] = _array([int(node.opcode) for node in csg_nodes], "u1")
    arrays["geometry_csg/child_offsets"] = _array(child_offsets, "<u8")
    arrays["geometry_csg/child_ref"] = _array(child_refs, "<u8")
    arrays["geometry_csg/node_hash"] = _hash_matrix(node.content_hash for node in csg_nodes)

    physics_rows = {1: [], 2: []}
    physics_lookup: dict[bytes, tuple[bytes, int]] = {}
    scene_driver_offsets = [0]
    scene_strength_offsets = [0]
    positions: list[np.ndarray] = []
    strengths: list[np.ndarray] = []
    physics_local_rows = []
    for scene in bundle.scenes:
        position_array = scene.driver_position.as_array()
        positions.append(position_array)
        scene_driver_offsets.append(scene_driver_offsets[-1] + len(position_array))
        if scene.driver_strength is None:
            scene_strength_offsets.append(scene_strength_offsets[-1])
        else:
            strength_array = scene.driver_strength.as_array()
            strengths.append(strength_array)
            scene_strength_offsets.append(scene_strength_offsets[-1] + len(strength_array))
        if scene.physics.codec_id == 0:
            physics_local_rows.append(UINT64_MAX)
        else:
            content_hash = scene.physics.content_hash
            content_frame = scene.physics.semantic_fields
            previous = physics_lookup.get(content_hash)
            if previous is not None:
                if previous[0] != content_frame:
                    raise ValueError("physics hash collision")
                physics_local_rows.append(previous[1])
            else:
                row = len(physics_rows[scene.physics.codec_id])
                physics_rows[scene.physics.codec_id].append(scene.physics)
                physics_lookup[content_hash] = (content_frame, row)
                physics_local_rows.append(row)
    arrays["scene_core/scene_id"] = _hash_matrix(scene.content_hash for scene in bundle.scenes)
    arrays["scene_core/geometry_row"] = _array(
        [bundle._geometry_by_hash[scene.geometry_id][1] for scene in bundle.scenes], "<u8"
    )
    arrays["scene_core/driver_offsets"] = _array(scene_driver_offsets, "<u8")
    arrays["scene_core/driver_position"] = _cat(positions, "<f8", (d,))
    arrays["scene_core/strength_offsets"] = _array(scene_strength_offsets, "<u8")
    arrays["scene_core/driver_strength"] = _cat(strengths, "<f8")
    arrays["scene_core/physics_codec_id"] = _array([scene.physics.codec_id for scene in bundle.scenes], "u1")
    arrays["scene_core/physics_local_row"] = _array(physics_local_rows, "<u8")

    exponential = physics_rows[1]
    log_pchip = physics_rows[2]
    arrays["scene_core/physics_exponential/tau"] = _stack(
        (physics.fields[0].as_array() for physics in exponential), "<f8", ()
    )
    arrays["scene_core/physics_exponential/payload_hash"] = _hash_matrix(
        physics.content_hash for physics in exponential
    )
    arrays["scene_core/physics_log_pchip/knots"] = _stack(
        (next(field for field in physics.fields if field.name == "knots").as_array()
         for physics in log_pchip), "<f8", (9,)
    )
    arrays["scene_core/physics_log_pchip/log_values"] = _stack(
        (next(field for field in physics.fields if field.name == "log_values").as_array()
         for physics in log_pchip), "<f8", (9,)
    )
    arrays["scene_core/physics_log_pchip/payload_hash"] = _hash_matrix(
        physics.content_hash for physics in log_pchip
    )
    return arrays


def _validate_typed_array(path: str, array: np.ndarray) -> None:
    if array.ndim < 1 or array.dtype.kind not in "iuf":
        raise ValueError(f"{path}: expected typed, at least rank-one array")
    if array.dtype.byteorder == ">":
        raise ValueError(f"{path}: big-endian canonical array")
    if array.dtype.kind == "f":
        if array.dtype != np.dtype("<f8") or not np.isfinite(array).all():
            raise ValueError(f"{path}: nonfinite or non-binary64 real array")
        bits = array.view("<u8")
        if np.any(bits == 0x8000000000000000):
            raise ValueError(f"{path}: negative zero is not canonical")


async def _ticker() -> None:
    # The current local Python runtime occasionally loses executor wakeups.
    # This periodic event-loop wake also makes LocalStore operations complete.
    while True:
        await asyncio.sleep(0.01)


def _run_async(coroutine: object) -> object:
    loop = asyncio.new_event_loop()
    ticker = loop.create_task(_ticker())
    try:
        return loop.run_until_complete(coroutine)
    finally:
        ticker.cancel()
        loop.run_until_complete(asyncio.gather(ticker, return_exceptions=True))
        loop.close()


def _physical_shape(array: np.ndarray) -> tuple[tuple[int, ...], tuple[int, ...]]:
    row_bytes = max(1, array.dtype.itemsize * int(np.prod(array.shape[1:], dtype=np.int64)))
    chunk_rows = max(1, min(max(1, array.shape[0]), 1_048_576 // row_bytes))
    tail = tuple(max(1, int(size)) for size in array.shape[1:])
    chunks = (chunk_rows, *tail)
    shards = (chunk_rows * 32, *tail)
    return chunks, shards


async def _write_arrays_async(path: Path, arrays: dict[str, np.ndarray]) -> None:
    group = await zarr.api.asynchronous.open_group(str(path), mode="w", zarr_format=3)
    for name, array in sorted(arrays.items()):
        _validate_typed_array(name, array)
        chunks, shards = _physical_shape(array)
        created = await group.create_array(
            name,
            data=array,
            chunks=chunks,
            shards=shards,
            compressors=[BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")],
        )
        if created.metadata.to_dict()["codecs"][0]["name"] != "sharding_indexed":
            raise AssertionError(f"{name}: Zarr v3 indexed sharding missing")


def write_bundle(path: str | Path, bundle: CanonicalBundle) -> None:
    """Write an M1 scratch SoA; it is never marked READY by this function."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = pack_arrays(bundle)
    _run_async(_write_arrays_async(path, arrays))


async def _read_arrays_async(path: Path, names: Iterable[str]) -> dict[str, np.ndarray]:
    group = await zarr.api.asynchronous.open_group(str(path), mode="r", zarr_format=3)
    arrays: dict[str, np.ndarray] = {}
    for name in sorted(names):
        node = await group.getitem(name)
        array = np.asarray(await node.getitem(slice(None)))
        _validate_typed_array(name, array)
        arrays[name] = array
    return arrays


def _offset_slice(offsets: np.ndarray, row: int) -> slice:
    return slice(int(offsets[row]), int(offsets[row + 1]))


def _local_offsets(offsets: np.ndarray, first: int, last: int) -> np.ndarray:
    return np.asarray(offsets[first:last + 1] - offsets[first], dtype="<u8")


def _decode_leaf(codec: CodecID, row: int, d: int, arrays: dict[str, np.ndarray]) -> Leaf:
    if codec in (CodecID.AFFINE_BOX, CodecID.AFFINE_BALL):
        base = "geometry_codecs/affine_box/" if codec == CodecID.AFFINE_BOX else "geometry_codecs/affine_ball/"
        kwargs = {name: arrays[base + name][row] for name in ("center", "matrix")}
    elif codec == CodecID.POLYGON_RINGS_2D:
        base = "geometry_codecs/polygon_rings_2d/"
        r = _offset_slice(arrays[base + "object_ring_offsets"], row)
        v = slice(int(arrays[base + "ring_vertex_offsets"][r.start]),
                  int(arrays[base + "ring_vertex_offsets"][r.stop]))
        kwargs = {
            "center": arrays[base + "center"][row],
            "matrix": arrays[base + "matrix"][row],
            "vertices": arrays[base + "vertices"][v],
            "ring_vertex_offsets": _local_offsets(arrays[base + "ring_vertex_offsets"], r.start, r.stop),
            "ring_role": arrays[base + "ring_role"][r],
            "ring_owner": arrays[base + "ring_owner"][r],
        }
    elif codec == CodecID.CONVEX_HALFSPACES:
        base = "geometry_codecs/convex_halfspaces/"
        p = _offset_slice(arrays[base + "object_plane_offsets"], row)
        kwargs = {"center": arrays[base + "center"][row], "matrix": arrays[base + "matrix"][row],
                  "normal": arrays[base + "normal"][p], "plane_offset": arrays[base + "plane_offset"][p]}
    elif codec == CodecID.ORIENTED_MESH_3D:
        base = "geometry_codecs/oriented_mesh_3d/"
        v = _offset_slice(arrays[base + "object_vertex_offsets"], row)
        sh = _offset_slice(arrays[base + "object_shell_offsets"], row)
        t = slice(int(arrays[base + "shell_triangle_offsets"][sh.start]),
                  int(arrays[base + "shell_triangle_offsets"][sh.stop]))
        kwargs = {"center": arrays[base + "center"][row], "matrix": arrays[base + "matrix"][row],
                  "vertices": arrays[base + "vertices"][v], "triangles": arrays[base + "triangles"][t],
                  "shell_triangle_offsets": _local_offsets(arrays[base + "shell_triangle_offsets"], sh.start, sh.stop),
                  "shell_role": arrays[base + "shell_role"][sh], "shell_owner": arrays[base + "shell_owner"][sh]}
    elif codec == CodecID.RADIAL_FOURIER_2D:
        base = "geometry_codecs/radial_fourier_2d/"
        f = _offset_slice(arrays[base + "coefficient_offsets"], row)
        kwargs = {"center": arrays[base + "center"][row], "matrix": arrays[base + "matrix"][row],
                  "coeff_a": arrays[base + "coeff_a"][f], "coeff_b": arrays[base + "coeff_b"][f]}
    elif codec == CodecID.RADIAL_FOURIER_SPHERE:
        base = "geometry_codecs/radial_fourier_sphere/"
        f = _offset_slice(arrays[base + "feature_offsets"], row)
        kwargs = {"center": arrays[base + "center"][row], "matrix": arrays[base + "matrix"][row],
                  "frequency": arrays[base + "frequency"][f], "coeff_a": arrays[base + "coeff_a"][f],
                  "coeff_b": arrays[base + "coeff_b"][f]}
    elif codec == CodecID.COSINE_FIELD:
        base = "geometry_codecs/cosine_field/"
        f = _offset_slice(arrays[base + "feature_offsets"], row)
        kwargs = {"amplitude": arrays[base + "amplitude"][f],
                  "frequency": arrays[base + "frequency"][f], "phase": arrays[base + "phase"][f],
                  "threshold": arrays[base + "threshold"][row]}
    else:
        raise ValueError(f"unknown codec ID {codec}")
    leaf = make_leaf(codec, d, **kwargs)
    if leaf.content_hash != _hash_at(arrays[base + "leaf_hash"], row):
        raise ValueError(f"{CODEC_NAMES[codec]} row {row}: leaf hash mismatch")
    return leaf


def _decode_bundle(arrays: dict[str, np.ndarray], dimension: int) -> CanonicalBundle:
    bundle = CanonicalBundle(dimension)
    ref_hash = arrays["geometry_ref/ref_hash"]
    ref_kind = arrays["geometry_ref/ref_kind"]
    codec_ids = arrays["geometry_ref/codec_id"]
    codec_rows = arrays["geometry_ref/codec_row"]
    csg_hash = arrays["geometry_csg/node_hash"]
    csg_opcode = arrays["geometry_csg/opcode"]
    csg_offsets = arrays["geometry_csg/child_offsets"]
    csg_child = arrays["geometry_csg/child_ref"]
    leaves_by_codec: dict[CodecID, list[Leaf]] = {}
    for codec in CodecID:
        prefix = {
            CodecID.AFFINE_BOX: "affine_box", CodecID.AFFINE_BALL: "affine_ball",
            CodecID.POLYGON_RINGS_2D: "polygon_rings_2d", CodecID.CONVEX_HALFSPACES: "convex_halfspaces",
            CodecID.ORIENTED_MESH_3D: "oriented_mesh_3d", CodecID.RADIAL_FOURIER_2D: "radial_fourier_2d",
            CodecID.RADIAL_FOURIER_SPHERE: "radial_fourier_sphere", CodecID.COSINE_FIELD: "cosine_field",
        }[codec]
        count = len(arrays[f"geometry_codecs/{prefix}/leaf_hash"])
        if (codec in (CodecID.POLYGON_RINGS_2D, CodecID.RADIAL_FOURIER_2D) and dimension != 2
                or codec in (CodecID.ORIENTED_MESH_3D, CodecID.RADIAL_FOURIER_SPHERE) and dimension != 3):
            if count:
                raise ValueError(f"{codec.name} is not valid in dimension {dimension}")
            leaves_by_codec[codec] = []
        else:
            leaves_by_codec[codec] = [_decode_leaf(codec, row, dimension, arrays) for row in range(count)]
    if not (len(ref_hash) == len(ref_kind) == len(codec_ids) == len(codec_rows)):
        raise ValueError("geometry reference table lengths disagree")
    for index, kind in enumerate(ref_kind):
        if kind == 0:
            try:
                codec = CodecID(int(codec_ids[index]))
            except ValueError as error:
                raise ValueError("unknown geometry codec ID") from error
            row = int(codec_rows[index])
            if row >= len(leaves_by_codec[codec]):
                raise ValueError("leaf codec row out of range")
            item = leaves_by_codec[codec][row]
            if bundle.add_leaf(item) != index:
                raise ValueError("duplicate or reordered reference row")
        elif kind == 1:
            if codec_ids[index] != 0:
                raise ValueError("CSG reference must have codec ID zero")
            row = int(codec_rows[index])
            if row >= len(csg_hash):
                raise ValueError("CSG node row out of range")
            child_slice = _offset_slice(csg_offsets, row)
            children = []
            for child_ref in csg_child[child_slice]:
                if int(child_ref) >= index:
                    raise ValueError("CSG child reference must precede parent")
                children.append(bundle.references[int(child_ref)].content_hash)
            try:
                node = CSGNode(dimension, CSGOpcode(int(csg_opcode[row])), tuple(children))
            except ValueError as error:
                raise ValueError("invalid CSG opcode or arity") from error
            if node.content_hash != _hash_at(csg_hash, row):
                raise ValueError("CSG node hash mismatch")
            if bundle.add_node(node) != index:
                raise ValueError("duplicate or reordered CSG reference")
            item = node
        else:
            raise ValueError("unknown geometry reference kind")
        if item.content_hash != _hash_at(ref_hash, index):
            raise ValueError("geometry reference hash mismatch")

    ids = arrays["geometry/geometry_id"]
    kinds = arrays["geometry/root_kind"]
    roots = arrays["geometry/root_ref"]
    dims = arrays["geometry/dimension"]
    if not (len(ids) == len(kinds) == len(roots) == len(dims)):
        raise ValueError("geometry table lengths disagree")
    for index in range(len(ids)):
        if int(dims[index]) != dimension:
            raise ValueError("geometry dimension differs from store")
        try:
            kind = RootKind(int(kinds[index]))
        except ValueError as error:
            raise ValueError("unknown geometry root kind") from error
        if kind == RootKind.EMPTY:
            if int(roots[index]) != UINT64_MAX:
                raise ValueError("EMPTY root must use UINT64_MAX")
            geometry = Geometry.empty(dimension)
        else:
            if int(roots[index]) >= len(bundle.references):
                raise ValueError("geometry root reference out of range")
            geometry = Geometry(dimension, kind, bundle.references[int(roots[index])].content_hash)
        if bundle.add_geometry(geometry) != index or geometry.content_hash != _hash_at(ids, index):
            raise ValueError("geometry ID mismatch or duplicate geometry row")

    pexp = arrays["scene_core/physics_exponential/tau"]
    pexp_hash = arrays["scene_core/physics_exponential/payload_hash"]
    pp_knots = arrays["scene_core/physics_log_pchip/knots"]
    pp_log = arrays["scene_core/physics_log_pchip/log_values"]
    pp_hash = arrays["scene_core/physics_log_pchip/payload_hash"]
    if len(pexp) != len(pexp_hash) or not (len(pp_knots) == len(pp_log) == len(pp_hash)):
        raise ValueError("physics table lengths disagree")
    physics: dict[int, list[Physics]] = {
        1: [make_physics(1, tau=value) for value in pexp],
        2: [make_physics(2, knots=knots, log_values=logs)
            for knots, logs in zip(pp_knots, pp_log, strict=True)],
    }
    for codec, hash_array in ((1, pexp_hash), (2, pp_hash)):
        for row, item in enumerate(physics[codec]):
            if item.content_hash != _hash_at(hash_array, row):
                raise ValueError("physics payload hash mismatch")

    scene_ids = arrays["scene_core/scene_id"]
    geometry_rows = arrays["scene_core/geometry_row"]
    driver_offsets = arrays["scene_core/driver_offsets"]
    driver_positions = arrays["scene_core/driver_position"]
    strength_offsets = arrays["scene_core/strength_offsets"]
    driver_strengths = arrays["scene_core/driver_strength"]
    physics_ids = arrays["scene_core/physics_codec_id"]
    physics_rows = arrays["scene_core/physics_local_row"]
    count = len(scene_ids)
    if not (len(geometry_rows) == len(physics_ids) == len(physics_rows) == count):
        raise ValueError("scene table lengths disagree")
    for row in range(count):
        if int(geometry_rows[row]) >= len(bundle.geometries):
            raise ValueError("scene geometry row out of range")
        codec = int(physics_ids[row])
        if codec == 0:
            if int(physics_rows[row]) != UINT64_MAX:
                raise ValueError("binary scene must have no physics table row")
            physics_item = Physics(0)
        elif codec in physics:
            if int(physics_rows[row]) >= len(physics[codec]):
                raise ValueError("physics local row out of range")
            physics_item = physics[codec][int(physics_rows[row])]
        else:
            raise ValueError("unknown physics codec")
        ds = _offset_slice(driver_offsets, row)
        ss = _offset_slice(strength_offsets, row)
        if codec == 0 and ss.start != ss.stop:
            raise ValueError("binary scene has strengths")
        if codec != 0 and (ss.stop - ss.start) != (ds.stop - ds.start):
            raise ValueError("additive driver and strength counts differ")
        scene = make_scene(bundle.geometries[int(geometry_rows[row])].content_hash,
                           dimension, physics_item, driver_positions[ds],
                           None if codec == 0 else driver_strengths[ss])
        if bundle.add_scene(scene) != row or scene.content_hash != _hash_at(scene_ids, row):
            raise ValueError("scene ID mismatch or duplicate scene row")
    return bundle


def read_bundle(path: str | Path, dimension: int) -> CanonicalBundle:
    """Reload all SoA tables, prove references/hashes, and reject silent repair."""
    expected = pack_arrays(CanonicalBundle(dimension))
    arrays = _run_async(_read_arrays_async(Path(path), expected))
    for name, empty in expected.items():
        actual = arrays[name]
        if actual.dtype != empty.dtype or actual.shape[1:] != empty.shape[1:]:
            raise ValueError(f"{name}: dtype or trailing shape mismatch")
    bundle = _decode_bundle(arrays, dimension)
    repacked = pack_arrays(bundle)
    for name, actual in arrays.items():
        if actual.shape != repacked[name].shape or not np.array_equal(actual, repacked[name]):
            raise ValueError(f"{name}: reloaded SoA differs from canonical repacking")
    return bundle
