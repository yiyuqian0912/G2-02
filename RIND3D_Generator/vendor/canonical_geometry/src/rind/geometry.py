"""Public structure-only query facade for supported algebraic scopes.

The facade intentionally accepts geometry records, never scenes or drivers.
All query coordinates are first converted to binary64; the native backend then
interprets their exact dyadic rational values.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
import math
from hashlib import sha256
from pathlib import Path
from functools import lru_cache

from .contracts import CSGNode, CSGOpcode, CodecID, Leaf, RootKind
from .ellipsoid3d import ExactEllipsoid3D
from .hybrid3d import HybridGeometry3D
from .storage import CanonicalBundle

try:
    from ._exact2d import ExactGeometry2D
except ImportError:  # The storage package remains usable before native build.
    ExactGeometry2D = None
try:
    from ._exact3d import ExactGeometry3D
except ImportError:
    ExactGeometry3D = None


@lru_cache(maxsize=2)
def backend_build_id(dimension: int = 2) -> str | None:
    if dimension == 2 and ExactGeometry2D is not None:
        from . import _exact2d as native
        source_name = "exact2d.cpp"
    elif dimension == 3 and ExactGeometry3D is not None:
        from . import _exact3d as native
        source_name = "exact3d.cpp"
    else:
        return None
    project = Path(__file__).resolve().parents[2]
    inputs = (Path(__file__), project / "native" / source_name,
              project / "native" / "CMakeLists.txt")
    if dimension == 3:
        inputs += (Path(__file__).with_name("ellipsoid3d.py"),
                   Path(__file__).with_name("hybrid3d.py"))
    source = b"".join(path.read_bytes() for path in inputs)
    source += native.backend_version.encode("ascii")
    if dimension == 3:
        import z3
        source += z3.get_version_string().encode("ascii")
    return f"cgalscope{dimension}d-" + sha256(source).hexdigest()[:24]


class QueryStatus(IntEnum):
    OK = 0
    OUTSIDE_DOMAIN = 1
    INVALID_INPUT = 2
    IN_STRUCTURE = 3
    ON_BOUNDARY = 4
    UNRESOLVED = 5
    UNSUPPORTED = 6
    ABORTED = 7


class Membership(IntEnum):
    FREE = 0
    INTERIOR = 1
    BOUNDARY = 2
    UNKNOWN = 255


class BoxClass(IntEnum):
    FREE = 0
    SOLID = 1
    MIXED = 2
    UNKNOWN = 255


class Clearance(IntEnum):
    LE = 0
    GT = 1
    UNRESOLVED = 255


class SegmentDecision(IntEnum):
    MISS = 0
    HIT = 1
    UNKNOWN = 255


@dataclass(frozen=True, slots=True)
class QueryResult:
    status: np.ndarray
    value: np.ndarray
    valid: np.ndarray
    reason_code: tuple[str, ...]
    evidence_ref: tuple[Any | None, ...]
    work: tuple[dict[str, int], ...]


@dataclass(frozen=True, slots=True)
class DistanceResult:
    status: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    empty_set: np.ndarray
    valid: np.ndarray
    reason_code: tuple[str, ...]
    work: tuple[dict[str, int], ...]


class UnsupportedGeometryError(ValueError):
    pass


def _points(values: object, dimension: int) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("coordinates cannot be converted to binary64") from exc
    if result.ndim == 1:
        result = result[None, :]
    if result.ndim != 2 or result.shape[1] != dimension:
        raise ValueError(f"expected [m,{dimension}] coordinates")
    return result


def _result(status: list[int], value: list[int], reasons: list[str],
            evidence: list[Any | None] | None = None,
            valid_override: list[bool] | None = None,
            work_units: list[int] | None = None) -> QueryResult:
    if evidence is None:
        evidence = [None] * len(status)
    valid = (valid_override if valid_override is not None else
             [s in (QueryStatus.OK, QueryStatus.ON_BOUNDARY) for s in status])
    if work_units is None:
        work_units = [0] * len(status)
    return QueryResult(
        np.asarray(status, dtype=np.uint8),
        np.asarray(value, dtype=np.uint8),
        np.asarray(valid, dtype=np.bool_),
        tuple(reasons), tuple(evidence),
        tuple({"precision_bits": 0, "subdivisions": 0,
               "branches": 0, "backend_work_units": int(work_units[i])}
              for i in range(len(status))),
    )


class GeometryView:
    """Build a certified geometry-only view from a canonical bundle.

    Algebraic 2D leaves and 3D polyhedral box/halfspace/mesh leaves support
    regularized CSG. 3D affine balls are supported as leaves in a top-level
    union; ambiguous mixed boundary and box queries return UNRESOLVED.
    """

    def __init__(self, geometry_id: bytes, geometry_reader: CanonicalBundle,
                 policy: dict[str, Any] | None = None):
        if len(geometry_id) != 32:
            raise ValueError("geometry ID must be 32 bytes")
        self.geometry_id = geometry_id
        self.policy = policy or {}
        self.max_events = int(self.policy.get("max_arrangement_events_per_item", 2**20))
        if self.max_events < 0:
            raise ValueError("max_arrangement_events_per_item must be nonnegative")
        self._bundle = geometry_reader
        geometry = next((item for item in geometry_reader.geometries
                         if item.content_hash == geometry_id), None)
        if geometry is None:
            raise KeyError("geometry ID not found")
        self.dimension = geometry.dimension
        if self.dimension != geometry_reader.dimension:
            raise ValueError("bundle and geometry dimension mismatch")
        self._backend = None
        self._unsupported_reason = ""
        native_class = ExactGeometry2D if self.dimension == 2 else ExactGeometry3D
        if native_class is None:
            self._unsupported_reason = (
                "native_exact_2d_not_built" if self.dimension == 2
                else "native_exact_3d_not_built"
            )
            return

        refs = {item.content_hash: item for item in geometry_reader.references}
        native = native_class()
        index_by_hash: dict[bytes, int] = {}

        def build(content_hash: bytes) -> int:
            if content_hash in index_by_hash:
                return index_by_hash[content_hash]
            item = refs.get(content_hash)
            if item is None:
                raise ValueError("missing referenced geometry record")
            if isinstance(item, CSGNode):
                children = [build(child) for child in item.children]
                index = native.add_csg(int(item.opcode), children)
            elif isinstance(item, Leaf):
                enabled = ((CodecID.AFFINE_BOX, CodecID.AFFINE_BALL,
                            CodecID.POLYGON_RINGS_2D, CodecID.CONVEX_HALFSPACES)
                           if self.dimension == 2 else
                           (CodecID.AFFINE_BOX, CodecID.CONVEX_HALFSPACES,
                            CodecID.ORIENTED_MESH_3D))
                if item.codec not in enabled:
                    raise UnsupportedGeometryError(f"codec_{item.codec.name.lower()}")
                center = item.field("center").as_array().tolist()
                matrix = item.field("matrix").as_array().ravel().tolist()
                if item.codec == CodecID.AFFINE_BOX:
                    index = native.add_box(center, matrix)
                elif item.codec == CodecID.AFFINE_BALL:
                    index = native.add_ball(center, matrix)
                elif item.codec == CodecID.POLYGON_RINGS_2D:
                    index = native.add_polygon(
                        center, matrix,
                        item.field("vertices").as_array().tolist(),
                        item.field("ring_vertex_offsets").as_array().tolist(),
                        item.field("ring_role").as_array().tolist(),
                        item.field("ring_owner").as_array().tolist(),
                    )
                elif item.codec == CodecID.CONVEX_HALFSPACES:
                    index = native.add_halfspaces(
                        center, matrix,
                        item.field("normal").as_array().tolist(),
                        item.field("plane_offset").as_array().tolist(),
                    )
                else:
                    index = native.add_mesh(
                        center, matrix,
                        item.field("vertices").as_array().tolist(),
                        item.field("triangles").as_array().tolist(),
                        item.field("shell_triangle_offsets").as_array().tolist(),
                        item.field("shell_role").as_array().tolist(),
                        item.field("shell_owner").as_array().tolist(),
                    )
            else:
                raise TypeError("unknown geometry reference")
            index_by_hash[content_hash] = index
            return index

        try:
            if self.dimension == 3 and geometry.root_kind != RootKind.EMPTY:
                ellipsoids: list[ExactEllipsoid3D] = []
                poly_roots: list[bytes] = []

                def collect_top_union(content_hash: bytes) -> None:
                    item = refs.get(content_hash)
                    if item is None:
                        raise ValueError("missing referenced geometry record")
                    if isinstance(item, CSGNode) and item.opcode == CSGOpcode.UNION:
                        for child in item.children:
                            collect_top_union(child)
                    elif isinstance(item, Leaf) and item.codec == CodecID.AFFINE_BALL:
                        ellipsoids.append(ExactEllipsoid3D(
                            item.field("center").as_array().tolist(),
                            item.field("matrix").as_array().tolist()))
                    else:
                        poly_roots.append(content_hash)

                collect_top_union(geometry.root_hash)
                roots = [build(content_hash) for content_hash in poly_roots]
                poly_root = (-1 if not roots else roots[0] if len(roots) == 1
                             else native.add_csg(int(CSGOpcode.UNION), roots))
                native.set_root(poly_root)
                if ellipsoids:
                    self._backend = HybridGeometry3D(
                        native, ellipsoids,
                        qfnra_rlimit=int(self.policy.get("qfnra_rlimit_per_item",
                                                        self.policy.get(
                                                            "max_solver_work_units_per_item",
                                                            4_194_304))))
                    return
            else:
                native.set_root(-1 if geometry.root_kind == RootKind.EMPTY
                                else build(geometry.root_hash))
        except UnsupportedGeometryError as exc:
            self._unsupported_reason = str(exc)
            return
        self._backend = native

    def _estimated_events(self, operation: str) -> int:
        if self._backend is None:
            return 0
        if self.dimension == 2:
            vertices, halfedges = self._backend.arrangement_counts()
            if operation == "point":
                return 1
            if operation == "segment":
                return halfedges + 1
            if operation == "clearance":
                return vertices + 2 * halfedges + 1
            if operation == "box":
                return vertices + 4 * halfedges + 1
        else:
            # This is only a deterministic outer gate; Nef internals are not
            # interruptible. A native operation may exceed this work estimate.
            return {"point": 1, "segment": 2, "clearance": 4,
                    "box": 4}[operation]
        raise ValueError("unknown event estimate")

    def capability(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "backend": (f"CGAL-exact-{'conic' if self.dimension == 2 else 'Nef+QF_NRA' if isinstance(self._backend, HybridGeometry3D) else 'Nef'}-{self.dimension}D"
                        if self._backend is not None else None),
            "version": (self._backend and __import__(
                f"rind._exact{self.dimension}d", fromlist=["backend_version"]).backend_version),
            "backend_build": backend_build_id(self.dimension) if self._backend is not None else None,
            "supported": self._backend is not None,
            "reason": self._unsupported_reason or None,
            "operations": ("classify_points", "classify_box", "segment_intersect",
                           "clearance_decide", "distance_query") if self._backend is not None else (),
            "evidence": "native exact algebraic witness for LE",
            "max_arrangement_events_per_item": self.max_events,
            "deterministic_outer_event_budget": True,
            "internal_cgal_calls_interruptible": False,
            "qfnra_rlimit_per_item": (self._backend.qfnra_rlimit
                                      if isinstance(self._backend, HybridGeometry3D) else None),
            "qfnra_solver_version": (__import__("z3").get_version_string()
                                     if isinstance(self._backend, HybridGeometry3D) else None),
            "ball_csg_scope": ("top_level_union_only"
                               if isinstance(self._backend, HybridGeometry3D) else None),
        }

    def is_empty(self) -> bool | None:
        return self._backend.is_empty() if self._backend is not None else None

    def is_full(self) -> bool | None:
        return self._backend.is_full() if self._backend is not None else None

    def classify_points(self, points: object) -> QueryResult:
        try:
            coordinates = _points(points, self.dimension)
        except ValueError:
            return _result([QueryStatus.INVALID_INPUT], [Membership.UNKNOWN], ["shape_or_dtype"])
        status: list[int] = []
        values: list[int] = []
        reasons: list[str] = []
        event_bound = self._estimated_events("point")
        work_units = []
        for p in coordinates:
            if not np.all(np.isfinite(p)):
                s, value, reason = QueryStatus.INVALID_INPUT, Membership.UNKNOWN, "nonfinite"
            elif np.any(p < 0) or np.any(p > 1):
                s, value, reason = QueryStatus.OUTSIDE_DOMAIN, Membership.UNKNOWN, "outside_domain"
            elif self._backend is None:
                s, value, reason = QueryStatus.UNSUPPORTED, Membership.UNKNOWN, self._unsupported_reason
            elif event_bound > self.max_events:
                s, value, reason = QueryStatus.UNRESOLVED, Membership.UNKNOWN, "arrangement_event_budget"
            else:
                try:
                    value = (self._backend.classify(*p) if self.dimension == 2
                             else self._backend.classify(p.tolist()))
                except RuntimeError as exc:
                    s, value, reason = QueryStatus.UNRESOLVED, Membership.UNKNOWN, str(exc)
                else:
                    s = QueryStatus.ON_BOUNDARY if value == Membership.BOUNDARY else QueryStatus.OK
                    reason = ""
            status.append(s)
            values.append(value)
            reasons.append(reason)
            work_units.append(event_bound if s in (QueryStatus.OK, QueryStatus.ON_BOUNDARY) else 0)
        return _result(status, values, reasons, work_units=work_units)

    def segment_intersect(self, a: object, b: object) -> QueryResult:
        try:
            starts, ends = _points(a, self.dimension), _points(b, self.dimension)
            if starts.shape != ends.shape:
                raise ValueError("segment endpoints shape mismatch")
        except ValueError:
            return _result([QueryStatus.INVALID_INPUT], [SegmentDecision.UNKNOWN], ["shape_or_dtype"])
        status, values, reasons, work_units = [], [], [], []
        event_bound = self._estimated_events("segment")
        for p, q in zip(starts, ends, strict=True):
            if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
                s, value, reason = QueryStatus.INVALID_INPUT, SegmentDecision.UNKNOWN, "nonfinite"
            elif np.any(p < 0) or np.any(p > 1) or np.any(q < 0) or np.any(q > 1):
                s, value, reason = QueryStatus.OUTSIDE_DOMAIN, SegmentDecision.UNKNOWN, "outside_domain"
            elif self._backend is None:
                s, value, reason = QueryStatus.UNSUPPORTED, SegmentDecision.UNKNOWN, self._unsupported_reason
            elif event_bound > self.max_events:
                s, value, reason = QueryStatus.UNRESOLVED, SegmentDecision.UNKNOWN, "arrangement_event_budget"
            else:
                try:
                    hit = (self._backend.segment_hit(*p, *q) if self.dimension == 2
                           else self._backend.segment_hit(p.tolist(), q.tolist()))
                except RuntimeError as exc:
                    s, value, reason = QueryStatus.UNRESOLVED, SegmentDecision.UNKNOWN, str(exc)
                else:
                    s = QueryStatus.OK
                    value = SegmentDecision.HIT if hit else SegmentDecision.MISS
                    reason = ""
            status.append(s)
            values.append(value)
            reasons.append(reason)
            work_units.append(event_bound if s == QueryStatus.OK else 0)
        return _result(status, values, reasons, work_units=work_units)

    def classify_box(self, bounds: object) -> QueryResult:
        try:
            boxes = np.asarray(bounds, dtype=np.float64)
            if boxes.shape == (2, self.dimension):
                boxes = boxes[None, ...]
            if boxes.ndim != 3 or boxes.shape[1:] != (2, self.dimension):
                raise ValueError("expected [m,2,d] bounds")
        except (TypeError, ValueError, OverflowError):
            return _result([QueryStatus.INVALID_INPUT], [BoxClass.UNKNOWN], ["shape_or_dtype"])
        status, values, reasons, work_units = [], [], [], []
        event_bound = self._estimated_events("box")
        for lo, hi in boxes:
            if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)) or np.any(lo > hi):
                s, value, reason = QueryStatus.INVALID_INPUT, BoxClass.UNKNOWN, "nonfinite_or_reversed"
            elif np.any(lo < 0) or np.any(hi > 1):
                s, value, reason = QueryStatus.OUTSIDE_DOMAIN, BoxClass.UNKNOWN, "outside_domain"
            elif self._backend is None:
                s, value, reason = QueryStatus.UNSUPPORTED, BoxClass.UNKNOWN, self._unsupported_reason
            elif event_bound > self.max_events:
                s, value, reason = QueryStatus.UNRESOLVED, BoxClass.UNKNOWN, "arrangement_event_budget"
            else:
                try:
                    value = (self._backend.classify_box(lo[0], lo[1], hi[0], hi[1])
                             if self.dimension == 2 else
                             self._backend.classify_box(lo.tolist(), hi.tolist()))
                except RuntimeError as exc:
                    s, value, reason = QueryStatus.UNRESOLVED, BoxClass.UNKNOWN, str(exc)
                else:
                    s, reason = QueryStatus.OK, ""
            status.append(s)
            values.append(value)
            reasons.append(reason)
            work_units.append(event_bound if s == QueryStatus.OK else 0)
        return _result(status, values, reasons, work_units=work_units)

    def clearance_decide(self, points: object, radius: object) -> QueryResult:
        try:
            coordinates = _points(points, self.dimension)
            radii = np.asarray(radius, dtype=np.float64)
            if radii.ndim == 0:
                radii = np.full(len(coordinates), radii.item(), dtype=np.float64)
            if radii.shape != (len(coordinates),):
                raise ValueError("radius must be scalar or [m]")
        except (TypeError, ValueError, OverflowError):
            return _result([QueryStatus.INVALID_INPUT], [Clearance.UNRESOLVED], ["shape_or_dtype"])
        status, values, reasons, evidence, work_units = [], [], [], [], []
        event_bound = self._estimated_events("clearance")
        for p, r in zip(coordinates, radii, strict=True):
            witness = None
            if not np.all(np.isfinite(p)) or not np.isfinite(r) or r < 0:
                s, value, reason = QueryStatus.INVALID_INPUT, Clearance.UNRESOLVED, "nonfinite_or_negative"
            elif np.any(p < 0) or np.any(p > 1):
                s, value, reason = QueryStatus.OUTSIDE_DOMAIN, Clearance.UNRESOLVED, "outside_domain"
            elif self._backend is None:
                s, value, reason = QueryStatus.UNSUPPORTED, Clearance.UNRESOLVED, self._unsupported_reason
            elif event_bound > self.max_events:
                s, value, reason = QueryStatus.UNRESOLVED, Clearance.UNRESOLVED, "arrangement_event_budget"
            else:
                try:
                    value, witness = (self._backend.clearance(*p, float(r))
                                      if self.dimension == 2 else
                                      self._backend.clearance(p.tolist(), float(r)))
                except RuntimeError as exc:
                    s, value, reason = QueryStatus.UNRESOLVED, Clearance.UNRESOLVED, str(exc)
                else:
                    s, reason = QueryStatus.OK, ""
            status.append(s)
            values.append(value)
            reasons.append(reason)
            evidence.append(witness)
            work_units.append(event_bound if s == QueryStatus.OK else 0)
        return _result(status, values, reasons, evidence, work_units=work_units)

    def verify_clearance_witness(self, point: object, radius: float, witness: Any) -> bool:
        if self._backend is None:
            return False
        p = _points(point, self.dimension)
        if len(p) != 1:
            raise ValueError("one witness verification point required")
        if self.dimension == 2:
            return self._backend.verify_clearance_witness(*p[0], float(radius), witness)
        return self._backend.verify_clearance_witness(p[0].tolist(), float(radius), witness)

    def distance_query(self, points: object, *, absolute_tolerance: float = 2**-40,
                       max_steps: int = 80) -> DistanceResult:
        """Return a certified enclosing interval for distance to the final solid.

        A finite lower endpoint is backed by a GT decision and a finite upper
        endpoint by LE. No rounded point estimate is presented as exact truth.
        """
        if not math.isfinite(absolute_tolerance) or absolute_tolerance <= 0:
            raise ValueError("absolute_tolerance must be positive and finite")
        if max_steps < 0 or max_steps > 1024:
            raise ValueError("max_steps must be in [0,1024]")
        try:
            coordinates = _points(points, self.dimension)
        except ValueError:
            coordinates = np.empty((1, self.dimension), dtype=np.float64)
            coordinates[:] = np.nan
        statuses: list[int] = []
        lower: list[float] = []
        upper: list[float] = []
        empties: list[bool] = []
        reasons: list[str] = []
        work: list[dict[str, int]] = []
        event_bound = self._estimated_events("clearance")
        for p in coordinates:
            lo, hi, empty = 0.0, 0.0, False
            units = 0
            if not np.all(np.isfinite(p)):
                status, reason = QueryStatus.INVALID_INPUT, "shape_or_nonfinite"
            elif np.any(p < 0) or np.any(p > 1):
                status, reason = QueryStatus.OUTSIDE_DOMAIN, "outside_domain"
            elif self._backend is None:
                status, reason = QueryStatus.UNSUPPORTED, self._unsupported_reason
            elif self._backend.is_empty():
                status, reason, hi, empty = QueryStatus.OK, "", math.inf, True
            elif (self._backend.classify(*p) if self.dimension == 2 else
                  self._backend.classify(p.tolist())) != Membership.FREE:
                status, reason = QueryStatus.OK, ""
            elif event_bound > self.max_events:
                status, reason, hi = QueryStatus.UNRESOLVED, "arrangement_event_budget", 2.0
            else:
                hi = 2.0  # Both Ω diameters are strictly below 2.
                def decide(r):
                    try:
                        return (self._backend.clearance(*p, r) if self.dimension == 2
                                else self._backend.clearance(p.tolist(), r))
                    except RuntimeError:
                        return Clearance.UNRESOLVED, None

                def verify(r, witness):
                    try:
                        return (self._backend.verify_clearance_witness(*p, r, witness)
                                if self.dimension == 2 else
                                self._backend.verify_clearance_witness(p.tolist(), r, witness))
                    except RuntimeError:
                        return False

                decision, witness = decide(hi)
                units += event_bound
                if decision != Clearance.LE or witness is None or not verify(hi, witness):
                    status, reason = QueryStatus.UNRESOLVED, "distance_upper_unproved"
                else:
                    status, reason = QueryStatus.OK, ""
                    for _ in range(max_steps):
                        if hi - lo <= absolute_tolerance:
                            break
                        mid = lo + (hi - lo) / 2
                        if mid <= lo or mid >= hi:
                            break
                        if units + event_bound > self.max_events:
                            status, reason = QueryStatus.UNRESOLVED, "arrangement_event_budget"
                            break
                        decision, witness = decide(mid)
                        units += event_bound
                        if decision == Clearance.GT:
                            lo = mid
                        elif decision == Clearance.LE and witness is not None and verify(mid, witness):
                            hi = mid
                        else:
                            status, reason = QueryStatus.UNRESOLVED, "clearance_unproved"
                            break
                    if status == QueryStatus.OK and hi - lo > absolute_tolerance:
                        status, reason = QueryStatus.UNRESOLVED, "distance_step_budget"
            statuses.append(status)
            lower.append(lo)
            upper.append(hi)
            empties.append(empty)
            reasons.append(reason)
            work.append({"precision_bits": 0, "subdivisions": 0,
                         "branches": 0, "backend_work_units": units})
        return DistanceResult(
            np.asarray(statuses, dtype=np.uint8),
            np.asarray(lower, dtype=np.float64),
            np.asarray(upper, dtype=np.float64),
            np.asarray(empties, dtype=np.bool_),
            np.asarray([s == QueryStatus.OK for s in statuses], dtype=np.bool_),
            tuple(reasons), tuple(work),
        )
