"""Conservative top-level-union facade for algebraic and transcendental leaves.

This is an integration prototype, not a READY backend: complete Ω-face proofs,
codec capability registration and transcendental distance bounds remain to be
implemented. It never infers a certified result from an ambiguous leaf.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import product
from typing import Any

import numpy as np

from .contracts import CSGNode, CSGOpcode, CodecID, Geometry, Leaf, RootKind
from .geometry import (BoxClass, Clearance, GeometryView, Membership,
                       QueryResult, QueryStatus, SegmentDecision,
                       UnsupportedGeometryError, _points, _result)
from .storage import CanonicalBundle
from .transcendental import (PRECISION_LADDER, AffineRadialPredicate,
                            CosineFieldPredicate, ProofTrace, _q,
                            capture_work, count_branch, count_native_work,
                            count_subdivision)


_TRANSCENDENTAL = {CodecID.RADIAL_FOURIER_2D,
                   CodecID.RADIAL_FOURIER_SPHERE, CodecID.COSINE_FIELD}


@dataclass(frozen=True, slots=True)
class BranchWitness:
    branch_index: int
    certificate: Any


class ExpandedGeometryView:
    """Query a declared top-level union, returning UNKNOWN on proof gaps."""

    def __init__(self, geometry_id: bytes, bundle: CanonicalBundle,
                 policy: dict[str, Any] | None = None):
        geometry = next((item for item in bundle.geometries
                         if item.content_hash == geometry_id), None)
        if geometry is None:
            raise KeyError("geometry ID not found")
        self.geometry_id = geometry_id
        self.dimension = geometry.dimension
        self.policy = policy or {}
        self._witness_seeds = tuple(
            np.asarray(item.field("center").as_array(), dtype=np.float64)
            for item in bundle.references
            if isinstance(item, Leaf) and item.codec != CodecID.COSINE_FIELD)
        precision_limit = int(self.policy.get("max_precision_bits", 1024))
        self.ladder = tuple(bits for bits in PRECISION_LADDER
                            if bits <= precision_limit)
        self.max_subdivisions = int(self.policy.get("max_subdivisions_per_item", 65536))
        if not self.ladder or self.max_subdivisions < 0:
            raise ValueError("invalid transcendental precision/subdivision policy")
        refs = {item.content_hash: item for item in bundle.references}
        algebraic = CanonicalBundle(self.dimension)
        for item in bundle.references:
            if isinstance(item, Leaf):
                algebraic.add_leaf(item)
            else:
                algebraic.add_node(item)

        def contains_transcendental(content_hash: bytes) -> bool:
            item = refs[content_hash]
            if isinstance(item, Leaf):
                return item.codec in _TRANSCENDENTAL
            return any(contains_transcendental(child) for child in item.children)

        branches = []
        algebraic_roots: list[bytes] = []

        def add_branch(content_hash: bytes) -> None:
            item = refs[content_hash]
            if isinstance(item, CSGNode) and item.opcode == CSGOpcode.UNION:
                for child in item.children:
                    add_branch(child)
            elif isinstance(item, Leaf) and item.codec in _TRANSCENDENTAL:
                branches.append(CosineFieldPredicate(item, self.ladder)
                                if item.codec == CodecID.COSINE_FIELD
                                else AffineRadialPredicate(item, self.ladder))
            elif contains_transcendental(content_hash):
                raise UnsupportedGeometryError(
                    "transcendental leaf under non-union CSG node")
            else:
                if not algebraic_roots:
                    branches.append(None)  # Replaced by one algebraic union view.
                algebraic_roots.append(content_hash)

        if geometry.root_kind == RootKind.REF:
            add_branch(geometry.root_hash)
        if algebraic_roots:
            if len(algebraic_roots) == 1:
                algebraic_root = algebraic_roots[0]
            else:
                node = CSGNode(self.dimension, CSGOpcode.UNION,
                               tuple(algebraic_roots))
                algebraic.add_node(node)
                algebraic_root = node.content_hash
            branch = Geometry(self.dimension, RootKind.REF, algebraic_root)
            algebraic.add_geometry(branch)
            view = GeometryView(branch.content_hash, algebraic, self.policy)
            if not view.capability()["supported"]:
                raise UnsupportedGeometryError(view.capability()["reason"])
            branches[branches.index(None)] = view
        self.branches = tuple(branches)

    def capability(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "backend": "CGAL+Arb-conservative-top-union-prototype",
            "supported": True,
            "ready_capable": False,
            "reason": "full Ω-face proofs and generator acceptance integration incomplete",
            "operations": ("classify_points", "classify_box", "segment_intersect",
                           "clearance_decide"),
            "topology_scope": "top-level union; transcendental leaves not nested",
            "work_counters_complete": True,
            "work_unit_definition": "Arb precision passes plus native backend units",
        }

    def _point(self, p: np.ndarray) -> Membership:
        if not self.branches:
            return Membership.FREE
        labels = []
        on_face = bool(np.any(p == 0) or np.any(p == 1))
        for branch in self.branches:
            count_branch()
            if isinstance(branch, GeometryView):
                result = branch.classify_points(p)
                count_native_work(result.work[0])
                label = (Membership(result.value[0]) if result.valid[0]
                         else Membership.UNKNOWN)
            else:
                label = branch.classify_point(p)
                if on_face and label == Membership.BOUNDARY:
                    label = Membership.UNKNOWN
            labels.append(label)
        if Membership.INTERIOR in labels:
            return Membership.INTERIOR
        if all(label == Membership.FREE for label in labels):
            return Membership.FREE
        if labels.count(Membership.BOUNDARY) == 1 and all(
                label in (Membership.BOUNDARY, Membership.FREE)
                for label in labels):
            return Membership.BOUNDARY
        return Membership.UNKNOWN

    def classify_points(self, points: object) -> QueryResult:
        try:
            coordinates = _points(points, self.dimension)
        except ValueError:
            return _result([QueryStatus.INVALID_INPUT], [Membership.UNKNOWN],
                           ["shape_or_dtype"])
        statuses, values, reasons, works = [], [], [], []
        for p in coordinates:
            work = ProofTrace().as_dict()
            if not np.all(np.isfinite(p)):
                status, value, reason = (QueryStatus.INVALID_INPUT,
                                         Membership.UNKNOWN, "nonfinite")
            elif np.any(p < 0) or np.any(p > 1):
                status, value, reason = (QueryStatus.OUTSIDE_DOMAIN,
                                         Membership.UNKNOWN, "outside_domain")
            else:
                with capture_work() as trace:
                    value = self._point(p)
                work = trace.as_dict()
                status = (QueryStatus.UNRESOLVED if value == Membership.UNKNOWN
                          else QueryStatus.ON_BOUNDARY if value == Membership.BOUNDARY
                          else QueryStatus.OK)
                reason = "composite_point_unproved" if status == QueryStatus.UNRESOLVED else ""
            statuses.append(status)
            values.append(value)
            reasons.append(reason)
            works.append(work)
        return replace(_result(statuses, values, reasons), work=tuple(works))

    def _segment_decision(self, p: np.ndarray, q: np.ndarray,
                          trace: ProofTrace) -> SegmentDecision:
        interior_endpoints = bool(np.all((p > 0) & (p < 1)) and
                                  np.all((q > 0) & (q < 1)))
        decisions = []
        for branch in self.branches:
            count_branch()
            if isinstance(branch, GeometryView):
                result = branch.segment_intersect(p, q)
                count_native_work(result.work[0])
                decision = (SegmentDecision(result.value[0])
                            if result.valid[0] else SegmentDecision.UNKNOWN)
            else:
                remaining = max(0, self.max_subdivisions - trace.subdivisions)
                decision = branch.segment_intersect(
                    p, q, max_subdivisions=remaining)
                if decision == SegmentDecision.HIT and not interior_endpoints:
                    decision = SegmentDecision.UNKNOWN
            decisions.append(decision)
            if decision == SegmentDecision.HIT:
                return SegmentDecision.HIT
        return (SegmentDecision.MISS if all(
            item == SegmentDecision.MISS for item in decisions)
            else SegmentDecision.UNKNOWN)

    def segment_intersect(self, a: object, b: object) -> QueryResult:
        try:
            starts, ends = _points(a, self.dimension), _points(b, self.dimension)
            if starts.shape != ends.shape:
                raise ValueError("segment endpoints shape mismatch")
        except ValueError:
            return _result([QueryStatus.INVALID_INPUT], [SegmentDecision.UNKNOWN],
                           ["shape_or_dtype"])
        statuses, values, reasons, works = [], [], [], []
        for p, q in zip(starts, ends, strict=True):
            work = ProofTrace().as_dict()
            if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
                status, value, reason = (QueryStatus.INVALID_INPUT,
                                         SegmentDecision.UNKNOWN, "nonfinite")
            elif (np.any(p < 0) or np.any(p > 1) or
                  np.any(q < 0) or np.any(q > 1)):
                status, value, reason = (QueryStatus.OUTSIDE_DOMAIN,
                                         SegmentDecision.UNKNOWN, "outside_domain")
            else:
                with capture_work() as trace:
                    value = self._segment_decision(p, q, trace)
                work = trace.as_dict()
                status = (QueryStatus.OK if value != SegmentDecision.UNKNOWN
                          else QueryStatus.UNRESOLVED)
                reason = "" if status == QueryStatus.OK else "composite_segment_unproved"
            statuses.append(status)
            values.append(value)
            reasons.append(reason)
            works.append(work)
        return replace(_result(statuses, values, reasons), work=tuple(works))

    def _box_decision(self, lo: np.ndarray, hi: np.ndarray,
                      trace: ProofTrace) -> BoxClass:
        classes = []
        domain_face_only = bool(np.any(lo == hi) and
                                (np.any(lo == 0) or np.any(hi == 1)))
        for branch in self.branches:
            count_branch()
            if isinstance(branch, GeometryView):
                result = branch.classify_box(np.stack((lo, hi)))
                count_native_work(result.work[0])
                kind = BoxClass(result.value[0]) if result.valid[0] else BoxClass.UNKNOWN
            else:
                kind = branch.classify_box(lo, hi)
                if domain_face_only and kind == BoxClass.SOLID:
                    kind = BoxClass.UNKNOWN
            classes.append(kind)
            if kind == BoxClass.SOLID:
                return BoxClass.SOLID
        if all(kind == BoxClass.FREE for kind in classes):
            return BoxClass.FREE
        return self._mixed_witness_search(
            lo, hi, max(0, self.max_subdivisions - trace.subdivisions))

    def classify_box(self, bounds: object) -> QueryResult:
        try:
            boxes = np.asarray(bounds, dtype=np.float64)
            if boxes.shape == (2, self.dimension):
                boxes = boxes[None, ...]
            if boxes.ndim != 3 or boxes.shape[1:] != (2, self.dimension):
                raise ValueError("expected [m,2,d] bounds")
        except (TypeError, ValueError, OverflowError):
            return _result([QueryStatus.INVALID_INPUT], [BoxClass.UNKNOWN],
                           ["shape_or_dtype"])
        statuses, values, reasons, works = [], [], [], []
        for lo, hi in boxes:
            work = ProofTrace().as_dict()
            if (not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)) or
                    np.any(lo > hi)):
                status, value, reason = (QueryStatus.INVALID_INPUT,
                                         BoxClass.UNKNOWN, "nonfinite_or_reversed")
            elif np.any(lo < 0) or np.any(hi > 1):
                status, value, reason = (QueryStatus.OUTSIDE_DOMAIN,
                                         BoxClass.UNKNOWN, "outside_domain")
            else:
                with capture_work() as trace:
                    value = self._box_decision(lo, hi, trace)
                work = trace.as_dict()
                status = QueryStatus.OK if value != BoxClass.UNKNOWN else QueryStatus.UNRESOLVED
                reason = "" if status == QueryStatus.OK else "composite_box_unproved"
            statuses.append(status)
            values.append(value)
            reasons.append(reason)
            works.append(work)
        return replace(_result(statuses, values, reasons), work=tuple(works))

    def _mixed_witness_search(self, lo: np.ndarray, hi: np.ndarray,
                              limit: int) -> BoxClass:
        """Find certified opposite point labels in the *world* box."""
        seen: set[Membership] = set()

        def observe(point: np.ndarray) -> bool:
            seen.add(self._point(point))
            return (Membership.FREE in seen and
                    bool(seen.intersection((Membership.INTERIOR,
                                            Membership.BOUNDARY))))

        for corner in product(*zip(lo, hi, strict=True)):
            if observe(np.asarray(corner, dtype=np.float64)):
                return BoxClass.MIXED
        if observe((lo + hi) / 2):
            return BoxClass.MIXED
        for seed in self._witness_seeds:
            if np.all(seed >= lo) and np.all(seed <= hi) and observe(seed):
                return BoxClass.MIXED
        queue = deque([(lo, hi)])
        splits = 0
        while queue and splits < limit:
            left, right = queue.popleft()
            axis = int(np.argmax(right - left))
            middle = left[axis] + (right[axis] - left[axis]) / 2
            if not left[axis] < middle < right[axis]:
                continue
            lower_hi = right.copy()
            upper_lo = left.copy()
            lower_hi[axis] = middle
            upper_lo[axis] = middle
            splits += 1
            count_subdivision()
            for child_lo, child_hi in ((left, lower_hi), (upper_lo, right)):
                if observe((child_lo + child_hi) / 2):
                    return BoxClass.MIXED
            queue.append((left, lower_hi))
            queue.append((upper_lo, right))
        return BoxClass.UNKNOWN

    def is_empty(self) -> bool | None:
        if not self.branches:
            return True
        statuses = []
        for branch in self.branches:
            if isinstance(branch, GeometryView):
                statuses.append(branch.is_empty())
            else:
                statuses.append(branch.classify_box([0]*self.dimension,
                                                     [1]*self.dimension) == BoxClass.FREE)
        return True if all(status is True for status in statuses) else None

    def is_full(self) -> bool | None:
        omega = np.asarray([[0]*self.dimension, [1]*self.dimension],
                           dtype=np.float64)
        for branch in self.branches:
            if isinstance(branch, GeometryView):
                if branch.is_full():
                    return True
            elif branch.classify_box(omega[0], omega[1]) == BoxClass.SOLID:
                return True
        return None

    def _clearance_decision(self, p: np.ndarray, r: float
                            ) -> tuple[Clearance, BranchWitness | None]:
        decisions = []
        for index, branch in enumerate(self.branches):
            count_branch()
            if isinstance(branch, GeometryView):
                result = branch.clearance_decide(p, r)
                count_native_work(result.work[0])
                decision = (Clearance(result.value[0]) if result.valid[0]
                            else Clearance.UNRESOLVED)
                certificate = result.evidence_ref[0]
            else:
                decision, certificate = self._primitive_clearance(branch, p, r)
            if decision == Clearance.LE and certificate is not None:
                return Clearance.LE, BranchWitness(index, certificate)
            decisions.append(decision)
        return (Clearance.GT if all(decision == Clearance.GT
                                    for decision in decisions)
                else Clearance.UNRESOLVED), None

    def clearance_decide(self, points: object, radius: object) -> QueryResult:
        try:
            coordinates = _points(points, self.dimension)
            radii = np.asarray(radius, dtype=np.float64)
            if radii.ndim == 0:
                radii = np.full(len(coordinates), radii.item(), dtype=np.float64)
            if radii.shape != (len(coordinates),):
                raise ValueError("radius must be scalar or [m]")
        except (TypeError, ValueError, OverflowError):
            return _result([QueryStatus.INVALID_INPUT], [Clearance.UNRESOLVED],
                           ["shape_or_dtype"])
        statuses, values, reasons, evidence, works = [], [], [], [], []
        for p, r in zip(coordinates, radii, strict=True):
            witness = None
            work = ProofTrace().as_dict()
            if not np.all(np.isfinite(p)) or not np.isfinite(r) or r < 0:
                status, value, reason = (QueryStatus.INVALID_INPUT,
                                         Clearance.UNRESOLVED, "nonfinite_or_negative")
            elif np.any(p < 0) or np.any(p > 1):
                status, value, reason = (QueryStatus.OUTSIDE_DOMAIN,
                                         Clearance.UNRESOLVED, "outside_domain")
            else:
                with capture_work() as trace:
                    value, witness = self._clearance_decision(p, float(r))
                work = trace.as_dict()
                status = (QueryStatus.OK if value != Clearance.UNRESOLVED
                          else QueryStatus.UNRESOLVED)
                reason = "" if status == QueryStatus.OK else "composite_clearance_unproved"
            statuses.append(status)
            values.append(value)
            reasons.append(reason)
            evidence.append(witness)
            works.append(work)
        return replace(_result(statuses, values, reasons, evidence),
                       work=tuple(works))

    def _primitive_clearance(self, branch: CosineFieldPredicate | AffineRadialPredicate,
                             p: np.ndarray, radius: float
                             ) -> tuple[Clearance, tuple[Fraction, ...] | None]:
        decision = branch.clearance_decide(p, radius)
        if decision != Clearance.LE:
            return decision, None
        world = tuple(_q(value) for value in p)
        if branch.classify_point(p) == Membership.INTERIOR:
            return Clearance.LE, world
        if (isinstance(branch, AffineRadialPredicate) and
                all(0 < value < 1 for value in world) and
                branch.classify_point(p) == Membership.BOUNDARY):
            return Clearance.LE, world
        if isinstance(branch, AffineRadialPredicate):
            center = branch.center
            if (all(0 <= value <= 1 for value in center) and
                    sum(((a-b)**2 for a, b in zip(world, center, strict=True)),
                        Fraction(0)) <= _q(radius)**2):
                return Clearance.LE, center
        return Clearance.UNRESOLVED, None

    def verify_clearance_witness(self, point: object, radius: float,
                                 witness: BranchWitness) -> bool:
        if not isinstance(witness, BranchWitness) or not 0 <= witness.branch_index < len(self.branches):
            return False
        branch = self.branches[witness.branch_index]
        if isinstance(branch, GeometryView):
            return branch.verify_clearance_witness(point, radius, witness.certificate)
        p = _points(point, self.dimension)
        if len(p) != 1 or not np.all(np.isfinite(p)) or not np.isfinite(radius) or radius < 0:
            return False
        q = witness.certificate
        if (not isinstance(q, tuple) or len(q) != self.dimension or
                not all(isinstance(value, Fraction) and 0 <= value <= 1 for value in q)):
            return False
        world = tuple(_q(value) for value in p[0])
        if sum(((a-b)**2 for a, b in zip(world, q, strict=True)),
               Fraction(0)) > _q(radius)**2:
            return False
        if isinstance(branch, CosineFieldPredicate):
            label = branch._classify_exact_point(q)
        else:
            label = branch.radial._classify_exact_point(branch._local(q))
        return (label == Membership.INTERIOR or
                (label == Membership.BOUNDARY and all(0 < value < 1 for value in q)))
