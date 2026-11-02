"""Private binary response service; never exposed through GeometryView."""

from __future__ import annotations

import numpy as np

from .contracts import Scene
from .geometry import (GeometryView, Membership, QueryResult, QueryStatus,
                       SegmentDecision, _points, _result)


class PrivateOracle:
    def __init__(self, scene: Scene, geometry_view: GeometryView):
        if scene.geometry_id != geometry_view.geometry_id:
            raise ValueError("scene/geometry identity mismatch")
        if scene.physics.codec_id != 0:
            raise ValueError("binary oracle requires binary_v1 physics")
        self._scene = scene
        self._geometry = geometry_view
        self._drivers = scene.driver_position.as_array()

    def _validate_query(self, points: object) -> tuple[np.ndarray | None, QueryResult]:
        try:
            coordinates = _points(points, self._scene.dimension)
        except ValueError:
            return None, _result([QueryStatus.INVALID_INPUT], [255], ["shape_or_dtype"])
        return coordinates, self._geometry.classify_points(coordinates)

    def query_binary_path_response(self, driver_id: int, points: object) -> QueryResult:
        coordinates, classification = self._validate_query(points)
        if coordinates is None:
            return classification
        if driver_id < 0 or driver_id >= len(self._drivers):
            return _result([QueryStatus.INVALID_INPUT] * len(coordinates),
                           [255] * len(coordinates), ["driver_index"] * len(coordinates))
        starts = np.broadcast_to(self._drivers[driver_id], coordinates.shape)
        paths = self._geometry.segment_intersect(starts, coordinates)
        status, values, reasons = [], [], []
        for i in range(len(coordinates)):
            if not classification.valid[i] or classification.value[i] != Membership.FREE:
                s = (QueryStatus.IN_STRUCTURE if classification.value[i] == Membership.INTERIOR
                     else QueryStatus.ON_BOUNDARY if classification.value[i] == Membership.BOUNDARY
                     else int(classification.status[i]))
                status.append(s)
                values.append(255)
                reasons.append("query_not_free" if s in (3, 4) else classification.reason_code[i])
            elif paths.status[i] != QueryStatus.OK:
                status.append(int(paths.status[i]))
                values.append(255)
                reasons.append(paths.reason_code[i])
            else:
                status.append(QueryStatus.OK)
                values.append(int(paths.value[i]))
                reasons.append("")
        return _result(status, values, reasons,
                       valid_override=[s == QueryStatus.OK for s in status])

    def query_response(self, points: object) -> QueryResult:
        coordinates, classification = self._validate_query(points)
        if coordinates is None:
            return classification
        status, values, reasons = [], [], []
        for i, point in enumerate(coordinates):
            if classification.value[i] == Membership.INTERIOR:
                status.append(QueryStatus.IN_STRUCTURE)
                values.append(255)
                reasons.append("query_in_structure")
                continue
            if classification.value[i] == Membership.BOUNDARY:
                status.append(QueryStatus.ON_BOUNDARY)
                values.append(255)
                reasons.append("query_on_boundary")
                continue
            if classification.status[i] != QueryStatus.OK:
                status.append(int(classification.status[i]))
                values.append(255)
                reasons.append(classification.reason_code[i])
                continue
            if not self._drivers.size:
                status.append(QueryStatus.OK)
                values.append(0)
                reasons.append("")
                continue
            unknown_status = None
            any_miss = False
            for driver in self._drivers:
                path = self._geometry.segment_intersect(driver, point)
                if path.status[0] == QueryStatus.OK and path.value[0] == SegmentDecision.MISS:
                    any_miss = True
                    break
                if path.status[0] != QueryStatus.OK and unknown_status is None:
                    unknown_status = int(path.status[0])
            if any_miss:
                status.append(QueryStatus.OK)
                values.append(1)
                reasons.append("")
            elif unknown_status is not None:
                status.append(unknown_status)
                values.append(255)
                reasons.append("driver_path_unproved")
            else:
                status.append(QueryStatus.OK)
                values.append(0)
                reasons.append("")
        return _result(status, values, reasons,
                       valid_override=[s == QueryStatus.OK for s in status])
