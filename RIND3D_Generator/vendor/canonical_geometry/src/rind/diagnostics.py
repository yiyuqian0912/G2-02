"""Rebuildable probe-grid diagnostics; never part of canonical truth or READY."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .geometry import GeometryView, Membership
from .publication import _canonical_json, validate_snapshot
from .storage import read_bundle


ADAPTER_ID = "probe-grid-center-v1"


def _batches(coordinates: np.ndarray, dimension: int):
    if dimension == 2:
        for first in range(0, len(coordinates), 16):
            xx, yy = np.meshgrid(coordinates, coordinates[first:first + 16])
            yield np.column_stack((xx.ravel(), yy.ravel()))
    else:
        for z in coordinates:
            xx, yy = np.meshgrid(coordinates, coordinates)
            yield np.column_stack((xx.ravel(), yy.ravel(),
                                    np.full(xx.size, z, dtype=np.float64)))


def probe_grid(root: str | Path, *, n: int | None = None,
               slots: list[int] | None = None,
               output: str | Path | None = None) -> dict:
    root = Path(root)
    quick = validate_snapshot(root)
    dimension = quick["dimension"]
    if n is None:
        n = 512 if dimension == 2 else 128
    if n < 2 or n > 1024:
        raise ValueError("probe grid N must be in [2,1024]")
    snapshot = root / "snapshots" / quick["snapshot_id"]
    bundle = read_bundle(snapshot / f"d{dimension}" / "core.zarr", dimension)
    ledger = pq.read_table(snapshot / "catalogs" / "candidate_ledger.parquet").to_pylist()
    ready = [row for row in ledger if row["candidate_state"] == "READY"]
    if slots is not None:
        requested = set(slots)
        ready = [row for row in ready if row["slot_index"] in requested]
        if {row["slot_index"] for row in ready} != requested:
            raise ValueError("some requested slots are not READY")
    coordinates = (np.arange(n, dtype=np.float64) + .5) / n
    records = []
    counts_by_geometry = {}
    for row in ready:
        cached = counts_by_geometry.get(row["geometry_id"])
        if cached is None:
            view = GeometryView(bytes.fromhex(row["geometry_id"]), bundle)
            counts = {"FREE": 0, "INTERIOR": 0, "BOUNDARY": 0, "UNRESOLVED": 0}
            scalar_checks = 0
            # An exact empty/full decision proves every center at once.
            uniform = "FREE" if view.is_empty() is True else (
                "INTERIOR" if view.is_full() is True else None)
            if uniform is not None:
                counts[uniform] = n ** dimension
                expected = Membership.FREE if uniform == "FREE" else Membership.INTERIOR
                for index in (0, n // 2, n - 1):
                    point = [coordinates[index]] * dimension
                    result = view.classify_points(point)
                    if not result.valid[0] or result.value[0] != expected:
                        raise AssertionError("uniform geometry disagrees with scalar point")
                    scalar_checks += 1
            else:
                for points in _batches(coordinates, dimension):
                    result = view.classify_points(points)
                    counts["FREE"] += int(np.count_nonzero(result.value == Membership.FREE))
                    counts["INTERIOR"] += int(np.count_nonzero(result.value == Membership.INTERIOR))
                    counts["BOUNDARY"] += int(np.count_nonzero(result.value == Membership.BOUNDARY))
                    counts["UNRESOLVED"] += int(np.count_nonzero(~result.valid))
                    for local in (0, len(points) // 2, len(points) - 1):
                        scalar = view.classify_points(points[local])
                        if (not np.array_equal(scalar.value, result.value[local:local + 1])
                                or not np.array_equal(scalar.status,
                                                      result.status[local:local + 1])):
                            raise AssertionError("grid query differs between batch and scalar")
                        scalar_checks += 1
            full_box = view.classify_box([[0] * dimension, [1] * dimension])
            if dimension == 2 and not full_box.valid[0]:
                raise ValueError("domain box diagnostic unresolved")
            if full_box.valid[0] and full_box.value[0] == 0 and counts["FREE"] != n ** dimension:
                raise AssertionError("FREE domain box disagrees with center grid")
            if full_box.valid[0] and full_box.value[0] == 1 and counts["FREE"] != 0:
                raise AssertionError("SOLID domain box disagrees with center grid")
            cached = (counts, scalar_checks, int(full_box.value[0]))
            counts_by_geometry[row["geometry_id"]] = cached
        counts, scalar_checks, box_class = cached
        records.append({
            "slot_index": row["slot_index"], "scene_id": row["scene_id"],
            "geometry_id": row["geometry_id"], "counts": counts,
            "domain_box_class": box_class,
            "scalar_batch_agreement_checks": scalar_checks,
        })
    result = {"adapter": ADAPTER_ID, "snapshot_id": quick["snapshot_id"],
              "dimension": dimension, "n": n,
              "grid_point_formula": "(index+0.5)/N",
              "canonical_truth_modified": False, "scenes": records}
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        destination.write_bytes(_canonical_json(result) + b"\n")
    return result
