"""Local dataset generation, verification, and geometry-query commands."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from .geometry import GeometryView, backend_build_id
from .storage import read_bundle

try:
    from . import _exact2d
except ImportError:
    _exact2d = None
try:
    from . import _exact3d
except ImportError:
    _exact3d = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rind", description="RIND dataset tools")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="report local backend availability")
    for command in ("validate-storage", "verify-hashes"):
        sub = commands.add_parser(command, help="reload and verify a scratch SoA bundle")
        sub.add_argument("bundle", type=Path)
        sub.add_argument("--dimension", type=int, choices=(2, 3), required=True)
    query = commands.add_parser("query", help="query a structure-only geometry")
    query_source = query.add_mutually_exclusive_group(required=True)
    query_source.add_argument("--bundle", type=Path, help="standalone SoA bundle")
    query_source.add_argument("--dataset", type=Path, help="committed dataset root")
    query.add_argument("--dimension", type=int, choices=(2, 3), required=True)
    query.add_argument("--geometry", required=True, help="64-character content hash")
    query.add_argument("--request", type=Path, required=True, help="JSON request file")
    generate = commands.add_parser("generate", help="build an atomic core snapshot")
    generate.add_argument("--dimension", type=int, choices=(2, 3), required=True)
    generate.add_argument("--slots", required=True, help="half-open start:stop, e.g. 0:4")
    generate.add_argument("--output", type=Path, required=True)
    decay = commands.add_parser("expand-decay", help="publish additive physics variants from READY binary truth")
    decay.add_argument("--source", type=Path, required=True, help="deep-auditable binary snapshot root")
    decay.add_argument("--output", type=Path, required=True)
    resume = commands.add_parser("resume", help="append adjacent slots to a committed dataset")
    resume.add_argument("--dataset", type=Path, required=True)
    resume.add_argument("--slots", required=True, help="half-open start:stop; start must equal current stop")
    validate = commands.add_parser("validate-core", help="verify CURRENT snapshot and core tables")
    validate.add_argument("--dataset", type=Path, required=True)
    validate.add_argument("--deep", action="store_true", help="recompute scientific acceptance gates")
    validate.add_argument("--replay", action="store_true", help="also replay candidate RNG streams")
    diagnose = commands.add_parser("diagnose", help="run a derived center-grid diagnostic")
    diagnose.add_argument("--dataset", type=Path, required=True)
    diagnose.add_argument("--adapter", choices=("probe-grid-center-v1",), required=True)
    diagnose.add_argument("--n", type=int, help="grid size; default 512 in 2D, 128 in 3D")
    diagnose.add_argument("--slots", help="comma-separated READY slot indices; default all")
    diagnose.add_argument("--output", type=Path)
    benchmark = commands.add_parser("benchmark", help="measure storage and query performance")
    benchmark.add_argument("--dataset", type=Path, required=True)
    benchmark.add_argument("--slot", type=int, default=0)
    benchmark.add_argument("--output", type=Path)
    report = commands.add_parser("report", help="write a human-readable snapshot report")
    report.add_argument("--dataset", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--deep", action="store_true")
    report.add_argument("--replay", action="store_true")
    inspect = commands.add_parser("inspect", help="launch the local snapshot inspector")
    inspect.add_argument("--dataset", type=Path, required=True)
    inspect.add_argument("--port", type=int, default=8501)
    staging = commands.add_parser("inspect-staging", help="inspect build journals and abandoned stages")
    staging.add_argument("--dataset", type=Path, required=True)
    return parser


def _json_result(result, operation: str, view: GeometryView, request: dict) -> dict:
    evidence = []
    for index, witness in enumerate(result.evidence_ref):
        if witness is None:
            evidence.append(None)
            continue
        point = np.asarray(request["points"], dtype=np.float64)
        if point.ndim == 1:
            point = point[None, :]
        radius = np.asarray(request["radius"], dtype=np.float64)
        r = float(radius if radius.ndim == 0 else radius[index])
        if view.dimension == 2:
            display_point = [witness.x_approx, witness.y_approx]
            durable = False
            certificate = None
        else:
            display_point = list(witness.approximate)
            durable = True
            certificate = list(witness.homogeneous)
        evidence.append({
            "approximate_coordinates_for_display": display_point,
            "verified_by_exact_backend": view.verify_clearance_witness(point[index], r, witness),
            "durable_certificate": durable,
            "certificate_homogeneous": certificate,
        })
    return {
        "operation": operation,
        "geometry_id": view.geometry_id.hex(),
        "status": result.status.tolist(),
        "value": result.value.tolist(),
        "valid": result.valid.tolist(),
        "reason_code": result.reason_code,
        "evidence": evidence,
        "capability": view.capability(),
    }


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "doctor":
            print(json.dumps({
                "native_exact_2d": _exact2d is not None,
                "native_exact_3d": _exact3d is not None,
                "cgal_version": (_exact2d or _exact3d).backend_version
                                if (_exact2d or _exact3d) is not None else None,
                "backend_build_2d": backend_build_id(2),
                "backend_build_3d": backend_build_id(3),
                "ready_2d_pilot_supported": _exact2d is not None,
                "ready_3d_pilot_supported": _exact3d is not None,
                "ready_3d_scope": "algebraic B/C/D pilot; balls in top-level union only"
                                  if _exact3d is not None else None,
            }, indent=2))
            return 0
        if arguments.command == "inspect-staging":
            from .journal import inspect_staging
            print(json.dumps(inspect_staging(arguments.dataset), indent=2))
            return 0
        if arguments.command == "inspect":
            if not 1 <= arguments.port <= 65535:
                raise ValueError("port must be in [1,65535]")
            from .publication import validate_snapshot
            dimension = validate_snapshot(arguments.dataset)["dimension"]
            inspector_name = "inspector.py" if dimension == 2 else "inspector3d.py"
            environment = os.environ.copy()
            environment.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
            environment.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
            return subprocess.call([
                sys.executable, "-m", "streamlit", "run",
                str(Path(__file__).with_name(inspector_name)),
                "--server.address", "127.0.0.1", "--server.port", str(arguments.port),
                "--", "--dataset", str(arguments.dataset.resolve()),
            ], env=environment)
        if arguments.command == "generate":
            if arguments.dimension == 2:
                from .publication import build_2d as build
            else:
                from .publication3d import build_3d as build
            left, right = arguments.slots.split(":", 1)
            print(json.dumps(build(int(left), int(right), arguments.output), indent=2))
            return 0
        if arguments.command == "expand-decay":
            from .publication_decay import build_decay
            print(json.dumps(build_decay(arguments.source, arguments.output), indent=2))
            return 0
        if arguments.command == "resume":
            from .publication import validate_snapshot
            if validate_snapshot(arguments.dataset)["dimension"] == 2:
                from .publication import build_2d as build
            else:
                from .publication3d import build_3d as build
            left, right = arguments.slots.split(":", 1)
            print(json.dumps(build(int(left), int(right), arguments.dataset, resume=True), indent=2))
            return 0
        if arguments.command == "validate-core":
            from .publication import audit_snapshot, validate_snapshot
            if arguments.replay and not arguments.deep:
                raise ValueError("--replay requires --deep")
            result = (audit_snapshot(arguments.dataset, replay=arguments.replay)
                      if arguments.deep else validate_snapshot(arguments.dataset))
            print(json.dumps(result, indent=2))
            return 0
        if arguments.command == "diagnose":
            from .diagnostics import probe_grid
            slots = (None if not arguments.slots else
                     [int(value) for value in arguments.slots.split(",")])
            result = probe_grid(arguments.dataset, n=arguments.n, slots=slots,
                                output=arguments.output)
            print(json.dumps(result, indent=2))
            return 0
        if arguments.command == "benchmark":
            from .benchmark import benchmark_snapshot
            print(json.dumps(benchmark_snapshot(arguments.dataset, slot=arguments.slot,
                                                output=arguments.output), indent=2))
            return 0
        if arguments.command == "report":
            from .reports import snapshot_report
            print(json.dumps(snapshot_report(arguments.dataset, arguments.output,
                                             deep=arguments.deep,
                                             replay=arguments.replay), indent=2))
            return 0
        if arguments.command == "query" and arguments.dataset is not None:
            from .publication import validate_snapshot
            snapshot_id = validate_snapshot(arguments.dataset)["snapshot_id"]
            bundle_path = (arguments.dataset / "snapshots" / snapshot_id /
                           f"d{arguments.dimension}" / "core.zarr")
        else:
            bundle_path = arguments.bundle
        bundle = read_bundle(bundle_path, arguments.dimension)
        if arguments.command in ("validate-storage", "verify-hashes"):
            print(json.dumps({
                "valid": True,
                "dimension": bundle.dimension,
                "references": len(bundle.references),
                "geometries": len(bundle.geometries),
                "scenes": len(bundle.scenes),
                "hashes_and_repacking_verified": True,
                "ready_publication_verified": False,
            }, indent=2))
            return 0
        if len(arguments.geometry) != 64:
            raise ValueError("geometry hash must be 64 hex characters")
        geometry_id = bytes.fromhex(arguments.geometry)
        view = GeometryView(geometry_id, bundle)
        request = json.loads(arguments.request.read_text())
        operation = request["operation"]
        if operation == "classify_points":
            result = view.classify_points(request["points"])
        elif operation == "segment_intersect":
            result = view.segment_intersect(request["a"], request["b"])
        elif operation == "classify_box":
            result = view.classify_box(request["bounds"])
        elif operation == "clearance_decide":
            result = view.clearance_decide(request["points"], request["radius"])
        else:
            raise ValueError(f"unsupported operation: {operation}")
        print(json.dumps(_json_result(result, operation, view, request), indent=2))
        return 0
    except (ValueError, KeyError, OSError) as exc:
        print(f"rind: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
