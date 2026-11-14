"""Single-writer committed snapshots for the exact 3D binary core pilot."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from .contracts import QUERY_SEMANTICS, SOLID_SEMANTICS
from .generation import CandidateInvalid, CandidateUnresolved, MASTER_SEED, MAX_ATTEMPTS
from .generation3d import (PROFILE_ID_3D, _exact_det3, make_binary_scene_3d,
                           propose_geometry_3d, sample_drivers_3d,
                           validate_candidate_3d)
from .geometry import GeometryView, backend_build_id
from .identity import DTypeTag as T
from .identity import FRAMING_ID, HASH_FAMILY, Field, digest, hash_bytes_field
from .journal import BuildJournal
from .publication import (_DATASET_DIR, _POLICY_PATH, _SPEC_PATH,
                          CONTRACT_VERSION, _bytes, _canonical_json,
                          _audit_candidate_journal, _checksums, _copy_candidate_journal,
                          _fsync_tree, _merge_bundle, _publish_snapshot, _save_json,
                          _sha, _table, validate_snapshot)
from .storage import CanonicalBundle, read_bundle, write_bundle


_PROFILE3_PATH = _DATASET_DIR / "profiles" / f"{PROFILE_ID_3D}.json"
GENERATOR_BUILD_3D = "rind-generator3d-impl2-" + hashlib.sha256(
    Path(__file__).with_name("generation3d.py").read_bytes() +
    (_DATASET_DIR / "native" / "exact3d.cpp").read_bytes()).hexdigest()[:24]


def _profile3() -> tuple[dict, bytes]:
    profile = json.loads(_PROFILE3_PATH.read_text())
    if (profile["profile_id"] != PROFILE_ID_3D or
            profile["master_seed"] != MASTER_SEED or
            profile["enabled_run_dimensions"] != [3]):
        raise ValueError("3D profile and generator constants disagree")
    return profile, hashlib.sha256(_canonical_json(profile)).digest()


def _lineage_id3(profile_digest: bytes, slot: int, attempt: int) -> bytes:
    return digest("lineage", (
        Field.numeric("master_seed", T.UINT64, MASTER_SEED),
        Field.text("profile_id", PROFILE_ID_3D),
        hash_bytes_field("profile_digest", profile_digest),
        Field.numeric("dimension", T.UINT8, 3),
        Field.numeric("slot_index", T.UINT64, slot),
        Field.numeric("attempt_index", T.UINT32, attempt),
        Field.text("generator_build", GENERATOR_BUILD_3D),
    ))


def _candidate_id3(lineage_id: bytes) -> bytes:
    return digest("candidate", (
        hash_bytes_field("lineage_id", lineage_id),
        Field.numeric("scene_variant_index", T.UINT32, 0),
    ))


def build_3d(start: int, stop: int, output: str | Path, *, resume: bool = False) -> dict:
    """Generate half-open slots with write/reload acceptance and atomic commit."""
    if start < 0 or stop <= start or stop > 128:
        raise ValueError("slots must be a nonempty half-open subset of [0,128)")
    profile, profile_digest = _profile3()
    policy = json.loads(_POLICY_PATH.read_text())
    output = Path(output)
    parent_id = None
    publish_start = start
    bundle = CanonicalBundle(3)
    ledger: list[dict] = []
    validation_reports: list[dict] = []
    if (output / "CURRENT").exists():
        if not resume:
            raise FileExistsError("a committed snapshot exists; use resume")
        parent = validate_snapshot(output)
        if parent["dimension"] != 3:
            raise ValueError("cannot resume a non-3D snapshot")
        parent_id = parent["snapshot_id"]
        parent_path = output / "snapshots" / parent_id
        parent_manifest = json.loads((parent_path / "manifest.json").read_text())
        if (parent_manifest["slots"][1] != start or
                parent_manifest["profile_digest"] != profile_digest.hex() or
                parent_manifest["generator_build"] != GENERATOR_BUILD_3D or
                parent_manifest["backend_build"] != backend_build_id(3)):
            raise ValueError("resume requires adjacent slots and identical 3D build")
        publish_start = parent_manifest["slots"][0]
        bundle = read_bundle(parent_path / "d3" / "core.zarr", 3)
        ledger = pq.read_table(parent_path / "catalogs" /
                               "candidate_ledger.parquet").to_pylist()
        validation_reports = [json.loads(row["report_json"]) for row in
                              pq.read_table(parent_path / "catalogs" /
                                            "query_validation_report.parquet").to_pylist()]
    elif resume:
        raise FileNotFoundError("cannot resume without CURRENT")
    output.mkdir(parents=True, exist_ok=True)
    staging_root = output / "staging" / uuid.uuid4().hex
    staging_root.mkdir(parents=True)
    journal = BuildJournal(staging_root / "journal.jsonl", {
        "dimension": 3, "slots": [start, stop], "parent_snapshot_id": parent_id,
        "profile_digest": profile_digest.hex(), "generator_build": GENERATOR_BUILD_3D,
        "backend_build": backend_build_id(3),
    })
    unfinished = set(range(start, stop))
    begun = time.monotonic()

    def record_terminal(row: dict) -> None:
        journal.append("candidate_terminal", {
            "candidate_id": row["candidate_id"], "state": row["candidate_state"],
            "geometry_id": row["geometry_id"], "scene_id": row["scene_id"],
            "reason_code": row["reason_code"],
        })

    try:
        for attempt in range(MAX_ATTEMPTS):
            if not unfinished:
                break
            wave: dict[int, bytes] = {}
            for slot in sorted(unfinished):
                lineage_id = _lineage_id3(profile_digest, slot, attempt)
                candidate_id = _candidate_id3(lineage_id)
                row = {
                    "slot_id": f"d3:{slot}", "slot_index": slot,
                    "attempt_index": attempt, "scene_variant_index": 0,
                    "candidate_id": candidate_id.hex(),
                    "lineage_id": lineage_id.hex(),
                    "profile_id": PROFILE_ID_3D,
                    "profile_digest": profile_digest.hex(),
                    "dimension": 3, "candidate_state": "PENDING",
                    "reason_code": "", "reason_detail": "",
                    "geometry_id": None, "scene_id": None,
                    "precision_bits_used": 0, "subdivisions_used": 0,
                    "generator_build": GENERATOR_BUILD_3D,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                ledger.append(row)
                journal.append("candidate_pending", {
                    "candidate_id": row["candidate_id"],
                    "slot_index": slot, "attempt_index": attempt,
                })
                try:
                    proposed, geometry = propose_geometry_3d(slot, attempt)
                    _merge_bundle(bundle, proposed)
                    row["geometry_id"] = geometry.content_hash.hex()
                    wave[slot] = geometry.content_hash
                    journal.append("candidate_progress", {
                        "candidate_id": row["candidate_id"], "stage": "geometry_proposed",
                        "geometry_id": row["geometry_id"],
                    })
                except CandidateInvalid as exc:
                    row.update(candidate_state="INVALID", reason_code=exc.reason,
                               reason_detail="proved structural proposal failure")
                    record_terminal(row)
                    if attempt == MAX_ATTEMPTS - 1:
                        unfinished.remove(slot)
            if not wave:
                continue
            geometry_store = staging_root / f"geometry-wave-{attempt}.zarr"
            write_bundle(geometry_store, bundle)
            reloaded = read_bundle(geometry_store, 3)
            scene_wave: dict[int, tuple[bytes, tuple[int, ...]]] = {}
            for slot, geometry_id in sorted(wave.items()):
                row = next(item for item in reversed(ledger)
                           if item["slot_index"] == slot and item["attempt_index"] == attempt)
                try:
                    view = GeometryView(geometry_id, reloaded, policy)
                    drivers = sample_drivers_3d(view, slot, attempt)
                    geometry = next(item for item in reloaded.geometries
                                    if item.content_hash == geometry_id)
                    scene = make_binary_scene_3d(reloaded, geometry, drivers.positions)
                    scene_wave[slot] = (scene.content_hash,
                                        drivers.proposals_per_driver)
                    row["scene_id"] = scene.content_hash.hex()
                    journal.append("candidate_progress", {
                        "candidate_id": row["candidate_id"], "stage": "scene_proposed",
                        "scene_id": row["scene_id"],
                        "driver_proposals_per_driver": drivers.proposals_per_driver,
                    })
                except CandidateInvalid as exc:
                    row.update(candidate_state="INVALID", reason_code=exc.reason,
                               reason_detail="proved structural or driver failure")
                    record_terminal(row)
                    if attempt == MAX_ATTEMPTS - 1:
                        unfinished.remove(slot)
                except CandidateUnresolved as exc:
                    row.update(candidate_state="UNRESOLVED", reason_code=exc.reason,
                               reason_detail="deterministic proof/proposal budget exhausted")
                    record_terminal(row)
                    unfinished.remove(slot)
            scene_store = staging_root / f"scene-wave-{attempt}.zarr"
            write_bundle(scene_store, reloaded)
            bundle = read_bundle(scene_store, 3)
            for slot, (scene_id, proposal_counts) in sorted(scene_wave.items()):
                row = next(item for item in reversed(ledger)
                           if item["slot_index"] == slot and item["attempt_index"] == attempt)
                geometry_id = bytes.fromhex(row["geometry_id"])
                scene = next(item for item in bundle.scenes
                             if item.content_hash == scene_id)
                view = GeometryView(geometry_id, bundle, policy)
                try:
                    report = validate_candidate_3d(scene, view, slot, attempt)
                except CandidateUnresolved as exc:
                    row.update(candidate_state="UNRESOLVED", reason_code=exc.reason,
                               reason_detail="acceptance probe unresolved")
                else:
                    row.update(candidate_state="READY", reason_code="",
                               reason_detail="acceptance-core-v1 passed")
                    report["slot_index"] = slot
                    report["attempt_index"] = attempt
                    report["driver_proposals_per_driver"] = proposal_counts
                    validation_reports.append(report)
                record_terminal(row)
                unfinished.remove(slot)
        if unfinished or any(row["candidate_state"] == "PENDING" for row in ledger):
            raise AssertionError("3D candidate slots are not terminal")
        ready = [row for row in ledger if row["candidate_state"] == "READY"]
        snapshot_id = _sha(_canonical_json({
            "profile_digest": profile_digest.hex(), "slots": [publish_start, stop],
            "parent_snapshot_id": parent_id,
            "generator_build": GENERATOR_BUILD_3D,
            "backend_build": backend_build_id(3),
            "terminal": [(row["candidate_id"], row["candidate_state"],
                          row["geometry_id"], row["scene_id"]) for row in ledger],
        }))
        snapshot = staging_root / "snapshot"
        (snapshot / "d3").mkdir(parents=True)
        (snapshot / "catalogs").mkdir()
        # Preserve a proposed scene even if its later acceptance proof failed:
        # its canonical parameters and identity remain inspectable in the
        # published attempt history, without granting that scene READY status.
        published = CanonicalBundle(3)
        _merge_bundle(published, bundle)
        for scene in bundle.scenes:
            published.add_scene(scene)
        write_bundle(snapshot / "d3" / "core.zarr", published)
        final_bundle = read_bundle(snapshot / "d3" / "core.zarr", 3)
        if {scene.content_hash for scene in final_bundle.scenes} != {
                scene.content_hash for scene in published.scenes}:
            raise AssertionError("3D scene changed during snapshot assembly")
        _table(snapshot / "catalogs" / "candidate_ledger.parquet", ledger)
        _table(snapshot / "catalogs" / "geometry_catalog.parquet", [
            {"dimension": 3, "geometry_id": item.content_hash.hex(),
             "row": i, "store": "d3/core.zarr"}
            for i, item in enumerate(published.geometries)
        ])
        _table(snapshot / "catalogs" / "scene_catalog.parquet", [
            {"dimension": 3, "scene_id": item.content_hash.hex(),
             "geometry_id": item.geometry_id.hex(), "row": i,
             "store": "d3/core.zarr"}
            for i, item in enumerate(published.scenes)
        ])
        _table(snapshot / "catalogs" / "content_lineage_map.parquet", [
            {"lineage_id": row["lineage_id"], "candidate_id": row["candidate_id"],
             "geometry_id": row["geometry_id"], "scene_id": row["scene_id"],
             "slot_id": row["slot_id"]} for row in ledger
        ])
        _table(snapshot / "catalogs" / "slot_summary.parquet", [
            {"slot_id": f"d3:{slot}",
             "attempts_started": sum(row["slot_index"] == slot for row in ledger),
             "final_state": next(row["candidate_state"] for row in reversed(ledger)
                                 if row["slot_index"] == slot)}
            for slot in range(publish_start, stop)
        ])
        _table(snapshot / "catalogs" / "candidate_events.parquet", [
            {"candidate_id": row["candidate_id"], "event": row["candidate_state"],
             "reason_code": row["reason_code"]} for row in ledger
        ])
        _table(snapshot / "catalogs" / "query_validation_report.parquet", [
            {"candidate_id": next(row["candidate_id"] for row in ledger
                                  if row["slot_index"] == report["slot_index"]
                                  and row["attempt_index"] == report["attempt_index"]),
             "report_json": _canonical_json(report).decode()}
            for report in validation_reports
        ])
        report = {
            "slots_requested": stop - publish_start,
            "slots_added_this_snapshot": stop - start,
            "attempts_started": len(ledger),
            "ready_candidates": len(ready),
            "unique_ready_scenes": len({row["scene_id"] for row in ready}),
            "invalid_attempts": sum(row["candidate_state"] == "INVALID" for row in ledger),
            "unresolved_attempts": sum(row["candidate_state"] == "UNRESOLVED" for row in ledger),
            "wall_clock_seconds_diagnostic": time.monotonic() - begun,
        }
        _table(snapshot / "catalogs" / "build_report.parquet", [report])
        _copy_candidate_journal(journal, snapshot)
        (snapshot / "profiles").mkdir()
        shutil.copy2(_PROFILE3_PATH, snapshot / "profiles" / _PROFILE3_PATH.name)
        shutil.copy2(_POLICY_PATH, snapshot / "numeric_policy.json")
        registry = {
            "hash_family": HASH_FAMILY, "framing_id": FRAMING_ID,
            "solid_semantics_id": SOLID_SEMANTICS,
            "query_semantics_id": QUERY_SEMANTICS,
            "codecs": ["affine_box_v1", "affine_ball_v1", "polygon_rings_2d_v1",
                       "convex_halfspaces_v1", "oriented_mesh_3d_v1",
                       "radial_fourier_2d_v1", "radial_fourier_sphere_v1",
                       "cosine_field_v1"],
        }
        _save_json(snapshot / "registry.json", registry)
        capability_by_geometry = {
            report["geometry_id"]: report["backend"]
            for report in validation_reports
        }
        if set(capability_by_geometry) != {row["geometry_id"] for row in ready}:
            raise AssertionError("READY geometry capability declarations are incomplete")
        manifest = {
            "schema_version": "rind-v3-impl1", "snapshot_id": snapshot_id,
            "parent_snapshot_id": parent_id,
            "spec_sha256": _sha(_bytes(_SPEC_PATH)),
            "implementation_contract_version": CONTRACT_VERSION,
            "hash_family": HASH_FAMILY, "framing_id": FRAMING_ID,
            "solid_semantics_id": SOLID_SEMANTICS,
            "query_semantics_id": QUERY_SEMANTICS,
            "codec_registry_digest": _sha(_canonical_json(registry)),
            "numeric_policy_digest": _sha(_canonical_json(policy)),
            "profile_digest": profile_digest.hex(),
            "generator_build": GENERATOR_BUILD_3D,
            "backend_build": backend_build_id(3),
            "rng_engine": "NumPy-PCG64",
            "dependency_lock_digest": _sha(_bytes(_DATASET_DIR / "requirements.lock")),
            "supported_dimensions": [3],
            "declared_capabilities": {
                "dimension": 3,
                "ready_geometry_capabilities": capability_by_geometry,
                "backend_variants": sorted({capability["backend"]
                                            for capability in capability_by_geometry.values()}),
            },
            "store_locations": {"d3": "d3/core.zarr"},
            "catalog_locations": [str(path.relative_to(snapshot)) for path in
                                  sorted((snapshot / "catalogs").glob("*.parquet"))],
            "candidate_journal_location": "catalogs/candidate_journal.jsonl",
            "slots": [publish_start, stop], "report": report,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "commit_protocol": "local-posix-rename-current-v1",
        }
        _save_json(snapshot / "manifest.json", manifest)
        _save_json(snapshot / "checksums.json", _checksums(snapshot))
        _save_json(snapshot / "COMMIT.json", {
            "snapshot_id": snapshot_id,
            "checksums_sha256": _sha(_bytes(snapshot / "checksums.json")),
        })
        _fsync_tree(snapshot)
        journal.append("operation_ready", {"snapshot_id": snapshot_id})
        _publish_snapshot(snapshot, output, snapshot_id, expected_current=parent_id)
        published = validate_snapshot(output)
        journal.append("operation_committed", {"snapshot_id": snapshot_id})
        return published
    except BaseException as error:
        try:
            journal.append("operation_aborted", {
                "error_type": type(error).__name__, "detail": str(error)[:500],
            })
        except Exception:
            pass
        raise


def _replay_failed_3d(row: dict, policy: dict) -> None:
    """Reject a published failure that deterministic 3D generation cannot reproduce."""
    slot, attempt = row["slot_index"], row["attempt_index"]
    geometry_id = scene_id = None
    reason = ""
    try:
        proposal, geometry = propose_geometry_3d(slot, attempt)
    except CandidateInvalid as error:
        observed, reason = "INVALID", error.reason
    else:
        geometry_id = geometry.content_hash.hex()
        view = GeometryView(geometry.content_hash, proposal, policy)
        try:
            drivers = sample_drivers_3d(view, slot, attempt)
        except CandidateInvalid as error:
            observed, reason = "INVALID", error.reason
        except CandidateUnresolved as error:
            observed, reason = "UNRESOLVED", error.reason
        else:
            scene = make_binary_scene_3d(proposal, geometry, drivers.positions)
            scene_id = scene.content_hash.hex()
            try:
                validate_candidate_3d(scene, view, slot, attempt)
            except CandidateUnresolved as error:
                observed, reason = "UNRESOLVED", error.reason
            else:
                observed = "READY"
    if (row["candidate_state"] != observed or row["reason_code"] != reason or
            row["geometry_id"] != geometry_id or row["scene_id"] != scene_id):
        raise ValueError(f"failed candidate replay differs: d3:{slot} attempt {attempt}")


def audit_snapshot_3d(root: str | Path, *, replay: bool = False) -> dict:
    """Independently recheck the published 3D records and acceptance reports."""
    quick = validate_snapshot(root)
    if quick["dimension"] != 3:
        raise ValueError("3D audit requires a 3D snapshot")
    root = Path(root)
    snapshot = root / "snapshots" / quick["snapshot_id"]
    manifest = json.loads((snapshot / "manifest.json").read_text())
    registry = json.loads((snapshot / "registry.json").read_text())
    profile = json.loads((snapshot / "profiles" / _PROFILE3_PATH.name).read_text())
    policy = json.loads((snapshot / "numeric_policy.json").read_text())
    if manifest["generator_build"] != GENERATOR_BUILD_3D:
        raise ValueError("3D generator build differs from snapshot provenance")
    if manifest["backend_build"] != backend_build_id(3):
        raise ValueError("3D geometry backend differs from snapshot provenance")
    if manifest["spec_sha256"] != _sha(_bytes(_SPEC_PATH)):
        raise ValueError("snapshot was built against a different spec")
    if manifest["codec_registry_digest"] != _sha(_canonical_json(registry)):
        raise ValueError("registry digest mismatch")
    if manifest["profile_digest"] != _sha(_canonical_json(profile)):
        raise ValueError("profile digest mismatch")
    if manifest["numeric_policy_digest"] != _sha(_canonical_json(policy)):
        raise ValueError("numeric policy digest mismatch")
    if (profile["profile_id"] != PROFILE_ID_3D or
            profile["enabled_run_dimensions"] != [3] or
            policy["policy_id"] != "rind-numeric-3-impl1" or
            manifest["supported_dimensions"] != [3]):
        raise ValueError("3D profile/numeric/dimension capability mismatch")
    bundle = read_bundle(snapshot / "d3" / "core.zarr", 3)
    from .contracts import CodecID, Geometry, Leaf, RootKind
    for item in bundle.references:
        if not isinstance(item, Leaf):
            continue
        if item.codec not in (CodecID.AFFINE_BOX, CodecID.AFFINE_BALL,
                              CodecID.ORIENTED_MESH_3D):
            raise ValueError("3D pilot contains a codec outside its declared mixture")
        if _exact_det3(item.field("matrix").as_array()) == 0:
            raise ValueError("stored 3D affine transform is singular")
        if item.codec == CodecID.ORIENTED_MESH_3D:
            if (item.field("shell_role").as_array().tolist() != [0] or
                    item.field("shell_triangle_offsets").as_array().shape != (2,)):
                raise ValueError("3D pilot mesh has unexpected shell recipe")
            isolated = CanonicalBundle(3)
            isolated.add_leaf(item)
            geometry = Geometry(3, RootKind.REF, item.content_hash)
            isolated.add_geometry(geometry)
            if not GeometryView(geometry.content_hash, isolated, policy).capability()["supported"]:
                raise ValueError("stored 3D mesh has no certified backend")
    catalogs = snapshot / "catalogs"
    ledger = pq.read_table(catalogs / "candidate_ledger.parquet").to_pylist()
    journal_verified = _audit_candidate_journal(snapshot, ledger)
    events = pq.read_table(catalogs / "candidate_events.parquet").to_pylist()
    slot_summary = pq.read_table(catalogs / "slot_summary.parquet").to_pylist()
    reports = pq.read_table(catalogs / "query_validation_report.parquet").to_pylist()
    geometry_catalog = pq.read_table(catalogs / "geometry_catalog.parquet").to_pylist()
    scene_catalog = pq.read_table(catalogs / "scene_catalog.parquet").to_pylist()
    lineage_map = pq.read_table(catalogs / "content_lineage_map.parquet").to_pylist()
    build_report = pq.read_table(catalogs / "build_report.parquet").to_pylist()
    if len(events) != len(ledger) or len(lineage_map) != len(ledger):
        raise ValueError("3D catalogs do not cover every attempt")
    if len(geometry_catalog) != len(bundle.geometries) or len(scene_catalog) != len(bundle.scenes):
        raise ValueError("3D catalog row count differs from canonical store")
    if any(row["geometry_id"] != geometry.content_hash.hex() or row["row"] != i
           for i, (row, geometry) in enumerate(zip(geometry_catalog,
                                                    bundle.geometries, strict=True))):
        raise ValueError("3D geometry catalog locator mismatch")
    if any(row["scene_id"] != scene.content_hash.hex() or row["row"] != i
           for i, (row, scene) in enumerate(zip(scene_catalog,
                                                 bundle.scenes, strict=True))):
        raise ValueError("3D scene catalog locator mismatch")
    expected_slots = list(range(*manifest["slots"]))
    if [row["slot_id"] for row in slot_summary] != [f"d3:{slot}" for slot in expected_slots]:
        raise ValueError("3D slot summary has missing, extra or reordered slots")
    rows_by_slot: dict[int, list[dict]] = {}
    profile_digest = bytes.fromhex(manifest["profile_digest"])
    for row, event, lineage in zip(ledger, events, lineage_map, strict=True):
        slot, attempt = int(row["slot_index"]), int(row["attempt_index"])
        rows_by_slot.setdefault(slot, []).append(row)
        expected_lineage = _lineage_id3(profile_digest, slot, attempt).hex()
        expected_candidate = _candidate_id3(bytes.fromhex(expected_lineage)).hex()
        if (row["lineage_id"] != expected_lineage or
                row["candidate_id"] != expected_candidate or
                event["candidate_id"] != expected_candidate or
                event["event"] != row["candidate_state"] or
                lineage["candidate_id"] != expected_candidate or
                lineage["geometry_id"] != row["geometry_id"] or
                lineage["scene_id"] != row["scene_id"]):
            raise ValueError("3D lineage, candidate or terminal event mismatch")
    if sorted(rows_by_slot) != expected_slots:
        raise ValueError("3D ledger does not cover requested slots")
    for summary in slot_summary:
        slot = int(summary["slot_id"].split(":")[1])
        rows = rows_by_slot[slot]
        if ([row["attempt_index"] for row in rows] != list(range(len(rows))) or
                summary["attempts_started"] != len(rows) or
                summary["final_state"] != rows[-1]["candidate_state"] or
                len(rows) > MAX_ATTEMPTS):
            raise ValueError("3D slot attempt sequence or final state mismatch")
        if any(row["candidate_state"] == "READY" for row in rows[:-1]):
            raise ValueError("READY 3D slot was silently retried")
    ready = [row for row in ledger if row["candidate_state"] == "READY"]
    declarations = manifest["declared_capabilities"]
    if "ready_geometry_capabilities" in declarations:
        declared = declarations["ready_geometry_capabilities"]
        if (declarations["dimension"] != 3 or
                set(declared) != {row["geometry_id"] for row in ready} or
                declarations["backend_variants"] != sorted({
                    capability["backend"] for capability in declared.values()})):
            raise ValueError("3D declared capability coverage mismatch")
    if len(reports) != len(ready):
        raise ValueError("3D acceptance reports do not cover READY candidates")
    report_by_id = {row["candidate_id"]: json.loads(row["report_json"]) for row in reports}
    if set(report_by_id) != {row["candidate_id"] for row in ready}:
        raise ValueError("3D acceptance report candidate IDs mismatch")
    geometries = {item.content_hash.hex(): item for item in bundle.geometries}
    scenes = {item.content_hash.hex(): item for item in bundle.scenes}
    referenced_geometries = {row["geometry_id"] for row in ledger
                             if row["geometry_id"] is not None}
    if set(geometries) != referenced_geometries:
        raise ValueError("published 3D geometry table differs from candidate attempts")
    referenced_scenes = {row["scene_id"] for row in ledger
                         if row["scene_id"] is not None}
    if set(scenes) != referenced_scenes:
        raise ValueError("published 3D scene table differs from candidate attempts")
    for row in ready:
        geometry = geometries[row["geometry_id"]]
        scene = scenes[row["scene_id"]]
        if scene.geometry_id != geometry.content_hash:
            raise ValueError("3D scene points to different geometry")
        if scene.physics.codec_id != 0 or not 1 <= len(scene.driver_position.as_array()) <= 4:
            raise ValueError("3D pilot physics or driver count mismatch")
        view = GeometryView(geometry.content_hash, bundle, policy)
        if not view.capability()["supported"]:
            raise ValueError("READY 3D geometry lacks certified query scope")
        if "ready_geometry_capabilities" in declarations and (
                declared[row["geometry_id"]] !=
                json.loads(_canonical_json(view.capability()))):
            raise ValueError("READY 3D geometry capability declaration differs")
        recomputed = validate_candidate_3d(scene, view, row["slot_index"],
                                           row["attempt_index"])
        stored = report_by_id[row["candidate_id"]]
        for key, value in recomputed.items():
            if stored.get(key) != json.loads(_canonical_json(value)):
                raise ValueError(f"3D acceptance evidence differs: {row['slot_id']} {key}")
        if replay:
            proposed_bundle, proposed_geometry = propose_geometry_3d(
                row["slot_index"], row["attempt_index"])
            if proposed_geometry.content_hash != geometry.content_hash:
                raise ValueError("3D geometry stream replay differs")
            proposed_view = GeometryView(proposed_geometry.content_hash,
                                         proposed_bundle, policy)
            drivers = sample_drivers_3d(proposed_view, row["slot_index"],
                                        row["attempt_index"])
            proposed_scene = make_binary_scene_3d(proposed_bundle,
                                                    proposed_geometry,
                                                    drivers.positions)
            if proposed_scene.content_hash != scene.content_hash:
                raise ValueError("3D driver stream replay differs")
            if list(drivers.proposals_per_driver) != stored["driver_proposals_per_driver"]:
                raise ValueError("3D driver proposal counters differ")
    if replay:
        for row in ledger:
            if row["candidate_state"] != "READY":
                _replay_failed_3d(row, policy)
    if len(build_report) != 1 or build_report[0]["ready_candidates"] != len(ready):
        raise ValueError("3D build report count mismatch")
    return {**quick, "scientific_audit": "passed", "replay": replay,
            "accepted_probe_reports_verified": len(ready),
            "affine_nonsingularity_checked": True,
            "slot_and_terminal_invariants_checked": True,
            "candidate_journal_verified": journal_verified,
            "failed_attempts_replayed": sum(
                row["candidate_state"] != "READY" for row in ledger) if replay else 0}
