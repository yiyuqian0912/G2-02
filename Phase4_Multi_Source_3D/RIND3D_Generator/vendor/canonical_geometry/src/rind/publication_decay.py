"""Immutable write/reload/publish expansion of READY binary truth into decay variants."""

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

import numpy as np
import pyarrow.parquet as pq

from .contracts import QUERY_SEMANTICS, SOLID_SEMANTICS, make_physics, make_scene
from .decay import AdditiveOracle, LogPchipLaw
from .generation_decay import (PROFILE_ID_DECAY, make_decay_variant,
                               stream_decay)
from .geometry import GeometryView, Membership, QueryStatus, backend_build_id
from .identity import DTypeTag as T
from .identity import FRAMING_ID, HASH_FAMILY, Field, digest, hash_bytes_field
from .journal import BuildJournal
from .publication import (_DATASET_DIR, _POLICY_PATH, _SPEC_PATH,
                          CONTRACT_VERSION, _bytes, _canonical_json,
                          _audit_candidate_journal, _checksums, _copy_candidate_journal,
                          _fsync_tree, _merge_bundle, _publish_snapshot, _save_json,
                          _sha, _table, audit_snapshot, validate_snapshot)
from .storage import CanonicalBundle, read_bundle, write_bundle


_PROFILE_PATH = _DATASET_DIR / "profiles" / f"{PROFILE_ID_DECAY}.json"
DECAY_BUILD = "rind-decay-impl1-" + hashlib.sha256(
    Path(__file__).with_name("decay.py").read_bytes() +
    Path(__file__).with_name("generation_decay.py").read_bytes()).hexdigest()[:24]


def _lineage(profile_digest: bytes, source_candidate_id: str,
             source_scene_id: str, dimension: int, slot: int,
             attempt: int, variant: int) -> bytes:
    return digest("lineage", (
        Field.text("profile_id", PROFILE_ID_DECAY),
        hash_bytes_field("profile_digest", profile_digest),
        hash_bytes_field("source_candidate_id", bytes.fromhex(source_candidate_id)),
        hash_bytes_field("source_scene_id", bytes.fromhex(source_scene_id)),
        Field.numeric("dimension", T.UINT8, dimension),
        Field.numeric("slot_index", T.UINT64, slot),
        Field.numeric("attempt_index", T.UINT32, attempt),
        Field.numeric("scene_variant_index", T.UINT32, variant),
        Field.text("generator_build", DECAY_BUILD),
    ))


def _candidate(lineage_id: bytes, variant: int) -> bytes:
    return digest("candidate", (
        hash_bytes_field("lineage_id", lineage_id),
        Field.numeric("scene_variant_index", T.UINT32, variant),
    ))


def _acceptance(scene, view: GeometryView, slot: int, attempt: int,
                variant: int) -> dict:
    """Recheck every algebraic/interval response without changing the scene."""
    if scene.physics.codec_id != variant:
        raise ValueError("decay variant index and physics codec differ")
    if variant == 1:
        tau = next(field for field in scene.physics.fields if field.name == "tau")
        if float(tau.as_array()) != .25:
            raise ValueError("fixed exponential control has a nonstandard tau")
    else:
        LogPchipLaw.from_physics(scene.physics, scene.dimension)
    points = stream_decay(scene.dimension, slot, attempt, variant,
                          "query").random((8, scene.dimension))
    membership = view.classify_points(points)
    response = AdditiveOracle(scene, view).query_response(points)
    if not np.all(membership.valid):
        raise RuntimeError("DECAY_ACCEPTANCE_MEMBERSHIP_UNPROVED")
    for i in range(len(points)):
        if membership.value[i] == Membership.FREE:
            if response.status[i] != QueryStatus.OK or not response.valid[i]:
                raise RuntimeError("DECAY_ACCEPTANCE_RESPONSE_UNPROVED")
        elif response.valid[i] or response.value[i] != 0:
            raise AssertionError("structure point exposed additive response")
    reverse = AdditiveOracle(scene, view).query_response(points[::-1])
    if (not np.array_equal(response.status, reverse.status[::-1]) or
            not np.array_equal(response.value, reverse.value[::-1])):
        raise AssertionError("decay response depends on query batch order")
    return {
        "protocol": "acceptance-decay-v1", "geometry_id": scene.geometry_id.hex(),
        "scene_id": scene.content_hash.hex(), "backend": view.capability(),
        "query_points": points.tolist(),
        "membership": membership.value.tolist(),
        "response_status": response.status.tolist(),
        "response_valid": response.valid.tolist(),
        "response_value": response.value.tolist(),
        "precision_bits": response.precision_bits.tolist(),
    }


def build_decay(source_root: str | Path, output: str | Path) -> dict:
    """Expand a deeply audited binary snapshot; publish only validated variants."""
    source_root, output = Path(source_root), Path(output)
    if (output / "CURRENT").exists():
        raise FileExistsError("decay output already contains a committed snapshot")
    source_check = audit_snapshot(source_root, replay=False)
    source_id, dimension = source_check["snapshot_id"], source_check["dimension"]
    source_path = source_root / "snapshots" / source_id
    source_manifest = json.loads((source_path / "manifest.json").read_text())
    source_bundle = read_bundle(source_path / f"d{dimension}" / "core.zarr", dimension)
    source_rows = pq.read_table(source_path / "catalogs" /
                                "candidate_ledger.parquet").to_pylist()
    source_scenes = {scene.content_hash.hex(): scene for scene in source_bundle.scenes}
    profile = json.loads(_PROFILE_PATH.read_text())
    if profile["profile_id"] != PROFILE_ID_DECAY or profile["master_seed"] != 20260921:
        raise ValueError("decay profile and generator disagree")
    profile_digest = hashlib.sha256(_canonical_json(profile)).digest()
    policy = json.loads(_POLICY_PATH.read_text())
    if policy["max_precision_bits"] != 1024:
        raise ValueError("decay precision ladder and numeric policy disagree")
    staging = output / "staging" / uuid.uuid4().hex
    staging.mkdir(parents=True)
    journal = BuildJournal(staging / "journal.jsonl", {
        "dimension": dimension, "source_snapshot_id": source_id,
        "profile_digest": profile_digest.hex(), "generator_build": DECAY_BUILD,
        "backend_build": backend_build_id(dimension),
    })
    begun = time.monotonic()
    bundle = CanonicalBundle(dimension)
    _merge_bundle(bundle, source_bundle)
    rows = []
    for source in source_rows:
        if source["candidate_state"] != "READY":
            continue
        binary = source_scenes[source["scene_id"]]
        if binary.physics.codec_id != 0:
            raise ValueError("source snapshot contains a nonbinary READY scene")
        for variant in (1, 2):
            lineage_id = _lineage(profile_digest, source["candidate_id"],
                                  source["scene_id"], dimension,
                                  source["slot_index"], source["attempt_index"],
                                  variant)
            candidate_id = _candidate(lineage_id, variant)
            scene = make_decay_variant(binary, source["slot_index"],
                                       source["attempt_index"], variant)
            bundle.add_scene(scene)
            rows.append({
                "slot_id": source["slot_id"],
                "slot_index": source["slot_index"],
                "attempt_index": source["attempt_index"],
                "scene_variant_index": variant,
                "source_candidate_id": source["candidate_id"],
                "source_binary_scene_id": source["scene_id"],
                "candidate_id": candidate_id.hex(),
                "lineage_id": lineage_id.hex(),
                "profile_id": PROFILE_ID_DECAY,
                "profile_digest": profile_digest.hex(),
                "dimension": dimension,
                "candidate_state": "PENDING", "reason_code": "", "reason_detail": "",
                "geometry_id": scene.geometry_id.hex(),
                "scene_id": scene.content_hash.hex(),
                "precision_bits_used": 0, "subdivisions_used": 0,
                "generator_build": DECAY_BUILD,
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
            journal.append("candidate_pending", {
                "candidate_id": candidate_id.hex(), "slot_index": source["slot_index"],
                "attempt_index": source["attempt_index"], "variant": variant,
                "source_candidate_id": source["candidate_id"],
            })
            journal.append("candidate_progress", {
                "candidate_id": candidate_id.hex(), "stage": "scene_proposed",
                "geometry_id": scene.geometry_id.hex(), "scene_id": scene.content_hash.hex(),
            })
    proposal_store = staging / "decay-proposals.zarr"
    write_bundle(proposal_store, bundle)
    reloaded = read_bundle(proposal_store, dimension)
    by_id = {scene.content_hash.hex(): scene for scene in reloaded.scenes}
    reports = []
    for row in rows:
        scene = by_id[row["scene_id"]]
        view = GeometryView(scene.geometry_id, reloaded, policy)
        try:
            report = _acceptance(scene, view, row["slot_index"],
                                 row["attempt_index"], row["scene_variant_index"])
        except RuntimeError as exc:
            row.update(candidate_state="UNRESOLVED", reason_code=str(exc),
                       reason_detail="finite certified acceptance budget exhausted")
        else:
            row.update(candidate_state="READY", reason_detail="acceptance-decay-v1 passed",
                       precision_bits_used=max(report["precision_bits"], default=0))
            reports.append({"candidate_id": row["candidate_id"],
                            "report_json": _canonical_json(report).decode()})
        journal.append("candidate_terminal", {
            "candidate_id": row["candidate_id"], "state": row["candidate_state"],
            "geometry_id": row["geometry_id"], "scene_id": row["scene_id"],
            "reason_code": row["reason_code"],
        })
    if any(row["candidate_state"] == "PENDING" for row in rows):
        raise AssertionError("decay candidate did not reach terminal state")
    ready = [row for row in rows if row["candidate_state"] == "READY"]
    published = CanonicalBundle(dimension)
    _merge_bundle(published, reloaded)
    ready_ids = {row["scene_id"] for row in ready}
    for scene in reloaded.scenes:
        if scene.content_hash.hex() in ready_ids:
            published.add_scene(scene)
    snapshot_id = _sha(_canonical_json({
        "kind": "decay-variants-v1", "source_snapshot_id": source_id,
        "profile_digest": profile_digest.hex(), "backend_build": backend_build_id(dimension),
        "decay_build": DECAY_BUILD,
        "terminal": [(row["candidate_id"], row["candidate_state"], row["scene_id"])
                     for row in rows],
    }))
    snapshot = staging / "snapshot"
    (snapshot / f"d{dimension}").mkdir(parents=True)
    (snapshot / "catalogs").mkdir()
    write_bundle(snapshot / f"d{dimension}" / "core.zarr", published)
    final = read_bundle(snapshot / f"d{dimension}" / "core.zarr", dimension)
    if {scene.content_hash for scene in final.scenes} != {
            scene.content_hash for scene in published.scenes}:
        raise AssertionError("decay scene changed during published write/reload")
    _table(snapshot / "catalogs" / "candidate_ledger.parquet", rows)
    _table(snapshot / "catalogs" / "geometry_catalog.parquet", [
        {"dimension": dimension, "geometry_id": geometry.content_hash.hex(),
         "row": i, "store": f"d{dimension}/core.zarr"}
        for i, geometry in enumerate(final.geometries)])
    _table(snapshot / "catalogs" / "scene_catalog.parquet", [
        {"dimension": dimension, "scene_id": scene.content_hash.hex(),
         "geometry_id": scene.geometry_id.hex(), "row": i,
         "store": f"d{dimension}/core.zarr"}
        for i, scene in enumerate(final.scenes)])
    _table(snapshot / "catalogs" / "content_lineage_map.parquet", [
        {"lineage_id": row["lineage_id"], "candidate_id": row["candidate_id"],
         "geometry_id": row["geometry_id"], "scene_id": row["scene_id"],
         "source_binary_scene_id": row["source_binary_scene_id"],
         "slot_id": row["slot_id"]} for row in rows])
    _table(snapshot / "catalogs" / "candidate_events.parquet", [
        {"candidate_id": row["candidate_id"], "event": row["candidate_state"],
         "reason_code": row["reason_code"]} for row in rows])
    _table(snapshot / "catalogs" / "query_validation_report.parquet", reports)
    report = {
        "source_ready_binary_candidates": sum(row["candidate_state"] == "READY"
                                              for row in source_rows),
        "variant_attempts": len(rows), "ready_candidates": len(ready),
        "unique_ready_scenes": len(ready_ids),
        "unresolved_candidates": len(rows) - len(ready),
        "wall_clock_seconds_diagnostic": time.monotonic() - begun,
    }
    _table(snapshot / "catalogs" / "build_report.parquet", [report])
    _copy_candidate_journal(journal, snapshot)
    (snapshot / "profiles").mkdir()
    shutil.copy2(_PROFILE_PATH, snapshot / "profiles" / _PROFILE_PATH.name)
    shutil.copy2(_POLICY_PATH, snapshot / "numeric_policy.json")
    registry = json.loads((source_path / "registry.json").read_text())
    registry["physics_codecs"] = ["exponential_v1", "log_pchip_v1"]
    _save_json(snapshot / "registry.json", registry)
    manifest = {
        "schema_version": "rind-v3-impl1", "dataset_kind": "decay-variants-v1",
        "snapshot_id": snapshot_id,
        "source_snapshot_id": source_id,
        "source_profile_digest": source_manifest["profile_digest"],
        "spec_sha256": _sha(_bytes(_SPEC_PATH)),
        "implementation_contract_version": CONTRACT_VERSION,
        "hash_family": HASH_FAMILY, "framing_id": FRAMING_ID,
        "solid_semantics_id": SOLID_SEMANTICS,
        "query_semantics_id": QUERY_SEMANTICS,
        "codec_registry_digest": _sha(_canonical_json(registry)),
        "numeric_policy_digest": _sha(_canonical_json(policy)),
        "profile_digest": profile_digest.hex(),
        "generator_build": DECAY_BUILD,
        "backend_build": backend_build_id(dimension),
        "rng_engine": "NumPy-PCG64",
        "dependency_lock_digest": _sha(_bytes(_DATASET_DIR / "requirements.lock")),
        "supported_dimensions": [dimension],
        "declared_capabilities": {
            "physics_codecs": ["exponential_v1", "log_pchip_v1"],
            "term_rounding": "Arb-certified-binary64-v1",
            "accumulation": "exact-rounded-sum-v1",
        },
        "store_locations": {f"d{dimension}": f"d{dimension}/core.zarr"},
        "catalog_locations": [str(path.relative_to(snapshot)) for path in
                              sorted((snapshot / "catalogs").glob("*.parquet"))],
        "candidate_journal_location": "catalogs/candidate_journal.jsonl",
        "slots": source_manifest["slots"], "report": report,
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
    _publish_snapshot(snapshot, output, snapshot_id, expected_current=None)
    published = validate_snapshot(output)
    journal.append("operation_committed", {"snapshot_id": snapshot_id})
    return published


def audit_decay(root: str | Path, *, replay: bool = False) -> dict:
    quick = validate_snapshot(root)
    root = Path(root)
    snapshot = root / "snapshots" / quick["snapshot_id"]
    manifest = json.loads((snapshot / "manifest.json").read_text())
    if manifest.get("dataset_kind") != "decay-variants-v1":
        raise ValueError("not a decay-variant snapshot")
    dimension = quick["dimension"]
    registry = json.loads((snapshot / "registry.json").read_text())
    profile = json.loads((snapshot / "profiles" / _PROFILE_PATH.name).read_text())
    policy = json.loads((snapshot / "numeric_policy.json").read_text())
    if (manifest["spec_sha256"] != _sha(_bytes(_SPEC_PATH)) or
            manifest["generator_build"] != DECAY_BUILD or
            manifest["backend_build"] != backend_build_id(dimension) or
            manifest["codec_registry_digest"] != _sha(_canonical_json(registry)) or
            manifest["profile_digest"] != _sha(_canonical_json(profile)) or
            manifest["numeric_policy_digest"] != _sha(_canonical_json(policy))):
        raise ValueError("decay manifest provenance differs from current contract")
    bundle = read_bundle(snapshot / f"d{dimension}" / "core.zarr", dimension)
    catalogs = snapshot / "catalogs"
    rows = pq.read_table(catalogs / "candidate_ledger.parquet").to_pylist()
    journal_verified = _audit_candidate_journal(snapshot, rows)
    events = pq.read_table(catalogs / "candidate_events.parquet").to_pylist()
    lineages = pq.read_table(catalogs / "content_lineage_map.parquet").to_pylist()
    reports = pq.read_table(catalogs / "query_validation_report.parquet").to_pylist()
    geometry_catalog = pq.read_table(catalogs / "geometry_catalog.parquet").to_pylist()
    scene_catalog = pq.read_table(catalogs / "scene_catalog.parquet").to_pylist()
    build_report = pq.read_table(catalogs / "build_report.parquet").to_pylist()
    if (len(rows) != len(events) or len(rows) != len(lineages) or
            len(geometry_catalog) != len(bundle.geometries) or
            len(scene_catalog) != len(bundle.scenes)):
        raise ValueError("decay catalog coverage mismatch")
    if any(row["geometry_id"] != geometry.content_hash.hex() or row["row"] != i
           for i, (row, geometry) in enumerate(
               zip(geometry_catalog, bundle.geometries, strict=True))):
        raise ValueError("decay geometry catalog locator mismatch")
    if any(row["scene_id"] != scene.content_hash.hex() or
           row["geometry_id"] != scene.geometry_id.hex() or row["row"] != i
           for i, (row, scene) in enumerate(
               zip(scene_catalog, bundle.scenes, strict=True))):
        raise ValueError("decay scene catalog locator mismatch")
    scenes = {scene.content_hash.hex(): scene for scene in bundle.scenes}
    report_by_id = {row["candidate_id"]: json.loads(row["report_json"])
                    for row in reports}
    ready = [row for row in rows if row["candidate_state"] == "READY"]
    if (len(report_by_id) != len(reports) or
            set(scenes) != {row["scene_id"] for row in ready} or
            set(report_by_id) != {row["candidate_id"] for row in ready}):
        raise ValueError("decay published scenes or acceptance reports differ from READY")
    profile_digest = bytes.fromhex(manifest["profile_digest"])
    source_profile_digest = bytes.fromhex(manifest["source_profile_digest"])
    if dimension == 2:
        from .publication import _candidate_id as source_candidate
        from .publication import _lineage_id as source_lineage
    else:
        from .publication3d import _candidate_id3 as source_candidate
        from .publication3d import _lineage_id3 as source_lineage
    source_variants: dict[tuple[str, str, int, int], set[int]] = {}
    candidate_ids: set[str] = set()
    for row, event, lineage in zip(rows, events, lineages, strict=True):
        variant = row["scene_variant_index"]
        source_key = (row["source_candidate_id"], row["source_binary_scene_id"],
                      int(row["slot_index"]), int(row["attempt_index"]))
        expected_source = source_candidate(source_lineage(
            source_profile_digest, row["slot_index"], row["attempt_index"])).hex()
        if (variant not in (1, 2) or row["candidate_id"] in candidate_ids or
                variant in source_variants.setdefault(source_key, set()) or
                row["slot_id"] != f"d{dimension}:{row['slot_index']}" or
                row["source_candidate_id"] != expected_source):
            raise ValueError("decay variant ledger has duplicate or malformed candidates")
        candidate_ids.add(row["candidate_id"])
        source_variants[source_key].add(variant)
        expected_lineage = _lineage(profile_digest, row["source_candidate_id"],
                                    row["source_binary_scene_id"], dimension,
                                    row["slot_index"], row["attempt_index"],
                                    variant).hex()
        expected_candidate = _candidate(bytes.fromhex(expected_lineage), variant).hex()
        if (row["lineage_id"] != expected_lineage or
                row["candidate_id"] != expected_candidate or
                event["candidate_id"] != expected_candidate or
                event["event"] != row["candidate_state"] or
                lineage["candidate_id"] != expected_candidate or
                lineage["source_binary_scene_id"] != row["source_binary_scene_id"] or
                lineage["geometry_id"] != row["geometry_id"] or
                lineage["scene_id"] != row["scene_id"] or
                lineage["slot_id"] != row["slot_id"]):
            raise ValueError("decay lineage or candidate event mismatch")
        if row["candidate_state"] != "READY":
            continue
        scene = scenes[row["scene_id"]]
        if scene.geometry_id.hex() != row["geometry_id"] or scene.physics.codec_id != variant:
            raise ValueError("decay scene/geometry/variant mismatch")
        binary = make_scene(scene.geometry_id, dimension, make_physics(0),
                            scene.driver_position.as_array())
        if binary.content_hash.hex() != row["source_binary_scene_id"]:
            raise ValueError("decay variant did not preserve source driver positions")
        replayed = make_decay_variant(binary, row["slot_index"],
                                      row["attempt_index"], variant)
        if replayed.content_hash != scene.content_hash:
            raise ValueError("decay physics stream replay differs")
        view = GeometryView(scene.geometry_id, bundle, policy)
        recomputed = _acceptance(scene, view, row["slot_index"],
                                 row["attempt_index"], variant)
        if json.loads(_canonical_json(recomputed)) != report_by_id[row["candidate_id"]]:
            raise ValueError("decay acceptance evidence differs")
    if (any(variants != {1, 2} for variants in source_variants.values()) or
            len(source_variants) != manifest["report"]["source_ready_binary_candidates"] or
            len(rows) != 2 * len(source_variants)):
        raise ValueError("decay variant ledger does not cover source READY candidates")
    if (len(build_report) != 1 or
            build_report[0]["variant_attempts"] != len(rows) or
            build_report[0]["ready_candidates"] != len(ready)):
        raise ValueError("decay build report count mismatch")
    return {**quick, "scientific_audit": "passed", "replay": replay,
            "accepted_probe_reports_verified": len(ready),
            "source_binary_scene_reconstruction_verified": True,
            "candidate_journal_verified": journal_verified}
