"""Single-writer local snapshots for the 2D core pilot.

Publication keeps each snapshot immutable and switches a small CURRENT pointer
only after Zarr, Parquet, hashes, and acceptance reports have been verified.
"""

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

import pyarrow as pa
import pyarrow.parquet as pq

from .contracts import (CSGNode, Geometry, Leaf, QUERY_SEMANTICS,
                        SOLID_SEMANTICS)
from .generation import (
    MASTER_SEED, MAX_ATTEMPTS, PROFILE_ID, CandidateInvalid,
    CandidateUnresolved, make_binary_scene, propose_geometry_2d,
    sample_drivers_2d, validate_candidate_2d, _exact_det, _ring_valid,
)
from .geometry import GeometryView, backend_build_id
from .identity import DTypeTag as T
from .identity import FRAMING_ID, HASH_FAMILY, Field, digest, hash_bytes_field
from .journal import BuildJournal, load_journal
from .storage import CanonicalBundle, read_bundle, write_bundle


GENERATOR_BUILD = "rind-generator-impl1-" + hashlib.sha256(
    Path(__file__).with_name("generation.py").read_bytes()).hexdigest()[:24]
CONTRACT_VERSION = "rind-implementation-contract-1"
_DATASET_DIR = Path(__file__).resolve().parents[2]
_PROFILE_PATH = _DATASET_DIR / "profiles" / f"{PROFILE_ID}.json"
_POLICY_PATH = _DATASET_DIR / "profiles" / "rind-numeric-3-impl1.json"
_SPEC_PATH = _DATASET_DIR / "RIND_Dataset_v3_Final_Spec_Generic_Plain.md"


def _bytes(path: Path) -> bytes:
    return path.read_bytes()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json(value) + b"\n")


def _profile() -> tuple[dict, bytes]:
    profile = json.loads(_PROFILE_PATH.read_text())
    if profile["profile_id"] != PROFILE_ID or profile["master_seed"] != MASTER_SEED:
        raise ValueError("profile and generator constants disagree")
    return profile, hashlib.sha256(_canonical_json(profile)).digest()


def _lineage_id(profile_digest: bytes, slot: int, attempt: int) -> bytes:
    return digest("lineage", (
        Field.numeric("master_seed", T.UINT64, MASTER_SEED),
        Field.text("profile_id", PROFILE_ID),
        hash_bytes_field("profile_digest", profile_digest),
        Field.numeric("dimension", T.UINT8, 2),
        Field.numeric("slot_index", T.UINT64, slot),
        Field.numeric("attempt_index", T.UINT32, attempt),
        Field.text("generator_build", GENERATOR_BUILD),
    ))


def _candidate_id(lineage_id: bytes) -> bytes:
    return digest("candidate", (
        hash_bytes_field("lineage_id", lineage_id),
        Field.numeric("scene_variant_index", T.UINT32, 0),
    ))


def _merge_bundle(destination: CanonicalBundle, source: CanonicalBundle) -> None:
    for item in source.references:
        if isinstance(item, Leaf):
            destination.add_leaf(item)
        elif isinstance(item, CSGNode):
            destination.add_node(item)
        else:
            raise TypeError("unknown reference type")
    for geometry in source.geometries:
        destination.add_geometry(geometry)


def _table(path: Path, rows: list[dict]) -> None:
    if not rows:
        # Keep an actual Parquet table even when a catalog is empty.
        table = pa.table({"_empty": pa.array([], type=pa.uint8())})
    else:
        table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd")


def _fsync_tree(path: Path) -> None:
    for directory, _, filenames in os.walk(path):
        for filename in filenames:
            with open(Path(directory) / filename, "rb") as handle:
                os.fsync(handle.fileno())
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _checksums(path: Path) -> dict[str, str]:
    return {
        str(file.relative_to(path)): _sha(file.read_bytes())
        for file in sorted(path.rglob("*")) if file.is_file()
        and file.name not in ("checksums.json", "COMMIT.json")
    }


def _verify_detached_snapshot(path: Path, snapshot_id: str) -> dict:
    """Verify a renamed snapshot even when CURRENT still points to its parent."""
    commit = json.loads((path / "COMMIT.json").read_text())
    checksum_bytes = _bytes(path / "checksums.json")
    manifest = json.loads((path / "manifest.json").read_text())
    if (commit.get("snapshot_id") != snapshot_id or
            manifest.get("snapshot_id") != snapshot_id or
            commit.get("checksums_sha256") != _sha(checksum_bytes) or
            json.loads(checksum_bytes) != _checksums(path)):
        raise ValueError("detached snapshot failed commit or file verification")
    return manifest


def _publish_snapshot(staged: Path, output: Path, snapshot_id: str,
                      *, expected_current: str | None) -> None:
    """Install or adopt a verified orphan, then atomically switch CURRENT.

    Repeating a build after a crash between snapshot rename and pointer switch
    produces the same content-derived ID. The orphan is reused only after its
    files and immutable scientific provenance match the newly checked staging.
    """
    pointer = output / "CURRENT"
    observed_current = pointer.read_text().strip() if pointer.exists() else None
    if observed_current != expected_current:
        raise RuntimeError("CURRENT changed while snapshot was being built")
    destination = output / "snapshots" / snapshot_id
    destination.parent.mkdir(exist_ok=True)
    staged_manifest = _verify_detached_snapshot(staged, snapshot_id)
    if destination.exists():
        existing_manifest = _verify_detached_snapshot(destination, snapshot_id)
        fields = ("parent_snapshot_id", "source_snapshot_id", "spec_sha256",
                  "profile_digest", "generator_build", "backend_build",
                  "supported_dimensions", "slots", "dataset_kind")
        if any(existing_manifest.get(key) != staged_manifest.get(key) for key in fields):
            raise ValueError("orphan snapshot provenance differs from rebuilt candidate")
        # Timestamps and wall-clock reports may differ across attempts; the
        # canonical arrays, evidence and catalog mappings must not differ.
        variable = {"manifest.json", "catalogs/build_report.parquet",
                    "catalogs/candidate_ledger.parquet"}
        existing_checksums = json.loads((destination / "checksums.json").read_text())
        staged_checksums = json.loads((staged / "checksums.json").read_text())
        if ({name: value for name, value in existing_checksums.items()
             if name not in variable} !=
            {name: value for name, value in staged_checksums.items()
             if name not in variable}):
            raise ValueError("orphan snapshot semantic files differ from rebuilt candidate")
        ledger_name = "catalogs/candidate_ledger.parquet"
        existing_ledger = destination / ledger_name
        staged_ledger = staged / ledger_name
        if existing_ledger.exists() != staged_ledger.exists():
            raise ValueError("orphan snapshot ledger presence differs")
        if existing_ledger.exists():
            volatile = {"created_at", "updated_at", "wall_clock_ms"}
            def stable_rows(path: Path) -> list[dict]:
                return [{key: value for key, value in row.items()
                         if key not in volatile}
                        for row in pq.read_table(path).to_pylist()]
            if stable_rows(existing_ledger) != stable_rows(staged_ledger):
                raise ValueError("orphan candidate ledger differs from rebuilt decisions")
    else:
        os.replace(staged, destination)
    parent_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    temporary = output / f".CURRENT.{uuid.uuid4().hex}.tmp"
    temporary.write_text(snapshot_id + "\n")
    with open(temporary, "rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, pointer)
    root_fd = os.open(output, os.O_RDONLY)
    try:
        os.fsync(root_fd)
    finally:
        os.close(root_fd)


def _copy_candidate_journal(journal: BuildJournal, snapshot: Path) -> None:
    """Freeze the candidate-decision prefix inside the immutable snapshot."""
    shutil.copy2(journal.path, snapshot / "catalogs" / "candidate_journal.jsonl")


def _audit_candidate_journal(snapshot: Path, ledger: list[dict]) -> bool:
    path = snapshot / "catalogs" / "candidate_journal.jsonl"
    manifest = json.loads((snapshot / "manifest.json").read_text())
    declared = manifest.get("candidate_journal_location")
    if declared not in (None, "catalogs/candidate_journal.jsonl"):
        raise ValueError("unsupported candidate journal locator")
    if declared is not None and not path.exists():
        raise ValueError("declared candidate journal is missing")
    if not path.exists():
        return False  # Historical snapshots predate the journal extension.
    state = load_journal(path)
    operation = state.events[0]["payload"]
    if (operation.get("dimension") != manifest["supported_dimensions"][0] or
            operation.get("profile_digest") != manifest["profile_digest"] or
            operation.get("generator_build") != manifest["generator_build"] or
            operation.get("backend_build") != manifest["backend_build"]):
        raise ValueError("published journal operation provenance differs")
    slots = operation.get("slots")
    if slots is not None:
        published_slots = manifest["slots"]
        if (not isinstance(slots, list) or len(slots) != 2 or
                not all(isinstance(value, int) for value in slots) or
                not published_slots[0] <= slots[0] < slots[1] or
                slots[1] != published_slots[1] or
                operation.get("parent_snapshot_id") != manifest["parent_snapshot_id"]):
            raise ValueError("published journal slot or parent provenance differs")
    elif operation.get("source_snapshot_id") != manifest.get("source_snapshot_id"):
        raise ValueError("published journal source provenance differs")
    covered = (ledger if slots is None else
               [row for row in ledger if slots[0] <= row["slot_index"] < slots[1]])
    expected = {row["candidate_id"]: row["candidate_state"] for row in covered}
    if (state.truncated_tail or len(expected) != len(covered) or
            state.candidate_states != expected or
            any(value == "PENDING" for value in state.candidate_states.values())):
        raise ValueError("published journal and terminal candidate ledger disagree")
    by_id = {row["candidate_id"]: row for row in covered}
    pending = {}
    terminal = {}
    for entry in state.events[1:]:
        payload = entry["payload"]
        candidate_id = payload.get("candidate_id")
        if entry["event"] == "candidate_pending":
            pending[candidate_id] = payload
        elif entry["event"] == "candidate_terminal":
            terminal[candidate_id] = payload
    if set(pending) != set(by_id) or set(terminal) != set(by_id):
        raise ValueError("published journal candidate event coverage differs")
    for candidate_id, row in by_id.items():
        start = pending[candidate_id]
        end = terminal[candidate_id]
        if (start.get("slot_index") != row["slot_index"] or
                start.get("attempt_index") != row["attempt_index"] or
                end.get("state") != row["candidate_state"] or
                end.get("geometry_id") != row["geometry_id"] or
                end.get("scene_id") != row["scene_id"] or
                end.get("reason_code") != row["reason_code"]):
            raise ValueError("published journal candidate payload differs from ledger")
    return True


def validate_snapshot(root: str | Path) -> dict:
    """Pin CURRENT once, then verify commit, files, SoA and terminal ledger."""
    root = Path(root)
    snapshot_id = (root / "CURRENT").read_text().strip()
    if not snapshot_id or "/" in snapshot_id or ".." in snapshot_id:
        raise ValueError("invalid CURRENT pointer")
    snapshot = root / "snapshots" / snapshot_id
    commit = json.loads((snapshot / "COMMIT.json").read_text())
    if commit["snapshot_id"] != snapshot_id:
        raise ValueError("commit snapshot identity mismatch")
    checksum_bytes = _bytes(snapshot / "checksums.json")
    if _sha(checksum_bytes) != commit["checksums_sha256"]:
        raise ValueError("commit/checksum digest mismatch")
    expected = json.loads(checksum_bytes)
    if expected != _checksums(snapshot):
        raise ValueError("snapshot file checksum mismatch")
    manifest = json.loads((snapshot / "manifest.json").read_text())
    if manifest["snapshot_id"] != snapshot_id:
        raise ValueError("manifest identity mismatch")
    if manifest["framing_id"] != FRAMING_ID or manifest["hash_family"] != HASH_FAMILY:
        raise ValueError("unsupported identity semantics")
    if manifest["query_semantics_id"] != QUERY_SEMANTICS:
        raise ValueError("unsupported query semantics")
    dimensions = manifest.get("supported_dimensions", [])
    if dimensions not in ([2], [3]):
        raise ValueError("unsupported snapshot dimension partition")
    dimension = dimensions[0]
    expected_store = {f"d{dimension}": f"d{dimension}/core.zarr"}
    if manifest.get("store_locations") != expected_store:
        raise ValueError("unexpected canonical store locator")
    if (manifest.get("candidate_journal_location") is not None and
            manifest["candidate_journal_location"] !=
            "catalogs/candidate_journal.jsonl"):
        raise ValueError("unsupported candidate journal locator")
    if (manifest.get("candidate_journal_location") is not None and
            not (snapshot / manifest["candidate_journal_location"]).exists()):
        raise ValueError("declared candidate journal is missing")
    bundle = read_bundle(snapshot / f"d{dimension}" / "core.zarr", dimension)
    ledger = pq.read_table(snapshot / "catalogs" / "candidate_ledger.parquet")
    records = ledger.to_pylist()
    terminal = {"READY", "INVALID", "UNRESOLVED"}
    if any(row["candidate_state"] not in terminal for row in records):
        raise ValueError("published ledger has nonterminal candidates")
    candidate_ids = [row["candidate_id"] for row in records]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate ledger has duplicate attempts")
    scene_ids = {scene.content_hash.hex() for scene in bundle.scenes}
    if any(row["scene_id"] not in scene_ids for row in records
           if row["candidate_state"] == "READY"):
        raise ValueError("READY candidate points to absent scene")
    return {
        "snapshot_id": snapshot_id,
        "dimension": dimension,
        "references": len(bundle.references),
        "geometries": len(bundle.geometries),
        "scenes": len(bundle.scenes),
        "attempts": len(records),
        "ready_candidates": sum(row["candidate_state"] == "READY" for row in records),
        "unique_ready_scenes": len({row["scene_id"] for row in records
                                    if row["candidate_state"] == "READY"}),
    }


def _replay_failed_2d(row: dict, policy: dict) -> None:
    """Reject a failure label that deterministic generation cannot reproduce."""
    slot, attempt = row["slot_index"], row["attempt_index"]
    geometry_id = scene_id = None
    reason = ""
    try:
        proposal, geometry = propose_geometry_2d(slot, attempt)
    except CandidateInvalid as error:
        observed = "INVALID"
        reason = error.reason
    else:
        geometry_id = geometry.content_hash.hex()
        view = GeometryView(geometry.content_hash, proposal, policy)
        try:
            drivers = sample_drivers_2d(view, slot, attempt)
        except CandidateInvalid as error:
            observed, reason = "INVALID", error.reason
        except CandidateUnresolved as error:
            observed, reason = "UNRESOLVED", error.reason
        else:
            scene = make_binary_scene(proposal, geometry, drivers.positions)
            scene_id = scene.content_hash.hex()
            try:
                validate_candidate_2d(scene, view, slot, attempt)
            except CandidateUnresolved as error:
                observed, reason = "UNRESOLVED", error.reason
            else:
                observed = "READY"
    if (row["candidate_state"] != observed or row["reason_code"] != reason or
            row["geometry_id"] != geometry_id or row["scene_id"] != scene_id):
        raise ValueError(f"failed candidate replay differs: d2:{slot} attempt {attempt}")


def audit_snapshot(root: str | Path, *, replay: bool = False) -> dict:
    """Recompute pilot scientific gates from the pinned published records."""
    quick = validate_snapshot(root)
    manifest = json.loads((Path(root) / "snapshots" / quick["snapshot_id"] /
                           "manifest.json").read_text())
    if manifest.get("dataset_kind") == "decay-variants-v1":
        from .publication_decay import audit_decay
        return audit_decay(root, replay=replay)
    if quick["dimension"] == 3:
        from .publication3d import audit_snapshot_3d
        return audit_snapshot_3d(root, replay=replay)
    root = Path(root)
    snapshot = root / "snapshots" / quick["snapshot_id"]
    manifest = json.loads((snapshot / "manifest.json").read_text())
    registry = json.loads((snapshot / "registry.json").read_text())
    profile = json.loads((snapshot / "profiles" / _PROFILE_PATH.name).read_text())
    policy = json.loads((snapshot / "numeric_policy.json").read_text())
    if manifest["generator_build"] != GENERATOR_BUILD:
        raise ValueError("current generator build differs from snapshot provenance")
    if manifest["backend_build"] != backend_build_id():
        raise ValueError("current geometry backend differs from snapshot provenance")
    if manifest["spec_sha256"] != _sha(_bytes(_SPEC_PATH)):
        raise ValueError("snapshot was built against a different spec")
    if manifest["codec_registry_digest"] != _sha(_canonical_json(registry)):
        raise ValueError("registry digest mismatch")
    if manifest["profile_digest"] != _sha(_canonical_json(profile)):
        raise ValueError("profile digest mismatch")
    if manifest["numeric_policy_digest"] != _sha(_canonical_json(policy)):
        raise ValueError("numeric policy digest mismatch")
    if profile["profile_id"] != PROFILE_ID or policy["policy_id"] != "rind-numeric-3-impl1":
        raise ValueError("unsupported pilot profile or numeric policy")
    if manifest["supported_dimensions"] != [2] or profile["enabled_run_dimensions"] != [2]:
        raise ValueError("dimension capability mismatch")
    if manifest["store_locations"] != {"d2": "d2/core.zarr"}:
        raise ValueError("unexpected canonical store locator")
    bundle = read_bundle(snapshot / "d2" / "core.zarr", 2)
    from .contracts import CodecID
    for item in bundle.references:
        if not isinstance(item, Leaf):
            continue
        if item.codec not in (CodecID.AFFINE_BOX, CodecID.AFFINE_BALL,
                              CodecID.POLYGON_RINGS_2D):
            raise ValueError("pilot contains a codec outside its declared distribution")
        if _exact_det(item.field("matrix").as_array()) == 0:
            raise ValueError("stored affine transform is singular")
        if item.codec == CodecID.POLYGON_RINGS_2D:
            if item.field("ring_role").as_array().tolist() != [0]:
                raise ValueError("pilot polygon recipe unexpectedly contains holes")
            _ring_valid(item.field("vertices").as_array())

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
        raise ValueError("catalogs do not cover every attempt")
    if len(geometry_catalog) != len(bundle.geometries) or len(scene_catalog) != len(bundle.scenes):
        raise ValueError("catalog row count differs from canonical store")
    if any(row["geometry_id"] != geometry.content_hash.hex() or row["row"] != i
           for i, (row, geometry) in enumerate(zip(geometry_catalog, bundle.geometries, strict=True))):
        raise ValueError("geometry catalog locator mismatch")
    if any(row["scene_id"] != scene.content_hash.hex() or row["row"] != i
           for i, (row, scene) in enumerate(zip(scene_catalog, bundle.scenes, strict=True))):
        raise ValueError("scene catalog locator mismatch")
    expected_slots = list(range(*manifest["slots"]))
    if [row["slot_id"] for row in slot_summary] != [f"d2:{slot}" for slot in expected_slots]:
        raise ValueError("slot summary has missing, extra or reordered slots")
    rows_by_slot: dict[int, list[dict]] = {}
    profile_digest = bytes.fromhex(manifest["profile_digest"])
    for row, event, lineage in zip(ledger, events, lineage_map, strict=True):
        slot, attempt = int(row["slot_index"]), int(row["attempt_index"])
        rows_by_slot.setdefault(slot, []).append(row)
        expected_lineage = _lineage_id(profile_digest, slot, attempt).hex()
        expected_candidate = _candidate_id(bytes.fromhex(expected_lineage)).hex()
        if (row["lineage_id"] != expected_lineage or
                row["candidate_id"] != expected_candidate or
                event["candidate_id"] != expected_candidate or
                event["event"] != row["candidate_state"] or
                lineage["candidate_id"] != expected_candidate or
                lineage["geometry_id"] != row["geometry_id"] or
                lineage["scene_id"] != row["scene_id"]):
            raise ValueError("lineage, candidate or terminal event mismatch")
    if sorted(rows_by_slot) != expected_slots:
        raise ValueError("candidate ledger does not cover requested slots")
    for summary in slot_summary:
        slot = int(summary["slot_id"].split(":")[1])
        rows = rows_by_slot[slot]
        if ([row["attempt_index"] for row in rows] != list(range(len(rows))) or
                summary["attempts_started"] != len(rows) or
                summary["final_state"] != rows[-1]["candidate_state"] or
                len(rows) > MAX_ATTEMPTS):
            raise ValueError("slot attempt sequence or final state mismatch")
        if any(row["candidate_state"] == "READY" for row in rows[:-1]):
            raise ValueError("READY slot was silently retried")
    ready = [row for row in ledger if row["candidate_state"] == "READY"]
    if len(reports) != len(ready):
        raise ValueError("acceptance reports do not cover every READY candidate")
    report_by_id = {row["candidate_id"]: json.loads(row["report_json"]) for row in reports}
    if set(report_by_id) != {row["candidate_id"] for row in ready}:
        raise ValueError("acceptance report candidate IDs mismatch")
    geometries = {item.content_hash.hex(): item for item in bundle.geometries}
    scenes = {item.content_hash.hex(): item for item in bundle.scenes}
    referenced_geometries = {row["geometry_id"] for row in ledger
                             if row["geometry_id"] is not None}
    if set(geometries) != referenced_geometries:
        raise ValueError("published geometry table differs from candidate attempts")
    referenced_scenes = {row["scene_id"] for row in ledger
                         if row["scene_id"] is not None}
    if set(scenes) != referenced_scenes:
        raise ValueError("published scene table differs from candidate attempts")
    for row in ready:
        geometry = geometries[row["geometry_id"]]
        scene = scenes[row["scene_id"]]
        if scene.geometry_id != geometry.content_hash:
            raise ValueError("scene points to a different geometry")
        if scene.physics.codec_id != 0 or not 1 <= len(scene.driver_position.as_array()) <= 4:
            raise ValueError("pilot physics or driver count mismatch")
        view = GeometryView(geometry.content_hash, bundle, policy)
        recomputed = validate_candidate_2d(scene, view, row["slot_index"], row["attempt_index"])
        stored = report_by_id[row["candidate_id"]]
        for key, value in recomputed.items():
            if stored.get(key) != json.loads(_canonical_json(value)):
                raise ValueError(f"acceptance evidence differs: {row['slot_id']} {key}")
        if replay:
            proposal_bundle, proposed_geometry = propose_geometry_2d(
                row["slot_index"], row["attempt_index"])
            if proposed_geometry.content_hash != geometry.content_hash:
                raise ValueError("geometry stream replay differs from published identity")
            proposed_view = GeometryView(proposed_geometry.content_hash, proposal_bundle, policy)
            drivers = sample_drivers_2d(proposed_view, row["slot_index"], row["attempt_index"])
            proposed_scene = make_binary_scene(proposal_bundle, proposed_geometry, drivers.positions)
            if proposed_scene.content_hash != scene.content_hash:
                raise ValueError("driver stream replay differs from published scene")
            if list(drivers.proposals_per_driver) != stored["driver_proposals_per_driver"]:
                raise ValueError("driver proposal counters differ")
    if replay:
        for row in ledger:
            if row["candidate_state"] != "READY":
                _replay_failed_2d(row, policy)
    if len(build_report) != 1 or build_report[0]["ready_candidates"] != len(ready):
        raise ValueError("build report count mismatch")
    return {**quick, "scientific_audit": "passed", "replay": replay,
            "accepted_probe_reports_verified": len(ready),
            "affine_nonsingularity_checked": True,
            "slot_and_terminal_invariants_checked": True,
            "candidate_journal_verified": journal_verified,
            "failed_attempts_replayed": sum(
                row["candidate_state"] != "READY" for row in ledger) if replay else 0}


def build_2d(start: int, stop: int, output: str | Path, *, resume: bool = False) -> dict:
    """Generate [start, stop) with official write/reload gates and atomic commit."""
    if start < 0 or stop <= start or stop > 128:
        raise ValueError("slots must be a nonempty half-open subset of [0,128)")
    profile, profile_digest = _profile()
    policy = json.loads(_POLICY_PATH.read_text())
    output = Path(output)
    parent_id = None
    publish_start = start
    bundle = CanonicalBundle(2)
    ledger: list[dict] = []
    validation_reports: list[dict] = []
    if (output / "CURRENT").exists():
        if not resume:
            raise FileExistsError("a committed snapshot exists; use resume to append slots")
        parent = validate_snapshot(output)
        parent_id = parent["snapshot_id"]
        parent_path = output / "snapshots" / parent_id
        parent_manifest = json.loads((parent_path / "manifest.json").read_text())
        if (parent_manifest["slots"][1] != start or
                parent_manifest["profile_digest"] != profile_digest.hex() or
                parent_manifest["generator_build"] != GENERATOR_BUILD or
                parent_manifest["backend_build"] != backend_build_id()):
            raise ValueError("resume requires adjacent slots and identical profile/backend build")
        publish_start = parent_manifest["slots"][0]
        bundle = read_bundle(parent_path / "d2" / "core.zarr", 2)
        ledger = pq.read_table(parent_path / "catalogs" / "candidate_ledger.parquet").to_pylist()
        validation_reports = [json.loads(row["report_json"]) for row in
                              pq.read_table(parent_path / "catalogs" /
                                            "query_validation_report.parquet").to_pylist()]
    elif resume:
        raise FileNotFoundError("cannot resume: no committed CURRENT snapshot")
    output.mkdir(parents=True, exist_ok=True)
    staging_root = output / "staging" / uuid.uuid4().hex
    staging_root.mkdir(parents=True)
    journal = BuildJournal(staging_root / "journal.jsonl", {
        "dimension": 2, "slots": [start, stop], "parent_snapshot_id": parent_id,
        "profile_digest": profile_digest.hex(), "generator_build": GENERATOR_BUILD,
        "backend_build": backend_build_id(),
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
                lineage_id = _lineage_id(profile_digest, slot, attempt)
                candidate_id = _candidate_id(lineage_id)
                row = {
                    "slot_id": f"d2:{slot}", "slot_index": slot,
                    "attempt_index": attempt, "scene_variant_index": 0,
                    "candidate_id": candidate_id.hex(),
                    "lineage_id": lineage_id.hex(),
                    "profile_id": PROFILE_ID, "profile_digest": profile_digest.hex(),
                    "dimension": 2, "candidate_state": "PENDING",
                    "reason_code": "", "reason_detail": "",
                    "geometry_id": None, "scene_id": None,
                    "precision_bits_used": 0, "subdivisions_used": 0,
                    "generator_build": GENERATOR_BUILD,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                ledger.append(row)
                journal.append("candidate_pending", {
                    "candidate_id": row["candidate_id"],
                    "slot_index": slot, "attempt_index": attempt,
                })
                try:
                    proposed, geometry = propose_geometry_2d(slot, attempt)
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
            reloaded = read_bundle(geometry_store, 2)
            scene_wave: dict[int, tuple[bytes, tuple[int, ...]]] = {}
            for slot, geometry_id in sorted(wave.items()):
                row = next(entry for entry in reversed(ledger)
                           if entry["slot_index"] == slot and entry["attempt_index"] == attempt)
                view = GeometryView(geometry_id, reloaded, policy)
                try:
                    driver_sample = sample_drivers_2d(view, slot, attempt)
                    geometry = next(item for item in reloaded.geometries
                                    if item.content_hash == geometry_id)
                    scene = make_binary_scene(reloaded, geometry, driver_sample.positions)
                    scene_wave[slot] = (scene.content_hash, driver_sample.proposals_per_driver)
                    row["scene_id"] = scene.content_hash.hex()
                    journal.append("candidate_progress", {
                        "candidate_id": row["candidate_id"], "stage": "scene_proposed",
                        "scene_id": row["scene_id"],
                        "driver_proposals_per_driver": driver_sample.proposals_per_driver,
                    })
                except CandidateInvalid as exc:
                    row.update(candidate_state="INVALID", reason_code=exc.reason,
                               reason_detail="proved no eligible driver")
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
            reloaded = read_bundle(scene_store, 2)
            bundle = reloaded
            for slot, (scene_id, proposal_counts) in sorted(scene_wave.items()):
                row = next(entry for entry in reversed(ledger)
                           if entry["slot_index"] == slot and entry["attempt_index"] == attempt)
                geometry_id = bytes.fromhex(row["geometry_id"])
                scene = next(item for item in bundle.scenes if item.content_hash == scene_id)
                view = GeometryView(geometry_id, bundle, policy)
                try:
                    report = validate_candidate_2d(scene, view, slot, attempt)
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
        if unfinished:
            raise AssertionError("candidate slots not terminal after attempt limit")
        if any(row["candidate_state"] == "PENDING" for row in ledger):
            raise AssertionError("candidate ledger contains PENDING after build")

        terminal_rows = [row for row in ledger if row["candidate_state"] == "READY"]
        snapshot_id = _sha(_canonical_json({
            "profile_digest": profile_digest.hex(), "slots": [publish_start, stop],
            "parent_snapshot_id": parent_id,
            "generator_build": GENERATOR_BUILD,
            "backend_build": backend_build_id(),
            "terminal": [(row["candidate_id"], row["candidate_state"],
                          row["geometry_id"], row["scene_id"])
                         for row in ledger],
        }))
        snapshot = staging_root / "snapshot"
        (snapshot / "d2").mkdir(parents=True)
        (snapshot / "catalogs").mkdir()
        write_bundle(snapshot / "d2" / "core.zarr", bundle)
        # Repacking after the scene roundtrip proves that the final store is
        # exactly the canonical content later exposed by the snapshot reader.
        final_bundle = read_bundle(snapshot / "d2" / "core.zarr", 2)
        if {s.content_hash for s in final_bundle.scenes} != {s.content_hash for s in bundle.scenes}:
            raise AssertionError("final scene store changed during snapshot assembly")
        _table(snapshot / "catalogs" / "candidate_ledger.parquet", ledger)
        _table(snapshot / "catalogs" / "geometry_catalog.parquet", [
            {"dimension": 2, "geometry_id": g.content_hash.hex(), "row": i,
             "store": "d2/core.zarr"} for i, g in enumerate(bundle.geometries)
        ])
        _table(snapshot / "catalogs" / "scene_catalog.parquet", [
            {"dimension": 2, "scene_id": s.content_hash.hex(), "geometry_id": s.geometry_id.hex(),
             "row": i, "store": "d2/core.zarr"} for i, s in enumerate(bundle.scenes)
        ])
        _table(snapshot / "catalogs" / "content_lineage_map.parquet", [
            {"lineage_id": row["lineage_id"], "candidate_id": row["candidate_id"],
             "geometry_id": row["geometry_id"], "scene_id": row["scene_id"],
             "slot_id": row["slot_id"]} for row in ledger
        ])
        summary = [{"slot_id": f"d2:{slot}",
                    "attempts_started": sum(row["slot_index"] == slot for row in ledger),
                    "final_state": next(row["candidate_state"] for row in reversed(ledger)
                                        if row["slot_index"] == slot)}
                   for slot in range(publish_start, stop)]
        _table(snapshot / "catalogs" / "slot_summary.parquet", summary)
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
            "ready_candidates": len(terminal_rows),
            "unique_ready_scenes": len({row["scene_id"] for row in terminal_rows}),
            "invalid_attempts": sum(row["candidate_state"] == "INVALID" for row in ledger),
            "unresolved_attempts": sum(row["candidate_state"] == "UNRESOLVED" for row in ledger),
            "wall_clock_seconds_diagnostic": time.monotonic() - begun,
        }
        _table(snapshot / "catalogs" / "build_report.parquet", [report])
        _copy_candidate_journal(journal, snapshot)
        (snapshot / "profiles").mkdir()
        shutil.copy2(_PROFILE_PATH, snapshot / "profiles" / _PROFILE_PATH.name)
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
            "generator_build": GENERATOR_BUILD,
            "backend_build": backend_build_id(),
            "rng_engine": "NumPy-PCG64",
            "dependency_lock_digest": _sha(_bytes(_DATASET_DIR / "requirements.lock")),
            "supported_dimensions": [2],
            "declared_capabilities": GeometryView(bundle.geometries[0].content_hash, bundle, policy).capability()
            if bundle.geometries else {},
            "store_locations": {"d2": "d2/core.zarr"},
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
        # A killed process cannot write this event; its last PENDING event is
        # still durable. Never turn an operational interruption into INVALID.
        try:
            journal.append("operation_aborted", {
                "error_type": type(error).__name__, "detail": str(error)[:500],
            })
        except Exception:
            pass
        raise
