"""Human-readable reports derived from a committed snapshot."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pyarrow.parquet as pq

from .publication import audit_snapshot, validate_snapshot


def snapshot_report(dataset: str | Path, output: str | Path, *,
                    deep: bool = False, replay: bool = False) -> dict:
    if replay and not deep:
        raise ValueError("replay requires deep audit")
    root = Path(dataset).resolve()
    destination = Path(output).resolve()
    result = audit_snapshot(root, replay=replay) if deep else validate_snapshot(root)
    snapshot = root / "snapshots" / result["snapshot_id"]
    if destination == snapshot or snapshot in destination.parents:
        raise ValueError("report output cannot modify a committed snapshot")
    manifest = json.loads((snapshot / "manifest.json").read_text())
    catalogs = snapshot / "catalogs"
    ledger = pq.read_table(catalogs / "candidate_ledger.parquet").to_pylist()
    counts = Counter(row["candidate_state"] for row in ledger)
    reasons = Counter((row["candidate_state"], row["reason_code"])
                      for row in ledger if row["reason_code"])
    record = {
        "snapshot_id": result["snapshot_id"],
        "profile_id": ledger[0]["profile_id"] if ledger else None,
        "profile_digest": manifest["profile_digest"],
        "backend_build": manifest["backend_build"],
        "slots": manifest["slots"],
        "integrity": "deep+replay" if replay else "deep" if deep else "quick",
        "scientific_audit": result.get("scientific_audit", "not_run"),
        "attempts_started": len(ledger),
        "candidate_states": dict(sorted(counts.items())),
        "unique_ready_scenes": result["unique_ready_scenes"],
        "unique_geometries": result["geometries"],
        "stored_references": result["references"],
        "reason_counts": [
            {"state": state, "reason_code": reason, "count": count}
            for (state, reason), count in sorted(reasons.items())
        ],
        "declared_capabilities": manifest["declared_capabilities"],
        "diagnostics": [],
    }
    for path in sorted((root / "diagnostics").glob("*.json")):
        try:
            diagnostic = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if diagnostic.get("snapshot_id") == result["snapshot_id"]:
            record["diagnostics"].append({"file": str(path),
                                          "kind": diagnostic.get("adapter", "benchmark")})
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "overview.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    state_lines = "\n".join(f"| {state} | {count} |"
                            for state, count in sorted(counts.items()))
    reason_lines = "\n".join(
        f"| {state} | {reason} | {count} |"
        for (state, reason), count in sorted(reasons.items())) or "| — | — | 0 |"
    diagnostics = "\n".join(f"- `{item['kind']}`: `{item['file']}`"
                            for item in record["diagnostics"]) or "- None measured for this snapshot."
    markdown = f"""# RIND snapshot report

Snapshot: `{record['snapshot_id']}`  
Profile: `{record['profile_id']}` (`{record['profile_digest']}`)  
Backend build: `{record['backend_build']}`  
Slots: half-open `{record['slots'][0]}:{record['slots'][1]}`  
Verification run for this report: **{record['integrity']}**. Scientific audit: **{record['scientific_audit']}**.

## Inventory

| Metric | Count |
|---|---:|
| Started attempts | {record['attempts_started']} |
| READY candidates | {counts.get('READY', 0)} |
| Unique READY scenes | {record['unique_ready_scenes']} |
| Unique geometries | {record['unique_geometries']} |
| Stored references | {record['stored_references']} |

## Candidate states

| State | Attempts |
|---|---:|
{state_lines}

## Recorded failure reasons

| State | Reason code | Attempts |
|---|---|---:|
{reason_lines}

## Derived diagnostics

{diagnostics}

Grid diagnostics and previews do not define canonical scene truth or READY status. The declared query scope is in `overview.json`; unsupported codecs are not implicitly certified.
"""
    (destination / "overview.md").write_text(markdown)
    return {"snapshot_id": result["snapshot_id"], "output": str(destination),
            "integrity": record["integrity"], "candidate_states": record["candidate_states"]}
