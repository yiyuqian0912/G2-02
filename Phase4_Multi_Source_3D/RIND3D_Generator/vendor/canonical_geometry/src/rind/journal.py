"""Fsynced, hash-chained operational events for uncommitted build staging.

The journal is evidence, never a substitute for the committed candidate
ledger. After interruption a builder may regenerate from CURRENT, leaving the
old staging directory and its PENDING records intact for inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path


TERMINAL = {"READY", "INVALID", "UNRESOLVED"}
EVENTS = {"operation_start", "candidate_pending", "candidate_progress",
          "candidate_terminal", "operation_ready", "operation_aborted",
          "operation_committed"}


def _bytes(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _transition(states: dict[str, str], event: str, payload: dict) -> None:
    candidate = payload.get("candidate_id")
    if event == "candidate_pending":
        if not isinstance(candidate, str) or not candidate or candidate in states:
            raise ValueError("candidate was already started or has no ID")
        states[candidate] = "PENDING"
    elif event == "candidate_progress":
        if states.get(candidate) != "PENDING":
            raise ValueError("progress requires a PENDING candidate")
    elif event == "candidate_terminal":
        state = payload.get("state")
        if state not in TERMINAL or states.get(candidate) != "PENDING":
            raise ValueError("candidate requires exactly one terminal transition")
        states[candidate] = state
    elif event in ("operation_ready", "operation_committed"):
        if any(state == "PENDING" for state in states.values()):
            raise ValueError("cannot publish an operation with PENDING candidates")


@dataclass(frozen=True, slots=True)
class JournalState:
    events: tuple[dict, ...]
    candidate_states: dict[str, str]
    truncated_tail: bool


def load_journal(path: str | Path) -> JournalState:
    raw = Path(path).read_bytes()
    lines = raw.split(b"\n")
    truncated = bool(lines[-1])
    states: dict[str, str] = {}
    events = []
    previous = "0" * 64
    for index, line in enumerate(lines[:-1]):
        if not line:
            raise ValueError("empty journal line")
        entry = json.loads(line)
        if not isinstance(entry, dict) or not isinstance(entry.get("payload"), dict):
            raise ValueError("journal event is not an object with a payload")
        claimed = entry.pop("digest")
        if (entry.get("sequence") != index or entry.get("previous") != previous or
                entry.get("event") not in EVENTS or
                sha256(_bytes(entry)).hexdigest() != claimed):
            raise ValueError("journal sequence or hash chain is invalid")
        if index == 0 and entry["event"] != "operation_start":
            raise ValueError("journal does not begin with operation_start")
        if index > 0 and entry["event"] == "operation_start":
            raise ValueError("operation_start repeated")
        _transition(states, entry["event"], entry["payload"])
        events.append({**entry, "digest": claimed})
        previous = claimed
    if not events:
        raise ValueError("journal has no durable operation_start event")
    return JournalState(tuple(events), states, truncated)


class BuildJournal:
    def __init__(self, path: str | Path, operation: dict):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("xb") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        parent_fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        self._states: dict[str, str] = {}
        self._sequence = 0
        self._previous = "0" * 64
        self.append("operation_start", operation)

    def append(self, event: str, payload: dict) -> str:
        if event not in EVENTS or (event == "operation_start" and self._sequence != 0):
            raise ValueError("unsupported or repeated journal event")
        states = self._states.copy()
        _transition(states, event, payload)
        entry = {"sequence": self._sequence, "previous": self._previous,
                 "event": event, "payload": payload}
        digest = sha256(_bytes(entry)).hexdigest()
        encoded = _bytes({**entry, "digest": digest}) + b"\n"
        descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND)
        try:
            offset = 0
            while offset < len(encoded):
                written = os.write(descriptor, encoded[offset:])
                if written == 0:
                    raise OSError("journal append made no progress")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._states = states
        self._sequence += 1
        self._previous = digest
        return digest


def inspect_staging(dataset_root: str | Path) -> dict:
    """Read-only inventory of committed, interrupted and legacy staging dirs."""
    root = Path(dataset_root)
    current = ((root / "CURRENT").read_text().strip()
               if (root / "CURRENT").exists() else None)
    stages = []
    for stage in sorted((root / "staging").glob("*")):
        if not stage.is_dir():
            continue
        path = stage / "journal.jsonl"
        if not path.exists():
            stages.append({"stage": stage.name, "journal_status": "legacy_or_missing"})
            continue
        try:
            state = load_journal(path)
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            stages.append({"stage": stage.name, "journal_status": "invalid",
                           "error": str(error)})
            continue
        committed = next((event["payload"]["snapshot_id"] for event in
                          reversed(state.events)
                          if event["event"] == "operation_committed"), None)
        proposed = next((event["payload"]["snapshot_id"] for event in
                         reversed(state.events)
                         if event["event"] == "operation_ready"), None)
        stages.append({
            "stage": stage.name, "journal_status": "truncated_tail"
            if state.truncated_tail else "valid",
            "events": len(state.events),
            "candidate_states": {
                status: sum(value == status for value in state.candidate_states.values())
                for status in ("PENDING", "READY", "INVALID", "UNRESOLVED")
            },
            "pending_candidate_ids": [candidate for candidate, status in
                                      state.candidate_states.items()
                                      if status == "PENDING"],
            "operation_committed_snapshot_id": committed,
            "proposed_snapshot_id": proposed,
            "orphan_snapshot_present": (proposed is not None and
                                        (root / "snapshots" / proposed).exists() and
                                        proposed != current),
            "is_current_snapshot": committed is not None and committed == current,
        })
    return {"dataset_root": str(root.resolve()), "current_snapshot_id": current,
            "stages": stages}
