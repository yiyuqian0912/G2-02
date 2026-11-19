"""Appending to a JSONL file after a crash left a torn final line must not create a corrupt middle line (found by the
2026-11-18 release canary resume test: a torn manifest line was merged with the next record)."""
import json

from rind3d_100k.storage import append_line, read_jsonl


def test_append_after_torn_tail(tmp_path):
    p = tmp_path / "physical.jsonl"
    append_line(p, json.dumps({"slot": 0, "status": "accepted"}))
    with open(p, "a") as f:
        f.write('{"slot": 99, "status": "acc')                 # crash mid-append
    assert [r["slot"] for r in read_jsonl(p)] == [0]          # torn tail ignored by readers
    append_line(p, json.dumps({"slot": 1, "status": "accepted"}))
    append_line(p, json.dumps({"slot": 2, "status": "accepted"}))
    assert [r["slot"] for r in read_jsonl(p)] == [0, 1, 2]
    assert p.read_text().endswith("\n") and "99" not in p.read_text()


def test_append_after_torn_only_line(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"slot": 5, "sta')
    append_line(p, json.dumps({"slot": 6}))
    assert read_jsonl(p) == [{"slot": 6}]
