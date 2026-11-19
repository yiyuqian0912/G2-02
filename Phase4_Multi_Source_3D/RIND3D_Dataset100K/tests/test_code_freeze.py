"""Code freeze of a campaign root (contract v3 labels.code_freeze): resuming under different labeller / physics /
pipeline code is refused unless explicitly allowed, and the change is logged."""
import json

import pytest

from rind3d_100k.coordinator import check_code, code_provenance, contract_digest


def test_code_provenance_is_stable_and_nonempty():
    a = code_provenance(); b = code_provenance()
    assert a == b and a["files"] > 20 and len(a["digest"]) == 64


def test_resume_with_changed_code_refused_unless_allowed(tmp_path):
    cur = dict(digest="a" * 64, files=1, trees=[])
    assert check_code(tmp_path, current=cur) == cur
    assert check_code(tmp_path, current=cur) == cur                       # same code: fine
    other = dict(cur, digest="b" * 64)
    with pytest.raises(SystemExit):
        check_code(tmp_path, current=other)
    check_code(tmp_path, allow_change=True, current=other)
    log = (tmp_path / "logs" / "code_changes.jsonl").read_text().splitlines()
    assert json.loads(log[0]) == dict(old="a" * 64, new="b" * 64)


def test_documentation_fields_do_not_change_contract_digest():
    from pathlib import Path
    c = json.loads((Path(__file__).resolve().parent.parent / "contract" / "RIND3D_100K_CONTRACT_v3.json").read_text())
    d = json.loads(json.dumps(c)); d["labels"].pop("normalization"); d["labels"].pop("code_freeze"); d["status"] = "x"
    assert contract_digest(c) == contract_digest(d)
    e = json.loads(json.dumps(c)); e["labels"]["fastlabel"]["n_tasks"] += 1
    assert contract_digest(c) != contract_digest(e)
