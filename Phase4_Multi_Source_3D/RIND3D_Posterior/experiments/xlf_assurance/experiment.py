"""Reproduce the wire check on an existing posterior without rerunning geometry."""
from fractions import Fraction as Q
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter

from ledger import decode, encode
from test_assurance import physical_corner


def main():
    root = Path(__file__).resolve().parent
    file = root.parent / "global_inverse/records/grids_heldout_raster/000004_w00_s0.json.gz"
    raw = file.read_bytes()
    d = json.loads(gzip.decompress(raw))
    weights = dict(zip(d["cells"], map(int, d["weights"])))
    assert len(weights) == len(d["cells"]) == 214
    assert sum(weights.values()) == int(d["denominator"])
    t0 = perf_counter()
    wire = encode(weights)
    p = decode(wire)
    exact = {c: Q(w, int(d["denominator"])) for c, w in weights.items()}
    drift = sum(abs(p.get(c,0)-v) for c,v in exact.items()) / 2
    wall = perf_counter() - t0
    old_bound = Q(int(d["tv_numerator"]), int(d["tv_denominator"]))
    ledger = physical_corner()
    state, _, upper = ledger.certify(Q(1,20))
    report = dict(archive=file.name, archive_sha256=hashlib.sha256(raw).hexdigest(),
                  cells=len(weights), wire_bytes=len(wire), wire_sha256=hashlib.sha256(wire).hexdigest(),
                  exact_wire_sum=str(sum(p.values())), encoding_tv=str(drift),
                  inherited_archive_bound=str(old_bound), composed_bound=str(drift+old_bound),
                  encoding_tv_approx=float(drift), wire_check_seconds=wall,
                  incomplete_fixture=dict(status=state, upper=str(upper), tolerance="1/20"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
