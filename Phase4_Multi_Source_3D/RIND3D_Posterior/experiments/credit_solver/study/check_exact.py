"""Compare released bytes against every complete rational reference in two frozen archives."""
import argparse
from fractions import Fraction
import gzip
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import evidence
import inputs
from cbc_run import warm_jit
from cbc_search import solve
from run import write_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    records = HERE.parent / "global_inverse" / "records"
    receipts = json.loads((records/"inputs.json").read_bytes())
    warm_jit()
    with args.output.open("x") as sink, TemporaryDirectory() as tmp:
        for archive in ("heldout_raster","fresh_raster"):
            with gzip.open(records/(archive+".jsonl.gz"), "rt") as stream:
                for ref in map(json.loads,stream):
                    if ref.get("grid_status") != "exact_normalized" or ref["unknown_pieces"] != 0:
                        continue
                    task = ref["task"]
                    key, slot = task["key"], task["slot"]
                    path = args.work/"sampling_impact/v3/scenes"/f"{slot//1000:04d}"/f"{slot:06d}.npz"
                    obs, ident = inputs.load(path,task["w"],task["s"])
                    expected = receipts[key]
                    assert ident["geometry_sha256"] == ref["geometry_sha256"] == expected["geometry"]
                    assert ident["observation_xyz_sha256"] == expected["classes_xyz"]
                    assert all(ident[k]==expected[k] for k in ("origin","side","resolution"))
                    start = perf_counter()
                    raw, diagnostic = solve(obs,key,Fraction(1,20),10)
                    label = Path(tmp)/"posterior.json"
                    write_result(label,ident,Fraction(1,20),raw,diagnostic)
                    rec = json.loads(label.read_bytes())
                    row = dict(key=key, family=task["family"], side=task["L"], archive=archive,
                               status=rec["status"], bound=rec["tv"], seconds=perf_counter()-start,
                               geometry=ident["geometry_sha256"], crop=ident["observation_xyz_sha256"])
                    if rec["status"]=="CERTIFIED":
                        p = evidence.check_artifact(rec,ident)
                        q = {int(c): Fraction(v) for c,v in ref["probabilities"].items()}
                        assert sum(q.values())==1 and all(v>0 for v in q.values())
                        tv = sum(abs(p.get(c,0)-q.get(c,0)) for c in p.keys()|q.keys())/2
                        assert tv <= Fraction(rec["tv"]) <= Fraction(1,20)
                        row.update(cells=len(p), exact_reference_cells=len(q), exact_error=str(tv),
                                   error_float=float(tv), wire_sha256=rec["wire_sha256"])
                    else:
                        assert rec["status"]=="INCOMPLETE" and "wire" not in rec
                    print(json.dumps(row,sort_keys=True),file=sink,flush=True)


if __name__=="__main__":
    main()
