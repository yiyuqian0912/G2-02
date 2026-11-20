"""Fresh-process integrity and exact decoded-simplex check. Does not rerun geometry proofs."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import inputs
import evidence

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("work",type=Path)
p.add_argument("artifact",type=Path)
a=p.parse_args()
obs,identity=inputs.load(a.work/"sampling_impact/v3/scenes/0000/000001.npz",2,0)
data=a.artifact.read_bytes()
r=json.loads(data)
wire=evidence.check_artifact(r,identity)
total=sum(wire.values(),Fraction())
if total!=1 or Fraction(r["tv"])>Fraction(1,20) or r["status"]!="CERTIFIED":
    raise ArithmeticError("tail artifact failed")
build=r["certificate"]["statement"]["build"]
now=json.loads(evidence._build_bytes())
matching=all(build[k]==now[k] for k in ("source_sha256","runtime_sha256","kernel","semantics"))
if not matching:
    raise ArithmeticError("scientific source/runtime fingerprint mismatch")
print(json.dumps(dict(artifact_sha256=hashlib.sha256(data).hexdigest(),wire_sha256=r["wire_sha256"],
    bytes=len(data),status=r["status"],bound=r["tv"],bound_float=float(Fraction(r["tv"])),
    cells=len(wire),exact_sum=str(total),crop=identity["observation_xyz_sha256"],
    geometry=identity["geometry_sha256"],build_commit=build["commit"],
    matching_runtime_and_source_fingerprints=matching),indent=2))
