"""Matched single-worker comparisons. Read only existing observations and post-run references."""
import argparse
import base64
from fractions import Fraction as Q
import gzip
from hashlib import sha256
import json
from pathlib import Path
import resource
import sys
import tempfile
from time import perf_counter, process_time

parser=argparse.ArgumentParser()
parser.add_argument("checkout",type=Path)
parser.add_argument("work",type=Path,help="v3 work directory containing sampling_impact/v3 and xlf_cbc")
parser.add_argument("output",type=Path)
parser.add_argument("--arms", choices=("baseline","patch","factorial"), required=True)
args=parser.parse_args()
config=json.loads(Path(__file__).with_name("tasks.json").read_bytes())
phase=args.checkout.resolve()/"Phase4_Multi_Source_3D"
here=phase/"RIND3D_Posterior/experiments/credit_solver"
sys.path[:0]=[str(here),str(phase/"RIND3D_Generator/src")]
import numpy as np
import inputs
from cbc_run import warm_jit
from cbc_search import solve
from ledger import decode
from run import write_result

def load(key):
    slot,w,s=key.split("_")
    return inputs.load(args.work/"sampling_impact/v3/scenes"/f"{int(slot)//1000:04d}"/(slot+".npz"),int(w[1:]),int(s[1:]))

def summary_ref(key,p,d,ident):
    if not p:
        return {}
    p=decode(p)
    out={}
    if key=="000004_w00_s0":
        with gzip.open(here.parent/"global_inverse/records/heldout_raster.jsonl.gz","rt") as f:
            row=next(r for r in map(json.loads,f) if r["task"]["key"]==key)
        assert row["unknown_pieces"]==0 and row["grid_status"]=="exact_normalized"
        assert row["geometry_sha256"]==ident["geometry_sha256"]
        ground={int(c):Q(v) for c,v in row["probabilities"].items()}
        tv=sum(abs(p.get(c,0)-ground.get(c,0)) for c in p.keys()|ground.keys())/2
        assert tv<=Q(d["tv"])
        out["tv_to_exact_214"]=float(tv)
    reference=args.work/"xlf_cbc/ref_cert/labels"/(key+"_tp_label.npz")
    if reference.exists():
        with np.load(reference) as z:
            q=dict(zip(z["cells"].tolist(),z["q"].tolist()))
            tv=sum(abs(float(p.get(c,0))-q.get(c,0)) for c in p.keys()|q.keys())/2
            out.update(tv_to_reference_float=tv,reference_bound_float=float(z["tv_bound"]))
            assert tv<=float(Q(d["tv"]))+float(z["tv_bound"])+1e-12
    return out

def parameters(key):
    if args.arms=="baseline":
        return [{}]
    if args.arms=="patch":
        return [dict(patch_depth=v) for v in (None,1,2)]
    return [dict(strategy=st,pairs=k) for st in ("interface","interface_random","random_classes","classes_spread")
            for k in (8,24)]

start=perf_counter()
warm_jit()
obs,_=load("000030_w01_s0")
solve(obs,"000030_w01_s0",Q(1,20),10)
warm=perf_counter()-start
tolerance=Q(config["tolerance"])
archive=args.work/"xlf_cbc/v301_canary_10s"
fl={r["key"]:r for r in json.loads((archive/"compare_canary.json").read_bytes())}
keys=config["factorial" if args.arms=="factorial" else "cases"]
with args.output.open("x") as sink,tempfile.TemporaryDirectory() as tmp:
    for i,key in enumerate(keys):
        arms=parameters(key)
        for arm in arms[i%len(arms):]+arms[:i%len(arms)]:
            start,cpu=perf_counter(),process_time()
            obs,identity=load(key)
            loaded=perf_counter()
            raw,d=solve(obs,key,tolerance,config["seconds"],config["cap"],**arm)
            computed=perf_counter()
            dst=Path(tmp)/"label.json"
            write_result(dst,identity,tolerance,raw,d)
            actual=json.loads(dst.read_bytes())
            wall=perf_counter()-start
            record=dict(key=key,arm=arm,side=obs.side,family=identity["family"],bodies=len(obs.bodies),
                        geometry=identity["geometry_sha256"],crop=identity["observation_xyz_sha256"],
                        warmup_once=warm,decode_s=loaded-start,solver_s=computed-loaded,total_s=wall,
                        cpu_s=process_time()-cpu,peak_process_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        status=actual["status"],reason=d["search_reason"],bound=d["tv"],cells=actual.get("cells",0),
                        wire_sha256=actual.get("wire_sha256"),unknown=d.get("unknown_volume"),lower=d.get("lower_volume"),
                        seed_volume=d.get("seed_volume"),seed_pieces=d.get("seed_pieces"),stats=d["stats"])
            if raw:
                assert base64.b64decode(actual["wire"])==raw and Q(d["tv"])<=tolerance
                record.update(summary_ref(key,raw,d,identity))
                if "certificate" in actual:
                    build=actual["certificate"]["statement"]["build"]
                    record.update(build_commit=build["commit"],build_sha256=sha256(json.dumps(build,sort_keys=True).encode()).hexdigest())
            else:
                assert actual["status"]=="INCOMPLETE" and "wire" not in actual
            archived=archive/"results"/(key+".json")
            if archived.exists():
                frozen=json.loads(archived.read_bytes())["problem"]
                if all(frozen[k]==identity[k] for k in ("origin","side","geometry_sha256","observation_xyz_sha256","channel")):
                    record["historical_fastlabel_seconds"]=fl[key].get("fl_seconds")
                    record["historical_fastlabel_status"]=fl[key].get("fl_status")
            print(json.dumps(record,sort_keys=True),file=sink,flush=True)
