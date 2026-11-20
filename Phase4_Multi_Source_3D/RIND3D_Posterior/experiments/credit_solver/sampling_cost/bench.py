"""Matched physical inputs and durable wire checks. Invoke separately per checked-out implementation."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import resource
import sys
import tempfile
from time import perf_counter,process_time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("kernel_dir",type=Path)
p.add_argument("work",type=Path)
p.add_argument("output",type=Path)
p.add_argument("--cases",nargs="+",required=True,help="slot_wNN_sN:soft_seconds")
a=p.parse_args()
sys.path.insert(0,str(a.kernel_dir.resolve()))
import inputs
from cbc_kernel import Kernel
from cbc_run import warm_jit
from cbc_search import solve
from run import write_result

def profile(name):
    original=getattr(Kernel,name)
    def timed(self,*args,**kwargs):
        start=perf_counter()
        try:
            return original(self,*args,**kwargs)
        finally:
            self.stats[name+"_profile_s"]+=perf_counter()-start
            self.stats[name+"_calls"]+=1
    setattr(Kernel,name,timed)

for op in ("snapshot","_commit"):
    profile(op)
t=perf_counter()
warm_jit()
warm=perf_counter()-t
tol=Fraction(1,20)
with a.output.open("x") as out,tempfile.TemporaryDirectory() as tmp:
    for entry in a.cases:
        key,budget=entry.split(":")
        slot,w,j=key.split("_")
        t,cpu=perf_counter(),process_time()
        obs,ident=inputs.load(a.work/"sampling_impact/v3/scenes/0000"/(slot+".npz"),int(w[1:]),int(j[1:]))
        raw,details=solve(obs,key,tol,int(budget))
        path=Path(tmp)/"label.json"
        write_result(path,ident,tol,raw,details)
        r=json.loads(path.read_bytes())
        if raw is not None:
            import evidence
            evidence.check_artifact(r,ident)
        print(json.dumps(dict(key=key,budget=int(budget),wall=perf_counter()-t,cpu=process_time()-cpu,
              warmup=warm,status=r["status"],tv=float(Fraction(r["tv"])) if r.get("tv") else None,
              cells=r.get("cells",0),wire=r.get("wire_sha256"),stats=details["stats"],trace=details.get("trace"),
              lower=details.get("lower_volume"),unknown=details.get("unknown_volume"),
              rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              geometry=ident["geometry_sha256"],crop=ident["observation_xyz_sha256"])),file=out,flush=True)

