"""Diagnose the one-receiver thin-slab tail against its historical matched FastLabel cost."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import resource
import sys
from tempfile import TemporaryDirectory
from time import perf_counter,process_time

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import numpy as np
import inputs
from cbc_run import warm_jit
from cbc_search import solve
from run import write_result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work",type=Path)
    parser.add_argument("final_panel",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    key="000001_w02_s0"
    baseline=next(r for r in map(json.loads,args.final_panel.read_text().splitlines())
                  if r["key"]==key and r["arm"].get("patch_depth") is None)
    target=10*baseline["historical_fastlabel_seconds"]
    warm_jit()
    scenes=args.work/"sampling_impact/v3/scenes/0000"
    obs,_=inputs.load(scenes/"000030.npz",1,0)
    solve(obs,"000030_w01_s0",Fraction(1,20),10)
    # Reserve four seconds for the final snapshot and I/O. Record real time even on overrun.
    # A second, deliberately slower attempt measures whether abstention persists beyond the target.
    with args.output.open("x") as output, TemporaryDirectory() as tmp:
        for budget in (int(target)-4,120):
            start,cpu=perf_counter(), process_time()
            obs,identity=inputs.load(scenes/"000001.npz",2,0)
            assert identity["geometry_sha256"]==baseline["geometry"]
            assert identity["observation_xyz_sha256"]==baseline["crop"]
            raw,d=solve(obs,key,Fraction(1,20),budget)
            path=Path(tmp)/"label.json"
            write_result(path,identity,Fraction(1,20),raw,d)
            rec=json.loads(path.read_bytes())
            wall=perf_counter()-start
            row=dict(key=key, soft_seconds=budget, wall_s=wall, cpu_s=process_time()-cpu,
                     matched_10x_seconds=target, peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     status=rec["status"], bound=rec["tv"], cells=rec.get("cells",0),
                     wire_sha256=rec.get("wire_sha256"), reason=d["search_reason"],
                     lower_volume=d["lower_volume"], unresolved_volume=d["unknown_volume"],
                     class_counts_xyz=np.bincount(obs.cls.reshape(-1)).tolist(), stats=d["stats"])
            print(json.dumps(row,sort_keys=True),file=output,flush=True)
            if rec["status"]=="CERTIFIED":
                break


if __name__=="__main__":
    main()
