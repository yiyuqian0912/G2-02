"""Run the S6 prototype (and optionally production v3.4) on listed tasks; one JSON line per task.
usage: run_s6.py <scene_dir> <w> <s> [prod]"""
from pathlib import Path
import json, sys, time
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "RIND3D_StudentResearch/s6"))
import numpy as np
from fastlabel_s6 import label_task_s6
sd, w, s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
t = time.perf_counter()
if len(sys.argv) > 4 and sys.argv[4] == "prod":
    from rind3d_1k.fastlabel import label_task
    lab = label_task(sd, w, s, base_seed=777)
else:
    lab = label_task_s6(sd, w, s, base_seed=777)
m = lab["meta"]
print(json.dumps(dict(scene=sd[-6:], window=w, source=s, status=lab["status"], path=m.get("path"), tv_bound=float(lab["tv_bound"]),
                      n_cells=m.get("n_cells"), vol=float(np.sum(lab["hi"])), wall=time.perf_counter() - t,
                      rejected=m.get("rejected_inside_solid"), drawn=m.get("drawn"), solid_removal=m.get("solid_removal"),
                      rounds=[(x.get("round"), x.get("tested"), x.get("K"), x.get("rejected_inside_solid")) for x in m.get("rounds", [])],
                      p_lower=m.get("p_lower")), default=float), flush=True)
