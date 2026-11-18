"""Run the XLF solver on one task and compare with a stored certified label (if given)."""
import json, sys, time
from pathlib import Path
import numpy as np
from rind3d_posterior import xlf  # noqa: F401  (numba cache dir)
from rind3d_posterior.certify import scene as SC, tvb, label as LB
from rind3d_posterior.xlf import solve as XS

W = __import__("os").environ.get("RIND3D_V3_WORK", "/home/yushu/Projects/RIND/RIND3D_100K_v3_work")
key = sys.argv[1]
t = next(x for x in json.loads(Path(W + "/cmp_v3/tasks.json").read_text()) if x["key"] == key)
t0 = time.perf_counter()
cls, o, N = SC.window_from_container(t["container"], t["w"], t["s"]); scene = SC.scene_from_container(t["container"])
t1 = time.perf_counter()
prob = XS.Problem(scene, cls, o, N)
sv = XS.Solver(prob).run(time_limit=float(sys.argv[2]) if len(sys.argv) > 2 else None)
cells, lo, hi, pool = sv.result()
b, Lt = tvb.tv_bound_exact(cells, lo, hi, pool)
out = dict(key=key, L=int(cls.shape[0]), load_s=t1 - t0, solve_s=sv.seconds, tv=float(b) if b is not None else None,
           Lt=float(np.sum(lo)), Ht=float(np.sum(hi) + pool), pool=pool, cells=int(np.sum(lo > 0)), stats=sv.stats,
           planes=sv.PLs.n, bsp_nodes=len(sv.bsp_parent))
ref = Path(f"{W}/perf/{key}_tp_label.npz")
if ref.exists():
    R = LB.load_label(ref)
    out["ref"] = dict(tv=float(R["tv_bound"]), Lt=float(np.sum(R["lo"])), Ht=float(np.sum(R["hi"]) + float(R["pool"])),
                      cells=int(np.sum(R["lo"] > 0)))
    # consistency of the two certified enclosures: lo_xlf <= hi_ref and lo_ref <= hi_xlf per cell
    d = {int(c): (l, h) for c, l, h in zip(R["cells"], R["lo"], R["hi"])}
    bad = 0
    for c, l, h in zip(cells, lo, hi):
        rl, rh = d.get(int(c), (0.0, float(R["pool"])))
        if l > rh + float(R["pool"]) or rl > h + pool:
            bad += 1
    for c, (rl, rh) in d.items():
        if rl > 0 and c not in set(cells.tolist()) and rl > pool:
            bad += 1
    out["enclosure_conflicts"] = bad
print(json.dumps(out, default=float))
