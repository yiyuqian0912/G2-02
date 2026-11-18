"""Near-solid OUT audit: for every OUT-claimed leaf of the engine's tree with a vertex near the solid, render points
on segments from that vertex towards the region centroid (inside the region by convexity) at geometric distances, and
record for every zero-mismatch point its distance to the solid (lower bound from facet planes, upper bound from a
closest-point solve)."""
import json, sys, io, numpy as np
from pathlib import Path
from scipy.optimize import minimize
from rind3d_posterior.certify import scene as SC, verify as V, label as LB
from rind3d.geometry import load_geometry
from rind3d.rendering.backends import BackendFactory
from rind3d_1k.physical import POLICY
W = __import__("os").environ.get("RIND3D_V3_WORK", "/home/yushu/Projects/RIND/RIND3D_100K_v3_work")
key = sys.argv[1]
t = next(x for x in json.loads(Path(W + '/cmp_v3/tasks.json').read_text()) if x["key"] == key)
cls, o, N = SC.window_from_container(t["container"], t["w"], t["s"]); scene = SC.scene_from_container(t["container"])
cert = LB.load_cert(f"{W}/perf/{key}_tp_cert.npz")
rep = V.verify_certificate(scene, cls, o, N, cert)
with np.load(t["container"], allow_pickle=False) as z:
    bundle = load_geometry(io.BytesIO(z["geometry"].tobytes()))
fac = BackendFactory(cuda_allowed=False); L = cls.shape[0]; free = cls != 2
A = [scene.RM[c, :int(scene.K[c])] for c in range(int(scene.C))]
def dist_lb(p):
    return min(max(0.0, float(np.max((Ac[:, :3] @ p + Ac[:, 3]) / np.linalg.norm(Ac[:, :3], axis=1)))) for Ac in A)
def dist_ub(p):
    best = np.inf
    for Ac in A:
        n = np.linalg.norm(Ac[:, :3], axis=1); G = Ac[:, :3] / n[:, None]; h = Ac[:, 3] / n
        r = minimize(lambda x: np.sum((x - p) ** 2), p, jac=lambda x: 2 * (x - p), method="SLSQP",
                     constraints=[dict(type="ineq", fun=lambda x: -(G @ x + h), jac=lambda x: -G)], options=dict(ftol=1e-30, maxiter=200))
        if np.all(G @ r.x + h <= 1e-15):
            best = min(best, float(np.linalg.norm(r.x - p)))
    return best
ts = np.logspace(-10, -2, 9)
nreg = nv = nr = 0; zero = []; minmis = []
for pos, Vp in rep["audit_out_regions"]:
    ctr = Vp.mean(0); near = [v for v in Vp if dist_lb(v) < 1e-6]
    if not near:
        continue
    nreg += 1
    for v in near:
        nv += 1
        for tt in ts:
            p = v + tt * (ctr - v)
            try:
                y, _, _ = fac.source(bundle, p, backend="cpu_native_certified", policy=POLICY).render(tuple(int(u) for u in o), L, N)
            except ValueError:
                continue
            nr += 1
            mism = int(np.sum((np.asarray(y, bool).transpose(2, 1, 0) != (cls == 1)) & free)); minmis.append(mism)
            if mism == 0:
                zero.append(dict(event=int(pos), t=float(tt), p=p.tolist(), lb=dist_lb(p), ub=dist_ub(p)))
print(json.dumps(dict(key=key, near_regions=nreg, near_vertices=nv, rendered=nr, zero_mismatch=len(zero),
                      zero_outside_shell=sum(1 for z in zero if z["lb"] > 2.0 ** -26),
                      zero_max_ub=max((z["ub"] for z in zero), default=None), shell=2.0 ** -26,
                      min_mismatch=np.percentile(minmis, [0, 5, 50]).tolist() if minmis else None, zero_first=zero[:5])))
