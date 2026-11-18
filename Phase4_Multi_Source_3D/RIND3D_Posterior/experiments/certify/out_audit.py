"""Physical-oracle audit of every OUT-claimed leaf of the engine's tree: points deep inside each region (half way to
its centroid) are rendered by the certified renderer; a point reproducing every FREE receiver would be a feasible
source the engine excluded."""
import json, sys, io, numpy as np, zlib
from pathlib import Path
from rind3d_posterior.certify import scene as SC, verify as V, label as LB
from rind3d.geometry import load_geometry
from rind3d.rendering.backends import BackendFactory
from rind3d_1k.physical import POLICY
W = __import__("os").environ.get("RIND3D_V3_WORK", "/home/yushu/Projects/RIND/RIND3D_100K_v3_work")
key = sys.argv[1]; npts = int(sys.argv[2]); conc = float(sys.argv[3]); toward = float(sys.argv[4])
t = next(x for x in json.loads(Path(W + '/cmp_v3/tasks.json').read_text()) if x["key"] == key)
cls, o, N = SC.window_from_container(t["container"], t["w"], t["s"]); scene = SC.scene_from_container(t["container"])
cert = LB.load_cert(f"{W}/perf/{key}_tp_cert.npz")
rep = V.verify_certificate(scene, cls, o, N, cert)
with np.load(t["container"], allow_pickle=False) as z:
    bundle = load_geometry(io.BytesIO(z["geometry"].tobytes()))
fac = BackendFactory(cuda_allowed=False); L = cls.shape[0]; free = cls != 2
rng = np.random.default_rng(zlib.crc32(key.encode()))
zero = []; feas = 0; tested = 0; minm = []; nwin = nsolid = nrend = 0; wlo = o / N; whi = (o + L) / N
for pos, Vp in rep["audit_out_regions"]:
    ctr = Vp.mean(0); best = None
    for _ in range(npts):
        p = toward * (rng.dirichlet(np.ones(len(Vp)) * conc) @ Vp) + (1 - toward) * ctr
        tested += 1
        if np.all(p >= wlo) and np.all(p <= whi):
            nwin += 1; continue
        try:
            y, _, _ = fac.source(bundle, p, backend="cpu_native_certified", policy=POLICY).render(tuple(int(v) for v in o), L, N)
        except ValueError:
            nsolid += 1; continue
        mism = int(np.sum((np.asarray(y, bool).transpose(2, 1, 0) != (cls == 1)) & free)); nrend += 1
        best = mism if best is None else min(best, mism)
        if mism == 0:
            lb = min(max(0.0, max((float(np.dot(scene.RM[c, i, :3], p) + scene.RM[c, i, 3]) / float(np.linalg.norm(scene.RM[c, i, :3]))
                                       for i in range(int(scene.K[c])))) ) for c in range(int(scene.C)))
            feas += 1; zero.append(dict(event=int(pos), p=p.tolist(), dist_lb=lb, outside_shell=bool(lb > 2.0 ** -26)))
            print('zero-mismatch point', key, 'event', pos, p.tolist(), 'dist lower bound', lb, flush=True)
    if best is not None: minm.append(best)
print(json.dumps(dict(key=key, out_regions=len(rep["audit_out_regions"]), points=tested, in_window=nwin, in_solid=nsolid, rendered=nrend, zero_mismatch_points=feas, zero=zero,
                      min_mismatch_quantiles=np.percentile(minm, [0, 10, 50]).tolist() if minm else None, stats=rep["stats"])))
