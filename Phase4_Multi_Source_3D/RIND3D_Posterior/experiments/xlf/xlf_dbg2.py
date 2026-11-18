import json, sys, numpy as np
from pathlib import Path
from rind3d_posterior import xlf
from rind3d_posterior.certify import scene as SC
from rind3d_posterior.xlf import solve as XS, pred as PR
W='/home/yushu/Projects/RIND/RIND3D_100K_v3_work'; key=sys.argv[1]
t = next(x for x in json.loads(Path(W + "/cmp_v3/tasks.json").read_text()) if x["key"] == key)
cls, o, N = SC.window_from_container(t["container"], t["w"], t["s"]); scene = SC.scene_from_container(t["container"])
prob = XS.Problem(scene, cls, o, N); sv = XS.Solver(prob)
orig_fine = sv._fine; orig_proc = sv._process
log = []
def proc(node):
    if node.fine > 0 and len(log) < 12:
        Vm, Vr, nv = sv._verts(node.slot)
        comps = np.zeros(prob.C, np.int64); nc = PR.relevant_comps(Vm, Vr, nv, prob.rlo, prob.rhi, prob.AL, prob.AH, prob.C, comps)
        rowflag = np.zeros(len(node.rows), np.int64)
        code, namb = PR.scan_rows(Vm, Vr, nv, node.rows, len(node.rows), cls, o, N, prob.L, comps[:nc].copy(), nc, prob.RM, prob.RR, prob.K, prob.ADJ, sv.amb, sv.cap_amb, rowflag, sv.wit, sv.rcap)
        sts = []
        for k in range(min(namb, 50)):
            X, Y, tt = sv.amb[k]; st, nh, obs = sv._recv(node.slot, X, Y, tt, comps[:nc].copy()); sts.append((st, nh, obs))
        from collections import Counter
        log.append((node.fine, nv, code, namb, Counter((a, c) for a, b, c in sts)))
        print('fine', node.fine, 'nv', nv, 'code', code, 'namb', namb, 'statuses (st,obs)', dict(Counter((a, c) for a, b, c in sts)), flush=True)
        if namb and len(log) < 4:
            X, Y, tt = sv.amb[0]; st, nh, obs = sv._recv(node.slot, X, Y, tt, comps[:nc].copy())
            for h in range(nh):
                lo, hi = PR.range_on(sv.hs_m[h], sv.hs_r[h], Vm, Vr, nv)
                print('    receiver', (X,Y,tt), 'hs', h, 'key', sv.hs_key[h].tolist(), 'sense', sv.hs_s[h], 'range on node', lo, hi)
    if len(log) >= 12:
        raise SystemExit
    return orig_proc(node)
sv._process = proc
try:
    sv.run(time_limit=300)
except SystemExit:
    pass
print(sv.stats)
