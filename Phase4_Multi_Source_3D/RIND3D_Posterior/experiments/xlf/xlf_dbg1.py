import json, numpy as np
from pathlib import Path
from rind3d_posterior import xlf
from rind3d_posterior.certify import scene as SC, label as LB, poly as PY
from rind3d_posterior.xlf import solve as XS, pred as PR
W='/home/yushu/Projects/RIND/RIND3D_100K_v3_work'; key='000049_w03_s0'
t = next(x for x in json.loads(Path(W + "/cmp_v3/tasks.json").read_text()) if x["key"] == key)
cls, o, N = SC.window_from_container(t["container"], t["w"], t["s"]); scene = SC.scene_from_container(t["container"])
R = LB.load_label(f"{W}/perf/{key}_tp_label.npz")
cv = 1/128**3
full = R["cells"][R["lo"] > cv*(1-1e-9)]
print('cells', len(R['cells']), 'full cells', len(full))
prob = XS.Problem(scene, cls, o, N); sv = XS.Solver(prob)
c = int(full[len(full)//2]); ix, iy, iz = c // 128**2, (c // 128) % 128, c % 128
b0 = np.array([ix, iy, iz]) / 128; b1 = b0 + 1/128
s = sv.sl.get(); ax = np.array([[sv.PLs.axis_plane(k, b0[k]), sv.PLs.axis_plane(k, b1[k])] for k in range(3)], np.int64)
PY.set_box(sv.PLs.PL, s, ax, *sv.sl.args())
Vm, Vr, nv = sv._verts(s)
comps = np.zeros(prob.C, np.int64); nc = PR.relevant_comps(Vm, Vr, nv, prob.rlo, prob.rhi, prob.AL, prob.AH, prob.C, comps)
print('cell', (ix,iy,iz), 'relevant comps', comps[:nc])
rowflag = np.zeros(len(prob.rows), np.int64)
code, namb = PR.scan_rows(Vm, Vr, nv, prob.rows, len(prob.rows), cls, o, N, prob.L, comps[:nc].copy(), nc, prob.RM, prob.RR, prob.K, prob.ADJ, sv.amb, sv.cap_amb, rowflag, sv.wit)
print('scan code', code, 'namb', namb, 'wit', sv.wit)
for k in range(min(namb, 8)):
    X, Y, tt = sv.amb[k]
    st, nh, obs = sv._recv(s, X, Y, tt, comps[:nc].copy())
    print(' amb', (X,Y,tt), 'obs', obs, 'recv status', st, 'nh', nh)
# per-comp prep
for q in range(nc):
    DM=np.zeros((4096,4)); DR=np.zeros((4096,4)); LMm=np.zeros((512,4)); LRr=np.zeros((512,4))
    a,b = PR._prep_comp(comps[q], Vm, Vr, nv, prob.RM, prob.RR, prob.K, prob.ADJ, DM, DR, LMm, LRr)
    print(' comp', comps[q], 'dark constraints', a, 'lit planes', b)
