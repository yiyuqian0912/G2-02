#!/usr/bin/env python3
"""Reference-Student smoke on a RIND3D-100K root: loader -> pilot features -> one forward / backward step.
No training campaign. Two stages, because the feature code (numba) and the Student (PyTorch) live in different venvs:

  $RIND3D_LABEL_PY scripts/student_smoke.py export <root> <out.npz> [--n 64] [--population Tstar]
  $RIND3D_TORCH_PY scripts/student_smoke.py step <out.npz>

export: `loader.flatten(population=...)` examples with certified posterior supervision; the model inputs are built
from `loader.student_inputs` only (SOLID crop, one response channel, L, origin) with the pilot's
`featlib.pooled` (4 x 32^3 uint8). Targets are the sparse certified label masses; the T* weight is kept as a loss
weight. step: the pilot `Student` (RIND3D_StudentResearch/pilot/student.py), weighted cross-entropy
(`target_ce`), one backward pass; reports the loss and the gradient norm."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parent
PILOT = REPO / "RIND3D_StudentResearch" / "pilot"
sys.path[:0] = [str(HERE), str(REPO / "RIND3D_Dataset1K"), str(REPO / "RIND3D_Posterior" / "src"), str(PILOT)]
import numpy as np  # noqa: E402


def export(root, out, n=64, population="Tstar"):
    from featlib import pooled
    from rind3d_100k.loader import RIND3D100K, flatten, student_inputs
    ds = RIND3D100K(root)
    feats, origin, side, tb, tc, tq, w, ids = [], [], [], [], [], [], [], []
    for ex in flatten(ds, population=population, missing="drop"):
        x = student_inputs(ex)
        f, *_ = pooled(np.ascontiguousarray(x["solid"]), np.ascontiguousarray(x["response"]), 32)
        i = len(feats)
        feats.append(f); origin.append(x["origin"]); side.append(x["L"])
        tb.append(np.full(len(ex["target"]["cells"]), i)); tc.append(ex["target"]["cells"]); tq.append(ex["target"]["q"])
        w.append(ex["weight"]); ids.append((ex["slot"], ex["window"], ex["source"], ex["stream"], ex["status"]))
        if len(feats) >= n:
            break
    np.savez(out, feats=np.stack(feats), origin=np.array(origin, np.int64), side=np.array(side, np.int64),
             tb=np.concatenate(tb).astype(np.int64), tc=np.concatenate(tc).astype(np.int64), tq=np.concatenate(tq),
             weight=np.array(w), ids=np.array(json.dumps(ids)))
    print(json.dumps(dict(examples=len(feats), streams=sorted({i[3] for i in ids}), statuses=sorted({i[4] for i in ids}),
                          weight_range=[float(min(w)), float(max(w))])))


def step(path, chunk=8):
    import torch
    from student import Student, target_ce
    z = np.load(path)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model = Student().to(dev)
    n = len(z["origin"]); wall = torch.from_numpy(z["weight"]).float()
    tb_all, tc_all, tq_all = z["tb"], z["tc"], z["tq"]
    loss_tot = 0.0
    for a in range(0, n, chunk):                              # micro-batches, gradient accumulation (6 GB GPUs)
        b = min(a + chunk, n); sel = (tb_all >= a) & (tb_all < b)
        feats = torch.from_numpy(z["feats"][a:b]).to(dev); origin = torch.from_numpy(z["origin"][a:b]).to(dev)
        side = torch.from_numpy(z["side"][a:b]).to(dev)
        out = model(feats, origin, side)
        ce = target_ce(model, out, origin, side, torch.from_numpy(tb_all[sel] - a).to(dev), torch.from_numpy(tc_all[sel]).to(dev),
                       torch.from_numpy(tq_all[sel]).float().to(dev))
        part = (wall[a:b].to(dev) * ce).sum() / wall.sum().to(dev)
        part.backward(); loss_tot += float(part)
    loss = torch.tensor(loss_tot)
    g = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None))
    res = dict(examples=int(n), loss=float(loss), loss_finite=bool(torch.isfinite(loss)),
               grad_norm=float(g), params=int(sum(p.numel() for p in model.parameters())), device=dev)
    print(json.dumps(res))
    return res


if __name__ == "__main__":
    if sys.argv[1] == "export":
        n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 64
        pop = sys.argv[sys.argv.index("--population") + 1] if "--population" in sys.argv else "Tstar"
        export(sys.argv[2], sys.argv[3], n, pop)
    else:
        r = step(sys.argv[2])
        sys.exit(0 if r["loss_finite"] and r["grad_norm"] > 0 else 1)
