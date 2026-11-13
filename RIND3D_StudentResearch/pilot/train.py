"""Pilot Student training: one arm per run.

usage: train.py <feat_dir> <out_dir> '<arm json>'
arm keys:
  name, seed
  target:  "posterior" (certified FastLabel / tau-pencil label field; label-or-one-hot fallback if a label is missing)
           or "onehot" (true-source cell)
  weight:  "b1k" | "trel" | "mis"  (data.py); the training pool is every mixed TRAINING-stratum row with weight > 0
  stratum: "boundary_mixed" (default; solid-separated, all-lit, all-dark are held out for OOD) or "mixed"
  subset:  optional {"scenes_frac": f} or {"examples_frac": f} (seeded; scene-count vs channel-count ablation)
  steps (8000), batch (32), ckpt (True: recompute cross blocks in backward; memory), lr (4e-4), wd (0.05), warmup (300), ckpt_every (1000), select_n (2000)
Selection: every ckpt_every steps, one-hot NLL on a fixed pilot-select set drawn from the arm's own target population
(b1k: release rows; trel / mis: natural rows), weighted; the best checkpoint is kept. pilot-eval rows are never read."""
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import Table, side_normalised, kish, ROLES, TYPES   # noqa: E402
from student import Student, target_ce                         # noqa: E402

DEF = dict(steps=8000, batch=32, ckpt=True, lr=4e-4, wd=0.05, warmup=300, ckpt_every=1000, select_n=2000, stratum="boundary_mixed",
           target="posterior", weight="b1k", seed=0, subset=None, workers=6)


def pool_mask(T, arm, role):
    r = T.rows
    types = (0,) if arm["stratum"] == "boundary_mixed" else (0, 1)
    m = (r["role"] == ROLES.index(role)) & np.isin(r["type"], types)
    if arm["weight"] == "b1k":
        m &= r["ds"] == 0
    elif arm["weight"] == "trel":
        m &= r["ds"] == 0
    return m


def subset_mask(T, m, sub, seed):
    if not sub:
        return m
    rng = np.random.default_rng(1000 + seed)
    r = T.rows
    if "scenes_frac" in sub:
        sc = np.unique(r["scene"][m]); keep = rng.choice(sc, int(round(len(sc) * sub["scenes_frac"])), replace=False)
        return m & np.isin(r["scene"], keep)
    if "examples_frac" in sub:
        idx = np.flatnonzero(m); keep = rng.choice(idx, int(round(len(idx) * sub["examples_frac"])), replace=False)
        out = np.zeros_like(m); out[keep] = True
        return out
    raise ValueError(sub)


class Batches(Dataset):
    """Item s = the batch of step s (indices drawn by the step's own RNG: reproducible and worker-independent)."""

    def __init__(self, T, idx, p, arm, n_steps):
        self.T, self.idx, self.cdf, self.arm, self.n = T, idx, np.cumsum(p) / np.sum(p), arm, n_steps

    def __len__(self):
        return self.n

    def __getitem__(self, s):
        rng = np.random.default_rng([self.arm["seed"], s, 7])
        pick = self.idx[np.minimum(np.searchsorted(self.cdf, rng.random(self.arm["batch"])), len(self.idx) - 1)]
        return collate(self.T, pick, self.arm["target"])


def collate(T, pick, target):
    r = T.rows[pick]
    feats = torch.from_numpy(T.feats(pick))
    origin = torch.from_numpy(np.stack([r["ox"], r["oy"], r["oz"]], 1).astype(np.int64))
    side = torch.from_numpy(r["Lp"].astype(np.int64))
    tb, tc, tq, fb = [], [], [], 0
    for b, i in enumerate(pick):
        cells = q = None
        if target == "posterior":
            cells, q, _ = T.label(i)
        if cells is None:                                  # one-hot target, or label-or-one-hot fallback
            cells = np.array([T.source_cell(i)]); q = np.ones(1, np.float32)
            fb += target == "posterior"
        tb.append(np.full(len(cells), b)); tc.append(cells); tq.append(q)
    return dict(feats=feats, origin=origin, side=side, tb=torch.from_numpy(np.concatenate(tb)),
                tc=torch.from_numpy(np.concatenate(tc)), tq=torch.from_numpy(np.concatenate(tq)), fallback=fb,
                src=torch.from_numpy(np.array([T.source_cell(i) for i in pick])))


@torch.no_grad()
def onehot_nll(model, B, dev):
    with torch.autocast("cuda", torch.bfloat16):
        out = model(B["feats"].to(dev), B["origin"].to(dev), B["side"].to(dev))
    n = len(B["src"])
    return target_ce(model, out, B["origin"].to(dev), B["side"].to(dev), torch.arange(n, device=dev), B["src"].to(dev),
                     torch.ones(n, device=dev)).cpu().numpy()


def main():
    T = Table(sys.argv[1]); out = Path(sys.argv[2]); arm = dict(DEF); arm.update(json.loads(sys.argv[3]))
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(arm["seed"]); np.random.seed(arm["seed"])
    dev = "cuda"
    m = subset_mask(T, pool_mask(T, arm, "pilot-train"), arm["subset"], arm["seed"])
    idx = np.flatnonzero(m)
    w = side_normalised(T.w[arm["weight"]], T.rows["Lp"])[idx]
    keep = w > 0; idx, w = idx[keep], w[keep]
    # selection set: arm's own population on pilot-select
    types = (0,) if arm["stratum"] == "boundary_mixed" else (0, 1)
    ms = (T.rows["role"] == ROLES.index("pilot-select")) & np.isin(T.rows["type"], types)
    ms &= (T.rows["ds"] == (0 if arm["weight"] == "b1k" else 1))      # B1K: release windows; T*: natural windows
    sidx = np.flatnonzero(ms)
    sw = side_normalised(T.w["b1k" if arm["weight"] == "b1k" else "nat"], T.rows["Lp"])[sidx]
    rng = np.random.default_rng(99)
    sel = sidx[np.sort(rng.choice(len(sidx), min(arm["select_n"], len(sidx)), replace=False, p=sw / sw.sum()))]
    info = dict(arm=arm, n_pool=int(len(idx)), n_scenes=int(len(np.unique(T.rows["scene"][idx]))), kish=kish(w),
                kish_frac=kish(w) / len(idx), n_select=int(len(sel)),
                side_shares={str(L): float(w[T.rows["Lp"][idx] == L].sum() / w.sum()) for L in (64, 128, 256)})
    print(json.dumps(info), flush=True)
    (out / "info.json").write_text(json.dumps(info, indent=1))
    model = Student(ckpt=bool(arm.get("ckpt", False))).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=arm["lr"], betas=(0.9, 0.98), weight_decay=arm["wd"])
    lr_at = lambda s: arm["lr"] * min(1, (s + 1) / arm["warmup"]) * (0.025 + 0.975 * 0.5 * (1 + math.cos(math.pi * min(1, s / arm["steps"]))))
    dl = DataLoader(Batches(T, idx, w, arm, arm["steps"]), batch_size=None, shuffle=False, num_workers=arm["workers"],
                    persistent_workers=False, prefetch_factor=4)
    sel_batches = [collate(T, sel[i:i + 32], "onehot") for i in range(0, len(sel), 32)]
    best, hist, t0, fbs = None, [], time.perf_counter(), 0
    log = open(out / "train_log.jsonl", "a")
    for s, B in enumerate(dl):
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        model.train()
        with torch.autocast("cuda", torch.bfloat16):
            o = model(B["feats"].to(dev, non_blocking=True), B["origin"].to(dev), B["side"].to(dev))
        loss = target_ce(model, o, B["origin"].to(dev), B["side"].to(dev), B["tb"].to(dev), B["tc"].to(dev),
                         B["tq"].to(dev)).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); fbs += B["fallback"]
        if s % 50 == 0:
            log.write(json.dumps(dict(step=s, loss=float(loss), gnorm=float(gn), lr=lr_at(s), wall=time.perf_counter() - t0,
                                      fallback=fbs)) + "\n"); log.flush()
        if (s + 1) % arm["ckpt_every"] == 0 or s + 1 == arm["steps"]:
            model.eval()
            nll = np.concatenate([onehot_nll(model, b, dev) for b in sel_batches])
            v = float(nll.mean())                      # sel was drawn proportional to the population weight
            rec = dict(step=s + 1, select_nll=v, train_loss=float(loss), wall=time.perf_counter() - t0, fallback=fbs)
            hist.append(rec); print(json.dumps(rec), flush=True)
            torch.save(model.state_dict(), out / f"ckpt_{s + 1}.pt")
            if best is None or v < best["select_nll"]:
                best = rec; torch.save(model.state_dict(), out / "best.pt")
    (out / "done.json").write_text(json.dumps(dict(best=best, history=hist, info=info, wall=time.perf_counter() - t0), indent=1))


if __name__ == "__main__":
    main()
