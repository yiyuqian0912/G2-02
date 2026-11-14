"""Train one lab arm (resumable).

usage: python -m slab.train <config.json | json string> [--out DIR]
Config keys (defaults in DEF): name, seed, model (generic | facet) + model keys (slab.models.build), target
(posterior | onehot), weight (mis | b1k | trel | nat; pilot data.py), subset ({"scenes_frac"} | {"examples_frac"}),
steps, batch, lr, wd, warmup, workers, curve_every, curve_n.
Data: train split (pilot-train), boundary-mixed rows only; batches drawn with probability proportional to the side-
normalised population weight (unweighted loss; same expected gradient); batch s is keyed by (seed, s), so a resumed run
sees exactly the batches it would have seen.
Every curve_every steps: one-hot NLL on a fixed dev subset (the learning curve; not used to pick checkpoints: the final
checkpoint is the arm's model). State (model, optimiser, step) is saved every curve_every steps for resumption."""
import json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .paths import OUT
from .data import LabTable, collate, to_dev, side_normalised, kish, SPLIT_ROLE
from .density import legal_logv, log_normalise, sparse_ce
from .models import build

DEF = dict(model="generic", target="posterior", weight="mis", seed=0, subset=None, steps=8000, batch=32, lr=4e-4,
           wd=0.05, warmup=300, workers=6, curve_every=1000, curve_n=1024, ckpt=True)


def train_pool(T, cfg):
    r = T.rows
    m = T.split_mask("train") & (r["type"] == 0)
    if cfg["weight"] in ("b1k", "trel"):
        m &= r["ds"] == 0
    if cfg["weight"] == "nat":
        m &= r["ds"] == 1
    sub = cfg.get("subset")
    if sub:
        rng = np.random.default_rng(1000 + int(sub.get("seed", cfg["seed"])))
        if "scenes_frac" in sub:
            sc = np.unique(r["scene"][m]); keep = rng.choice(sc, int(round(len(sc) * sub["scenes_frac"])), replace=False)
            m &= np.isin(r["scene"], keep)
        elif "examples_frac" in sub:
            idx = np.flatnonzero(m); keep = rng.choice(idx, int(round(len(idx) * sub["examples_frac"])), replace=False)
            mm = np.zeros_like(m); mm[keep] = True; m = mm
    idx = np.flatnonzero(m)
    w = side_normalised(T.w[cfg["weight"]], r["Lp"])[idx]
    keep = w > 0
    return idx[keep], w[keep]


class Batches(Dataset):
    def __init__(self, T, idx, w, cfg, start, n):
        self.T, self.idx, self.cdf, self.cfg, self.start, self.n = T, idx, np.cumsum(w) / np.sum(w), cfg, start, n

    def __len__(self):
        return self.n - self.start

    def __getitem__(self, j):
        s = self.start + j
        rng = np.random.default_rng([self.cfg["seed"], s, 7])
        pick = self.idx[np.minimum(np.searchsorted(self.cdf, rng.random(self.cfg["batch"])), len(self.idx) - 1)]
        B = collate(self.T, pick, self.cfg["target"]); B["step"] = s
        return B


def loss_of(model, B):
    logits, ctx = model(B)
    logp = log_normalise(logits, legal_logv(B["origin"], B["side"]))
    return sparse_ce(logp, B["tb"], B["tc"], B["tf"], B["tq"]), logp, ctx


@torch.no_grad()
def dev_curve(model, batches, dev):
    model.eval(); out = []
    for B in batches:
        B = to_dev(B, dev)
        with torch.autocast("cuda", torch.bfloat16):
            logits, _ = model(B)
        logp = log_normalise(logits.float(), legal_logv(B["origin"], B["side"]))
        out.append((-logp[torch.arange(len(B["sc"]), device=dev), B["sc"], B["sf"]]).cpu().numpy())
    model.train()
    return float(np.concatenate(out).mean())


def main():
    arg = sys.argv[1]
    cfg = dict(DEF); cfg.update(json.loads(Path(arg).read_text()) if os.path.exists(arg) else json.loads(arg))
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else OUT / "runs" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    if (out / "final.pt").exists():
        print("already finished", out); return
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    dev = "cuda"
    T = LabTable()
    idx, w = train_pool(T, cfg)
    info = dict(n_pool=int(len(idx)), n_scenes=int(len(np.unique(T.rows["scene"][idx]))), kish=kish(w))
    model = build(cfg).to(dev)
    info["params"] = int(sum(p.numel() for p in model.parameters()))
    (out / "info.json").write_text(json.dumps(info, indent=1)); print(json.dumps(info), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.98), weight_decay=cfg["wd"])
    start = 0
    if (out / "state.pt").exists():
        st = torch.load(out / "state.pt", map_location=dev, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); start = st["step"]
        print(json.dumps(dict(resumed_at=start)), flush=True)
    lr_at = lambda s: cfg["lr"] * min(1, (s + 1) / cfg["warmup"]) * (0.025 + 0.975 * 0.5 * (1 + math.cos(math.pi * min(1, s / cfg["steps"]))))
    didx = T.eval_set("tstar", "dev")
    didx = np.sort(np.random.default_rng(5).choice(didx, min(cfg["curve_n"], len(didx)), replace=False))
    curve_batches = [collate(T, didx[i:i + 16], "onehot") for i in range(0, len(didx), 16)]
    dl = DataLoader(Batches(T, idx, w, cfg, start, cfg["steps"]), batch_size=None, shuffle=False, num_workers=cfg["workers"],
                    prefetch_factor=4, persistent_workers=False)
    log = open(out / "train_log.jsonl", "a"); t0 = time.perf_counter(); fbs = 0
    model.train()
    for B in dl:
        s = B.pop("step"); B = to_dev(B, dev)
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        with torch.autocast("cuda", torch.bfloat16):
            ce, _, ctx = loss_of(model, B)
        loss = ce.mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); fbs += B["fallback"]
        if s % 50 == 0:
            rec = dict(step=s, loss=float(loss.detach()), gnorm=float(gn), lr=lr_at(s), wall=time.perf_counter() - t0, fallback=fbs)
            if "eps_logit" in ctx and bool(ctx["fmask"].any()):
                m = ctx["fmask"]
                rec.update(eps_med=float(torch.sigmoid(ctx["eps_logit"][m]).median()), s0_med=float(ctx["s0"][m].median()),
                           sth_med=float(ctx["sth"][m].median()))
            log.write(json.dumps(rec) + "\n"); log.flush()
        if (s + 1) % cfg["curve_every"] == 0 or s + 1 == cfg["steps"]:
            v = dev_curve(model, curve_batches, dev)
            rec = dict(step=s + 1, dev_nll=v, train_loss=float(loss), wall=time.perf_counter() - t0)
            print(json.dumps(rec), flush=True)
            with open(out / "curve.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=s + 1), out / "state.tmp.pt")
            os.replace(out / "state.tmp.pt", out / "state.pt")
    torch.save(model.state_dict(), out / "final.pt")
    (out / "done.json").write_text(json.dumps(dict(info=info, wall=time.perf_counter() - t0, fallback=fbs), indent=1))


if __name__ == "__main__":
    main()
