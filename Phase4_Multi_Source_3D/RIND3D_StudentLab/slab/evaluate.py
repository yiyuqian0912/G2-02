"""Shared evaluation of lab models (and of the frozen pilot checkpoints) on declared row sets.

usage: python -m slab.evaluate <run_dir | pilot:<ckpt.pt>> [--sets tstar,b1k,sep,alllit,alldark] [--split dev]
                                [--confirm] [--tag NAME] [--ckpt final.pt] [--override JSON]
--override: evaluation-time model options merged into the run config (e.g. '{"quad": 4}' for a quadrature change at test
time; architecture-changing keys would fail at load). Requires --tag, so the default evaluation file is never replaced.
--split confirm (pilot-eval = the 1K val split) requires --confirm and is appended to OUT/CONFIRM_LOG.jsonl; it is
reserved for the pre-declared confirmatory comparisons. The sealed 1K test split is not in the feature table.
Per example (nats, 128^3 label cells; support = cube minus the closed window box; u = uniform-legal prior):
  nll = -log p(true cell); nll_prior; ig = nll_prior - nll
  hpd = randomised HPD rank of the true cell (U(0,1) under calibration); ent = H(p); brier16 (16^3 coarse cells)
  with a certified label q: ce = -sum q log p; hq = H(q); kl = ce - hq; ce_prior; ig_ref = ce_prior - hq (label's IG);
  supp = p(supp q)
Writes <run_dir>/eval_<split>[_<tag>].npz with per-example arrays per set, plus strata columns."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch

from .paths import OUT, facets_file
from .data import LabTable, collate, to_dev, cf_of_cell
from .density import legal_logv, log_normalise
from .models import build, Generic

SETS = "tstar,b1k,sep,alllit,alldark"


def pilot_model(ck, dev):
    """A frozen pilot checkpoint, read out through the hierarchical density it was trained with (exactly)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "RIND3D_StudentResearch" / "pilot"))
    from student import Student, full_logmass
    m = Student().to(dev); m.load_state_dict(torch.load(ck, map_location=dev)); m.eval()

    def logp_of(B):
        with torch.autocast("cuda", torch.bfloat16):
            out = m(B["feats"], B["origin"], B["side"])
        lm = full_logmass(m, out, B["origin"], B["side"])            # (B, 128^3) label order
        from .density import to_cf
        return to_cf(lm.view(-1, 128, 128, 128))
    return logp_of


def lab_model(run, dev, which="final.pt", override=None):
    cfg = json.loads((Path(run) / "config.json").read_text()); cfg.update(override or {})
    m = build(cfg).to(dev); m.load_state_dict(torch.load(Path(run) / which, map_location=dev)); m.eval()

    def logp_of(B):
        with torch.autocast("cuda", torch.bfloat16):
            logits, _ = m(B)
        return log_normalise(logits.float(), legal_logv(B["origin"], B["side"]))
    return logp_of


@torch.no_grad()
def eval_rows(T, idx, logp_of, dev, bs=8, seed=12345):
    rng = np.random.default_rng(seed)
    keys = ("nll", "nll_prior", "hpd", "ent", "brier16", "ce", "hq", "kl", "ce_prior", "ig_ref", "supp")
    M = {k: np.full(len(idx), np.nan) for k in keys}
    for a in range(0, len(idx), bs):
        pick = idx[a:a + bs]
        B = to_dev(collate(T, pick, "onehot"), dev)
        lp = logp_of(B).float()                                      # (b, 4096, 512)
        lu = log_normalise(torch.zeros_like(lp), legal_logv(B["origin"], B["side"]))
        b = len(pick); ar = torch.arange(b, device=dev)
        lpt = lp[ar, B["sc"], B["sf"]]
        p = lp.exp(); flat = lp.flatten(1); pf = p.flatten(1)
        gt = (flat > lpt[:, None]); eq = (flat == lpt[:, None])
        V = torch.from_numpy(rng.random(b)).to(dev).float()
        hpd = (pf * gt).sum(1) + V * (pf * eq).sum(1)
        ent = -torch.where(pf > 0, pf * flat, torch.zeros_like(pf)).sum(1)
        Pc = p.sum(-1)
        e = torch.zeros_like(Pc); e[ar, B["sc"]] = 1
        sl = slice(a, a + b)
        M["nll"][sl] = (-lpt).cpu().numpy(); M["nll_prior"][sl] = (-lu[ar, B["sc"], B["sf"]]).cpu().numpy()
        M["hpd"][sl] = hpd.cpu().numpy(); M["ent"][sl] = ent.cpu().numpy(); M["brier16"][sl] = ((Pc - e) ** 2).sum(1).cpu().numpy()
        for j, i in enumerate(pick):
            cells, q, _ = T.label(i)
            if cells is None:
                continue
            c, f = cf_of_cell(cells)
            c = torch.from_numpy(c).to(dev); f = torch.from_numpy(f).to(dev); qt = torch.from_numpy(q).to(dev).double()
            lq = lp[j, c, f].double(); lqu = lu[j, c, f].double()
            k = a + j
            M["ce"][k] = float(-(qt * lq).sum()); M["hq"][k] = float(-(qt * torch.log(qt)).sum())
            M["kl"][k] = M["ce"][k] - M["hq"][k]; M["supp"][k] = float(lq.exp().sum())
            M["ce_prior"][k] = float(-(qt * lqu).sum()); M["ig_ref"][k] = M["ce_prior"][k] - M["hq"][k]
    return M


def main():
    tgt = sys.argv[1]
    arg = lambda n, d: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d
    sets = arg("--sets", SETS).split(","); split = arg("--split", "dev"); tag = arg("--tag", "")
    if split == "confirm" and "--confirm" not in sys.argv:
        raise SystemExit("the confirm split needs --confirm (reserved for declared confirmatory comparisons)")
    dev = "cuda"
    ver = json.loads((Path(tgt) / "config.json").read_text()).get("facets", "v1") if not tgt.startswith("pilot:") else "v1"
    T = LabTable(facets=facets_file(ver))
    if tgt.startswith("pilot:"):
        ck = tgt[6:]; logp_of = pilot_model(ck, dev)          # outputs go to the lab, never into the pilot's run directory
        od = OUT / "pilot_evals" / Path(ck).parent.name; od.mkdir(parents=True, exist_ok=True)
        outp = od / f"eval_{split}{'_' + tag if tag else ''}.npz"
    else:
        ov = json.loads(arg("--override", "{}"))
        if ov and not tag:
            raise SystemExit("--override needs --tag")
        logp_of = lab_model(tgt, dev, arg("--ckpt", "final.pt"), ov); outp = Path(tgt) / f"eval_{split}{'_' + tag if tag else ''}.npz"
    res = {}; t0 = time.time()
    for s in sets:
        idx = T.eval_set(s, split)
        M = eval_rows(T, idx, logp_of, dev)
        r = T.rows[idx]
        res.update({f"{s}/{k}": v for k, v in M.items()})
        res.update({f"{s}/idx": idx, f"{s}/scene": r["scene"], f"{s}/Lp": r["Lp"], f"{s}/family": r["family"], f"{s}/k": r["k"],
                    f"{s}/m": r["m"], f"{s}/ds": r["ds"], f"{s}/nf": T.nf[idx] if T.nf is not None else np.zeros(len(idx))})
        print(json.dumps(dict(set=s, n=int(len(idx)), ig=float(np.nanmean(M["nll_prior"] - M["nll"])),
                              kl=float(np.nanmean(M["kl"])), n_lab=int(np.sum(~np.isnan(M["ce"]))), wall=round(time.time() - t0))), flush=True)
    np.savez_compressed(outp, **res)
    if split == "confirm":
        with open(OUT / "CONFIRM_LOG.jsonl", "a") as f:
            f.write(json.dumps(dict(target=tgt, sets=sets, time=time.strftime("%Y-%m-%dT%H:%M:%S"))) + "\n")
    print(json.dumps(dict(out=str(outp))))


if __name__ == "__main__":
    main()
