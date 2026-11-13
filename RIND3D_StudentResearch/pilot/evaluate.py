"""Evaluate a pilot checkpoint on pilot-eval rows (the 1K val split; never used for training or selection).

usage: evaluate.py <feat_dir> <ckpt.pt> <out.npz> [--sets tstar,b1k,ood] [--role pilot-eval]
Row sets:
  tstar : NAT-1K rows, boundary-mixed (in-distribution, T* population; equal weights within side)
  b1k   : release rows, boundary-mixed (in-distribution, B1K population; weight omega_hat)
  ood   : NAT-1K rows of type solid-separated / all-lit / all-dark, and release solid-separated rows (never trained on)
Per-example metrics (nats; 128^3 label cells; the Student's support is the cube minus the closed window box):
  nll        -log p(cell of the true source)            nll_prior: same for the uniform-legal prior u
  hpd_u      randomised HPD rank of the true cell: p(p > p_true) + V p(p = p_true); U(0,1) under calibration
  brier16    Brier score of the 16^3 coarse-cell masses against the true coarse cell
  ent        entropy of p (cells); eff_support = exp(ent); ig = KL(p || u)
  with a certified reference label q (if one exists): ce = -sum q log p, hq = H(q), kl = ce - hq, supp_mass = p(supp q),
  ce_prior = -sum q log u, ig_ref = KL(q || u)
Also: coarse-cell reliability histograms (predicted mass vs. hit frequency) per set.
Subsamples (seed 20261113, identical for all arms): b1k 6000 rows; ood_nat_alllit 3000 rows incl. every row with a reference label."""
import json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import Table, ROLES, side_normalised       # noqa: E402
from student import Student, full_logmass, axis_overlap, NFINE   # noqa: E402

BINS = np.concatenate([[0], np.logspace(-7, 0, 29)])


def uniform_logmass(origin, side):
    ov = axis_overlap(origin, side, NFINE)               # (B, 3, 128)
    win = ov[:, 0, :, None, None] * ov[:, 1, None, :, None] * ov[:, 2, None, None, :]
    v = (1 - win).reshape(origin.shape[0], -1).clamp_min(0)
    return torch.log(v / v.sum(1, keepdim=True)).masked_fill(v <= 0, float("-inf"))


def main():
    T = Table(sys.argv[1]); ck = sys.argv[2]; outp = sys.argv[3]
    arg = lambda n, d: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d
    sets = arg("--sets", "tstar,b1k,ood").split(","); role = ROLES.index(arg("--role", "pilot-eval"))
    dev = "cuda"
    model = Student().to(dev); model.load_state_dict(torch.load(ck, map_location=dev)); model.eval()
    r = T.rows
    base = r["role"] == role
    chosen = {}
    if "tstar" in sets:
        chosen["tstar"] = np.flatnonzero(base & (r["ds"] == 1) & (r["type"] == 0))
    if "b1k" in sets:
        chosen["b1k"] = np.flatnonzero(base & (r["ds"] == 0) & (r["type"] == 0))
    if "ood" in sets:
        for t, nm in ((1, "sep"), (2, "alllit"), (3, "alldark")):
            chosen[f"ood_nat_{nm}"] = np.flatnonzero(base & (r["ds"] == 1) & (r["type"] == t))
        chosen["ood_rel_sep"] = np.flatnonzero(base & (r["ds"] == 0) & (r["type"] == 1))
    # declared subsamples of the two largest sets (identical for every arm; rows with a reference label always kept)
    caps = {"b1k": 6000, "ood_nat_alllit": 3000}
    srng = np.random.default_rng(20261113)
    for name, cap in caps.items():
        idx = chosen.get(name)
        if idx is None or len(idx) <= cap:
            continue
        has_ref = np.array([T.label_path(i).exists() or (T.label_path(i).parent.parent / "ood" / T.label_path(i).name).exists()
                            for i in idx]) if name.startswith("ood") else np.zeros(len(idx), bool)
        keep = np.flatnonzero(has_ref)
        rest = np.flatnonzero(~has_ref)
        pick = srng.choice(rest, max(cap - len(keep), 0), replace=False)
        chosen[name] = np.sort(idx[np.concatenate([keep, pick])])
    res = {}
    rng = np.random.default_rng(12345)
    for name, idx in chosen.items():
        M = {k: np.full(len(idx), np.nan) for k in ("nll", "nll_prior", "hpd_u", "brier16", "ent", "ig", "ce", "hq", "kl",
                                                   "supp_mass", "ce_prior", "ig_ref", "ent_prior")}
        rel_p = np.zeros(len(BINS) - 1); rel_h = np.zeros(len(BINS) - 1); rel_n = np.zeros(len(BINS) - 1)
        for a in range(0, len(idx), 8):
            pick = idx[a:a + 8]
            rr = r[pick]
            feats = torch.from_numpy(T.feats(pick)).to(dev)
            origin = torch.from_numpy(np.stack([rr["ox"], rr["oy"], rr["oz"]], 1).astype(np.int64)).to(dev)
            side = torch.from_numpy(rr["Lp"].astype(np.int64)).to(dev)
            with torch.no_grad():
                with torch.autocast("cuda", torch.bfloat16):
                    out = model(feats, origin, side)
                lm = full_logmass(model, out, origin, side)            # (b, 128^3)
                lu = uniform_logmass(origin, side)
                p = lm.exp()
                tc = torch.tensor([T.source_cell(i) for i in pick], device=dev)
                ar = torch.arange(len(pick), device=dev)
                lpt = lm[ar, tc]
                gt = (lm > lpt[:, None]); eq = (lm == lpt[:, None])
                V = torch.from_numpy(rng.random(len(pick))).to(dev).float()
                hpd = (p * gt).sum(1) + V * (p * eq).sum(1)
                plogp = torch.where(p > 0, p * lm, torch.zeros_like(p))
                ent = -plogp.sum(1)
                ig = (plogp - torch.where(p > 0, p * lu, torch.zeros_like(p))).sum(1)
                pu = lu.exp(); ent_u = -torch.where(pu > 0, pu * lu, torch.zeros_like(pu)).sum(1)
                Pc = p.view(-1, 16, 8, 16, 8, 16, 8).sum((2, 4, 6)).reshape(len(pick), -1)
                ix, iy, iz = tc // (NFINE * NFINE), (tc // NFINE) % NFINE, tc % NFINE
                cc = (ix // 8) * 256 + (iy // 8) * 16 + iz // 8
                e = torch.zeros_like(Pc); e[ar, cc] = 1
                brier = ((Pc - e) ** 2).sum(1)
                Pcn = Pc.cpu().numpy().ravel(); en = e.cpu().numpy().ravel()
                bi = np.clip(np.searchsorted(BINS, Pcn, side="right") - 1, 0, len(BINS) - 2)
                rel_p += np.bincount(bi, weights=Pcn, minlength=len(BINS) - 1)
                rel_h += np.bincount(bi, weights=en, minlength=len(BINS) - 1)
                rel_n += np.bincount(bi, minlength=len(BINS) - 1)
            sl = slice(a, a + len(pick))
            M["nll"][sl] = (-lpt).cpu().numpy(); M["nll_prior"][sl] = (-lu[ar, tc]).cpu().numpy()
            M["hpd_u"][sl] = hpd.cpu().numpy(); M["brier16"][sl] = brier.cpu().numpy()
            M["ent"][sl] = ent.cpu().numpy(); M["ig"][sl] = ig.cpu().numpy(); M["ent_prior"][sl] = ent_u.cpu().numpy()
            for b, i in enumerate(pick):
                cells, q, st = T.label(i)
                if cells is None:
                    continue
                ct = torch.from_numpy(cells).to(dev); qt = torch.from_numpy(q).to(dev).double()
                lq = lm[b, ct].double(); lqu = lu[b, ct].double()
                j = a + b
                M["ce"][j] = float(-(qt * lq).sum()); M["hq"][j] = float(-(qt * torch.log(qt)).sum())
                M["kl"][j] = M["ce"][j] - M["hq"][j]; M["supp_mass"][j] = float(lq.exp().sum())
                M["ce_prior"][j] = float(-(qt * lqu).sum()); M["ig_ref"][j] = M["ce_prior"][j] - M["hq"][j]
        rows = r[idx]
        res[name] = dict(idx=idx, scene=rows["scene"], Lp=rows["Lp"], family=rows["family"], k=rows["k"], m=rows["m"],
                         type=rows["type"], w_b1k=T.w["b1k"][idx], w_nat=T.w["nat"][idx],
                         litfrac=rows["nlit"] / np.maximum(rows["nlit"] + rows["ndark"], 1),
                         solidfrac=rows["nsol"] / (rows["Lp"].astype(float) ** 3), rel_p=rel_p, rel_h=rel_h, rel_n=rel_n, **M)
        print(json.dumps(dict(set=name, n=int(len(idx)), nll=float(np.nanmean(M["nll"])) if len(idx) else None,
                              nll_prior=float(np.nanmean(M["nll_prior"])) if len(idx) else None,
                              n_labels=int(np.sum(~np.isnan(M["ce"]))))), flush=True)
    flat = {f"{s}/{k}": v for s, d in res.items() for k, v in d.items()}
    np.savez_compressed(outp, **flat)


if __name__ == "__main__":
    main()
