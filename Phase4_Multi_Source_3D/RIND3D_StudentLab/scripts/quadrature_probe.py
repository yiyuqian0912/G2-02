"""Non-learned probe of the cell-integral approximation (X only; the true source is used only to score).

P0 evaluates each plane factor once at the cell centre and accounts for the 8-voxel cell by inflating every plane's
variance by 8^2/12. For a point-determined window (>= 3 planes meeting at small angles, since the window subtends only
~0.1 rad at the source) this inflation is applied to every plane independently and spreads the product along the
ill-conditioned depth direction. Probe: integrate the product over the cell by m^3 midpoint sub-points with the
residual sub-cell variance (8/m)^2/12, i.e. log p_cell = logmeanexp_j sum_k phi_k(x_j). m = 1 reproduces P0.

usage: quadrature_probe.py [--ms 1,2,4,2s] [--split dev] [--batch 8] [--out name]   ('2s': sparse M=2; '2sa': sparse + amplitude, as in arm F5; '1a': amplitude only)
Writes $RIND3D_LAB_OUT/quadrature_<out>.npz (per-row IG for each m) and prints side-balanced IG by stratum."""
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slab.data import LabTable, collate, to_dev           # noqa: E402
from slab.density import cell_centres, legal_logv, NC, NF  # noqa: E402
from slab.paths import OUT, facets_file                     # noqa: E402
from slab.analyze import boot, _stat                        # noqa: E402
from slab.plane_triton import plane_factor_q_fused, plane_factor_qs_fused   # noqa: E402

CELL = 8.0


def factor_quadrature(B, m, min_inliers=100, eps=1e-3, s0=0.5, kappa=1.0):
    """(B, 4096, 512) log of the cell-averaged product of P0's robust plane factors, m^3 midpoint sub-points (fused
    kernel slab.plane_triton.plane_factor_q_fused; tested against explicit sub-points in tests/test_lab.py)."""
    Fc = B["facets"].float()
    mask = (Fc[..., 6] >= min_inliers) & (torch.arange(Fc.shape[1], device=Fc.device)[None] < B["nf"][:, None])
    sth = kappa * Fc[..., 7].clamp_min(0.2) / (Fc[..., 8].clamp_min(1.0) * Fc[..., 6].clamp_min(1).sqrt())
    e = torch.full_like(sth, math.log(eps / (1 - eps)))
    if isinstance(m, str):              # '2s': sparse quadrature; suffix 'a': amplitude-consistent (the F5 kernel: '2sa')
        M = int(m.rstrip("sa"))
        return plane_factor_qs_fused(Fc[..., :3], Fc[..., 3:6], torch.full_like(e, s0), sth, e, mask, M=M, amp="a" in m)
    return plane_factor_q_fused(Fc[..., :3], Fc[..., 3:6], torch.full_like(e, s0), sth, e, mask, M=m)


@torch.no_grad()
def main():
    opt = lambda n, d: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d
    ms = [x if not x.isdigit() else int(x) for x in opt("--ms", "1,2,4").split(",")]
    split, bs, name = opt("--split", "dev"), int(opt("--batch", "8")), opt("--out", "v1")
    assert split == "dev", "exploratory probe: dev only"
    T = LabTable(facets=facets_file("v1"))
    idx = T.eval_set("tstar", split)
    dev = torch.device("cuda")
    res = {m: np.zeros(len(idx)) for m in ms}
    t0 = time.perf_counter()
    for i in range(0, len(idx), bs):
        pick = idx[i:i + bs]
        B = to_dev(collate(T, pick, "onehot"), dev)
        lv = legal_logv(B["origin"], B["side"])
        ar = torch.arange(len(pick), device=dev)
        lu = lv - torch.logsumexp(lv.flatten(1), 1)[:, None, None]
        for m in ms:
            z = factor_quadrature(B, m) + lv
            lp = z - torch.logsumexp(z.flatten(1), 1)[:, None, None]
            res[m][i:i + len(pick)] = (lp[ar, B["sc"], B["sf"]] - lu[ar, B["sc"], B["sf"]]).cpu().numpy()
        if i % (bs * 50) == 0:
            print(json.dumps(dict(done=i + len(pick), wall=round(time.perf_counter() - t0))), flush=True)
    r = T.rows[idx]
    np.savez(OUT / f"quadrature_{name}.npz", idx=idx, scene=r["scene"], Lp=r["Lp"], **{f"ig_m{m}": res[m] for m in ms})
    summ = {}
    for m in ms:
        summ[m] = boot(res[m], r["scene"], r["Lp"])
        print(f"m={m}: dev T* IG {summ[m]['mean']:.3f} [{summ[m]['lo']:.3f}, {summ[m]['hi']:.3f}]")
        if m != ms[0]:
            d = boot(res[m] - res[ms[0]], r["scene"], r["Lp"])
            print(f"   paired vs m={ms[0]}: {d['mean']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]")


if __name__ == "__main__":
    main()
