"""Lab tests: exact normalisation, plane factor, and the facet law on an analytic synthetic window.
Run: .venv/bin/python -m pytest -q tests/test_lab.py"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slab.density import legal_logv, log_normalise, plane_factor, cell_centres, SIGMA_CELL2   # noqa: E402
from slab.facets import facets_of, interface_points                                          # noqa: E402


def test_flat_normalisation_and_window_exclusion():
    torch.manual_seed(0)
    origin = torch.tensor([[0, 0, 0], [500, 37, 900]]); side = torch.tensor([64, 64])
    lv = legal_logv(origin, side)
    lp = log_normalise(torch.randn(2, 4096, 512), lv)
    assert torch.allclose(lp.exp().flatten(1).sum(1).double(), torch.ones(2, dtype=torch.float64), atol=1e-5)
    # cells fully inside the box (8-voxel cells, box [0, 64)^3: cells 0..7 per axis) have zero mass
    assert torch.isinf(lp[0, 0, 0]) and lp[0, 0, 0] < 0
    # uniform logits = uniform-legal prior: equal mass per unit legal volume
    lu = log_normalise(torch.zeros(1, 4096, 512), lv[:1]).exp()
    tot_legal = 1024 ** 3 - 64 ** 3
    assert abs(float(lu.max()) - 512.0 / tot_legal) < 1e-9


def test_plane_factor_matches_direct():
    torch.manual_seed(1)
    B, K = 2, 3
    mu = torch.rand(B, K, 3) * 1024; n = torch.nn.functional.normalize(torch.randn(B, K, 3), dim=-1)
    s0 = torch.rand(B, K) + 0.2; sth = torch.rand(B, K) * 1e-2; e = torch.randn(B, K) * 3
    mask = torch.tensor([[True, False, True], [True, True, True]])
    pf = plane_factor(mu, n, s0, sth, e, mask, checkpoint=False)
    C = cell_centres("cpu").double()
    for b in range(B):
        ref = torch.zeros(4096, 512, dtype=torch.float64)
        for k in range(K):
            if not mask[b, k]:
                continue
            D = C - mu[b, k].double(); d = D @ n[b, k].double(); r2 = (D * D).sum(-1)
            s2 = SIGMA_CELL2 + s0[b, k].double() ** 2 + r2 * sth[b, k].double() ** 2
            eps = torch.sigmoid(e[b, k].double())
            ref += torch.log(eps + (1 - eps) * torch.exp(-d * d / (2 * s2)))
        assert torch.allclose(pf[b].double(), ref, atol=2e-3), float((pf[b].double() - ref).abs().max())


def _segment_hits_box(s, x, lo, hi):
    """closed segment s -> x (rows of x) meets the closed axis-aligned box [lo, hi] (slab method)."""
    d = x - s; t0 = np.zeros(len(x)); t1 = np.ones(len(x))
    for a in range(3):
        with np.errstate(divide="ignore", invalid="ignore"):
            ta = (lo[a] - s[a]) / d[:, a]; tb = (hi[a] - s[a]) / d[:, a]
        tmin = np.minimum(ta, tb); tmax = np.maximum(ta, tb)
        par = d[:, a] == 0
        inside = (s[a] >= lo[a]) & (s[a] <= hi[a])
        tmin = np.where(par, np.where(inside, -np.inf, np.inf), tmin); tmax = np.where(par, np.where(inside, np.inf, -np.inf), tmax)
        t0 = np.maximum(t0, tmin); t1 = np.minimum(t1, tmax)
    return t0 <= t1


def test_facet_law_synthetic_box_occluder():
    """A box occluder outside a 64^3 window casts a shadow into it; every extracted facet plane contains the source.
    Oblique shadow planes (generic position). Near-axis-aligned planes are the worst case of the voxel staircase
    (about 1 voxel at 330 voxels for slope 0.025); the learned facet widths absorb that."""
    L = 64; s = np.array([-300.0, -140.0, 210.0])                 # source, window voxel units (window = [0, 64)^3)
    lo, hi = np.array([-150.0, -60.0, 100.0]), np.array([-120.0, -20.0, 135.0])  # hidden occluder (outside the window)
    g = np.arange(L) + 0.5
    Z, Y, X = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    lit = (~_segment_hits_box(s, pts, lo, hi)).reshape(L, L, L)
    sol = np.zeros((L, L, L), bool)
    assert lit.any() and (~lit).any()
    F, n = facets_of(sol, lit)
    assert len(F) >= 2 and n > 100
    # the two largest facets (generic orientations) contain the source to 0.1 voxel at ~350 voxels; smaller or
    # axis-parallel facets can be off by 1-2 voxels (staircase quantisation): measured on real data in validate_facets.py
    for f in sorted(F, key=lambda f: -f[6])[:2]:
        assert abs(float((s - f[:3]) @ f[3:6])) < 0.1, f
