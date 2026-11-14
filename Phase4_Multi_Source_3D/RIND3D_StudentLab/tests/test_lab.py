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


def test_triton_plane_factor_matches_torch():
    if not torch.cuda.is_available():
        return
    from slab.plane_triton import plane_factor_fused
    torch.manual_seed(2)
    B, K = 2, 3; dev = "cuda"
    mu = (torch.rand(B, K, 3, device=dev) * 1024).requires_grad_(); n0 = torch.randn(B, K, 3, device=dev)
    n = torch.nn.functional.normalize(n0, dim=-1).detach().requires_grad_()
    s0 = (torch.rand(B, K, device=dev) + 0.2).requires_grad_(); sth = (torch.rand(B, K, device=dev) * 1e-2).requires_grad_()
    e = (torch.randn(B, K, device=dev) * 3).requires_grad_()
    mask = torch.tensor([[True, False, True], [True, True, True]], device=dev)
    g = torch.randn(B, 4096, 512, device=dev)
    ref = plane_factor(mu, n, s0, sth, e, mask, checkpoint=False)
    gr = torch.autograd.grad((ref * g).sum(), (mu, n, s0, sth, e))
    out = plane_factor_fused(mu, n, s0, sth, e, mask)
    go = torch.autograd.grad((out * g).sum(), (mu, n, s0, sth, e))
    assert torch.allclose(out, ref, atol=1e-3), float((out - ref).abs().max())
    for a, b in zip(go, gr):
        assert torch.allclose(a, b, rtol=2e-3, atol=1e-2 * float(b.abs().max()) + 1e-6), (a, b)


def test_triton_two_hypothesis_factor_matches_torch():
    if not torch.cuda.is_available():
        return
    from slab.plane_triton import plane_factor2_fused, plane_factor2_torch
    torch.manual_seed(3)
    B, K = 2, 2; dev = "cuda"
    mu = (torch.rand(B, K, 3, device=dev) * 1024).requires_grad_()
    n = torch.nn.functional.normalize(torch.randn(B, K, 3, device=dev), dim=-1).detach().requires_grad_()
    s0 = (torch.rand(B, K, device=dev) + 0.2).requires_grad_(); sth = (torch.rand(B, K, device=dev) * 1e-2).requires_grad_()
    e = (torch.randn(B, K, device=dev) * 2).requires_grad_(); pi = torch.randn(B, K, device=dev).requires_grad_()
    lr = (torch.randn(B, K, device=dev) - 3).requires_grad_()
    mask = torch.tensor([[True, False], [True, True]], device=dev)
    g = torch.randn(B, 4096, 512, device=dev)
    args = (mu, n, s0, sth, e, pi, lr)
    ref = plane_factor2_torch(*args, mask)
    gr = torch.autograd.grad((ref * g).sum(), args)
    out = plane_factor2_fused(*args, mask)
    go = torch.autograd.grad((out * g).sum(), args)
    assert torch.allclose(out, ref, atol=2e-3), float((out - ref).abs().max())
    for a, b in zip(go, gr):
        assert torch.allclose(a, b, rtol=3e-3, atol=1e-2 * float(b.abs().max()) + 1e-6), (a, b)


def test_triton_quadrature_factor_matches_torch():
    """Sub-cell quadrature kernel (values and gradients) vs explicit sub-points; the reference runs on the CPU."""
    if not torch.cuda.is_available():
        return
    from slab.plane_triton import plane_factor_q_fused, plane_factor_q_torch, plane_factor_fused
    torch.manual_seed(3)
    B, K, M = 1, 3, 2
    mu = torch.rand(B, K, 3) * 1024; n = torch.nn.functional.normalize(torch.randn(B, K, 3), dim=-1)
    s0 = torch.rand(B, K) + 0.2; sth = torch.rand(B, K) * 1e-2; e = torch.randn(B, K) * 3
    mask = torch.tensor([[True, False, True]])
    g = torch.randn(B, 4096, 512)
    leaves = [t.clone().requires_grad_() for t in (mu, n, s0, sth, e)]
    ref = plane_factor_q_torch(*leaves, mask, M=M)
    gr = torch.autograd.grad((ref * g).sum(), leaves)
    cl = [t.cuda().requires_grad_() for t in (mu, n, s0, sth, e)]
    out = plane_factor_q_fused(*cl, mask.cuda(), M=M)
    go = torch.autograd.grad((out * g.cuda()).sum(), cl)
    assert torch.allclose(out.cpu(), ref, atol=2e-3), float((out.cpu() - ref).abs().max())
    for a, b in zip(go, gr):
        assert torch.allclose(a.cpu(), b, rtol=3e-3, atol=1e-2 * float(b.abs().max()) + 1e-6), (a, b)
    # M = 1 is the cell-centre factor
    o1 = plane_factor_q_fused(*[t.cuda() for t in (mu, n, s0, sth, e)], mask.cuda(), M=1)
    o0 = plane_factor_fused(*[t.cuda() for t in (mu, n, s0, sth, e)], mask.cuda())
    assert torch.allclose(o1, o0, atol=1e-4)
