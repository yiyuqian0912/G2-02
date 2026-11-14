"""Fused Triton kernels for the robust plane factor of slab.density (forward and analytic backward).

phi(s) = sum_k logaddexp(LE_k, L1_k - u_k(s)),  u = d^2 / (2 sigma^2),  d = n.(C - mu),  r2 = |C - mu|^2,
sigma^2 = c0 + S0 + r2 * STH.  Parameters per (b, k): P = [mu(3), n(3), S0, STH, LE, L1] (float32).
Cell centres C are generated from the (c, f) index (no memory reads). The backward reduces, per block of cells, the
gradients of the 10 parameters; partial sums (B, n_blocks, K, 10) are summed in torch (deterministic)."""
import math
import torch
import triton
import triton.language as tl

from .density import SIGMA_CELL2, NC, NF

BLOCK = 1024
NCELL = NC ** 3 * NF ** 3


@triton.jit
def _centre(j):
    c = j // 512; f = j % 512
    cx = c // 256; cy = (c // 16) % 16; cz = c % 16
    fx = f // 64; fy = (f // 8) % 8; fz = f % 8
    x = ((cx * 8 + fx).to(tl.float32) + 0.5) * 8.0
    y = ((cy * 8 + fy).to(tl.float32) + 0.5) * 8.0
    z = ((cz * 8 + fz).to(tl.float32) + 0.5) * 8.0
    return x, y, z


@triton.jit
def _fwd(P, OUT, K: tl.constexpr, NCELL: tl.constexpr, C0: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0); blk = tl.program_id(1)
    j = blk * BLOCK + tl.arange(0, BLOCK)
    x, y, z = _centre(j)
    acc = tl.zeros([BLOCK], tl.float32)
    for k in tl.static_range(K):
        base = P + (b * K + k) * 10
        mx = tl.load(base + 0); my = tl.load(base + 1); mz = tl.load(base + 2)
        nx = tl.load(base + 3); ny = tl.load(base + 4); nz = tl.load(base + 5)
        s0 = tl.load(base + 6); sth = tl.load(base + 7); le = tl.load(base + 8); l1 = tl.load(base + 9)
        dx = x - mx; dy = y - my; dz = z - mz
        d = nx * dx + ny * dy + nz * dz
        r2 = dx * dx + dy * dy + dz * dz
        s2 = C0 + s0 + r2 * sth
        a = l1 - d * d / (2.0 * s2)
        m = tl.maximum(le, a)
        acc += m + tl.log(1.0 + tl.exp(-tl.abs(le - a)))
    tl.store(OUT + b * NCELL + j, acc)


@triton.jit
def _bwd(P, G, PART, K: tl.constexpr, NCELL: tl.constexpr, C0: tl.constexpr, BLOCK: tl.constexpr, NBLK: tl.constexpr):
    b = tl.program_id(0); blk = tl.program_id(1)
    j = blk * BLOCK + tl.arange(0, BLOCK)
    x, y, z = _centre(j)
    g = tl.load(G + b * NCELL + j)
    for k in tl.static_range(K):
        base = P + (b * K + k) * 10
        mx = tl.load(base + 0); my = tl.load(base + 1); mz = tl.load(base + 2)
        nx = tl.load(base + 3); ny = tl.load(base + 4); nz = tl.load(base + 5)
        s0 = tl.load(base + 6); sth = tl.load(base + 7); le = tl.load(base + 8); l1 = tl.load(base + 9)
        dx = x - mx; dy = y - my; dz = z - mz
        d = nx * dx + ny * dy + nz * dz
        r2 = dx * dx + dy * dy + dz * dz
        s2 = C0 + s0 + r2 * sth
        u = d * d / (2.0 * s2)
        a = l1 - u
        w = 1.0 / (1.0 + tl.exp(le - a))                  # d phi / d a
        gw = g * w
        ds = d / s2; us = u / s2
        o = PART + ((b * NBLK + blk) * K + k) * 10
        # mu_j: w (d/s2) n_j - 2 w (u/s2) STH D_j
        tl.store(o + 0, tl.sum(gw * (ds * nx - 2.0 * us * sth * dx), 0))
        tl.store(o + 1, tl.sum(gw * (ds * ny - 2.0 * us * sth * dy), 0))
        tl.store(o + 2, tl.sum(gw * (ds * nz - 2.0 * us * sth * dz), 0))
        # n_j: -w (d/s2) D_j
        tl.store(o + 3, tl.sum(-gw * ds * dx, 0))
        tl.store(o + 4, tl.sum(-gw * ds * dy, 0))
        tl.store(o + 5, tl.sum(-gw * ds * dz, 0))
        tl.store(o + 6, tl.sum(gw * us, 0))               # S0
        tl.store(o + 7, tl.sum(gw * us * r2, 0))          # STH
        tl.store(o + 8, tl.sum(g * (1.0 - w), 0))         # LE
        tl.store(o + 9, tl.sum(gw, 0))                    # L1


class PlaneFactorFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, P):
        P = P.contiguous().float(); B, K, _ = P.shape
        out = torch.empty(B, NCELL, device=P.device, dtype=torch.float32)
        _fwd[(B, NCELL // BLOCK)](P, out, K=K, NCELL=NCELL, C0=SIGMA_CELL2, BLOCK=BLOCK)
        ctx.save_for_backward(P)
        return out.view(B, NC ** 3, NF ** 3)

    @staticmethod
    def backward(ctx, g):
        (P,) = ctx.saved_tensors; B, K, _ = P.shape; nblk = NCELL // BLOCK
        part = torch.empty(B, nblk, K, 10, device=P.device, dtype=torch.float32)
        _bwd[(B, nblk)](P, g.contiguous().float(), part, K=K, NCELL=NCELL, C0=SIGMA_CELL2, BLOCK=BLOCK, NBLK=nblk)
        return part.sum(1)


def plane_factor_fused(mu, n, s0, sth, eps_logit, mask):
    """Same contract as slab.density.plane_factor (fp32, any autocast state)."""
    with torch.autocast("cuda", enabled=False):
        le = torch.nn.functional.logsigmoid(eps_logit.float()); l1 = torch.nn.functional.logsigmoid(-eps_logit.float())
        le = torch.where(mask, le, torch.zeros_like(le)); l1 = torch.where(mask, l1, torch.full_like(l1, -1e4))
        P = torch.cat([mu.float(), n.float(), (s0.float() ** 2)[..., None], (sth.float() ** 2)[..., None], le[..., None], l1[..., None]], -1)
        return PlaneFactorFn.apply(P)


# ---------------------------------------------------------------- two-hypothesis facets (shadow plane | occluder face)
# phi(s) = logaddexp(LE, L1 + logaddexp(LPI - u, LQ + logsigmoid(d * TI))), d = n.(C - mu) with n oriented to the lit side.
# Parameters per (b, k): [mu(3), n(3), S0, STH, LE, L1, LPI, LQ, TI] (13); gradients for the first 12 (TI fixed).
@triton.jit
def _logsig(x):
    return tl.minimum(x, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(x)))


@triton.jit
def _lae(a, b):
    m = tl.maximum(a, b)
    return m + tl.log(1.0 + tl.exp(-tl.abs(a - b)))


@triton.jit
def _fwd2(P, OUT, K: tl.constexpr, NCELL: tl.constexpr, C0: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0); blk = tl.program_id(1)
    j = blk * BLOCK + tl.arange(0, BLOCK)
    x, y, z = _centre(j)
    acc = tl.zeros([BLOCK], tl.float32)
    for k in tl.static_range(K):
        base = P + (b * K + k) * 13
        mx = tl.load(base + 0); my = tl.load(base + 1); mz = tl.load(base + 2)
        nx = tl.load(base + 3); ny = tl.load(base + 4); nz = tl.load(base + 5)
        s0 = tl.load(base + 6); sth = tl.load(base + 7); le = tl.load(base + 8); l1 = tl.load(base + 9)
        lpi = tl.load(base + 10); lq = tl.load(base + 11); ti = tl.load(base + 12)
        dx = x - mx; dy = y - my; dz = z - mz
        d = nx * dx + ny * dy + nz * dz
        r2 = dx * dx + dy * dy + dz * dz
        s2 = C0 + s0 + r2 * sth
        inner = _lae(lpi - d * d / (2.0 * s2), lq + _logsig(d * ti))
        acc += _lae(le, l1 + inner)
    tl.store(OUT + b * NCELL + j, acc)


@triton.jit
def _bwd2(P, G, PART, K: tl.constexpr, NCELL: tl.constexpr, C0: tl.constexpr, BLOCK: tl.constexpr, NBLK: tl.constexpr):
    b = tl.program_id(0); blk = tl.program_id(1)
    j = blk * BLOCK + tl.arange(0, BLOCK)
    x, y, z = _centre(j)
    g = tl.load(G + b * NCELL + j)
    for k in tl.static_range(K):
        base = P + (b * K + k) * 13
        mx = tl.load(base + 0); my = tl.load(base + 1); mz = tl.load(base + 2)
        nx = tl.load(base + 3); ny = tl.load(base + 4); nz = tl.load(base + 5)
        s0 = tl.load(base + 6); sth = tl.load(base + 7); le = tl.load(base + 8); l1 = tl.load(base + 9)
        lpi = tl.load(base + 10); lq = tl.load(base + 11); ti = tl.load(base + 12)
        dx = x - mx; dy = y - my; dz = z - mz
        d = nx * dx + ny * dy + nz * dz
        r2 = dx * dx + dy * dy + dz * dz
        s2 = C0 + s0 + r2 * sth
        u = d * d / (2.0 * s2)
        a1 = lpi - u; a2 = lq + _logsig(d * ti)
        inner = _lae(a1, a2)
        w = 1.0 / (1.0 + tl.exp(le - l1 - inner))            # non-outlier responsibility
        r1 = 1.0 / (1.0 + tl.exp(a2 - a1))                   # shadow-plane share
        sm = 1.0 / (1.0 + tl.exp(d * ti))                    # sigmoid(-d ti)
        gd = g * w * (-r1 * d / s2 + (1.0 - r1) * ti * sm)   # d phi / d d
        gs = g * w * r1 * u / s2                             # d phi / d s2
        o = PART + ((b * NBLK + blk) * K + k) * 12
        tl.store(o + 0, tl.sum(-gd * nx - 2.0 * gs * sth * dx, 0))
        tl.store(o + 1, tl.sum(-gd * ny - 2.0 * gs * sth * dy, 0))
        tl.store(o + 2, tl.sum(-gd * nz - 2.0 * gs * sth * dz, 0))
        tl.store(o + 3, tl.sum(gd * dx, 0))
        tl.store(o + 4, tl.sum(gd * dy, 0))
        tl.store(o + 5, tl.sum(gd * dz, 0))
        tl.store(o + 6, tl.sum(gs, 0))
        tl.store(o + 7, tl.sum(gs * r2, 0))
        tl.store(o + 8, tl.sum(g * (1.0 - w), 0))
        tl.store(o + 9, tl.sum(g * w, 0))
        tl.store(o + 10, tl.sum(g * w * r1, 0))
        tl.store(o + 11, tl.sum(g * w * (1.0 - r1), 0))


class PlaneFactor2Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, P):
        P = P.contiguous().float(); B, K, _ = P.shape
        out = torch.empty(B, NCELL, device=P.device, dtype=torch.float32)
        _fwd2[(B, NCELL // BLOCK)](P, out, K=K, NCELL=NCELL, C0=SIGMA_CELL2, BLOCK=BLOCK)
        ctx.save_for_backward(P)
        return out.view(B, NC ** 3, NF ** 3)

    @staticmethod
    def backward(ctx, g):
        (P,) = ctx.saved_tensors; B, K, _ = P.shape; nblk = NCELL // BLOCK
        part = torch.empty(B, nblk, K, 12, device=P.device, dtype=torch.float32)
        _bwd2[(B, nblk)](P, g.contiguous().float(), part, K=K, NCELL=NCELL, C0=SIGMA_CELL2, BLOCK=BLOCK, NBLK=nblk)
        gp = part.sum(1)
        return torch.cat([gp, torch.zeros_like(gp[..., :1])], -1)


def two_hyp_params(mu, n_lit, s0, sth, eps_logit, pi_logit, log_rho, mask, tau=4.0):
    """Parameter rows for PlaneFactor2Fn. n_lit: unit normals oriented to the lit side. Masked slots contribute 0."""
    le = torch.nn.functional.logsigmoid(eps_logit); l1 = torch.nn.functional.logsigmoid(-eps_logit)
    lpi = torch.nn.functional.logsigmoid(pi_logit); lq = torch.nn.functional.logsigmoid(-pi_logit) + log_rho
    le = torch.where(mask, le, torch.zeros_like(le)); l1 = torch.where(mask, l1, torch.full_like(l1, -1e4))
    ti = torch.full_like(le, 1.0 / tau)
    return torch.stack([mu[..., 0], mu[..., 1], mu[..., 2], n_lit[..., 0], n_lit[..., 1], n_lit[..., 2], s0 ** 2, sth ** 2,
                        le, l1, lpi, lq, ti], -1).float()


def plane_factor2_fused(*a, **kw):
    with torch.autocast("cuda", enabled=False):
        return PlaneFactor2Fn.apply(two_hyp_params(*[t.float() if torch.is_tensor(t) and t.dtype != torch.bool else t for t in a], **kw))


def plane_factor2_torch(mu, n_lit, s0, sth, eps_logit, pi_logit, log_rho, mask, tau=4.0):
    """Reference implementation (tests)."""
    from .density import cell_centres
    P = two_hyp_params(mu, n_lit, s0, sth, eps_logit, pi_logit, log_rho, mask, tau)
    C = cell_centres(P.device).view(-1, 3)
    out = 0
    for k in range(P.shape[1]):
        p = P[:, k]
        D = C[None] - p[:, None, :3]; d = (D * p[:, None, 3:6]).sum(-1); r2 = (D * D).sum(-1)
        s2 = SIGMA_CELL2 + p[:, 6:7] + r2 * p[:, 7:8]
        inner = torch.logaddexp(p[:, 10:11] - d * d / (2 * s2), p[:, 11:12] + torch.nn.functional.logsigmoid(d * p[:, 12:13]))
        out = out + torch.logaddexp(p[:, 8:9], p[:, 9:10] + inner)
    return out.view(-1, NC ** 3, NF ** 3)


# ---------------------------------------------------------------- sub-cell quadrature of the single-hypothesis factor
# The cell-centre factor inflates every plane's variance by the cell term 8^2/12 independently, which over-spreads the
# product of several nearly coplanar-normal planes along the ill-conditioned (depth) direction. The quadrature version
# integrates the product over the cell with M^3 midpoint sub-points and the residual sub-cell variance (8/M)^2/12:
#   phi_cell = logmeanexp_j sum_k phi_k(x_j).
# Backward: d phi_cell / d S_j = exp(S_j - phi_cell - log M^3) (forward output saved); partial sums per (block, j, k).
@triton.jit
def _phi(x, y, z, base):
    mx = tl.load(base + 0); my = tl.load(base + 1); mz = tl.load(base + 2)
    nx = tl.load(base + 3); ny = tl.load(base + 4); nz = tl.load(base + 5)
    s0 = tl.load(base + 6); sth = tl.load(base + 7); le = tl.load(base + 8); l1 = tl.load(base + 9)
    return mx, my, mz, nx, ny, nz, s0, sth, le, l1


@triton.jit
def _fwdq(P, OUT, K: tl.constexpr, NCELL: tl.constexpr, C0: tl.constexpr, BLOCK: tl.constexpr, M: tl.constexpr):
    b = tl.program_id(0); blk = tl.program_id(1)
    j = blk * BLOCK + tl.arange(0, BLOCK)
    x0, y0, z0 = _centre(j)
    lse = tl.full([BLOCK], -1e30, tl.float32)
    for qa in tl.static_range(M):
        for qb in tl.static_range(M):
            for qc in tl.static_range(M):
                x = x0 + ((qa + 0.5) / M - 0.5) * 8.0
                y = y0 + ((qb + 0.5) / M - 0.5) * 8.0
                z = z0 + ((qc + 0.5) / M - 0.5) * 8.0
                S = tl.zeros([BLOCK], tl.float32)
                for k in tl.static_range(K):
                    mx, my, mz, nx, ny, nz, s0, sth, le, l1 = _phi(x, y, z, P + (b * K + k) * 10)
                    dx = x - mx; dy = y - my; dz = z - mz
                    d = nx * dx + ny * dy + nz * dz
                    s2 = C0 + s0 + (dx * dx + dy * dy + dz * dz) * sth
                    S += _lae(le, l1 - d * d / (2.0 * s2))
                lse = _lae(lse, S)
    tl.store(OUT + b * NCELL + j, lse - tl.log(M * M * M * 1.0))


@triton.jit
def _bwdq(P, G, OUTF, PART, K: tl.constexpr, NCELL: tl.constexpr, C0: tl.constexpr, BLOCK: tl.constexpr,
          NBLK: tl.constexpr, M: tl.constexpr):
    b = tl.program_id(0); blk = tl.program_id(1)
    j = blk * BLOCK + tl.arange(0, BLOCK)
    x0, y0, z0 = _centre(j)
    g = tl.load(G + b * NCELL + j)
    lse = tl.load(OUTF + b * NCELL + j) + tl.log(M * M * M * 1.0)
    for qa in tl.static_range(M):
        for qb in tl.static_range(M):
            for qc in tl.static_range(M):
                x = x0 + ((qa + 0.5) / M - 0.5) * 8.0
                y = y0 + ((qb + 0.5) / M - 0.5) * 8.0
                z = z0 + ((qc + 0.5) / M - 0.5) * 8.0
                S = tl.zeros([BLOCK], tl.float32)
                for k in tl.static_range(K):
                    mx, my, mz, nx, ny, nz, s0, sth, le, l1 = _phi(x, y, z, P + (b * K + k) * 10)
                    dx = x - mx; dy = y - my; dz = z - mz
                    d = nx * dx + ny * dy + nz * dz
                    s2 = C0 + s0 + (dx * dx + dy * dy + dz * dz) * sth
                    S += _lae(le, l1 - d * d / (2.0 * s2))
                gq = g * tl.exp(S - lse)
                q = (qa * M + qb) * M + qc
                for k in tl.static_range(K):
                    mx, my, mz, nx, ny, nz, s0, sth, le, l1 = _phi(x, y, z, P + (b * K + k) * 10)
                    dx = x - mx; dy = y - my; dz = z - mz
                    d = nx * dx + ny * dy + nz * dz
                    r2 = dx * dx + dy * dy + dz * dz
                    s2 = C0 + s0 + r2 * sth
                    u = d * d / (2.0 * s2)
                    w = 1.0 / (1.0 + tl.exp(le - (l1 - u)))
                    gw = gq * w
                    ds = d / s2; us = u / s2
                    o = PART + (((b * NBLK + blk) * (M * M * M) + q) * K + k) * 10
                    tl.store(o + 0, tl.sum(gw * (ds * nx - 2.0 * us * sth * dx), 0))
                    tl.store(o + 1, tl.sum(gw * (ds * ny - 2.0 * us * sth * dy), 0))
                    tl.store(o + 2, tl.sum(gw * (ds * nz - 2.0 * us * sth * dz), 0))
                    tl.store(o + 3, tl.sum(-gw * ds * dx, 0))
                    tl.store(o + 4, tl.sum(-gw * ds * dy, 0))
                    tl.store(o + 5, tl.sum(-gw * ds * dz, 0))
                    tl.store(o + 6, tl.sum(gw * us, 0))
                    tl.store(o + 7, tl.sum(gw * us * r2, 0))
                    tl.store(o + 8, tl.sum(gq * (1.0 - w), 0))
                    tl.store(o + 9, tl.sum(gw, 0))


class PlaneFactorQFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, P, M):
        P = P.contiguous().float(); B, K, _ = P.shape
        out = torch.empty(B, NCELL, device=P.device, dtype=torch.float32)
        c0 = (8.0 / M) ** 2 / 12.0
        _fwdq[(B, NCELL // BLOCK)](P, out, K=K, NCELL=NCELL, C0=c0, BLOCK=BLOCK, M=M)
        ctx.save_for_backward(P, out); ctx.M = M
        return out.view(B, NC ** 3, NF ** 3)

    @staticmethod
    def backward(ctx, g):
        P, out = ctx.saved_tensors; M = ctx.M; B, K, _ = P.shape; nblk = NCELL // BLOCK
        part = torch.empty(B, nblk, M ** 3, K, 10, device=P.device, dtype=torch.float32)
        _bwdq[(B, nblk)](P, g.contiguous().float(), out, part, K=K, NCELL=NCELL, C0=(8.0 / M) ** 2 / 12.0, BLOCK=BLOCK,
                         NBLK=nblk, M=M)
        return part.sum((1, 2)), None


def plane_factor_q_fused(mu, n, s0, sth, eps_logit, mask, M=2):
    """Sub-cell quadrature version of plane_factor_fused (M^3 midpoint sub-points per 8-voxel cell)."""
    with torch.autocast("cuda", enabled=False):
        le = torch.nn.functional.logsigmoid(eps_logit.float()); l1 = torch.nn.functional.logsigmoid(-eps_logit.float())
        le = torch.where(mask, le, torch.zeros_like(le)); l1 = torch.where(mask, l1, torch.full_like(l1, -1e4))
        P = torch.cat([mu.float(), n.float(), (s0.float() ** 2)[..., None], (sth.float() ** 2)[..., None], le[..., None], l1[..., None]], -1)
        return PlaneFactorQFn.apply(P, M)


def plane_factor_q_torch(mu, n, s0, sth, eps_logit, mask, M=2):
    """Reference implementation (tests): explicit sub-points, torch autograd."""
    from .density import cell_centres
    le = torch.nn.functional.logsigmoid(eps_logit); l1 = torch.nn.functional.logsigmoid(-eps_logit)
    C = cell_centres(mu.device).view(-1, 3); c0 = (8.0 / M) ** 2 / 12.0
    offs = (torch.arange(M, device=mu.device, dtype=torch.float32) + 0.5) / M * 8.0 - 4.0
    lse = None
    for a in offs:
        for b in offs:
            for c in offs:
                X = C + torch.stack([a, b, c])
                S = 0
                for k in range(mu.shape[1]):
                    D = X[None] - mu[:, k, None]; d = (D * n[:, k, None]).sum(-1); r2 = (D * D).sum(-1)
                    s2 = c0 + s0[:, k, None] ** 2 + r2 * sth[:, k, None] ** 2
                    phi = torch.logaddexp(le[:, k, None].expand_as(d), l1[:, k, None] - d * d / (2 * s2))
                    S = S + torch.where(mask[:, k, None], phi, torch.zeros_like(phi))
                lse = S if lse is None else torch.logaddexp(lse, S)
    return (lse - math.log(M ** 3)).view(-1, NC ** 3, NF ** 3)


# ---------------------------------------------------------------- sparse quadrature (used by F5)
# A cell uses the quadrature only if >= 2 planes are non-negligible at its centre (Gaussian part within e^-8 of the
# floor, evaluated with the cell-centre variance); elsewhere at most one plane differs from its floor and the
# cell-centre formula is kept. Blocks without any such cell skip the sub-point loop (most of the volume).
# AMP: both paths approximate the cell *average* of the same function f = eps + (1 - eps) exp(-d^2 / 2 v0),
# v0 = s0^2 + r^2 sth^2: a Gaussian smoothed by variance c has amplitude sqrt(v0 / (v0 + c)). Without it, the two paths
# give a single plane different mass relative to the floor (peak-normalised kernels of different widths), and the switch
# between them distorts lines (measured: -0.12 nats on two-plane windows at M = 2).
@triton.jit
def _term(x, y, z, base, C0X: tl.constexpr, AMP: tl.constexpr):
    mx, my, mz, nx, ny, nz, s0, sth, le, l1 = _phi(x, y, z, base)
    dx = x - mx; dy = y - my; dz = z - mz
    d = nx * dx + ny * dy + nz * dz
    r2 = dx * dx + dy * dy + dz * dz
    v0 = s0 + r2 * sth
    s2 = C0X + v0
    u = d * d / (2.0 * s2)
    a = l1 - u
    if AMP:
        a += 0.5 * tl.log(v0 / s2)
    # d a / d v0 (through s2 and the amplitude)
    av = u / s2
    if AMP:
        av += 0.5 * C0X / (v0 * s2)
    return a, le, d, s2, av, dx, dy, dz, nx, ny, nz, sth, r2


@triton.jit
def _store_grads(o, gq, a, le, d, s2, av, dx, dy, dz, nx, ny, nz, sth, r2):
    w = 1.0 / (1.0 + tl.exp(le - a))
    gw = gq * w
    ds = d / s2
    tl.store(o + 0, tl.sum(gw * (ds * nx - 2.0 * av * sth * dx), 0))
    tl.store(o + 1, tl.sum(gw * (ds * ny - 2.0 * av * sth * dy), 0))
    tl.store(o + 2, tl.sum(gw * (ds * nz - 2.0 * av * sth * dz), 0))
    tl.store(o + 3, tl.sum(-gw * ds * dx, 0))
    tl.store(o + 4, tl.sum(-gw * ds * dy, 0))
    tl.store(o + 5, tl.sum(-gw * ds * dz, 0))
    tl.store(o + 6, tl.sum(gw * av, 0))
    tl.store(o + 7, tl.sum(gw * av * r2, 0))
    tl.store(o + 8, tl.sum(gq * (1.0 - w), 0))
    tl.store(o + 9, tl.sum(gw, 0))


@triton.jit
def _subsum(x, y, z, P, b, K: tl.constexpr, C0: tl.constexpr, AMP: tl.constexpr, BLOCK: tl.constexpr):
    S = tl.zeros([BLOCK], tl.float32)
    for k in tl.static_range(K):
        a, le, d, s2, av, dx, dy, dz, nx, ny, nz, sth, r2 = _term(x, y, z, P + (b * K + k) * 10, C0, AMP)
        S += _lae(le, a)
    return S


@triton.jit
def _fwdqs(P, OUT, SEL, K: tl.constexpr, NCELL: tl.constexpr, C0C: tl.constexpr, C0: tl.constexpr, BLOCK: tl.constexpr,
           M: tl.constexpr, AMP: tl.constexpr):
    b = tl.program_id(0); blk = tl.program_id(1)
    j = blk * BLOCK + tl.arange(0, BLOCK)
    x0, y0, z0 = _centre(j)
    res = tl.zeros([BLOCK], tl.float32); cnt = tl.zeros([BLOCK], tl.int32)
    for k in tl.static_range(K):
        a, le, d, s2, av, dx, dy, dz, nx, ny, nz, sth, r2 = _term(x0, y0, z0, P + (b * K + k) * 10, C0C, AMP)
        res += _lae(le, a)
        cnt += (a >= le - 8.0).to(tl.int32)
    sel = cnt >= 2
    if tl.max(sel.to(tl.int32), 0) > 0:
        lse = tl.full([BLOCK], -1e30, tl.float32)
        for qa in tl.static_range(M):
            for qb in tl.static_range(M):
                for qc in tl.static_range(M):
                    S = _subsum(x0 + ((qa + 0.5) / M - 0.5) * 8.0, y0 + ((qb + 0.5) / M - 0.5) * 8.0,
                                z0 + ((qc + 0.5) / M - 0.5) * 8.0, P, b, K, C0, AMP, BLOCK)
                    lse = _lae(lse, S)
        res = tl.where(sel, lse - tl.log(M * M * M * 1.0), res)
    tl.store(OUT + b * NCELL + j, res)
    tl.store(SEL + b * NCELL + j, sel.to(tl.int8))


@triton.jit
def _bwdqs(P, G, OUTF, SEL, PART, K: tl.constexpr, NCELL: tl.constexpr, C0C: tl.constexpr, C0: tl.constexpr,
           BLOCK: tl.constexpr, NBLK: tl.constexpr, M: tl.constexpr, AMP: tl.constexpr):
    b = tl.program_id(0); blk = tl.program_id(1)
    j = blk * BLOCK + tl.arange(0, BLOCK)
    x0, y0, z0 = _centre(j)
    g = tl.load(G + b * NCELL + j)
    sel = tl.load(SEL + b * NCELL + j) > 0
    gc = tl.where(sel, 0.0, g)
    Q: tl.constexpr = M * M * M
    for k in tl.static_range(K):                                   # cell-centre path (slot Q)
        a, le, d, s2, av, dx, dy, dz, nx, ny, nz, sth, r2 = _term(x0, y0, z0, P + (b * K + k) * 10, C0C, AMP)
        _store_grads(PART + (((b * NBLK + blk) * (Q + 1) + Q) * K + k) * 10, gc, a, le, d, s2, av, dx, dy, dz,
                     nx, ny, nz, sth, r2)
    if tl.max(sel.to(tl.int32), 0) > 0:                             # quadrature path (slots 0..Q-1; zero-filled otherwise)
        lse = tl.where(sel, tl.load(OUTF + b * NCELL + j) + tl.log(Q * 1.0), 1e30)
        gs = tl.where(sel, g, 0.0)
        for qa in tl.static_range(M):
            for qb in tl.static_range(M):
                for qc in tl.static_range(M):
                    x = x0 + ((qa + 0.5) / M - 0.5) * 8.0
                    y = y0 + ((qb + 0.5) / M - 0.5) * 8.0
                    z = z0 + ((qc + 0.5) / M - 0.5) * 8.0
                    S = _subsum(x, y, z, P, b, K, C0, AMP, BLOCK)
                    gq = gs * tl.exp(tl.minimum(S - lse, 0.0))
                    q = (qa * M + qb) * M + qc
                    for k in tl.static_range(K):
                        a, le, d, s2, av, dx, dy, dz, nx, ny, nz, sth, r2 = _term(x, y, z, P + (b * K + k) * 10, C0, AMP)
                        _store_grads(PART + (((b * NBLK + blk) * (Q + 1) + q) * K + k) * 10, gq, a, le, d, s2, av,
                                     dx, dy, dz, nx, ny, nz, sth, r2)


class PlaneFactorQSFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, P, M, amp):
        P = P.contiguous().float(); B, K, _ = P.shape
        out = torch.empty(B, NCELL, device=P.device, dtype=torch.float32)
        sel = torch.empty(B, NCELL, device=P.device, dtype=torch.int8)
        _fwdqs[(B, NCELL // BLOCK)](P, out, sel, K=K, NCELL=NCELL, C0C=SIGMA_CELL2, C0=(8.0 / M) ** 2 / 12.0, BLOCK=BLOCK,
                                    M=M, AMP=amp)
        ctx.save_for_backward(P, out, sel); ctx.M = M; ctx.amp = amp
        return out.view(B, NC ** 3, NF ** 3)

    @staticmethod
    def backward(ctx, g):
        P, out, sel = ctx.saved_tensors; M = ctx.M; B, K, _ = P.shape; nblk = NCELL // BLOCK
        part = torch.zeros(B, nblk, M ** 3 + 1, K, 10, device=P.device, dtype=torch.float32)
        _bwdqs[(B, nblk)](P, g.contiguous().float(), out, sel, part, K=K, NCELL=NCELL, C0C=SIGMA_CELL2,
                          C0=(8.0 / M) ** 2 / 12.0, BLOCK=BLOCK, NBLK=nblk, M=M, AMP=ctx.amp)
        return part.sum((1, 2)), None, None


def plane_factor_qs_fused(mu, n, s0, sth, eps_logit, mask, M=2, amp=True):
    """Sparse quadrature: M^3 sub-points only in cells where >= 2 planes are non-negligible; cell-centre factor elsewhere.
    amp: amplitude-consistent cell average (see above); M = 1 with amp=True is the amplitude-corrected cell-centre factor."""
    with torch.autocast("cuda", enabled=False):
        le = torch.nn.functional.logsigmoid(eps_logit.float()); l1 = torch.nn.functional.logsigmoid(-eps_logit.float())
        le = torch.where(mask, le, torch.zeros_like(le)); l1 = torch.where(mask, l1, torch.full_like(l1, -1e4))
        P = torch.cat([mu.float(), n.float(), (s0.float() ** 2)[..., None], (sth.float() ** 2)[..., None], le[..., None], l1[..., None]], -1)
        return PlaneFactorQSFn.apply(P, M, bool(amp))


def plane_factor_qs_torch(mu, n, s0, sth, eps_logit, mask, M=2, amp=True):
    """Reference implementation of the sparse rule (tests; any device)."""
    from .density import cell_centres
    le = torch.nn.functional.logsigmoid(eps_logit); l1 = torch.nn.functional.logsigmoid(-eps_logit)
    C = cell_centres(mu.device).view(-1, 3); B, K, _ = mu.shape

    def terms(X, c):
        out = []
        for k in range(K):
            D = X[None] - mu[:, k, None]; d = (D * n[:, k, None]).sum(-1); r2 = (D * D).sum(-1)
            v0 = s0[:, k, None] ** 2 + r2 * sth[:, k, None] ** 2; s2 = c + v0
            a = l1[:, k, None] - d * d / (2 * s2) + (0.5 * torch.log(v0 / s2) if amp else 0)
            out.append(a)
        return out

    A = terms(C, SIGMA_CELL2); cen = 0; cnt = 0
    for k in range(K):
        phi = torch.logaddexp(le[:, k, None].expand_as(A[k]), A[k])
        cen = cen + torch.where(mask[:, k, None], phi, torch.zeros_like(phi))
        cnt = cnt + ((A[k] >= le[:, k, None] - 8) & mask[:, k, None]).long()
    sel = cnt >= 2
    c0 = (8.0 / M) ** 2 / 12.0
    offs = (torch.arange(M, device=mu.device, dtype=torch.float32) + 0.5) / M * 8.0 - 4.0
    lse = None
    for ox in offs:
        for oy in offs:
            for oz in offs:
                At = terms(C + torch.stack([ox, oy, oz]), c0); S = 0
                for k in range(K):
                    phi = torch.logaddexp(le[:, k, None].expand_as(At[k]), At[k])
                    S = S + torch.where(mask[:, k, None], phi, torch.zeros_like(phi))
                lse = S if lse is None else torch.logaddexp(lse, S)
    out = torch.where(sel, lse - math.log(M ** 3), cen)
    return out.view(B, NC ** 3, NF ** 3), sel.view(B, NC ** 3, NF ** 3)
