"""The exact local feasibility region of Model B (docs/VOXEL_PHYSICS_DESIGN.md, Theorem 10).

For a window observation X = (sigma, r) on W (sigma = the observed SOLID cells; r on the FREE cells) and a candidate
source q outside the closed window box B_W, legal w.r.t. sigma:
  (a) no active receiver's closed segment [q, x] touches a cell of sigma;
  (b) every inactive receiver y whose segment touches no cell of sigma has, on [q, y], a cell C outside W that touches
      no active receiver's segment and lies farther than rho from q.
L_B(X) = {q : (a) and (b)} is EXACTLY the union of the Model B feasible sets F_B(V, X) over all voxel worlds V that
agree with sigma on W: "within" because a blocker of an unexplained inactive receiver must lie outside W and off every
active segment; "onto" because the maximal world sigma + A(q), A(q) = {C outside W : C touches no active segment,
dist(q, C) > rho}, reproduces X from q. It depends on the observation only: a certified X-only source region.

Two implementations:
* `local_region_exact(...)`: rational reference (closed segments against closed cells, `touched_cells_exact`);
  `maximal_world(...)` builds sigma + A(q) for the converse check. Small grids only.
* `LocalRegionGPU`: two CuPy kernels per candidate. Soundness of the PRUNING decision does not need exact arithmetic:
  - `mark`: for every active receiver, the cells its segment certainly enters are marked forbidden; a cell entered at
    a near tie (two crossing parameters within tie_rel) is not marked (under-marking keeps more blockers available);
  - `check`: for every unexplained inactive receiver, every cell its segment may touch is a candidate blocker,
    including all cells of a near-tie block (over-approximation);
  - candidates are not filtered by rho (keeping more candidates is the safe side);
  - degenerate geometry (q within 1e-9 voxel of a lattice plane, a segment parallel to an axis) gives "keep q".
  Hence GPU-pruned implies pruned by the exact rule. (a) and the unexplained set come from the exact renderer of the
  observed SOLID cells alone (`cuda_dda.GPURenderer`)."""
import itertools
from fractions import Fraction

import numpy as np

from .visibility import touched_cells_exact
from .world import VoxelWorld

RHO = 2.0 ** -26


def _window_cells(o, L):
    o = np.asarray(o, np.int64)
    g = np.stack(np.meshgrid(np.arange(L), np.arange(L), np.arange(L), indexing="ij"), -1).reshape(-1, 3)   # z, y, x
    return o[None] + g[:, ::-1]                                                                             # x, y, z


def _dist2_point_cell(q, c, N):
    q = [Fraction(float(v)) for v in q]
    return sum((min(max(q[a], Fraction(int(c[a]), N)), Fraction(int(c[a]) + 1, N)) - q[a]) ** 2 for a in range(3))


def local_region_exact(N, o, L, sol, lit, q, rho=RHO):
    """Exact membership of q in L_B(X). sol, lit: bool [Z, Y, X] of the window (lit only meaningful on FREE cells).
    Returns dict(in_region, pruned_a, unexplained, unblockable, legal)."""
    o = np.asarray(o, np.int64); q = np.asarray(q, float)
    cells = _window_cells(o, L); s = sol.reshape(-1); a = lit.reshape(-1) & ~s
    W = {tuple(c) for c in cells.tolist()}; S = {tuple(c) for c in cells[s].tolist()}
    Wsig = VoxelWorld(N); Wsig.add_cells(cells[s])
    legal = bool(Wsig.legal_source(q, rho)) and not np.all((q >= o / N) & (q <= (o + L) / N))
    X = (cells + 0.5) / N
    touched = {}
    for i in np.flatnonzero(~s):
        touched[i] = touched_cells_exact(q, X[i], N)
    pruned_a = any(touched[i] & S for i in np.flatnonzero(a))
    forbid = set()
    for i in np.flatnonzero(a):
        forbid |= touched[i] - W
    unexpl = [i for i in np.flatnonzero(~s & ~a) if not (touched[i] & S)]
    rr = Fraction(rho) ** 2
    unblock = 0
    for i in unexpl:
        cand = [c for c in touched[i] - W - forbid if all(0 <= v < N for v in c) and _dist2_point_cell(q, c, N) > rr]
        unblock += int(len(cand) == 0)
    return dict(in_region=bool(legal and not pruned_a and unblock == 0), pruned_a=bool(pruned_a), unexplained=len(unexpl),
                unblockable=unblock, legal=legal)


def maximal_world(N, o, L, sol, lit, q, rho=RHO):
    """sigma + A(q): the observed SOLID cells plus every cell outside W that touches no active segment from q and lies
    farther than rho from q (the converse construction of Theorem 10)."""
    o = np.asarray(o, np.int64); q = np.asarray(q, float)
    cells = _window_cells(o, L); s = sol.reshape(-1); a = lit.reshape(-1) & ~s
    W = {tuple(c) for c in cells.tolist()}
    X = (cells + 0.5) / N
    forbid = set()
    for i in np.flatnonzero(a):
        forbid |= touched_cells_exact(q, X[i], N)
    rr = Fraction(rho) ** 2
    add = [c for c in itertools.product(range(N), repeat=3) if c not in W and c not in forbid and _dist2_point_cell(q, c, N) > rr]
    V = VoxelWorld(N); V.add_cells(cells[s]); V.add_cells(np.array(add, np.int64).reshape(-1, 3))
    return V


_SRC = r"""
__device__ __forceinline__ bool inwin(const int* c, int ox, int oy, int oz, int L) {
  return c[0] >= ox && c[0] < ox + L && c[1] >= oy && c[1] < oy + L && c[2] >= oz && c[2] < oz + L;
}
__device__ __forceinline__ bool ingrid(const int* c, int N) {
  return c[0] >= 0 && c[1] >= 0 && c[2] >= 0 && c[0] < N && c[1] < N && c[2] < N;
}
__device__ __forceinline__ long long cid(const int* c, int N) { return ((long long)c[0] * N + c[1]) * N + c[2]; }

// mode 0: mark certain cells outside W as forbidden. mode 1: return 1 iff some possibly-touched cell outside W is not
// forbidden (a candidate blocker), 0 if none, 2 if degenerate (caller keeps q).
__device__ int walk(int mode, const double* S, int ex, int ey, int ez, int N, int ox, int oy, int oz, int L,
                    double tie_rel, unsigned int* forbid) {
  double X[3] = { (2.0 * ex + 1.0) / (2.0 * N), (2.0 * ey + 1.0) / (2.0 * N), (2.0 * ez + 1.0) / (2.0 * N) };
  double D[3]; int c[3], step[3];
  for (int a = 0; a < 3; ++a) {
    D[a] = X[a] - S[a];
    double v = S[a] * N, f = floor(v);
    if (v - f < 1e-9 || f + 1.0 - v < 1e-9 || fabs(D[a]) * N < 1e-9) return 2;
    c[a] = (int)f; step[a] = D[a] > 0 ? 1 : -1;
  }
  // start cell: certainly touched
  if (!inwin(c, ox, oy, oz, L)) {
    long long id = cid(c, N);
    if (mode == 0) atomicOr(&forbid[id >> 5], 1u << (id & 31));
    else if (!((forbid[id >> 5] >> (id & 31)) & 1u)) return 1;
  }
  for (int it = 0; it < 3 * N + 3; ++it) {
    double tm[3];
    for (int a = 0; a < 3; ++a) {
      double nxt = step[a] > 0 ? (double)(c[a] + 1) / N : (double)c[a] / N;
      tm[a] = (nxt - S[a]) / D[a];
    }
    int i0 = 0, i1 = 1, i2 = 2, t;
    if (tm[i1] < tm[i0]) { t = i0; i0 = i1; i1 = t; }
    if (tm[i2] < tm[i1]) { t = i1; i1 = i2; i2 = t; }
    if (tm[i1] < tm[i0]) { t = i0; i0 = i1; i1 = t; }
    double t1 = tm[i0];
    if (t1 > 1.0) return mode == 0 ? 0 : 0;                 // the segment ends in the current cell (inside W)
    int nt = 1;
    if (tm[i1] - tm[i0] <= tie_rel * t1) { nt = 2; if (tm[i2] - tm[i1] <= tie_rel * t1) nt = 3; }
    int ax[3] = { i0, i1, i2 };
    if (nt > 1 && mode == 1) {
      // every intermediate cell of the tie block may be touched (closed semantics or float ordering)
      for (int m = 1; m < (1 << nt) - 1; ++m) {
        int cc[3] = { c[0], c[1], c[2] };
        for (int b = 0; b < nt; ++b) if ((m >> b) & 1) cc[ax[b]] += step[ax[b]];
        if (ingrid(cc, N) && !inwin(cc, ox, oy, oz, L)) {
          long long id = cid(cc, N);
          if (!((forbid[id >> 5] >> (id & 31)) & 1u)) return 1;
        }
      }
    }
    for (int b = 0; b < nt; ++b) c[ax[b]] += step[ax[b]];   // the cell after the (tied) crossings: certain
    if (!ingrid(c, N)) return 2;
    if (inwin(c, ox, oy, oz, L)) return 0;                  // inside the open window box from here on
    long long id = cid(c, N);
    if (mode == 0) atomicOr(&forbid[id >> 5], 1u << (id & 31));
    else if (!((forbid[id >> 5] >> (id & 31)) & 1u)) return 1;
  }
  return 2;
}

extern "C" __global__ void mark(const int* rec, int n, double sx, double sy, double sz, int N, int ox, int oy, int oz, int L,
                                double tie_rel, unsigned int* forbid) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double S[3] = { sx, sy, sz };
  walk(0, S, rec[3 * i], rec[3 * i + 1], rec[3 * i + 2], N, ox, oy, oz, L, tie_rel, forbid);
}

extern "C" __global__ void check(const int* rec, int n, double sx, double sy, double sz, int N, int ox, int oy, int oz, int L,
                                 double tie_rel, unsigned int* forbid, unsigned char* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double S[3] = { sx, sy, sz };
  out[i] = (unsigned char)walk(1, S, rec[3 * i], rec[3 * i + 1], rec[3 * i + 2], N, ox, oy, oz, L, tie_rel, forbid);
}
"""


class LocalRegionGPU:
    """Membership tests q in L_B(X) for one window observation, on the GPU (CuPy; separate process from the certified
    polytope CUDA runtime). `test(q)` returns dict(pruned, by, unexplained, unblockable, degenerate, seconds)."""

    def __init__(self, N, o, L, sol, lit, tie_rel=1e-12):
        import cupy as cp
        from .cuda_dda import GPURenderer
        self.cp = cp; self.N = int(N); self.o = np.asarray(o, np.int64); self.L = int(L); self.tie = float(tie_rel)
        self.sol = np.asarray(sol, bool); self.lit = np.asarray(lit, bool) & ~self.sol
        cells = _window_cells(self.o, self.L)
        self.cells = cells; self.free = ~self.sol.reshape(-1)
        Wsig = VoxelWorld(self.N); Wsig.add_cells(cells[self.sol.reshape(-1)])
        self.Wsig = Wsig; self.R = GPURenderer(Wsig, tie_rel=tie_rel)
        self.lit_rec = cp.asarray(cells[self.lit.reshape(-1)].astype(np.int32))
        self.forbid = cp.zeros(self.N ** 3 // 32, cp.uint32)
        self.k_mark = cp.RawKernel(_SRC, "mark", options=("--fmad=false",))
        self.k_check = cp.RawKernel(_SRC, "check", options=("--fmad=false",))

    def test(self, q):
        import time
        cp = self.cp; t0 = time.perf_counter(); q = np.asarray(q, float)
        o, L, N = self.o, self.L, self.N
        if np.all((q >= o / N) & (q <= (o + L) / N)) or not self.Wsig.legal_source(q):
            return dict(pruned=True, by="domain", seconds=time.perf_counter() - t0)
        l3, _ = self.R.window(q, o, L)                         # exact: lit under the observed SOLID cells alone
        if np.any(self.lit & ~l3):
            return dict(pruned=True, by="a", seconds=time.perf_counter() - t0)
        unexpl = (~self.sol) & (~self.lit) & l3                 # inactive, not explained inside the window
        n_u = int(unexpl.sum())
        if n_u == 0:
            return dict(pruned=False, by=None, unexplained=0, unblockable=0, degenerate=0, seconds=time.perf_counter() - t0)
        rec_u = cp.asarray(self.cells[unexpl.reshape(-1)].astype(np.int32))
        self.forbid.fill(0); thr = 256
        args = (np.float64(q[0]), np.float64(q[1]), np.float64(q[2]), np.int32(N), np.int32(o[0]), np.int32(o[1]), np.int32(o[2]),
                np.int32(L), np.float64(self.tie))
        n_l = int(self.lit_rec.shape[0])
        if n_l:
            self.k_mark(((n_l + thr - 1) // thr,), (thr,), (self.lit_rec, np.int32(n_l)) + args + (self.forbid,))
        out = cp.zeros(n_u, cp.uint8)
        self.k_check(((n_u + thr - 1) // thr,), (thr,), (rec_u, np.int32(n_u)) + args + (self.forbid, out))
        r = cp.asnumpy(out)
        unb = int((r == 0).sum()); deg = int((r == 2).sum())
        return dict(pruned=unb > 0, by="b" if unb > 0 else None, unexplained=n_u, unblockable=unb, degenerate=deg,
                    seconds=time.perf_counter() - t0)
