"""GPU rendering of Model B responses for one window (CuPy/NVRTC). Run it in a process WITHOUT the certified polytope
CUDA runtime (the repository rule: incompatible CUDA runtimes live in separate processes).

Each thread walks one receiver's closed segment from the source with Amanatides-Woo steps in float64
(`--fmad=false`), checking the occupancy bit of every cell entered.
* A step whose two smallest crossing parameters agree within `tie_rel` (an edge or vertex crossing, where the closed
  segment touches 4 or 8 cells) is FLAGGED, not guessed. The host recomputes flagged receivers with the exact rational
  reference `visibility.free_exact`.
* Unflagged decisions are those of the exact rule: a crossing gap above tie_rel is far larger than the float error.
Output classes: 0 inactive, 1 active, 3 SOLID receiver (inactive by convention)."""
import numpy as np

from .visibility import free_exact

_SRC = r"""
extern "C" __global__ void render(const unsigned char* bits, int N, double sx, double sy, double sz,
                                  int ox, int oy, int oz, int L, double tie_rel, unsigned char* out) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long n = (long long)L * L * L;
  if (i >= n) return;
  int x = (int)(i % L), y = (int)((i / L) % L), z = (int)(i / ((long long)L * L));
  int ex = ox + x, ey = oy + y, ez = oz + z;                      // receiver cell
  long long ei = ((long long)ex * N + ey) * N + ez;
  if ((bits[ei >> 3] >> (7 - (ei & 7))) & 1) { out[i] = 3; return; }
  double X[3] = { (2.0 * ex + 1.0) / (2.0 * N), (2.0 * ey + 1.0) / (2.0 * N), (2.0 * ez + 1.0) / (2.0 * N) };
  double S[3] = { sx, sy, sz };
  int c[3], step[3], e[3] = { ex, ey, ez };
  double tmax[3], tdel[3];
  for (int a = 0; a < 3; ++a) {
    c[a] = (int)floor(S[a] * N);
    double d = X[a] - S[a];
    step[a] = d > 0 ? 1 : (d < 0 ? -1 : 0);
    if (step[a] != 0) {
      double nxt = step[a] > 0 ? (double)(c[a] + 1) / N : (double)c[a] / N;
      tmax[a] = (nxt - S[a]) / d; tdel[a] = (1.0 / N) / fabs(d);
    } else { tmax[a] = 1e300; tdel[a] = 1e300; }
  }
  for (int it = 0; it < 3 * N + 3; ++it) {
    int a1 = 0; if (tmax[1] < tmax[a1]) a1 = 1; if (tmax[2] < tmax[a1]) a1 = 2;
    double t1 = tmax[a1];
    if (t1 > 1.0) { out[i] = 1; return; }                         // the segment ends in the current cell
    double t2 = 1e300;
    for (int a = 0; a < 3; ++a) if (a != a1 && tmax[a] < t2) t2 = tmax[a];
    if (t2 - t1 <= tie_rel * t1) { out[i] = 2; return; }           // edge / vertex crossing: exact on the host
    c[a1] += step[a1];
    double nxt = step[a1] > 0 ? (double)(c[a1] + 1) / N : (double)c[a1] / N;   // fresh, no accumulated error
    tmax[a1] = (nxt - S[a1]) / (X[a1] - S[a1]);
    if (c[0] == e[0] && c[1] == e[1] && c[2] == e[2]) { out[i] = 1; return; }
    if (c[0] < 0 || c[1] < 0 || c[2] < 0 || c[0] >= N || c[1] >= N || c[2] >= N) { out[i] = 2; return; }
    long long ci = ((long long)c[0] * N + c[1]) * N + c[2];
    if ((bits[ci >> 3] >> (7 - (ci & 7))) & 1) { out[i] = 0; return; }
  }
  out[i] = 2;
}

extern "C" __global__ void render_skip(const unsigned char* bits, const unsigned char* bocc, int N, int BS,
                                       double sx, double sy, double sz, int ox, int oy, int oz, int L, double tie_rel,
                                       unsigned char* out) {
  // Two-level traversal: inside an EMPTY 8^3 block (bocc bit 0) the walk advances block by block; on entering a
  // non-empty block it restarts the exact cell walk from the entry point. Every crossing (cell or block level) uses
  // the same tie test as `render`, and an entry cell that is not decided away from a cell boundary is FLAGGED.
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long n = (long long)L * L * L;
  if (i >= n) return;
  int x = (int)(i % L), y = (int)((i / L) % L), z = (int)(i / ((long long)L * L));
  int e[3] = { ox + x, oy + y, oz + z };
  long long ei = ((long long)e[0] * N + e[1]) * N + e[2];
  if ((bits[ei >> 3] >> (7 - (ei & 7))) & 1) { out[i] = 3; return; }
  int NB = N / BS;
  double X[3] = { (2.0 * e[0] + 1.0) / (2.0 * N), (2.0 * e[1] + 1.0) / (2.0 * N), (2.0 * e[2] + 1.0) / (2.0 * N) };
  double S[3] = { sx, sy, sz }, D[3];
  int step[3], c[3], eb[3];
  for (int a = 0; a < 3; ++a) { D[a] = X[a] - S[a]; step[a] = D[a] > 0 ? 1 : (D[a] < 0 ? -1 : 0); c[a] = (int)floor(S[a] * N); eb[a] = e[a] / BS; }
  double t = 0.0;                                                 // current parameter (start of the current cell / block)
  for (int it = 0; it < 6 * N + 6; ++it) {
    int b[3] = { c[0] / BS, c[1] / BS, c[2] / BS };
    long long bi = ((long long)b[0] * NB + b[1]) * NB + b[2];
    bool empty = !((bocc[bi >> 3] >> (7 - (bi & 7))) & 1);
    bool endblock = (b[0] == eb[0] && b[1] == eb[1] && b[2] == eb[2]);
    if (empty && !endblock) {
      // exit parameters of the current block
      double tm[3];
      for (int a = 0; a < 3; ++a) {
        if (step[a] == 0) { tm[a] = 1e300; continue; }
        double bnd = step[a] > 0 ? (double)((b[a] + 1) * BS) / N : (double)(b[a] * BS) / N;
        tm[a] = (bnd - S[a]) / D[a];
      }
      int a1 = 0; if (tm[1] < tm[a1]) a1 = 1; if (tm[2] < tm[a1]) a1 = 2;
      double t1 = tm[a1]; double t2 = 1e300;
      for (int a = 0; a < 3; ++a) if (a != a1 && tm[a] < t2) t2 = tm[a];
      if (t1 > 1.0) { out[i] = 1; return; }
      if (t2 - t1 <= tie_rel * t1) { out[i] = 2; return; }
      // enter the next block: cell index along a1 is the first cell of the new block, the others from the point
      t = t1;
      for (int a = 0; a < 3; ++a) {
        if (a == a1) { c[a] = step[a] > 0 ? (b[a] + 1) * BS : b[a] * BS - 1; continue; }
        double v = (S[a] + t * D[a]) * N; double f = floor(v);
        if (v - f <= tie_rel * N || f + 1.0 - v <= tie_rel * N) { out[i] = 2; return; }    // too close to a cell boundary
        c[a] = (int)f;
      }
      if (c[0] < 0 || c[1] < 0 || c[2] < 0 || c[0] >= N || c[1] >= N || c[2] >= N) { out[i] = 2; return; }
      long long ci = ((long long)c[0] * N + c[1]) * N + c[2];
      if (c[0] == e[0] && c[1] == e[1] && c[2] == e[2]) { out[i] = 1; return; }
      if ((bits[ci >> 3] >> (7 - (ci & 7))) & 1) { out[i] = 0; return; }
      continue;
    }
    // cell-level step from cell c (exact crossing parameters recomputed from c)
    double tm[3];
    for (int a = 0; a < 3; ++a) {
      if (step[a] == 0) { tm[a] = 1e300; continue; }
      double nxt = step[a] > 0 ? (double)(c[a] + 1) / N : (double)c[a] / N;
      tm[a] = (nxt - S[a]) / D[a];
    }
    int a1 = 0; if (tm[1] < tm[a1]) a1 = 1; if (tm[2] < tm[a1]) a1 = 2;
    double t1 = tm[a1]; double t2 = 1e300;
    for (int a = 0; a < 3; ++a) if (a != a1 && tm[a] < t2) t2 = tm[a];
    if (t1 > 1.0) { out[i] = 1; return; }
    if (t2 - t1 <= tie_rel * t1) { out[i] = 2; return; }
    c[a1] += step[a1]; t = t1;
    if (c[0] == e[0] && c[1] == e[1] && c[2] == e[2]) { out[i] = 1; return; }
    if (c[0] < 0 || c[1] < 0 || c[2] < 0 || c[0] >= N || c[1] >= N || c[2] >= N) { out[i] = 2; return; }
    long long ci = ((long long)c[0] * N + c[1]) * N + c[2];
    if ((bits[ci >> 3] >> (7 - (ci & 7))) & 1) { out[i] = 0; return; }
  }
  out[i] = 2;
}
"""


class GPURenderer:
    def __init__(self, world, tie_rel=1e-12):
        # crossing parameters are recomputed from the cell index at every step (error a few ulp, ~1e-15 relative), so a
        # relative tie tolerance of 1e-12 keeps a 1000x safety margin while flagging ~1000x fewer rays than 1e-9
        import cupy as cp
        self.cp = cp; self.world = world; self.N = world.N; self.tie_rel = float(tie_rel)
        self.bits = cp.asarray(world.dense_bits())
        self.k = cp.RawKernel(_SRC, "render", options=("--fmad=false",))
        self.k2 = cp.RawKernel(_SRC, "render_skip", options=("--fmad=false",))
        # 8^3-block occupancy (bit set iff the block has an occupied cell), from the dense bitset
        BS = 8; NB = self.N // BS
        dense = np.unpackbits(world.dense_bits()).reshape(self.N, self.N, self.N)
        bo = dense.reshape(NB, BS, NB, BS, NB, BS).any(axis=(1, 3, 5))
        self.BS = BS; self.bocc = cp.asarray(np.packbits(bo.ravel()))
        self.stats = dict(windows=0, flagged=0)

    def window(self, s, o, L, skip=True):
        """(active bool [Z, Y, X], solid bool [Z, Y, X]) of the L^3 window with origin o (x, y, z), exact.
        skip=True uses the two-level kernel (empty 8^3 blocks crossed in one step)."""
        cp = self.cp; n = L ** 3
        out = cp.empty(n, cp.uint8); thr = 256
        if skip:
            self.k2(((n + thr - 1) // thr,), (thr,), (self.bits, self.bocc, np.int32(self.N), np.int32(self.BS), np.float64(s[0]), np.float64(s[1]),
                                                    np.float64(s[2]), np.int32(o[0]), np.int32(o[1]), np.int32(o[2]), np.int32(L),
                                                    np.float64(self.tie_rel), out))
        else:
            self.k(((n + thr - 1) // thr,), (thr,), (self.bits, np.int32(self.N), np.float64(s[0]), np.float64(s[1]), np.float64(s[2]),
                                                   np.int32(o[0]), np.int32(o[1]), np.int32(o[2]), np.int32(L), np.float64(self.tie_rel), out))
        r = cp.asnumpy(out).reshape(L, L, L)
        fl = np.argwhere(r == 2)
        for z, y, x in fl:
            X = (np.array([o[0] + x, o[1] + y, o[2] + z]) * 2 + 1) / (2.0 * self.N)
            r[z, y, x] = 1 if free_exact(self.world, s, X) else 0
        self.stats["windows"] += 1; self.stats["flagged"] += len(fl)
        return r == 1, r == 3
