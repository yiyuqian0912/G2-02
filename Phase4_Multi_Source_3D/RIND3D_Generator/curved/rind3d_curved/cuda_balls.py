"""Certified CUDA rendering of closed-ball occlusion and membership for 64^3 receiver tiles (CuPy, NVRTC).

Every arithmetic operation uses directed-rounding intrinsics (__dadd_rd/__dadd_ru, __dmul_rd/__dmul_ru), so each
interval contains the exact rational value; subtraction is addition of the exact negation. A voxel whose sign is
not certified is flagged UNKNOWN and recomputed exactly on the CPU (`spheres.segment_hits_ball_exact` /
`ball_contains_exact`). Receivers are (2 v + 1) / (2 N), exact in binary64. Output layout: [Z, Y, X]."""
import numpy as np

from .spheres import segment_hits_ball_exact, ball_contains_exact

_SRC = r"""
struct I { double lo, hi; };
__device__ __forceinline__ I mk(double v) { I r; r.lo = v; r.hi = v; return r; }
__device__ __forceinline__ I add(I a, I b) { I r; r.lo = __dadd_rd(a.lo, b.lo); r.hi = __dadd_ru(a.hi, b.hi); return r; }
__device__ __forceinline__ I sub(I a, I b) { I r; r.lo = __dadd_rd(a.lo, -b.hi); r.hi = __dadd_ru(a.hi, -b.lo); return r; }
__device__ __forceinline__ I mul(I a, I b) {
  double p0 = __dmul_rd(a.lo, b.lo), p1 = __dmul_rd(a.lo, b.hi), p2 = __dmul_rd(a.hi, b.lo), p3 = __dmul_rd(a.hi, b.hi);
  double q0 = __dmul_ru(a.lo, b.lo), q1 = __dmul_ru(a.lo, b.hi), q2 = __dmul_ru(a.hi, b.lo), q3 = __dmul_ru(a.hi, b.hi);
  I r; r.lo = fmin(fmin(p0, p1), fmin(p2, p3)); r.hi = fmax(fmax(q0, q1), fmax(q2, q3)); return r; }
__device__ __forceinline__ I sq(I a) { I m = mul(a, a); if (a.lo <= 0.0 && a.hi >= 0.0) m.lo = 0.0; return m; }
// three-valued: 1 true, 0 false, -1 unknown
__device__ __forceinline__ int le0(I v) { return v.hi <= 0.0 ? 1 : (v.lo > 0.0 ? 0 : -1); }
__device__ __forceinline__ int lt0(I v) { return v.hi < 0.0 ? 1 : (v.lo >= 0.0 ? 0 : -1); }
__device__ __forceinline__ int gt0(I v) { return v.lo > 0.0 ? 1 : (v.hi <= 0.0 ? 0 : -1); }
__device__ __forceinline__ int t_and(int a, int b) { return (a == 0 || b == 0) ? 0 : ((a == 1 && b == 1) ? 1 : -1); }
__device__ __forceinline__ int t_or(int a, int b) { return (a == 1 || b == 1) ? 1 : ((a == 0 && b == 0) ? 0 : -1); }

extern "C" __global__ void balls_tile(const double* balls, int nb, double sx, double sy, double sz,
                                      const int* origins, int T, int N, unsigned char* out_all) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= T * T * T) return;
  int ox = origins[3 * blockIdx.y], oy = origins[3 * blockIdx.y + 1], oz = origins[3 * blockIdx.y + 2];
  unsigned char* out = out_all + (size_t)blockIdx.y * T * T * T;
  int x = i % T, y = (i / T) % T, z = i / (T * T);
  double twoN = 2.0 * (double)N;
  double X[3] = { (double)(2 * (ox + x) + 1) / twoN, (double)(2 * (oy + y) + 1) / twoN, (double)(2 * (oz + z) + 1) / twoN };
  double S[3] = { sx, sy, sz };
  int blocked = 0, solid = 0;
  for (int b = 0; b < nb; ++b) {
    I w[3], d[3], e[3];
    for (int k = 0; k < 3; ++k) {
      w[k] = sub(mk(S[k]), mk(balls[4 * b + k]));
      d[k] = sub(mk(X[k]), mk(S[k]));
      e[k] = sub(mk(X[k]), mk(balls[4 * b + k]));
    }
    I rr = mul(mk(balls[4 * b + 3]), mk(balls[4 * b + 3]));
    I A = add(add(sq(d[0]), sq(d[1])), sq(d[2]));
    I B = add(add(mul(w[0], d[0]), mul(w[1], d[1])), mul(w[2], d[2]));
    I C = sub(add(add(sq(w[0]), sq(w[1])), sq(w[2])), rr);
    I E = sub(add(add(sq(e[0]), sq(e[1])), sq(e[2])), rr);
    I D = sub(mul(A, C), sq(B));
    int hit = t_or(t_or(le0(C), le0(E)), t_and(t_and(lt0(B), gt0(add(A, B))), le0(D)));
    blocked = (b == 0) ? hit : t_or(blocked, hit);
    int ins = le0(E);
    solid = (b == 0) ? ins : t_or(solid, ins);
  }
  if (nb == 0) { blocked = 0; solid = 0; }
  // bits: 0 blocked, 1 blocked unknown, 2 solid, 3 solid unknown
  out[i] = (unsigned char)((blocked == 1 ? 1 : 0) | (blocked == -1 ? 2 : 0) | (solid == 1 ? 4 : 0) | (solid == -1 ? 8 : 0));
}
"""


class BallTiles:
    """GPU tiles of (blocked-by-a-ball, inside-a-ball) for one ball set; exact by construction."""

    def __init__(self, balls_array):
        import cupy as cp
        from cupy.cuda import driver
        self.cp = cp; self.drv = driver
        # The certified polytope backend owns a separate CUDA context (rind3d.rendering.cuda_runtime, cuCtxCreate) and
        # leaves it current. CuPy works in the device's primary context: make that current around every CuPy call and
        # restore the previous context afterwards, so neither backend sees the other's handles.
        cp.cuda.runtime.getDeviceCount()                     # initialises the driver if nothing else has
        prev = driver.ctxGetCurrent(); driver.ctxSetCurrent(0)
        try:
            cp.cuda.Device(0).use()
            self.b = np.ascontiguousarray(np.asarray(balls_array, "<f8").reshape(-1, 4))
            self.db = cp.asarray(self.b) if len(self.b) else cp.zeros(4, cp.float64)
            self.k = cp.RawKernel(_SRC, "balls_tile", options=("--fmad=false",))
            self.k.compile()
            self.ctx = driver.ctxGetCurrent()
        finally:
            driver.ctxSetCurrent(prev)
        self.stats = dict(tiles=0, unknown=0)

    def tile(self, s, origin, T=64, N=1024):
        """(blocked bool [T,T,T] ZYX, solid bool [T,T,T] ZYX) for source s and tile origin (x, y, z)."""
        return self.tiles(s, [origin], T, N)[0]

    def tiles(self, s, origins, T=64, N=1024):
        """Batched: one launch for all tiles; returns [(blocked, solid)] in order."""
        cp = self.cp
        n = T ** 3; origins = np.ascontiguousarray(np.asarray(origins, np.int32).reshape(-1, 3))
        prev = self.drv.ctxGetCurrent(); self.drv.ctxSetCurrent(self.ctx)
        try:
            out = cp.empty(n * len(origins), cp.uint8)
            thr = 256
            self.k(((n + thr - 1) // thr, len(origins)), (thr,), (self.db, np.int32(len(self.b)), np.float64(s[0]), np.float64(s[1]), np.float64(s[2]),
                                                                cp.asarray(origins), np.int32(T), np.int32(N), out))
            allo = cp.asnumpy(out).reshape(len(origins), T, T, T)
        finally:
            self.drv.ctxSetCurrent(prev)
        return [self._resolve(s, o, origin, T, N) for o, origin in zip(allo, origins)]

    def _resolve(self, s, o, origin, T, N):
        blocked = (o & 1).astype(bool); solid = (o & 4).astype(bool)
        ub = np.argwhere(o & 2); us = np.argwhere(o & 8)
        self.stats["tiles"] += 1; self.stats["unknown"] += len(ub) + len(us)
        for z, y, x in ub:
            X = (np.array([origin[0] + x, origin[1] + y, origin[2] + z]) * 2 + 1) / (2.0 * N)
            blocked[z, y, x] = any(segment_hits_ball_exact(c[:3], c[3], s, X) for c in self.b)
        for z, y, x in us:
            X = (np.array([origin[0] + x, origin[1] + y, origin[2] + z]) * 2 + 1) / (2.0 * N)
            solid[z, y, x] = any(ball_contains_exact(c[:3], c[3], X) for c in self.b)
        return blocked, solid
