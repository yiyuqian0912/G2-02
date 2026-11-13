// Batched certified point-source verification (RIND3D FastLabel research).
//
// For a batch of candidate sources p_k and one window W, decide EXACTLY whether every FREE receiver x of W has the
// observed class, i.e. lit(x) == [x outside every closed shadow polyhedron S_C(p_k)].
//
// Arithmetic (every decision is exact; nothing is approximated):
//  * shadow_rows: the rows of S_C(p) are built from the exact body-row enclosures with directed-rounding binary64
//    interval arithmetic. Each output row is an enclosure of a POSITIVE multiple of the exact rational row of
//    rind3d.rendering.polyhedra.shadow_planes, so every sign decision below is a decision about the same half-space.
//    A source whose face signs c_i cannot be certified (|c_i| at rounding level) is flagged and redone on the host
//    with the exact rational construction.
//  * FP32 filter: rows are normalised (max |coef| <= 1); with voxel coordinates exact in binary32 and fmad disabled,
//    |fl32(v) - v| <= 1.25 * 2^-20 < E32 = 2^-18 (derivation in GPUVERIFY.md). |fl32(v)| > E32 decides the sign.
//  * Otherwise the row is evaluated with the directed binary64 intervals of the certified renderer.
//  * Otherwise the voxel is UNKNOWN and is decided by the host with the exact native (GMP) predicate.
// Semantics: closed bodies, closed shadows (v <= 0 inside), voxel centres (i + 0.5) / N, SOLID from the stored mask
// (the stored mask itself is checked once per window by the same kernel in solid mode against the geometry rows).
struct I { double lo, hi; };
__device__ __forceinline__ I add(I a, I b) { return {__dadd_rd(a.lo, b.lo), __dadd_ru(a.hi, b.hi)}; }
__device__ __forceinline__ I sub(I a, I b) { return {__dsub_rd(a.lo, b.hi), __dsub_ru(a.hi, b.lo)}; }
__device__ __forceinline__ I mul(I a, I b) {
  double l = fmin(fmin(__dmul_rd(a.lo, b.lo), __dmul_rd(a.lo, b.hi)), fmin(__dmul_rd(a.hi, b.lo), __dmul_rd(a.hi, b.hi)));
  double h = fmax(fmax(__dmul_ru(a.lo, b.lo), __dmul_ru(a.lo, b.hi)), fmax(__dmul_ru(a.hi, b.lo), __dmul_ru(a.hi, b.hi)));
  return {l, h};
}
// identical to cuda_certified.cu: interval coefficient times a coordinate range [lo, hi] with 0 <= lo <= hi
__device__ __forceinline__ I axis(I a, double lo, double hi) {
  if (a.lo == 0. && a.hi == 0.) return {0., 0.};
  if (a.lo >= 0.) return {__dmul_rd(a.lo, lo), __dmul_ru(a.hi, hi)};
  if (a.hi <= 0.) return {__dmul_rd(a.lo, hi), __dmul_ru(a.hi, lo)};
  return {__dmul_rd(a.lo, hi), __dmul_ru(a.hi, hi)};
}

#ifndef E32
#define E32 3.814697265625e-06f   /* 2^-18 */
#endif
#define WMAX 9.313225746154785e-10 /* 2^-30: maximum enclosure width for a row to get an FP32 copy */
#define MAXF 64

__device__ __forceinline__ float mn(float a, float b) { return a < b ? a : b; }   // NaN-propagating via b
__device__ __forceinline__ float mx(float a, float b) { return a > b ? a : b; }

__device__ void emit_row(I r0, I r1, I r2, I r3, double* o64, float4* o32, int* bad) {
  double m = fmax(fmax(fmax(fabs(r0.lo), fabs(r0.hi)), fmax(fabs(r1.lo), fabs(r1.hi))),
                  fmax(fmax(fabs(r2.lo), fabs(r2.hi)), fmax(fabs(r3.lo), fabs(r3.hi))));
  bool z = r0.lo <= 0. && r0.hi >= 0. && r1.lo <= 0. && r1.hi >= 0. && r2.lo <= 0. && r2.hi >= 0. && r3.lo <= 0. && r3.hi >= 0.;
  if (z || !(m > 0.) || !isfinite(m)) { *bad = 1; m = 1.; }
  I r[4] = {r0, r1, r2, r3}; float f[4]; bool ok = true;
  for (int a = 0; a < 4; ++a) {
    double lo = __ddiv_rd(r[a].lo, m), hi = __ddiv_ru(r[a].hi, m);
    o64[2 * a] = lo; o64[2 * a + 1] = hi;
    if (!(hi - lo <= WMAX)) ok = false;
    f[a] = __double2float_rn(0.5 * lo + 0.5 * hi);
  }
  if (!ok) { float q = __int_as_float(0x7fffffff); f[0] = f[1] = f[2] = f[3] = q; }
  *o32 = make_float4(f[0], f[1], f[2], f[3]);
}

// grid (K, ceil(nb / 32)), block 32: thread = (source k, body b)
// status bits per source: 1 = a face sign or a row is not certified (host builds exact rows),
// 2 = clearance > r not certified by a strict face separator for some body (host runs legal_source),
// 4 = no face of some body faces the source (source in or on a closed body; host legal_source rejects it).
// Clearance certificate (same as rind3d lite_acceptance.verify_clearance 'strict-separating-planes'): for every body
// some face row with value v = a.p + d > 0 and v^2 > r^2 |a|^2, evaluated with directed intervals.
extern "C" __global__ void shadow_rows(const double* g, const int* go, int nb, const double* src, int K,
                                       const int* rstart, int rows_per_src, double* R64, float4* R32, int* rcnt, int* status,
                                       double r2) {
  int k = blockIdx.x, b = blockIdx.y * blockDim.x + threadIdx.x;
  if (k >= K || b >= nb) return;
  double p[3] = {src[3 * k], src[3 * k + 1], src[3 * k + 2]};
  int f0 = go[b], f1 = go[b + 1], nf = f1 - f0;
  if (nf > MAXF) { status[k] = 2; rcnt[k * nb + b] = 0; return; }
  I c[MAXF]; signed char sg[MAXF]; int bad = 0, clear = 0, nneg = 0;
  for (int i = 0; i < nf; ++i) {
    const double* f = g + 8 * (f0 + i); I v = {f[6], f[7]};
    for (int a = 0; a < 3; ++a) v = add(v, axis({f[2 * a], f[2 * a + 1]}, p[a], p[a]));
    c[i] = {-v.hi, -v.lo};
    sg[i] = c[i].hi < 0. ? -1 : (c[i].lo > 0. ? 1 : 0);
    if (!sg[i]) bad = 1;
    if (sg[i] < 0) ++nneg;
    if (!clear && v.lo > 0.) {
      double n2 = 0.;
      for (int a = 0; a < 3; ++a) { double m = fmax(fabs(f[2 * a]), fabs(f[2 * a + 1])); n2 = __dadd_ru(n2, __dmul_ru(m, m)); }
      if (__dmul_rd(v.lo, v.lo) > __dmul_ru(r2, n2)) clear = 1;
    }
  }
  if (!clear) atomicOr(status + k, 2);
  if (!nneg) { atomicOr(status + k, 4); rcnt[k * nb + b] = 0; return; }
  size_t base = (size_t)k * rows_per_src + rstart[b]; int n = 0;
  if (!bad) {
    for (int i = 0; i < nf; ++i) if (sg[i] < 0) {
      const double* f = g + 8 * (f0 + i);
      emit_row({f[0], f[1]}, {f[2], f[3]}, {f[4], f[5]}, {f[6], f[7]}, R64 + 8 * (base + n), R32 + base + n, &bad); ++n;
    }
    for (int i = 0; i < nf; ++i) if (sg[i] < 0) for (int j = 0; j < nf; ++j) if (sg[j] > 0) {
      const double* fi = g + 8 * (f0 + i); const double* fj = g + 8 * (f0 + j);
      I nr[3];
      for (int a = 0; a < 3; ++a) nr[a] = sub(mul(c[j], {fi[2 * a], fi[2 * a + 1]}), mul(c[i], {fj[2 * a], fj[2 * a + 1]}));
      I d = {0., 0.};
      for (int a = 0; a < 3; ++a) d = add(d, axis(nr[a], p[a], p[a]));
      emit_row(nr[0], nr[1], nr[2], {-d.hi, -d.lo}, R64 + 8 * (base + n), R32 + base + n, &bad); ++n;
    }
  }
  rcnt[k * nb + b] = n;
  if (bad) atomicOr(status + k, 1);
}

__device__ __forceinline__ int box_body(const float4* R, int r0, int n, float lx, float ly, float lz, float hx, float hy, float hz) {
  bool in = true;
  for (int r = r0; r < r0 + n; ++r) {
    float4 f = R[r];
    float a0 = f.x * lx, a1 = f.x * hx, b0 = f.y * ly, b1 = f.y * hy, c0 = f.z * lz, c1 = f.z * hz;
    float vmin = ((mn(a0, a1) + mn(b0, b1)) + mn(c0, c1)) + f.w;
    float vmax = ((mx(a0, a1) + mx(b0, b1)) + mx(c0, c1)) + f.w;
    if (vmin > E32) return 0;
    if (!(vmax < -E32)) in = false;
  }
  return in ? 1 : -1;
}

__device__ __forceinline__ int pt_body(const float4* R, const double* R6, int r0, int n, float x, float y, float z,
                                       double dx, double dy, double dz) {
  bool unk = false;
  for (int r = r0; r < r0 + n; ++r) {
    float4 f = R[r];
    float v = ((f.x * x + f.y * y) + f.z * z) + f.w;
#ifdef UNK_BAND
    if (fabsf(v) < UNK_BAND) { unk = true; continue; }     // validation build only: force the exact host path
#endif
    if (v > E32) return 0;
    if (v < -E32) continue;
    const double* g = R6 + 8 * (size_t)r; I w = {g[6], g[7]};
    w = add(w, axis({g[0], g[1]}, dx, dx)); w = add(w, axis({g[2], g[3]}, dy, dy)); w = add(w, axis({g[4], g[5]}, dz, dz));
    if (!isfinite(w.lo) || !isfinite(w.hi)) { unk = true; continue; }
    if (w.lo > 0.) return 0;
    if (!(w.hi <= 0.)) unk = true;
  }
  return unk ? -1 : 1;
}

// One block (64 threads) per work item (source k, 16^3 brick). Bits are the stored big-endian packed ZYX crops.
// solid_mode = 0: care = FREE (stored SOLID bit clear), expected inside = NOT lit (inside = in some shadow).
// solid_mode = 1: care = all voxels, expected inside = SOLID bit, rows = geometry bodies (one "source").
// lcode[leaf]: 0 every care voxel expects 0, 1 every care voxel expects 1, 2 mixed, 3 no care voxel (per 8^3 leaf).
// dump (optional, validation): per window voxel, 0/1 predicted inside, 2 unknown (only for source 0 of the batch).
extern "C" __global__ void verify(const float4* R32, const double* R64, const int* rstart, const int* rcnt, int rows_per_src,
                                  int nb, const int* work, int nwork, const unsigned char* lit, const unsigned char* sol,
                                  int solid_mode, const signed char* lcode, int ox, int oy, int oz, int L, int N,
                                  int* nmis, int* rec_n, int* rec, int rcap, int* nunk, int* unk, int ucap, unsigned char* dump) {
  int wi = blockIdx.x; if (wi >= nwork) return;
  int k = work[2 * wi], br = work[2 * wi + 1];
  const int nbr = L >> 4; int bx = br % nbr, by = (br / nbr) % nbr, bz = br / (nbr * nbr);
  const float4* R = R32 + (size_t)k * rows_per_src; const double* R6 = R64 + (size_t)k * rows_per_src * 8;
  const int* cnt = rcnt + k * nb;
  __shared__ unsigned long long s_mask, l_mask[8]; __shared__ int s_in, l_mode[8], s_cnt, s_min, s_max;
  const float h = 1.0f / (2 * N);                    // exact: N is a power of two
  int x0 = bx * 16, y0 = by * 16, z0 = bz * 16;      // window coordinates
  if (threadIdx.x == 0) { s_mask = 0ull; s_in = 0; s_cnt = 0; s_min = 0x7fffffff; s_max = -1; }
  __syncthreads();
  {
    float lx = (2 * (ox + x0) + 1) * h, ly = (2 * (oy + y0) + 1) * h, lz = (2 * (oz + z0) + 1) * h;
    float hx = (2 * (ox + x0 + 15) + 1) * h, hy = (2 * (oy + y0 + 15) + 1) * h, hz = (2 * (oz + z0 + 15) + 1) * h;
    for (int b = threadIdx.x; b < nb; b += blockDim.x) {
      int r = box_body(R, rstart[b], cnt[b], lx, ly, lz, hx, hy, hz);
      if (r == 1) atomicOr(&s_in, 1); else if (r < 0) atomicOr(&s_mask, 1ull << b);
    }
  }
  __syncthreads();
  if (threadIdx.x < 8) {
    int l = threadIdx.x, nl = L >> 3;
    int lxo = (l & 1) * 8, lyo = ((l >> 1) & 1) * 8, lzo = (l >> 2) * 8;
    int code = lcode[(((z0 + lzo) >> 3) * nl + ((y0 + lyo) >> 3)) * nl + ((x0 + lxo) >> 3)];
    int mode; unsigned long long lm = 0ull;
    if (code == 3) mode = 0;
    else if (s_in) mode = (code == 1 && !dump) ? 0 : 2;
    else {
      int lin = 0;
      float lx = (2 * (ox + x0 + lxo) + 1) * h, ly = (2 * (oy + y0 + lyo) + 1) * h, lz = (2 * (oz + z0 + lzo) + 1) * h;
      float hx = lx + 14 * h, hy = ly + 14 * h, hz = lz + 14 * h;   // centres i + 0.5 .. i + 7.5 (exact)
      for (unsigned long long m = s_mask; m; m &= m - 1) {
        int b = __ffsll((long long)m) - 1;
        int r = box_body(R, rstart[b], cnt[b], lx, ly, lz, hx, hy, hz);
        if (r == 1) { lin = 1; break; }
        if (r < 0) lm |= 1ull << b;
      }
      if (lin) mode = (code == 1 && !dump) ? 0 : 2;
      else if (!lm) mode = (code == 0 && !dump) ? 0 : 1;
      else mode = 3;
    }
    l_mode[l] = mode; l_mask[l] = lm;
  }
  __syncthreads();
  int lc = 0, lmin = 0x7fffffff, lmax = -1;
  for (int l = 0; l < 8; ++l) {
    int mode = l_mode[l]; if (!mode) continue;
    unsigned long long lm = l_mask[l];
    int lxo = (l & 1) * 8, lyo = ((l >> 1) & 1) * 8, lzo = (l >> 2) * 8;
    for (int v = threadIdx.x; v < 512; v += blockDim.x) {
      int wx = x0 + lxo + (v & 7), wy = y0 + lyo + ((v >> 3) & 7), wz = z0 + lzo + (v >> 6);
      int idx = (wz * L + wy) * L + wx;
      int sb = (sol[idx >> 3] >> (7 - (idx & 7))) & 1;
      int e;
      if (solid_mode) e = sb;
      else { if (sb) continue; e = 1 - ((lit[idx >> 3] >> (7 - (idx & 7))) & 1); }
      int pin;
      if (mode == 1) pin = 0;
      else if (mode == 2) pin = 1;
      else {
        float x = (2 * (ox + wx) + 1) * h, y = (2 * (oy + wy) + 1) * h, z = (2 * (oz + wz) + 1) * h;
        double dx = (ox + wx + .5) / N, dy = (oy + wy + .5) / N, dz = (oz + wz + .5) / N;
        bool u = false; pin = 0;
        for (unsigned long long m = lm; m; m &= m - 1) {
          int b = __ffsll((long long)m) - 1;
          int r = pt_body(R, R6, rstart[b], cnt[b], x, y, z, dx, dy, dz);
          if (r == 1) { pin = 1; break; }
          if (r < 0) u = true;
        }
        if (u && !pin) {
          int q = atomicAdd(nunk + k, 1); if (q < ucap) unk[(size_t)k * ucap + q] = idx;
          if (dump && k == 0) dump[idx] = 2;
          continue;
        }
      }
      if (dump && k == 0) dump[idx] = (unsigned char)pin;
      if (pin != e) { ++lc; lmin = min(lmin, idx); lmax = max(lmax, idx); }
    }
  }
  if (lc) { atomicAdd(&s_cnt, lc); atomicMin(&s_min, lmin); atomicMax(&s_max, lmax); }
  __syncthreads();
  if (threadIdx.x == 0 && s_cnt) {
    atomicAdd(nmis + k, s_cnt);
    int q = atomicAdd(rec_n + k, 2);
    if (q + 1 < rcap) { rec[(size_t)k * rcap + q] = s_min; rec[(size_t)k * rcap + q + 1] = s_max; }
  }
}

// verify2: one block (128 threads) per (source k, 64^3 tile). Same decisions as `verify` (same predicates, same
// expected values, same per-16^3-brick (min, max) violation records); only the work schedule differs:
// tile box test -> brick box tests (64 bricks x tile-masked bodies, in parallel) -> leaf box tests (512 leaves) ->
// voxels of unresolved leaves. tcode / bcode / lcode: observation codes of tiles / bricks / leaves (0, 1, 2, 3 as lcode).
extern "C" __global__ void verify2(const float4* R32, const double* R64, const int* rstart, const int* rcnt, int rows_per_src,
                                   int nb, const int* work, int nwork, const unsigned char* lit, const unsigned char* sol,
                                   int solid_mode, const signed char* lcode, const signed char* bcode, const signed char* tcode,
                                   int ox, int oy, int oz, int L, int N,
                                   int* nmis, int* rec_n, int* rec, int rcap, int* nunk, int* unk, int ucap, unsigned char* dump) {
  int wi = blockIdx.x; if (wi >= nwork) return;
  int k = work[2 * wi], t = work[2 * wi + 1];
  const int nt = L >> 6, nbr = L >> 4, nl = L >> 3;
  int tx = t % nt, ty = (t / nt) % nt, tz = t / (nt * nt);
  int x0 = tx * 64, y0 = ty * 64, z0 = tz * 64;
  const float4* R = R32 + (size_t)k * rows_per_src; const double* R6 = R64 + (size_t)k * rows_per_src * 8;
  const int* cnt = rcnt + k * nb;
  const float h = 1.0f / (2 * N);
  __shared__ unsigned long long t_mask, b_mask[64], l_mask[512];
  __shared__ int t_in, b_in[64], b_cnt[64], b_min[64], b_max[64], nlist, nbod;
  __shared__ unsigned char l_mode[512]; __shared__ short list[512]; __shared__ signed char bod[64];
  int tid = threadIdx.x;
  if (tid == 0) { t_mask = 0ull; t_in = 0; nlist = 0; }
  if (tid < 64) { b_mask[tid] = 0ull; b_in[tid] = 0; b_cnt[tid] = 0; b_min[tid] = 0x7fffffff; b_max[tid] = -1; }
  __syncthreads();
  int tc = tcode[t];
  if (tc == 3) return;
  {
    float lx = (2 * (ox + x0) + 1) * h, ly = (2 * (oy + y0) + 1) * h, lz = (2 * (oz + z0) + 1) * h;
    float hx = lx + 126 * h, hy = ly + 126 * h, hz = lz + 126 * h;          // centres i + 0.5 .. i + 63.5
    for (int b = tid; b < nb; b += blockDim.x) {
      int r = box_body(R, rstart[b], cnt[b], lx, ly, lz, hx, hy, hz);
      if (r == 1) atomicOr(&t_in, 1); else if (r < 0) atomicOr(&t_mask, 1ull << b);
    }
  }
  __syncthreads();
  if (!dump && ((t_in && tc == 1) || (!t_in && !t_mask && tc == 0))) return;
  if (tid == 0) { int n = 0; for (unsigned long long m = t_mask; m; m &= m - 1) bod[n++] = (signed char)(__ffsll((long long)m) - 1); nbod = n; }
  __syncthreads();
  // bricks: (brick, body) pairs
  if (!t_in) {
    for (int p = tid; p < 64 * nbod; p += blockDim.x) {
      int bi = p / nbod, b = bod[p % nbod];
      int bx = x0 + (bi & 3) * 16, by = y0 + ((bi >> 2) & 3) * 16, bz = z0 + (bi >> 4) * 16;
      float lx = (2 * (ox + bx) + 1) * h, ly = (2 * (oy + by) + 1) * h, lz = (2 * (oz + bz) + 1) * h;
      int r = box_body(R, rstart[b], cnt[b], lx, ly, lz, lx + 30 * h, ly + 30 * h, lz + 30 * h);
      if (r == 1) atomicOr(&b_in[bi], 1); else if (r < 0) atomicOr(&b_mask[bi], 1ull << b);
    }
  } else if (tid < 64) b_in[tid] = 1;
  __syncthreads();
  // leaves
  for (int q = tid; q < 512; q += blockDim.x) {
    int bi = q >> 3, l = q & 7;
    int bxg = tx * 4 + (bi & 3), byg = ty * 4 + ((bi >> 2) & 3), bzg = tz * 4 + (bi >> 4);
    int bc = bcode[(bzg * nbr + byg) * nbr + bxg];
    int bx = x0 + (bi & 3) * 16, by = y0 + ((bi >> 2) & 3) * 16, bz = z0 + (bi >> 4) * 16;
    int lxo = (l & 1) * 8, lyo = ((l >> 1) & 1) * 8, lzo = (l >> 2) * 8;
    int code = lcode[(((bz + lzo) >> 3) * nl + ((by + lyo) >> 3)) * nl + ((bx + lxo) >> 3)];
    int mode; unsigned long long lm = 0ull;
    if (code == 3 || bc == 3) mode = 0;
    else if (b_in[bi]) mode = (code == 1 && !dump) ? 0 : 2;
    else if (!b_mask[bi]) mode = (code == 0 && !dump) ? 0 : 1;
    else {
      int lin = 0;
      float lx = (2 * (ox + bx + lxo) + 1) * h, ly = (2 * (oy + by + lyo) + 1) * h, lz = (2 * (oz + bz + lzo) + 1) * h;
      for (unsigned long long m = b_mask[bi]; m; m &= m - 1) {
        int b = __ffsll((long long)m) - 1;
        int r = box_body(R, rstart[b], cnt[b], lx, ly, lz, lx + 14 * h, ly + 14 * h, lz + 14 * h);
        if (r == 1) { lin = 1; break; }
        if (r < 0) lm |= 1ull << b;
      }
      if (lin) mode = (code == 1 && !dump) ? 0 : 2;
      else if (!lm) mode = (code == 0 && !dump) ? 0 : 1;
      else mode = 3;
    }
    l_mode[q] = (unsigned char)mode; l_mask[q] = lm;
    if (mode) { int j = atomicAdd(&nlist, 1); list[j] = (short)q; }
  }
  __syncthreads();
  // voxels of listed leaves
  for (int j = 0; j < nlist; ++j) {
    int q = list[j], bi = q >> 3, l = q & 7, mode = l_mode[q];
    unsigned long long lm = l_mask[q];
    int bx = x0 + (bi & 3) * 16 + (l & 1) * 8, by = y0 + ((bi >> 2) & 3) * 16 + ((l >> 1) & 1) * 8, bz = z0 + (bi >> 4) * 16 + (l >> 2) * 8;
    int lc = 0, lmin = 0x7fffffff, lmax = -1;
    for (int v = tid; v < 512; v += blockDim.x) {
      int wx = bx + (v & 7), wy = by + ((v >> 3) & 7), wz = bz + (v >> 6);
      int idx = (wz * L + wy) * L + wx;
      int sb = (sol[idx >> 3] >> (7 - (idx & 7))) & 1;
      int e;
      if (solid_mode) e = sb;
      else { if (sb) continue; e = 1 - ((lit[idx >> 3] >> (7 - (idx & 7))) & 1); }
      int pin;
      if (mode == 1) pin = 0;
      else if (mode == 2) pin = 1;
      else {
        float x = (2 * (ox + wx) + 1) * h, y = (2 * (oy + wy) + 1) * h, z = (2 * (oz + wz) + 1) * h;
        double dx = (ox + wx + .5) / N, dy = (oy + wy + .5) / N, dz = (oz + wz + .5) / N;
        bool u = false; pin = 0;
        for (unsigned long long m = lm; m; m &= m - 1) {
          int b = __ffsll((long long)m) - 1;
          int r = pt_body(R, R6, rstart[b], cnt[b], x, y, z, dx, dy, dz);
          if (r == 1) { pin = 1; break; }
          if (r < 0) u = true;
        }
        if (u && !pin) {
          int qq = atomicAdd(nunk + k, 1); if (qq < ucap) unk[(size_t)k * ucap + qq] = idx;
          if (dump && k == 0) dump[idx] = 2;
          continue;
        }
      }
      if (dump && k == 0) dump[idx] = (unsigned char)pin;
      if (pin != e) { ++lc; lmin = min(lmin, idx); lmax = max(lmax, idx); }
    }
    if (lc) { atomicAdd(&b_cnt[bi], lc); atomicMin(&b_min[bi], lmin); atomicMax(&b_max[bi], lmax); }
  }
  __syncthreads();
  if (tid < 64 && b_cnt[tid]) {
    atomicAdd(nmis + k, b_cnt[tid]);
    int qq = atomicAdd(rec_n + k, 2);
    if (qq + 1 < rcap) { rec[(size_t)k * rcap + qq] = b_min[tid]; rec[(size_t)k * rcap + qq + 1] = b_max[tid]; }
  }
}
