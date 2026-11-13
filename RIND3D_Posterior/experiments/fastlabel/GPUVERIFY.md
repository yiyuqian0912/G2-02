# Batched certified point-source verification on the GPU (`verify.cu`, `gpuverify.py`)

## Question decided

Given a candidate source p and one label's window W (stored lit crop r_j, stored SOLID crop), decide **exactly**
whether p ∈ F_W^j:

1. **Window box.** p ∉ B̄(W) (host, exact).
2. **Legality.** p ∈ D(G): (0,1)³, not in solid, clearance > 2⁻²⁶.
3. **Observation.** For every FREE voxel centre x of W, lit_G(p, x) = r_j(x).
   * lit_G(p, x) = [x ∉ S_C(p) for every closed convex component C].
   * S_C(p) is the closed shadow polyhedron of `rind3d.rendering.polyhedra.shadow_planes`.

On failure, it also returns a deterministic set of violated receivers: per failing 16³ brick, the minimum and maximum
violating voxel index. These become new phase-1 constraints.

## Pipeline per batch of K candidates (one GPU submission under the cross-process lock)

| step | where | what |
|---|---|---|
| 1 | host | window-box and open-cube checks (vectorised) |
| 2 | GPU `shadow_rows`, thread per (source, body) | face signs c_i = −(r_i·p + d_i), negative rows, pair rows c_j r_i − c_i r_j with d = −n·p, in directed binary64 intervals; normalisation by max\|coef\| with directed division; FP32 copies. Also the clearance certificate: some face with v = a·p + d > 0 and v² > r²\|a\|² (directed), the same test as `verify_clearance` 'strict-separating-planes'. Status bits: 1 = sign or row not certified, 2 = clearance not certified, 4 = no face faces p. |
| 3 | GPU `verify`, block of 64 threads per (source, 16³ brick containing a FREE voxel) | brick box test per body (FP32 interval) → body mask; per 8³ leaf box test over the masked bodies; leaves are skipped when the box result already matches the leaf's observation code (all-lit / all-dark); otherwise per voxel over the leaf's unresolved bodies: FP32 filter → FP64 directed intervals → UNKNOWN. Comparison with the stored lit bit on the device; per-source mismatch count, brick min/max violating voxel, UNKNOWN list. |
| 4 | host | flagged sources: exact `legal_source`; exact rational rows (`fastplanes`) for status 1, then a second launch. UNKNOWN voxels of otherwise passing sources: native GMP predicate (`native.points`), as in the renderer. |

**Data and launches:**
* Device-resident per label: the stored packed crops (big-endian ZYX, uploaded as stored) and the 8³ leaf codes.
* Device-resident per geometry: the exact body-row enclosures.
* Per batch: 2 kernel launches, uploads of K × 3 coordinates and the work list, and compact downloads (4 ints per source, plus violating-voxel and UNKNOWN lists only when needed).

## Numerics (why every decision is exact)

**Coefficient enclosures.** Body rows are exact rationals, normalised so that max|coef| = 1. They are stored as
1-ulp binary64 enclosures (`encode_fast`, identical to `cuda_backend.encode`).
* Interval arithmetic with directed rounding gives enclosures of the exact pair-row coefficients. These are positive multiples of the rows of `shadow_planes`.
* Positive scaling does not change any half-space.
* Duplicate rows in the original are removed by `dict.fromkeys`; duplicates do not change an intersection.
* Exact-zero rows are dropped by the original; an interval row whose four coefficients all contain 0 is flagged and redone exactly.

**FP32 filter.**
* Normalised enclosures have |coef| ≤ 1. Rows wider than 2⁻³⁰ get NaN FP32 copies and always fall through to FP64.
* f = fl32(mid) satisfies |f − q| ≤ 2⁻²⁴ + 2⁻³⁰.
* Voxel centres (2i+1)·2⁻¹¹ and box corners are exact in binary32. `--fmad=false --ftz=false`.
* Then |fl32(((f_x x + f_y y) + f_z z) + f_w) − v| ≤ γ₄ · 4(1 + 2⁻²³) + 4(2⁻²⁴ + 2⁻³⁰) ≤ 1.25 · 2⁻²⁰ < E32 = 2⁻¹⁸. The same holds for box minima and maxima.
* So `v > E32` proves v > 0, and `v < −E32` proves v < 0.
* NaN comparisons are false, so NaN rows are never decided in FP32.

**FP64 fallback** (identical to `cuda_certified.cu`). `lo > 0` proves outside. `hi ≤ 0` proves inside: closed, so a voxel
exactly on a plane is inside.

**Box tests** use only the FP32 filter. An undecided box simply means "evaluate the voxels", so no correctness depends on it.

**SOLID.** Taken from the stored crop. It is checked once per label by the same kernel in solid mode: the geometry
bodies act as the "shadow", all voxels are care voxels, and UNKNOWN voxels are decided by the native membership
predicate. Any disagreement raises an error.

**Determinism.**
* The pass/fail decision, mismatch counts and the brick (min, max) sets do not depend on thread scheduling.
* The host sorts the violation lists.
* There is no cross-block early exit; it would make the reported receivers timing-dependent.

## Gate A (validation, `validate_verify.py`, `gatea/*.jsonl`)

**Windows:** one window per (family, side) of the 9 DEV scenes (`bench_plan.json`), 27 windows. Train split only.

**Candidates per window:**
* 40 uniform in the phase-1 outer bound B;
* 16 near the true source, scales 1e-9 … 1e-3;
* 12 at engine vertices (+1e-12);
* 12 uniform in the world;
* up to 24 within one ulp of a body face plane, which forces uncertifiable signs and the exact host rows.

**Reference:** the certified CUDA renderer (`gpurender.render_tiles`, bit-identical to `packed_batch`) over **all** tiles,
with no culling and exact UNKNOWN fallback.

**Checks:**
* dump mode, per candidate: the full predicted FREE-voxel field equals the reference;
* batch mode, all candidates at once: decision agreement, and every reported violated receiver is a true mismatch.

| build | windows | legal candidates | fields identical | batch disagreements | false violations | notes |
|---|---|---|---|---|---|---|
| default (E32 = 2⁻¹⁸) | 27 | 2684 | 2684 | 0 | 0 | 1096 host-row sources; 4 genuine UNKNOWN voxels decided exactly |
| `E32=1e-3` (forces FP64 for \|v\| < 1e-3) | 9 | 904 | 904 | 0 | 0 | exercises the FP64 interval path |
| `UNK_BAND=2e-4` (forces UNKNOWN for \|v\| < 2e-4) | 9 | 896 | 896 | 0 | 0 | 643,872 voxels decided by the native GMP predicate |

**An earlier harness bug,** found by this gate and fixed before the runs above:
* A second launch for host-row sources overwrote the first launch's device lists before they were downloaded.
* Effect: wrong reported receivers; decisions were unaffected.
* The results of that first attempt are kept in `gatea/old_flow/`.

## Speed (uncontended, `bench_verify.py`, 104 candidates from a phase-1 B)

| window | current tester (adaptive2, culling) | batched GPU | speedup |
|---|---|---|---|
| L64, 5 bodies | 20.1 ms/point | 0.23 ms/point (GPU 0.07) | 87× |
| L128, 17 bodies | 23.6 | 0.57 (GPU 0.09) | 42× |
| L256, 17 bodies | 44.2 | 0.67 (GPU 0.19) | 66× |

These timings still include the host `legal_source` (0.16–0.47 ms/point). It has since been replaced by the GPU
clearance certificate, with the host check only for flagged sources.

## Portability (H20 / other GPUs)

* The kernel is plain CUDA C++ compiled by NVRTC for the device's compute capability, through the existing `Runtime`.
* It uses only `__d{add,sub,mul,div}_r{d,u}`, which are IEEE directed-rounding intrinsics on all architectures since sm_20.
* The batch size is bounded by a device-memory budget per worker (`mem_budget`, default 96 MB).
* There is no host-CPU assumption. Re-run Gate A on a new device type before use; it takes about 10 minutes.
