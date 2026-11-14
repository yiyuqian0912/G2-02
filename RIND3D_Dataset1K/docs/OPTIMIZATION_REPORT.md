# RIND3D posterior-label and scene-production optimization report (Phase 3, 2026-11-12)

> **2026-11-13:** the ≈ 140 h tau-pencil labelling projection discussed here was superseded by FastLabel v3.4 (1K backfill 4.33 h; `../../RIND3D_StudentResearch/D_BACKFILL_REPORT.md`). Paths under `RIND3D_Research/` are in the archive tag `archive/rind3d-research-2026-11-13`.

## Setup and accounting

* **Machine.** Ryzen 7 5800H (8 cores / 16 threads), 31 GB RAM, RTX 3060 Laptop (6 GB, weak FP64).
* **Engine.** `RIND3D_Posterior` (tau-pencil engine; CPU reference and exact GPU pass).
* **Accounting.** All times are measured. Four quantities are kept separate:
  * **worker-seconds**: wall time of one task in one worker, with the machine fully loaded;
  * **CPU-s**: process CPU time;
  * **GPU use**: by number of workers and segments;
  * **machine throughput**: tasks per wall hour, with all 16 hardware threads busy.
* **No CPU-equivalent conversion.** GPU work is never converted into CPU-equivalent seconds.

Earlier Phase-2 evidence is in `../RIND3D_Research/posterior_labels/phase2_gpu/` (`docs/PHASE2_STEP3_GPU_PASS.md`, `evidence/`).

## Direction A: window sides and complete production cost

### Stage-B sample (1024³, 16-worker pool)

* **Sample.** A seeded stratified random sample of informative (window, source) tasks from 16 fresh canary scenes: k = 1–4, all nine v0.1 families, no thinning.
* **Configuration.** 4 GPU-assisted + 12 CPU workers; queue capacity 2^25 entries.

| side | n | worker-s p50 | mean | p95 | max | certified (TV ≤ 0.05) |
|---|---|---|---|---|---|---|
| 64 | 24 | 3.7 | 6.2 | 17.8 | 28 | 24/24 |
| 128 | 24 | 8.9 | 24.3 | 87.8 | 134 | 24/24 |
| 256 | 48 | 32.4 | 106.7 | 439 | 555 | 48/48 |

The cost is a heavy tail of large windows. The L256 mean is 3.3× its median, and the most expensive single labels need 1–2.5·10⁹ entry evaluations.

### Window profiles: complete per-scene labelling cost

Means from the table above. Labels per window under perfect m-balance = 2.5 on every side (the simulated controller achieves this, section F of `SELECTION_LAW_1K.md`).

| profile (windows 64/128/256) | labels per scene | worker-s per scene | machine s per scene (16 workers) | 1000 scenes |
|---|---|---|---|---|
| **1K contract, 16/24/24** | 160 | 8 100 | 507 | **≈ 141 h** |
| previous P64, 5/15/44 | 160 | 12 700 | 795 | ≈ 221 h |

**Result.** The 16/24/24 profile is 1.6× cheaper than 5/15/44. Even so, labels dominate the complete pipeline by two orders of magnitude.

### Physical stage per scene (measured, 1024³, 16 canary scenes)

| stage | per scene |
|---|---|
| certified CUDA fields | 2–12 s (≈ 2.5 s per source) |
| exact count index | < 0.1 s per source |
| selection, including 3 × 1024 rate proposals | 1–8 s |
| materialisation of the source-separated window observations | 5–21 s |
| **total** | **7–42 s** |
| storage | ≈ 2–3 MB (observations) + 0.8 MB per source field |

### Uninformative channels (all-lit / all-dark in W)

* At 128³ they cost 1–580 CPU-s, against a mean of 1 s for informative channels. The feasible sets are large.
* Their posteriors are deferred, with slot status `not_computed`; the slots are kept.

### Large windows: queue capacity is a correctness issue, not only a cost issue

* With the Phase-1 capacity (2^24 queue entries), 3/89 early L256 labels exceeded the queue and were reported with large UNDET mass (TV bound 0.27–1.0). They are stored as `uncertified`.
* At capacity 2^26, 2 of the 3 certify (TV 0 and 2.4e-9) at the same cost.
* The third still has one UNDET region. It hits the per-region polytope size limit (72 faces / 160 vertices).
* **Production policy:**
  * tier 1 runs every label at capacity 2^25;
  * tier 2 automatically retries any label with TV bound > 0.05 at capacity 2^26;
  * tier 3 stores any remaining label as `uncertified`, never as certified.

### Negative results

| Lever | Result |
|---|---|
| Per-cell node budgets (early stopping by refinement depth) | No speedup at any budget (4–256 nodes per cell). The cost is per-node entry evaluation, not deep refinement. |
| Stage-1 depth d1 = 5 / 6 | Mixed: 10× faster on one window, 1.3–1.5× slower on others. Not adopted. |

## Direction B: GPU queue memory and workers

| Configuration (160-task stream, val mix) | windows/h |
|---|---|
| CPU × 16 | 1801 |
| 2 GPU + 14 CPU (FP64 GPU pass) | 2530 |
| 4 GPU + 12 CPU (FP32v, device-queue cap 2^22, ≤ 435 MB pool per worker) | 2449 |
| 6 GPU + 10 CPU | **2794 (1.55×)** |
| 8 GPU + 8 CPU | 2773 |

* **Bounded queues.** The device queue is capped per worker, with an exact CPU fallback for oversize passes. This replaced the earlier OOM at 4 workers.
* **Saturation.** The GPU saturates at about 6 workers.
* **No MPS.** CUDA MPS would let processes share the GPU concurrently, but it fails to initialise on this GeForce laptop GPU.
* **Single-process multi-threaded GPU server.**
  * Design: numba `nogil` engines, one CUDA context, per-thread streams and thread-local pass state.
  * Correctness: digests are identical to sequential runs, with 6 threads 2× faster than sequential (measured under load).
  * Status: the implementation is in the package (`gpu_free_pool=False`), but it is not yet benchmarked against the process pool on an idle machine.

## Direction C: FP32/FP64 hybrid

* **Status-only filter (Phase 2).** Decides only 4–29% of component evaluations; no gain.
* **Value-aware FP32.**
  * How it works: FP32 classifies every gate and edge with rigorous, directed-rounding error bounds that contain the FP64 engine's values. FP64 is computed only for (i) the edges whose τ values enter Unew/Lnew and (ii) uncertain decisions, including `receiver_limited`, which is decided by FP32 bounds when certain.
  * Correctness: records are bit-identical on all tested windows.
  * Coverage: 99.99% of component evaluations are resolved in FP32; edge-level FP64 is needed for about 9%; full FP64 fallbacks are about 1e-4.
  * Gain: the kernel is only 1.4–1.7× faster (1.26 → 0.73 s on 2f308fe3_109), because per-segment launch and synchronisation now dominate. End-to-end per-window latency improves about 15%.
* **Caveat.** The bounds are derived for the whole operation sequence. The compiler flags are `--fmad=false`, with directed-rounding intrinsics for every bound operation. Bit-identity of the outputs is a software equivalence test, not a proof.

## Direction D: selection and rendering

See `RENDERING_BENCHMARK.md`. Summary: certified full fields plus exact count indices make selection nearly free (O(1) exact counts per proposal, rate streams included).

## Direction E: hardware

* **This machine.** The measured machine-level ceiling is ≈ 2800 windows/h on the val mix (1.55× CPU), and ≈ 620–1440 balanced-1K labels/h for the large-window-heavy 1K mix.
* **What would help, by structure** (not by extrapolated FLOPS):
  1. a GPU with fast FP64 and more memory (A100/H100-class), so that more GPU workers fit and FP64 kernels stop dominating;
  2. a many-core CPU node, since labels are embarrassingly parallel per (window, source).
* **Cloud.** No paid cloud was used.
* **Estimate.** At the measured per-label worker-seconds, the 1K campaign needs ≈ 8100 worker-s per scene ≈ 2.25 worker-hours per scene, i.e. ≈ 2250 worker-hours in total. A 64-core node would finish in ≈ 1.5–2 days.

## Direction F: early stopping with TV guarantees

* **Node budgets.** Negative result; see Direction A.
* **Relaxed sets.** For a relaxed set B ⊇ F, TV(q_B, q_F) = 1 − π(F)/π(B). This needs a certified lower bound on π(F), which requires the full constraint set on the IN regions, the expensive part. No useful bound was obtained for the tiny feasible sets of 1024³ windows (volumes 1e-21 – 1e-5).
* **Not done.** Sampling-based certification was not pursued. The acceptance probability of naive rejection sampling equals vol(F)/vol(D) ≤ 1e-5.
* **The real lever** is the heavy tail of individual large windows (queue blow-up from dual-tree expansions). Next steps:
  * smarter receiver-block ordering;
  * a source-space subdivision that keeps the expansion fronts bounded.

## Recommended production architecture (adopted for 1K v1)

1. **Physical stage.**
   * Runs sequentially in slot order, so the balancing controller is deterministic.
   * Per scene: certified CUDA full fields, exact count indices, quota selection with thinning, rate streams, source-separated observations.
   * Cost: ≈ 23 s per scene at 1024³, using one CPU core and the GPU briefly.
2. **Label stage.**
   * A resumable task queue over (scene, window, source), restricted to informative channels.
   * Scenes are processed in batches of 4, heaviest windows first within a batch, so complete scenes accumulate.
   * Workers: a mixed pool of 5 GPU-assisted + 10 CPU workers, exact and bit-identical to the CPU reference.
3. **Certification.**
   * Every slot stores its deterministic TV bound and status.
   * Slots with TV bound > 0.05 are retried automatically at the larger queue capacity (tier 2).
   * Anything left is stored as `uncertified` and never presented as qualified.
4. **Validation.**
   * Every scene gets the structural validator.
   * A seeded sample of scenes also gets physics spot checks against the canonical certified and independent predicates.
5. **Scaling beyond this machine.** Labels are embarrassingly parallel. The same code runs on a many-core CPU node (the `cpu` engine), or on several GPU nodes each with its own process pool. No algorithm or contract change is needed.
