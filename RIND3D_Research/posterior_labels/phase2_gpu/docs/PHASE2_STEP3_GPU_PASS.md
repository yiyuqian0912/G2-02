# Phase 2, step 3: the repeated-evaluation bottleneck, and an exact GPU evaluation pass

Date: 2026-11-11.

Machine: local laptop.
* CPU: Ryzen 7 5800H, 8 cores / 16 threads.
* RAM: 31 GB.
* GPU: RTX 3060 Laptop, 6 GB.

No Phase-1 jobs were running. Every timing below comes from an otherwise idle machine.

## 1. Where the time goes (measured)

### Differential timing (`code/p5_delta.py`, `outputs/P5_DELTA.json`)

Method: one operation is executed twice. The extra CPU time is the cost of that operation.

| window | side, N | eval_comp2 | hull test | entry copy | node context |
|---|---|---|---|---|---|
| bdd0eb2e_18 | 64, 1024 | 58% | 4.7% | 1.5% | 0.2% |
| 2f308fe3_109 | 128, 1024 | 72% | 9.9% | 1.4% | 0.04% |
| e324ba2f_53 | 8, 128 | 13% | 21% | 3.8% | 3.1% |

### Pass timer (`code/p6_split.py`, `outputs/P6_SPLIT.json`)

At 1024³ the per-node entry loop takes 81–94% of wall time. Region geometry (clipping, τ ranges, splits, emission) is small.

At 128³ a label costs ~0.2 CPU-s. That is already below the 1 CPU-s target, and the time is dominated by fixed per-window overhead.

**The bottleneck is therefore the 1024³ entry loop.** Of its full component evaluations, 75–96% are exact repeats of (item, component, aspect) within the window (`P3_DUPCOUNT_*`).

## 2. Eliminating repeated evaluation on the CPU: measured negative

All variants below produce identical records.

| approach | full-evaluation reduction | CPU time vs reference |
|---|---|---|
| p3: per-entry records inherited along the ancestor chain | 1.03–1.17× | 0.81–1.0× (slower) |
| p4: window-wide exact item-record cache (hash) | 2–50× | 0.24–0.53× at 1024³ (2–4× slower); 0.74–0.82× at 128³ |

Why these fail:
* **p3.** Most repeats are not ancestor re-evaluations. They are the same item expansions recomputed in different region branches.
* **p4.** These repeats are scattered across the region tree. A window-wide cache must hold ~17M distinct records (2f308fe3_109) and misses DRAM on almost every lookup. Even a hit costs more than recomputing, since in-engine eval_comp2 is ~100–170 ns.
* **Allocation-free hull test.** No measurable gain once code-generation effects are removed. It is not used.

**Conclusion.** On a CPU, caching item records cannot remove the repeated-evaluation cost. Recomputation is cheaper than retrieval. What can remove the cost is making each repeated evaluation almost free, which is what massively parallel FP64 evaluation does.

## 3. The exact GPU evaluation pass (`code/tpl/engine_g2.py`, `gpu_pass2.py`, `run_g1.py`)

### Why per-entry evaluation is safe to parallelise

Each queue entry's outcome depends only on the entry and on per-pass region data, never on other entries. The outcome is one of:
* drop;
* survivor;
* expand into 8 children;
* refute the region.

Its contribution to the collapsed bounds Unew/Lnew is likewise independent of other entries.

Only four things depend on order:
* the first refuter in FIFO order (it also feeds the learned ring);
* the queue-capacity guard;
* survivor overflow;
* the order of survivors and children.

### Implementation

1. **CPU prefix.** The CPU processes the FIFO prefix of every pass, at least `gpu_prefix` = 64 entries, to catch ring-learned refuters cheaply.
2. **Handoff.** As soon as the remaining FIFO segment holds ≥ `gpu_min` = 1024 entries, it goes to the GPU.
3. **GPU segment evaluation.** The GPU evaluates whole FIFO segments in FP64 with `--fmad=false`. The operations are the CPU's, in the CPU's order:
   * item box;
   * gate;
   * corner face values `base (+a)(+b)(+g)`;
   * `(m*fb)/ff`;
   * min/max;
   * the DARK/LIT entry logic;
   * the receiver-limited rule;
   * `hull_misses_component` for X components.
4. **Order-dependent parts, preserved exactly:**
   * **Unew/Lnew:** order-encoded atomicMin/atomicMax.
   * **First refuter:** atomicMin over FIFO positions.
   * **Survivors and children:** compacted on the device in FIFO order by one packed int64 inclusive scan (survivor rank in the low 32 bits, expansion rank in the high 32 bits). Segment k+1 is the children of segment k, so this is exactly the CPU's FIFO.
   * **Guards:** if a segment could reach the CPU's queue-capacity guard or survivor overflow, the pass is redone on the CPU from scratch. This is exact because no host state the CPU pass reads has been modified. It was never triggered in any test.
5. **Residency.** The queue stays on the device during a pass. Per pass, the host uploads the FIFO segment plus the packed region data (pinned, asynchronous) and downloads the survivors, Unew/Lnew and counters.

### Exactness (tested, not assumed)

* **1024³:** all 12 val windows (`outputs/G1_VAL12_v3.json`).
* **128³:** Stage-A smoke windows.

Every tested window produced **bit-identical records** (sha256 of cell, lo, hi, kind) and identical decision statistics. The one exception is the evaluation counters on passes that end at a refuter, because the GPU also evaluated the entries after it.

The certificates are therefore exactly the Phase-1 certificates. No floating-point GPU test replaces a conservative predicate: the GPU computes the same FP64 values as the CPU engine and makes the same decisions.

### Single-process latency at 1024³

These are wall seconds for one window, one CPU thread plus the GPU. The engine is g1 (static `ne ≥ gpu_min` gating); g2's dynamic handoff fixes the one regression, see below.

| window | side | reference (CPU) | GPU pass | speedup | entry evaluations |
|---|---|---|---|---|---|
| 1c79650a_23 | 64 | 1.21 | 0.88 | 1.38 | 4.9e6 |
| bdd0eb2e_179 | 256 | 2.20 | 0.94 | 2.35 | 8.9e6 |
| bdd0eb2e_18 | 64 | 4.08 | 2.02 | 2.02 | 3.0e7 |
| a636b0aa_130 | 128 | 5.61 | 1.58 | 3.56 | 5.0e7 |
| bdb46e76_33 | 64 | 8.72 | 2.28 | 3.83 | 4.2e7 |
| 5200796c_231 | 256 | 8.99 | 11.56 → **1.49 with g2** | 0.78 → 6.25 | 7.7e7 |
| 2f308fe3_109 | 128 | 20.99 | 3.81 | 5.51 | 9.1e7 |
| 23584efd_55 | 64 | 24.15 | 5.22 | 4.63 | 8.3e7 |
| 82b22342_130 | 128 | 52.95 | 8.90 | 5.95 | 2.6e8 |
| 66a6ee76_173 | 256 | 50.01 | 6.77 | 7.39 | 3.7e8 |
| acaf483e_106 | 128 | 116.00 | 16.29 | 7.12 | 4.7e8 |
| ae6f5493_252 | 256 | 265.12 | 28.74 | 9.23 | 1.1e9 |
| **sum** | | **560.0** | **89.0 (g1)** | **6.3** | |

**The 5200796c_231 regression.** With g1, every pass of this window started below 1024 entries and grew only by expansion, so the GPU was never used. g2 hands off dynamically when the remaining queue reaches `gpu_min`, which fixes it.

**g2 CPU path.** The CPU entry loop now lives in its own njit function (`cpu_seg`). This removes a 23% slowdown that the objmode block caused in the hot loop. The g2 CPU-only path is within 4–5% of the reference.

### Cost breakdown and the FP32 filter

Breakdown for 2f308fe3_109 after the transfer optimisations (pinned, asynchronous, device compaction), out of 3.4–3.7 s wall:
* GPU kernel (FP64) plus scan: ~1.15 s;
* region upload: ~0.4 s;
* end-of-pass download: ~0.6 s;
* scatter: ~0.16 s;
* rest of the engine on the CPU: ~1.2 s.

**FP32 filter (rigorous, directed rounding, FP64 fallback).**
* It can decide only results that need no τ values: NOCOVER, COVER and AMBG.
* In practice it decides 4–29% of component evaluations. Most are PARTIAL and need the FP64 τ values for Unew/Lnew and the receiver-limited rule.
* It brings no end-to-end gain and is off by default (exactness was verified with it on).
* A value-aware FP32 path is possible but would save under 1 s per window here.

## 4. Throughput against the multicore CPU (`code/throughput.py`, `outputs/TP*_*.json`)

### Setup

* **Tasks:** the 10 val windows other than the two heaviest (acaf483e_106, ae6f5493_252). Each is repeated, and tasks are ordered heaviest first.
* **Engines:**
  * CPU workers run the reference engine;
  * GPU-assisted workers run engine_g2 + the GPU pass;
  * each worker has its own CUDA context on the one GPU.
* **Waiting:** after the first runs, waiting uses blocking CUDA events. A spin-wait was measured to burn a full core.

### Short stream: 40 tasks

This run is straggler-bound. With 16 workers, the wall time is roughly the latency of the heaviest single task under load (~75 s).

| configuration | wall s | windows/h | CPU-s per window |
|---|---|---|---|
| GPU-assisted × 1 (spin sync) | 135.3 | 1064 | 3.1 |
| GPU-assisted × 2 (spin sync) | 84.1 | 1712 | 4.0 |
| GPU-assisted × 2 (blocking sync) | 92.7 | 1553 | 2.9 |
| GPU-assisted × 4 or × 8 | out of GPU memory (6 GB; ~0.8–1.2 GB per worker on the largest windows) | | |
| CPU × 8 | 103.9 | 1386 | 20.3 |
| CPU × 16 | 78.4 | 1837 | 26.7 |

### Long stream: 160 tasks (the decisive measurement)

| configuration | wall s | windows/h | CPU-s per window |
|---|---|---|---|
| CPU × 16 (all hardware threads) | 319.8 | 1801 | 28.0 |
| **2 GPU-assisted + 14 CPU workers** | 227.7 | **2530 (1.40×)** | 19.4 |

### Reading

* **Per label.** GPU assistance cuts the CPU cost of a label by 5–9× (2.9 vs 18–28 CPU-s).
* **Per machine.** The machine-level gain over a fully used 16-thread CPU is only **1.40×**. A GPU-assisted worker took ~4.6 s per task under contention, against ~3.4 s alone.
* **Why the gain is capped.** The laptop GPU has low FP64 throughput (~0.9e8 entry evaluations/s across all GPU workers), and only two workers' buffers fit in VRAM.
* **Levers.** There are two independent ones:
  1. A value-aware FP32 path. The status-only filter in section 3 is not enough. Most evaluations are PARTIAL, and their FP64 τ values are needed only for the Unew/Lnew contributors (the DARK unique candidate's undecided edges, and the LIT single-undecided edge) and for uncertain receiver-limited decisions.
  2. Memory-bounded device queues, so more GPU workers fit.

## 5. P64 scene-level batches (`code/p64_scene.py`, `outputs/P64_128/*/P64_REPORT.json`, `outputs/P64_1024/*`)

### Design (declared)

* **Scene and source.** Validation-split v0.1 geometry with its stored single source. The v0.3 source prior is the v0.1 driver rule.
* **Proposals.**
  * For each physical side Lp ∈ {64, 128, 256}, M origins are drawn uniformly on the 128³ lattice.
  * The stream is keyed by (seed, scene, Lp) only, so it is **independent of the source**.
  * M is one global constant, not tuned per scene or per source.
* **Acceptance.** A window is accepted when it has lit-FREE and dark-FREE receivers **and** the source lies outside the closed box.
  * At 128³: the Phase-1 Stage-A float renderer; windows with an undecided receiver are skipped and counted (none occurred).
  * At 1024³: stored exact fields.
* **Fixed budget.** All accepted windows are kept, with no quota. λ_W(s) is therefore independent of s, and every label is exactly Uniform(F_W) on this **declared informative population**. This population is not the natural distribution.
* **Pilot.** The count per scene varies. The pilot scene (769d6508, M = 64) accepted 20% of proposals. M = 108 was then fixed for the 128³ scenes.

### 128³ results

16 CPU workers, reference engine.

| scene | proposals | accepted (64/128/256) | preprocess s | selection s | label wall s | label CPU s (per label) | storage | max TV bound |
|---|---|---|---|---|---|---|---|---|
| 769d6508 (pilot, M = 64) | 192 | 38 (6/8/24) | 0.38 | 1.45 | 0.69 | 6.2 (0.16) | 3.3 MB | 2.2e-15 |
| f97c87e3 | 324 | 109 (7/31/71) | 0.38 | 2.31 | 5.54 | 53.7 (0.49) | 11.8 MB | 2.9e-11 |
| 30178e73 | 324 | 121 (23/25/73) | 0.38 | 2.38 | 6.06 | 59.1 (0.49) | 11.2 MB | 1.9e-08 |
| ffb068ff | 324 | 101 (11/27/63) | 0.37 | 2.98 | 7.15 | 64.7 (0.64) | 16.8 MB | 1.3e-08 |

* **Certification.** Every label is certified, with TV bound ≤ 1.9e-8 (369/369 ≤ 0.05).
* **Cost.** A complete scene with ~64 labels at 128³ costs ~0.4 s preprocessing, ~1.5–2 s selection rendering and ~32 CPU-s of labels, i.e. ~3–4 s wall on this machine.
* **Where the time goes.** At 128³ the labeller is not the bottleneck; the per-window fixed overhead dominates, and the GPU does not help.

### Exact on-device capacity guard (added after the first 1024³ P64 run)

**The problem.** Heavy L = 256 windows reach the engine's queue capacity. For example, f97c87e3_L256_037 hits max_entries = 16.78M = 2^24 − 1024, and the CPU capacity guard fires 7.0M times. GPU pass v2 handled this conservatively: it redid the pass on the CPU, 42 times for this window, giving 107 s with only 6.4 s of GPU kernel time.

**The fix.** The guard is now replayed exactly on the device. The CPU keeps every expansion candidate after the first A = ⌊(NQ − 1024 − nq)/8⌋ in FIFO order as a survivor. A fix-up kernel turns those candidates into survivors, and the inclusive scan is recomputed. Segments are also processed in chunks of ≤ 2^21 entries, which bounds the per-segment buffers. The FIFO order is unchanged.

**Result for f97c87e3_L256_037.**

| run | time | vs reference |
|---|---|---|
| reference CPU | 174.2 s | 1× |
| GPU pass, CPU fallback on guard | 107 s | 1.6× |
| **GPU pass, exact on-device guard** | **23.0 s wall, 13.0 CPU-s** | **7.6×** |

Records and decision statistics are identical, including the 7,007,498 guard events.

### First 1024³ P64 run (scene f97c87e3, M = 64 per side, mixed pool with 2 GPU + 14 CPU workers, pre-guard-fix engine)

* **Selection.** 192 proposals, 64 accepted (5/15/44 by side), 128 uninformative, none containing the source. Selection took 109 s, single-process, reading stored 1024³ fields.
* **Labels.** 273 s wall and 2968 CPU-s, i.e. 46 CPU-s per label. The L = 256 windows dominate.
* **Certification.** All 64 labels certified, max TV bound 1.2e-4.
* The final runs with the guard fix (mixed pool and CPU-only baseline) are in `outputs/P64_1024/f97c87e3_{mix,cpu}`.

## 6. Final measurements with the final engine

Final engine: engine_g2 + GPU pass with the exact on-device guard and blocking events.

### Single-window latency at 1024³ (`outputs/G2_VAL12_final.json`)

All 12 val windows are **bit-identical** to the reference (records and decision statistics). There were 0 CPU fallbacks.

| window | reference s | GPU pass wall s (CPU s) | speedup |
|---|---|---|---|
| 1c79650a_23 | 1.23 | 0.97 (0.85) | 1.28 |
| bdd0eb2e_179 | 2.47 | 0.95 (0.83) | 2.60 |
| bdd0eb2e_18 | 4.12 | 2.21 (1.99) | 1.86 |
| a636b0aa_130 | 5.49 | 1.44 (1.09) | 3.81 |
| bdb46e76_33 | 8.64 | 2.45 (1.96) | 3.53 |
| 5200796c_231 | 8.86 | 1.53 (0.99) | 5.81 |
| 2f308fe3_109 | 20.54 | 3.68 (2.67) | 5.58 |
| 23584efd_55 | 23.13 | 5.74 (4.59) | 4.03 |
| 82b22342_130 | 52.83 | 8.30 (5.55) | 6.36 |
| 66a6ee76_173 | 49.50 | 6.73 (3.97) | 7.36 |
| acaf483e_106 | 117.67 | 16.01 (10.14) | 7.35 |
| ae6f5493_252 | 263.23 | 28.42 (15.98) | 9.26 |
| **sum** | **557.7** | **78.4 (50.6)** | **7.1× wall; 11× fewer CPU-s** |

### P64 at 1024³, scene f97c87e3, complete scene

64 windows (5/15/44 by side). All certified, max TV bound 1.2e-4.

| pool (16 workers) | selection s | label wall s | label CPU s (per label) | windows handled by GPU workers |
|---|---|---|---|---|
| CPU only | 101 | 300.3 | 3183 (49.7) | 0 |
| 2 GPU-assisted + 14 CPU | 106 | **224.2 (1.34×)** | 2117 (33.1) | 25 |

* The label digests are identical between the two pools (64/64).
* Selection here means reading stored fields in a single process. In production it is part of observation generation, and it parallelises.

### Summary against the targets

* **Accuracy.** Every label produced in Phase 2 is certified with a deterministic TV bound ≤ 1.2e-4. That is 369 labels at 128³ and 64 at 1024³, plus 12 val windows.
* **Cost at 128³.** 0.16–0.64 CPU-s per label, under the 1 CPU-s target.
* **Cost at 1024³.** ~33–50 CPU-equivalent seconds per label for a P64 scene dominated by L = 256 windows. That is far above the target (≤ 5 s ceiling).
  * The exact GPU pass gives 7.1× single-window latency.
  * Against a fully used 16-thread CPU it gives only 1.34–1.40× machine throughput, because the consumer GPU's FP64 rate is low and only two GPU workers fit in 6 GB.
* **Remaining levers, in order of expected gain:**
  1. **A value-aware FP32 GPU path.** FP64 would be needed only for the Unew/Lnew contributors and for uncertain decisions. The FP32 rate on this GPU is 64× the FP64 rate.
  2. **Memory-bounded (ring) device queues,** so that all CPU threads can be GPU-assisted.
  3. **Fewer queue-capacity stalls on L = 256 windows.** At NQ = 2^24 they turn expansions into extra region splits. Changing NQ changes the records (still exact), so it needs its own validation.
  4. **512³** remains a last resort and has not been used.
