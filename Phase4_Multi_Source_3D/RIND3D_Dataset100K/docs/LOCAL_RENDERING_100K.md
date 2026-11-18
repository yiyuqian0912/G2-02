# Report A — Local-only response production: measured comparison and recommendation

**Measurement setup:**
* RTX 3060 Laptop (6 GB), Ryzen 7 5800H, unified environment;
* certified rind3d renderer: CUDA directed-rounding intervals plus exact CPU fallback for UNKNOWN voxels;
* contract `rind3d-100k-contract-v1`, quotas 4/4/4.

All numbers are **laptop measurements** unless marked otherwise.

## 1. Primitives

| primitive | cost | exactness |
|---|---|---|
| certified CUDA 64³ tile, batch 1 / 8 / 32 | 0.76 / 0.26 / 0.15 ms per tile | exact (certified intervals plus exact fallback) |
| full 1024³ field per source (render_stream, 4 CPU packers) | ≈ 2.0 s | exact |
| exact CPU receiver probe (`native.points`) | 0.5–0.7 µs per receiver (one thread) | exact |
| exact rational all-active hull certificate | ≈ 0.1 ms per channel | proof (sufficient condition) |
| coarse 128³ field (N = 128, 8 tiles) | ≈ 20 ms per source | exact for the coarse world, **not** for 1024³ receivers |
| NVRTC compile of the renderer | 1.2 s per process (once) | |

## 2. Strategies

| | strategy | what it computes | status |
|---|---|---|---|
| A | full global fields plus exact count index (the 1K pipeline) | k certified 1024³ fields; every proposal is O(1) | baseline, run with 4/4/4 quotas |
| B | random local proposals with sparse exact screening | exact CPU probes prove informativeness; render otherwise | part of L |
| C | coarse 128³ screening | coarse field predicts informativeness | **measured; rejected for selection (biased)** |
| D | hierarchical screening | probes → hull certificate → progressive tiles with early stop → full render of accepted windows | part of L |
| E | batched GPU-local rendering | tiles batched (32 per call) per channel; geometry resident per scene | part of L |
| **L** | **B + D + E (`rind3d_100k/local_eval.py`)** | lazy exact decisions; only accepted windows are fully rendered | **recommended** |

## 3. Correctness

**L against A** (same seed, slot, plan and quotas; 12 scenes, k = 1..4; thinning tables c ∈ [0.05, 1]):
* identical accepted windows on every scene;
* identical Λ̂ on all three sides;
* **0 mismatches in 2.9×10⁹ response bits**.

**Physics spot checks** of stored local crops:
* against the certified CPU predicate: 4,608 receivers, 0 mismatches;
* against the independent predicate: 576 receivers, 0 mismatches.

**Coarse screen C** against exact fine informativeness (18 scenes; 2,040 random source-excluded channel-proposals per
side):

| side | exactly informative | missed by coarse (false negatives) | coarse false positives |
|---|---|---|---|
| 64 | 367 | 50 (**13.6%**) | 2 (0.12%) |
| 128 | 697 | 35 (5.0%) | 2 (0.15%) |
| 256 | 1354 | 31 (2.3%) | 1 (0.15%) |

**Reading of C:**
* A coarse rejection step would remove 13.6% of the informative L = 64 channels: thin response-inactive regions, slits, grazing
  boundaries.
* That is a geometry- and source-dependent selection that changes the per-channel posterior and cannot be repaired by
  the scene-level ω.
* Coarse fields are cheap, but they cannot *certify* anything about 1024³ receivers. C is therefore not used, not
  even as a pre-filter. Using it only to *order* exact work would be law-neutral, but it brings no measurable saving:
  rendering is already lazy.

## 4. Cost

**L against A on the same scenes** (12 scenes, scalar R = 128, one GPU worker):
* L: 1.5–6.4 s per scene;
* A: 3.3–14.3 s per scene.
* Equal decisions; L is 2–3× faster.

**Ablation on one k = 4 scene** (R = 256, c ≡ 1):

| step added | scene wall | tiles | note |
|---|---|---|---|
| naive local (unpack every tile, render all unresolved channels) | 30.0 s | 38,881 | Python unpacking dominates |
| popcount on packed tiles + progressive early stop | 8.5 s | 36,140 | |
| + exact all-active hull certificate + incremental channel resolution | 6.6 s | 24,453 | 846 of 2,484 channel checks certified without rendering |
| more probes (4×) | 9.2 s | 22,903 | probes are not the bottleneck |
| R = 128 instead of 256 | 5.2 s | 13,422 | the rate stream dominates the remaining cost |
| full-field pipeline A, same scene | 11.4 s | | |

**Where the remaining cost is** (k = 4 scene, R = 256):
* The rate stream for ω uses ≈ 85% of the tiles, L = 256 the most (18k of 23.6k tiles).
* The contract therefore uses per-side R = 256 / 128 / 64. Its ESS cost is 0.94 / 0.97 / 0.98 (SELECTION_LAW_100K §3).

**96-scene dev canary** (contract settings, controller with 32-scene epochs, one GPU worker):

| quantity | value |
|---|---|
| accepted / attempts | 96 / 96 |
| wall per scene: mean / p50 / p95 / max | **3.7 / 3.9 / 6.1 / 6.7 s** (selection ≈ 2.5 s, serialisation ≈ 1.3 s CPU) |
| tiles per scene: mean / p95 | 3,988 / 6,661 (≈ 0.6 s GPU) |
| channel checks certified all-active without rendering | 41% (median 37%) |
| scene container | 0.36 MB mean (p95 0.64 MB) |
| GPU memory | ≈ 0.6 GB per physical worker |

**100K physical stage [projection]:**
* On the laptop: 100,000 × 3.7 s = 103 GPU-worker-hours.
* On 16 GPUs: with one worker per GPU and laptop-equal speed per worker, ≈ 6.4 h. With two workers per GPU, if the
  CPU part (≈ 1.3–2 s per scene) is the bottleneck, ≈ 3–4 h.
* It is **not** the dominant cost of the 100K campaign (see Report D). Measure on the H20 with `bench_scaling.py`.

## 5. Recommendation

**Use L.** It gives:
* exact local-only responses;
* lazy exact decisions;
* hull certificates;
* progressive batched tiles;
* full rendering only of the 12 accepted windows.

No global response field is ever computed or stored. The decisions are provably the exact rule's, and were verified
bit-for-bit against the full-field pipeline.

**Remaining optimisation headroom:**
* move window serialisation and statistics off the GPU worker (≈ 35% of the scene wall is CPU-side packing and
  statistics);
* batch tiles across the k sources (the renderer binds one source per call today).
