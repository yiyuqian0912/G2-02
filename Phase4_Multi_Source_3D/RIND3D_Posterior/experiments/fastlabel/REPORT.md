# RIND3D label-cost research: certified fast posterior labels (status 2026-11-12, laptop)

**Goal (user brief):** label the 1K backfill, 159,802 informative channels, in ≤ 4 h (max 6 h) on this laptop. The
hardware is a Ryzen 7 5800H (8C/16T), an RTX 3060 6 GB and 31 GB RAM. Requirements:
* TV ≤ 0.05;
* valid deterministic or statistical guarantees;
* the same target (Uniform(F_W^j) cell masses on 128³);
* the canonical unflattened storage.

**Verdict: not reached.** The best production-like configuration measures **≈ 18.7 h**, against ≈ 141 h for the
production engine (≈ 7.5× at the machine level). Per label on an unbiased random sample, it is 12× against the
reference engine.
* The remaining gap to 6 h is **≈ 3.1×**; to 4 h, ≈ 4.7×.
* Every label produced is certified, and no label is fabricated or relaxed.

## 1. Baselines (measured)

**Reference engine** (TPE v3.1, CPU, tier 1) on an unbiased random sample: 120 tasks, 40 per side, from 110 train
scenes (`random120.txt`, `refs.jsonl`).

| side | mean CPU-s | p50 | p95 | max |
|---|---|---|---|---|
| L64 | 11.4 | 6.0 | 36.6 | 75.8 |
| L128 | 32.2 | 20.5 | 93.3 | 222 |
| L256 | 150.5 | | 235.7 | 422 |

**1K mix** (label shares 24 / 38 / 38%): ≈ 72 CPU-s per label, i.e. ≈ 200 h on 16 threads. The production estimate,
with GPU-assisted workers, was 141 h.

## 2. Cost anatomy and negative results

These are in `../twophase/RESEARCH_NOTES.md`:
* cost tracks receiver-item evaluations;
* most of it is refuting candidate regions;
* sentinels and a deeper stage 1 do not help;
* phase-1 subsets made of pyramid blocks do not help;
* emission is the cost for very broad posteriors.

## 3. Methods

**3.1 Exact outer bound.** Phase 1 runs the TPE engine with a *point subset* X′ of the receivers: 1024 boundary plus
512 interior to start. This gives B = F(X′) ⊇ F, which is sound for any subset. B comes with exact cell masses q_B
and exact convex pieces.

**3.2 Portfolio, in order:**
1. **Single-cell support.** If B lies in one 128³ cell, the label is δ_c.
   * F ⊆ B ⊆ c and F ≠ ∅, because the true source generated the observation.
   * TV = 0, provided vol(F) > 0, which holds almost surely.
2. **Exact small B.** If B has ≤ 64 pieces, run the exact phase 2 restricted to B by the outer-bound filter
   (`engine_p2x`).
3. **L64 exact race.** A budgeted exact phase 2; kept if it completes.
4. **Statistical tier** (§3.3).
5. **Exact fallback.** Rare with geometric δ splitting (0 of 691 in run4d).

**3.3 Statistical tier.** It uses the brief's Direction B identity, with Direction C sampling.

*Identity.* F ⊆ B and both posteriors are uniform, so **TV(q_B, q_F) = 1 − vol(F)/vol(B)**. Cell-level TV is ≤ this.

*Discovery.* Uniform samples in B (piece chosen ∝ exact volume, then a tetrahedralisation) are tested **exactly**:
1. legal source (certified clearance);
2. outside the closed window box;
3. every FREE receiver matches the observation.

Check 3 uses the certified CUDA renderer, with exact CPU recomputation of UNKNOWN voxels. Pure tiles are certified
without rendering by the exact shadow polyhedra of the point source (float-filtered predicates, as in the engine).
The receivers violated by failing samples are added to X′, and phase 1 is re-run restricted to B. These are cutting
planes; B stays ⊇ F.

*Certification.* At most 8 attempts, with attempt c at level δ·2⁻ᶜ (n_c = 104, 117, …). Certify iff all n_c fresh
samples pass, i.e. the one-sided Clopper–Pearson bound gives p ≥ 0.95. Hence **P(a false certificate) ≤ δ = 0.01
per label**.

*Dataset level.* The expected number of false certificates is ≤ 0.01 × (number of statistical labels whose true
p < 0.95). In practice p ≈ 0.99+ after discovery: the validation run measured p̂ = 0.9965 on a hard case. A
simultaneous dataset-wide 99% guarantee would need δ/N per label, e.g. n ≈ 320 for 60k statistical labels, at ≈ 2.5×
the test cost.

*Stored metadata* for a statistical label: status `certified_stat`, ε, δ, n, k and p_lower.

**3.4 Exactness-preserving speedups** (each verified identical to the original):

| speedup | gain | verification |
|---|---|---|
| class tensor by a numba tiled transpose | 0.99 → 0.16 s at L256 | identical arrays |
| lazy block pyramid | | |
| integer shadow planes | 7.3 → 1.7 ms | 100 sources identical |
| exact fast coefficient enclosures | 2.84 → 0.98 ms | identical |
| split renderer (lock only around GPU work) | | bit-identical on 1440 tiles |
| one CUDA context per process; per-scene caches | | |

## 4. Accuracy

**36-window benchmark** (eval36.jsonl) and the **random 120** (eval120b.jsonl):
* every certified label matches its certified reference within **TV ≤ 0.015** (L64 ≤ 0.007, L128 ≤ 0.004,
  L256 ≤ 0.015);
* one reference was itself uncertified (engine tv_bound 0.42 after 222 s). Our statistical label was independently
  validated with 2000 fresh exact tests: p̂ = 0.9965, 99% CI [0.9915, 0.9990] (`validate_mc.py`).

## 5. Cost frontier

All numbers are full-machine. Production-like runs label all 691 informative channels of 4 random train scenes with
15 persistent workers (`runner.py`).

| configuration | s per label (worker) | 1K projection (16 workers) | notes |
|---|---|---|---|
| reference engine (random 120) | 72 | ≈ 200 h (production estimate 141 h with GPU assist) | |
| two-phase exact, B-filter (36-window benchmark) | ≈ 13 (L256 5.4×, L64 0.9×) | ≈ 36 h | exact |
| portfolio v2, fresh process per task (random 120) | 6.0 | 16.7 h | GPU contention and per-label CUDA init hidden |
| run4: persistent workers, no GPU lock | 7.55 | 21.0 h | 15 CUDA contexts thrash: 3–8 ms per tile instead of 0.07 |
| run4b: lock around whole point test | 10.5 | 31 h | lock held during CPU work; worse |
| run4c: lock only around GPU work, geometric δ | 7.53 | 20.9 h | exact fallbacks 9 → 0 |
| run4d: + exact point-shadow tile culling | 7.22 | 20.0 h | tiles −44% |
| **run4e: + smaller initial subset (1024 + 512)** | **6.75** | **18.7 h** (machine wall incl. tail: 20.9 h) | phase 1 −44% |

**Where the time goes** (run4e, 691 labels; sum of worker time 4.66k s):

| component | share |
|---|---|
| exact point tests | 60% |
| phase 1 | 18% |
| exact phase 2 | 7% |
| L64 race | 5% |
| setup + stage 1 | 6% |
| other / Python | ~4% |

The GPU was 77% busy.
* **Renderer kernel cost per 64³ tile** varies 0.04–0.45 ms with the geometry (e.g. 17 bodies / 127 faces is the
  slowest).
* **CPU per exact point test** is ≈ 3–6 ms: exact plane construction and encoding, which scale with the face count,
  plus driver calls.

## 6. What prevents 4–6 h, and the most promising path

1. **Exact point tests** (≈ 140 per statistical label; ≈ 33% of the labels; 60% of all time). This is the floor of the statistical
   certificate at δ = 0.01, ε = 0.05: n ≥ 90.
   * **Next:** a dedicated certified batched point-test kernel. Expected **≥ 3×** on tests.
     - It batches many candidate sources per launch, over only the mixed tiles' receivers.
     - It compares on the device, transferring a mismatch count instead of 128 KB per tile.
     - It builds shadow planes from integer rows in C.
2. **Phase 1** (≈ 1.2 s per label after the smaller subset). Options: the exact GPU pass for phase-1 node
   evaluation, numba construction of point items, and stage-1 tuning.
3. **Broad L64 / L128 tails** (1–2M pieces from fragmentation and emission). Emit whole IN polytopes and certify a
   cell-union outer bound statistically.
4. **Integration.** A `certified_stat` labeller tier in `rind3d_posterior`, with regression tests against references
   and scene-ordered persistent workers. This is the same architecture as the 100K coordinator.

**Estimated combined effect of 1–3:** ≈ 2–3× (to ≈ 6–9 h). That would approach, but likely not reach, the 6 h
maximum on this laptop. On the 16 × H20 server the GPU side is not a constraint, so these methods cut the 100K label
work from ≈ 33,400 to ≈ 5,500 laptop-thread-hours [projection: 2.94M labels × 6.75 s].

---

# Checkpoint 4 (2026-11-12, paused on request): FastLabel v3 — batched certified GPU verification

**New code:**
* `verify.cu`, `gpuverify.py`: batched exact point verification. Design and numerics are in `GPUVERIFY.md`.
* `adaptive3.py`: v3 pipeline with a round-wise k-tolerant certificate and keyed per-label streams. Default policy: dataset-simultaneous δ = 0.01 / 159,802.
* `runner3.py`, `analyze3.py`: throughput benchmark and its analysis.
* `fastwin.py` v3 helpers (numba class tensor, boundary mask and top items; identical outputs, `test_fastwin2.py`).
* `bench_plan.json`: DEV / HOLDOUT / GATEA plan, train split only.

**Audit:** `CERTIFICATE_AUDIT.md`. v2 is valid per label at δ = 0.01. Findings:
* seed 0 for every label;
* float-margin tile culling;
* all-pass waste.

All three are removed in v3.

**Gate A passed** (`gatea/`). On 27 windows (9 families × 3 sides), 2684 legal candidates gave bit-identical FREE-voxel
fields against the certified renderer. The FP64-band build (904 candidates) and the forced-UNKNOWN build (896
candidates; 643,872 voxels decided exactly) were also identical. There were 0 batch disagreements and 0 false violations.

**Speed of point tests:** 40–90× per point uncontended (0.23–0.67 ms vs 20–44 ms). Point tests are now 2–8% of the time.

**Throughput, production-like** (same 691 labels / 4 scenes as run4e, 15 workers; DEV tuning, not the final assessment):

| run | config | labels/s | 1K projection |
|---|---|---|---|
| run4e (v2) | | 2.13 | 20.9 h |
| v3a | v3 defaults (race on, subset 1024 + 512) | 5.73 | 7.75 h |
| v3b | race off | 5.93 | 7.48 h |
| v3c | subset 512 + 256 | 6.55 | 6.77 h |
| v3d | race off + 512 + 256 | 7.36 | 6.03 h |
| v3e | race off + small B statistical | 6.20 | 7.16 h |
| v3f | race off + 512 + 256 + small B statistical | 7.54 | 5.89 h |
| v3g | race off + 256 + 128 | 8.31 | 5.34 h |
| **v3h** | **race off + 256 + 128 + small B statistical** | **8.80** | **5.04 h** |
| v3i | race off + 512 + 256 + lighter stage 1 | 8.00 | 5.55 h |

v3j (stage-1 depth 3) and v3k (16 workers) were interrupted by the pause.

* Every run had 0 errors and 0 exact fallbacks.
* Every statistical label was certified at the dataset-simultaneous level (p_lower ≥ 0.950).
* Every deterministic label has tv_bound ≤ 3.3e-7.

**Resume from here:**
1. Finish the stage-1 / worker-count sweep (v3j, v3k) and combine with v3h.
2. Representative benchmark on the 18 DEV scenes, then the final assessment on the 18 HOLDOUT scenes (Gate C/D):
   * latency quantiles;
   * per family and side;
   * tail;
   * scene-affine scheduling (cold scene loads ≈ 0.9 s each).
3. Accuracy: v3 on the refs (random120) and the release labels (scenes 0–3), plus `validate_n` Monte Carlo checks.
4. Integration as a `certified_stat` tier in the production labeller:
   * package `rind3d_posterior.fastlabel`;
   * a `rind3d_1k.labels` engine;
   * loader status handling;
   * a GPU lock in the Supervisor;
   * tests.
5. Do not start the 1K backfill before Gates C/D and the user's go.

---

# Checkpoint 5 (2026-11-12, paused on request): FastLabel v3.3

**Changes since checkpoint 4.** Every change is validated identical, or is a valid change of the certificate design.
1. Lazy chunk sampling (`sampler3.BSampler`): vectorised folding, verified identical to the scalar version on 100k inputs; per-label triangulation cache.
2. Vectorised `build_filter2` / `items_from_vox2` (`fastbf.py`). Identical arrays on 6 windows; 1.75 s → 0.15 s on a 289k-piece B.
3. Process-wide reusable device buffers, and a per-process cache of the per-window solid check.
4. `verify2` kernel: one block per (source, 64³ tile), with bricks and leaves classified in parallel. Bit-identical on 4 windows (384 candidates). Speed is about the same as `verify`; the kernel is bound by voxel visits along shadow boundaries.
5. **Two-stage certificate per round** (`design2`; audited in CERTIFICATE_AUDIT.md):
   * n₁ = 512 with a₁ = 3, else n₂ = 1024 with a₂ = 17;
   * exact α = 2.9e-8 ≤ δ_r at the dataset-simultaneous level;
   * the same power as single-stage;
   * 402 of 457 labels stop at 512.
6. Cache-blocked class tensor (`fast_cls3`): identical on 29 windows. Under 15-way contention it takes 0.095 s, against 0.68 s for `fast_cls2` and 0.36 s for the original. `leaf_codes` is now read in memory order (identical).
7. Defaults are now the v3h configuration: race off, initial subset 256 + 128, small B certified statistically.
8. **v3.3: the vertex pool grows ×4 as needed and is kept for later rounds.**
   * Before the fix, very broad posteriors (≈ 1.1M pieces, ≈ 900k cells) fell back to the exact engine: 41–522 s each in DEV-18.
   * With the fix they certify in about 14 s (4 labels re-run individually).

**Throughput on the run4e sample** (691 labels, 15 workers):

| run | labels/s | 1K projection |
|---|---|---|
| v3p (items 1–3) | 9.05 | 4.91 h |
| v3q (+ two-stage) | 10.34 | 4.29 h |
| v3r (+ fast_cls3) | 11.22 | 3.96 h |

In v3r, p99 is 5.5 s and the GPU is 65% busy.

**DEV-18** (18 scenes, 2,562 labels): not completed yet.
* `runs/dev18_v3r_partial_vpoolbug.jsonl`: 1,982 labels, stopped. It exposed the vertex-pool tail of item 8.
* `runs/dev18_v33_partial.jsonl`: 584 labels with v3.3, stopped by the pause.

**Resume from here:**
1. Re-run DEV-18 with v3.3 to completion:
   `runner3.py 0 15 0 runs/dev18_v33.jsonl --gpu-permits 1 --scenes <dev>`
   then `analyze3.py … --strata`.
2. Freeze the configuration, then run HOLDOUT-18 (`bench_plan.json` "holdout") as the final assessment.
3. Accuracy: `runner3.py 0 15 0 runs/acc.jsonl --tasks runs/acc_tasks.txt --kw '{"validate_n": 4096}'` (116 refs and 333 release labels).
4. Production integration, as planned:
   * a `rind3d_1k.fastlabel` package;
   * a `fastlabel` engine in `labels.py` (status `certified_stat`, certificate in meta);
   * an optional Supervisor worker initialiser for the GPU semaphore;
   * an opt-in `accept_statistical` flag in the loader;
   * tests.

**Separate task (user, 2026-11-12):** the scientific foundations audit. Its brief is `RIND3D_Audit/BRIEF.md`. A background audit agent was started and then stopped by the pause before writing any results; relaunch it on resume. Constraints: no GPU, at most one niced CPU process, write only in `RIND3D_Audit/`.

---

# Checkpoint 6 (2026-11-13): FastLabel v3.4 — final assessment and production integration

**v3.4 = v3.3 plus an exact box fast path in the sampler.** A piece whose 8 vertices are the 8 corners of its bounding
box is that box, so it is sampled directly. The law is unchanged. It removes about 97% of the sampling time on
1M-piece bounds: 1.23 s → 0.04 s per 512 points.

The configuration was frozen before the HOLDOUT run.

## C. Performance (production-like, 15 workers, full machine, `runner3.py`, all informative channels of the scenes)

| set | scenes | labels | labels/s | 1K projection | p50 / p95 / p99 / max (s) | errors | exact fallbacks |
|---|---|---|---|---|---|---|---|
| run4e sample (v2 baseline) | 4 | 691 | 2.13 | 20.9 h | | 0 | 1 |
| DEV-18 (tuning set) | 18 | 2562 | 9.06 | 4.90 h | 0.81 / 3.8 / 23.6 / 37.8 | 0 | 0 |
| **HOLDOUT-18 (never used for tuning)** | 18 | 2520 | **12.23** | **3.63 h** | 0.72 / 2.5 / 14.5 / 34.5 | 0 | 0 |
| **pooled DEV + HOLDOUT** | 36 | 5082 | ≈ 10.4 | **4.27 h** | | 0 | 0 |

* Scene-bootstrap of the pooled projection (`runs/projection_v34.txt`, 20,000 resamples of the 36 scenes, machine efficiency 0.98 measured):
  * 90% interval **[3.53, 5.03] h**;
  * 99% upper bound **5.36 h**.
  * Per-scene cost varies from 0.48 to 3.61 worker-s per label; this is the dominant uncertainty.
* GPU utilisation is 34–40%. Peak memory is ≈ 16 GB of 31 GB.
* By family (HOLDOUT, mean s/label): occluder_clusters 0.53 … corridors 2.58. The heaviest tails come from very broad
  posteriors (thin sheets, ≈ 10⁵–10⁶ cells), mostly at L64 / L128.
* Worker-time components (HOLDOUT):

| component | share |
|---|---|
| phase 1 (tau-pencil on a receiver subset) | 45% |
| stage 1 | 24% |
| sampling | 14% |
| point tests | 6% |
| setup | 5% |
| verifier initialisation | 3% |

## Accuracy (`runs/acc_v34.jsonl`, `runs/accuracy_v34.txt`)

Run on every task that has a certified reference (449):

| reference | n | max TV | mean TV |
|---|---|---|---|
| certified tau-pencil v3.1 references (random 120) | 113 | 0.0164 | 0.0011 |
| production release labels (scenes 0–3) | 334 | 0.0147 | 0.0013 |

* Two references with a vacuous bound were excluded.
* **Independent Monte Carlo check.** This is empirical and NOT part of the certificate: 4096 fresh exact tests for each of the 315 statistical labels.
  * p̂ = vol(F)/vol(B): min 0.981, median 1.000.
  * No 99% interval lies below 0.95.
  * This is consistent with every certificate.

## E. Certification frontier

All rows use the GPU verifier. Costs are tests per statistical label; the round-1 thresholds are shown.

| policy | per-label δ | n₁ / a₁, n₂ / a₂ | P(certify) at p = 0.99 | tests per statistical label (measured) |
|---|---|---|---|---|
| v2 all-pass | 0.01 | 104 / 0 | 0.35 (then retries) | ≈ 140 + discovery |
| v3 single-stage, dataset-simultaneous | 6.3e-8 | 1024 / 17 | 0.98 | 1229 |
| **v3.2+ two-stage, dataset-simultaneous (default)** | 6.3e-8 | 512 / 3, 1024 / 17 | 0.98 | **787–806** |

* One certificate also bounds other ε′: P(TV > 0.10 and certified) ≤ P(Bin(1024, 0.10) ≤ 17) ≈ 4e-27.
* Confidence (δ) and the error bound (ε) are reported separately and never added.

## F. Recommendation

**The full 159,802-label backfill is projected to finish within the 6 h maximum on this laptop.** The pooled point
estimate is 4.27 h, with a 90% interval of 3.53–5.03 h and a 99% upper bound of 5.36 h. Under the GPU-batched pipeline
this is ≈ 33× fewer machine hours than the original 141 h.

Every label carries one of two certificates:
* a deterministic one: single-cell support (TV 0), or an exact path;
* a statistical one: TV ≤ 0.05, with P(any false statistical certificate in the whole 1K) ≤ 0.01.

The **4 h preferred target is not reliably met.** The point estimate exceeds it by about 7%, and the scene-level spread
is wide.

**Strongest next levers**, by measured share:
1. Phase 1 (45%): a GPU port of the subset-phase item evaluation, or sharing stage-1 / phase-1 work across the source channels of a window.
2. Stage 1 (24%): GPU sentinel box tests.
3. Sampling (14%): a numba convex-hull tetrahedralisation instead of Qhull for non-box pieces.

**Go / no-go:** GO technically, pending the user's decision. The backfill command and the meaning of the label statuses
are in `RIND3D_Dataset1K/docs/FASTLABEL_1K.md`.

## G. Production integration (`yushu/3d`)

* **`rind3d_1k.fastlabel`** (production copy of the verified research code, v3.4):
  * `pipeline.label_task`;
  * `gpuverify` / `verify.cu`;
  * `fastplanes`, `window`, `sampler`, `filters`, `engine`.
* **`rind3d_1k.labels`:**
  * engine `fastlabel` (`run_task`, `supervised_task`), with status `certified_stat` and a certificate in the meta;
  * statistical records are per-cell volumes of B (kind 4).
* **`rind3d_1k.supervise`:** optional per-worker initialiser (the GPU semaphore).
* **`scripts/run_labels.py --engine fastlabel`.**
* **`rind3d_1k.loader`:** `flatten(..., accept_statistical=False)`; labels expose `certificate` / `engine`.
* **Tests:** `tests/test_fastlabel.py` (design α and monotonicity, verifier vs the original renderer, labels vs release
  labels, determinism, task-file round trip in a temporary scene copy) and `tests/test_supervise.py`. All pass.

**Reproduce:**
* `runner3.py 0 15 0 out.jsonl --gpu-permits 1 --scenes <bench_plan.json holdout>`, then `analyze3.py out.jsonl "<summary line>"` and `--strata`;
* accuracy: `--tasks runs/acc_tasks.txt --kw '{"validate_n": 4096}'`.
