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
