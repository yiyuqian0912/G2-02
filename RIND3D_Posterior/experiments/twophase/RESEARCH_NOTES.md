# Cost-reduction research (brief "Full 1K dataset in hours"), checkpoint 2026-11-12 ≈ 13:00 EST

**Status:** paused to work on the RIND3D-100K H20 feasibility task. Resume from here.

## 1. Where the reference engine spends its time (measured)

**Release labels (333, train scenes 0–3; `labels_log.jsonl`):**
* L256 is 92% of CPU. Its cost tracks entry evaluations, with corr(log cpu, log evals) = 0.97.
* Median L256 label: 5.9e8 evaluations over only ≈ 960 nodes.
* Per-label output: 53% of L256 labels are a single 128³ cell; 38% are broad (> 8 cells).

**Depth profiles** (`prof1.py`; windows 000003/45/1 single-cell and 000001/52/1 broad):
* Stage 1 leaves 50 / 152 candidate depth-4 boxes; stage 2 refutes 49 / 139 of them.
* Refutation of those boxes holds ≈ 85% of the evaluations.
* Evaluations per node grow with depth (2.8–3.1 M at depth 3–4), because blocks get expanded into receivers.

## 2. Negative results

| Experiment | Result |
|---|---|
| More stage-1 sentinels (512 → 16384) | Candidates 50 → 49. Single-receiver box tests cannot refute these boxes; refutation needs combined constraints. |
| Deeper stage 1 (`d1` 5 / 6 / 7) | Slower: 96 / 227 / 741 candidates; stage 2 takes 50 / 66 / 94 s instead of 53 s (`sweep_stage1.jsonl`) |
| Phase-1 subsets made of pyramid blocks (`min_level` ≥ 1) | No gain: blocks are expanded back into receivers (`subset_variants.jsonl`) |
| Clearance-shell term | Not a cost driver (12.7 s with it, 12.3 s without, on a broad L64 window) |
| Broad L64 window | Cost is BSP fragmentation (514k nodes, 241k IN pieces, only 5e7 evaluations), not item evaluation |
| Broad L256 posteriors (thin sheets, 2k–9k cells) | Mass is spread evenly: 95% of the mass needs 60–90% of the records. Truncating small pieces would save little. |

## 3. Two-phase exact labeller (`twophase.py`) — exact by construction

**Construction:**
1. **Phase 1.** Run the TPE engine with a *subset* X′ of single-receiver items: 16384 random boundary receivers plus 4096 interior receivers. The subset must consist of level-0 items. This gives B = F(X′) ⊇ F, which is sound for any subset.
2. **Phase 2.** Run the full engine with all receivers on the stage-1 roots. The B-filter is added in `tpl/engine_p2x.py`, an experimental copy: a node whose bounding box misses B's cells or sub-boxes is OUT. This is sound because F ⊆ B.

**Results on 36 stratified windows:**
* Exactness: cell-mass TV vs the reference ≤ 3e-16, and the certified volume ratio is 1.000000 (`bench_bfilter.jsonl`).

| Side | Sum of CPU-s, reference → B-filter | Speedup | Comment |
|---|---|---|---|
| L256 | 1622 → 300 | **5.4×** | Concentrated posteriors 20–35×. Broad thin sheets only 2–3×; 270 of the 300 s. |
| L128 | 427 → 171 | 2.5× | |
| L64 | 155 → 179 | 0.87× | Phase 1 plus stage-1 overhead dominate |

* Weighted by workload this is ≈ 4.6× overall. The target is 24–35×.
* Cover mode (one root per phase-1 piece, `bench_cover.jsonl`) is better for concentrated posteriors and catastrophic for broad ones, because each root re-evaluates all ≈ 255k top items.

**Tightness of phase 1:** vol(F)/vol(B) ≥ 0.99 for most broad windows (L64 and L256 broad: 0.99–1.0). For concentrated windows it is ≪ 1 (B is small but loose), where exact phase 2 is cheap anyway.

## 4. Next steps when resuming

1. **Statistical tier for broad B** (Direction B identity). The label is q_B, the exact cell masses of phase 1. TV(q_B, q_F) = 1 − vol(F)/vol(B). Certify by n uniform samples in B with an exact point test against all receivers: n = 90 with all feasible gives ≥ 0.95 at 99%. Per-label versus dataset-wide simultaneous coverage must be stated separately.
   * A point test costs ≈ one pass over the top items with point-source decisions, ≈ 5–50 ms. It needs a numba point-source checker, or the engine on a degenerate root.
2. **Adaptive subset size** by side and boundary size. For L64, skip phase 1 or use 2–4k points; possibly multi-level X1 ⊂ X2 with B-filters.
3. **Fixed overhead:** stage 1 takes 0.3–1 s, and `top_items` and `point_items` are Python loops. Move them to numba.
4. **GPU:** port the B-filter to `engine_g2` (exact GPU pass) for phase 2 on broad windows.
5. **Physical stage** (Goal B, 6.8 h): materialisation takes 15–21 s per scene (CPU), fields 9–12 s.

---

# Checkpoint 2 (2026-11-12 ≈ 15:40 EST): adaptive constraint generation with a statistical certificate

**Code:** `experiments/fastlabel/` (`adaptive2.py` is current; `adaptive.py` is the earlier version).
**Status:** paused for a machine reboot. Resume from "Next steps" below.

## Method (exact outer bound; statistical certificate only on the volume ratio)

**Construction:**
1. **Phase 1.** Run the TPE engine with a point subset X′ (2048 boundary + 1024 interior receivers). This gives
   B = F(X′) ⊇ F, with exact cell masses q_B.
2. **Paths, in order:**
   * **single-cell support:** if B lies in one 128³ cell, the label is δ_c (TV 0; F ≠ ∅ because the true source
     generated the observation; vol(F) > 0 almost surely);
   * **exact small B:** if B has ≤ 64 pieces, run the exact phase 2 with the B-filter (cheap);
   * **L64 exact race:** a budgeted exact phase 2; kept if it completes;
   * **otherwise:** the statistical tier.
3. **Statistical tier.**
   * **Discovery rounds.** 16 uniform samples in B, each tested EXACTLY: legality (clearance), exclusion from the
     window box, and a certified CUDA render of the window compared bit for bit. Every violated receiver is added to
     X′, and phase 1 is rerun restricted to the current B.
   * **Certification.** On a clean discovery round: n = 112 fresh samples (at most C = 3 attempts, each at level
     δ/3, Bonferroni). Certify iff all pass (one-sided Clopper–Pearson: p = vol(F)/vol(B) ≥ 0.95 at 99%).
   * **Guarantee.** TV(q_B, q_F) ≤ 1 − p ≤ 0.05, and P(false certificate) ≤ δ = 0.01 per label. A dataset-wide
     simultaneous guarantee would need δ/N, e.g. n ≈ 320 for N = 60k statistical-tier labels.
4. **Fallback.** Exact phase 2 on B.

**Exactness-preserving speedups, all verified identical:**
* `fastwin.fast_cls`: numba tiled transpose, 0.99 → 0.16 s at L256;
* lazy block pyramid;
* `fastplanes`: integer shadow planes, 7.3 → 1.7 ms, identical on 100 sources;
* `fastplanes.encode_fast`: 2.84 → 0.98 ms, identical;
* certified tile culling (`tilecull.py`, rpl predicates);
* renderer batch 128.

## Results

**36-window benchmark** (eval36, contended):

| side | speedup vs reference | max TV vs reference |
|---|---|---|
| L256 | 23× | ≤ 0.002 |
| L128 | 4.9× | ≤ 0.004 |
| L64 | 2.0× | ≤ 0.011 |

**Unbiased random sample** (120 tasks, 40 per side, 110 train scenes; refs = TPE v3.1; eval120b.jsonl):

| side | new mean (s) | p95 (s) | ref mean (s) | speedup |
|---|---|---|---|---|
| L64 | 7.3 | 26 | 11.4 | 1.5× |
| L128 | 5.3 | 10.5 | 32.2 | 6.1× |
| L256 | 6.0 | 14 | 150.5 | 25× |

* **1K mix (shares 24 / 38 / 38%):** 6.0 s per label vs 72 s, **12×**. Projection: **≈ 16.7 h** on 16 workers
  (CPU worker time). GPU: ≈ 1,125 tiles per label → 7.5–15 GPU-hours. The 4–6 h target is NOT reached.
* **Paths:**
  * L256: 13 single-cell, 8 exact-small, 19 statistical;
  * L128: 27 statistical;
  * L64: 20 race, 8 exact-small.
* **Accuracy:** every certified label matches its certified reference with TV ≤ 0.015. One reference
  (000195/24/0) was itself uncertified (engine tv_bound 0.42 in 222 s). Our statistical label was validated by
  2000 fresh exact tests: p̂ = 0.9965, 99% CI [0.9915, 0.9990] (`validate_mc.py`).
* **Cost breakdown** (mix, first run):
  * phase 1: 2.1 s;
  * point tests: 1.8 s;
  * exact phase 2: 0.9 s (mostly L64);
  * setup + stage 1 + other: 1.4 s.
* **Open items from the runs:**
  * one task failed in eval120b (000316/15/2: check its error);
  * two tasks of eval120 ran over 27 min and were killed at the reboot (000127/10/3, 000388/10/0).

## Next steps

1. **GPU is the binding constraint for point tests**, at 0.15–0.36 ms per 64³ tile; 32³ tiles are not cheaper per
   voxel enough. The next lever is a dedicated exact batched point-test kernel: many sources × only mixed-tile
   receivers, FP32 filter + FP64/rational fallback, ≈ 1 ms per point instead of ≈ 13–25 ms.
2. **L64 tails** (phase 1 up to 21 s with 5 records; race up to 2.7 s). Try nbd ≈ 512 for L64, or exact-first for
   L64 with a stricter budget.
3. **Per-scene caching** in a production runner: geometry, SceneObs, tester and numba warm-up; setup ≈ 0.5 s →
   ≈ 0.15 s. Also a stage-1 max_tests sweep.
4. **Integration:** a production labeller tier in `rind3d_posterior` with status `certified_stat` (eps, delta, n, k
   in the metadata), plus a regression test against references.
