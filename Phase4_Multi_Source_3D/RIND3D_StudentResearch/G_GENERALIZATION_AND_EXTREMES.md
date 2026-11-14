# G — Generalisation analysis, and cost-gated support for extreme responses

## 1. Experiment B: does a mixed-only Student generalise to unseen response types?

**Set-up** (`F_OOD_BENCHMARK.md`, protocol fixed before training):
* Students were trained only on boundary-mixed channels of pilot-train scenes.
* They were evaluated once on natural channels of pilot-eval scenes (the 1K val split) for three held-out response types they never saw: solid-separated, all-lit, all-dark.
* Primary score: information gain over the uniform-legal prior, from the one-hot NLL of the true source. It is a strictly proper score for the true conditional p*(s | X), so it needs no constructed reference.
* Certified full-world references (360 natural extreme channels; 499 solid-separated) add support-mass and adaptivity diagnostics.

### 1.1 Results (four posterior Students; one-hot Students alongside)

| stratum (n) | information gain, nats: FL_B1K s0 / s1, FL_MIS s0 / s1 | one-hot (OH_B1K s0 / s1, OH_MIS s0 / s1) | calibration MAE (FL) | support mass / prior's (FL) | adaptivity ρ(KL(p‖u), KL(q_G‖u)) (FL) |
|---|---|---|---|---|---|
| in-distribution T* (6,321), for reference | 3.77 / 3.88 / 3.98 / 3.96 | 1.14 / 1.38 / 1.60 / 1.88 | 0.03 | 200–270 | 0.63–0.65 |
| **C: solid-separated, release (440)** | **0.71 / 0.75 / 0.76 / 0.66**, all CIs > 0 | −0.25 / −0.18 / −0.09 / −0.91 | 0.03–0.08 | 2.6–2.7 (CIs > 1) | **0.31–0.40, all CIs > 0** |
| C: solid-separated, natural (59) | 0.89 / 0.95 / 1.04 / 0.86, all CIs > 0 | +0.14 / +0.02 / +0.26 / −0.48 | 0.04–0.07 | 3.6–4.0 (CIs > 1) | 0.21–0.30 (CIs include 0; n small) |
| **B: all-dark (2,043; 180 refs)** | **0.20 / 0.24 / 0.26 / 0.25**, all CIs > 0 | −0.64 / −0.74 / −0.36 / −1.36 | 0.03–0.04 | 1.41–1.60 (CIs > 1) | 0.27, 0.20, 0.08, 0.27; 3 of 4 CIs > 0 |
| **A: all-lit (3,000; 180 refs)** | **−0.02 / −0.03 / −0.04 / −0.07** (CIs touch 0 or lie below it) | −0.30 / −1.04 / −0.96 / −1.75 | 0.09–0.13 | 1.8–1.9 (CIs include 1) | 0.09–0.14 (mostly include 0) |

**Information available in each stratum.** The full-world label gives an upper bound on what any Student could extract:

| stratum | KL(q_G ‖ u), mean | recovered by FL Students |
|---|---|---|
| T* | 10.3 nats | 37–39% |
| solid-separated | 1.9–2.0 | 34–53% |
| all-dark | 1.63 | 12–18% |
| all-lit | **0.36** | ≈ 0 |

(`experiments/out/pilot_info_recovered.json`)

### 1.2 Verdict per stratum (the four criteria of F)

| stratum | gain > 0 | calibration ≤ 0.10 | support mass > prior | adaptivity ρ > 0 | verdict |
|---|---|---|---|---|---|
| C: solid-separated | ✓ | ✓ | ✓ | ✓ on the release set (n = 440); n.s. on the natural set (n = 59) | **pass** |
| B: all-dark | ✓ (+0.20–0.26) | ✓ | ✓ (1.4–1.6×) | ✓ in 3 of 4 Students, weak (ρ ≈ 0.2–0.27) | **partial pass**: more than a generic prior, much less than the available information |
| A: all-lit | ✗ (≈ 0, slightly negative) | ✓ for B1K Students (0.09), ✗ for one MIS seed (0.13) | ✗ (CI includes 1) | ✗ | **fail**: prior-like output |
| any stratum, one-hot Students | ✗ (negative on A, B and released C) | ✗ (0.22–0.35) | — | — | **fail**: confidently wrong |

## 2. Interpretation: meaningful geometric reasoning, or a trivial prior?

1. **Solid-separated windows: transferable constraints.**
   * The Students never saw a mixed window without a lit/dark boundary, yet on these windows they recover 34–53% of the full-world information. That is the same share as in-distribution.
   * Their mass on the physically feasible region is 2.6–4× the prior's.
   * Their per-window information content tracks the reference's (ρ 0.3–0.4).
   * What transfers is the use of lit and dark *regions* and of the SOLID crop, not only local boundary patterns. This is the strongest generalisation result of the pilot.
2. **All-dark windows: some real structure, mostly prior.**
   * Gains are positive and calibrated, and the feasible region gets 1.4–1.6× the prior's mass. So the Student does more than output its prior. Consistent with the physics, it shifts mass behind the visible solids.
   * But it recovers only 12–18% of the available information, and adaptivity across windows is weak. It has learnt a coarse "dark means behind something" rule, not the detailed occlusion geometry.
3. **All-lit windows: correctly uninformative, slightly overconfident.**
   * Even the full world says little here: 0.36 nats on average, because all-lit posteriors cover 7–95% of the cube.
   * The marginal over hidden geometry says less still, so a near-prior answer is close to correct. The posterior Students produce exactly that.
   * But they do not distinguish the all-lit windows whose geometry *does* constrain the source (ρ ≈ 0.1). They are mildly overconfident: gain −0.02 to −0.07, coverage MAE 0.09–0.13.
4. **The comparison that matters is with the one-hot Students.**
   * Trained on the same mixed windows, they are confidently wrong on every unseen type, losing up to 1.75 nats against the prior on all-lit.
   * Recognising an uninformative observation and staying near the prior is itself learnt behaviour. Here it comes from posterior supervision, not from the architecture: the architecture and data are identical.
5. **Limits.**
   * One model size, 8k steps, 684 training scenes, two seeds for the main arms.
   * OOD sets come from about 120 pilot-eval scenes; the natural solid-separated set is small (59 channels).
   * References are full-world posteriors. The marginal posterior under hidden geometry was not constructed (no counterfactual ensembles). Proper scores against the true source evaluate the marginal without that construction.
   * Success on solid-separated and partial success on all-dark are evidence of transferable geometric reasoning. They are not proof of a complete physical understanding.

## 3. Can all-lit, all-dark and solid-separated windows be labelled efficiently? (addendum)

### 3.1 Solid-separated mixed windows: already handled by the validated solver

* The generator counts them as informative. The FastLabel point items fall back to interior receivers when there is no lit/dark boundary.
* The exact verifier tests every FREE voxel. Nothing in the certificate depends on a lit/dark boundary.
* All solid-separated channels of the 1K are part of the backfill: 2.6% of informative channels (audit E9). Their coverage and cost are reported in §3.4.
* **No separate solver is needed.**

### 3.2 All-dark: the S6 defect and its fix

**Mechanism** (audit 06 §3, confirmed here):
* A source inside a closed solid sees no receiver, so it is consistent with "all dark".
* Phase 1's outer bound B ⊇ F therefore keeps solid interiors.
* Uniform samples of B that land inside a solid are rejected by the exact verifier as illegal, and they carry no violated receivers.
* The sequential test cannot reach p = vol(F)/vol(B) ≥ 0.95 once the solid interior is more than 5% of B.
* The label then falls back to the exact engine, whose bound is loose for the same reason.

**Fix** (`s6/fastlabel_s6.py`; a research prototype, production code unchanged):

| step | what | why it is exact |
|---|---|---|
| outer bound | B′ = B \ ∪ closed solids | F ⊆ D(G) and D(G) excludes the solids, so F ⊆ B′ |
| test | uniform samples of B; samples inside a closed solid (exact `context.contains`, the predicate of `legal_source`) are discarded and do not count | the accepted samples are i.i.d. Uniform(B′), so the v3.4 two-stage design applies unchanged to p′ = vol(F)/vol(B′) |
| label | q_B′. For each piece: removed if inside a body; kept if separated by a face row; otherwise vol(P) − vol(P ∩ ∪C), with float64 half-space intersections and exact inclusion–exclusion over ≤ 8 bodies | degenerate slivers that qhull cannot handle are kept whole. The label then covers a set ⊇ B′, and the extra mass enters a deterministic term η |
| certificate | TV(q_B′, q_F) ≤ ε + η, except with probability δ (same per-label δ policy) | η is recorded per label: float volume error (generous 1e-12 relative) plus kept slivers |

**Results on the audit's 18 all-dark tasks** (E11 tasks; train-split scenes; `s6/s6_summary.md`):

| | production v3.4 (E11) | S6 prototype |
|---|---|---|
| certified | **9 / 18** (3 uncertified, 6 timeouts at 150 s) | **18 / 18** |
| solid volume removed / vol(B′) | — | 0 to 15% (median 1.1%) |
| η | — | ≤ 2.0·10⁻⁸ (median 4·10⁻¹⁴); in the 180 OOD all-dark references: ≤ 2.8·10⁻⁶ |
| wall per label, under full contention from the backfill | 3–113 s | 17–273 s |

**Independent empirical validation** (`s6/validate_s6.py`; not part of the certificate). On the three labels with the
most solid removed:

| task | removed / vol(B′) | exactly feasible prior points | outside the label support | MC vol(F) / vol(B′) |
|---|---|---|---|---|
| 000234/w5/s2 | 9.8% | 11,144 of 60,000 | **0** | 0.987 ± 0.008 |
| 000014/w2/s3 | 14.8% | 22,549 of 60,000 | **0** | 0.993 ± 0.005 |
| 000257/w45/s0 | 10.6% | 19,897 of 60,000 | **0** | 0.990 ± 0.006 |

So S6 makes all-dark windows certifiable: 53,590 exactly feasible points over three labels, none outside a support. The certificate is consistent with independent Monte Carlo.

### 3.3 All-lit

* All-lit labels certify with the unchanged v3.4 pipeline (audit E11: 18/18).
* Their posteriors are broad: 10⁵–2·10⁶ cells, covering 7–95% of the cube. They are far from the prior.
* Their cost is set by the size of B (phase 1 and sampling over ~10⁶ cells), not by certification difficulty. There is no redundancy shortcut: the receiver constraints are redundant, but the *region* they carve out is large and intricate (holes fixture: cones through openings).

### 3.4 Cost gate (full-machine measurements)

**Natural extreme channels.**
* Sample: the declared random OOD sample, 60 per (side, type), of NAT-1K pilot-eval scenes (`ood/label_ood.py`).
* Run: 12 workers, 09:47 → 10:23, sharing the machine with Student training.
* Comparison: mixed channels from the backfill (15 workers, no training).

| type | engine | certified | worker wall per label, median / mean / p90 (s) | posterior cells (median) | throughput, full machine |
|---|---|---|---|---|---|
| boundary-mixed (backfill) | v3.4 | 100% (159,469) | 0.8 / 1.1 / — | 1–2,080 by side | **10.2 labels/s** |
| solid-separated mixed (backfill) | v3.4 | 100% (3,924) | 11 / 13.4 / — | 3–6·10⁵ | about 1.1 labels/s |
| all-lit L64 / L128 / L256 | v3.4 | **180 / 180** | 24 / 24 / 30; 26 / 29 / 42; 35 / 33 / 46 | 1.5–1.8·10⁶ | **≈ 0.4 labels/s** |
| all-dark L64 / L128 / L256 | S6 prototype | **180 / 180** | 82 / 113 / 269; 88 / 108 / 236; 116 / 121 / 237 | 4–6·10⁵ | **≈ 0.12 labels/s** |
| whole OOD sample | — | 360 / 360 | 71.5 s mean (25,737 worker-s) | — | 360 labels in 36 min |

**Gate, criterion by criterion** (addendum §2):

| criterion | all-lit | all-dark | solid-separated |
|---|---|---|---|
| 1. scientifically correct (properly conditioned Uniform(F) under T*) | yes: the same theorem, X-measurable stratum | yes, with S6 (B′ ⊇ F; Monte Carlo validated) | yes: already in the contract |
| 2. adequately certified (TV ≤ 0.05) | yes: 180/180 statistical | yes: 180/180 statistical + η ≤ 2.8·10⁻⁶; research prototype, not production-reviewed | yes: 100% |
| 3. cost comparable to mixed labels | **no: ≈ 25× per label** | **no: ≈ 80–100× per label** | **no: ≈ 12×** (already paid in the 1K) |
| 4. scalable within the 1K / 100K budgets | **no.** 1K: 25,795 natural-equivalent all-lit channels ≈ 18 h extra. 100K: × ~20 | **no.** 1K: 13,443 channels ≈ 31 h extra | it is 2.5% of labels but about 23% of the 1K label compute. In the 100K, consider X-measurable thinning of this stratum if the budget binds |
| 5. compatible with multi-source sampling | only for channels drawn by an X-measurable rule from *natural* windows (not the release's stored extreme channels, audit M4) | same | yes |

### 3.5 Decision

* **Not added** to the main supervised population of the 1K or 100K.
  * Production labelling of extremes fails the cost gate: criteria 3 and 4, by one to two orders of magnitude.
  * The physical channels stay stored with status `not_computed` (1K: 39,109).
  * The 12-window quota and the backfill contract are unchanged.
* **Kept as an evaluation resource:**
  * the 360 certified natural OOD references (`labels/ood/` in NAT-1K, pilot-eval scenes);
  * the one-hot proper-score evaluation on all 11,153 natural pilot-eval extreme channels.
* **S6** is a validated research prototype. Before any production use it needs:
  * a review of the float half-space volume step; an exact rational path would remove η;
  * a speed-up of the Python solid-removal step (20–90 s per label);
  * a decision on the exact-engine fallback (the prototype has none: unresolved cases are recorded, never invented).
* **Possible future versioned profile.** An optional supplementary labelled extreme set: a few thousand X-measurably sampled natural channels, about 1 GPU-laptop day. It would be a separate, explicitly authorised ablation and is not added to Student training by default.
