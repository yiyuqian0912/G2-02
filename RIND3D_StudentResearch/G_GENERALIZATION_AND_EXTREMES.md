# G — Generalisation analysis, and cost-gated support for extreme responses

## 1–2. Generalisation of the mixed-only Student (Experiment B)

(filled in from the pilot evaluation; see §4)

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
| η | — | ≤ 2.0·10⁻⁸ (median 4·10⁻¹⁴) |
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
| 2. adequately certified (TV ≤ 0.05) | yes: 180/180 statistical | yes: 180/180 statistical + η ≤ 2·10⁻⁸; research prototype, not production-reviewed | yes: 100% |
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
