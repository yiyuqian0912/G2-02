# A — Research plan: global source inference from local shadows

Branch `yushu/3d-experiment`. This is a working plan. It was revised after the Stage A audit, whose numbers are in §1. Results are in `B_RESULTS.md`.

## 0. Task (unchanged; `RIND3D_StudentResearch/A_TASK_DEFINITION.md`)

**Input.** The Student sees X = (SOLID ∩ W, r_j = R_j|FREE(W), L, o). That is the local geometry, one channel's binary response, and the window side and origin.

**Output.** p(s | X) over the 128³ source cells outside the closed window box.

**Target.** p*(s | X) = E_T*[q_G(s) | X]. This marginalises the hidden world. The certified labels q_G = Uniform(F_W(G)) are per-world posteriors, not targets to be matched one by one.

**Hidden from the Student.** Other channels, K, m, the mask, ids, weights and the label status.

## 1. Stage A audit: the main finding

### 1.1 Shadow posteriors are projective

The lit/dark interface inside the window consists of FREE receiver pairs of opposite response. That interface is a piece of the shadow boundary.

In free space, the shadow boundary of a point source is a cone with its apex at the source. With convex polyhedral occluders it is a union of **planar facets, and every facet plane contains the source**. This holds whether or not the occluder is visible.

The facets are measurable from X. Greedy local RANSAC on the interface midpoints finds them (`slab/facets.py`; it uses no privileged data).

On the dev split, NAT-1K boundary-mixed rows (5,218 rows, 102 scenes):

| facet size (inliers) | facets | true source within 2 voxels of the facet plane | median distance |
|---|---|---|---|
| ≥ 1000 | 11,750 | **95–96%** | **0.04–0.09 voxel**, at a median ~700 voxels from the window |
| 100–999 | 2,121 | 68–79% | 0.5–1.0 voxel |
| 24–99 | 196 | 19% | 8–11 voxels |

Facets per row, mean: 1.4 at L64, 2.0 at L128, 3.4 at L256.

The certified labels agree. Their supports are one cell thick: a plane at L64, a line at L128 (median sd 0.3, 0.3, 15 cells), and a single cell in 56% of L256 labels.

### 1.2 The pilot Student does not use this structure

The pilot Student (FL_MIS_s0, 3.9–4.0 nats information gain) puts 29% of its mass within ±8 voxels of a single-facet window's plane. The label puts 100% there.

**Non-learned, X-only estimators** (dev, side-balanced information gain over the uniform-legal prior):

| estimator | IG (nats) |
|---|---|
| pilot Student | 3.92 |
| **uniform prior × Π_k [ε + exp(−d_k² / 2σ_k²)] over the X-measured facets** | **7.18** |
| Student × facet factor | 7.3 (pooled 8.17) |
| full-world label (privileged upper reference) | 10.28 |

The earlier interpretation (`E_STUDENT_PILOT.md` §2.6) attributed the gap of about 6.4 nats between the Student and the labels to hidden geometry and capacity. **About half of that gap is X-measurable projective information that the Student fails to extract.**

### 1.3 Cause

The pooled 32³ input still fixes the plane to 0.2–0.7 voxel: weighted PCA of the pooled boundary channel on single-facet windows. So the information is present in the input.

What fails is the computation. A plane fitted over 64–256 voxels has to be extrapolated about 700 voxels with sub-cell precision. A CNN plus Fourier-query cross-attention decoder has no operator for this.

## 2. Questions, in priority order

| | question | why it matters | decisive comparison |
|---|---|---|---|
| **Q1** | Does an explicit projective factor, computed from X with learned reliability and width, let a Student recover the X-measurable information? | It tests the central mechanism and is the basis for the other questions | **G0** (generic) vs **F1** (generic + facet factor): same encoder, data, budget and seeds |
| **Q2** | Which part of the projective computation must be built in? | It separates "plane extraction" from "plane extrapolation" as the bottleneck | F1 ablations: fixed facet head (no learning); facet-only (no generic field); generic with 4× steps (does scale fix it?); learned plane extraction from pooled features (slot PCA) |
| **Q3** | Does posterior supervision still beat one-hot once the inductive bias is right? (H1) | The pilot's +2.2–2.6 nats came from memorisation by a generic model | one-hot vs posterior for G0 and F1 |
| **Q4** | How do the models scale with independent worlds? (H4) | Data efficiency of the structured model vs the generic one | scene fractions 1/8, 1/4, 1/2, 1 for G0 and F1 |
| **Q5** | What remains after the projective constraints, and is the model calibrated about it? (H5) | It separates irreducible hidden-world uncertainty from model error | residual vs label information by facet count; along-plane analysis; HPD calibration |
| **Q6** | Does it generalise? (H6) | Facet-free strata (solid-separated, all-lit, all-dark) need other mechanisms | dev OOD strata; window-size transfer |

**Deferred.** Counterfactual hidden-world ensembles (§8 of the brief). The resampled hidden geometry would have to reproduce the observed shadow facets exactly, which is a conditional sampler the generator does not provide. A cheaper partial version is noted in §6.

## 3. Evaluation protocol (fixed before the comparisons)

**Splits.** These are the pilot's hash-declared, scene-disjoint roles:

| split | scenes | use |
|---|---|---|
| train | pilot-train, 684 | training only |
| dev | pilot-select, 102 | every exploratory comparison and OOD development |
| confirm | pilot-eval = 1K val split, 121 | used once, for a small pre-declared set of final comparisons (`--confirm`, logged) |
| 1K test | — | sealed; never opened |

**Primary score.** Held-out information gain IG = −log u(s_true) + log p(s_true), on dev T* rows (NAT-1K natural windows, boundary-mixed). Sides are pooled with equal shares.
* IG is strictly proper for p*. It needs no reference posterior and no privileged data.
* KL(q_G ‖ p) and CE to the labels are secondary diagnostics. Their floor is not zero (§0).

**Also reported:**
* B1K (release windows);
* HPD calibration MAE;
* per side and per facet count;
* OOD strata (solid-separated, all-lit, all-dark) on dev rows, never trained on.

**Uncertainty.**
* 95% scene-bootstrap intervals; all channels of a scene resample together.
* Paired differences on identical rows.
* At least 2 seeds for every decision-relevant arm, and 3 for the main comparison.

**Checkpoint.** Each arm is represented by its final checkpoint (cosine schedule). No checkpoint selection on dev, so dev comparisons carry no selection bias. One-hot arms also report their best dev-curve step, flagged as optimistic.

**Training.** Unless stated otherwise every arm uses the pilot's protocol:
* AdamW 4·10⁻⁴, 8k steps × 32, bf16;
* T*-MIS weights (balance heuristic, release + NAT-1K);
* boundary-mixed training rows only;
* no augmentation.

**Population.** Every arm trains on T* via the MIS weights. No unweighted oversampling.

## 4. Order of work and cost

Laptop RTX 3060, 6 GB.

| stage | runs | GPU time |
|---|---|---|
| A | audit; facet law; non-learned bounds; G0 reproduction; pilot checkpoints re-evaluated | done / ~1 h |
| B | Q1: G0 × 3 seeds, F1 × 3 seeds | ~3.5 h |
| C | Q2 ablations (fixed, facet-only, slot-PCA, G0 4× steps); Q3 one-hot arms | ~4 h |
| D | Q4 learning curves (3 fractions × 2 models × 2 seeds) | ~6 h |
| E | Q5/Q6 analyses (no training); confirmatory evaluation of the final arms on the confirm split | ~1 h |

**CPU.** Facet extraction for all 194,532 rows takes ~2 h on 12 cores, once.

## 5. Risks

* **Facet-extraction errors.** About 4% of large facets miss the source, for example coplanar merges. Mitigation: robust factors with a learned outlier floor ε_k. Errors are measured per facet size.
* **Label-free OOD strata have no facets.** The model falls back to the generic field there. Q6 checks it does not become overconfident.
* **The 1K scale.** 684 training worlds. Results are 1K-scale results and are not presented as 100K-scale ones. The protocol (splits, metrics, models) transfers unchanged.
* **A facet-structured model could exploit artefacts of the voxel staircase.** Facets are computed from the same X at train and test time, and the facet law (§1.1) is physical, not an artefact. The test is generalisation to strata and sides with different facet statistics.

## 6. Ideas held in reserve

* **Lit-cone visibility factor.** Every lit receiver must see the source through the visible window geometry, which gives a hard X-measurable support constraint. It is the only geometric information in all-lit windows.
* **Partial counterfactuals.** Perturb hidden objects that avoid the source's view of the window. X is unchanged and q_G changes, which shows non-identifiability directly. It needs FastLabel runs (~1 s each).

## 7. Pre-declared confirmatory comparisons (written 2026-11-14 ~06:30, before the Q2–Q6 dev results)

Only F1 s0 and G0 s0/s1 had been seen on dev when this was written. The confirm split (1K val) is opened once, for
these comparisons only, with the frozen final checkpoints.

| id | comparison | sets | decision rule |
|---|---|---|---|
| C1 | F1 − G0, both seeds (s0, s1), paired | T*, B1K | primary; confirmed if the scene-bootstrap 95% interval excludes 0 in both seed pairs |
| C2 | P0 (non-learned facets) − G0 s0 | T*, B1K | confirms that X-measurable projective information exceeds what the generic Student extracts |
| C3 | the best dev projective arm (F3 if it beats F1 on dev T* by more than twice the F1 seed spread, else F1) vs G0 | solid-separated, all-lit, all-dark (with the 360 certified OOD references of this split) | reported descriptively: calibration, IG, KL where references exist |
| C4 | G0 s0 − pilot FL_MIS_s0 | T* | reproduction on the pilot's own evaluation scenes |

Everything else (ablations, scaling, generalisation arms) stays dev-only.

**Amendment A1 (written 2026-11-14 ~09:10, before any dev result of F3, F4 or F5 existed; the confirm split still unopened).**
* A new arm, **F5 = F1 + sub-cell quadrature of the facet product**, was added after the failure analysis (`docs/B_RESULTS.md` §9), on dev evidence from F1, G0 and P0 only.
* The best-projective-arm rule of C3 becomes: among F3, F4 and F5, pick the one with the largest dev T* IG.
  * It replaces F1 only if it beats F1 s0 on dev T* by more than twice the F1 seed spread (2 × 0.03 = 0.06 nats). Otherwise F1 is used.
* If the selected arm is not F1, it is also evaluated on confirm T*/B1K (C1′: selected − G0 s0, descriptive), and both C1 seeds stay F1.

**Amendment A2 (written 2026-11-14 ~10:30, after the F5 dev results; the confirm split still unopened).**
* F5 s0 evaluated with M = 4 sub-points at test time ("F5q4") is added to confirm as a **secondary, descriptive** comparison: F5q4 − F5 and F5q4 − F1 on T*/B1K.
* It is dev-selected (+0.12 on dev), so its confirm result is a check of that selection, not a pre-registered test.

**Amendment A3 (written 2026-11-14 ~10:55, after the F3 dev result, before the F4, F1drop and F6 results; the confirm split still unopened).**
* **F6** = F5 + facet dropout (0.2) + availability flag + leave-one-out consensus descriptors, single-hypothesis factor. Motivation: F3 repaired solid-separated windows (+0.28), and quadrature has no occluder branch.
* F6 joins the A1 candidate set: F3, F4, F5, F6.
* Unchanged rule: the arm with the largest dev T* IG, if it beats F1 s0 by more than 0.06 nats, else F1.
* If the selected arm is F6, the A2 secondary comparison uses F6 with M = 4 at test time instead of F5.

## 8. Carrying the protocol to the 100K (no production change proposed here)

* **Facets are a per-channel function of X.** They come from the stored crop and response, so they can be computed for any release without touching the contract.
  * Measured cost: 194,532 rows in 30 min on 11 laptop cores (~0.1 core-s per row; L256 dominates).
  * At ~12 M 100K channels: ~35 core-hours, a small fraction of the label budget.
  * Storage is 8 × 12 float32 per channel (≈ 4.6 GB for 12 M rows), against ~1.5 TB for pooled 4×32³ inputs. The input features, not the facets, are the scaling problem: compute them on the fly.
* **The evaluation protocol is unchanged.** Side-balanced T* IG with scene bootstrap, dev / confirm / sealed test.
  * A 100K run should keep the dev scenes of this study as an anchor set, so that 1K and 100K numbers stay comparable.
* **What the 1K cannot answer** (H4): whether a generic decoder eventually learns the projective operator with ~100× more worlds. The Q4 learning curves give the slope at 86–684 worlds; G0long gives the step budget.
