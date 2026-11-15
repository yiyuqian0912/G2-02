# B — Results: projective structure, projective Students, and what remains

**Status.** Complete for the 2026-11-14 study: all queued arms are run and analysed. Every number is exploratory **dev-split** evidence unless it is marked *confirm*.
* dev = pilot-select, 102 scenes of the 1K train split;
* confirm = the 1K val split, used once (§10);
* the 1K test split is sealed and was never read.

**Conventions.**
* The headline score is side-balanced information gain (IG) over the uniform-legal prior on dev T* rows: NAT-1K natural windows, boundary-mixed, 5,218 rows.
* IG is −log u(s) + log p(s) at the true source. It is strictly proper for p*(s | X) and uses no reference posterior.
* Brackets are 95% scene-bootstrap intervals.
* "Paired" means identical rows.
* The 1K scale (684 training worlds) is not presented as a 100K-scale result.

Run outputs are in `$RIND3D_LAB_OUT/runs/<arm>/`: config, curves, `final.pt`, `eval_dev.npz`. Configurations are in `configs/`.

## 1. Stage A: audit and baseline reproduction

**Reproduction.** The pilot architecture, retrained in the lab with a flat 128³ readout (G0), reproduces the pilot checkpoints on dev:

| arm | T* IG | B1K IG | KL(q_G ‖ p) | HPD-MAE | IG at L64 / L128 / L256 |
|---|---|---|---|---|---|
| pilot FL_MIS s0 (frozen checkpoint) | 3.92 [3.80, 4.05] | 3.87 | 6.36 | 0.046 | 3.61 / 3.93 / 4.22 |
| pilot FL_MIS s1 | 3.89 [3.77, 4.02] | 3.83 | 6.39 | 0.048 | 3.57 / 3.89 / 4.20 |
| **G0 s0** (lab, retrained) | 3.90 [3.78, 4.04] | 3.83 | 6.37 | 0.046 | 3.56 / 3.89 / 4.26 |
| G0 s1 | 3.87 [3.75, 4.01] | 3.80 | 6.41 | 0.041 | 3.56 / 3.86 / 4.21 |

* G0 s0 − pilot s0 (paired): −0.02 [−0.04, +0.01].
* The flat readout and the lab pipeline therefore change nothing. Seed spread is 0.03 nats.
* The full-world labels carry 10.28 nats on the same rows. The pilot's reading of the 6.4-nat KL gap was "hidden geometry plus capacity" (`RIND3D_StudentResearch/E_STUDENT_PILOT.md` §2.6). **The audit refutes it.**

## 2. The projective structure of shadow posteriors

### 2.1 The facet law

**Statement.**
* For a point source, the boundary of the shadow volume of a convex polyhedral occluder is a union of planar facets.
* Each facet lies in the plane through the source and one silhouette edge.
* So every planar piece of the lit/dark FREE interface in a window lies on a plane that **contains the source**, wherever the occluder is: visible, hidden, near or far.

**Extraction from X.**
* Interface points are the midpoints of 6-adjacent lit-FREE/dark-FREE receiver pairs of the full-resolution crop.
* They are segmented into up to 8 planar facets by greedy local RANSAC (`slab/facets.py`, threshold 0.9 voxel).
* Only the SOLID crop and the channel's response are read.

**Synthetic check.** `tests/test_lab.py` uses an analytic box occluder and a point source. The two largest facets pass within 0.1 voxel of the source at ~350 voxels.

**Real dev windows** (`scripts/validate_facets.py`; the true source is used only for scoring):

| facet size (inliers) | L64: median distance / within 2 voxels | L128 | L256 |
|---|---|---|---|
| ≥ 1000 | 0.09 vox / 96% | 0.05 / 95% | 0.04 / 95% |
| 100–999 | 1.0 / 69% | 0.73 / 76% | 0.48 / 80% |
| 24–99 | 10.6 / 19% | 8.0 / 15% | 7.8 / 20% |

* Facets per row: 1.4 / 2.0 / 3.4 at L64 / L128 / L256.
* Median source distance: ~700 voxels.
* The certified labels show the same geometry. Their supports are one cell thick: planes at L64 and lines at L128 (median sd 0.3 / 0.3 / 15 cells), and single cells in 56% of L256 labels.
  * one facet: the source lies on a plane;
  * two facets: on a line;
  * three or more: at a point.

### 2.2 The exception: occluder interfaces (a new phenomenon)

On train-split rows, 3.8% of large facets miss the source by more than 4 voxels (median 123 voxels). The rate depends strongly on family:

| family | thin_slabs | occluder_clusters | the other 7 |
|---|---|---|---|
| large facets violating the law | **17%** | 6% | 0.8–1.9% |

**Mechanism** (verified against the privileged geometry):
* **96% of the violating thin_slabs facets lie on a face plane of a solid** (median offset 0.0 voxel), against 3% of law-obeying facets.
* Thin slabs have half-thickness log-uniform in [2⁻²⁰, 2⁻⁵] of the cube, so about 60% are thinner than half a voxel.
* Voxel-centre sampling cannot see them in the SOLID mask, yet they block light. The lit/dark interface is then **the occluder itself**, with the source in the lit half-space, not a shadow boundary through the source.

**Consequence.** X's "local geometry" is incomplete even inside the window. An interface is one of two physically different objects:
* a **shadow facet**: its plane contains the source;
* an **occluder facet**: the source lies on its lit side.

### 2.3 How much information the projective structure carries (non-learned, X only)

| estimator (dev T*) | IG | KL to labels | HPD-MAE |
|---|---|---|---|
| G0 (generic Student) | 3.90 | 6.37 | 0.046 |
| **P0: uniform-legal × Π_k [ε + exp(−d_k² / 2σ_k²)], fixed ε = 10⁻³, no learning** | **7.20 [6.98, 7.42]** | 3.11 | 0.024 |
| P1: G0 × the same factor, post hoc | 7.42 [7.18, 7.66] | 2.88 | 0.118 (overconfident) |
| full-world labels (privileged) | 10.28 | 0 | — |

**Notation.**
* d_k is the distance from the cell centre to facet plane k.
* σ_k² = (cell)² + s0² + (r σ̂_θ)², where σ̂_θ = rms / (sd₁ √n) comes from the facet's fit.

**Readings.**
* P0 − G0 = +3.30 [3.15, 3.46] nats, with no training.
* The pooled 32³ input still fixes the plane: weighted PCA of the pooled boundary channel on single-facet windows misses the source by 0.2 / 0.3 / 0.7 voxel (L64 / L128 / L256).
* **Diagnosis.** The pilot architecture does not lack information. It lacks an operator that extrapolates a plane fitted over 64–256 voxels to ~700 voxels with sub-cell precision.

## 3. Q1: generic vs projective Student

**F1** is G0 plus the robust plane factor of every X-measured facet:

    log p(s) ∝ generic(s) + Σ_k log(ε_k + (1 − ε_k) exp(−d_k(s)² / 2σ_k(s)²))

* ε_k, s0_k and the angular width are predicted per facet from its descriptors, the other facets and the encoder context.
* The planes themselves are computed from X, never learned.

**Cost.**
* parameters: 2.11 M vs 2.00 M;
* step time 0.188 vs 0.168 s (fused Triton kernel, `slab/plane_triton.py`).

**Training.** Same data, schedule, 8k steps and T*-MIS weights as G0.

| arm | T* IG | B1K IG | KL(q_G ‖ p) | HPD-MAE | IG at L64 / L128 / L256 |
|---|---|---|---|---|---|
| G0 s0 / s1 | 3.90 / 3.87 | 3.83 / 3.80 | 6.37 / 6.41 | 0.046 / 0.041 | 3.56 / 3.89 / 4.26 |
| **F1 s0** | **7.72 [7.48, 7.94]** | 7.73 | **2.58** | 0.053 | 6.08 / 7.36 / 9.71 |
| **F1 s1** | **7.71 [7.48, 7.94]** | 7.74 | 2.57 | 0.055 | 6.09 / 7.35 / 9.70 |

**Paired differences** (F1 − G0 under the same seed):

| | T* | B1K |
|---|---|---|
| seed 0 | **+3.81 [3.67, 3.95]** | +3.90 |
| seed 1 | **+3.84 [3.70, 3.98]** | +3.94 |

* F1 s1 − F1 s0 = −0.00 [−0.02, +0.01].
* F1 − P0 = +0.51 [0.46, 0.57]: the learned head and the generic field together add half a nat over the non-learned factor.
* F1 − P1 (post-hoc product) = +0.29 [0.24, 0.35]: joint training beats a post-hoc combination.

**By facet count** (F1 s0, T* IG; `docs/fig/ig_by_facets.png`):

| facets | 0 | 1 | 2 | 3 | ≥ 4 |
|---|---|---|---|---|---|
| G0 | 0.44 | 2.62 | 4.30 | 4.72 | 5.19 |
| P0 | 0.00 | 4.12 | 7.99 | 10.25 | 11.63 |
| F1 | 0.04 | 4.66 | 8.78 | 10.46 | 11.80 |
| label | 6.4 | 7.1 | 11.0 | 14.0 | 14.5 |

**Readings.**
* G0 saturates near 5 nats whatever the geometry.
* The projective models scale with the number of facets, as the facet law predicts: one facet gives a plane, two a line, three or more a point.

**Training dynamics** (`docs/fig/curves_G0_F1.png`).
* F1 reaches its dev NLL plateau (≈ 6.0) by step 1,000.
* G0 is still descending at step 8,000 (12.9 → 10.4).

**G0long: is the generic Student merely under-trained?** The same G0, with a 4× step budget (32k steps, cosine schedule over 32k):

| arm | steps | T* IG | B1K IG | KL | HPD-MAE (signed) | IG by facets 1 / 2 / 3 / ≥ 4 |
|---|---|---|---|---|---|---|
| G0 | 8k | 3.90 | 3.83 | 6.37 | 0.046 (−0.038) | 2.62 / 4.30 / 4.72 / 5.19 |
| **G0long** | 32k | **4.81 [4.67, 4.95]** | 4.75 | 5.48 | 0.110 (−0.110) | 3.32 / 5.38 / 5.75 / 6.08 |
| F1 | 8k | 7.72 | 7.73 | 2.58 | 0.053 (−0.053) | 4.66 / 8.78 / 10.46 / 11.80 |

* G0long − G0 = +0.91 [0.85, 0.97]: G0 was under-trained.
* But the dev curve flattens (9.60 → 9.54 over the last 6k steps), and F1 − G0long is still **+2.90 [2.77, 3.04]**.
* The extra steps buy sharpness, not correctness. G0long is over-confident: its nominal 50% HPD regions contain the source 34% of the time.
* With 4× compute, the generic decoder recovers under a quarter of the gap. At 1K worlds it does not discover the projective operator by optimisation alone. Whether more *data* does is Q4 (§6).

**Posteriors** (`docs/fig/posteriors_F1_G0.png`; median-IG dev rows, chosen by rule).
* G0 predicts coarse-cell-scale bands.
* F1 reproduces the label's one-cell-thin sheets and lines and a near-point at L256 with 4 facets.
* F1's worst dev row (IG −16) is a thin_slabs window: F1 trusted an occluder-interface facet (§2.2).

**Out of distribution** (dev; never trained on):

| | solid-separated | all-lit | all-dark |
|---|---|---|---|
| F1 − G0, s0 | −0.21 [−0.34, −0.06] | 0.00 | +0.09 |
| F1 − G0, s1 | −0.04 [−0.22, +0.12] | +0.09 | +0.03 |

F1's generic field co-adapts with the facet factor: almost every boundary-mixed training row has facets. On facet-free windows it can be weaker than G0, which is the motivation for F3's facet dropout.

## 4. Q2: what must be built in?

### 4.1 Information vs operator: G1 (the decisive control)

**G1** is G0 plus the same X-measured facets as F1, entered as K extra key/value tokens of every cross-attention block. The facet embedding covers:
* the centroid (Fourier features);
* the normal;
* the plane offset n·μ;
* size and fit quality.

G1 has **the information F1 has, but no geometric decoding operator**.

| arm | T* IG | B1K IG | KL | IG by facets: 1 / 2 / 3 / ≥ 4 |
|---|---|---|---|---|
| G0 s0 | 3.90 | 3.83 | 6.37 | 2.62 / 4.30 / 4.72 / 5.19 |
| **G1 s0** (facet tokens) | **3.81 [3.68, 3.95]** | 3.75 | 6.46 | 2.59 / 4.19 / 4.65 / 4.95 |
| F1 s0 (facet factor) | 7.72 | 7.73 | 2.58 | 4.66 / 8.78 / 10.46 / 11.80 |

**Paired differences:**
* G1 − G0 = **−0.09 [−0.13, −0.06]**;
* F1 − G1 = **+3.91 [3.76, 4.04]**.

**Conclusion.** Giving a generic decoder the exact shadow planes does not help at all. Access to the projective information is not the bottleneck; the operator that turns it into a global constraint is.

* A cross-attention decoder can form logits that are linear in n·s, but its output is a soft average of token values, read out by smooth position features: Fourier frequencies ≤ 8 over the cube, a 16³ coarse grid, and a fine head that is linear in the query embedding.
* A sheet one cell (8 voxels) thick, 700 voxels from the evidence, needs a function that is sharp in d = n·(s − μ). The architecture has no inductive path to it at 1K data scale.
* The explicit factor log(ε + exp(−d²/2σ²)) supplies exactly that function. Its parameters are the X-measured planes.

### 4.2 Which learned parts matter? (explicit-facet family, seed 0, dev T*)

| arm | generic field | facet head | T* IG | Δ vs P0 | KL | HPD-MAE |
|---|---|---|---|---|---|---|
| P0 | none (uniform-legal) | fixed (ε = 10⁻³, s0 = 0.5, κ = 1), no training | 7.20 | — | 3.11 | 0.024 |
| F1only | none | learned | 7.45 [7.22, 7.68] | +0.25 [0.20, 0.29] | 2.86 | 0.033 |
| F1fix | learned jointly | fixed | 7.64 [7.42, 7.86] | +0.44 [0.39, 0.49] | 2.66 | 0.048 |
| F1 | learned jointly | learned | 7.72 | +0.52 | 2.58 | 0.053 |

**Readings.**
* **Geometry carries the result.** P0 − G0 = +3.30 nats with no learning at all. Every learned component together adds +0.5.
* **The generic field adds along-plane information** (+0.27 = F1 − F1only): prior, visibility and sidedness.
* **Learned facet reliability adds +0.08 to +0.25.** It adds more when no generic field can absorb errors.
* With the current descriptors the head cannot recognise occluder-interface facets, which motivates F3 and F4.
* **Calibration.** HPD-MAE rises from 0.024 (P0) to 0.053 (F1), still within the pilot's 0.03–0.05 range.
* (F1fix uses facets with ≥ 24 inliers and P0 those with ≥ 100.)

### 4.3 The cell integral: a non-learned probe (motivated by the failure analysis, §9)

**Probe.** `scripts/quadrature_probe.py`: P0's fixed facet factor, with one change.
* The product of the plane factors is integrated over each 8-voxel cell with m³ midpoint sub-points, each using the residual sub-cell variance (8/m)²/12.
* This replaces evaluating every plane at the cell centre with the full cell variance 8²/12.
* m = 1 reproduces P0 exactly (7.202).
* No training. Dev T*, paired with m = 1:

| stratum | rows | P0 (m = 1) | m = 2 | m = 4 | m = 2 − m = 1 | m = 4 − m = 1 |
|---|---|---|---|---|---|---|
| all | 5,218 | 7.20 | 7.41 | 7.45 | **+0.21 [0.17, 0.24]** | **+0.25 [0.20, 0.30]** |
| 1 large facet, law-obeying | 1,469 | 4.48 | 4.40 | 4.29 | −0.08 | −0.19 |
| 2, law-obeying | 1,266 | 8.64 | 8.72 | 8.63 | +0.08 | −0.01 |
| ≥ 3, law-obeying | 1,726 | 10.93 | 11.55 | 11.84 | **+0.62 [0.54, 0.70]** | **+0.91 [0.81, 1.01]** |
| law-violating | 623 | 5.57 | 5.99 | 6.16 | +0.42 | +0.59 |

**Readings.**
* **The hypothesis of §9 holds.** On point-determined windows, about a third of the 3.5-nat P0-to-label gap is the cell-integral approximation, not missing information.
  * The cell-centre factor inflates every plane independently. For planes that meet at small angles, this smears the product over several cells in depth.
* **On single-plane windows, quadrature with P0's hand-set width loses 0.1–0.2 nats.**
  * The cell inflation had been acting as extra width that the fixed s0 = 0.5 lacks.
  * The learned head of F1 sets its own width, so this cost is not intrinsic.
* **Consequence for the Student.** F1 cannot learn this: its variance floor is the fixed cell term.

**Making it affordable: sparse quadrature.**
* Sub-points are used only in cells where ≥ 2 planes are non-negligible at the centre (0.1–0.6% of cells, in 2–7% of the kernel blocks). The cell-centre formula is kept elsewhere.
* Cost: ~3× the cell-centre kernel at M = 2, against ~20× for full quadrature.
* **Pitfall.** The plane factors are peak-normalised, so kernels of different width give a plane different mass relative to the outlier floor.
  * Switching formulas at the edge of the sparse region then distorts lines: two-plane rows lose 0.12 nats at M = 2 and 0.46 at M = 4, while full quadrature gains there.
  * **Fix:** let both paths approximate the cell *average* of the same function f = ε + (1 − ε) exp(−d²/2v0), with v0 = s0² + r²σθ². A Gaussian smoothed by variance c then carries the amplitude √(v0 / (v0 + c)).
  * Probe with this amplitude-consistent semantics (`1a` = cell-centre with amplitude; same fixed ε, s0, κ):

| stratum | 1a | sparse M = 2 | sparse M = 4 | M = 2 − 1a | M = 4 − 1a |
|---|---|---|---|---|---|
| all | 6.98 | 7.23 | 7.35 | **+0.26 [0.22, 0.29]** | **+0.38 [0.33, 0.42]** |
| 1, law-obeying | 4.15 | 4.15 | 4.15 | 0.00 | 0.00 |
| 2, law-obeying | 8.18 | 8.39 | 8.46 | +0.21 | +0.28 |
| ≥ 3, law-obeying | 10.90 | 11.50 | 11.81 | +0.60 | +0.91 |
| law-violating | 5.86 | 6.13 | 6.21 | +0.27 | +0.35 |

* With consistent semantics, quadrature helps every multi-plane stratum and leaves single planes unchanged (by construction).
* The amplitude itself costs 0.22 nats at P0's *hand-set* ε, because it shifts the plane/floor balance. A learned head re-tunes ε and s0.
* **Arm F5** = F1 + amplitude-consistent sparse quadrature, M = 2 (M = 4 would cost several times more per step).
  * The fused kernels are tested against a torch reference of the same rule, values and gradients, with and without the amplitude.
  * F5 is the only change relative to F1. It was added before any F3–F5 dev result (amendment A1 of the plan).

### 4.4 Can the planes be learned from the Student's own input? (F2)

**F2.** The same generic field as G0, plus a plane operator whose planes are *learned* from the pooled input (4 × 32³ per channel), with no RANSAC facets:
* slot attention over the top 3,072 pooled boundary cells;
* weighted PCA planes (eigh plus shifted power iterations);
* a self-supervised planarity loss computed on X only;
* the same facet head and plane factor as F1.

| arm | planes from | T* IG | KL | IG L64 / L128 / L256 | IG by facet count 1 / 2 / 3 / ≥ 4 |
|---|---|---|---|---|---|
| G0 | — | 3.90 | 6.37 | 3.56 / 3.89 / 4.26 | 2.62 / 4.30 / 4.72 / 5.19 |
| **F2** | pooled input (learned) | **5.62 [5.45, 5.81]** | 4.66 | 5.48 / 5.59 / 5.78 | 4.00 / 6.30 / 6.94 / 6.43 |
| F1 | full-resolution crop (RANSAC) | 7.72 | 2.58 | 6.08 / 7.36 / 9.71 | 4.66 / 8.78 / 10.46 / 11.80 |

**Paired differences:**
* F2 − G0 = **+1.71 [1.60, 1.83]**: the operator alone, on the same input as G0.
* F1 − F2 = **+2.10 [1.95, 2.24]**: plane precision.

Solid-separated: F2 − G0 = −0.16, the same co-adaptation as F1.

**Readings.**
* **The decoding operator is the larger single ingredient.** On G0's own input, a learned plane operator gains 1.7 nats, whereas G1 (the exact planes as tokens, without the operator) loses 0.09.
* **The input representation is the second bottleneck.**
  * The pooled input discretises a 64–256-voxel window into 32³ cells (2–8 voxels per cell).
  * F2 is flat in window size and saturates at ~6.5 nats with ≥ 3 planes. F1 instead rises to 9.7 at L256, where full-resolution facets are most precise (§2.1).
  * Sub-voxel plane geometry must come from full-resolution X. Any 100K Student should either carry X-measured facets (cheap: §8 of the plan) or a full-resolution boundary encoder.
* F2 costs ~2× the step time of F1 (slot attention and PCA). Its dev NLL plateaued at 8.6 from step 6k, so more steps would not close the gap.

### 4.5 F5: the Student with an exact-enough cell integral

**F5** = F1 + the amplitude-consistent sparse quadrature (M = 2) of §4.3. It is the only change, with the same seed, data and steps.

| arm | T* IG | B1K IG | KL | HPD-MAE | IG L64 / L128 / L256 | sep | all-lit | all-dark |
|---|---|---|---|---|---|---|---|---|
| F1 s0 | 7.72 | 7.73 | 2.58 | 0.053 | 6.08 / 7.36 / 9.71 | 0.37 | −0.05 | 0.30 |
| F1 s1 | 7.71 | 7.74 | 2.57 | 0.055 | 6.09 / 7.35 / 9.70 | 0.42 | −0.02 | 0.32 |
| **F5 s0** | **7.97 [7.74, 8.20]** | **7.98** | **2.32** | **0.031** | 6.17 / 7.57 / 10.18 | 0.42 | −0.03 | 0.29 |
| F5 s1 | 7.97 [7.74, 8.20] | 7.96 | 2.32 | 0.032 | 6.17 / 7.56 / 10.18 | 0.33 | −0.07 | 0.29 |

**Paired differences:**
* F5 − F1 = **+0.25 [0.22, 0.29]** on T* and +0.25 [0.22, 0.29] on B1K. The seed difference F1 s0 − s1 is 0.00 [−0.01, 0.02].
* **Second seed:** F5 s1 − F1 s1 = **+0.26 [0.23, 0.29]**, and F5 s1 − F5 s0 = 0.00 [−0.02, 0.02].
* F5 − G0 = **+4.07 [3.92, 4.21]**.
* On the OOD strata, F5 is equal to or slightly better than F1.

**By failure stratum** (F1 → F5; label in brackets):
* point-determined law-obeying rows: 11.00 → **11.62** (14.42);
* two-plane rows: 9.41 → 9.67 (11.15);
* single-plane rows: 5.01 → 5.03 (7.21), unchanged as designed;
* ≥ 3-facet violating rows: 8.73 → 9.04.
* The pooled label gap falls from 2.66 to 2.32 nats per row. The point-determined share falls from 0.93 to 0.70.

**Calibration.**
* Law-obeying rows: HPD-MAE 0.036 → 0.019.
* Point-determined rows are now slightly under-confident (signed +0.025), which leaves room for more sub-points.
* Violating rows stay over-confident (−0.171): quadrature does not address occluder interfaces. F3 and F4 target them.

**More sub-points at test time.** The same F5 checkpoint (trained with M = 2) was evaluated with M = 4 (`slab.evaluate --override '{"quad": 4}' --tag q4`):

| | T* IG | B1K IG | KL | HPD-MAE | IG L64 / L128 / L256 |
|---|---|---|---|---|---|
| F5, M = 2 | 7.97 | 7.98 | 2.32 | 0.031 | 6.17 / 7.57 / 10.18 |
| F5, M = 4 at test time | **8.10 [7.86, 8.34]** | 8.11 | **2.20** | **0.017** | 6.19 / 7.66 / 10.43 |

* The paired gain is +0.12 [0.11, 0.14], so F5 at M = 4 is +0.38 [0.34, 0.42] over F1. The returns diminish: +0.25 from M = 1 to 2, then +0.12 from 2 to 4.
* This is a post-hoc, dev-selected variant. It goes to confirm only as a declared secondary comparison (plan, amendment A2).

**Where the residual is now: degenerate pencils.**
* "≥ 3 large facets" does not guarantee a point. Facets cast by one straight edge all contain the line through that edge and the source: a *pencil* of planes, which fixes a line, not a point.
* Measure: the smallest singular value σ_min of the stacked unit normals of the large facets (0 = all normals perpendicular to one common line).
* Law-obeying ≥ 3-facet dev rows, by σ_min tertile:

| σ_min | rows | label | G0 | F1 | F5 | F5, M = 4 | label − F5 (M = 4) |
|---|---|---|---|---|---|---|---|
| 0.000–0.061 (near-pencil) | 576 | 14.38 | 5.43 | 10.37 | 10.94 | 11.26 | 3.12 |
| 0.061–0.166 | 576 | 14.55 | 5.15 | 11.91 | 12.63 | 13.00 | 1.54 |
| 0.166–0.797 (well-conditioned) | 576 | 14.54 | 5.97 | 12.73 | 13.28 | 13.71 | **0.83** |

* **Where X fixes a point, the X-only Student is within 0.8 nats of the full-world label.** That label knows every hidden body.
* The remaining point-stratum gap sits on near-pencil rows: 30% of the stratum has σ_min < 0.05. There X determines a line, and the label's depth comes from geometry outside the window.
  * This is the regime of the depth counterfactuals (§7). Their floor, 0.15 nats on average over a narrow family, is a lower bound; the true X-only floor on pencil rows is unknown.
* `docs/fig/zoom_F1_F5.png` shows three point-determined dev rows drawn at random (seed 0, not selected on outcome):
  * F5 narrows the posterior streak across the window-to-source direction;
  * on pencil-like rows, mass still spreads *along* it.

**What the head learned.**
* F1's facet head had widened the planes: median s0 = 1.27 voxels, ε ≈ 6·10⁻⁴.
* F5's head makes them sharp: s0 ≈ 10⁻³, so the width now comes only from the fitted angular term; ε ≈ 10⁻⁵.
* With the cell integral done properly, the model no longer needs to blur planes to hedge. This confirms §9's reading that F1's wide kernels were compensating for the cell-centre approximation.

### 4.6 Robust facets: F3 (two-hypothesis factor + facet dropout), targeted at failure mode 3

**F3** = F1 with three changes:
1. Each facet is a mixture: a shadow plane through the source, or an occluder face with the source in its lit half-space (lit side from facets v2), plus the outlier floor.
2. Facet dropout: p = 0.2 per training example.
3. A facet-availability flag for the generic field.

| | T* | B1K | solid-separated | all-lit | all-dark | T* KL | HPD-MAE |
|---|---|---|---|---|---|---|---|
| F1 | 7.72 | 7.73 | 0.37 | −0.05 | 0.30 | 2.58 | 0.053 |
| F3 | 7.75 | 7.78 | **0.64** | −0.01 | 0.27 | 2.54 | 0.049 |
| F3 − F1 (paired) | +0.03 [0.01, 0.05] | +0.05 [0.03, 0.08] | **+0.28 [0.18, 0.37]** | +0.05 [0.02, 0.08] | −0.03 [−0.07, 0.00] | | |

**Breakdown of F3 − F1:**
* law-violating rows (12% of T*): **+0.23 [0.09, 0.35]**;
* law-obeying rows: +0.01 [−0.01, 0.02];
* by family: thin_slabs +0.12 [0.01, 0.22], occluder_clusters +0.07 [0.02, 0.10].

**Readings.**
* **Solid-separated windows are repaired.** F3 scores 0.64 there, against 0.37 for F1 and 0.57 for G0. The projective Student no longer pays for its facet factor on facet-free windows.
  * The single-change arm F1drop (dropout + flag only, no occluder hypothesis) shows that the repair comes entirely from dropout: see the attribution below.
* **Occluder interfaces are only partly handled.** F3 recovers 0.23 of the ~4–6-nat label gap on violating rows. Its calibration there does not improve: signed HPD −0.191 vs −0.184.
  * The mixture lets a facet be "either", but with only size, fit and lit-side descriptors the head cannot tell which. It hedges a little and stays confidently wrong when it chooses the shadow reading.
  * The slab face behind an occluder interface is itself invisible in X (§2.2), so the distinction must come from evidence beyond the facet:
    * agreement with the other facets' common intersection (F4: leave-one-out consensus; violators have median z 4.0 against 0.06);
    * or a visible solid edge lying on the plane inside the window (a visible caster), which is not yet a descriptor.
* By the pre-declared rule, F3 does not replace F1 on T* (+0.03 < 0.06). F5 remains the best projective arm (F5 − F3 = +0.22 [0.19, 0.26]).

**F4** = F3 + leave-one-out consensus descriptors, i.e. each facet's agreement with the least-squares intersection of the others: log1p of the residual, log1p of its variance, and a z score.

| | T* | B1K | sep | all-lit | T* KL | HPD-MAE | violating rows, signed HPD |
|---|---|---|---|---|---|---|---|
| F3 | 7.75 | 7.78 | 0.64 | −0.01 | 2.54 | 0.049 | −0.191 |
| F4 | 7.77 | 7.82 | 0.58 | 0.01 | 2.52 | 0.054 | −0.201 |

**Paired differences:**
* F4 − F3 = +0.03 [0.01, 0.04] on T*;
* on violating rows, **+0.12 [0.04, 0.19]**: +0.23 where < 3 large facets, +0.11 where ≥ 3;
* on law-obeying rows, +0.02;
* F4 − F1 = +0.06 [0.03, 0.09];
* F5 − F4 = +0.20 [0.16, 0.23].

**Reading (failure mode 3).**
* Together, the occluder hypothesis and the consensus statistic recover ~0.35 nats on law-violating rows, out of a 4–6-nat gap.
* Neither reduces the over-confidence there.
* The X-measurable evidence that a facet is an occluder face exists (§2.2 consensus z), but the learned head does not turn it into calibrated doubt at this data scale. Per facet, the training signal is weak: only 3.8% of large training facets violate the law (§2.2).
* This failure mode stays open (§11).

**Attribution by single-change arms** (paired, dev; F1 → F1drop → F3 → F4):

| change | T* | solid-separated | law-violating T* rows | law-obeying T* rows |
|---|---|---|---|---|
| + facet dropout 0.2 and availability flag (F1drop − F1) | +0.02 [0.00, 0.04] | **+0.27 [0.17, 0.36]** | **+0.16 [0.04, 0.28]** | +0.01 |
| + two-hypothesis factor (F3 − F1drop) | +0.01 [−0.01, 0.02] | +0.01 | +0.07 [−0.00, 0.14] | 0.00 |
| + consensus descriptors (F4 − F3) | +0.03 [0.01, 0.04] | −0.06 | **+0.12 [0.04, 0.19]** | +0.02 |

* Dropout with an availability flag is the cheap, decisive fix for facet-free windows. It also helps on violating rows: the generic field learns not to lean on facets.
* The occluder branch adds little (+0.07, borderline); consensus adds +0.12 on violating rows.
* F6 (amendment A3, declared before these results) combines F5 with dropout, flag and consensus, without the occluder branch.



### 4.7 F6 (F5 + dropout, flag, consensus) and the confirm selection

| | T* | B1K | sep | all-lit | all-dark | T* KL | HPD-MAE |
|---|---|---|---|---|---|---|---|
| F5 | 7.97 | 7.98 | 0.42 | −0.03 | 0.29 | 2.32 | 0.031 |
| F6 | 7.96 [7.73, 8.19] | 7.96 | 0.39 | −0.03 | 0.26 | 2.33 | 0.039 |

**Paired F6 − F5:**
* T* −0.01 [−0.03, 0.01];
* violating rows +0.12 [−0.02, 0.28];
* law-obeying rows −0.02 [−0.04, −0.01].

**Readings.**
* Combining the components does not add up. Consensus and dropout bring their small violating-row gain, but cost a little on law-obeying rows.
* **Unresolved: dropout did not repair facet-free windows under quadrature.**
  * Solid-separated rows have no facets at all (0 of 486 dev rows), so only the generic field acts there.
  * With the cell-centre factor, dropout + flag lifted them from 0.37–0.42 (F1 seeds) to 0.58–0.64 (F1drop, F3, F4). With quadrature, F6 stays at 0.39.
  * The seed spread on this small set is ~0.1 nats (G0 s0/s1: 0.57/0.46). The cause is untested.

**Confirm selection (amendments A1/A3, applied mechanically).**
* Dev T* IG of the candidates: F3 7.75, F4 7.77, **F5 7.97**, F6 7.96.
* F5 is the largest and beats F1 s0 by 0.25 > 0.06. **The best projective arm is F5_s0**, and its M = 4 test-time variant is the A2 secondary comparison.
* Recorded before the confirm split was opened.

## 5. Q3 (H1): does posterior supervision still matter once the inductive bias is right?

Same architectures, same rows and MIS weights, 8k steps, seed 0. The only change is the target: a one-hot at the true
source cell ("OH") instead of the certified q_G. `best.pt` is the step with the lowest true-source NLL on 1,024 of
the 5,218 dev T* rows. On the 4,194 rows that did not take part in that selection, the paired differences are
unchanged to ±0.01, so selection bias is negligible.

| arm | target | checkpoint | T* IG | sep IG | all-lit IG | all-dark IG | T* KL | HPD-MAE |
|---|---|---|---|---|---|---|---|---|
| G0 | posterior | final | 3.90 | 0.57 | −0.05 | 0.22 | 6.37 | 0.046 |
| OHG0 | one-hot | final | 0.75 | −2.14 | −2.91 | −2.28 | 9.54 | 0.456 |
| OHG0 | one-hot | best (step 2k) | 1.37 | −0.70 | −1.06 | −0.47 | 8.89 | 0.278 |
| F1 | posterior | final | 7.72 | 0.37 | −0.05 | 0.30 | 2.58 | 0.053 |
| OHF1 | one-hot | final | 5.14 | −2.54 | −3.03 | −2.53 | 5.09 | 0.393 |
| OHF1 | one-hot | best (step 1k) | 7.23 | −0.44 | −0.62 | −0.06 | 3.04 | 0.134 |

| paired T* IG difference | final | best one-hot checkpoint |
|---|---|---|
| G0 − OHG0 | 3.16 [2.92, 3.38] | 2.53 [2.41, 2.67] |
| F1 − OHF1 | 2.57 [2.35, 2.79] | **0.48 [0.39, 0.59]** |

**Readings.**
* **H1 holds under both inductive biases.** Posterior supervision wins on every set, at every checkpoint.
* **Its size depends on the architecture.** With early stopping the gap falls from 2.5 nats (G0) to 0.5 (F1). The projective operator supplies most of what the dense label was teaching.
* **One-hot training memorises.** OHF1 train loss falls to ~2 nats while dev NLL rises from 6.4 (step 1k) to 8.3 (step 8k). OHG0 behaves the same way from step 2k on.
  * There are 684 training worlds with one source each. Many windows share a source, so a one-hot target lets the network learn the world → source map.
  * The certified posterior q_G is the conditional expectation of the one-hot target given the full world and the observation, i.e. a Rao–Blackwellised target. Since p*(s | X) = E[q_G | X], both targets have the same population optimum, but q_G removes the source-sampling noise that one-hot training fits.
* **Early-stopped one-hot F1 reaches exactly what geometry alone gives.** OHF1-best scores 7.23 against P0's 7.20, the non-learned facet estimator. Under one-hot supervision, the learned parts add nothing beyond the X-measured planes before they start to overfit. Posterior labels are what make the learned along-plane field (+0.27, §4.2) and the reliability head learnable at this scale.
* **Calibration and OOD are where one-hot hurts most.**
  * Even the best OHF1 checkpoint has HPD-MAE 0.134, against 0.053.
  * It scores negative IG on solid-separated and all-lit windows, i.e. worse than the uniform-legal prior.
  * One-hot gradients reward sharp mass at the source. Nothing in them teaches the model to stay diffuse where X is uninformative.
* **Scale caveat.** At 100K worlds the one-hot noise averages over ~150× more sources. The early-stopped gap would be expected to shrink further, but this cannot be tested here. The data-scaling arms (§6) give the local slope.

## 6. Q4 (H4): how do the models scale with independent worlds?

**Design.**
* G0 and F1 were trained on 1/8, 1/4 and 1/2 of the 684 training scenes: 86, 171 and 342 scenes, with 14,781, 30,489 and 61,070 pooled rows.
* At each fraction both models see the same scenes. Each fraction is a separate seeded draw, so the subsets are **not nested**.
* Everything else matches §3: 8k steps × 32, seed 0, MIS weights, dev T*.
* **Deviation from the plan:** one seed per point, not two. Scene-bootstrap bands are reported.

**Which checkpoint.** At small fractions F1 overfits after ~1k steps: F1sc125's dev NLL rises from 6.21 at step 1k to 6.49 at step 8k. Scoring the final step would therefore mix "less data" with "more passes over it".
* Each point is the checkpoint with the lowest dev NLL. As in §5, that selection uses 1,024 of the 5,218 dev T* rows.
* For G0_s0 and F1_s0 the final step is that minimum: G0_s0 has no `best.pt` but its dev curve falls monotonically, and F1_s0's `best.pt` scores identically to its final checkpoint.
* With the final checkpoints instead, F1 would be 7.10 / 7.41 / 7.59 / 7.72 nats: early stopping recovers 0.34 nats at 1/8.

| training scenes | G0 T* IG | F1 T* IG | F1 − G0 | G0 HPD-MAE | F1 HPD-MAE |
|---|---|---|---|---|---|
| 86 (1/8) | 1.69 [1.56, 1.84] | 7.45 [7.19, 7.69] | 5.76 [5.53, 5.98] | 0.340 | 0.092 |
| 171 (1/4) | 2.71 [2.56, 2.86] | 7.50 [7.26, 7.74] | 4.78 [4.60, 4.97] | 0.287 | 0.088 |
| 342 (1/2) | 3.47 [3.35, 3.60] | 7.63 [7.39, 7.86] | 4.15 [3.99, 4.31] | 0.130 | 0.035 |
| 684 (all) | 3.90 [3.78, 4.04] | 7.72 [7.48, 7.94] | 3.81 [3.67, 3.95] | 0.046 | 0.053 |

| gain per doubling (paired) | 86 → 171 | 171 → 342 | 342 → 684 |
|---|---|---|---|
| G0 | +1.02 [0.89, 1.16] | +0.76 [0.65, 0.88] | +0.43 [0.37, 0.49] |
| F1 | +0.05 [−0.03, 0.13] | +0.13 [0.07, 0.19] | +0.09 [0.06, 0.13] |

Paired across fractions means on identical dev rows. The training subsets differ, so the uncertainty from the subset draw is not in these intervals.

![dev T* IG vs training worlds](fig/scaling.png)

**Readings.**
* **H4 holds: the projective Student is far more data-efficient.**
  * With 86 worlds, F1 already scores 7.45 nats: 96% of its full-data score, and 1.9× G0 trained on all 684.
  * Over an 8× increase in data, F1 gains 0.27 nats and G0 gains 2.21.
  * The projective operator is built in, so the data only has to teach the reliability head and the along-plane field.
* **G0's returns diminish at a fixed step budget.** Its gain per doubling falls 1.02 → 0.76 → 0.43, and the F1 − G0 gap shrinks by 0.98, then 0.63, then 0.34.
  * A log-linear extrapolation of G0 to F1's level is not supported by these points.
  * The slope is also confounded with compute. At 684 scenes, 8k × 32 samples are ~2 passes over the pool, and G0long (§3) gains +0.91 from 4× more steps at the same data. Part of the flattening is under-training, not data saturation.
  * What these points do show: at fixed compute, more worlds do not make the generic decoder discover the projective operator at 1K scale. Whether ~100× more worlds *and* matched compute would close the gap cannot be tested here.
* **G0 is also much worse calibrated at small data.** HPD-MAE is 0.34 at 86 scenes against 0.05 at 684, because G0 memorises the few training sources. F1's calibration error stays below 0.1 at every fraction.
* **Large windows show the difference most clearly.** At 1/8, G0's L256 IG (0.95) is below its L64 IG (2.11), while F1's L256 IG is its highest (9.39). Large windows carry the most plane information (§2.3). A generic decoder needs data to use it; the projective factor uses it from the start.
* **The facet-free weakness does not depend on data.** Solid-separated F1 − G0, final checkpoints: −0.16 [−0.52, 0.28], −0.17 [−0.36, 0.01], −0.15 [−0.27, −0.04] and −0.21 [−0.34, −0.06] at 1/8 → 1. The co-adaptation of the generic field (§3) is structural, not a small-data effect.
* **Scale caveat.** These are 86–684-world results on the 1K release, one seed per point. They give the local slope at 1K scale, not 100K-scale behaviour.

## 7. Q5: hidden worlds with an identical observation (counterfactuals, brief §8)

Two exact constructions on dev rows (`scripts/counterfactual.py`, `scripts/counterfactual_depth.py`). Every edited world
is labelled with the production FastLabel v3.4 pipeline, and the true source is in every counterfactual support (100%).

**(a) Remove every body disjoint from H = conv({s} ∪ window box)** (2-voxel margin). By construction X is unchanged
(crop and response). 240 rows; 7 with nothing removable; 233 labelled.

| | L64 | L128 | L256 | all |
|---|---|---|---|---|
| TV(q_G, q_G′): median / mean | 4·10⁻⁴ / 0.034 | 10⁻⁸ / 0.064 | 0 / 0.018 | 10⁻⁸ / 0.039 |
| JSD(q_G, q_G′), mean (nats) | 0.013 | 0.027 | 0.009 | 0.016 |
| pairs with TV > 0.5 | 0% | 5% | 3% | 2.6% |

Geometry outside the source–window hull is practically irrelevant to the full-world posterior: 90% of pairs agree
within the 0.05 certificate tolerance. The pairwise floor of any X-only Student is about 0.02 nats, against
pair-averaged KL of 2.70 (F1), 3.27 (P0) and 6.52 (G0).

**(b) Scale every hidden caster about the source**, O′ = s + λ(O − s) with λ ∈ {0.6, 0.8, 1.25, 1.6}. The shadow
cone from s is invariant, but the hidden depth changes. A world is accepted only if:
* O′ stays outside the box and inside the cube;
* s stays legal;
* **the exact FastLabel verifier confirms that s reproduces the stored observation on every FREE receiver**. It
  rejected one candidate whose response changed on 58 voxels.

Counts:
* 120 rows; 34 have no hidden caster (every shadow comes from visible bodies, 16 of 40 at L256);
* 231 accepted worlds;
* 86 rows with a set of 3.7 X-identical worlds on average.

| | L64 | L128 | L256 | all |
|---|---|---|---|---|
| mean entropy range across the set (nats) | 1.05 | 0.59 | 0.48 | 0.74 |
| floor: generalised JSD of the set, mean / median | 0.21 / 0.24 | 0.11 / 0.04 | 0.12 / 0.00 | **0.15 / 0.07** |
| G0: mean KL to the set | 5.00 | 6.06 | 8.18 | 6.24 |
| F1: mean KL to the set = floor + KL(mixture ‖ p) | 2.28 | 2.66 | 2.54 | **2.48 = 0.15 + 2.33** |

**Readings.**
* Hidden caster depth is a real, irreducible source of uncertainty: the same X admits posteriors whose entropy differs by up to ~1 nat (`docs/fig/counterfactual_depth_max.png`).
* F1's single prediction covers the whole family, a band on the X-determined plane.
* In this tested range the floor is small (0.15 nats) next to F1's excess (2.33). The family is a lower bound on the full set of X-equivalent worlds (caster shapes, larger depth ranges), so the true floor is higher.

**Where the residual is X-measurable.**
* Windows with ≥ 3 large facets (2,050 dev rows): the least-squares intersection of the X-measured planes lies a median 1.4–1.7 voxels from the true source, and in the source's own 8-voxel label cell in 52–58% of rows.
* There the label carries 14.0–14.4 nats and F1 11.3–12.0.

**Conclusion.** Much of F1's remaining 2.3–2.6-nat KL is model error, not hidden-world uncertainty:
* wide kernels hedge against unreliable facets;
* the cell integral is approximate.

This motivates F3 (occluder hypothesis) and F4 (consensus).

## 8. Q6 (H6): does the Student generalise?

Three kinds of shift, all on dev. The OOD sets are never trained on.
* (a) A window size never seen in training.
* (b) Scene families never seen in training.
* (c) The facet-free hard cases. These are summarised from §3, §6 and §10.

Arms (a) and (b) are retrained copies of G0 and F1: same budget, seed 0, final checkpoints. Their only difference is the training-row filter.

### 8.1 Window-size transfer: trained on L64 + L128 only, evaluated on L256

* The side arms G0side and F1side train on 72,868 rows of all 684 scenes, against 124,502 for G0 and F1.
* Both models carry a learned embedding of the window-size bucket (L64 / L128 / L256: `side_emb`, plus a one-hot in the decoder). In these arms the L256 bucket receives no gradient.
* The diagnostic `side_bucket_max = 1` is applied only at evaluation, on dev. It makes L256 windows reuse the L128 bucket. As a check, the L64 and L128 rows are bit-identical with and without it.

| dev T*, L256 rows (n = 2,920) | IG | vs the same model trained on all sides | HPD-MAE | 50% HPD coverage − 0.5 |
|---|---|---|---|---|
| G0 (all sides) | 4.26 [4.10, 4.42] | — | 0.057 | −0.070 |
| G0side | 2.64 [2.54, 2.75] | −1.62 [−1.72, −1.52] | 0.147 | −0.200 |
| G0side, L256 bucket → L128 | 3.30 [3.16, 3.43] | −0.97 [−1.03, −0.91] | 0.039 | +0.049 |
| F1 (all sides) | 9.71 [9.43, 9.99] | — | 0.061 | −0.080 |
| F1side | 9.70 [9.43, 9.97] | −0.01 [−0.05, 0.03] | 0.021 | +0.023 |
| F1side, L256 bucket → L128 | 9.76 [9.48, 10.03] | +0.05 [0.02, 0.09] | 0.013 | −0.022 |

On the trained sides (L64 + L128, n = 2,298) the side arms match the full arms: G0side − G0 +0.07 [0.02, 0.12], F1side − F1 +0.03 [0.00, 0.06].

**Readings.**
* **The projective Student transfers to an unseen window size with no loss**, and is better calibrated there than when trained on it. This is expected if the transferable part is the plane factor: a plane through the source is the same object at every window size. The learned reliability head only has to rate a facet's fit quality, which does not depend on the window size.
* **The generic Student does not transfer.**
  * Without L256 training it loses 1.62 nats at L256 and becomes over-confident.
  * Two-fifths of that loss (0.65) is the untrained size embedding.
  * The remaining 0.97 nats is a real failure to extrapolate: the encoder sees an L256 window at 8 voxels per pooled cell, a scale it never saw.
  * With the clamp, G0side at L256 (3.30) is below its own L128 score (3.88), while F1side at L256 (9.76) is well above its L128 score (7.40).
* **The generic field transfers poorly for both models.** On facet-free all-lit L256 windows (n = 485), both side arms lose 0.16–0.22 nats against their all-sides counterparts, clamp or not. F1's size transfer therefore comes from the plane factor, not from its generic field.

### 8.2 Family transfer: corridors and thin_slabs held out of training

**Design.**
* G0fam and F1fam train on the other seven families: 521 of 684 scenes, 92,191 rows.
* Of the 5,218 dev T* rows, 497 are corridors and 691 are thin_slabs. These are the two hardest families for both full models (F1: 6.97 and 6.48 nats, against 8.03 on the rest).
* thin_slabs is the main source of occluder-interface facets (§2.2). Holding it out tests whether the exception to the facet law has to be seen in training.

| dev T* stratum | n | G0fam − G0 | F1fam − F1 | F1fam IG | HPD-MAE F1fam / F1 |
|---|---|---|---|---|---|
| other families (trained) | 4,030 | −0.01 [−0.05, 0.03] | −0.07 [−0.12, −0.03] | 7.95 | 0.067 / 0.039 |
| corridors (held out) | 497 | +0.06 [0.02, 0.10] | +0.05 [−0.04, 0.15] | 7.01 | 0.060 / 0.052 |
| thin_slabs (held out) | 691 | **−0.40 [−0.52, −0.24]** | **−0.23 [−0.34, −0.10]** | 6.25 | 0.160 / 0.134 |

The table below splits the held-out rows by whether any large facet (≥ 100 inliers) misses the true source by more than 4 voxels. This is the §9 stratification; the true source is used only to stratify.

| held-out stratum | n | G0fam − G0 | F1fam − F1 | F1 IG (trained on it) |
|---|---|---|---|---|
| thin_slabs, every large facet obeys | 452 | −0.21 [−0.37, −0.08] | −0.08 [−0.16, −0.01] | 7.05 |
| thin_slabs, violating | 228 | −0.82 [−1.18, −0.31] | **−0.65 [−0.81, −0.42]** | 5.44 |
| corridors, every large facet obeys | 438 | +0.04 [−0.05, 0.11] | +0.05 [−0.03, 0.15] | 7.30 |
| corridors, violating | 43 | +0.63 [−0.01, 1.55] | −0.04 [−0.26, 0.09] | 5.74 |

**Readings.**
* **Where the facet law holds, the projective Student transfers to unseen families.** On held-out corridors F1 loses nothing. On thin_slabs rows whose large facets all obey the law it loses 0.08 nats, close to the −0.07 on the trained families. That is about what 26% fewer training rows would cost at the §6 slope of ~0.1 nats per doubling.
* **The exception has to be seen in training.**
  * 33% of thin_slabs rows carry a law-violating facet. Without thin_slabs worlds in training, F1 loses 0.65 nats on those rows, and its calibration error on the family rises from 0.134 to 0.160.
  * The reliability head learns to distrust occluder-interface facets from examples of them. A Student trained on families with few sub-voxel occluders will be over-confident on families with many.
  * The family composition of a training set therefore matters through the rate of occluder interfaces, not through the plane geometry.
* **The generic Student behaves the same way, with larger losses:** −0.40 on thin_slabs, −0.82 on its violating rows. Even untrained on them, F1fam scores 6.25 nats on thin_slabs, about twice G0 trained on them (3.20).
* **Calibration degrades for both models on every stratum,** including the trained families: HPD-MAE 0.077 vs 0.042 for G0 and 0.067 vs 0.039 for F1. This matches the small-data trend of §6 and is not specific to the held-out families.

### 8.3 The facet-free hard cases, and the position-only control (Gblind)

**Gblind** is G0 with the window content zeroed: the same architecture and capacity, seeing only the window's origin and size. It measures how much a Student can get from *where* a window is, without looking at it.

| dev | T* | solid-separated | all-lit | all-dark |
|---|---|---|---|---|
| Gblind IG | −0.07 [−0.11, −0.03] | −0.22 [−0.28, −0.15] | −0.29 [−0.33, −0.24] | +0.08 [0.00, 0.18] |
| G0 − Gblind | +3.97 [3.85, 4.11] | +0.79 [0.60, 0.98] | +0.24 [0.18, 0.28] | +0.13 [−0.02, 0.27] |
| F5 − Gblind | +8.04 [7.81, 8.27] | +0.64 [0.47, 0.84] | +0.26 [0.21, 0.31] | +0.21 [0.08, 0.32] |
| label IG (confirm, §10) | — | 1.90 | 0.36 | 1.63 |

**Readings.**
* **Position alone carries no transferable information.** Gblind scores slightly *below* the uniform-legal prior, and its dev NLL rises from 14.55 to 14.64 during training: it fits the positions of the 684 training sources, and those do not carry over to new worlds. The uniform-legal prior is therefore the right position-only baseline, and every IG in this report comes from the window content.
* **On facet-free windows the generic field extracts something, but little.** G0 gets 0.13–0.79 nats beyond position on the three hard sets, F5 0.21–0.64. The full-world labels hold 0.36–1.90 nats there (confirm). How much of that is X-measurable is unknown (§10).
* **The projective Students' solid-separated deficit is stable.**
  * It is −0.21 on dev (s0), −0.44 on confirm (F5), and the same at every training fraction (§6).
  * The facet factor is idle on these windows. The generic field co-adapted with it during training and is weaker than G0's alone.
  * Facet dropout repaired this only with the cell-centre factor (F1drop, §4.6).

**Verdict on H6.** H6 is supported where the facet law holds, and not supported elsewhere.
* **Where the facet law holds, the projective Student generalises.**
  * Unseen window size: no loss, better calibration.
  * Unseen families: no loss on law-obeying rows.
* **The occluder-interface exception does not generalise:** −0.65 nats on violating rows of an unseen family.
* **Neither model generalises on facet-free windows:** at most 0.8 nats beyond position, against 0.4–1.9 in the labels.
* **The generic Student generalises worse on every axis:** −0.97 nats on the unseen size after removing the embedding confound, and −0.40 on thin_slabs.

## 9. Failure analysis (deliverable E): where the remaining 2.7 nats go

Run with `scripts/failure_analysis.py` on dev T*. The true source is used only to *score* facets.
* A row is **law-violating** when some large facet (≥ 100 inliers) misses the true source by more than 4 voxels.
* Law-violating rows are 11.9% of all rows, but 33% of thin_slabs and 23% of occluder_clusters rows (cf. §2.2).

| stratum (large facets) | rows | label IG | G0 | P0 | F1 | label − F1 | share of the pooled gap |
|---|---|---|---|---|---|---|---|
| 0 | 2.6% | 7.11 | 0.66 | 0.00 | 1.71 | 5.37 | 0.14 |
| 1, law-obeying | 28.2% | 7.21 | 2.82 | 4.48 | 5.01 | 2.21 | 0.62 |
| 2, law-obeying | 24.3% | 11.15 | 4.81 | 8.64 | 9.41 | 1.86 | 0.45 |
| ≥ 3, law-obeying (point-determined) | 33.1% | 14.42 | 5.56 | 10.93 | 11.00 | 2.80 | **0.93** |
| 1, violating | 1.7% | 7.12 | 0.29 | 1.17 | 2.05 | 5.04 | 0.09 |
| 2, violating | 2.3% | 9.37 | 2.06 | 3.68 | 3.70 | 5.86 | 0.14 |
| ≥ 3, violating | 7.9% | 13.08 | 3.75 | 7.91 | 8.73 | 3.75 | 0.30 |

IG is side-balanced within each stratum. The gap shares are pooled means per dev T* row; they sum to 2.66.

**Three failure modes, in order of size.**
1. **Precision on point-determined windows (0.93 nats, 35% of the gap).**
   * With ≥ 3 law-obeying planes, X determines the source to ~1.5 voxels (§7). The label is nearly a one-cell delta (14.4 nats), yet P0 and F1 both stop at ~11.
   * Learning adds nothing over the fixed factor here (F1 − P0 = +0.07), and F1 is well calibrated (signed HPD −0.018). It is precise in the wrong way: it spreads mass over a few cells along the depth direction, rather than placing it in the wrong place.
   * Hypothesis: the cell-integral approximation.
     * The window subtends ~0.1 rad at the source, so the planes meet at small angles. Depth is ill-conditioned by a factor of ~10 relative to the in-plane directions.
     * The cell-centre factor inflates *every* plane's variance by the cell term 8²/12, independently. Multiplying the inflated planes spreads the product over several cells in depth.
     * The exact cell integral of the product does not have this problem → probe below, and arm F5.
2. **Plane-only and line-only windows (1.07 nats).** One plane leaves a 2-D sheet and two planes a 1-D line. Here the label's extra information is along-plane: hidden geometry, visibility and prior. The generic field contributes +0.5 over P0 in these strata. This is the part where more data and a better generic field should help (Q4).
3. **Occluder interfaces and mis-fits (0.53 nats, 11.9% of rows).**
   * On violating rows F1 trails the label by 3.8–5.9 nats.
   * There it is strongly *over-confident*: signed HPD −0.184, against −0.035 on law-obeying rows. It places a sharp sheet on a plane that does not contain the source.
   * The facet head cannot recognise these facets from size and fit descriptors. F3 (two-hypothesis factor) and F4 (consensus) target exactly this.

**Calibration (H5).** HPD-MAE / signed deviation by stratum. Negative means over-confident: the true source lies outside the nominal high-density regions too often.

| | all | law-obeying | violating | no large facet | ≥ 3 law-obeying |
|---|---|---|---|---|---|
| G0 | 0.046 / −0.038 | 0.046 / −0.029 | 0.102 / −0.102 | 0.035 / −0.035 | 0.047 / −0.022 |
| P0 | 0.024 / +0.024 | 0.057 / +0.057 | 0.222 / −0.222 | 0.039 / +0.035 | 0.019 / −0.012 |
| F1 | 0.053 / −0.053 | 0.036 / −0.035 | 0.184 / −0.184 | 0.049 / −0.049 | 0.018 / −0.018 |

The headline HPD-MAE of each model hides two opposite regimes:
* near-perfect calibration where the facet law holds;
* strong over-confidence where it fails.

P0's good overall MAE (0.024) is a cancellation: it is under-confident on law-obeying rows (+0.057) and over-confident on violating rows (−0.222).

## 10. Confirmatory evaluation (1K val split, opened once)

**Protocol.**
* Run once on 2026-11-14 with `scripts/confirm.sh F5_s0`, after the selection of §4.7 was committed (06ad9ca).
* Pre-declared comparisons: plan §7 with amendments A1–A3.
* Final checkpoints only; all 10 evaluations are logged in `$RIND3D_LAB_OUT/CONFIRM_LOG.jsonl`. Outputs are in `docs/confirm/`.
* Confirm T*: 6,321 natural windows, all with certified labels. B1K: 4,000 release windows.

| run | T* IG | B1K IG | T* KL | HPD-MAE | T* IG L64 / L128 / L256 |
|---|---|---|---|---|---|
| pilot FL_MIS s0 | 3.98 [3.85, 4.09] | — | 6.37 | 0.037 | 3.47 / 4.08 / 4.38 |
| G0 s0 | 3.96 [3.85, 4.07] | 3.92 | 6.38 | 0.033 | 3.39 / 4.05 / 4.44 |
| G0 s1 | 3.94 [3.81, 4.05] | 3.91 | 6.40 | 0.027 | 3.40 / 4.00 / 4.42 |
| P0 (no learning) | 7.32 [7.12, 7.51] | 7.40 | 3.02 | 0.036 | 5.28 / 7.00 / 9.68 |
| F1 s0 | 7.91 [7.72, 8.09] | 7.98 | 2.47 | 0.023 | 5.97 / 7.64 / 10.10 |
| F1 s1 | 7.91 [7.72, 8.10] | 7.99 | 2.46 | 0.023 | 5.99 / 7.64 / 10.10 |
| **F5 s0** | **8.13 [7.93, 8.31]** | **8.23** | **2.23** | **0.010** | 6.04 / 7.82 / 10.51 |
| F5 s0, M = 4 at test (A2) | 8.25 [8.06, 8.44] | 8.37 | 2.12 | 0.011 | 6.06 / 7.92 / 10.76 |

| id | comparison (paired) | T* | B1K | dev T* | verdict |
|---|---|---|---|---|---|
| C1 | F1 − G0, s0 | **+3.94 [3.81, 4.07]** | +4.06 [3.94, 4.19] | +3.81 | confirmed |
| C1 | F1 − G0, s1 | **+3.97 [3.85, 4.10]** | +4.08 [3.97, 4.21] | +3.84 | confirmed |
| C2 | P0 − G0 s0 | **+3.36 [3.21, 3.51]** | +3.49 [3.34, 3.63] | +3.30 | confirmed |
| C1′ | F5 − G0 s0 | +4.16 [4.04, 4.29] | +4.32 [4.19, 4.46] | +4.07 | descriptive |
| C1′ | F5 − F1 s0 | **+0.22 [0.19, 0.25]** | +0.25 [0.22, 0.28] | +0.25 | replicated |
| A2 | F5 (M = 4) − F5 | +0.12 [0.11, 0.14] | +0.13 [0.12, 0.16] | +0.12 | replicated (secondary) |
| A2 | F5 (M = 4) − F1 s0 | +0.34 [0.31, 0.37] | +0.39 [0.35, 0.42] | +0.38 | replicated (secondary) |
| C4 | G0 s0 − pilot FL_MIS s0 | −0.02 [−0.04, 0.01] | — | −0.02 | reproduction confirmed |

**C3: hard cases with certified references (G0 s0 vs F5 s0).**

| set | rows (with reference) | label IG | G0 IG / KL | F5 IG / KL | F5 − G0, IG | F5 − G0, KL |
|---|---|---|---|---|---|---|
| solid-separated | 499 (499) | 1.90 | 0.89 / 1.14 | 0.46 / 1.51 | **−0.44 [−0.55, −0.30]** | +0.37 [0.26, 0.45] |
| all-lit | 2,500 (180) | 0.36 | −0.05 / 0.38 | −0.03 / 0.36 | +0.02 [−0.00, 0.05] | −0.02 |
| all-dark | 2,043 (180) | 1.63 | 0.22 / 1.38 | 0.25 / 1.37 | +0.01 [−0.04, 0.07] | −0.01 |

The IG columns are on the reference rows; the paired IG differences are over all rows of each set.

**Readings.**
* **Every dev effect replicates on held-out worlds**, with magnitudes within 0.15 nats of dev. Seed agreement is again ≤ 0.03.
* **F5 is also the best calibrated model.** HPD-MAE is 0.010 on confirm T*, against 0.023 for F1 and 0.033 for G0.
* **The solid-separated deficit is real and larger out of sample.** F5 trails G0 by 0.44 nats there (dev, F1: −0.04 to −0.21). These windows have no facets, and the projective Students' generic field is weaker than G0's.
  * On dev, dropout + flag repaired this with the cell-centre factor (F1drop: +0.27), but not with quadrature (F6, §4.7).
  * No dropout arm was declared for confirm, so the repair is dev-only evidence.
* **Hard cases carry information no Student extracts.**
  * The full-world label has 1.6–1.9 nats on all-dark and solid-separated windows, and 0.36 on all-lit. Both models stay at 0–0.9.
  * How much of this is X-measurable is unknown: these labels condition on the hidden world, and windows without lit/dark interfaces have no facets.

## 11. Conclusions (deliverable F)

**The research question.** How can a model infer a global 3-D source distribution from limited local geometry and binary observations, and what makes the inference accurate, efficient and generalisable?

**Answer at the 1K scale.**
* The observation contains a *projective* structure: every planar lit/dark interface lies on a plane through the source.
* A Student is accurate when its output representation can express that structure *exactly*. Concretely: an exactly normalised density over the 128³ cells, factorised as a learned generic field times analytic plane factors whose planes are measured from X, each factor integrated over the cell.
* That design doubles the information gain of the established Student (3.9 → 8.1 nats on confirm T*) and cuts KL to the full-world label from 6.4 to 2.2 nats. It trains in ~1k steps, and its non-learned core needs no training at all.
* **Efficient:** it reaches 96% of its score from 86 training worlds (§6), where the generic Student reaches 43% of its own.
* **Generalisable:** it transfers to an unseen window size with no loss, and to unseen families wherever the facet law holds (§8).
* **Where it falls short:**
  * the exception to the law (occluder interfaces), which must be seen in training;
  * facet-free windows, where no Student extracts much;
  * near-pencil facet sets, which fix a line rather than a point.

### What is demonstrated (dev, replicated on the confirm split)

1. **The facet law, and its exception.**
   * Large X-measured facets lie a median 0.04–0.09 voxel from the true source at distances of ~700 voxels.
   * The exception is occluder interfaces: sub-voxel slabs invisible in the SOLID mask. They are 3.8% of large facets, mostly thin_slabs and occluder_clusters.
2. **The established Student's gap was an operator deficit, not hidden information** (H2).
   * A non-learned product of facet factors beats the pilot Student by 3.36 nats on confirm (C2).
   * Giving the generic decoder the same planes as attention tokens does not help (G1: −0.09 on dev).
   * A projective decoder gains +3.94 / +3.97 per seed (C1).
3. **The probability representation matters at the sub-cell level** (H3).
   * Integrating the facet product over each cell, rather than evaluating it at the centre, adds +0.22 on confirm (F5 − F1), +0.34 with 4³ sub-points.
   * HPD calibration error falls to 0.010.
   * The consistent cell *average* (amplitude-corrected) is essential for a sparse implementation (§4.3).
4. **Posterior supervision beats one-hot supervision under every inductive bias** (H1).
   * Its margin shrinks from 2.5 to 0.5 nats once the operator is right (early-stopped one-hot).
   * One-hot memorises the 684 training worlds, is badly calibrated, and scores below the prior on OOD strata.
5. **Hidden-world uncertainty is small next to the remaining model error, except on degenerate pencils** (H5).
   * Editing hidden geometry while keeping X identical changes the label by 0.016 (body removal) and 0.15 nats (caster depth).
   * Where the X-measured planes fix a point, F5 (M = 4) is within 0.83 nats of the full-world label.
   * The largest remaining gap (3.1 nats) is on near-pencil facet sets, where X fixes only a line.
6. **Calibration splits into two regimes.** Models are near-perfectly calibrated where the facet law holds and strongly over-confident on law-violating rows. Headline HPD-MAE averages the two.
7. **The projective Student is data-efficient; the generic one is data- and compute-limited** (H4, dev only, §6).
   * With 86 training worlds F1 scores 7.45 nats, 96% of its 684-world score and 1.9× G0 trained on all 684.
   * G0 gains 2.2 nats over the same 8× range, with shrinking returns per doubling (1.02 → 0.76 → 0.43) at a fixed step budget.
   * 4× more steps at full data add +0.91 to G0 (G0long) but leave F1 − G0long at +2.90, and make G0 over-confident.
   * At 1K scale, neither more steps nor more worlds lets the generic decoder find the projective operator.
8. **The projective Student generalises where the facet law holds** (H6, dev only, §8).
   * Trained without L256 windows, F1 loses nothing on them (−0.01 [−0.05, 0.03]) and is better calibrated. G0 loses 0.97 nats even after the untrained size embedding is factored out.
   * Trained without corridors and thin_slabs, F1 loses nothing on held-out rows where every large facet obeys the law. It loses 0.65 nats on thin_slabs rows with an occluder-interface facet: the reliability head must see that exception in training.
   * A position-only control (Gblind) scores −0.07 nats, so all reported information comes from the window content.

### What is falsified

* "The pilot's 6.4-nat KL gap is hidden geometry plus capacity" (`E_STUDENT_PILOT.md` §2.6). At least 4.2 nats of it is X-measurable: confirm KL falls from 6.38 (G0) to 2.12 (F5, M = 4) with X-only inputs.
* "Access to the planes suffices" (G1).
* "A two-hypothesis (occluder | shadow) factor fixes occluder interfaces" (F3 − F1drop: +0.07 on violating rows, no calibration gain).
* "Planes learned from the pooled 32³ input are enough" (F2: +1.7 over G0, but −2.1 vs full-resolution facets).

### What is inconclusive

* Why facet dropout repairs facet-free windows with the cell-centre factor (F1drop, +0.27) but not with quadrature (F6).
* How much of the hard-case label information (all-dark 1.6, solid-separated 1.9 nats on confirm) is X-measurable at all.
* The X-only floor on pencil-degenerate rows.
* Whether the generic decoder would find the projective operator with ~100× more worlds *and* matched compute. At fixed compute, its gain per doubling shrinks (§6); one seed per point.
* Whether a training set with more sub-voxel occluders would teach the reliability head a transferable rule, or only a per-family rate (§8.2).

### Strongest design, and the most efficient next experiments

* **Strongest design: F5**, run with M = 4 sub-points at evaluation. It adds no learned parameters beyond F1 and costs ~1.3× its wall time (2,138 vs ~1,615 s for 8k steps plus evaluation).
* **For a 100K Student:**
  * keep X-measured facets as an input stream (~0.1 core-s per channel, §8 of the plan);
  * keep the plane-factor decoder with sparse quadrature;
  * add facet dropout with an availability flag. It is cheap, repaired facet-free windows with the cell-centre factor (dev), and should be verified with quadrature.
* **Most efficient next experiments, ranked by the measured gap share:**
  1. Point-precision on near-pencil facet sets. Use a depth prior learned along the pencil line, conditioned on the window context; test it on the pencil stratum of §4.5.
  2. Occluder-face detection from X: a visible-caster descriptor (is a visible solid edge on the plane?) feeding the facet head; target the violating stratum (0.48 nats of F5's gap).
  3. The facet-free hard cases (solid-separated, all-dark): the generic field is the only source of information there. Data scale alone does not fix them: F1's solid-separated deficit against G0 is the same at every training fraction (§6).

### Novel contributions

* The facet law as a measurable property of binary shadow observations, and the occluder-interface exception (sub-voxel occluders invisible in the geometry channel).
* The G1 control separating information from operator.
* A projective product-of-experts Student with an exactly normalised flat 128³ density and an amplitude-consistent sparse cell integral, with fused, tested GPU kernels.
* Exact X-identical counterfactual worlds (verified by the FastLabel verifier) that bound the hidden-world floor.

These are 1K-scale results: 684 training worlds, dev and confirm on the 1K splits. They are not 100K results.
