# B — Results: projective structure, projective Students, and what remains

**Status.** Working document. Every number is exploratory **dev-split** evidence unless it is marked *confirm*.
* dev = pilot-select, 102 scenes of the 1K train split;
* confirm = the 1K val split, used once (§9);
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
* G0 is still descending at step 8,000 (12.9 → 10.4). The G0long arm (32k steps) tests whether it ever catches up.

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
