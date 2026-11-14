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

(table filled in §3 below as seeds complete)
