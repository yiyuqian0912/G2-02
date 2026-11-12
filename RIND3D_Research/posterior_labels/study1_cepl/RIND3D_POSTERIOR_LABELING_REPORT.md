# Efficient, error-controlled global source-posterior labels for RIND3D: research report

**Date:** 2026-11-11.

**Scope:** RIND3D-Lite v0.1, single source, binary response.

**Data used:**
* geometry, fields and windows of the canonical v0.1 1K campaign, read-only;
* the **train** split for development and the **validation** split for every reported real-window number;
* the test split was never opened.

**Code:** `code/rpl/` (numba engine and utilities), `code/` (drivers), `tests/`.

**Evidence:** `outputs/`. Every table in `docs/RESULTS_TABLES.md` is generated from those JSON files by
`code/make_tables.py`.

**Mathematics:** `docs/METHOD.md`, with the definitions, lemmas, proofs and certificate.

**Markers:**
* **[P]** proved;
* **[V]** verified against an independent implementation;
* **[M]** measured;
* **[E]** extrapolated;
* **[A]** assumption.

---

## Executive summary

### 1. The target is well defined, and it is a polyhedral set [P]

With full geometry, the window's full-resolution response and the uniform source prior, the posterior is
the uniform law on the feasible set F_W. F_W is a finite Boolean combination of convex polyhedra, built
from the exact shadow polyhedra S_C(x) of every FREE receiver of the window.

Exact cell masses are therefore *polytope volumes*. No temperature, mismatch threshold or receiver bank is
part of the definition.

### 2. The v0.1 selection adds a factor that cannot be computed window-locally [P]

The v0.1 window-selection code makes P(window selected | s) depend on the source through the *whole* field:
* every Sobol proposal is tested for informativeness;
* the composition of the pool's lit-ratio bins drives bin choice and RNG use.

So the exact v0.1 posterior carries a global selection factor λ_W(s).

We label with the *natural source-excluded posterior* Uniform(F_W). For training it is a correct
Rao–Blackwell target: its population optimum equals that of true-source NLL. It deviates from the v0.1
posterior by at most η/2 in TV, where 1+η is the variation of λ_W over F_W. A small change to the
selection law (§D.4) makes it exact.

### 3. A certified exact labeller exists and is correct on controlled problems [V]

CEPL combines:
* a source-space octree;
* a hierarchical receiver test with two certificates proved here (convex shadow polyhedra, and a 64-corner
  all-blocked test);
* exact polytope volumes in the leaves;
* a rigorous TV bound.

On thin, extended and disconnected feasible sets spanning about 9,000 cells at 128³, its volumes match
brute force (every receiver tested for every sample point):
* to within 0.01–0.24%, and 1.4% on the corner case;
* the label TV equals the brute-force Monte Carlo noise floor;
* no feasible point was found outside its support.

It certifies TV ≤ 0.0024–0.0064 on those problems. A 65k-point Sobol design expects about 2 hits there.

### 4. On real windows, exact posteriors are microscopic in volume but often extended in space [M]

On the validation side-64 windows (`docs/RESULTS_TABLES.md`), certified feasible volumes range from
6.4×10⁻¹⁶ to 3.6×10⁻⁴.

They are often spread over many 128³ cells: thin tubes and slabs, not points. Certified *lower* bounds
(sound even for incomplete runs) put most of the posterior mass outside the true source's cell in several
windows:
* the information-rich window f402ba60/69 (complete): **≥ 0.98** of the mass outside the true source's
  128³ cell and **≥ 0.72** outside its 16³ cell;
* the minority-poor windows 13233791/48 and 11036406/80: ≥ 0.999 and ≥ 0.996 at 128³;
* in total, 5 of the 23 windows are certified to put more than half their mass outside the true 128³ cell, 12 of
  the 23 have certified mass in two or more cells, and none is certified one-hot.

Where they can be computed, exact labels therefore carry substantial information beyond the one-hot NLL
target. The train window 0084e469/0, run to completion in 2.1 h, has positive certified mass in 142 cells,
with ≥ 0.97 outside the true cell.

The receiver model matters enormously. On window f0be2f5f/82:
* given a 256-receiver bank, the exact posterior has volume 4.0×10⁻⁷ and 4% of its mass in the true cell;
* given the full window, it has 6.4×10⁻¹⁶ and all certified mass in the true cell.

Banks do not approximate the full posterior.

### 5. Certified labels at full receiver resolution are out of reach at production scale [M, E]

No real window reached a non-vacuous certificate. Within a budget of 1.5×10⁵–6×10⁵ nodes:
* most windows do not finish;
* those that finish have undetermined volume far above their certified volume.

That happens because the bound is relative to vol(F_W) ≈ 10⁻¹⁶–10⁻¹². Every leaf where many overlapping,
non-convex shadow constraints meet — up to 3×10⁴ per leaf, near silhouette vertices and multi-occluder
shadows — must be resolved below that scale. The controlled corner case T3 shows the same failure.

Cost:
* on 23 validation side-64 windows (`docs/RESULTS_TABLES.md`), one laptop core spends a median of 56 min and up
  to 2.7 h per window, at up to ~65 ms per node;
* only 2 of the 23 windows finished their 1.5×10⁵–6×10⁵-node budget, and 0 of 23 were certified at TV ≤ 0.05;
* one further window was terminated after more than 10⁴ s without finishing;
* per-box cost grows with the window's shadow-boundary voxels: 4.5k, 16k and 70k at sides 64, 128 and
  256.

Scene-level sharing is real but bounded: boundary voxels recur 3.3–4.8× across a scene's windows, and the
search over source space is window-specific.

### 6. Recommendation

* **Keep direct true-source NLL as the main supervision.** The exact full-geometry posterior is
  well-defined, and often far from one-hot. It would be a better, lower-variance target than NLL, but we
  cannot yet deliver it with a known error bound at any affordable cost.
* **Do not adopt bank-based, tempered or sample-only posteriors as a substitute.** They are different
  targets (§B, §C.4).
* **The most promising route** is to declare a coarser observation shared by the Student and the label,
  e.g. receivers on the Student's 32³ grid (§F.4). The Rao–Blackwell condition then holds by construction,
  the posterior is thicker, and exact certified labels cost bank-like 30–40 s per window (measured).
* **Engine fix needed by any certified route.** A BSP-subtraction replacement for inclusion–exclusion is
  required first (§F.3).

---

## A. What exactly is the correct posterior?

### A.1 Implementation facts (read from code; `docs/METHOD.md` §1)

**Geometry.**
* Canonical binary64 parameters, decoded to exact rational H-representations
  (`rind3d.rendering.polyhedra.decode_convex`).
* The solid is a finite union of closed, bounded, full-dimensional convex polyhedra, at most 64 rows
  each, all strictly inside Ω = (0,1)³.

**Visibility.**
* A receiver is the voxel centre (v+½)/1024, in XYZ order.
* It is dark iff the closed segment [s, x] meets the solid; touching counts.
* SOLID voxels are flagged and carry 0. The FREE shadow is `~response & ~solid`.

We checked our float renderer against the stored v0.1 windows of three training scenes: 0 mismatches on
6.8×10⁶ FREE voxels (`code/check_render.py`) [V].

**Sources.**
* Continuous, not voxel-centred: s ~ PCG64 float64 on the 2⁻⁵³ grid, uniform on {d(s, Σ) > 2⁻²⁶}.
* An eligibility test R(s) runs on 16³-block supports of the whole field. 3 of 1,003 proposals were
  rejected.

**Windows.**
* Sides 64, 128 and 256, with quotas 85, 85 and 86.
* The half-open box must exclude s, and the window must contain both a lit-FREE and a dark-FREE voxel.
* The pool comes from a scrambled Sobol origin stream seeded per attempt.
* A mixture of natural, ratio-bin-uniform and extreme components is selected, with a spatial-diversity
  gain (`stratified_windows.select`). The full step-by-step s-dependence analysis is in `METHOD.md` §2.2.

**Resolutions.**
* The observation is the full window: 64³–256³ receivers.
* The Student sees 4×32³ pooled channels.
* The Student outputs 16³ coarse × 8³ fine = 128³ cells.

**Single- vs multi-source.**
* v0.1 is single-source.
* The production v0.2 multi-source additive profile (K = 2–4) has a posterior over *sets* of sources with
  continuous strengths. It is not polyhedral in the same sense, so it is out of scope here.

### A.2 The full-geometry posterior [P]

    p(s | G, W, O_W)  ∝  1[s ∈ F_W] · R(s) · λ_W(s)

F_W is the set of sources in the legal domain, outside B(W), that reproduce every FREE receiver of W
(`METHOD.md` §2.1).

Because every lit receiver's shadow polyhedron contains the solid, F_W lies in FREE space automatically.
Only the 2⁻²⁶ clearance shell needs a separate bound: its volume is ≤ area(∂Σ)·2⁻²⁶.

### A.3 Uniform is justified only for the natural law, and that is the label we define [P]

The prior is uniform, so R ≡ 1 and λ_W constant on F_W would give a uniform posterior. In v0.1, λ_W is not
constant: it is roughly 1/P_b(G, s), the probability that a uniform origin is valid and falls in W's
lit-ratio bin, with pool-truncation and diversity corrections.

We therefore define the label as q_W = Uniform(F_W) and keep the selection effect explicit:
* **Bias bound.** TV(q_W, v0.1 posterior) ≤ η/2, where max λR / min λR ≤ 1+η on F_W.
* **When it vanishes.** Where F_W lies within ~10⁻⁵ of s*, alternative sources' fields differ only in
  boundary voxels, so λ_W is constant to high accuracy. This is an argument, not a certificate.
* **When it matters.** The real-window measurements (§C.3) show that F_W is often extended along thin
  tubes or slabs across tens to thousands of cells. There, distant alternative sources have genuinely
  different fields, and η is not small a priori. We did not measure it: doing so needs full-field renders
  for alternative sources.
* **How to make it exact.** The design fix is in §D.4.

### A.4 The full-geometry posterior vs the local-only posterior the Student can learn [P]

The Student input X = φ(O_W, W) is a function of Z = (G, W, O_W). Hence E[q_W(s) | X] = p(s | X).

Training with soft labels q_W has the *same* population optimum as true-source NLL, the local-only
posterior p(s | X), and a gradient variance that is never larger. The Student cannot reconstruct hidden
geometry; that is a limit of X, not of the labels.

The benefit of q_W over the one-hot NLL target is the conditional variance it removes. That is zero
whenever q_W is concentrated in the true source's output cell. This is why the regime split in §C.3
decides the practical value.

### A.5 What precision is needed [P]

Only the label's cell masses enter the Student's cross-entropy. For softmax logits the CE gradient is
p_θ − q, so a label error Δq biases the logit gradient by exactly Δq, with L1 norm 2·TV.

**TV over the 128³ cell masses** is therefore the operative metric. TV over 16³ is never larger.

The certificate (`METHOD.md` §5; `code/rpl/bound.py`) is a deterministic TV bound from per-cell volume
intervals. It is exactly 0 for a certified one-hot label. Statistical and empirical statements are
labelled as such throughout.

## B. Why earlier approaches struggled, and the method that replaces them

| Earlier approach | Obstacle (now quantified) |
|---|---|
| Gibbs q ∝ e^{−C/τ}, τ = 0.05 | It is a tempered pseudo-posterior, not the deterministic posterior. Its NLL at s* was −3.35 versus −8.44 for the bank-hard posterior (Student study). |
| Uniform or scrambled-Sobol source sampling | The discovery ceiling is n·vol(F). Real feasible volumes are 10⁻¹⁶–10⁻¹², so the expected hits are 10⁻¹¹–10⁻⁷ even at 65k points. On the controlled slab (vol 3×10⁻⁵), 65k points expect 2 hits for a 9,757-cell label. |
| Receiver banks (256–512) | The exact bank posterior is ~10⁹ times larger than the full one, with 4% vs 100% mass in the true cell (§C.4). It is a different target. |
| Certified box-only octrees | Undetermined volume decays like cost^{−1/2}, the same rate as Monte Carlo (Gibbs study). A bound relative to vol(F) ≈ 10⁻¹⁵ is vacuous at any feasible depth. |

**CEPL** (`docs/METHOD.md` §4; `code/rpl/engine3.py`) replaces the box test in the leaves with **exact
polytope volumes**.

* **Lemma 1.** Each receiver's shadow is a convex polyhedron with explicit rows from the front faces and
  silhouette edges.
* **Lemma 2.** A box pair is all-blocked by one convex C iff its 64 corner segments are.
* **Lemma 3.** The all-visible certificates.
* **Lemma 4.** Penumbrae of convex boxes are convex.

**Engineering that turns this into a practical search** (each item changes cost only, never results):
* ~640 *sentinel* receivers at the root;
* *learned refuters*;
* a cover stage (single receivers only) followed by a leaf stage;
* incremental clipping, in which an empty polytope certifies OUT;
* non-convex constraints filtered against the convex polytope before inclusion–exclusion.

Engine v3 reproduces v2's T1 result bit for bit, about 1.6× faster (100 s vs 164 s). On the same 20k-node
budget of a real window it runs in 23 s against v2's 76 s.

## C. Verification, measured performance and failure cases

All tables are in `docs/RESULTS_TABLES.md`.

### C.1 Component tests [V]

* **Polytope clipper vs qhull.** 400 random polytopes: worst relative volume error 3.8×10⁻¹⁴. Slabs down to
  10⁻¹⁰ thick are accurate (`tests/test_polytope.py`).
* **Shadow rows.** The silhouette-edge rows, the full Fourier–Motzkin rows and the canonical exact GMP
  oracle agree on 6.6×10⁶ test points from 6 real scenes, with 0 disagreements (`tests/test_shadow.py`).
* **TV bound.** Valid against exhaustive vertex enumeration in 3,000 random instances; median slack
  2×10⁻⁴ (`tests/test_bound.py`).

### C.2 Controlled reference problems [V]

The reference problems are T1 (thin extended slab), T2 (the same slab split by an occluding plank, so
disconnected) and T3 (corner tube). Each brute-force reference tests every one of the 4,096 window
receivers for every sample point. It uses a separate code path: direct segment tests, no octree, no shadow
rows, no polytopes.

A fourth case, T4, puts the occluder's face exactly on a receiver-centre plane, testing tangency.

* **Volumes.**

  | Case | Engine | Brute force | Difference |
  |---|---|---|---|
  | T1 | 3.1369×10⁻⁵ | 3.1372×10⁻⁵ | 0.01% |
  | T2 | 2.719×10⁻⁵ | 2.713×10⁻⁵ | 0.24% |
  | T3 | 2.27×10⁻⁸ | 2.30×10⁻⁸ | 1.4% |
  | T4 | 1.5594×10⁻³ | 1.5592×10⁻³ | 0.01% |

* **Completeness.** Among 2×10⁶ uniform points, the feasible points (62 in T1, 58 in T2) all fall inside
  the engine's support: **0 misses**.
* **Label accuracy.** The TV between the engine label and the stratified brute-force label equals the
  brute-force noise floor (T1: 0.0686 vs 0.0689). The engine is the more accurate of the two.
* **Certificates.**
  - T4: TV ≤ 1.7×10⁻⁵, 13,310 cells, 29 s (engine v3). 11,590 receiver–component tests were flagged
    ambiguous (receiver exactly on a face plane) and handled conservatively;
  - T1: TV ≤ 0.0024, 9,757 cells, 100 s (engine v3);
  - T2: TV ≤ 0.0064, 8,906 cells, 582 s (engine v2);
  - T3: bound **0.64**, i.e. *not certified*. Near silhouette vertices, lit receivers contribute
    non-convex constraints beyond the inclusion–exclusion term cap, and those leaves stay undetermined at
    depth 12. The volume *estimate* is still correct to 1.4%. This is the failure mode that dominates on
    real windows.

### C.3 Real windows (validation, side 64) [M]

The per-window table and its summary lines are in `docs/RESULTS_TABLES.md`. Main observations:

* **Soundness.** The true source's cell carries certified positive volume in every window that finished
  (f0be2f5f/82, f402ba60/69, and train 0084e469/0).
* **Extended posteriors are common.** In 5 of 23 windows (13233791/48, c6375435/15, ee197809/16,
  11036406/80 and f402ba60/69), certified lower bounds put 0.82–0.999 of the mass outside the true 128³
  cell, and 0.33–0.72 outside the true 16³ cell in three of them. 12 of the 23 have certified positive
  mass in at least two cells; none is certified one-hot.
  - These windows include information-rich ones: f402ba60/69 has 10 components and 76k minority voxels.
  - Elsewhere positive certified mass covers 66–3,518 cells, but the bounds are still uninformative
    because the runs did not reach the truth's neighbourhood within budget.
* **Microscopic volumes.**
  - The feasible volumes certified so far range from 6.4×10⁻¹⁶ (f0be2f5f/82, all in the true cell) to
    3.6×10⁻⁴ (13233791/48, two dark voxels).
  - Across that range, the sampled windows cover both one-cell and many-cell supports.
* **No real window is certified.**
  - The tight TV bound is ≈ 1 everywhere, because U ≫ L.
  - In finished windows U comes from leaves left undetermined at depth 13 by the non-convex budget:
    18,296 such leaves for 0084e469/0, and up to 3.3×10⁴ constraints per leaf.
  - In unfinished windows it comes from unvisited boxes.
  - The labels in the side-car format would all be UNCERTIFIED (NLL fallback).
* **Cost.**
  - Wall time per window ranges from 254 s to 9,766 s at 1.5×10⁵–6×10⁵ nodes (median 3,358 s).
  - The two complete validation windows took 672 s and 3,733 s.
  - The complete train window took 7,727 s and 4.8×10⁵ nodes.
* **Regime frequency** (all 29,696 validation windows; minority class = min(lit, dark) FREE voxels):

  | Side | Minority ≤ 256 voxels | Minority ≤ 1024 voxels | Median minority |
  |---|---|---|---|
  | 64 | 7.9% | 13.5% | 3.9×10⁴ |
  | 128 | 5.5% | 9.2% | 2.5×10⁵ |
  | 256 | 4.7% | 7.6% | 1.8×10⁶ |

  Extended posteriors are *not* confined to minority-poor windows (f402ba60/69, 0084e469/0). The minority
  count is therefore not a sufficient screen.

### C.4 Receiver banks vs the full window [M]

Window f0be2f5f/82:

| Receivers | Exact volume | Cells | Mass in true cell | Certified TV bound | Time |
|---|---|---|---|---|---|
| 256 (stratified bank) | 4.0×10⁻⁷ | 95 | 0.040 | 0.047 | 42 s |
| 4,096 | 1.8×10⁻⁹ | 26 | 0.067 | 0.39 | 30 s |
| Full (239k FREE) | 6.4×10⁻¹⁶ | 1 | 1.0 | — (see §C.3) | 672 s |

Bank posteriors are *cheap and certifiable*, but they condition on less than the Student sees. Because
X ⊄ σ(bank), they are **not** Rao–Blackwell targets for this Student.

### C.5 Thin, disconnected and rare regions, boundaries and tangencies

* **Thin and disconnected sets** are handled by construction:
  - nothing is sampled;
  - every region not certified OUT stays in the bound;
  - exact leaves resolve slabs 10⁻³ thick without subdividing them (T1, T2).
* **Rare tiny sets** (10⁻¹⁶) are found when they exist (f0be2f5f/82).
* **Proving the absence of other tiny sets** is the expensive step.
* **Boundaries and tangency.** Closed-segment semantics give closed shadow polyhedra, and boundaries have
  measure zero.
  - A receiver exactly on a face plane of C (|f_i(x)| ≤ 10⁻¹²) is flagged AMBIGUOUS and never certified.
  - 0 such receivers occurred in the measured windows.
  - Filtered decisions use a 10⁻¹¹ margin on unit-normalised planes, against evaluation errors below
    10⁻¹⁴.
  - Undecided signs always resolve conservatively (CUT), so they can cost time but cannot change a result.
* **What is not machine-checked.** The certificate is not machine-checked. It assumes the float
  predicates' error bounds and float polytope volumes (relative error ~10⁻¹³). Upgrading to exact
  rationals is possible: the canonical package already has GMP predicates.

## D. Dataset integration plan

1. **Where.** Add a post-sealing job, after field generation and window selection, so that neither is
   changed. Its inputs are the canonical geometry, the stored window bits and the private true source; the
   true source is used only to report diagnostics, never by the label. Labels are computed per window
   against a declared budget.
2. **What is stored.** One immutable side-car file per scene, `posterior_labels.v1.npz`
   (`code/rpl/labels.py`), holding per window:
   * status: CERT_ONEHOT / CERT_SPARSE / UNCERTIFIED;
   * the TV bound;
   * the certified volume L and undetermined volume U;
   * sparse fine-cell indices (u32) with masses (f32);
   * cost diagnostics.

   File-level metadata records the engine version and code SHA, engine parameters, tolerance, geometry and
   source identities, and the physics, prior and selection definitions.

   **Sizes:** a one-hot label needs 8 bytes. A broad label needs 8 bytes per cell, i.e. 10–80 KB for
   1,000–10,000 cells. `to_hierarchical` converts a label into the Student's 16³-coarse plus 8³-fine CE
   target without any physics at training time.
3. **Training policy.**
   * CERT_ONEHOT is identical to the NLL target at 128³.
   * CERT_SPARSE replaces the one-hot target with the certified label.
   * UNCERTIFIED falls back to true-source NLL. An uncertified label is never emitted as a target.
   * The Student sees no status or bound.
4. **Making the label exact under the generator.**
   * For the next release, draw window proposals independently of s, accept on O_W and the declared
     condition only (as the v0.3 streams already do), and train with per-sample weight ∝ P_b(G, s*).
     P_b(G, s*) is exactly computable from the true field via the existing rate stream.
   * Then Uniform(F_W) is the exact posterior of the weighted training law, and the η/2 term disappears.
   * The v0.1 ratio-bin balancing and pool truncation cannot be neutralised by one weight.

## E. Scaling assessment

Measured values are separated from extrapolated ones. The numbers come from `code/cost_model.py`, written
to `outputs/COST_MODEL.json`.

**Measured** (one laptop core per window, validation side 64, 23 windows):
* Wall time per window: median 3,358 s, mean 3,879 s, max 9,766 s.
* Complete: 2 of 23 (672 s and 3,733 s). Certified: 0.
* Incomplete windows are charged their budget, so these figures are lower bounds on the cost of a
  certificate.
* Other runs:
  - complete train window 0084e469/0: 7,727 s and 4.8×10⁵ nodes, not certified;
  - 256-receiver bank posterior: 42 s, certified 0.047;
  - controlled side-16 windows: 29–582 s, certified 1.7×10⁻⁵–0.0064 (T1, T2, T4).

**Cost drivers [M]:**
* Per-box cost grows with the window's boundary receivers: 4.5k, 16k and 70k at sides 64, 128 and 256.
* The number of boxes grows with the spatial extent of the posterior, and with the arrangement complexity
  near microscopic feasible sets.

**Projection [E].** The declared rule: the side-64 mean, scaled by the boundary-voxel ratio, with no
retries; this is optimistic because no window was certified.

| Windows | Per-window cost (s) at side 64 / 128 / 256 | Total CPU-core-hours | Core-years | With an ideal cross-window receiver cache (÷ 4.27) |
|---|---|---|---|---|
| 25.6M (100K × 256) | 3.9×10³ / 1.4×10⁴ / 6.1×10⁴ | 1.9×10⁸ | 2.2×10⁴ | 4.4×10⁷ core-hours |
| 10.16M (~102 per scene) | same | 7.5×10⁷ | 8.5×10³ | 1.8×10⁷ core-hours |

True-source NLL needs one 24-byte source record per scene and no physics at label time.

**Reading.** Certified full-resolution labels are not a production-scale method in this form. Even a
100–1000× engine improvement would leave 10⁵–10⁶ core-hours, without yet a certificate on real windows.

## F. Recommendation

1. **Main supervision: true-source NLL.**
   * The exact full-geometry posterior is often far from one-hot. That makes it scientifically attractive:
     it is a Rao–Blackwell target with lower variance than NLL.
   * But on real windows we could not certify it at any affordable budget (§C.3, §E). An uncertified
     label is no better founded than the earlier Teachers.
   * NLL has the same population optimum and needs no physics.
2. **Do not build production supervision on bank-based, tempered or sample-only posteriors.**
   * They are different targets: a 256-receiver bank posterior has 4% of its mass where the
     full-observation posterior has 100% (§C.4).
   * Sample-only completeness is impossible below the n·v ceiling.
3. **Research prerequisite for any certified route.** Replace inclusion–exclusion by an exact BSP
   subtraction:
   * a convex piece is removed from a polytope as a disjoint union of at most m convex parts;
   * empties are pruned at each step;
   * this avoids the 2^k blow-up behind every vacuous real-window certificate (T3, f0be2f5f/82,
     0084e469/0).

   Add parallel subtrees and a per-receiver shadow-row cache shared across a scene's windows (≤ 3.3–4.8×).
   Re-measure on the same declared validation sample. This is a contained engineering study (CPU only).
4. **The route most likely to make certified soft labels practical: change what is conditioned on.**
   * Declare the Student's observation and the label's conditioning observation to be the *same* receiver
     subset, e.g. a fixed 16³ or 32³ grid of window receivers, instead of pooled channels from all voxels.
   * The Rao–Blackwell condition Z ⊇ X then holds by construction, and the posterior is broader, so it is
     more informative relative to NLL.
   * Measured on one window: the exact posterior given 256 receivers was certified to TV ≤ 0.047 in
     42 s; with 4,096 receivers the bound was 0.39 after 30 s. Certification cost grows with the receiver
     count, so a 32³ grid (32,768 receivers) is untested and would need item 3 first.
   * This changes the Student's input contract, which is a design decision for the team.
5. **For the next release: adopt the selection-law fix (§D.4)** — proposals independent of s, acceptance
   on O_W only, and a per-sample weight P_b(G, s*). Without it, no window-local label is the exact
   posterior of the training distribution.

**What was not done:**
* no production runs;
* no GPU and no cloud compute;
* local CPU only, on at most 12 cores, with every run bounded by a declared budget;
* nothing committed;
* no dataset, checkpoint or earlier artifact modified.
