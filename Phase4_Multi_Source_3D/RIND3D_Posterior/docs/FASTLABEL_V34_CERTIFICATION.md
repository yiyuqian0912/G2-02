# FastLabel v3.4: statistical certification of the RIND3D-100K v3 posterior labels

**Status (2026-11-18).** FastLabel v3.4 is the production labeller of RIND3D-100K v3, under contract
`rind3d-100k-contract-v3`. This document states what its `certified_stat` labels guarantee and proves it. It lists the
assumptions the proof needs and validates the labels against independent exact references.

**This is not a deterministic certificate.** It is a *statistical* certificate. Over the labeller's random streams, the
probability that the campaign contains any accepted statistical label with TV > 0.05 is at most 0.01. A given label can
be wrong. Its probability of being wrong is at most 1.37·10⁻⁹.

Every statement is marked with one of three tags:

* **[proven]**: follows from the definitions and the code, by the argument given here.
* **[conditional]**: proven under the numerical assumptions of §6.1, which are stated explicitly and are not proven.
* **[empirical]**: measured. The scripts and recorded outputs are in
  `RIND3D_Posterior/experiments/fastlabel_v34_validation/`.

Earlier records: [CERTIFICATE_AUDIT.md](../experiments/fastlabel/CERTIFICATE_AUDIT.md) (the v3 derivation, which this
document supersedes for the 100K), [GPUVERIFY.md](../experiments/fastlabel/GPUVERIFY.md) (verifier and Gate A),
[FASTLABEL_CORE.md](FASTLABEL_CORE.md).

## 0. Summary

**Theorem** (§3.5) **[conditional]**. Assume the numerical terms of §6.1 total η ≤ 10⁻⁶ for every B the labeller
builds. The measured total is ≤ 7·10⁻¹⁰.

1. **Per label.** For every fixed scene, window and channel with Vol(F) > 0:

       P( status = certified_stat  and  TV(q̂, q_F) > 0.05 ) ≤ Σ_r α_r(1/20 − η) ≤ 1.37·10⁻⁹,

   where q̂ is the stored label, read under its normalisation rule (§6.3).

2. **Campaign.** Let the v3 campaign make at most N_max = 4,800,000 statistical label attempts. The label-unit bound is
   100,000 scenes × 12 windows × ≤ 4 channels. Then:

       P( ∃ accepted certified_stat label with TV(q̂, q_F) > 0.05 ) ≤ 4.8·10⁶ × 1.37·10⁻⁹ ≤ 0.0066 ≤ 0.01.

The probability is over the labeller's keyed random streams only, for the fixed realised dataset.

**Verdict on Gate B: PASS, with conditions.** Gate B asks for a documented, mathematically justified statistical
certification at the stated target. The conditions are listed in §10.

| item | result |
|---|---|
| containment F ⊆ B | **[conditional]**. Sound up to float decision margins (§6.1 N1). **[empirical]** missing mass ≤ 7·10⁻¹⁰ (relative) in every exact comparison, and no cell with reference mass is absent from B. |
| sampling law | **[proven]** piece ∝ volume `hi`, then uniform in the piece; no rejection sampling. Error from the float hulls ≤ 3.3·10⁻⁸ (§6.1 N2). |
| point predicate (GPU verifier) | **[proven]** exact for closed segments, closed solid, clearance > 2⁻²⁶ and the closed window box (§3.3). **[empirical]** Gate A, plus agreement with exact \|F\|/\|B\| (§7.4). |
| statistical test | **[proven]** exact rational α_r ≤ δ_r for all 12 rounds at the v3 budget, valid down to the boundary 1/20 − 10⁻⁶ (§5) |
| adaptivity / optional stopping | **[proven]** fresh draws per round, B_r measurable w.r.t. the past, union over rounds (§3.4) |
| zero feasible volume | **[proven]** never `certified_stat` except with the same ≤ 1.37·10⁻⁹ probability; ends `uncertified` or with a deterministic bound (§3.6) |
| serialized vector | per-cell binary64 volumes of B. The label is their exact normalisation; the loader's float q is within TV 1.2·10⁻¹⁰ of it **[proven]**, measured ≤ 1.3·10⁻¹⁶ (§6.3) |
| familywise budget | δ = 0.01 / 4.8·10⁶ = 2.083·10⁻⁹ per label; exact false-certificate probability ≤ 1.36·10⁻⁹ (§5) |
| empirical TV vs references | 50 statistical labels, 24 with an independent exact reference. Max TV ≤ 0.0141 where the reference is tight, and ≤ 0.0437 against the loosest admissible enclosure. **No label above 0.05.** 25 single-cell labels are exact (§7) |
| defects fixed (commit `bc6d860`) | D1: run options could re-roll seeds or change δ. D2: the validator did not check the certificate's stream, δ or design. H1: no curved-occluder guard (§8) |

## 1. Target posterior

The definitions follow `docs/SEMANTICS.md`, `GOALS.md` and `docs/RIND3D/GLOSSARY.md`. They are restated as the
code implements them.

* **Solid.** S = ∪_c C_c, a union of closed convex polytopes C_c = {x : a_ci·x + d_ci ≤ 0} with exact rational rows.
  These come from the canonical geometry bundle; the float64 copies are `P` and `K`.
* **Source domain.** D(G) = {s ∈ (0,1)³ : dist(s, S) > 2⁻²⁶}. This also puts s outside the closed solid.
* **Window.** The window W has origin o, side L and resolution N. Its receivers are the voxel centres
  x = (o + t + ½)/N. Its closed box is B̄(W) = [o/N, (o+L)/N]³.
* **Response.** For receiver x and source s: R_G(s)(x) = 2 if x ∈ S (SOLID). Otherwise R_G(s)(x) = 0 (inactive, legacy
  "dark") iff the closed segment [s, x] meets S, and 1 (active, legacy "lit") if it does not.
* **Observation and feasible set.** The observation of channel j is X = R_G(s_j*)|_W. The feasible set is
  F = F_W^j = {s ∈ D(G) \ B̄(W) : R_G(s)|_W = X}.
* **Target.** q_F(c) = Vol(F ∩ c) / Vol(F), the uniform measure on F pushed forward to the 128³ grid. It is defined when
  Vol(F) > 0. It is not a delta at the generating source s_j*.

  It is the posterior under a uniform prior on D(G), *conditional on the complete hidden geometry*. This is a Teacher
  target. It is not the Student's posterior over worlds that share the window.

## 2. The implemented estimator

Code: `RIND3D_Posterior/src/rind3d_posterior/fastlabel/`. Line numbers refer to commit `858fff7`. `bc6d860` shifts
only `coordinator.py`, `validate.py` and `inputs.py` after line 98.

### 2.1 Per label (`pipeline.label_window_fast`, `pipeline.py:127-260`)

1. **Setup.**
   * Window classes `cls` are read from the stored packed crops (`window.py`).
   * Three keyed streams come from `SeedSequence([base_seed, slot, w, s, 3]).spawn(3)` (`label_rng`, `pipeline.py:121`):
     * `seed_items`: the initial receivers;
     * `rng_s`: the sample points;
     * `rng_c`: the subsampling of violated receivers.
2. **Stage 1.** A coarse octree (depth 4) of candidate boxes is refuted by sentinel receivers (`engine.stage1`). It never
   certifies IN.
3. **Round r = 1…R (R = 12).**
   * **B_r.** The tau-pencil engine (`engine.stage2`, `tpl/engine_p2(x).py`) builds the feasible set of the receiver
     subset X′_r. Initially X′ has 256 random boundary receivers and 128 random interior FREE receivers. For r ≥ 2,
     regions that miss the outer filter of B_{r−1} are dropped (`engine_p2x.bf_hit`).
   * **Representation.** B_r is a list of *records*. Each record is a disjoint convex polytope (a BSP leaf) inside one
     128³ cell. It has float64 vertices (`voff`, `vn`, `vpool`) and a float64 volume `hi`. Its kind is IN (every receiver
     of X′ satisfied) or UNDET (budget/ambiguity; still part of B).
   * **Deterministic exits.** If B_r lies in a single cell, the label is one-hot (`single_cell_support`, `:157-159`).
     `exact_small_B` and `exact_race` are disabled by the production DEFAULTS (`small_records = race_nodes = 0`).
   * **Statistical test** (`:166-212`):
     * δ_r = δ·2⁻ʳ.
     * The thresholds are fixed in advance: (a₁, a₂) = `design2(512, 1024, 0.05, δ_r)`.
     * Points i.i.d. from `BSampler(res, rng_s)` (§2.2) are drawn in chunks of 128, 384 and 512, i.e. at
       checkpoints 128, 512 and 1024.
     * Every point is decided exactly by `gpuverify.WindowVerifier.test_batch` (§3.3). K counts the failures.
     * The round stops with no certificate as soon as K > a₂.
     * It certifies at stage 1 if K ≤ a₁ after 512 points, and at stage 2 if K ≤ a₂ after 1024 points (`:200`).
   * **Otherwise.** The violated receivers of the failing points (at most 64 per point and 2048 per round) join X′.
     The filter `bf = build_filter2(res)` is built, and the next round starts.
4. **After R rounds, or on a sampler failure.** `fallback_exact` runs the deterministic engine on B with every
   receiver. The status is `certified` / `certified_loose` / `uncertified` from its deterministic `tv_bound` (`:252-255`).
5. **Statistical label** (`:233-249`):
   * the stored arrays are q_B = per-cell sums of `hi` of the certifying round's records (`cell_masses(res.cell, res.hi,
     res.hi)`), with lo = hi;
   * status `certified_stat`, `tv_bound` = ε = 0.05;
   * the certificate (eps, delta_label, delta_round, round, stage, n, K, a1, a2, n1, n_max, policy, n_tasks, seed) is
     stored in the label meta.

### 2.2 Sampling in B (`sampler.py`) [proven]

* **No rejection sampling.**
* **Piece choice.** A record i is chosen with probability hi_i / Σhi (`rng.choice(p=...)`).
* **Point in the piece.**
  * A piece that is an exact axis-aligned box (8 vertices = the box corners) is sampled directly as lo + size·U³.
  * Otherwise the vertices are tetrahedralised (Delaunay, `QJ Pp`, deterministic qhull seed).
  * A tetrahedron is chosen with probability ∝ its volume, and a point is drawn uniformly in it by the Rocchini–Cignoni
    barycentric fold. The fold is an exact measure-preserving map of the unit cube onto the simplex.

So the law of one point is Q_r = Σ_i (hi_i/Σhi) · U(H_i), where H_i is the hull of the piece's float vertices.

### 2.3 100K serialization and loading

* `rind3d_100k/labels.py:_fastlabel_task` stores per-cell arrays (dcell, lo, hi), the status, `tv_bound` and the
  certificate.
* Records reach consolidated `labels/scenes/*.labels.npz` only through the journal (§8).
* The loader (`loader.py:121-136`) returns q = lo / lo.sum() in binary64. It accepts a label iff the meta status is in
  {certified, certified_stat}, the label has cells, and tv_bound ≤ 0.05 (`:230`).

## 3. Assurance theorem and proof

Notation:

* |A| is Lebesgue volume, and U_A the uniform law on A.
* TV(P, Q) = sup_A |P(A) − Q(A)|.
* push(·) is the image measure under the cell map x ↦ cell(x).

### 3.1 Lemma 1 (nested uniforms) [proven]

If F ⊆ B and 0 < |F| < ∞, then TV(U_B, U_F) = 1 − |F|/|B|.

*Proof.* U_F − U_B is positive exactly on F, so TV = U_F(F) − U_B(F) = 1 − |F|/|B|. ∎

### 3.2 Lemma 2 (imperfect outer set and sampling law) [proven]

Let Q be any probability law with support B* and Q(F) = 1 − θ. Let μ = |F \ B*|/|F| and η_S = TV(Q, U_{B*}). Then

    TV(Q, U_F) ≤ θ + 2 η_S + μ.

*Proof.* Use the triangle inequality TV(Q, U_F) ≤ TV(Q, U_{B*}) + TV(U_{B*}, U_{F∩B*}) + TV(U_{F∩B*}, U_F).

* The first term is η_S.
* By Lemma 1, the second is 1 − |F∩B*|/|B*| = 1 − U_{B*}(F). Since U_{B*}(F) ≥ Q(F) − η_S, it is ≤ θ + η_S.
* The third is 1 − |F∩B*|/|F| = μ, again by Lemma 1 (nested sets F∩B* ⊆ F). ∎

**Projection.** The label is q̂ = Σ_i (hi_i/Σhi) δ_{cell_i}. If every H_i lies in cell_i, then q̂ = push(Q) exactly.
Let η_cell = TV(q̂, push Q) cover the general case, and use the data-processing inequality
TV(push Q, push U_F) ≤ TV(Q, U_F). Then:

    TV(q̂, q_F) ≤ θ + η,   η := 2 η_S + μ + η_cell.          (★)

This is where the projection "cannot increase TV". It is used only through (★).

### 3.3 Lemma 3 (the point test decides s ∈ F exactly) [proven; implementation validated by Gate A]

For a binary64 point s, `WindowVerifier.test_batch` (`gpuverify.py:350-447`) returns ok iff s ∈ F. The four
components follow.

**(a) Domain.**
* *Window box.* The box test `o/N ≤ s ≤ (o+L)/N` (`:355`) is exact, since the bounds are dyadic.
* *Open cube.* The open-cube test is exact.
* *Legality.* A plain source is legal by a GPU *strict-separator* certificate (`verify.cu:shadow_rows`): for every body,
  some face with v = a·s + d > 0 and v² > r²|a|², evaluated with directed rounding.
  * Since C ⊆ {a·x + d ≤ 0}, dist(s, C) ≥ v/|a| > r.
  * Otherwise the host runs the exact `legal_source` (`inputs.py:30`).
  * A legal source flagged "inside a body" raises an error. It is never a silent pass.

**(b) Shadow lemma (closed segment ⇔ shadow rows).**

Let C = {f_i ≤ 0} with s ∉ C, and define c_i = −f_i(s). Then [s, x] ∩ C ≠ ∅ iff all of the following hold:
* f_i(x) ≤ 0 for every *front* face (c_i < 0);
* f_i(x) ≤ 0 for every face with c_i = 0;
* (c_j a_i − c_i a_j)·x + (c_j d_i − c_i d_j) ≤ 0 for every pair (front i, back j with c_j > 0).

*Proof.* Write y(t) = s + t(x − s) with t ∈ [0, 1], so that f_i(y(t)) = (1−t) f_i(s) + t f_i(x).
1. A front face (f_i(s) > 0) is satisfiable only if f_i(x) < f_i(s). It then gives the lower bound
   t ≥ f_i(s)/(f_i(s) − f_i(x)), which is ≤ 1 iff f_i(x) ≤ 0.
2. A back face (f_j(s) < 0) with f_j(x) > f_j(s) gives the upper bound t ≤ −f_j(s)/(f_j(x) − f_j(s)) > 0. A back face
   with f_j(x) ≤ f_j(s) holds for all t ≥ 0.
3. A face with c_i = 0 needs f_i(x) ≤ 0, because t = 0 is excluded (s ∉ C).
4. Fourier–Motzkin: max(lower) ≤ min(upper). Cross-multiplying the positive denominators gives exactly the pair rows.
   When f_j(x) ≤ f_j(s), the pair row is implied by the front row. ∎

This is `rind3d.rendering.polyhedra.shadow_planes`, the construction of the certified renderer that produced X.
* The GPU builds enclosures of positive multiples of these rows with directed binary64 intervals.
* An uncertifiable face sign or row width sends the source to exact host rows (`fastplanes`, which is identical to
  `shadow_planes`).
* "Inactive" means: inside the closed shadow of some body. Rows are tested as ≤ 0.

**(c) Sign decisions.**
1. *FP32 filter.* Rows are normalised to max|coef| ≤ 1. Coordinates are exact in binary32, and the code is compiled
   with `--fmad=false`. Then |fl32(v) − v| ≤ 1.25·2⁻²⁰ < E32 = 2⁻¹⁸ (GPUVERIFY.md; CERTIFICATE_AUDIT §4). The same
   bound holds for the box minima and maxima.
2. *FP64 intervals* otherwise.
3. *UNKNOWN* voxels otherwise. These are decided by the exact native GMP predicate. An uncertified answer raises an
   error.

**(d) SOLID.** The stored SOLID crop is checked once per window against exact membership (`solid_check`). Class-2
receivers are source-independent, so they constrain nothing.

**Validation of the implementation [empirical].**
* Gate A: on every candidate class (uniform in B, near the true source, near engine vertices, uniform in the world,
  within 1 ulp of face planes, and forced FP64 / forced exact builds), `field_diff = batch_disagree = bad_violation = 0`
  (GPUVERIFY.md).
* §7.4: the failure frequency of 20,000 verifier calls on U_B matches the *exact* |F|/|B| from Astra's rational
  solutions.

### 3.4 Lemma 4 (sequential test under adaptivity) [proven]

Let H_{r−1} be the σ-field of everything the labeller computed before the first test point of round r. That is:
* all earlier test points and their verdicts;
* the `seed_items` and `rng_c` outputs consumed so far;
* every engine output.

Assumption N4 (§6.1): the streams are independent i.i.d. Uniform(0,1) sequences.

**1. B_r and θ_r are H_{r−1}-measurable.** The engine, the filter and the receiver set are deterministic functions of
H_{r−1}, so the record list and hence Q_r are too. In particular, B_r never depends on round-r test points. Those enter
X′ only for round r + 1.

**2. Fresh draws.** The number of `rng_s` outputs consumed before round r is a stopping time. So the subsequent outputs
are i.i.d. uniforms independent of H_{r−1} (the strong Markov property of an i.i.d. sequence). Hence, given H_{r−1},
the round-r points are i.i.d. Q_r. By Lemma 3, the pass indicators are i.i.d. Bernoulli(1 − θ_r), with
θ_r := 1 − Q_r(F).

**3. Fixed thresholds and monotonicity.** The certificate event of round r is E_r = {K₅₁₂ ≤ a₁} ∪ {K₁₀₂₄ ≤ a₂}. Here
K_n counts failures among the first n points, and (a₁, a₂) depend on r only.
* Early stopping on K > a₂ only removes outcomes outside E_r.
* E_r is increasing in the pass indicators. Couple them as 1[U_k < 1 − θ]: then P_θ(E_r) is non-increasing in θ.
* Therefore, for any θ₀ and on {θ_r ≥ θ₀}: P(E_r | H_{r−1}) ≤ α_r(θ₀) := P_{θ₀}(E_r).
* α_r(θ₀) is computed exactly in rational arithmetic (§5).

**4. Union over rounds.** Then

       P( certify at some r with θ_r ≥ θ₀ ) ≤ Σ_r E[ 1{round r reached} α_r(θ₀) ] ≤ Σ_{r=1}^{12} α_r(θ₀).

   The data-dependent number of rounds, the adaptive choice of receivers, the reuse of failing points as constraints and
   the deterministic exits are all covered. None of them changes the law of the round-r points given H_{r−1}.

**5. Same B.** The label is built from the same `res` that `BSampler` sampled. A vertex-pool re-run (`:176-181`) rebuilds
the same B deterministically and resamples from the new object. ∎

### 3.5 Theorem (per label, and campaign) [conditional on §6.1, η ≤ 10⁻⁶]

Fix the scene, window and channel with Vol(F) > 0, and let η bound (★) for every B_r. If the label is `certified_stat`
at round r and TV(q̂, q_F) > ε = 1/20, then by (★) θ_r > 1/20 − η. By Lemma 4 with θ₀ = 1/20 − 10⁻⁶:

    P( certified_stat ∧ TV(q̂, q_F) > 0.05 ) ≤ Σ_r α_r(1/20 − 10⁻⁶) ≤ 1.37·10⁻⁹ < δ = 2.083·10⁻⁹.

**Campaign corollary.** Let the campaign make A ≤ N_max = 4.8·10⁶ statistical label attempts (§4). The union bound
needs no independence between labels, so:

    P( ∃ certified_stat label with TV > 0.05 ) ≤ A × 1.37·10⁻⁹ ≤ 0.0066 ≤ family_delta = 0.01.

The 0.01 bound holds for up to 7.3·10⁶ attempts. ∎

### 3.6 Zero or tiny feasible volume [proven]

* **Vol(F) = 0.** q_F is undefined. Then Q_r(F) = 0 for the continuous law (the float realisation is covered by N2). So
  θ_r = 1, and `certified_stat` occurs with probability ≤ Σα_r.
* **No feasible point at all.** K reaches a₂ + 1 within the first chunk of every round. After R rounds, or as soon as
  every failure lacks a violated receiver (all failures are domain rejections, `:213-214`), the label goes to
  `fallback_exact`. A deterministic bound with Σlo = 0 is 1.0, which gives `uncertified`.
* **Positive volume.** Every certificate has n − K ≥ 511 exactly verified feasible points, so F ≠ ∅. Positive volume
  itself is part of the statistical statement. The true source s_j* ∈ F always, by construction.
* **Single-cell labels.** These need F ≠ ∅ (true source) and F ⊆ B (N1).

## 4. Exact scope

* **Randomness.** Only the labeller's keyed streams (N4).
  * The dataset (geometry, windows, observations) is fixed. The theorem holds for every realised dataset. Its own
    streams are distinct SeedSequence entropy tuples: `[seed, slot, stream]` for the dataset, and
    `[base_seed, slot, w, s, 3]` spawned for the labeller.
  * The base seed 20261118 is fixed by the contract before labelling. The statement is a property of the procedure. It
    is not a probability statement about a label after the fact.
* **Which labels.** Status `certified_stat` only.
  * `certified` single-cell labels are deterministic given N1.
  * `certified` labels from the exact fallback carry the tau-pencil engine's float `tv_bound`. They depend on the same
    engine soundness assumption as the tau-pencil reference labels (CERTIFICATE_AUDIT C1, A1). They are outside this
    theorem.
* **Label units (contract v3).** A label unit is (slot, window, channel) with an informative (direct-interface) channel.
  * Windows per scene: 12. The quota is 4 + 4 + 4; coverage slots come out of the quota (`sampling.side_streams`).
  * Channels: k ≤ 4.
  * Hence at most 100,000 × 12 × 4 = 4.8·10⁶ units, and `n_tasks` = 4.8·10⁶.
  * The realised number of informative units is smaller.
* **Attempts and re-rolls.** An *attempt* is one run of the procedure on a key with one code and configuration.
  * A rerun with the same key, code and configuration is a *replay*: the result is bit-identical (verified 3/3, §7.6),
    so it is not a new attempt.
  * A rerun with a different `base_seed`, δ or code version is a new attempt and must be counted in A.
  * Since `bc6d860`, the contract pins the seed and budget (D1), and `validate` rejects foreign seeds, keys or budgets
    (D2).
  * Code changes are not pinned (H3). The release procedure must freeze the FastLabel code and DEFAULTS
    (`config_sha256`), or count reruns after a change.
* **Not covered.**
  * The Student's world-averaged posterior.
  * Curved occluders. These are excluded from T*_v3, and FastLabel now refuses curved containers (H1).
  * Labels produced under contract v2 or the 1K budget (δ = 0.01/159,802). The theorem applies to the 1K with that δ.

## 5. Statistical parameters (exact, `design_check.py`)

**Parameters.** ε = 0.05; n₁ = 512, n₂ = 1024; R = 12; δ = 0.01 / 4.8·10⁶ = 2.083·10⁻⁹; δ_r = δ 2⁻ʳ.

**How the thresholds were checked.** The thresholds come from `design2` (scipy floats). They were recomputed in exact
rational arithmetic at the true boundary 1/20; note that the float 0.05 is slightly above 1/20.

| r | δ_r | a₁ | a₂ | α_r(1/20) exact | α_r(1/20 − 10⁻⁶) | ≤ δ_r | a₂ maximal |
|---|---|---|---|---|---|---|---|
| 1 | 1.04e-9 | 1 | 14 | 4.899e-10 | 4.902e-10 | yes | yes |
| 2 | 5.21e-10 | 1 | 14 | 4.899e-10 | 4.902e-10 | yes | yes |
| 3 | 2.60e-10 | 1 | 13 | 2.073e-10 | 2.075e-10 | yes | yes |
| 4 | 1.30e-10 | 0 | 13 | 1.016e-10 | 1.017e-10 | yes | yes |
| 5–6 | 6.5e-11, 3.3e-11 | 0 | 12 | 2.720e-11 | 2.722e-11 | yes | yes |
| 7 | 1.63e-11 | 0 | 11 | 9.046e-12 | 9.053e-12 | yes | yes |
| 8 | 8.14e-12 | 0 | 10 | 4.961e-12 | 4.964e-12 | yes | yes |
| 9–10 | 4.1e-12, 2.0e-12 | −1 | 10 | 1.031e-12 | 1.032e-12 | yes | yes |
| 11–12 | 1.0e-12, 5.1e-13 | −1 | 9 | 1.888e-13 | 1.890e-13 | yes | yes |
| Σ | 2.08e-9 | | | **1.3595e-9** | 1.3605e-9 | yes | |

**Clopper–Pearson.** The decision is the exact two-stage binomial test above. It is not a Clopper–Pearson rule. The
stored `p_lower` is the CP lower bound at δ_r for the final (n, K), and it is informational. Every certified (n, K) has
P(Bin(n, ε) ≤ K) ≤ α ≤ δ_r, so `p_lower` ≥ 0.95 whenever a certificate is issued. The CP interval alone would ignore
the interim look, so validity comes from α_r.

**Power and cost per round** (exact recursion over the checkpoints).

| failure rate θ = 1 − \|F\|/\|B\| | 0 | 0.001 | 0.002 | 0.005 | 0.01 | 0.02 | 0.03 | 0.05 |
|---|---|---|---|---|---|---|---|---|
| P(certify), v3 δ (round 1) | 1 | 1 | 1 | 0.9997 | 0.905 | 0.086 | 5e-4 | 0 |
| E[points tested], v3 δ | 512 | 560 | 652 | 884 | 1006 | 976 | 731 | 515 |
| P(certify), 1K δ = 0.01/159,802 | 1 | 1 | 1 | 1 | 0.983 | 0.260 | 0.005 | 0 |
| P(certify), per-label δ = 0.01 | 1 | 1 | 1 | 1 | 1 | 0.994 | 0.651 | 0.004 |

**Cost of the campaign-level δ.**
* The number of points per round is unchanged (≤ 1024).
* The thresholds tighten from the 1K's (3, 17) to (1, 14).
* So a B is certified only when it is already ≥ 99 % tight. Looser B's (1–2 % slack) need another refinement round
  instead.
* An all-pass single-stage test at this δ would need ⌈ln δ_r / ln 0.95⌉ ≈ 403 points with zero failures.

**Measured [empirical]** over 50 statistical labels at the production configuration (cmp_v3: 18 tasks each at L = 64,
128 and 256):
* rounds per label: 1 round (6 labels), 2 rounds (39), 3 rounds (5);
* tested points per label: mean 878, max 2176;
* 74 % of the final rounds stop at stage 1;
* the verifier costs ~0.1–0.3 s per 1024 points on an RTX 3060 laptop GPU.

## 6. Numerical assumptions, volumes and serialization

### 6.1 Assumptions behind η in (★) [stated, not proven]

**N1 Containment: μ = |F \ B*| / |F| ≤ η₁.**

B is a superset of F in exact arithmetic:
* dropping receivers only removes constraints;
* the filter only removes regions that miss a box enclosure of B_{r−1};
* the window box is excluded.

The code decides with float margins: plane values 10⁻¹¹, gating AMARG 10⁻¹², and tau comparisons 10⁻¹²(1+|t|)/|f|
(CERTIFICATE_AUDIT A1, A2). Almost all margins are conservative: an undecided comparison causes more splitting and never
an OUT decision. Three are not:
* the window-box rule declares a region OUT when its box lies within the window box *expanded* by 10⁻¹²
  (`engine_p2.py:699-707`), so a ≤ 10⁻¹²-thick shell next to the box can be lost;
* `poly.clip` snaps vertices within 10⁻¹³ of a plane (EPS);
* the filter boxes are vertex bounding boxes ± 10⁻¹³ (`filters.py`), queried ± 10⁻¹².

Each loss is a shell of thickness ≲ 10⁻¹² around a decision surface. It is negligible unless F itself is that thin.

**N2 Sampling law: η_S = TV(Q_r, U_{B*}) ≤ η₂.**

The causes are: `hi` vs hull volume (≤ 3.3·10⁻⁸ relative per piece, ≤ 1.2·10⁻¹¹ mass-weighted, CERTIFICATE_AUDIT C3),
Delaunay joggle, and binary64 realisation of continuous points.

**N3 Projection: η_cell ≤ η₃.**

A piece hull may cross its cell boundary by float error.

**N4 PRNG.**

PCG64 streams under SeedSequence are modelled as independent i.i.d. uniforms.

**The theorem needs η = η₁ + 2η₂ + η₃ ≤ 10⁻⁶.**

[empirical] Every exact comparison in §7 is consistent with η ≤ 7·10⁻¹⁰. The rigorous references show:
* largest relative excess of a proved lower bound of |F ∩ c| over the float volume of B ∩ c: 7.0·10⁻¹⁰ (Astra) and
  6·10⁻¹⁸ (verified tau-pencil);
* no cell where a reference proves mass and B has none;
* the sampler/verifier failure counts match the exact |F|/|B| (§7.4).

### 6.2 Volumes of B

The per-cell volumes Σhi are float64 polytope volumes. They are **not enclosures**: their relative error is ~10⁻¹³–10⁻⁹
for small pieces, and the volume error enters (★) through η₂ and η₃. The certificate never needs |B| itself, only the
law Q_r that was actually sampled and its cell image q̂.

### 6.3 Serialized vector and normalisation [proven]

* **What is stored.** The binary64 volumes b_c = vol(B ∩ c), with lo = hi = b. These are not probabilities.
* **The label.** By rule, the label is q_c = b_c / Σ b in exact real arithmetic. Every binary64 is a rational, so this
  is well defined.
* **Astra's counterexample.** A float vector whose float sum is 1.0 need not be normalised: 3 × fl(1/3) = 1 − 2⁻⁵⁴. It
  therefore does not apply to the stored object.
* **What the loader returns.** It returns q̂_c = fl(b_c / fl(Σb)). For n cells and any summation order:
  * fl(Σb) = Σb (1 + θ) with |θ| ≤ γ_{n−1};
  * q̂_c = q_c (1 + δ_c)/(1 + θ) with |δ_c| ≤ u = 2⁻⁵³.
  * Hence ½ Σ_c |q̂_c − q_c| ≤ γ_{n+1}/2 < 1.2·10⁻¹⁰ for n ≤ 128³.
* **Measured** on 75 labels [empirical]:
  * max |Σq̂ − 1| = 2.6·10⁻¹⁶;
  * max TV(q̂, q) = 1.3·10⁻¹⁶;
  * 25/75 are exactly normalised (the one-hot labels).
* **Effect on the theorem.** The rounding is part of η₃. The dtype is float64 throughout: `loader.py:136`, `labels.py:62-66`.
* **Schema.** The 100K schema needs no change. The rule is now stated in the loader docstring.
  * Proposed contract text (for the release branch): `labels.normalization = "the label is lo / sum(lo) in exact
    arithmetic on the stored binary64 values; tv_bound refers to it; any binary64 normalisation is within TV 1.2e-10"`.
  * Storing exactly normalised dyadic weights (integers summing to 2⁵³) is a valid alternative. It is not needed for the
    certificate.

## 7. Empirical validation

Scripts are in `experiments/fastlabel_v34_validation/`. Every TV is computed exactly in rational arithmetic, with no
sampling. The outputs are recorded in `results/`.

### 7.1 References and method

| reference | tasks | nature | independent of FastLabel's float engine? |
|---|---|---|---|
| `tp_engine`: cmp_v3 tau-pencil labels | 54 (cmp_v3) | deterministic per-cell [lo, hi] from the same float engine, with all receivers and no filter | no (shares A1); tests the receiver subsetting, the rounds and the filter |
| `tp_cert`: certify-verified tau-pencil labels | 15 (cmp_v3) | per-cell lo ≤ Vol(F ∩ c) ≤ hi + pool, re-proved by the exact certify verifier | yes |
| `astra`: Astra's rational global inverse | 4 cmp_v3 + 21 fresh tasks (26 comparisons); 14 exact solutions | proved inner pieces (exact rationals) plus a proved upper bound of \|F\| | yes |

**How the TV is bounded.** Take the label q = b/Σb and a reference enclosure (lo, hi, pool). Let qt = lo/Σlo, and let
t_ref be the exact greedy supremum of TV(qt, v/|v|) over the enclosure (`certify.tvb` algorithm). For every q_F
consistent with the reference:

    TV(q, qt) − t_ref ≤ TV(q, q_F) ≤ TV(q, qt) + t_ref.

So the upper end is the worst case over the reference. A *violation* would be TV(q, qt) − t_ref > 0.05.

**Other quantities.**
* The pre-projection gap 1 − |F|/|B| is enclosed by [1 − (Σhi + pool)/Σb, 1 − Σlo/Σb].
* The containment excess is Σ_c (lo_c − b_c)⁺ / Σlo.

**Where the labels come from.**
* The 21 fresh tasks (`run_fl.py`) are Astra's held-out and fresh windows. They use the same v3 canary root and were
  labelled at the exact v3 production configuration (n_tasks = 4.8·10⁶, base_seed 20261118). They also have a 20,000-point
  validation stream that is separate from the certificate.
* The cmp_v3 labels were produced with the same configuration (`experiments/localfirst/compare.py`).

### 7.2 Results by reference (`results/SUMMARY.json`)

| reference | comparisons | statistical | max TV upper (stat.) | max 1 − \|F\|/\|B\| upper | max containment excess (relative) | TV lower > 0.05 |
|---|---|---|---|---|---|---|
| tp_engine | 54 | 39 | 0.0144 (000059_w11_s3; t_ref 6.9e-3) | 0.0142 | 5.7e-10 | 0 |
| tp_cert | 15 | 10 | 0.0098 (000007_w09_s2) | 0.0097 | 6.1e-18 | 0 |
| astra | 30 | 17 | 0.0437 (000025_w04_s0; t_ref 2.2e-2) | 0.0218 | 7.0e-10 | 0 |

**Per statistical label, against the tightest reference (50 labels; 24 with an independent reference).**
* max TV upper: 0.0437. This is 000025_w04_s0, where Astra's enclosure is loose (t_ref = 0.022). Its validation stream
  gives 1 − p̂ = 2.5·10⁻⁴, so the true TV is ≈ 3·10⁻⁴.
* max TV where t_ref < 10⁻³: **0.0141** (000038_w00_s1). Astra's enclosure there has slack 1.4·10⁻¹⁵, so the true value
  is 0.01411 to 14 digits.
* median TV upper: 4.9·10⁻⁴.

**The 10 largest TV upper bounds:**

| key | L | reference | TV(q, q_F) ≤ | t_ref | 1 − \|F\|/\|B\| |
|---|---|---|---|---|---|
| 000025_w04_s0 | 128 | astra (held-out) | 0.0437 | 2.2e-2 | ≤ 0.0218 (validation: 2.5e-4) |
| 000012_w00_s2 | 64 | astra (held-out) | 0.0220 | 1.2e-2 | ≤ 0.0117 (validation: 0 / 20,000) |
| 000059_w11_s3 | 256 | tp_engine | 0.0144 | 6.9e-3 | [7.8e-4, 7.6e-3] |
| 000038_w00_s1 | 64 | astra (archive) | **0.01411** | 1.4e-15 | 0.01422 |
| 000004_w04_s0 | 128 | astra (held-out) | 0.0134 | 1.2e-2 | ≤ 0.0117 |
| 000037_w06_s0 | 128 | tp_engine | 0.0123 | 4.2e-11 | 0.0128 |
| 000019_w02_s0 | 64 | tp_engine | 0.0117 | 8.5e-16 | 0.0119 |
| 000007_w09_s2 | 256 | tp_cert | 0.0098 | 1.1e-4 | [9.63e-3, 9.74e-3] |
| 000049_w03_s0 | 64 | tp_cert | 0.0073 | 2.6e-12 | 7.27e-3 |
| 000069_w01_s0 | 64 | tp_cert | 0.0065 | 7.0e-6 | 6.58e-3 |

**Single-cell labels (`certified`, 25).** All agree with the references. TV = 0 against tp_engine and tp_cert, and
≤ 10⁻²⁷ against Astra's exact one-cell solutions.

### 7.3 Containment and per-cell bias

* **Containment** [empirical].
  * No reference proves mass in a cell that B lacks: "reference mass missing" = 0 in all 99 comparisons.
  * A proved lower bound exceeds B's float cell volume only at the float-noise level: ≤ 7.0·10⁻¹⁰ of the total mass, and
    ≤ 1.8·10⁻⁷ of one cell's own mass.
  * Against the verified tau-pencil enclosures the excess is ≤ 6·10⁻¹⁸.
* **Systematic bias** [empirical]. These figures are over the statistical labels with a tight reference
  (t_ref < 10⁻³).
  * On F's support, q_B(c)/q_F(c) is essentially the constant p = |F|/|B|. The mass-weighted 5 %, 50 % and 95 %
    quantiles of the ratio differ from p by a median of 10⁻¹⁰.
  * The largest deviation is 5·10⁻³, at 000061_w01_s1. There the rim B \ F shares cells with F, and 9 % of F's cells
    have a ratio > 1.
  * The missing mass 1 − p lies on that rim. Cells outside F's support carry ≤ 0.7 % of q_B.
  * Where 1 − p > 10⁻⁴, TV(q_B, q_F) / (1 − p) lies in [0.40, 1.00]. So the projection removes at most 60 % of the
    pre-projection gap, and usually almost nothing. Bound (★) is close to tight.
  * The error is always support inflation (a uniform down-weighting plus a rim), never missing support.
  * The observed 1 − |F|/|B| is ≤ 1.4 % for every label with a tight reference, far below ε. This matches the power
    profile of §5: a B is certified only when it is ≲ 1–2 % loose.

### 7.4 Verifier and sampler against exact volumes (fresh runs, references with t_ref < 10⁻³)

| key | validation failures / 20,000 | exact 1 − \|F\|/\|B\| | binomial p-value |
|---|---|---|---|
| 000001_w00_s0 | 30 | [1.64e-3, 1.98e-3] | 0.71 |
| 000006_w00_s0 | 7 | 4.160e-4 | 0.82 |
| 000009_w00_s0 | 7 | 4.142e-4 | 0.83 |
| 000004_w00_s0, 000004_w08_s0, 000006_w04_s0, 000026_w04_s0 | 0 | ≤ 3.4e-10 | 1.0 |

The exact GPU verifier, applied to points of `BSampler`, rejects at the rate fixed by Astra's exact rational |F|/|B|.
This is an end-to-end check of Lemma 3 and of N2.

### 7.5 Serialization and certificate rules

* **Serialization.** See §6.3: 75/75 labels lie within the proven bound.
* **Certificate rules.** `validate.check_stat_certificate` accepts all 11 fresh production certificates against
  contract v3. Each certificate records, for example: `delta_label = 2.0833e-9`,
  `policy = "simultaneous over 4800000 planned labels"`, `seed = [20261118, slot, w, s]`.

### 7.6 Replay

Rerunning FastLabel on 3 cmp_v3 tasks (000038_w00_s1, 000007_w09_s2, 000059_w11_s3) in a new process reproduces the
saved labels bit for bit (`results/replay_check.txt`). So a same-key rerun is a replay, not a re-roll.

## 8. Pipeline integrity (100K), defects and fixes

Every path below was audited: worker → pool → coordinator → journal → consolidation → loader.

**Result: no timed-out, killed, errored, OOM or partially written task can produce a certified-looking record**
[proven by code inspection].
* **Worker and pool.** Exceptions return `{key, error}` without arrays (`labels.py:128-130`). Timeouts, crashes and RSS
  overruns are reported by the pool without a payload (`pool.py:91-109`).
* **Coordinator.** Only `kind == "result"` with arrays enters the journal (`coordinator.py:398-406`). Failures become
  `timeout`, `memory_limit` or `engine_failure` with no cells.
* **Journal.**
  * Records carry a sha256, and torn records are ignored.
  * The index line is written after the fsync.
  * The first record of a key wins.
* **Consolidation.** The status map is copied from the meta status.
* **Loader.** It gates on the meta status, the presence of cells and `tv_bound` ≤ 0.05.
* **Zero-mass labels.** These are always `uncertified`.
* **Timeouts.** `timeouts_s` (1800/3600/7200 s by side) are part of the contract digest.

Defects found and fixed in commit `bc6d860`, each with regression tests in
`RIND3D_Dataset100K/tests/test_fastlabel_certificate.py` (no GPU):

| id | defect | fix |
|---|---|---|
| **D1** (real) | `run_node --fastlabel-base-seed / --fastlabel-n-tasks` silently overrode the contract (`coordinator.py`: `fl.update(o["fastlabel"])`). For example, `--retry-failed --fastlabel-base-seed X` re-rolled failed labels outside the union bound. The "never re-roll" rule was documentation only. | `coordinator.fastlabel_config`: options may only fill fields the contract lacks. A conflicting base_seed, n_tasks, family_delta or eps raises `SystemExit`. Test: `test_fastlabel_options_cannot_override_contract`. |
| **D2** (real) | `validate` accepted a `certified_stat` label with a foreign seed, a wrong δ, an inconsistent stage, a wrong design, or lo ≠ hi. It checked only kind, eps, 0 < δ < 1, K ≤ a₂ and n_tasks. | `validate.check_stat_certificate`. It checks: the contract eps and tv_bound; delta_label = family_delta/n_tasks; n_tasks; seed = [contract base_seed, slot, w, s]; delta_round = δ·2⁻ʳ; a stage decision consistent with (n₁, n_max, a₁, a₂); (a₁, a₂) = design2(...); q_B arrays finite, ≥ 0, lo = hi, positive total; engine version. The status map must also equal the meta status. Tests: 1 acceptance test plus 16 single-mutation rejections. |
| **H1** (guard) | FastLabel had no curved-occluder guard. The v3 profile excludes spheres (polyhedral families only; the curved contract has no label engine), but `window_from_100k` would have silently labelled a curved container against its polytope-only bundle. | `inputs.window_from_100k` raises on a non-empty `balls` array. Test: `test_window_from_100k_refuses_curved_containers`. |
| design regression | (none found) | `test_v3_design_exact_budget`: exact α_r ≤ δ_r at 1/20 for the contract budget. |

Remaining hardening (not fixed here; for the release branch):
* **H2.** The loader trusts that `validate_labels` / `verify_root` was run. It does not re-check certificates or the
  consolidated sha256.
* **H3.** `config_sha256` is recorded but not pinned. A code or DEFAULTS change between a run and its `--retry-failed` is
  an unrecorded new procedure. Pin the FastLabel version and DEFAULTS hash in the contract, and check them in `validate`.
* `run_node.py` defaults to contract v2. A v3 campaign must pass `--contract …_v3.json`. The root's CONTRACT.json
  digest prevents mixing afterwards.

## 9. Limitations

1. **Not a deterministic certificate.** Each `certified_stat` label is wrong with probability ≤ 1.37·10⁻⁹. The campaign
   statement (≤ 0.01) relies on attempt accounting (§4).
2. **F ⊆ B is not proven.** It rests on the float margins of the tau-pencil engine (N1). Three rules are
   non-conservative at the 10⁻¹²–10⁻¹³ level. The theorem tolerates a relative missing mass of up to 10⁻⁶, and the
   measured excess is ≤ 7·10⁻¹⁰. A pathological F that is thinner than 10⁻¹² near a decision surface would fall outside
   the analysis.
3. **Volumes of B are float, not enclosed** (§6.2).
4. **FP32 filter bound.** The verifier's FP32 error bound is a derivation (GPUVERIFY.md). Its implementation is
   validated by Gate A and §7.4, not machine-checked.
5. **PRNG idealisation (N4).** The base seed is published.
6. **Coverage of the empirical validation.** 50 statistical labels from the v3 canary scenes (L = 64…256), 24 of them
   with independent references.
   * The Astra references cover L = 64 (11 tasks), 128 (4) and 256 (10). Every L = 256 one is a single-cell solution.
   * The largest-TV cases with loose enclosures (000025_w04_s0, 000012_w00_s2) are bounded only to ≤ 0.044 and ≤ 0.022
     by the references. Their validation streams indicate ≲ 10⁻³.
7. **Scope of the target.** It is conditional on the complete geometry (Teacher target). Exact-fallback `certified`
   labels carry the engine's float `tv_bound` and are outside the theorem.

## 10. Why this is appropriate for the first release, and the Gate B decision

* **Justified.** The guarantee is explicit, mathematically justified and conservative. Its constants are verified in
  exact arithmetic, and the design keeps a margin of ≥ 5.8 % per round at the boundary 1/20 − 10⁻⁶.
* **Failure is conservative.** A failed test never yields a statistical label. It refines B, or falls back to the exact
  engine and records a deterministic, weaker or `uncertified` status, which the loader does not accept.
* **Agrees with independent references.** On every independent exact reference, the labels agree to the precision the
  references allow. Where the references are tight, the largest TV is 0.014, and no label exceeds 0.05. The bias is
  understood: uniform down-weighting by |F|/|B| plus a thin rim.
* **Recorded and reproducible.** Each label carries a reproducible certificate (n, K, round, stage, δ, seed), and replay
  is bit-identical.
* **Cost.** Median 1.6 s per label, against 12.4 s for the full tau-pencil (LOCALFIRST_COMPARISON.md). This is what
  makes the 100K feasible.

**Gate B: PASS**, under these conditions:
1. Port `bc6d860` (D1, D2, H1 and the tests).
2. The contract keeps n_tasks = 4.8·10⁶, family_delta = 0.01, eps = 0.05 and base_seed = 20261118, and the release
   documents the normalisation rule (§6.3).
3. The release procedure freezes the FastLabel code and DEFAULTS for the campaign (H3), or counts every post-change
   rerun as an extra attempt. The budget allows up to 7.3·10⁶ attempts.
4. User-facing documentation describes `certified_stat` as a statistical certificate (§0). It must not be described as
   deterministic.

## 11. Reproduction

```bash
source Phase4_Multi_Source_3D/RIND3D_HANDOFF/env.sh; export PYTHONDONTWRITEBYTECODE=1
cd Phase4_Multi_Source_3D/RIND3D_Posterior/experiments/fastlabel_v34_validation
$RIND3D_LABEL_PY -B design_check.py results/design_check.json             # §5 (CPU, ~1 min)
$RIND3D_LABEL_PY -B compare_refs.py results/compare_cmp_v3.jsonl           # §7, cmp_v3 vs tp_engine / tp_cert / astra
$RIND3D_LABEL_PY -B run_fl.py results/fresh_astra --validate 20000 <keys>  # GPU, one process (keys: results/fresh_astra.log)
$RIND3D_LABEL_PY -B compare_refs.py results/compare_astra_fresh.jsonl results/fresh_astra
$RIND3D_LABEL_PY -B summarize.py                                           # results/SUMMARY.json
cd ../../../RIND3D_Dataset100K && $RIND3D_LABEL_PY -m pytest -q tests/test_fastlabel_certificate.py
```

Inputs are read-only:
* `RIND3D_100K_v3_work/cmp_v3`, `xlf/ref/labels` and `perf`;
* Astra's `experiments/global_inverse/records/grids*`, loaded as JSON only. No Astra code is imported.
