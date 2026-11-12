# Error-controlled global source-posterior labels for RIND3D: fixed target, the tau-pencil method, certificates

Status markers:
* **[P]** proved here;
* **[V]** verified numerically against an independent implementation;
* **[M]** measured;
* **[A]** an assumption, stated explicitly;
* **[E]** model-based estimate.

## 1. Fixed target (Deliverable A)

### 1.1 Generative law of RIND3D-Lite v0.1 (read from code)

* **Geometry.** Σ = ∪_c C_c is a finite union of closed convex polyhedra C_c = {x : f_k(x) = n_k·x + d_k ≤ 0},
  with canonical rational rows. We use their float64 roundings, renormalised to unit normals.
* **Source.** `lite_acceptance.sample_source`: p ~ U(0,1)³ (PCG64 float64, i.e. the 2⁻⁵³ grid), rejected if
  p is in the solid or d(p, Σ) ≤ 2⁻²⁶. This is uniform on D(G) = {p ∈ (0,1)³ : d(p, Σ) > 2⁻²⁶}.
  * A further eligibility test R(s) (`source_eligibility.certificate`, on 16³-block supports of the whole
    field) rejects the source and redraws it with the same geometry.
  * So does an INCOMPLETE window construction (`campaign_scene.prepare`).
* **Field.** A FREE receiver at the voxel centre x = (v + ½)/1024 is dark iff the closed segment [s, x] meets Σ.
  SOLID voxels carry no constraint.
* **Windows (v0.1 "mixed" mode, `stratified_windows.select`).** Per side σ ∈ {64, 128, 256}:
  * a pool of 8 × 85 (86) accepted proposals: uniform integer origins from a scrambled Sobol stream; accepted
    iff the window is informative (lit-FREE and dark-FREE both present) and excludes the source voxel;
  * then 85 (86) picks, each from one of three components:
    * natural — any remaining pool member;
    * uniform — the lit-ratio bin with the largest dataset-level deficit;
    * extreme — bins 0 and 9;
  * each pick is the best spatial gain among 32 random eligible members.

  The Sobol scramble and pick RNG come from `keyed_seed(window_seed, scene, …)`. The HMAC seed protocol
  derives them independently of the source seed.

### 1.2 The posterior that is the labelling target [P]

For a window W with observation O = O_W (solid mask and FREE responses):

    q(s | O, K, W)  ∝  1[s ∈ F_W] · R(s) · A(s) · λ_W(s)        on D(G),

where:
* F_W = {s ∈ D(G) \ B(W) : every FREE receiver x of W has [s, x] ∩ Σ ≠ ∅ ⇔ O(x) = dark};
* R(s) ∈ {0, 1} is the eligibility test;
* A(s) is the probability that window construction is COMPLETE for source s;
* λ_W(s) = P(W ∈ selected set | K, s), marginalising the source-independent selection randomness.

**Definition (natural posterior).** q_nat = Uniform(F_W). It equals q exactly iff R·A·λ_W is constant on F_W.

### 1.3 Resolution, metric, success criterion

* Output cells: the 128³ grid of the Student (16³ coarse × 8³ fine).
* Label: cell masses q_c = ∫_{c} q.
* Metric: TV(q̂, q) = ½ Σ_c |q̂_c − q_c| over the 128³ cells. This bounds the logit-gradient bias of
  softmax cross-entropy: ‖Δq‖₁ = 2·TV.
* A window is **resolved** only if its *reported upper bound* is ≤ 0.05 (primary) or ≤ 0.10 (secondary), with
  one of:
  * **deterministic** — a proof under exact arithmetic, implemented with filtered float64 predicates
    (§5); or
  * **statistical** — ≥ 99% coverage over the labeller's own sampling randomness, for that window.

  Vacuous or missing bounds count as unresolved.

## 2. Selection-law mismatch (what can and cannot be said)

* **R(s).** It rejected 3 of 1,003 v0.1 sources. On F_W it is constant except in rare cases [A, not
  verified per window].
* **λ_W(s).** It depends on s through the *whole* field: the pool composition and the bin availability.
  Marginalising the Sobol scramble gives, to leading order [E: model; pool truncation, spatial-gain competition
  and bin-availability effects ignored],

      λ_W(s) ≈ k_nat / (N_o P_valid(σ, s)) + k_b / (N_o P_b(σ, s)),

  where:
  * P_valid is the fraction of the N_o origins that give a valid window;
  * P_b is the fraction valid and in W's lit-ratio bin b (b is fixed by O);
  * k_nat and k_b are the natural and the bin-b picks.
* **Measured.** `code/selection_study.py` renders the full 1024³ field for sources drawn from the exact q_nat
  and estimates these frequencies with common random origins.
* **Consequence.** The discrepancy TV(q_nat, q) is estimated from those samples (`code/selection_summary.py`,
  which also reports the noise floor of the estimate).
  - It vanishes at the cell level when q_nat lies in one cell: reweighting inside a cell leaves the cell masses
    unchanged.
  - It is 0.02–0.11 for the extended posteriors measured (report §A).
* **The v0.3 inclusion design** (RIND3D_Training_v03 `docs/SELECTION_LAW.md`) is different:
  - proposals are uniform origins drawn independently of (G, S);
  - acceptance depends on O_W and "no source in the box";
  - documented weights reproduce the source-excluded natural population, normalised per (scene, side).

  The posterior of that weighted population is

      q_v0.3(s) ∝ 1[s ∈ F_W] / (1 − P_cs(s)),

  where P_cs(s) = ∏_k n_k(s)/(1025 − σ) is the fraction of origins whose window contains s. This factor is
  explicit; its variation over F_W is at most 3.8% for σ = 256 (0.3% for σ = 128, 0.03% for σ = 64), so
  TV(q_nat, q_v0.3) ≤ 0.019 and it can be applied exactly. [P, given the documented design.]
* **Consequence (continued).** Where F_W is extended, it is
  *not* negligible (§7 of the report). The labels computed in this study are exact for q_nat. For the v0.1
  target they carry this model-estimated discrepancy, which is reported separately and **not** included in
  any certified bound.

## 3. Geometry: the tau-pencil representation

Notation: f_i(p) = n_i·p + d_i are the face functions of C = C_c.

**Lemma 1 (shadow polyhedron, first study, [P, V]).** For s ∉ C with no f_i(s) = 0,

    S_C(s) := {x : [s, x] ∩ C ≠ ∅} = {x : f_i(x) ≤ 0 ∀ i ∈ F(s)} ∩ {g_e(s, x) ≤ 0 ∀ e ∈ E(s)},

where:
* F(s) = {i : f_i(s) > 0} are the front faces;
* E(s) are the silhouette edges, i.e. adjacent pairs e = (f, b) with f ∈ F(s) and b ∉ F(s);
* g_e(s, x) = f_f(s) f_b(x) − f_b(s) f_f(x).

The rows of Lemma 1 are (c_b n_f − c_f n_b)·(x − s) with c = −f(s), and expanding gives exactly g_e.
They were verified against the canonical exact oracle on 6.6×10⁶ points (`rpl` tests).

**Definition (aspect, pencil coordinate).** A source region R has a *fixed aspect* for C if every f_i has a
constant strict sign on R. For a silhouette edge e = (f, b) define

    τ_e(p) = f_b(p) / f_f(p).

A receiver x is **gated** if f_i(x) < 0 for every front face i.

**Lemma 2 (orthant form) [P].** On a region with fixed aspect, for every gated receiver x and every s in the
region,

    x ∈ S_C(s)   ⇔   τ_e(s) ≤ τ_e(x)   for every silhouette edge e.

An ungated receiver (f_i(x) > 0 for some front face i) is never in S_C(s).

*Proof.* f_f(s) > 0 > f_f(x). Dividing g_e(s, x) ≤ 0 by f_f(s) f_f(x) < 0 reverses it:
f_b(x)/f_f(x) ≥ f_b(s)/f_f(s). The ungated case is the first group of rows of Lemma 1. ∎

So every receiver of a window is an *orthant* in the common coordinates (τ_e(s))_e of the region. The
level sets {τ_e = t} = {f_b − t f_f = 0} are the pencil of planes through the edge line.

**Lemma 3 (quasi-linearity) [P].** On {f_f > 0} (or on {f_f < 0}), every sub- and super-level set of τ_e is a
half-space. Hence:
* on a convex polytope R, min and max of τ_e(s) are attained at vertices of R;
* on a box Q of receiver centres that is gated (f_f < 0 on Q), min and max of τ_e(x) are attained at the 8
  corner receivers of Q.

**Lemma 4 (exact collapses) [P].** Let R have fixed aspect, and let Q be a pure block of receivers (all dark
FREE or all lit FREE), gated for C.
* **(a) Dark, unique candidate.** Suppose no other component can shadow any receiver of Q from R (certified
  NOCOVER). Then "every receiver of Q dark for s" ⇔ τ_e(s) ≤ min_{x∈Q} τ_e(x) for all e. These are |E| half-spaces
  f_b(s) − u_e f_f(s) ≤ 0, with u_e computed at the corners by Lemma 3.
* **(b) Lit, one undecided edge.** Suppose that for all but one edge e* we have max_R τ_e ≤ min_Q τ_e (that
  edge can never exclude). Then "no receiver of Q shadowed by C for s" ⇔ τ_{e*}(s) > max_{x∈Q} τ_{e*}(x): one
  half-space.
* **(c) Certificates.**
  * COVER (every receiver of Q in S_C(s) for every s ∈ R) ⇔ Q gated and max_R τ_e ≤ min_Q τ_e for all e.
  * NOCOVER (no receiver of Q shadowed for any s ∈ R, up to a measure-zero set) ⇐ Q ungated, or
    min_R τ_e ≥ max_Q τ_e for some e.

*Proof.* (a) and (b) follow from Lemma 2 by intersecting orthants. The extremes over Q are at corners by
Lemma 3. (c) is Lemma 2 applied to the extreme receivers. ∎

**Lemma 2W (window-side form) [P, V].** Exchange the roles of source and receiver in Lemma 1 (S_C is symmetric:
x ∈ S_C(s) ⇔ s ∈ S_C(x)). Let F(x) = {i : f_i(x) > 0} be the faces of C seen from the receiver and E(x) its
silhouette edges. Then for s with f_i(s) < 0 for all i ∈ F(x):

    x ∈ S_C(s)   ⇔   f_b(s)/f_f(s) ≥ f_b(x)/f_f(x)   for every e = (f, b) ∈ E(x),

and if f_i(s) > 0 for some i ∈ F(x), then x ∉ S_C(s). With coord_e(p) = −f_b(p)/f_f(p), this is the same orthant
form coord(s) ≤ coord(x), so Lemma 3 and Lemma 4 carry over unchanged.

*Proof.* Lemma 1 from x, divided by f_f(x) f_f(s) < 0. ∎

If no face plane of C meets the box of the window's receiver centres, F(x) is the same for every receiver. Source
regions then need to be split only on the |F| planes bounding G_C = {s : f_i(s) ≤ 0, i ∈ F}, not on the other face
planes of C. TPE uses this form for such components ("W-mode", engine v2/b3) and the source-side form otherwise.
`tests/test_tau_lemmas.py` checks Lemma 2 and Lemma 2W against the segment predicate: 800,000 random pairs, 0
mismatches. A variant with several receiver-side aspect classes per component (engine v3) was implemented and
was correct, but it was slower on real windows, so it is not used for the reported numbers.

**Consequence.** Thousands of receivers along a shadow facet collapse to *one* plane per edge. Receivers deep
inside lit or dark regions are decided as whole blocks. Only two kinds remain as genuinely non-convex
constraints:
* lit receivers with ≥ 2 undecided edges (near silhouette vertices);
* dark receivers coverable by ≥ 2 components.

## 4. Algorithm (TPE, `code/tpl/engine_v2.py`)

1. **Stage 1** (accelerator only; it never certifies IN): an octree to depth d₁ (4).
   * Boxes are refuted by sentinel receivers with the validated generic box test of the first study
     (`rpl.engine2.receiver_status`).
   * Boxes inside B(W) are OUT by definition.
2. **Stage 2**: a BSP over convex polytopes, starting from the surviving depth-d₁ boxes. Each region carries:
   * logical aspect flags, set by face splits;
   * logical τ bounds, set by τ-splits and collapses;
   * the list of still-active items (pure blocks or receivers, each with a component mask).

   Per region:
   * **window box:** split on a face of B(W) if it crosses;
   * **relevance:** component c is irrelevant if conv(R ∪ hull(W)) misses C. This uses the first study's
     certified AABB and face-plane separation;
   * **evaluation passes (Lemma 4):**
     * drop satisfied items;
     * OUT on a contradiction;
     * collapse to half-spaces, then clip and repeat;
     * expand undecided blocks into their 8 children.

     A component whose aspect is not fixed is tested only with the aspect-free hull test.
   * **outcome:**
     * OUT;
     * IN, when no item remains: exact polytope volume, split exactly over the 128³ cells by a recursive
       BSP on the cell grid;
     * otherwise split:
       * on a face plane of a component still needed with unfixed aspect;
       * else on the most central undecided pencil plane of a surviving receiver;
       * else on the cell grid (or by geometric bisection, if all survivors are near-tangent).
   * **budget:** a declared node budget; any region left when it is exhausted is recorded **UNDET** with its
     full volume as the upper bound. Nothing is ever dropped.
3. **Certificate.** Per cell: [L_c, U_c] = [Σ IN volumes, Σ IN + UNDET volumes]. The bound is
   tv_bound(L, U) (first study, `rpl/bound.py`, rigorous). It is 0 when no UNDET mass remains.

## 4b. Clearance shell

The legal domain excludes {d(s, Σ) ≤ 2⁻²⁶}. For each IN polytope P, `code/shell_check.py` bounds the excluded
part from above by Σ_c vol(P ∩ {f_i ≤ 2⁻²⁶ ∀ i}), an exact half-space intersection. It subtracts that from the
cell's lower bound and recomputes the TV bound.

## 5. What "deterministic" means here [A, V]

The proof is for exact arithmetic. The implementation uses float64:
* sign decisions with explicit margins (gating 10⁻¹² on unit normals; region τ ranges widened by
  10⁻¹²(1 + |τ|)/f_f);
* τ thresholds as float quotients;
* polytope clipping with 10⁻¹³ snapping;
* volumes by the divergence theorem, whose clipper was tested against qhull to 4×10⁻¹⁴ relative in the first
  study.

An undecided comparison never certifies anything; it causes splitting or UNDET. The residual numerical error
(volume rounding, and slivers between float and exact thresholds) is not machine-checked. It is
characterised by two independent brute-force checks (§6), and by the agreement of different engine
configurations to ~10⁻¹³ in TV. This is a float-filtered deterministic certificate, not a formal proof.

## 6. Independent verification (`code/tpl/verify.py`)

All checks use brute force: every FREE receiver of the window is tested with the float Cyrus–Beck segment
predicate. No τ coordinates, polytopes or octree are involved.
* **Soundness:** points uniform inside emitted IN polytopes must all be feasible.
* **Completeness:** feasible points found by uniform sampling, in a box around the support and in the whole
  world, must lie inside an emitted polytope.
* **Volume:** the Monte Carlo feasible volume in that box must match the certified volume.

## 7. Statistical comparison methods (`code/tpl/stat.py`)

* **Cover MC.** Stratified uniform sampling of the stage-1 cover cells.
* **Prior MC.** Uniform sampling with sample splitting: the first half fixes the support set, and the second
  half gives the intervals, plus a phantom cell holding the mass outside the support.

Both methods:
* test feasibility exactly (brute force);
* use Clopper–Pearson intervals with Bonferroni over cells and sides, fixed sample sizes and no adaptive
  stopping, so the per-window coverage is ≥ 99% over the sampling randomness;
* certify TV(q̂, q) ≤ TV(q̂, q_L) + tv_bound(L, U).
