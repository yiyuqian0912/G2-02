# Exact posterior labels for RIND3D single-source windows: definitions, lemmas, certificate

Status markers: **[P]** proved here; **[V]** verified numerically against an independent implementation;
**[A]** an assumption, stated explicitly; **[M]** measured.

## 1. The physics, as implemented (RIND3D-Lite v0.1)

* World Ω = (0,1)³; grid N = 1024; receivers are voxel centres x = (v + ½)/N.
* Geometry G: the solid Σ = ∪_c C_c is a finite union of bounded, full-dimensional, closed convex
  polyhedra C_c = {x : n_k·x + d_k ≤ 0}. Every component lies strictly inside Ω. The canonical rows are
  exact rationals, decoded from binary64 parameters (`rind3d.rendering.polyhedra.decode_convex`).
* Response: a FREE receiver x is **dark** iff the *closed* segment [s, x] meets Σ (touching and tangency
  count as hits). It is **lit** otherwise. SOLID voxels carry 0 and are flagged separately; they impose no
  constraint on s.
* Source prior: s ~ Uniform(A_G), with A_G = {p ∈ 2⁻⁵³ℤ³ ∩ (0,1)³ : d(p, Σ) > 2⁻²⁶}, drawn by PCG64
  rejection. At 2⁻⁵³ resolution this is the uniform law on the open legal domain D(G) for every purpose
  here.
* Windows: side σ ∈ {64, 128, 256}. The box is the half-open B(W) = [o, o+σ)/N. A window is valid only if
  s ∉ B(W) and it contains both a lit-FREE and a dark-FREE voxel. Sources are continuous, never
  voxel-centred.

## 2. The posterior target

### 2.1 Feasible set [P]

For a window W with observation O_W (solid mask plus responses at the FREE voxels of W):

    F_W = { s ∈ D(G) \ B(W) : for every FREE receiver x ∈ W,  [s, x] ∩ Σ ≠ ∅  ⇔  O_W(x) = dark }.

The observation is deterministic, so the likelihood is the indicator 1[s ∈ F_W].

Every lit receiver x has C_c ⊂ S_c(x) (Lemma 1), so F_W ⊂ Ω \ Σ automatically. The clearance shell
{0 < d(s, Σ) ≤ 2⁻²⁶} is the only part of the legal-domain constraint that the geometric computation does
not enforce. Its volume is at most area(∂Σ)·2⁻²⁶ (≈ 1.5×10⁻⁸·area). It is carried as an explicit
additive term in the bound.

### 2.2 Posterior under the real v0.1 generator [P]

Let λ_W(s) = P(W is among the 256 selected windows | G, s). Then

    p(s | G, W, O_W)  ∝  1[s ∈ F_W] · R(s) · λ_W(s),

where R(s) ∈ {0, 1} is the source-eligibility test (`source_eligibility.py`). R(s) is a function of the
16³-block supports of the lit and dark sets over the **whole** field.

`stratified_windows.select` makes λ_W(s) depend on s through the whole field, not only through O_W:
* the informative and source-exclusion tests applied to every Sobol proposal;
* when the pool stops filling;
* the ratio-bin composition of the pool, which drives bin choice and RNG consumption.

Given the per-attempt seeds and `balance_before`, λ_W is deterministic in s and can be replayed exactly
for a single candidate s. Doing that for all of F_W would need the full 1024³ field of every alternative
source.

### 2.3 The label we construct, and why it is a correct training target [P]

We define the **natural source-excluded posterior**

    q_W(s) = Uniform(F_W)   (Lebesgue measure restricted to F_W, normalised).

It is the exact posterior under a generator that proposes windows independently of s and accepts them
by a rule that depends only on O_W and on s ∉ B(W). For example, the v0.3 informative stream, read
without per-scene quota normalisation (`SELECTION_LAW.md`).

* *Rao–Blackwell property.* The Student sees X = φ(O_W, W): pooled lit, dark, solid and boundary
  channels. With Z = (G, W, O_W), every label of the form p(s | Z) satisfies E[p(s | Z) | X] = p(s | X).
  So cross-entropy against p(s | Z) has the same population optimum as true-source NLL, and its gradient
  has no larger variance. The variance saving is large only where p(s | Z) is spread at the Student's
  output resolution.
* *Mismatch to v0.1.* q_W differs from the v0.1 posterior by the reweighting R(s)·λ_W(s), normalised. If
  R·λ varies by a factor of at most (1 + η) over F_W, the total-variation distance between the two is at
  most η/2.
  - R(s) = 1 for all but 3 of 1003 proposed sources, so it is constant on F_W except in rare cases [A].
  - λ_W is approximately 1/P_b(G, s): the probability that a uniform origin gives a valid window in the
    observed lit-ratio bin b. This holds up to pool truncation and the spatial-diversity term.
* *Exact correction for future releases.* Use proposals independent of s, acceptance that depends only on
  O_W and the declared condition, and per-sample weights ∝ P_b(G, s*). P_b is computable from the true
  source's stored field. Under that design q_W is the exact posterior of the weighted training law. In
  §6 this design change is given as the way to make labels exact.

### 2.4 Output resolution and metric

The Student's density is piecewise constant on 128³ fine cells (16³ coarse cells × 8³ fine cells). It is
uniform within a cell on the legal part. Cross-entropy therefore depends on the label only through its
cell masses q_c = vol(F_W ∩ c)/vol(F_W).

For softmax logits the CE gradient is p_θ − q. A label error Δq biases the logit gradient by exactly Δq,
whose L1 norm is 2·TV. So **TV over the 128³ cell masses** is the natural label metric. TV over the 16³
coarse masses is never larger (data processing).

## 3. Geometry lemmas

**Lemma 1 (shadow polyhedron) [P, V].** Let C = {f_i ≤ 0} be convex and r ∉ C, with α_i = f_i(r) ≠ 0 for
all i. Then S_C(r) := {s : [r, s] ∩ C ≠ ∅} is the convex polyhedron

    {f_i(s) ≤ 0, i ∈ F} ∩ {(c_j n_i − c_i n_j)·(s − r) ≤ 0, i ∈ F, j ∈ B},

where F = {α > 0}, B = {α < 0} and c = −α.

*Proof sketch.* Write the segment as r + t(s − r) with t ∈ [0, 1]. Eliminating t by Fourier–Motzkin from
f_i(r) + t·n_i·(s − r) ≤ 0 gives exactly these rows. Rows for pairs i ∈ F, j ∈ B whose faces are not
adjacent are implied by the silhouette-edge rows: the cone from r over C is bounded by the planes
through r and the silhouette edges, and beyond the entry point the ray stays inside every front-face
half-space.

*Verification.* 6.6×10⁶ test points on real scenes: 0 disagreements between the reduced rows, the full
Fourier–Motzkin rows and the canonical exact oracle (`tests/test_shadow.py`).

By symmetry of segments, S_C(r) is also the set of sources s that would make receiver r dark through C.

**Lemma 2 (all-blocked box pairs) [P].** For convex boxes B and R and convex C, every segment [s, x] with
s ∈ B, x ∈ R meets C iff all 64 corner-to-corner segments meet C. Equivalently, B ⊂ S_C(w) for each of
R's 8 corners w.

*Proof.* For a corner v of B, S_C(v) is convex and contains R's corners, so it contains R. For any x ∈ R,
symmetry then puts every corner of B in S_C(x), which is convex, so B ⊂ S_C(x).

**Lemma 3 (all-visible box pairs) [P].** Every segment from B to R misses C iff conv(B ∪ R) ∩ C = ∅. Two
sufficient certificates are used: a face plane of C strictly positive on both boxes, or separation of the
axis-aligned bounding boxes. They are incomplete near grazing; incompleteness only causes further
subdivision.

**Lemma 4 (convex penumbra) [P].** For convex B and C, ∪_{s∈B} S_C(s) is convex. This is noted for
future certificates and is not used by the current code.

## 4. Algorithm (CEPL: certified exact-polytope labeller)

1. **Source-space octree** over Ω. Each box carries the receivers and receiver blocks of W that are not
   yet certified for an ancestor. The root list holds ~640 *sentinel* single receivers (sampled boundary,
   lit and dark voxels) plus the whole-window block.
2. **Box–receiver decisions** use Lemmas 1–3 with filtered float64 predicates: a sign is accepted only
   with a margin of 10⁻¹¹ on unit-normalised planes. The possible outcomes are:
   * *consistent* — drop the receiver;
   * *contradiction* — the box is certified **OUT**;
   * *undecided* — split the block, or keep the single receiver as an *active* constraint.
3. **Exact leaves.** When only single receivers remain, F_W ∩ B is a Boolean combination of convex
   polyhedra:
   * a lit receiver whose shadow is cut by one plane gives a half-space;
   * a dark receiver with one cutting component gives a convex set;
   * anything else is a non-convex constraint (SUB or UNI) with convex pieces.

   Its volume is computed exactly:
   * clip the box by the convex rows;
   * drop every non-convex piece that is certified disjoint from, or certified to contain, the polytope;
   * apply inclusion–exclusion over the few that remain.
4. **Bounds.**
   * IN and EXACT records give lower = upper.
   * Undetermined leaves (depth or term budget) carry lower 0 and upper = volume of their convex
     relaxation.
   * Boxes left by the node budget carry their full volume.

## 5. Certificate [P]

Let L_c and U_c be the summed lower and upper bounds in cell c, L = Σ L_c and U = Σ (U_c − L_c), plus the
clearance-shell term. For the label q̂_c = L_c / L:

    TV(q̂, q) ≤ U / (L + U).

*Proof.* Write m_c = L_c + a_c with 0 ≤ a_c ≤ u_c and A = Σ a_c. Then
Σ_c |m_c/(L+A) − L_c/L| ≤ 2A/(L+A) ≤ 2U/(L+U).

A tighter bound, used for every reported number, is computed by `code/rpl/bound.py`:

    TV ≤ max_{0 ≤ e ≤ U−L} Σ_c e_c (1 − L_c/L) / (L + Σ_c e_c),

which is solved exactly by a greedy order on w_c = 1 − L_c/L. It follows from
|L e_c − E L_c| ≤ e_c (L − L_c) + L_c (E − e_c). It is exactly 0 for a certified one-hot label
(all uncertainty in the single cell holding all certified mass). It is checked against exhaustive vertex
enumeration in `tests/test_bound.py` (3,000 cases, never below the exact supremum).

Certified statements about *departure from one-hot* need no normalisation and hold even for incomplete
runs. The mass outside the true cell c* lies in

    [ L(¬c*) / (U(c*) + L(¬c*)),  1 − L(c*) / (L(c*) + U(¬c*)) ].

This is a deterministic statement about the mathematical algorithm. The implementation adds floating-point
polytope volumes (relative error ~10⁻¹³, `tests/test_polytope.py`) and filtered sign decisions (sound
under the stated margins). It is **not** a machine-checked certificate.

*Important consequence.* The bound is relative to vol(F_W), which is often ~10⁻¹³ (§7). Any unresolved box
with an upper bound above that scale makes the bound vacuous. Certified labels therefore require
*complete* resolution of the source space, not just a small absolute unresolved volume.
