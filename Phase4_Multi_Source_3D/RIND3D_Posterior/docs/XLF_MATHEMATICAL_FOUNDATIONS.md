# XLF — Mathematical foundations (Part A)

> **Status: incomplete.** XLF was frozen as a research track on 2026-11-18, before this document was finished.
> FastLabel v3.4 is the production labeller.
>
> The definitions (§1) and the single-receiver geometry (§2–3) are written with full proofs. The local-first
> theory (§4) and the error theorems (§6) give statements with proofs or proof sketches, as marked.
>
> **Unresolved proof gaps and open questions**
> * **G-A1.** §4.5, quantitative localisation: proven for slabs. For true wedges with unknown edge correspondence
>   there is only a union bound; no sharp volume bound exists.
> * **G-A2.** §4.6, an exact characterisation of which receivers bind (a minimal sufficient receiver set): open.
> * **G-A3.** §6.3: the exact supremum of the TV bound over the enclosure box is a subset maximisation. Its
>   complexity is not settled (knapsack-like), and the implemented greedy bound is only a relaxation.
> * **G-A4.** §5, selection effects: the label is the posterior given the event {R(s)|_W = X}. It equals the
>   population posterior only if window and channel selection are ignorable. This is the open audit item M1,
>   a modelling decision for the user.
> * **G-A5.** §8, curved occluders: only the convexity-based lemmas transfer. No exact predicate design exists.
> * **G-A6.** The literature connections (§7) are pointers only. No systematic survey was made.

Labels used throughout: **proven** (complete proof here), **proof sketch** (the argument is given and gaps are named),
**conditional** (proven under a stated hypothesis), **conjectured**, **empirical** (measured, not proven).
No statement here depends on Python, Numba or binary64.

## 1. The inverse problem (A1)

* **World and geometry.**
  - Ω = [0,1]³.
  - Geometry G: components K_1, …, K_m.
    - Each component is K_c = {x : f_{c,i}(x) ≤ 0, i = 1..n_c}, with affine f of rational coefficients (the canonical
      rows of `decode_convex`).
    - Each K_c is bounded, convex, regular closed (nonempty interior) and contained in (0,1)³ (SEMANTICS.md).
  - The solid is S = ∪_c K_c, which is closed.
* **Receivers.**
  - N is a power of two.
  - The window is W = {r_t = (o + t + ½)/N : t ∈ {0..L−1}³}.
  - The window box is the closed box B_W = [o/N, (o+L)/N]³.
  - The free receivers are W_f = W \ S.
* **Response.**
  - R_G(s)(r) = 1 (lit) iff the closed segment [s, r] misses S; R_G(s)(r) = 0 (dark) otherwise. Contact blocks.
  - The observation is X : W → {0, 1, SOLID}, with X(r) = SOLID iff r ∈ S. Otherwise X(r) = R_G(s*)(r) for the
    generated source s*.
* **Domain.** D = D(G, W) = {s ∈ (0,1)³ : dist₂(s, S) > ρ} \ B_W, with ρ = 2⁻²⁶.
  - The distance is Euclidean, as in the generator's `clearance`: strict separating planes, with an exact fallback.
* **Feasible set.** F = {s ∈ D : R_G(s)(r) = X(r) for all r ∈ W_f}.
* **Target.** The prior is uniform on D (the declared sampling law). The posterior is π_X(A) = Vol(A ∩ F)/Vol(F)
  when Vol(F) > 0.
  - On the 128³ grid of closed cells c, the target is q_c = Vol(F ∩ c)/Vol(F).
  - Cells overlap only in null sets, so Σ_c q_c = 1.

**Proposition 1.1 (boundaries are null; proven).** ∂F has Lebesgue measure zero. Hence open/closed conventions at
contact, at the window box and at the clearance level do not change π_X.

*Proof.*
1. By Cor. 2.5, F is D intersected with a finite Boolean combination of closed convex polyhedra.
2. ∂D lies in ∂Ω ∪ ∂B_W ∪ ⋃_c ∂(K_c ⊕ B_ρ). Each of these is the boundary of a convex body, hence null.
3. Finite unions of null sets are null. ∎

**Proposition 1.2 (zero-volume posteriors are a null event; proven).** The set of s* ∈ D for which Vol(F(s*)) = 0 is
null.

*Proof.* "Same observation" partitions D into at most 2^{|W_f|} classes. The union of the zero-volume classes is a
finite union of null sets. ∎

Consequence: when Vol(F) = 0 the posterior is undefined, and a labeller must abstain. It must never output a
distribution. XLF abstains unless it proves Vol(F) ≥ Σ lo_c > 0.

## 2. One receiver: the exact inverse shadow

Notation:
* C_c(p) = {x : [p, x] ∩ K_c ≠ ∅}. The relation is symmetric: x ∈ C_c(p) ⇔ p ∈ C_c(x).
* For an apex p ∉ K_c, write a_i = f_i(p), b_i = f_i(x), Φ⁺ = {i : a_i > 0} (nonempty), Φ⁰ = {a_i = 0},
  Φ⁻ = {a_i < 0}.
* B_ij(p, x) = a_i b_j − a_j b_i.

**Theorem 2.1 (segment lemma, all pairs, zero-inclusive gates; proven).** For p ∉ K_c, [p, x] meets K_c iff both:
* (gates) b_i ≤ 0 for every i ∈ Φ⁺ ∪ Φ⁰;
* (pairs) B_ij(p, x) ≤ 0 for every i ∈ Φ⁺, j ∈ Φ⁻.

No aspect condition is needed.

*Proof.*
1. We need t ∈ [0,1] with (1−t)a_i + t b_i ≤ 0 for all i. Since p ∉ K_c, t > 0.
2. Rows in Φ⁰ give b_i ≤ 0.
3. Rows in Φ⁺ give b_i ≤ 0 together with t ≥ τ_i = a_i/(a_i − b_i) ∈ (0,1].
4. Rows in Φ⁻ with b_j > 0 give t ≤ σ_j = −a_j/(b_j − a_j).
5. Rows in Φ⁻ with b_j ≤ 0 give nothing.
6. The interval [max τ, min σ] is nonempty iff every pair satisfies τ_i ≤ σ_j. Cross-multiplying positive denominators
   gives B_ij ≤ 0.
7. When b_j ≤ 0, B_ij ≤ 0 holds automatically, given the gate on i. ∎

**Theorem 2.2 (convexity; proven).** C_c(p) is a closed convex polyhedron.
* For p ∉ K_c, C_c(p) = cone_p(K_c) ∩ ⋂_{i ∈ Φ⁺∪Φ⁰}{f_i ≤ 0}.
* For p ∈ K_c, C_c(p) = ℝ³.

Proof: convexity follows from Thm 2.1 (an intersection of halfspaces, once the apex is fixed).
Directly: if x_k = p + t_k(k_k − p) with t_k ≥ 1, then a convex combination equals p + T(k − p), where T ≥ 1 and
k ∈ K_c. ∎

**Theorem 2.3 (silhouette reduction; conditional, proven under its hypotheses).**
* Let E be the set of row pairs that are both tight on a common edge of K_c, i.e. share ≥ 2 distinct vertices. This
  is the `certify.scene` ADJ rule.
* Suppose either K_c is full-dimensional, or every a_i ≠ 0 (strict aspect).
* Then Thm 2.1 remains exact with the pairs restricted to (i, j) ∈ E. Any superset of E is also exact.

*Proof.*
1. The cone cone_p(K) is closed, pointed and polyhedral. It is 3-dimensional: K is full-dimensional, or, under strict
   aspect, p ∉ aff K. The latter holds because the implicit-equality rows of a flat K vanish on aff K.
2. Hence cone_p(K) is the intersection of the halfspaces of its facets G.
3. Each facet is p + cone(F − p) for the face F = K ∩ H of its supporting plane H, and dim F ≥ 1.
4. If dim F = 2: H ∝ f_i for the facet row, with a_i = 0, so this is a gate.
5. If F is an edge e: by LP duality the normal cone at relint e is generated by the rows tight on e, so
   H = Σ_{k∈act(e)} λ_k f_k with λ ≥ 0 and Σ λ_k a_k = 0.
6. With M = Σ_{A⁺} λ_i a_i, the rows in A⁺ ∪ A⁻ satisfy Σ_{A⁺∪A⁻} λ_k f_k = M⁻¹ Σ_{i∈A⁺, j∈A⁻} λ_i λ_j B_ij, and the
   rows in A⁰ are gates.
7. All the pairs used are tight on e, so they lie in E.
8. Every pair (i ∈ Φ⁺, j ∈ Φ⁻) is a valid necessary condition: B_ij ≤ 0 on K and B_ij vanishes at p, so it holds on
   cone_p(K). This gives the superset claim. ∎

**Counterexample 2.4 (the hypothesis is needed; exact, `test_xlf_math_lemmas.py`).**
* Take K = [0.4, 0.6]² × {½}, a flat square, with apex p = (0.2, 0.5, 0.5) in its plane, and x = (0.8, 0.9, 0.5).
* The two side rows that bound the in-plane wedge share only one vertex, so they are not in E.
* Gates plus E-pairs claim a hit, but [p, x] misses K.
* The implementation is safe because it demands strict signs at every apex (IMPL test). Regular-closed scenes never
  contain flat components.

**Corollary 2.5 (exact feasible set; proven).**
F = D ∩ ⋂_{X(r)=0} ⋃_c C_c(r) ∩ ⋂_{X(r)=1} ⋂_c (ℝ³ \ C_c(r)).

**Proposition 2.6 (event surfaces; proven for full-dimensional K).** ∂C_c(r) \ K_c is contained in the silhouette
planes P(r, e) = aff(r ∪ e). So, for sources outside S, response changes happen only on planes through a receiver and
an occluder edge: the EV events of aspect-graph theory, with the receiver as the vertex.

*Proof.*
1. Suppose x ∈ ∂C(r) \ K lies in int cone_r(K).
2. The ray from r through x then crosses int K, with t_in < t_out.
3. x ∉ K forces t_out < 1.
4. By continuity, a whole neighbourhood of x satisfies t_in ≤ t_out < 1, so x would be interior to C(r).
   Contradiction. ∎

## 3. Region-level predicates

Let Q = conv V be a source polytope.

* **Lemma 3.1 (guaranteed shadow D; proven).** r is dark via c for every s ∈ Q iff V ⊆ C_c(r), iff
  r ∈ I_c(Q) := ⋂_{v∈V} C_c(v).
  - The proof is convexity of C_c(r) plus symmetry.
  - The statement is exact without any aspect condition (Astra's L2).
  - The form "∀s ∃c" is weaker than "∃c ∀s". So a single-component test is sound but incomplete: two blockers can each
    cover part of Q. This is the common-blocker cost.
* **Lemma 3.2 (convex localisation S; proven).** Suppose exactly the constraints A of C_c(r) straddle Q, and every
  other constraint is ≤ 0 on V. Then C_c(r) ∩ Q = Q ∩ ⋂_{g∈A}{g ≤ 0}.
* **Lemma 3.3 (pencil collapse; proven).**
  - B_ij(v, r) = cross(w_v, w_r) with w = (f_i, f_j), which is linear in w_v.
  - If w_v ∈ cone(w_a, w_b), the constraint at v is implied by the constraints at a and b. So only the angular
    extremes bind.
  - This is the Tau-pencil collapse applied to the source vertices instead of the receivers. It is one lemma, not two
    methods.
* **Lemma 3.4 (separation L; proven).** r is lit via c for every s ∈ Q iff conv(Q ∪ {r}) ∩ K_c = ∅.
  - Equivalently, some plane g has g ≤ 0 on K_c and g > 0 on conv(Q ∪ {r}). The proof is strict separation of compact
    convex sets.
  - Sound candidate planes:
    - facets f_i;
    - pencil planes λf_i + μf_j with λ, μ ≥ 0;
    - constraints of C_c(r). For a pair constraint, g(r) = 0, but r ∉ K_c, so the segment misses K_c.
  - "Two planes per edge" is linearity in (λ, μ). A linear function that is positive somewhere on a 2-D cone is
    positive on one of its extreme rays. The sinusoid argument is the same statement in angular coordinates.
* **Theorem 3.5 (possible shadow; proven).** For Q ∩ K = ∅, ⋃_{s∈Q} C_K(s) = K + cone(K − Q), which is convex.
  - Proof: k₀ + μ(k₁ − s) = k + μ(k − s) with k = (k₀ + μk₁)/(1 + μ) ∈ K.
  - Its facets are separating common tangent planes of K and Q of three kinds:
    - K-facet;
    - (K-edge, Q-vertex), the EV kind;
    - (K-vertex, Q-edge), the VE kind.
* **Remark 3.6 (incompleteness of the implemented lit families; proven by exact example).**
  - The receiver-form family {facets, edge-pencil extremes} omits the VE planes. The source-form family (constraints
    of C_c(r)) can also fail.
  - Example: K = [0,1]³, Q = [(1.3, .9, .9), (.9, 1.3, .9)], r = (.9, .9, 1.3).
    - The triangle misses K, because x + y + z = 3.1 on it.
    - No facet, edge pencil or C_K(r) constraint separates it.
  - Consequence: soundness is unaffected, but receivers near occluder vertices stay ambiguous and force splits.
  - Both families are complete for points, by Thm 2.1. This gives the asymptotic completeness used in Part B.

## 4. The local-first principle (A2)

Interface pair: adjacent free receivers l, k with X(l) = 1 and X(k) = 0.

* **Theorem 4.1 (exact pair constraint; proven).** F ⊆ (⋃_c C_c(k)) \ (⋃_c C_c(l)). No assumption is needed.
* **Theorem 4.2 (Proposition A′; proven in LOCALFIRST_E123.md §2, re-checked).**
  - Hypothesis H1: [l, k] misses S.
  - Conclusion: every s ∈ F has an edge e with e ∩ conv{s, l, k} ≠ ∅, and e is a silhouette from some p ∈ [l, k].
* **Counterexample 4.3 (A′ is strictly weaker than Thm 4.1; exact).**
  - Take s = 0, l = (10, 1, 0), k = (10, −1, 0), K = [4, 5] × [−0.1, 0.1]².
  - A′ admits s, through the edge {x = 4, y = −0.1} at barycentric coordinates (3/20, 1/4, 3/5).
  - But [s, k] misses K, so s is infeasible.
* **Theorem 4.4 (the plane statement, made exact; proven).**
  - If a continuous shadow-boundary piece Γ ⊆ ∂C(s) \ K is generated by edge e, then s ∈ aff(e ∪ Γ). This follows
    from Prop 2.6.
  - With a finite binary observation only the sliver remains: s lies between P(l, e) and P(k, e), with pencil angle
    θ_e(s) ∈ [θ_e(k), θ_e(l)]. The source is in general on **neither** plane.
  - The edge e is not determined by X. Over all admissible edges and components, the constraint is a union of slivers.
* **Theorem 4.5 (localisation by independent wedges; proven for slabs, G-A1 for wedges).**
  - Within distance D of line ℓ_e, a wedge of angle Δ lies in a slab of half-width D·sin(Δ/2) about its bisector.
  - Three slabs with unit normals forming the matrix 𝐍 confine s to a parallelepiped of diameter
    ≤ 2‖𝐍⁻¹‖₂‖(D_i sin(Δ_i/2))‖₂.
  - One wedge gives a sheet, two give a tube, three give a cell of size O(h·D/d).
* **Remark 4.6 (independence and the EEE degeneracy; proven).** The normals of P(s, e₁), P(s, e₂), P(s, e₃) are
  linearly dependent iff s lies on a common transversal of the three edge lines. That set is the EEE event quadric
  (Gigus–Canny–Seidel).
  - Coaxial or collinear edges share a pencil and are redundant (Lemma 3.3).
  - Localisation degrades to a tube near the quadric.
* **Theorem 4.7 (interfaces alone do not determine F; proven by example).** An all-lit window has no interface, yet it
  excludes sources behind an occluder. Interior receivers bind, so local-first regions are outer bounds only, and the
  exact posterior needs every receiver (G-A2: which receivers suffice is open).
* **Theorem 4.8 (multiple explanations; proven by example).**
  - With two occluders, the feasible set of one interface pair is a non-convex union of two wedges: s₁ and s₂ are
    feasible, their midpoint is not.
  - Any method that commits to one explanation (H2) can lose feasible mass. XLF keeps every explanation by leaving the
    receiver open.
* **Empirical (E1, LOCALFIRST_E123.md).** Interface regions held 82–97% of the log-volume information on 54 tasks.
  This is not a theorem.

## 5. Global geometry and posterior preservation (A3)

* **Theorem 5.1 (proven).** F is a function of (G, W, X) only.
  - An algorithm computes Vol(F ∩ c), or certified bounds on it, from G and X with no access to s* and no response
    information beyond X. It therefore targets π_X(·) ∝ 1_D · 1[R_G(·)|_W = X], the posterior of the declared model.
  - Using global geometry does not change the conditioning, because the posterior is conditional on G by definition.
    The uses are:
    - irrelevance pruning by AABBs, an exact set identity;
    - unions over all components and edges, which keep every explanation;
    - exact complements.
* The Student sees less than G. The Bayes-optimal Student prediction under a proper score is the mixture E[q_G | view].
  This is the standard proper-scoring result.
* G-A4: the result is the population posterior only if selection of (W, channel) depends on s* through X alone.

## 6. Error theorems (A4)

* **Theorem 6.1 (proven).** If I ⊆ F ⊆ U (up to null sets) and Vol(I) > 0, then
  TV(q_I, q_F) = 1 − |I|/|F| ≤ 1 − |I|/|U|. The cell projection does not increase TV.
* **Theorem 6.2 (cell enclosures; proven; this is `certify.tvb`).**
  - Assume lo_c ≤ v_c ≤ hi_c + p_c with Σ p_c ≤ P, and Lt = Σ lo_c > 0.
  - Then TV(lo/Lt, q) ≤ sup_{0≤e≤x} Σ_c e_c w_c/(Lt + E), with w_c = 1 − lo_c/Lt, x_c = hi_c − lo_c, and the pool
    entering with w = 1.
  - The supremum is attained greedily. The bound is ≤ 1 − Lt/Ht, so it is never worse than Thm 6.1.
  - It is 0 for single-cell posteriors, whatever the volume uncertainty.
* **Theorem 6.3 (exact supremum; proven).** sup TV over the enclosure box = max_A x_A (1 − lo_A/Lt)/(Lt + x_A), with
  pool mass added to x_A.
  - The tvb greedy bound dominates it. It is strictly larger when the extra mass spreads over several cells with lo > 0.
  - Tested exactly in `test_xlf_numerics.py`.
  - G-A3: the complexity of computing the exact supremum is open.
* **Theorem 6.4 (stored label; proven).** TV(q̂, q) ≤ TV(q̂, lo/Lt) + bound, with the first term computed exactly.
  - q̂ is not exactly normalised.
  - Renormalising adds at most TV(q̂, lo/Lt).
* **Error sources, kept separate:**
  - exact reconstruction: never claimed by XLF;
  - unresolved geometry: UNDET volume, slabs, the clearance shell;
  - enclosure width: numerical;
  - label rounding.

## 7. Literature (pointers; G-A6)

* Visibility and shadows:
  - shadow volumes (Crow 1977);
  - aspect graphs and EV/EEE events (Gigus, Canny, Seidel 1991; Plantinga, Dyer 1990);
  - visibility skeleton (Durand, Drettakis, Puech 1997);
  - discontinuity meshing (Heckbert 1992);
  - antipenumbra (Teller 1992).
* Exact computation:
  - plane-based geometry (Sugihara, Iri 1989; Campen, Kobbelt 2010);
  - SoS (Edelsbrunner, Mücke 1990);
  - adaptive predicates (Shewchuk 1997);
  - floating-point error bounds (Higham 2002).
* Certification:
  - certifying algorithms (McConnell, Mehlhorn, Näher, Schweitzer 2011);
  - shadow carving (Savarese et al. 2007);
  - bas-relief ambiguity (Kriegman, Belhumeur 2001).

## 8. Curved occluders (G-A5)

* What transfers: Lemma 3.1 (convexity of C_K(x) for any convex K), Lemma 3.4 (separation) and Thm 3.5.
* What does not transfer: Thm 2.1/2.3 and the plane structure.
  - Boundary rays are tangent, so s lies in the tangent plane at an unknown tangency point.
  - Finite observations give curved slivers between the tangent cones from l and k.
* See `G2-02-v3-curved/.../CURVED_ASSESSMENT.md`.
