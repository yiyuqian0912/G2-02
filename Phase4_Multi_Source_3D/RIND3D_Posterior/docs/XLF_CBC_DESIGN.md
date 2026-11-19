# XLF correct-by-construction labeller: design (kernel v2)

Branch `claude/xlf-correct-by-construction`. Builds on Astra's theorem (`XLF_PROOF_FOUNDATIONS.md`), Astra's
transition design (`XLF_CORRECT_BY_CONSTRUCTION.md`) and Astra's exact geometry (`experiments/global_inverse/`).
Code: `experiments/credit_solver/cbc_kernel.py` and `screen2.py` (trusted), `cbc_search.py` (untrusted).

Assurance labels (used in all XLF_CBC documents):

| label | meaning |
|---|---|
| **MATHEMATICALLY PROVEN** | a complete argument is written here or in Astra's documents; not machine-checked |
| **IMPLEMENTATION-VERIFIED** | the property of the code itself is mechanically proved (z3) over the stated ranges |
| **EXACT-DIFFERENTIAL-TESTED** | the code agrees with an independent exact (rational or integer) reference on the stated test population; not a proof |
| **RUNTIME-CERTIFIED** | an exact check runs inside every labelling run and aborts (no certificate) on failure |
| **EMPIRICALLY TESTED** | agreement with references or audits on real tasks; not a proof over continuous regions |
| **CONDITIONAL** | holds if a stated, unverified assumption holds |
| **OPEN** | not established |

## 1. Target and certificate

One problem consists of:

* the canonical convex obstacles K₁…K_m (integer rows, f ≤ 0);
* a v3 window (origin o, side L ∈ {64, 128, 256}, N = 1024);
* one complete FREE-channel observation, with classes 0/1/2 per receiver centre (o+i+½)/N.

The set D of legal sources consists of points in the open unit cube, outside the closed window box, at distance
greater than 2⁻²⁶ from every obstacle. F is the set of points of D that reproduce every FREE bit (closed-segment
rule). The target is the 128³ push-forward of uniform volume on F.

**Certificate (PROVEN, Astra; frontier form).** Suppose the kernel holds three things:

* IN pieces Q_k ⊆ F with disjoint interiors;
* integer lower credits lower[c] ≤ vol(⋃Q_k ∩ c)·2¹⁶⁰ and upper credits upper_in[c] ≥ the same;
* live pieces P_h with F ⊆ ⋃Q_k ∪ ⋃P_h (a.e.).

Then, for any decoded normalized payload p,

```
TV(p, q) ≤ 1 − Σ_c min(p_c, a_c),   a_c = l_c / (l_c + U_{−c}),
U_{−c} = Σ_{c'≠c} upper_in[c'] + Σ_{h: P_h ⊄ c} ⌈vol P_h⌉.
```

`kernel.wire_bound` evaluates this with integer floors, which round the bound outward. A live piece that the
search never visits is charged in full. Dropping a piece therefore requires a proof that it contains no feasible
volume.

## 2. State and invariants

The State is immutable: live pieces, `lower`, `upper_in`, the next handle, and the window-excluded flag. Each piece
holds its closed convex polytope, its residual obligation set R_P (or "all FREE receivers except `done`") and a
`deferred` flag.

| invariant | statement |
|---|---|
| I1 cover | F ⊆ ⋃ IN ∪ ⋃ live, up to Lebesgue-null sets |
| I2 soundness of IN | every IN piece ⊆ F (a.e.) |
| I3 disjointness | live and IN pieces have pairwise disjoint interiors |
| I4 obligations | for live P, every FREE receiver outside R_P is reproduced by every source of P |
| I5 exclusion | after `exclude_window`, every live piece misses the open window box and the obstacle interiors |
| I6 credits | `lower[c]` and `upper_in[c]` are the floor and ceiling sums of exact cell volumes of the IN pieces |

All transitions are atomic. Results are computed into scratch values and published by one `_commit`. A `Limit`
or arithmetic exception before the commit leaves the state unchanged. A stale handle raises before any change.

## 3. Transitions and preservation (PROVEN)

| transition | effect | why I1–I6 are preserved |
|---|---|---|
| `exclude_window` | every piece minus the closed window box, minus each closed obstacle (exact polytope differences) | F misses both sets (I1); children are subsets (I3, I4) |
| `condition_pair(d, a)` | each piece P ↦ P ∩ ⋃_c C_c(d) ∖ ⋃_c C_c(a), with first-cover ownership | exact inverse cones (Astra's elimination theorem); feasible sources satisfy both receivers (I1); children are disjoint subsets |
| `split(h, row)` | P ↦ P ∩ {row ≤ 0}, P ∩ {row ≥ 0}; any integer row | the halves cover P and overlap only on a null plane (I1, I3); obligations are inherited by inclusion (I4) |
| `screen(h)` | classify the receivers of R_P (§4): contradiction → drop P; nothing left → settle; else R_P := ambiguous subset | the screen lemma (§4): a contradiction means P ∩ F is null (I1); removed receivers are proved on all of P (I4) |
| settle (inside `screen`) | P ↦ legal parts (credited IN, exact per-cell volumes) + clearance shell (live, deferred); the forbidden part is dropped | with R_P = ∅, I4 and I5, every legal source of P is in F (I2); the clearance partition is exact (outward offset cover with a directed square root, inward 2⁻²⁶ translates); dropped points are within 2⁻²⁶ of an obstacle (I1) |
| `enforce(h, R)` | exact cone cuts for the receivers R (first-cover for inactive, subtraction for active); children have R_P ∖ R | Astra's exact cones (I1, I4); children are disjoint subsets (I3) |

The search chooses handles, receivers and planes. It can make the solver slow or abstain. It cannot change the
classification of a receiver, a volume or a credit. Every kernel method validates its arguments:

* receivers must be FREE and inside the window;
* split planes must be integer rows that are not constant;
* the obligation list is checked for sortedness and range inside the screen.

## 4. The screen lemma

For a live piece P and an obstacle K (rows fᵢ ≤ 0), define two receiver regions.

* **Guaranteed shadow** `inner_K(P)` = ⋂_{v ∈ vert P} S_K(v). Here S_K(v) is the set of receivers r for which
  [v, r] meets K; it is the cone formula with the roles of source and receiver exchanged.
  * For each facet pair, only the vertex with the tightest ratio is kept. Given fᵢ(r) ≤ 0, that row implies the
    rows of the other vertices.
  * **Claim.** If r ∈ inner_K(P), every source of P is blocked by K. The vertices lie in C_K(r), and C_K(r) is
    convex in s.
  * At a vertex on or inside K, the vertex rows over-constrain, so the region becomes smaller. This is sound but
    weak.
* **Possible shadow** `outer_K(P)` ⊇ ⋃_{s∈P∖K} S_K(s), built from three kinds of row:
  1. facet rows fᵢ(r) ≤ 0 where P ⊆ {fᵢ ≥ 0};
  2. pair rows with the minimal ratio where fᵢ > 0 > fⱼ on all of P (by the mediant inequality the ratio is
     minimal at a vertex);
  3. **separating support planes** h with K ⊆ {h ≤ 0} and P ⊆ {h ≥ 0}. These are valid because
     O = K + cone(K − P) ⊆ {h ≤ 0}. They are proposed by a floating-point hull (`penumbra.directions`) and kept
     only after an exact integer check on every vertex of K and of P.
  When P misses K and every facet of O is proposed, `outer_K(P)` = O exactly.
* **Directed fixed point.** Every row is normalized to 2⁴⁵ and rounded: guaranteed rows up, possible rows down.
  Receiver homogeneous coordinates (2(o+i)+1, …, 2N) are positive, so rounding up shrinks the guaranteed region and
  rounding down grows the possible region. Range: |coef| ≤ 2⁴⁵ and coordinates ≤ 2¹¹, so |value| < 2⁵⁸. The
  kernel checks this range on every screen.
* **Decision.**

  | receiver | condition | conclusion |
  |---|---|---|
  | active | in some guaranteed shadow | contradiction |
  | active | in no possible shadow | proved |
  | inactive | in no possible shadow | contradiction |
  | inactive | in some guaranteed shadow | proved |
  | otherwise | | ambiguous |

  Contradictions and proofs hold for every source of P ∖ ⋃K, which is all of P up to a null set by I5.
* **Monotonicity.** For P′ ⊆ P, inner(P′) ⊇ inner(P) and outer(P′) ⊆ outer(P) as exact sets. A receiver proved on
  P is therefore proved on P′, which justifies inheritance (I4).
* **Scanlines.** Within one scanline, each group region is an interval in x, computed by exact integer floor and
  ceiling division. `scan_collect` covers the full window; `scan_ids` covers a sorted residual list with identical
  arithmetic. Their agreement is a unit test.

## 4b. Candidate-body masks and restricted screens (PROVEN)

For each ambiguous receiver r of a screened piece P, the screen returns
m(r) = {K : r ∈ rounded possible shadow of K over P}.

* Rounding down only enlarges a possible shadow. Hence K ∉ m(r) implies that r lies outside the true union of
  shadows of K over P, so no source of P, or of any P′ ⊆ P, has its segment to r blocked by K.
* **Consequences, all exact:**
  1. `enforce` restricts the exact inverse cones of r to the bodies in m(r). For an inactive r it takes the
     first-cover union over m(r). For an active r it subtracts only those cones. The excluded cones miss P′ almost
     everywhere.
  2. A residual screen of P′ builds envelopes only for ⋃_{r ∈ R_P′} m(r). Decisions use these bodies plus the
     established fact that excluded bodies neither block nor guarantee a block. Such a screen is never weaker than
     an all-body screen; this is unit-tested.
* Children inherit m together with R (I4).

## 4c. Exact cell credits (PROVEN)

An IN piece Q is integrated by an octree over its grid range, each node carrying the exact clip of Q to its block:

* a block whose 8 corners satisfy every facet row of Q lies inside the convex Q and is credited at the exact cell
  volume 2⁻²¹ (2¹³⁹ units);
* a single boundary cell is credited with ⌊vol·2¹⁶⁰⌋ for the lower credit and ⌈vol·2¹⁶⁰⌉ for the upper credit;
* the facet rows are validated: every vertex satisfies them, and each vanishes on its face.

Conservation is exact: vol(Q)·2¹⁶⁰ − Σ lower ∈ [0, #boundary cells).

## 4d. The exact possible shadow (MATHEMATICALLY PROVEN; v2.9b)

**Lemma O.** Let K, P be compact convex sets with disjoint interiors and U = ⋃_{s∈P} S_K(s) the
set of receivers blocked by K for some source of P. Then U is convex and its closure is the
Minkowski sum O = K + cone(K − P).

*Proof.* r ∈ U iff r = M k − (M − 1) s for some k ∈ K, s ∈ P, M ≥ 1 (the segment [s, r] meets
K at k = s + (r − s)/M). For two such points with (k₁, s₁, M₁), (k₂, s₂, M₂) and t ∈ [0, 1],
the combination t r₁ + (1 − t) r₂ equals M k − (M − 1) s with M = tM₁ + (1 − t)M₂,
k = (tM₁k₁ + (1 − t)M₂k₂)/M ∈ K and s = (t(M₁−1)s₁ + (1 − t)(M₂−1)s₂)/(M − 1) ∈ P (if M = 1
the combination lies in K ⊆ U). So U is convex.
U ⊆ K + cone(K − P) because r = k + (M − 1)(k − s). Conversely k₁ + λ(k₂ − s) is the limit as
t → 0 of (1 − t)k₁ + t(k₂ + (λ/t)(k₂ − s)), a convex combination of points of U. ∎

**Facets of O.** The facet normals of a Minkowski sum of a polytope and a polyhedral cone come
from three kinds of face pairs: a facet of K; a facet of cone(K − P); or an edge of K together
with an extreme ray k − p. Every non-K facet therefore lies on a plane that separates K from P
and contains (an edge of K and a vertex of P) or (a vertex of K and an edge of P); a facet of
cone(K − P) also contains such a pair (its plane touches K and P along faces of dimension ≥ 0
whose sum spans it). By the separation lemma (§4) every such plane is valid.

**What was missing.** v2.8 used the facets of K that P lies outside, the all-vertex pair
rows, and the facets of cone(K − P) proposed by `penumbra.directions`. The mixed facets
(edge of K × vertex of P, edge of P × vertex of K) that are not cone facets were never proposed.
Whenever P straddles a facet plane of K the possible shadow was strictly larger than O. On
stuck pieces, 10⁵ lit receivers were ambiguous while exact cuts showed that they did not cut
the piece (531 of 537 batch cuts changed nothing).

**Implementation.** `shadow_facets.mixed_candidates` (untrusted) enumerates the candidate
planes with a binary64 preselection and builds each row exactly from three homogeneous
vertices; `cbc_kernel.separators` (trusted) keeps a row only if
max_{v ∈ vert(K ∩ cube)} h(v) ≤ 0 ≤ min_{v ∈ vert P} h(v) in exact integers. With every
candidate proposed, the possible shadow equals O up to the directed rounding of §4.
Effect: ambiguous receivers on stuck pieces 10⁵ → 10¹–10² (§XLF_CBC_RESULTS).

## 4e. Receiver convexity (MATHEMATICALLY PROVEN; used by the search, v2.9c)

**Lemma H.** For a source s outside a convex body K, the shadow S_K(s) = {s + λ(k − s) : k ∈ K,
λ ≥ 1} is convex (same argument as for C_K(r)). Hence for any receiver set R,
⋂_{r ∈ R} C_K(r) = ⋂_{r ∈ ext conv R} C_K(r): a source reproduces "R is dark because of K" iff
it reproduces the extreme points of conv R.

**Use.** For a piece P, let R_K be the dark ambiguous receivers whose only possible blocker is
K (candidate mask = {K}). The search enforces only the convex-hull vertices of R_K (per
scanline, only the minimum and maximum x can be extreme), each an exact intersection with
C_K(r) that never fragments the piece. The next screen then proves all of R_K, because every
vertex of the cut piece lies in every C_K(r), r ∈ R_K, so R_K lies in the exact guaranteed
shadow (up to the directed rounding). At L = 256 this replaces 1.6–9.9 × 10⁶ ambiguous
receivers by 50–80 exact cuts. The lemma is used only for efficiency: the kernel enforces
each proposed receiver exactly and re-proves everything in later screens, so a wrong hull
can only cost time.

## 4f. Contact vertices in the guaranteed shadow (MATHEMATICALLY PROVEN; v2.9c)

For a vertex v of P inside the closed body K, every segment [v, r] meets K, so S_K(v) is all
of space. The guaranteed shadow is therefore computed from the vertices outside K only:
r ∈ ⋂_{v ∉ K} S_K(v) puts every vertex (those in K trivially) into C_K(r), and P = conv(vert P)
⊆ C_K(r). If all vertices lie in K, then P ⊆ K is illegal a.e. and the guaranteed shadow is
the whole window. (Envelope evaluated the rows of contact vertices, which over-constrained:
sound, but it left dark receivers behind obstacle contacts ambiguous.)

## 4g. Polytope arithmetic certificates (MATHEMATICALLY PROVEN; RUNTIME-CERTIFIED)

**FS (face-structure invariant).** The face cycles of a polytope Q tile the boundary of
conv(vert Q) exactly once, each as a strictly convex polygon. `polycheck.check` verifies:
(a) primitive vertices with w > 0;
(b) every face spans a supporting plane, with all vertices ≤ 0 and the face's vertices = 0;
(c) strictly convex cycles that wind once;
(d) every directed edge used once and its reverse once (after orienting faces outward);
(e) distinct face planes;
(f) faces connected, with a single face cycle around each vertex;
(g) Euler characteristic 2.

**Lemma FS.** (a)–(f) imply that the union S of the faces is homeomorphic to the boundary of
H = ⋂ face rows and that H = conv(vert Q).

*Proof sketch.* Each face lies on a supporting plane of H, hence in ∂H, so the inclusion
φ: S → ∂H is defined. φ is a local homeomorphism at face interiors and at edge interiors,
because the two faces of an edge lie on opposite sides of it on ∂H by (d). Near a vertex the
faces form one cycle (f) of consecutive angular sectors. If they wound around the vertex more
than once, two faces on distinct planes (e) would overlap in an open subset of ∂H, where the
supporting plane is unique: a contradiction. So φ is a covering map of the sphere ∂H by the
connected surface S, hence a homeomorphism. Then ∂H ⊆ S ⊆ conv(vert Q) ⊆ H, so
H = conv(vert Q). ∎

Consequences: the fan volume is vol(conv vert Q); the face rows cut out exactly Q (octree
inside tests); every face edge is an edge of Q (split's edge walk).

**Lemma V (split).** For a polytope P and a plane meeting its interior, the vertices of
P ∩ {row ≤ 0} are exactly the vertices of P with row ≤ 0 together with the points where an
edge of P with strictly opposite end signs crosses the plane. `Poly.split` computes exactly
these: exact signs, exact primitive crossing points, Sutherland–Hodgman clipping of each
strictly convex face cycle, and the cap as the strict convex hull of the plane points in an
injective projection. Cap points never lie inside a cap edge, so no T-junctions arise. FS is
preserved (distinct planes, because a face plane would put P on one side of it).

**Per-split certificate (`polycheck.verify_split`, `CBC_VERIFY_SPLITS=1`).** For every split
inside every transition, the vertex sets of both halves must equal the two sets of Lemma V,
computed independently from the parent's (FS-certified) edges, and both halves must pass
`check`. Inductively from the cube, every polytope the kernel ever creates is then FS-valid,
and every split is exact: the halves cover the parent, overlap only on the plane, and carry
exact volumes. This certifies the cover invariant I1 for the polytope arithmetic in every run
without a physical replay. It is on by default. Cost in pure Python: +23% wall time over the canary, and up to
about 2× on cut-dominated runs (XLF_CBC_RESULTS §5).

Independently of `CBC_VERIFY_SPLITS`, every polytope that receives an IN credit, and every
live polytope charged in a snapshot, passes `check` (RUNTIME-CERTIFIED since v2.9a).

## 4h. Release discipline (by construction; v2.9a)

* The kernel records every (payload, bound) it computes in `snapshot`.
* `Kernel.release(tolerance)` returns CERTIFIED only for the best recorded pair, and only if
  its bound is ≤ tolerance. It returns INCOMPLETE otherwise.
* The trusted driver `cbc_kernel.certify` owns the budget and the kernel. It runs the
  untrusted search behind an exception barrier: a time limit, a fragment cap or any search
  exception is an abstention. It writes the status, payload and bound from the release only.
* Any ArithmeticError inside a transition (a failed structural, conservation or range guard,
  or an overflow) taints the kernel. No further transitions run, and the release is ERROR.
* The search gets read-only views of the kernel's obligation arrays and copies of polytopes.
  Proposed split rows must be integer, non-constant and bounded by 2⁴⁰⁹⁶ (this bounds every
  vertex size, because each vertex is the intersection of three facet planes).

Soundness is therefore independent of the search policy. This is also tested: the chaos
search drives the kernel with random legal operations and misuse attempts, and checks every
snapshot against an exact reference (XLF_CBC_RESULTS §3).

## 5. Search policy (untrusted; v2.9d)

1. Condition on 24 maximin interface pairs (Astra's protocol), then exclude the window and the obstacles.
2. Always work on the largest open piece, because it dominates the unknown term.
3. Park pieces smaller than 1/2000 of the proved lower volume, within a parking budget of a quarter of the
   tolerance. Parked pieces stay charged.
4. After a screen with n > 512 ambiguous receivers, take the **hull step** (§4e) first: enforce the convex-hull
   vertices of each single-blocker dark set. The set comes from the stored obligation list, or, if the list
   overflowed, from a scanline-extreme scan (`hull_proposals.scan_extremes`). The step is skipped for a lineage
   whose last hull step did not shrink the obligations by 20%.
5. Otherwise, after a screen with n ambiguous receivers:
   * if the piece touches an obstacle and n > 1500:
     * if it lies outside one obstacle facet and touches its plane, split by a parallel plane at half its height;
     * otherwise split by a dyadic axis plane half way from the contact vertex (apex isolation);
   * if the previous split shrank R by more than 20% and n > 1500, split at the long-axis midpoint;
   * otherwise enforce up to 96 ambiguous receivers exactly, in this priority order:
     1. single-blocker dark receivers (non-fragmenting);
     2. receivers wrong at the anchor source;
     3. interface receivers.

     At most 16 fragmenting receivers go into one batch: multi-blocker dark receivers (unions of cones) and lit
     receivers (subtractions).
6. Snapshot the exact bound after an IN settlement when the float hint says it can be near the tolerance. Stop at
   TV ≤ tolerance, when no work remains, or at the time budget. The kernel's release decides the status.

**Why the obstacle-contact rule matters.** Shadow boundaries pass through occluder edges, so feasible cones often
have an apex on an obstacle. At that vertex both envelopes degenerate: nearly every receiver is ambiguous while the
whole piece is consistent. In Astra's kernel this caused 37 of the 56 failures. §4d and §4f remove most of the
degeneracy on the envelope side.

## 6. Exploiting the posterior theorem directly (analysis, v2.9)

The certificate needs only l_c ≤ μ(F ∩ C_c) and the frontier charges, not a reconstruction of F. Two
consequences were examined.

1. **Unknown mass inside a cell is free for that cell.** A live piece inside cell c is not charged in U₋c. For a
   concentrated posterior, refining or grid-splitting live pieces inside the dominant cell cannot improve the bound
   beyond the other cells' terms. Splitting multi-cell live pieces at 128-grid planes is a pure kernel split, with
   no physics, and turns their charge into per-cell charges (the capacity form).
   * **Measured on the 54 canary failures with positive lower mass:** the bound with every live piece decomposed
     exactly per cell equals the frontier bound in every case (to three digits). The tail consists of spread
     posteriors over 10³–2 × 10⁵ cells with almost no mass in any single cell. **EMPIRICALLY TESTED** (no effect
     on the tail), so grid-aware refinement was not adopted.
2. **How loose is B = 1 − Σ_c min(p_c, a_c) for the released payload p = l/Σl?** For a cell set S, let U(S) be
   the largest unknown and slack mass that can lie in S. The exact worst case over the posteriors consistent with
   the credits is max_S U(S)(L − l(S)) / (L (L + U(S))), with L = Σ l. **MATHEMATICALLY PROVEN** (elementary):
   the adversary maximizes q(S) − p(S) by placing all admissible unknown mass in S. Consequences:
   * For spread posteriors the maximum is attained by S = the cells with little payload that the unknown pieces
     reach, and it approaches B (both ≈ U/(L + U)).
   * Only when the unknown mass can lie only in cells that already carry most of the payload (e.g. two cells
     of equal mass) is B up to 2× pessimistic.
   * Moreover, no payload can guarantee less than half the TV diameter of the consistent set, so even a
     minimax-optimal payload gains at most 2×.
   * Because the hard tail is spread, a refined bound would gain little. It is recorded, not implemented
     (it would add a combinatorial maximisation to the trusted base).

The productive direction was instead to reduce the geometric work that determines U: exact possible shadows (§4d),
receiver convexity (§4e) and contact vertices (§4f).

## 7. What the design does not claim

* Termination or a latency bound for every input. Abstention (INCOMPLETE) is always available.
* Exact recovery of F. Only the TV certificate is claimed.
* Protection against hostile code in the same process. This is a single-owner research model: the search is
  assumed buggy at worst, not malicious at the interpreter level.
* Machine-checked proofs of the geometric lemmas. The arguments above are manual; the integer arithmetic of the
  screens is mechanized (z3), and the polytope arithmetic is runtime-certified when `CBC_VERIFY_SPLITS=1`.
  `XLF_CBC_IMPLEMENTATION.md` lists the code-path review, the tests and the remaining gaps.
