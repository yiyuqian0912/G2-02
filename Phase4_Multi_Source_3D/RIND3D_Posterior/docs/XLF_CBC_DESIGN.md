# XLF correct-by-construction labeller: unified specification and design (v3.0)

Branch `claude/xlf-correct-by-construction`. This design builds on:
* Astra's theorem (`XLF_PROOF_FOUNDATIONS.md`);
* Astra's transition design (`XLF_CORRECT_BY_CONSTRUCTION.md`);
* Astra's exact geometry (`experiments/global_inverse/`).

Code: `experiments/credit_solver/cbc_kernel.py`, `screen2.py`, `polycheck.py` (trusted); `cbc_search.py`,
`shadow_facets.py`, `hull_proposals.py` (untrusted).

**v3.0 has one solver, one trusted configuration and one certification contract.** There is no certified or speed
mode: the per-split re-verification of v2.9e is replaced by a code-level proof of the split primitive (§8), and it
survives only as an external test oracle.

Assurance labels (used in all XLF_CBC documents):

| label | meaning |
|---|---|
| **PROVEN** | a complete manual argument: mathematical (about the abstract operation) or code-level (about the actual source, statement by statement); not machine-checked |
| **IMPLEMENTATION-VERIFIED** | the exact expression evaluated by the code is mechanically proved (z3: `smt_screen.py`, `smt_kernel.py`) over the stated domain |
| **EXACT-TESTED** | the code agrees with an independent exact (rational or integer) reference on the stated test population; not a proof |
| **EMPIRICALLY TESTED** | agreement with references, audits or chaos campaigns on real tasks; not a proof over continuous regions |
| **CONDITIONAL** | holds if a stated, unverified assumption holds |
| **OPEN** | not established |

## 0. Three kinds of correctness

| kind | statement | how it is established |
|---|---|---|
| **A. Conceptual** | The continuous problem is well posed, and the frontier posterior theorem holds. The problem: legal sources, closed-segment visibility, the target as the push-forward of uniform volume on F onto 128³ cells. The theorem: for sound credits, every decoded normalized payload p satisfies TV(p, q) ≤ 1 − Σ min(p_c, a_c) (§1). | PROVEN (Astra; restated in §1). The per-cell inequality and the integer rounding are IMPLEMENTATION-VERIFIED (`smt_kernel.py` T1–T3). |
| **B. Algorithmic** | An abstract solver whose state changes only through the transitions of §2a never violates I1–I6. A release from any reachable state is a valid certificate for its own bytes, whatever the search does. | PROVEN (§2a–§4h): each transition preserves the invariants, the release reads only invariant-protected quantities, and the search cannot call anything else. |
| **C. Implementation** | The Python/Numba/C++ routines realize those transitions exactly. | Mixed, per primitive: code-level proofs (§8), z3 lemmas for the arithmetic cores, exact differential tests, and a few retained cheap runtime guards. `XLF_CBC_IMPLEMENTATION.md` §1–§3 lists exactly what is proven and what is assumed. |

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

## 2a. The trusted state and the only allowed transitions (unified specification)

**Trusted state** (owned by one `Kernel` object; nothing else holds a reference to any mutable part of it):

* **Problem** (`private_problem`, immutable after construction): obstacle rows as tuples of Python ints, the class
  array (a read-only copy), the window origin and side, and the derived caches (obstacles clipped to the cube,
  their facet geometry, the clearance covers).
* **`State`** (immutable dataclass, replaced atomically by `_commit`): live pieces (polytope, residual obligations
  R_P as a read-only array or "all but `done`", read-only candidate masks, `deferred`), `lower`, `upper_in`, the
  next handle and the window-excluded flag.
* **Release record:** the best (bound, payload) pair computed by `snapshot`, plus the taint flag.

**Allowed transitions** (public kernel methods; arguments are validated; each commits once or raises with the
state unchanged):

| transition | argument the search may choose | effect |
|---|---|---|
| `condition_pair(d, a)` | one dark and one lit receiver | every piece P ↦ P ∩ ⋃C(d) ∖ ⋃C(a) |
| `exclude_window()` | — | every piece minus the closed window box and the closed obstacles |
| `split(h, row)` | integer row (non-constant, \|coef\| ≤ 2⁴⁰⁹⁶) | P ↦ (P ∩ {row ≤ 0}, P ∩ {row ≥ 0}); both halves kept |
| `screen(h)` | — | classify R_P: drop P (contradiction), settle P (credit + shell), or shrink R_P |
| `enforce(h, ids)` | integer receiver ids (FREE, in the window) | exact cone cuts; children with R_P minus ids |
| `snapshot()` | — | payload + exact bound on the decoded payload; the best pair is recorded |
| `release(tol)` | exact rational tolerance | CERTIFIED iff the best recorded bound ≤ tol, else INCOMPLETE; ERROR if tainted |

Read-only queries (`handles`, `is_open`, `open_handles`, `volume`, `pending_count`, `poly` (a copy), `contact`,
`receiver_classes`, `envelope_rows` (copies), `positive`, `lower_total_float`, `unknown_volume`) return immutable
values or copies.

**Invariants** I1–I6 are as in §2. **Theorem B (algorithmic soundness).** Every state reachable from the initial
state (one live piece, the unit cube) by any sequence of allowed transitions satisfies I1–I6. Every release from
such a state is either INCOMPLETE/ERROR or CERTIFIED with TV(decode(payload), q) ≤ bound ≤ tol.
*Proof.* By induction over transitions, using the preservation table of §3 and the lemmas of §4–§4h; the release
evaluates the frontier theorem (§1) on its own bytes. The search only chooses the arguments in the table above, and
every argument is validated before use, so the search policy does not appear in the proof. ∎

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

## 4d. The exact possible shadow (PROVEN; v2.9b)

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

## 4e. Receiver convexity (PROVEN; used by the search, v2.9c)

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

## 4f. Contact vertices in the guaranteed shadow (PROVEN; v2.9c)

For a vertex v of P inside the closed body K, every segment [v, r] meets K, so S_K(v) is all
of space. The guaranteed shadow is therefore computed from the vertices outside K only:
r ∈ ⋂_{v ∉ K} S_K(v) puts every vertex (those in K trivially) into C_K(r), and P = conv(vert P)
⊆ C_K(r). If all vertices lie in K, then P ⊆ K is illegal a.e. and the guaranteed shadow is
the whole window. (Envelope evaluated the rows of contact vertices, which over-constrained:
sound, but it left dark receivers behind obstacle contacts ambiguous.)

## 4g. Polytope arithmetic: FS, Lemma V and the split primitive (PROVEN; code-level proof in §8)

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

**Per-split certificate (v2.9d–e; since v3.0 a test oracle, `polycheck.instrument_splits`).** For every split
inside every transition, the vertex sets of both halves must equal the two sets of Lemma V,
computed independently from the parent's (FS-certified) edges, and both halves must pass
`check`. Inductively from the cube, every polytope the kernel ever creates is then FS-valid,
and every split is exact: the halves cover the parent, overlap only on the plane, and carry
exact volumes. This certifies the cover invariant I1 for the polytope arithmetic in every run
without a physical replay. It is on by default. Cost in pure Python: +23% wall time over the canary, and up to
about 2× on cut-dominated runs (XLF_CBC_RESULTS §5).

v3.0 replaces this per-split runtime check by Theorem P (§8), a code-level proof that `Poly.split` satisfies
Lemma V and preserves FS. The check is kept as an independent oracle: in the test suite, in the chaos campaigns,
and as `polycheck.instrument_splits()` for audits. It never failed on the 494,099 canary splits or the 1,009 chaos
splits of v2.9e (EMPIRICALLY TESTED), consistent with the proof.

Every polytope that receives an IN credit, and every live polytope charged in a snapshot, still passes `check`.
This is a cheap guard (0.5% of wall time) on the only two places where volumes enter the bound; see
IMPLEMENTATION §3 for why it is retained.

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

## 8. Code-level proofs of the trusted primitives (PROVEN, manual; v3.0)

These arguments are about the actual source as of v3.0 (`global_inverse/poly.py`, `solver.py`, `envelopes.py`,
`raster.py`, `credit_solver/cbc_kernel.py`, `polycheck.py`, `kernel.py`, `xlf_assurance/ledger.py`). Python
integers and `Fraction` are exact, and `tuple`/`dict`/`set` operations have their documented semantics.
Arithmetic steps marked [z3 X] are mechanized in `smt_kernel.py` (lemma X).

### 8.1 `Poly.split(row)` (Theorem P)

**Precondition.** P satisfies FS (§4g) with pairwise distinct face planes, and all vertices are primitive with w > 0.
`Poly.cube()` satisfies this by inspection, and `polycheck.check(Poly.cube())` passes.

**Theorem P.** `split` returns (P, None) iff every vertex has row ≤ 0, (None, P) iff every vertex has row ≥ 0 and
some vertex has row > 0, and otherwise (N, Q) with:
* vert N = {v : row(v) ≤ 0} ∪ X and vert Q = {v : row(v) ≥ 0} ∪ X, where X is the set of crossing points of the
  edges with strictly opposite end signs (Lemma V);
* N and Q both satisfy FS with distinct face planes.

Hence conv N = P ∩ {row ≤ 0} and conv Q = P ∩ {row ≥ 0}.

*Proof, following the code.*
1. `side(row)` returns −1 iff every vertex value is ≤ 0, and +1 iff all are ≥ 0 and one is > 0. Each per-vertex
   sign is exact: the filter decides only when \|v\| > 10⁻¹²·Σ\|terms\| with normal coordinates (error ≤ 6u·Σ\|terms\|,
   §IMPLEMENTATION 3), and the exact integer value decides otherwise. So the two early returns are exactly the
   first two cases. `signs` is then computed exactly for every vertex; the repeated max/min tests are redundant and
   consistent. In the main path both a strictly negative and a strictly positive vertex exist.
2. **Crossing points.** For consecutive cycle entries (last, p) with u = s(last) and v = s(p) of strictly opposite
   sign, the code builds h = primitive(\|v\|·last + \|u\|·p). Then row(h) = 0 [z3 S1a], h lies strictly inside
   the segment with w > 0 [z3 S1b], and `primitive` divides by the positive gcd. The value is symmetric in the two
   endpoints, so both faces sharing that edge (FS) obtain the identical canonical tuple. The `cut` cache only
   avoids recomputation.
3. **Clipping one face** f = (q₀ … q_{k−1}), which is strictly convex and cyclic. The loop appends, for each edge
   (q_{i−1}, q_i), the crossing point if the signs are strictly opposite, then q_i if s(q_i) ≤ 0. This is
   Sutherland–Hodgman.
   * The vertex signs along a strictly convex cycle change from − to + and back at most once each (a line meets
     the boundary of a strictly convex polygon in at most two points, or along one edge).
   * If all s ≤ 0, then a = f.
   * If all s ≥ 0, then a holds at most two zero vertices, because three zero vertices of a strictly convex
     polygon cannot lie on one line. Such a face is dropped by `len(a) >= 3`. A face with all s = 0 would lie in the
     cut plane, which supports P; then every vertex of P is on one side, so this cannot occur in the main path.
   * Otherwise a = (entry, negatives…, exit), where entry and exit are crossing points or zero vertices. This is
     the cycle of f ∩ {row ≤ 0}: a strictly convex polygon with ≥ 3 corners. A crossing point is a corner because
     the edge crosses the line transversally. At a zero vertex the polygon turns from an edge leaving the line to
     the chord on the line.
   * The same holds for b with {row ≥ 0}.
4. **Faces of N.** The facets of N = P ∩ {row ≤ 0} are the 2-dimensional f ∩ {row ≤ 0}, plus the cap
   P ∩ {row = 0}. The code emits exactly the first kind (`len(a) >= 3` iff 2-dimensional, by step 3) and the cap.
   The planes stay distinct: the face planes are distinct by hypothesis, and the cut plane differs from each
   because both signs occur.
5. **The cap.** `cap` collects every crossing point and every zero vertex.
   * These are exactly the vertices of the cap polygon C. A zero vertex is extreme in P, hence in C. A crossing
     point is the transversal intersection of an edge with the plane, a 0-face of C. Every vertex of C is of one
     of these two kinds (Lemma V).
   * All cap points are therefore in strictly convex position, and none lies inside an edge of C: an edge of C
     is f ∩ {row = 0} for a face f, and its interior meets ∂f in no vertex because f is strictly convex.
   * `cap_hull` drops the coordinate with the largest \|row\| component. That projection is injective on the plane.
     It sorts exactly (Fraction keys) and runs Andrew's monotone chain with the exact homogeneous orientation
     [z3 S2a, S3], keeping strict left turns. On points in strictly convex position this returns all of them in
     cyclic order. C is 2-dimensional (the plane meets int P), so it has ≥ 3 vertices.
6. **Edge pairing of N.** An edge of N is one of two kinds.
   * A portion of an edge of P. The two incident faces of P are both clipped, and the portion's endpoints (vertex
     or crossing point) are canonical, so it appears in both cycles.
   * A chord f ∩ {row = 0}, shared by the clipped f and the cap. If f ∩ {row ≤ 0} is only that chord (f touches the
     plane along an edge e and lies in {row ≥ 0}), the other face at e has negative vertices and supplies the
     clipped face. If both faces at e were in {row ≥ 0}, C would contain e but no 2-dimensional neighbourhood of
     its midpoint, contradicting that C is 2-dimensional with e on its boundary.

   Connectivity, single vertex links and Euler characteristic 2 hold because the face list is exactly the boundary
   complex of the convex polytope N.
7. `Poly(...)` sets the vertices to the union of face vertices, which is vert N. The argument for Q is symmetric. ∎

**Consequences.** By induction from the cube, every polytope the kernel creates satisfies FS. Every split preserves
the cover: conv N ∪ conv Q = P, with overlap only on the plane. The fan volume of each polytope is exact.
`native/cells.cpp` `split` is a statement-by-statement transliteration of steps 1–6 in GMP integers (exact signs,
the same crossing formula, the same clipping loop, the same monotone chain with exact cross-multiplied
comparisons). It is EXACT-TESTED against the Python path: per-cell credits are identical.

**Evidence beyond the proof** (not a substitute for it):
* EXACT-TESTED: independent half-space vertex enumeration on degenerate random cuts and real cones, and FS checks
  (`test_poly_exact.py`).
* EMPIRICALLY TESTED: 494,099 canary splits and all chaos splits passed `verify_split`; certified-mode and
  speed-mode outputs are byte-identical on all 786 channels both certified.

### 8.2 Set algebra built from `split`

* `within(P, rows)`. Loop invariant: `poly` = P ∩ {r₁ … r_t ≤ 0}, or None once it has empty interior (split's
  positive-only case). Result: P ∩ ⋂{rᵢ ≤ 0}.
* `without(P, rows)`.
  * If some row has `side > 0`, P lies in {r ≥ 0}, its intersection with the closed convex set has empty
    interior, and [P] is returned.
  * Otherwise, loop invariant: `result` = {P ∩ {r₁ … r_{t−1} ≤ 0} ∩ {r_t ≥ 0}}, pairwise interior-disjoint, and
    `poly` = P ∩ {r₁ … r_t ≤ 0}. The final `poly` (the removed part) is discarded. The union of the result is P
    minus the closed set, up to null sets.
* `subtract_many(parts, sets)` iterates `without`. `apply_pair` computes ⋃ⱼ (P ∩ C_j(d) ∖ ⋃_{k<j} C_k(d))
  ∖ ⋃ₖ C_k(a): for each hit j, `pieces` = (P ∩ C_j(d)) ∖ C_j(a), then the other lit cones and the previous dark
  cones are subtracted. Previous dark cones that missed P (not in `hits`) are null on P, so they need not be
  subtracted. The pieces are disjoint (first cover).
* `Kernel._exact_cut`: subtraction for lit receivers; first cover over the candidate cones for dark receivers (the
  same argument). Candidate masks are justified in §4b.
* `exclude_window`: `subtract_many` with the six window rows (u = n·e_axis: x_axis ≥ o/n, x_axis ≤ (o + L)/n) and
  then with each closed obstacle.
* `clearance_partition`: `outside` = P minus each enlarged body (without), and `possible` = the parts inside some
  enlarged body (within, first cover). So outside ∪ possible = P, disjointly. The shell is `possible` minus K and
  its six 2⁻²⁶ translates. `cleared(p)` holds when one facet row puts every vertex at distance > 2⁻²⁶ [z3 Q2], so
  the whole convex piece is at that distance. The offset rows use the directed square root [z3 Q1, Q2].

### 8.3 Cones, envelopes and separators

* `cone(body, r)`:
  * The rows are the facets with h = f(r) ≥ 0, plus hᵢ f_j − h_j fᵢ for front i (hᵢ > 0) and back j (h_j < 0).
    By [z3 C1, C2] the segment from r meets {fᵢ ≤ 0, f_j ≤ 0, tangent ≤ 0} exactly when these rows hold, for one
    front, one back and one tangent facet.
  * For several facets the feasible t form an intersection of intervals [lbᵢ, 1] and [0, ub_j]. In one
    dimension such a family intersects iff every pair does (Helly), so the pairwise rows suffice.
  * A receiver with no front facet is inside K; the code raises (abstain).
  * Dropping all-zero rows drops the trivially true 0 ≤ 0.
* `Envelope`:
  * inner keeps, per pair (i, j), the row of maximal h_j/hᵢ, which implies the others given fᵢ(r) ≤ 0 [z3 E1].
    That facet row is present because some vertex has fᵢ ≥ 0.
  * outer keeps the facet rows where every vertex has fᵢ ≥ 0, and per all-vertex pair the row of minimal h_j/hᵢ
    [z3 E2]. The ratio is quasi-linear on P, so its extremes are at vertices.
  * `extreme_pair` compares ratios by cross-multiplication with positive denominators [z3 S3 analogue].
  * Since v2.9c, inner uses only vertices outside K (§4f).
* `separators`: exact check max over vert(K ∩ cube) ≤ 0 ≤ min over vert P, then [z3 E3]. `penumbra.directions`
  performs the same exact check internally.
* `raster.fixed` and the scanline arithmetic: IMPLEMENTATION-VERIFIED (`smt_screen.py`).

### 8.4 Accounting, serialization and release

* `_commit`:
  * It deletes every removed handle before inserting children, so a stale or duplicate handle raises KeyError
    before any change.
  * Children receive fresh handles, and the new State is built completely before one assignment.
  * Credits are added only through its `lower` and `upper` arguments, which only `_settle` supplies, from
    `_grid_credit` of the legal parts of the piece being removed in the same commit. A piece can therefore be
    credited at most once, and only when it leaves the live set (no double counting).
* `_grid_credit`:
  * Interior blocks are those whose eight corners satisfy every validated facet row; by convexity they lie in Q and
    receive exact cell volumes.
  * Boundary cells receive floor/ceil of the exact volume of the clip, with clips by Theorem P.
  * Conservation against the exact Q volume is checked.
* `snapshot`:
  * The unknown charge sums ⌈vol⌉ over every live piece, deferred ones included.
  * `single_cell` uses exact integer grid bounds of the vertices.
  * `wire_bound` evaluates the frontier bound on `decode(payload)` with integer floors [z3 T1–T3].
* `encode_int`:
  * It assigns exact ticks summing to 2⁵² (largest remainder via integer `divmod`).
  * Each value n/2⁵² is an exact binary64.
  * `decode` checks sorted ids, finite values in (0, 1] and an exact sum of 1.
* `release`: it returns the payload of the best recorded snapshot only if that bound is ≤ the exact tolerance.
  `certify` writes status, payload and bound only from the release.

## 9. What the design does not claim

* Termination or a latency bound for every input. Abstention (INCOMPLETE) is always available.
* Exact recovery of F. Only the TV certificate is claimed.
* Protection against hostile code in the same process. This is a single-owner research model: the search is
  assumed buggy at worst, not malicious at the interpreter level.
* Machine-checked proofs of the geometric lemmas or of the control flow of the code. The arguments are manual
  (mathematical and code-level, §8). The arithmetic cores are mechanized: z3 proves the screen arithmetic
  (`smt_screen.py`) and 16 kernel lemmas (`smt_kernel.py`). `XLF_CBC_IMPLEMENTATION.md` lists exactly what is
  proven, tested and assumed.
