# XLF correct-by-construction labeller: design (kernel v2)

Branch `claude/xlf-correct-by-construction`. Builds on Astra's theorem (`XLF_PROOF_FOUNDATIONS.md`), Astra's
transition design (`XLF_CORRECT_BY_CONSTRUCTION.md`) and Astra's exact geometry (`experiments/global_inverse/`).
Code: `experiments/credit_solver/cbc_kernel.py` and `screen2.py` (trusted), `cbc_search.py` (untrusted).

Status words: **PROVEN** (argued here or in Astra's documents, not machine-checked), **IMPLEMENTED**, **TESTED**,
**OPEN**.

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

## 5. Search policy (untrusted)

1. Condition on 24 maximin interface pairs (Astra's protocol), then exclude the window and the obstacles.
2. Always work on the largest open piece, because it dominates the unknown term.
3. Park pieces smaller than 1/2000 of the proved lower volume. They stay charged.
4. After a screen with n ambiguous receivers:
   * if the piece touches an obstacle and n > 1500, split it by a dyadic axis plane half way from the contact
     vertex (apex isolation);
   * if the previous split shrank R by more than 20% and n > 1500, split at the long-axis midpoint;
   * otherwise enforce up to 96 ambiguous receivers exactly, those wrong at the anchor source first, then interface
     receivers.
5. Snapshot the exact bound after every IN settlement. Stop at TV ≤ tolerance, when no work remains, or at the time
   budget.

**Why the obstacle-contact rule matters.** Shadow boundaries pass through occluder edges, so feasible cones often
have an apex on an obstacle. At that vertex both envelopes degenerate: nearly every receiver is ambiguous while the
whole piece is consistent. In Astra's kernel this caused 37 of the 56 failures.

## 6. What the design does not claim

* Termination or a latency bound for every input. Abstention (INCOMPLETE) is always available.
* Exact recovery of F. Only the TV certificate is claimed.
* Protection against hostile code in the same process. This is a single-owner research model.
* Machine-checked proofs. The arguments above are manual. `XLF_CBC_IMPLEMENTATION.md` lists the tests and the
  remaining gaps between these lemmas and the code.
