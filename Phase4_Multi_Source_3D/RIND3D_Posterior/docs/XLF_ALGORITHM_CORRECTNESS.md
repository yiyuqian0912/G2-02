# XLF — Algorithmic correctness (Part B)

> **Status: incomplete.** XLF was frozen on 2026-11-18. The abstract algorithm, its invariants and the soundness,
> coverage and integration theorems below are complete at the level of exact mathematical objects.
>
> **Unresolved proof gaps and open questions**
> * **G-B1.** Asymptotic completeness (§5.2) is proven for exact oracles with fair refinement. For the finite-precision
>   oracles there is only a sketch: the residual undecided volume is O(u) times the area of the event surfaces.
> * **G-B2.** No fixed-time completion guarantee exists, and none is claimed. The edge-fan lower bound (§6) is a
>   sketch.
> * **G-B3.** The actual refinement policy (multires thresholds, fine stage, stalled-node handling) is not proven fair.
>   Termination holds only through the depth and time budgets.
> * **G-B4.** The cost model (§6) has not been validated against measurements (Part D).
> * **G-B5.** Certificate completeness: the abstract invariant (P) requires both children of every split. See
>   IMPLEMENTATION_ASSURANCE R1 (Astra's counterexample): the checker must enforce this.

## 1. Primitives (specifications only)

Each oracle may answer "unknown". Only a positive answer carries a claim. Here Q is a closed convex polytope, r a free
receiver, and c a component.

| Primitive | A positive answer guarantees |
|---|---|
| DARK_c(Q, r) | every s ∈ Q has [s, r] ∩ K_c ≠ ∅ (Lemma 3.1) |
| LIT_c(Q, r) | every s ∈ Q has [s, r] ∩ K_c = ∅ (Lemma 3.4) |
| REL(Q) | a superset of the components meeting some segment from Q to W |
| IN_D(Q) | Q \ D is null (window box, clearance, world) |
| OUT_D(Q) | Q ∩ D is null (inside B_W, or inside some K_c) |
| SPLIT(Q, h) | the pair (Q ∩ {h ≤ 0}, Q ∩ {h ≥ 0}), exactly |
| VOL(Q ∩ c) | an interval containing Vol(Q ∩ c) |
| TVB(lo, hi, pool) | Thm 6.2 of the foundations |

From these, define two receiver-level predicates:
* CONS(Q, r) holds if either:
  - X(r) = 0 and some c ∈ REL has DARK_c; or
  - X(r) = 1 and every c ∈ REL has LIT_c.
* REF(Q, r) holds if either:
  - X(r) = 1 and some c has DARK_c; or
  - X(r) = 0 and every c ∈ REL has LIT_c.

## 2. Abstract algorithm XLF-A

```
P := {(Ω, O = W_f)};  Leaves := ∅                    # O: receivers not yet proven consistent on the region
while P ≠ ∅ and budget remains:
    (Q, O) := policy.pick(P)                          # any order
    if OUT_D(Q):                 Leaves += (Q, OUT);  continue
    for r in policy.subset(O):                        # any subset, any order
        if REF(Q, r):            Leaves += (Q, OUT(r)); next region
        if CONS(Q, r):           O := O \ {r}
    if O = ∅ and IN_D(Q):        Leaves += (Q, IN);   continue
    either  P += {(Q⁻, O), (Q⁺, O)} for any plane h   # SPLIT; both children, the open set inherited (or a superset)
    or      Leaves += (Q, UNDET)
on budget exhaustion: every (Q, O) remaining in P becomes UNDET
lo_c := Σ_IN VOL⁻(Q∩c);  hi_c := Σ_{IN ∪ UNDET} VOL⁺(Q∩c)     (sums rounded outward)
status := certified iff Σ lo_c > 0; label q̂ := round(lo/Σlo); τ := TVB + TV(q̂, lo/Σlo)
```

The policy (`policy.*`) is untrusted. It covers:
* the split plane;
* the receivers tested;
* collective block scans;
* local-first clipping by rounded constraint planes with δ-slabs;
* pencil and solid splits;
* caps and hints.

The local-first fine stage is SPLIT plus re-processing. The outer piece becomes a leaf only when REF proves it.

## 3. Invariants

* **(P) Partition.** The regions of P ∪ Leaves are closed convex polytopes with pairwise null intersections, and their
  union is Ω.
* **(O) Open set.** For every (Q, O) in P, every IN leaf and every r ∈ W_f \ O: r is consistent on all of Q.
* **(L) Leaf truth.**
  - OUT: Vol(Q ∩ F) = 0.
  - IN: Vol(Q \ F) = 0.
  - UNDET: no claim.

**Preservation (proven).** Each statement follows from the primitive specifications.
* *Initialisation.* (P) and (O) hold trivially.
* *SPLIT.* The children tile the parent exactly, so (P) is preserved. Consistency on Q implies consistency on any
  subset, so (O) is preserved with the inherited O.
* *CONS.* Removing r keeps (O) by definition.
* *REF.* Every s ∈ Q contradicts X(r), so OUT is true.
* *OUT_D.* OUT is true.
* *IN.* O = ∅ together with (O) means every free receiver is consistent on Q, and IN_D gives Q ⊆ D up to a null set.
  So Q ⊆ F up to a null set.
* *UNDET and budget exhaustion.* No claim is made, so (L) holds vacuously.

An "unknown" answer never produces a claim. It only leaves r in O or the region in P.

## 4. Soundness, coverage, integration (B2, B3)

**Theorem 4.1 (soundness; proven).** At termination, for every cell c:

lo_c ≤ Σ_IN Vol(Q∩c) ≤ Vol(F∩c) ≤ Σ_{IN∪UNDET} Vol(Q∩c) ≤ hi_c.

*Proof.*
1. By (P), the leaves tile Ω.
2. By (L), F ∩ c is, up to null sets, the union of (IN leaves ∩ c) and parts of (UNDET leaves ∩ c).
3. The VOL enclosures and outward rounding give the outer inequalities. ∎

**Consequences.**
* No positive-volume feasible region is discarded without justification. Only OUT leaves drop mass, and each carries a
  REF witness or an OUT_D proof.
* No region is counted twice, by (P).
* Normalisation and TV follow from Thm 6.2/6.4 of the foundations. The guarantee covers multiple blockers and
  overlapping explanations, because DARK needs one component for all of Q and otherwise r stays open.

**Theorem 4.2 (final deterministic guarantee; conditional).** Assume:
* the implementation realises the primitives;
* the certificate satisfies (P), including both children for every split;
* status = certified.

Then TV(q̂, q*) ≤ τ, where q* is the exact posterior of the declared model.

## 5. Termination and abstention (B4)

1. **Termination (proven).** Every split increases depth, regions deeper than d_max become UNDET, and a split has two
   children. So at most 2^{d_max+1} − 1 regions are visited. The wall-clock or node budget also terminates the run;
   the remaining P becomes UNDET, or the run is reported as incomplete.
2. **Asymptotic completeness (conditional; G-B1).** Assume:
   - the oracles are exact and complete at points, which holds by Thm 2.1 of the foundations;
   - refinement is fair, i.e. diameters tend to 0.

   Then the undecided volume tends to 0. The undecided points lie in the finite union of event planes, facet planes,
   ∂B_W and the clearance surface, which is null.
3. **Abstention (proven sound).** Report "certified" only if Σ lo > 0 and τ ≤ target. Otherwise abstain and use the
   fallback.
4. **No fixed-time guarantee (G-B2).**
   - Vol(F) can be arbitrarily small relative to a cell.
   - Near an occluder edge the event planes form a fan around the edge line. Any box that meets the line is ambiguous
     for every fan receiver, so bisection alone needs ~ε^{-1/2} boxes to reach undecided volume ε. Pencil splits need
     only O(#binding angles). This is a sketch.

## 6. Complexity (sketch; G-B4)

* **Per-region cost.**
  - prep: O(Σ_c |V(Q)| n_c²).
  - Collective scan: O(#visited receiver blocks × active constraints).
  - Receiver-space convexity of I_c(Q) means only blocks near its boundary are refined.
* **Near ∂F.**
  - Bisection: O(Area³/ε²) leaves for undecided volume ε.
  - Clipping by binding constraint planes: O(m) cuts per cell and slab volume O(Area·δ), with δ ≈ 2⁻⁴⁹. This is the
    computational content of local-first.
* **Geometric drivers.**
  - the number of relevant components and facets;
  - binding receivers per cell;
  - occluder edges and vertices inside F (fans);
  - the clearance shell where F touches a body. It is currently bisected, and an offset-plane split is recommended.

## 7. Proof complexity (B5)

* The proof uses the 8 primitive specifications, 3 invariants and 2 integration theorems. All of them are independent
  of the search policy.
* What is proven once, for the algorithm:
  - preservation of (P), (O) and (L) by each step;
  - Thm 4.1;
  - the TV bound.
* What is checked per run: Σ lo > 0, the computed τ, and that the run used the verified implementation.
* The legacy Tau-pencil verifier had to re-derive float decisions (snapping, derived signs, symbolic facts) on objects
  that were never exact. Its proof was trace-level.
* XLF's claims are local to leaves, and its proof is Hoare-style over exact objects.
* The replay checker re-establishes (O) and (L) for IN leaves without inheritance. It therefore validates the
  *implementation* of the (O) bookkeeping and of (P), not the mathematics.
