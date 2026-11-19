# XLF — Implementation and numerical assurance (Part C)

> **Status: incomplete.** XLF was frozen on 2026-11-18. FastLabel v3.4 is the production labeller.
>
> This document records the audit of `xlf/` and `certify/{ar,poly,scene,tvb}`: cc512b3 plus the fixes in 4d44da5.
> It is incomplete, and `xlf/` must not be used for production labels without the replay checker.
>
> **Unresolved gaps and open questions**
> * **G-C1.** The inheritance obligation (O) was reviewed call site by call site (§3), but nothing checks it
>   mechanically. The replay checker is the only per-run evidence.
> * **G-C2.** The end-to-end exact audit (synthetic scenes, exact rational audit of every leaf claim, and Σ of exact
>   leaf volumes = 1) was designed but not finished. The prototype was stopped at the freeze.
> * **G-C3.** The fault-injection and adversarial-policy tests (§6) were designed but not run.
> * **G-C4.** The near-zero argument for the former γ₃ filter (D4) is a written proof sketch, not machine-checked.
> * **G-C5.** There is no runtime guard for the error-free-transformation preconditions (no underflow or overflow,
>   §4.6).
> * **G-C6.** Checker certificate completeness is risk **R1** below. It is not fixed in code.
> * **G-C7.** Efficiency findings (§7) are analysed, not measured.

## 1. Trusted computing base (minimal)

| Primitive | Code | Reduces to |
|---|---|---|
| exact sign of a plane at a vertex | `poly.sos_sign`, `ar.det3/det4` | Shewchuk expansions + SoS; filter with an a priori bound |
| exact clip / partition | `poly.clip` | signs (above); combinatorics of a simple perturbed polytope |
| vertex enclosure | `poly.vertex_enclosure` | Cramer's rule, a priori bound or exact expansions |
| interval evaluation | `pred.aff, range_on, pair_coef, comb_coef, _cross_lo, _le_range, _gt_ray` | Higham γ_n bounds + `dn`/`up` |
| lemma logic | `pred.recv_test, _prep_comp, scan_core, scan_tree` | Thm 2.1/2.3, Lemmas 3.1–3.4 of the foundations |
| volume | `poly.volume`, IN_BOX in `solve._in_or_split`, `solve._cell_volumes` | fan determinants, interval sums |
| domain tests | `_side`, `_clear`, OUT_WINDOW, OUT_SOLID (+ `inside` map, `_plane_pair` offset) | signs, interval evaluation |
| bookkeeping | `solve._split`, rows inheritance, leaf recording, `add_dn`/`add_up` | invariant (O), (P) |
| TV and label | `tvb` | exact rationals |
| inputs | `scene.ExactScene` (RM ± RR exact check, exact vertices, ADJ, AABBs), window bits | Fractions |

Untrusted:
* the split and clip plane choices, extreme selection, caps, hints, sentinels and multires thresholds;
* stalled-node splits, and the δ of `_plane_pair` *when used for clipping*.

**Exception since cc512b3.** `_plane_pair`'s offset is now trusted. The `inside` map turns it into an OUT_SOLID claim.

## 2. Defects found and fixed (commit 4d44da5)

* **D1 (solver and checker; numerical soundness).** Per-cell lo/hi were accumulated with round-to-nearest `+`. The
  stored lo_c could exceed the exact sum of the pieces' lower bounds, and hi_c could fall below. The fix is `add_dn` /
  `add_up` (two_sum residual decides the direction). The regression test is `test_directed_accumulation_of_cell_bounds`.
* **D2 (checker accepted a false claim).** `check._inner_plane_ok` tested `A₃ − h₃ > E(1 + 2⁻³⁰)`, but E omits the
  rounding of sc·h_k.
  - Exact counterexample: f = −3x + y − 3·2⁻³⁰ (RR = 0). Here fl(3·fl(1/3)) = 1, so the computed E = 0. An offset of
    2⁻⁸² is accepted, yet 4 of the 8 vertices of the accepted piece lie outside the facet.
  - Fix: an exact rational test, sup over the unit cube of (f/sc − A) < 0.
  - The solver's own construction is valid. With offset ≥ max(8E, 2⁻⁴⁹) and the true error ≤ E + Σ|h_k|·u ≤ E + 4u,
    the margin is ≥ 11u. Property-tested.
* **D3 (kernel; formal).** `poly.volume` covered the rounding of the radius sum only by INFL = 1 + 64u. That is enough
  for at most ~64 fan triangles, and MV = 256 allows more. The fix is outward interval accumulation.
* **D4 (kernel; formal).** `_affine_at` and `affine_range` used γ₃ for three products plus three additions. γ₄ is
  needed.
  - Recorded counterexample: an error of 3.36 u·ab against a bound of 3 u·ab.
  - sos_sign decisions were not affected. Near zero the running error is ≤ 2u·ab + O(u²). This is a proof sketch
    (G-C4).

All defects have magnitudes of a few ulps. None was observed to change a reported label. Each broke a stated
enclosure invariant.

## 3. Obligations verified by review (no defect found)

* **Lemma logic.**
  - `_prep_comp` builds I_c(Q) exactly: zero facet values make the component unavailable, and the domination test uses
    interval cross products.
  - `recv_test` implements Lemma 3.2 and Lemma 3.4 (source form) with strict signs.
  - The DARK, LIT and convex statuses are sound.
* **ADJ.** It is exact: rows sharing ≥ 2 exact vertices, a superset of the edge pairs. This is the hypothesis of Thm
  2.3. Flat components pass `ExactScene` but are safe under strict aspect.
* **Row ranges.** `_le_range` / `_gt_ray` are correct by monotonicity of the exact bound in t, with only the boundary
  receivers verified by directed rounding. This needs x, y, z > 0 and N a power of two, both true for the data:
  N = 1024 and o ∈ [11, 1016].
* **scan_tree.**
  - Per-level state reuse is correct under DFS.
  - Pure and mixed blocks are handled correctly, and SOLID receivers are skipped.
  - `stop_level = lb` marks every undecided level-lb block.
  - SC_MANY never marks.
* **Inheritance (O) call sites.** Every child receives the marked blocks (`nst`) or a superset (`node.rows`):
  - `_bisect` (block-level, full-level, SC_MANY);
  - `_solid_split`, `_stuck_split`, `_pencil_split`;
  - `_fine` and `_clip_sequence` (all pieces re-processed);
  - `_in_or_split` window and clearance splits.
* **Fine stage.** Outer pieces are re-proven by `recv_test` on their own polytope. Slabs become UNDET or are
  re-processed. The `_plane_pair` E bound over [0,1]³ is irrelevant to these claims.
* **IN_BOX.** Exact for dyadic box coordinates, and rigorous for any binary64 box: there are ≤ 5 roundings, within the
  2⁻⁵⁰ margin.
* **`_cell_volumes`.** Cell-range tracking is correct, and the clip to [0,1]³ is justified.
* **`_clear` and OUT_WINDOW.**
  - `_clear` matches D: Euclidean distance via facet distance or AABB gap, with rigorous norms.
  - OUT_WINDOW uses the closed box, and OUT_SOLID the closed component.
* **SoS.** Signs are consistent across the run, and the checker's appended planes do not change existing signs.
  Children tile parents. The limit argument holds: volumes and affine extremes of the perturbed polytopes equal those
  of the exact ones.
* **tvb.** The derivation was checked; it is a valid relaxation of the exact supremum (Thm 6.3 of the foundations).

## 4. Preconditions (assumptions, not checked at run time)

1. Numba's binary64 uses round-to-nearest, no FMA contraction and no fastmath. `ar.self_test` is a partial check.
2. Numba cache freshness is keyed by source hash (`xlf.__init__`).
3. The kernel has no overflow, and no underflow inside the EFTs.
   - Sufficient: all nonzero PL entries in [2⁻²⁰⁰, 2²⁰⁰].
   - Observed: occluder coefficients ≥ 6e-5, bisection values ≥ 2⁻¹⁶¹.
4. Receivers are positive and dyadic, and L is a power of two (`pyramid`).
5. Components are bounded, and full-dimensional or handled under strict aspect.

## 5. Risks (open)

* **R1 (Astra's counterexample: an incomplete partition tree accepted as a global certificate).** The checker rebuilds
  only the nodes it reaches. It flags an internal node only when the node has *no* children.
  - It would accept a certificate in which a split node has a single child, or two children that are not
    complementary halfspaces of one plane. The missing region carries feasible mass that no leaf accounts for, and the
    TV bound is then invalid.
  - The solver always records both children, but the checker must not rely on the solver.
  - **Required fix.** The checker must verify all of the following:
    - every BSP node is either a leaf with exactly one claim, or has exactly two children with refs {2p, 2p + 1} for
      the same plane p;
    - no node is both a leaf and internal;
    - every node is reachable from the root and processed;
    - no node is left unprocessed, which also covers runs cut by the time limit.
  - The volume ledger Σ_leaves vol ∋ 1 is a cheap independent cross-check.
* **R2.** The checker shares the whole TCB (pred, poly, ar, scene, tvb) with the solver. It cannot detect a kernel or
  lemma bug.
* **R3.** Since cc512b3, the correctness of `_plane_pair`'s offset is load-bearing. Its proof is in D2, and the
  checker re-verifies it exactly after the fix.
* **R4.** (O) has no per-run evidence without the checker (G-C1).
* **R5.** Preconditions 1–3 of §4 are unguarded (G-C5).

## 6. Evidence (tests)

* `tests/test_xlf_math_lemmas.py`: 30 exact tests.
* `tests/test_xlf_numerics.py`: 26 enclosure and regression tests.
* Existing tests: `test_xlf_pred.py`, `test_certify_kernel.py`, all passing after the fixes (298 passed).
* Designed, not run (G-C2, G-C3):
  - an exact end-to-end leaf audit on synthetic scenes;
  - an adversarial policy test (random split planes; must stay sound);
  - a fault-injection test (empty inherited rows; the checker and the exact audit must flag false IN leaves).

## 7. Efficiency findings (analysis only)

* **Lit planes and VE facets.** The lit families omit the (K-vertex, Q-edge) planes (foundations Remark 3.6), so
  receivers near occluder vertices stay ambiguous. Adding VE planes would fix this.
* **Clearance shell.** Clearance failures are bisected down to 1e-9. When F touches a body this can explode. Recommended:
  an offset-plane split at f_i = ρ‖n_i‖ (the slab becomes UNDET), or an OUT_SHELL claim from vertex distance
  witnesses.
* **Edge fans.** Near occluder edges, bisection cannot shrink ambiguity (Part B §5); pencil splits are needed.

## 8. Per-run certification record (minimal)

* **Identity.**
  - code identity: the source hash covering the TCB and the solver;
  - Numba and NumPy versions;
  - `ar.self_test` passed.
* **Inputs.** Digests of the geometry bytes and the observation bits, and the window (o, L, N).
* **Results.**
  - per-cell (cell, lo, hi) with directed accumulation, and pool;
  - Lt > 0 (exact);
  - τ = TVB + TV(q̂, lo/Lt), computed exactly and rounded up;
  - q̂;
  - status, one of certified / abstained / incomplete, with the reason.
* **Checker.** Its result with version and outcome, or a recorded "not run" together with the assurance-model
  justification.
* **Ledger.** Leaf counts per kind, and the volume ledger Σ_leaves [vlo, vhi] ∋ 1.

## 9. Is per-label replay necessary?

* **Mathematically, no.** Theorem 4.2 of Part B gives a per-run guarantee from the invariants alone.
* **For this implementation, yes, at present.**
  - (O) and the certificate structure (R1, R4) have no other per-run evidence.
  - D1–D4 show that the numerical code had real defects, which only a review found.
* Dropping the checker would require all of the following:
  - R1 fixed;
  - G-C1 to G-C3 closed: the inheritance proof made mechanical, for example with runtime assertions, and the
    end-to-end, adversarial and fault-injection tests in CI;
  - a frozen code hash;
  - the checker run on an offline random sample of production labels, with zero failures over a predeclared sample
    size.

  Even then, the residual risk would include any shared-TCB bug (R2), which no replay checker of this design can detect.
