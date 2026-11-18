# XLF — a self-certifying, local-first exact posterior labeller (research prototype)

Branch `yushu/3d-xlf-research`, package `rind3d_posterior.xlf`. Research only: not merged, not a production labeller.

## 1. Why a new solver

The v3 pipeline certified the float tau-pencil engine after the fact. The engine decides with binary64 predicates:
it projects new vertices onto the cutting plane and snaps vertices within 1e-13. Its regions are therefore not the
exact polytopes its planes define. The verifier had to rebuild each region from outer and inner shifted copies. It
also had to re-derive signs the engine had only guessed ("symbolic facts", "derived signs"). It cut clearance-shell
wedges around occluder edges where float claims were false within 1e-15 of an edge. Finally it pooled bisection
leftovers. The verifier grew almost as complex as the solver and ran 3–10× slower.

The audit of the engine's exclusions found no bug that loses feasible mass: 0 feasible sources among ~21.9K points
rendered inside every excluded region of three tasks (`experiments/certify/out_audit*.py`). The cost is
architectural: proof obligations are discharged after the fact, on objects that were never exact.

XLF inverts this. The solver works only on exact objects and records a claim only after proving it, so the
computation is the proof.

## 2. Mathematical core

Notation: receiver r (exact dyadic point), source s, convex occluder K_c = {f_i <= 0}. The observation is DARK iff the
closed segment [s, r] meets the closed solid.

* **Segment lemma** (exact, certify/check.py). For an apex p with every f_i(p) strictly signed, let Phi(p) = {i : f_i(p) > 0}.
  Then [p, x] meets K_c iff f_i(x) <= 0 for i in Phi(p), and B_ij(p, x) = f_i(p) f_j(x) - f_j(p) f_i(x) <= 0 for the
  silhouette pairs (i in Phi, j not in Phi, adjacent). As a set of x this is the shadow volume C_c(p): a convex
  polyhedron. By symmetry of segments, x in C_c(p) iff p in C_c(x).
* **Lemma D (guaranteed shadow).** r is dark for every s in a convex polytope Q via c iff V(Q) is a subset of C_c(r),
  iff r lies in I_c(Q) = the intersection of C_c(v) over the vertices v of Q.
  *Reduction:* for each silhouette pair only the two angular extremes of w_v = (f_i(v), f_j(v)) bind. B_ij(v, r) is
  linear in w_v; any w_v proven, by interval cross products, to lie in the cone of the two extremes is implied.
* **Lemma L (guaranteed lit).** Suppose a plane g has g <= 0 on K_c, g > 0 on V(Q), and g > 0 at r. Then no segment
  from Q to r meets K_c. The candidate planes g are:
  * the constraints of C_c(r) (source form; the plane passes through r);
  * facets f_i;
  * the pencil planes lam f_i + mu f_j, lam, mu >= 0 (receiver form).

  For each edge, the feasible angles form an arc of length at most pi/2. By the sinusoid argument, "positive at r
  for some angle in the arc" holds iff it holds at one of the two arc ends, so two planes per edge suffice.
* **Lemma S (local inverse geometry).** Suppose exactly the constraints A of C_c(r) straddle Q and all others are <= 0
  on V(Q). Then C_c(r) ∩ Q = {g <= 0 for g in A} ∩ Q.
  * A DARK receiver with a single non-lit component is therefore a convex constraint on Q: the intersection of A.
  * A LIT receiver is convex when each non-lit component has a single straddling constraint.

  All pair constraints of one occluder edge pass through that edge's line (a pencil), and only their angular
  extremes bind. The interface receivers of the window thus reduce, edge by edge, to two planes: the "critical
  receivers" that pin the source.

## 3. Algorithm

An exact BSP of the source domain [0, 1]^3 is built on `certify.poly`:
* Planes are binary64 rows read as exact rationals.
* Signs are exact, using determinant filters plus Shewchuk expansions, with Simulation of Simplicity.
* Children {h <= 0} and {h >= 0} tile their parent exactly.

Per node Q:
1. **Collective classification** (`pred.prep_all` + `pred.scan_core`).
   * For every relevant component, compute receiver-space data once per node: the Lemma D constraints and the
     verified Lemma L planes.
   * Each window row is then a 1-D problem. Every constraint gives a rigorous t-range, checked at the boundary
     receivers with directed rounding.
   * A receiver whose observation is contradicted on Q is an OUT witness.
   * At most `rcap` receivers left ambiguous are re-tested exactly in the source form.
   * Children re-scan only the rows that held an ambiguous receiver. Recent witness rows are scanned first
     (any contradicted receiver is a valid witness).
2. **Coarse:** while more than `cap_fine` receivers are ambiguous, bisect (at cell boundaries while the node spans
   cells).
3. **Fine (local first).**
   * The source-form test of every ambiguous receiver collects the Lemma S halfspaces.
   * Per (edge pencil, sense) group, only the angular extremes are kept.
   * The node is cut by each kept plane h: binary64 rounding of the constraint, plus parallel copies h ± delta.
     delta (>= 2^-49) bounds the rounding, so both sides are provable.
     * The outer piece is OUT by its receiver, re-proven before it is recorded.
     * The slab is UNDET; its volume is an upper bound.
     * The inner piece is classified again from step 1.
   * Receivers with non-convex constraints split the node by one of their planes.
4. **Leaves.**
   * IN: every free receiver is proven consistent on the node, and the node is proven inside D. "Inside D" means
     exact side tests against the window box, and clearance > 2^-26 via an AABB gap or a facet with
     f_i > eps |n_i|.
   * OUT: a witness receiver.
   * UNDET: depth limits and slabs.
5. **Integration.** Per-cell volume enclosures come from the kernel's mid-radius fan volumes. Exact boxes give
   exact products. Other leaves are clipped recursively at cell boundaries, with explicit cell ranges.
   `certify.tvb` then bounds TV(q_hat, q) exactly in rationals.

## 4. Trusted computing base

* `certify.ar` (error-free transformations, expansions, filters).
* `certify.poly` (plane-based polytopes, exact clip, volume enclosure).
* `certify.scene` (canonical rows, exact vertices, adjacency).
* `certify.tvb`.
* `xlf.pred` (≈700 lines: the lemmas above with a priori error bounds).
* The leaf and partition logic of `xlf.solve`.

Plane choice, extreme selection, grouping, hints and caps are search heuristics. A wrong choice costs time or
precision, never correctness: every leaf claim is re-proven on the leaf's own exact polytope.

Tests: `tests/test_xlf_pred.py` checks recv_test and scan_rows against an exact rational segment oracle on random
occluders and on windows placed across shadow boundaries. Every status is exercised (dark, lit, convex, non-convex).
Under lemma S, the convex halfspaces must reproduce the observation exactly.

## 5. Status and measurements

See `docs/XLF_RESULTS.md` (to be written from `RIND3D_100K_v3_work/xlf/batch*.jsonl`).
