# Certified deterministic posterior labels (RIND3D-100K v3)

This document states exactly what a certified label guarantees and why. It also lists the code that realizes each
step, the trusted base, and how to re-verify a released label. Measured cost and coverage are in
`CERTIFIED_LABELS_RESULTS.md`.

## 1. The certified statement

For one scene G, one window W and one source-specific response channel, the released artifact is a distribution
q̂ over the 128³ output cells together with a number ε (`tv_bound`). It comes with a proof that

**TV(q̂, q) ≤ ε, where q(c) = Vol(F ∩ c) / Vol(F) and Vol(F) > 0.**

The proof is deterministic. It is not a confidence statement. The definitions are those of `docs/RIND3D/SEMANTICS.md`:

* **Geometry.** The solid is S = ∪_c C_c, with C_c = {y : a_i·y + d_i ≤ 0}. The rows are binary64 parameters read
  as exact rationals by the canonical decoder `rind3d.rendering.polyhedra.decode_convex`. Every component is bounded,
  and the verifier checks this exactly (`certify.scene.bounded`).
* **Response.** A FREE receiver x is a voxel centre (o + v + ½)/N with N = 1024. It is response-inactive for source s
  iff the closed segment [s, x] meets the closed solid; contact blocks.
* **Observation X.** The stored binary response of every FREE receiver of the window: the zstd-packed bits of the
  scene container, decoded independently by `certify.scene.window_from_container`. A SOLID receiver carries no
  constraint.
* **Legal domain.** D(G) is the set of points of (0, 1)³ at distance > 2⁻²⁶ from the solid and outside the closed
  window box [o/N, (o+L)/N]³.
* **Feasible set.** F(G, X) = {s ∈ D(G) : the response of every FREE receiver equals X}.
* **Measure zero.** Boundaries of regions (planes, the window box, contact sets) have Lebesgue measure 0 and do not
  affect q. Every proof below holds for almost every s in a region, which is all that volumes need.
* **Undefined posteriors.** If no positive feasible volume is certified (Σ lo_c = 0), the status is `undefined` and
  no distribution is released. Nothing is normalized silently.

The posterior is that of the stored observation. A renderer error would change the target but cannot make a
certificate wrong: the verifier proves statements about the stored bits, not about the renderer.

## 2. Architecture: untrusted search, trusted check

```
scene container ──► labeller (untrusted)                       ──► certificate (proof tree, *_cert.npz)
                     stage1_cert, Local-first planner,                     │
                     engine_cert (tau-pencil BSP)                          ▼
scene container ──► verifier (trusted, certify/*) ◄────────────────────────┘
                     re-proves every claim from exact geometry and the stored bits
                     ──► per-cell enclosures [lo_c, hi_c], pooled mass P, TV bound ──► label (*_label.npz)
```

* **The labeller.** It is the existing tau-pencil engine with float margins (`tpl/engine_cert.py`), plus the
  Local-first planner (`localfirst/certclip.py`). It only *proposes* a proof tree:
  * **Splits** are named by exact geometry: an axis plane at a dyadic coordinate, a component facet (c, i), or an
    occluder-edge pencil through a receiver (c, i, j, x).
  * **Leaves** carry claims: IN, OUT (with a witness receiver or block), or UNDET.
  * **Local-first** adds root drops and root clips.
  * The certificate contains **no floating-point geometry**, only names and dyadic axis coordinates.
* **The verifier** (`rind3d_posterior.certify`) shares no code with the labeller. It materializes every plane
  itself and proves every claim with exact or error-bounded arithmetic.
  * A claim it cannot prove only moves volume from "certified" to "undetermined". It can never create certified
    volume.
  * The released bounds are the verifier's own.
* **Re-verification.** `python -m rind3d_posterior.certify.label verify` re-runs the verifier from the scene
  container, the window/channel indices, the stored label and the certificate. It then checks the stored q̂
  against its own enclosures (§4.9).

## 3. The proof tree and its true regions

Each stage-1 root box (octree depth 4) is the root of a binary tree. Its true regions are defined by exact
half-spaces:
- axis planes x_k ≤ v (v dyadic, exact);
- facet half-spaces f_i ≤ 0 or f_i ≥ 0;
- pencil half-spaces σ g_x ≤ 0 or σ g_x ≥ 0, with g_x(s) = f_f(x) f_b(s) − f_b(x) f_f(s);
- the Local-first wedge half-spaces (§4.2).

The two children of a split are closed half-spaces of the same exact plane, so the true leaves tile the root up
to measure zero.

Pencil and facet planes have rational coefficients that binary64 cannot represent. The verifier therefore keeps
two binary64 representations of each true region, from the coefficients' float approximation and its rigorous
error bound E on [0, 1]³:
* the **outer polytope**, every named plane shifted outward by E, a superset of the true region;
* the **inner planes**, shifted inward by E, which bound a subset.

Interval facts are proved on the outer polytope, so they hold on the true region.

The **symbolic facts** of a path are exact by definition of the true region:
- after a facet split, the sign of that facet;
- after a pencil split or clip at receiver y, the sign of B_fb(y, s).

They decide the conditions that are tight on the region's boundary. A float plane could never certify those.

## 4. The proof chain

Every item gives the statement, the argument, the implementation and the tests.

### 4.1 Canonical geometry and response predicates
* **Exact input.** Rows are exact rationals. The verifier encloses each coefficient (binary64 value plus radius,
  and a double-double hi + lo with a 10⁻³³ residual). It computes exact component vertices (Fractions), exact
  edges, outward-rounded bounding boxes, and an exact boundedness test.
  Code: `certify/scene.py`. The engine's facet indexing is checked against the canonical facets
  (`check_engine_rows`), because event recipes name facets by index.
* **Segment–polytope criterion (no adjacency, no silhouettes).** Take s outside C with every facet strictly
  signed at s, F = {f_i(s) > 0} and B = {f_j(s) < 0}. Then [s, x] meets C iff:
  - f_i(x) ≤ 0 for every i ∈ F, and
  - B_ij(s, x) = f_i(s) f_j(x) − f_j(s) f_i(x) ≤ 0 for every (i, j) ∈ F × B.

  *Proof.* Write f_k(s + t(x−s)) = (1−t) f_k(s) + t f_k(x). Each facet gives a bound on t ∈ [0, 1]. Eliminating
  t gives exactly these conditions. The roles of s and x are symmetric (the receiver form).
  Code: `certify/check.py`, module docstring. Tests: `tests/test_certify_predicates.py` against an exact rational
  segment oracle.
* **Box form.** B_ij is affine in s for fixed x and in x for fixed s. Its extremes over (polytope R) × (receiver
  box Q) are therefore attained at vertex × corner pairs. With fixed signs this collapses to ranges of
  τ_ij = f_j/f_i: COVER iff max_R τ ≤ min_Q τ (the tau-pencil lemma).
  - The inverse shadow of a receiver is convex in s, so "dark from every vertex" implies dark on the whole region.
  - NOCOVER is proved by one condition violated on every pair: a strict facet separation, a pair row, or an AABB
    separation.
  - Undecided stays undecided (the receiver block is expanded).

### 4.2 Proposition A′ and the Local-first restrictions
* **Proposition A′.** The full proof is in `docs/LOCALFIRST_E123.md` §2. Assume polytope occluders, a direct
  response interface (l active, k inactive, 6-adjacent FREE receivers) and [l, k] free of the solid (H1). Then
  every feasible s has an admissible occluder edge e meeting conv{s, l, k}.
* **Drop.** If no admissible edge meets conv(Q ∪ [l, k]), then F ∩ Q = ∅.
* **Clip (wedge lemma).** If only e = (c, i, j) can meet conv(Q ∪ [l, k]), every feasible s ∈ Q lies on the sliver
  s = c' + t(c' − p'), with c' ∈ e, p' ∈ [l, k] and t ≥ 0. There g_p(s) = −tμ g_p(q), where g_p is the pencil
  through line(e) and the endpoint p and q is the other endpoint. Hence F ∩ Q ⊆ {sign(g_p(q)) · g_p ≤ 0}.
* **Verification.** Code: `certify/propa.py`.
  - H1 is checked by a strict facet separation of both endpoints (double-double values) or an AABB separation.
  - Admissibility comes from strict signs at l and k.
  - The separation of each other edge from conv(Q ∪ {l, k}) uses explicit axes and error-bounded projections of
    the exact points and the enclosed edge endpoints.
  - The orientation σ comes from the certified sign of g_p(q). The applied plane is the outer one, a superset.
* **Effect of failures.** A clip that fails is not applied: the root stays larger, which is sound. A drop that
  fails puts the whole root box into the pool, which is sound but loose.
* **Locality.** Proposition A′ uses the global occluder edge list and an exact free-edge test. It is a
  global-geometry-assisted constraint, not a purely local one (`docs/LOCAL_INVERSE_GEOMETRY.md`, Theorems 3 and 3′).

### 4.3 No feasible source is lost in restriction
- Only proved clips are applied. Each applied plane is a superset of the true half-space.
- Dropped roots are proved, or pooled with their full volume.
- Stage-1 boxes that are not roots are proved:
  - inside the closed window box, by exact comparison; or
  - refuted by the named sentinel receiver over the whole box.

  Otherwise they are pooled.
- The stage-1 terminal boxes are checked to tile the unit cube exactly: their volumes sum to 1 and no box
  contains another.

### 4.4 Subdivision and classification
- **Exact kernel** (`certify/poly.py`):
  - Plane-based polytopes: a vertex is a triple of planes, and its sign against a plane is
    sign(det₄) · sign(det₃), computed exactly with Shewchuk expansions behind a static filter.
  - Simulation of Simplicity (b_id → b_id + ε^id) removes every degeneracy. Combinatorics are therefore exact and
    children tile parents exactly.
  - Tests: 95 kernel tests against exact rational polytope volumes, including planes through vertices,
    coincident planes and slivers two ulps thick.
- **IN** (`tree.verify_root`, `_check_claim`): every receiver item of the path is decided on the true leaf, with
  full block expansion.
  - Every DARK block has a COVERing component.
  - Every LIT block has NOCOVER for all components. This also proves the leaf misses the solid, since a source in
    the solid sees no active receiver.
  - The leaf is outside the window box interior (exact axis tests).
- **OUT:** the named witness refutes the leaf; otherwise any item of the path's list does (search mode).
- **Symbolic facts.**
  - Facet signs and pencil signs come from the path.
  - The derived-sign identity transfers a fact on receiver y to any receiver x of the same pencil:
    **f_i(y) B_ij(x, s) = f_i(x) B_ij(y, s) + B_ij(x, y) f_i(s)**. Its sign is certified when the terms agree
    (`check.derived_sign`).
  - Test: `test_derived_fact_identity`.
- **Clearance shell.** A region touching an occluder edge inside the solid can contain points where a claim is
  genuinely false, within about 10⁻¹⁵ of the solid.
  - The wedge N = R ∩ {f_i ≥ −η} ∩ {f_j ≥ −η} is dropped when every vertex of N is certified to lie within 2⁻²⁶ of
    the body (distance to an exact edge point, or inside). This is valid by convexity of the 2⁻²⁶-neighbourhood,
    which contains no legal source.
  - The remaining two convex pieces are re-proved.
- **Safety net.** A claim that still fails is re-proved on at most 48 axis-bisection pieces. Pieces that still fail
  are UNDET.

### 4.5 Per-cell volume enclosures
- **Leaf volumes.** The verifier splits each IN or UNDET leaf into 128³ cells with exact grid planes, carrying exact
  cell-index bounds.
  - Each piece's volume is a rigorous interval from fan tetrahedra over the enclosed vertices (mid-radius
    arithmetic).
  - Unit-axis boxes use an exact product of dyadic sides, rounded outward.
- **IN leaves:**
  - lo_c is the volume of the leaf's outer polytope ∩ (inner planes of the path), minus an upper bound of its part
    within 2⁻²⁶ of the solid. That bound comes from an outward-dilated polytope per component.
  - hi_c is the volume of the outer polytope.
- **UNDET leaves:** [0, outer volume].
- **Pooled mass:** regions that cannot be split per cell, plus unproved stage-1 boxes and drops, go into one pooled
  upper volume P.
- Per cell, lo_c and hi_c are exact rational sums of the binary64 records, rounded outward (`verify.cell_enclosures`).

### 4.6 Numerical error model
- **Floating-point model.** IEEE-754 binary64, round to nearest even, no fused multiply-add contraction (numba's
  default `fastmath=False`).
  - `ar.self_test()` checks the error-free transformations on adversarial inputs before every verification.
  - It fails if contraction or extended precision is present.
- **Error bounds.**
  - Every expression evaluated in binary64 carries an a priori bound γ_n · Σ|terms| (Higham, Lemma 3.1/3.3) plus
    2⁻¹⁰⁰⁰ for underflow, inflated by (1 + 64u).
  - Comparisons use endpoints widened so that they stay rigorous after the final rounding (`check._wid`).
  - Exact signs use expansions. Ill-conditioned vertex triples use exact determinants.
  - Receiver facet values are double-double, so their error is relative to |f(x)| even under cancellation.
- **Compiled-code identity.** The numba cache directory is keyed by the SHA-256 of the jitted modules. Numba's own
  cache would not notice a changed callee in another module (`certify/__init__.py`).

### 4.7 Normalization
Normalization is done in exact rational arithmetic (`certify/tvb.py`):
- Lt = Σ lo_c, which must be > 0 (this proves Vol(F) > 0);
- the reference estimate is q̃ = lo/Lt;
- the released q̂ is the binary64 rounding of q̃.

### 4.8 The TV bound
Let the true masses be v_c = lo_c + e_c, with 0 ≤ e_c ≤ hi_c − lo_c, plus pooled mass of total at most P.

Step 1. Write E = Σ e_c and w_c = 1 − lo_c/Lt. Then

  |Lt e_c − lo_c E| ≤ e_c (Lt − lo_c) + lo_c (E − e_c),

and therefore

  TV(q̃, q) ≤ Σ_c e_c w_c / (Lt + E).

Step 2. Pooled mass counts like a cell with w = 1, the largest possible weight. The supremum over the box is
attained greedily by adding cells in decreasing w.

Step 3. The released bound is TV(q̂, q) ≤ TV(q̂, q̃) + TV(q̃, q). The first term is computed exactly from the stored
binary64 q̂. The sum is rounded up.

### 4.9 Serialization and reload
- **Format.** The label and certificate are `.npz` files of int64/float64 arrays with no pickles. Binary64 values
  round-trip bit-exactly. `meta` holds the SHA-256 of:
  - the geometry bytes;
  - the observation bits;
  - the certificate file contents.
- **What `verify_label` checks:**
  - the digests;
  - a fresh run of the verifier on the certificate;
  - recomputing TV(q̂ stored, q) from its own enclosures and requiring it ≤ the stored `tv_bound`;
  - Σ q̂ = 1 (to 10⁻⁹).

## 5. Trusted computing base

Trusted:
- the canonical decoder `decode_convex` and the scene container format (geometry bytes and observation bits);
- `rind3d_posterior.certify` (≈ 2,500 lines);
- NumPy, numba/LLVM and CPython;
- IEEE-754 hardware arithmetic, checked by the self test;
- `fractions.Fraction`.

Not trusted:
- the labeller and its float margins;
- the Local-first planner;
- the archived tau-pencil volumes;
- FastLabel;
- every GPU path.

Also outside the TCB: the renderer that produced the observation. The certificate is about the stored bits.

## 6. Problems found and how they are resolved

| issue | cause | resolution |
|---|---|---|
| Stored lower volumes exceed cell capacity (`000049_w03_s0`, cell 1678663: lo = 0x1.0000000000080p-21 > 2⁻²¹; 1,445 single records and 11,304 cells in this rerun) | After interpolating a vertex on a clipping plane, the engine's float clipper projects it onto that plane (`V -= dd·a`) and snaps vertices within 10⁻¹³. Both move vertices off the other planes that define them. Root-box faces drift by 1–2 ulps and the drift propagates into cell pieces. The recorded volume is the exact volume of a vertex set lying outside the cell. Summation and serialization are not the cause. | The labeller's volumes are no longer used. The verifier's kernel has exact combinatorics and rigorous volume intervals; full cells are exact (`experiments/certify/diag_volume.py`). |
| Tau-pencil volumes differ from exact posteriors by up to 2×10⁻⁹ per cell and 4.6×10⁻¹⁰ in total; claimed TV 10⁻¹⁴ against a true 1.5×10⁻¹⁰ (independent review of Astra's exact fixtures) | same float clipper and margins | same |
| Labeller OUT claims that are false on parts of a leaf: a witness block that is not refuting everywhere near an occluder face, and claims false within 10⁻¹⁵ of an occluder edge | float hull tests with 10⁻¹¹ margins; regions touching occluder faces and edges | The verifier re-proves claims. Edge wedges inside the 2⁻²⁶ shell are dropped by domain. Other failures are re-proved piecewise or left UNDET. A physical-oracle audit confirmed every failing region is infeasible. |
| Local-first E2 harness: coarse budget-stopped leaves mapped to a single cell (mass loss on 3 tasks) | research harness `_filter_from_leaves` | Fixed in d1929c6 with a regression test. Production clips are re-proved by `propa`; the E2 box filter is not used. |
| Single-cell-support exit claimed TV 0 without certifying Vol(F) > 0 under the full observation (LF prototype, FastLabel) | outer bound B of a receiver subset used as support | Not used. Positivity Lt > 0 is required (§4.7). |
| Unbounded random polytopes broke a separation shortcut (test generator) | AABB of the vertices does not bound an unbounded body | Exact boundedness check in `certify/scene.py`. |
| Clipping by a coincident plane of opposite orientation returned EMPTY without clearing the slot (verifier kernel, found by the engine cross-check before any release) | kernel bug | Fixed, with regression tests `test_empty_clip_clears_slot_and_cells_tile` and `test_emit_cell_split_counts_each_cell_once`. |

## 7. Re-verifying a released label

```bash
source RIND3D_Posterior/experiments/certify/env.sh        # interpreter, PYTHONPATH
$PY -m rind3d_posterior.certify.label verify --container SCENE.npz --window W --channel S \
    --label KEY_label.npz --cert KEY_cert.npz              # exit 0 and "ok": true when the label is re-proved
```

Labels can also be produced with `... label produce [--local-first] ...`. Everything is CPU-only and deterministic.

## 8. Scope and limitations
- **Polyhedral solids only.** Curved bodies (balls) have no certificate and are excluded from the certified
  population (`docs/CURVED_ASSESSMENT.md`).
- **TV only.** The certificate bounds TV between the released distribution and the exact posterior of the stored
  observation. It says nothing about a Student's error, and nothing about uncertainty over hidden geometry: the
  target is conditional on the full geometry.
- **Tightness depends on the labeller.** A labeller tree whose claims fail broadly leads to UNDET volume. The bound
  stays valid but becomes loose. Measured coverage is in `CERTIFIED_LABELS_RESULTS.md`.
