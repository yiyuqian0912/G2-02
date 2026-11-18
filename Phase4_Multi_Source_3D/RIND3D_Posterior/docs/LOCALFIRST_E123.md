# Local inverse geometry, executed: certified interface regions (E1), their computational value (E2), and the exact local region of voxel physics (E3)

Branches: `yushu/3d-localfirst-research` (E1, E2; this note) and `yushu/3d-voxel-physics` (E3; `docs/VOXEL_PHYSICS_DESIGN.md`).
Data: `RIND3D_100K_v3_work/{e12, e2_profile, e3}`, outside Git. Tasks: the 54 `cmp_v3` tasks of `LOCALFIRST_COMPARISON.md`.
Laptop: RTX 3060 Laptop 6 GB, Ryzen 7 5800H, one thread per process.

**The question.** Does the geometry of locally observed binary-response interfaces constrain the continuous
driver-source position strongly and safely, before expensive global posterior certification?

**Summary of the answers (details in §7–9).**
* **Information: yes.**
  - In continuous physics, certified interface constraints that need only the occluder edge list carry at least 82%
    of the posterior's log-volume information. B ∩ R reaches 97%, and a median 91× less volume than B.
  - In exact fixtures, the region equals F.
  - Choosing constraints from local planes is not better than random. The predeclared E1 rule fails on that clause
    only.
* **Certification: yes, under stated assumptions.**
  - Proposition A′ (polytopes, H1 decided exactly, no H2) contains every feasible source, with no exceptional set.
  - Its predicates are verified against exact rational references, and no violation appeared in 1.6 M samples of F.
* **Computation: conditional.**
  - Box-level restriction cannot help (1.05–1.07×, at the 1.10× oracle).
  - Certified polytope-level root clipping, a 30-line engine change, speeds up the exact phase 1.54× at the median (up
    to 4.6×), with identical certificates.
  - End to end it is a heavy-tail optimisation: 0.95× median, 1.29× for heavy labels, max 57.7 → 32.4 s.
  - FastLabel stays faster.
* **Purely local (voxel physics): yes, as a theorem.**
  - The exact X-only region over all hidden worlds is computable.
  - It is under 1% of the cube in 34 of 37 windows, and locally about as tight as the fixed-world F.
  - The binary-response interface alone does the pruning in every window without observed SOLID.

It splits into three questions that are answered separately:
* **A. Information.** Do the interfaces carry it?
* **B. Certification.** Under which assumptions is it safe to prune with it?
* **C. Computation.** Can a deterministic labeller exploit it?

## 1. The right representation of the inverse information

**What a transition is, in source space.**
* Visibility of a fixed receiver point x from a moving source s changes only where the segment [s, x] grazes the
  solid.
* For polyhedra this is the edge–vertex visual event of the aspect-graph literature, with the receiver as the vertex.
  Gigus, Canny and Seidel, *IEEE TPAMI* 13(6), 1991, name the two polyhedral events "an edge and a vertex project
  onto each other" and "three edges meet".
* In source space each event is the planar wedge {e + t(e − x), t ≥ 0} beyond an occluder edge e: part of the plane
  through x and e.
* A response transition between two adjacent free receivers l and k is a transition of the moving point p along
  [l, k]. Its event set in source space is the thin sliver between the two event planes (through e and l, through e
  and k), beyond e.

**The feasible set.** F is therefore a union of cells of the arrangement of these event wedges. This is the viewpoint
space partition of Plantinga and Dyer (*IJCV* 5, 1990) restricted to point visibility. Every response receiver
contributes one wedge per occluder edge.

**The plane intuition, made precise.**
* "A planar interface puts the source on a plane" is the statement that the slivers of many interface edges
  explained by one occluder edge e intersect in a thin wedge around the plane through e and the interface piece.
* In the pencil of planes around line(e), that wedge is an interval of one angle-like parameter τ_e.
* Two independent pieces explained by different edges give a line, and three give a point.

**What is new here.** The tau-pencil engine already uses exactly this pencil parametrization for its exact constraint
compression (`tpl/engine_p2.py` docstring: every receiver is an orthant in the coordinates τ_e(s)). What is new is to
use the *interface* edges as constraints that need no visibility computation, before the exact phase.

**Formulations considered, and their verdicts:**

| formulation | uses | safe? | verdict |
|---|---|---|---|
| F1. one least-squares plane per component, intersect (Prop. 4 stabbing sets) | X only | only under H1 and H2; H2 fails at the *true* source for 67/118 RANSAC components (audit §3.4), up to 1.5 voxel | rejected for pruning |
| F2. set-valued planes: the digital-plane preimage of each component | X only | needs H2 (one plane per component), which no X-only test certifies; Theorem 3′ is an explicit failure | diagnostic only |
| F3. per-edge event slivers, union over every admissible occluder edge, intersected over edges (Prop. A′) | X + the occluder *edge list* | **yes**: needs H1 per edge, checked exactly; no H2 | **selected for Model A (§2–4)** |
| F4. F3 compressed per occluder edge (one τ-interval per edge and piece) | X + edges | only where every edge of the piece has a single admissible explanation on the region; certifiable per box by the same octree | the bridge to an engine (§5) |
| F5. X-only region over all hidden worlds (Model A) | X only | full measure for v2 (Theorem 3); not computable for v3 | impossibility (open for v3) |
| F6. X-only region over all hidden worlds (Model B, voxel physics) | X only | **yes, exact** (Theorem 10) | **selected for E3 (§6)** |

**Related work, and how far each result transfers:**
* *Crow 1977, shadow volumes.* The forward version of the event-wedge construction.
* *Kriegman and Belhumeur, "What shadows reveal about object structure" (ECCV 1998; JOSA A 18, 2001).* When the
  occluders are *unknown*, shadows determine the scene only up to a generalized bas-relief family. This is the
  continuous analogue of our Level-A obstruction (Theorem 3): without the geometry, local shadows do not pin the
  source.
* *Savarese et al., "3D reconstruction by shadow carving" (IJCV 2007).* Prove that carving away regions
  inconsistent with observed shadows keeps a conservative estimate. This is the same "never discard a consistent
  configuration" invariant as ours, for the dual problem (unknown shape, known light).
* *Laurentini 1994, the visual hull.* The dual of our problem: known viewpoints, unknown shape.
* *Discontinuity meshing (Heckbert 1992; Drettakis and Fiume 1994) and the visibility skeleton (Durand, Drettakis and
  Puech 1997).* Enumerate the same edge–vertex events for area-light shadow computation.
* *Caveat.* All of these concern exact continuous shadow boundaries. Our observation is a sampled binary field, and
  that is precisely why H1 is needed: §2's proposition is about the sampled edge, not about a continuous boundary.

## 2. Proposition A′ (corrected and strengthened from the audit's Proposition A)

**Assumptions.**
* S = C₁ ∪ … ∪ C_m is a union of compact, full-dimensional convex polytopes. This covers every v3 body.
* Closed-solid, closed-segment visibility.
* (l, k) is a direct interface edge of the observation: free receivers at distance h, l response-active,
  k response-inactive.
* H1: the closed segment [l, k] misses S. This is decided exactly (`edge_free_mask`: float with an exact rational
  re-check).

**Proposition A′.** For every s′ ∈ F(G, X) there are a point p ∈ [l, k] and an edge e of some body C with
1. e ∩ [s′, p] ≠ ∅;
2. [s′, p] meets S but not int S;
3. e is a silhouette edge of C seen from p: with g₁, g₂ the outward rows of the two facets adjacent to e,
   g₁(p)·g₂(p) ≤ 0.

The quantifier is over all of F, not over the generated source: every feasible source reproduces the observation, so
l is active and k inactive for each of them. **There is no exceptional set.** The audit's version excluded sources in
facet planes; the facet case below removes that exception.

**Proof.**
1. *A boundary point on the edge.* Fix s′ ∈ F. In(s′) = {y : [s′, y] ∩ S ≠ ∅} is closed, by compactness of S. The set
   {t ∈ [0, 1] : l + t(k − l) ∈ In(s′)} is closed, contains 1 (k is inactive) and not 0 (l is active). Let t* be its
   minimum and p = l + t*(k − l). Then p ∈ In(s′) is a limit of points outside In(s′), so p ∈ ∂In(s′). By H1, p ∉ S.
2. *The segment only grazes.* If [s′, p] contained a point z = s′ + τ(p − s′) of int S, then τ ∈ (0, 1), and
   s′ + τ(p″ − s′) ∈ int S for every p″ near p. So p would be interior to In(s′), a contradiction. Hence [s′, p]
   meets S only in ∂S, and it does meet S because p ∈ In(s′). This is item 2.
3. *Where the contact can lie.* Take c ∈ [s′, p] ∩ S. Then c ≠ s′ (sources are outside S) and c ≠ p. Take a body
   C ∋ c and the minimal face f of C with c ∈ relint f.
   - *f = C.* Then c ∈ int C, which step 2 excludes.
   - *f a facet, s′ ∉ aff f.* The line meets aff f only at c, so s′ and p lie strictly on opposite sides. Segments
     [s′, p″] with p″ near p cross aff f near c, hence inside relint f ⊂ S. So p ∈ int In(s′), a contradiction.
   - *f a facet, s′ ∈ aff f.* Then the whole segment, p included, lies in aff f. The sub-segment [c, p] starts in f
     and ends outside it (p ∉ S), so it meets the relative boundary of f, a union of edges of C. Replace c by such a
     point c′ ∈ [s′, p] on an edge e of f. Since p ∈ aff f, g_f(p) = 0, which gives item 3 for this e.
   - *f an edge e, c ∈ relint e.* Write u = p − c. If g₁(p), g₂(p) > 0, then g_j(c − εu) = −ε g_j(p) < 0 for both
     adjacent facets. Near c, C coincides with the dihedral wedge {g₁ ≤ 0, g₂ ≤ 0}, so c − εu ∈ int C, and that point
     lies on [s′, c]: excluded by step 2. If both are negative, c + εu ∈ int C lies on [c, p]: excluded likewise.
     Hence g₁(p) g₂(p) ≤ 0.
   - *f a vertex v.* If some facet at v has g(p) = 0, take an edge of that facet through v: g = 0 gives item 3. If
     every facet at v were strictly front-facing (g > 0), the same local argument puts v − εu in int C; if every
     facet were back-facing, v + εu is in int C. Both are excluded. So both kinds occur. Around v the facets form a
     cycle (the vertex figure), so some consecutive front/back pair shares an edge e ∋ v with g₁(p) g₂(p) < 0.

   In every case c lies on an edge e satisfying item 3, and c ∈ [s′, p] gives item 1. ∎

**Consequences used by the implementation.**
* **Necessary condition.** Since p ∈ [l, k], e meets the triangle conv{s′, l, k}.
* **Region per edge.** Let E(l, k) be the edges that pass item 3 for some p ∈ [l, k] ("admissible"). Item 3 is
  linear in p, so it suffices to test the endpoints. Define Z(l, k) = ∪_{e ∈ E(l,k)} {s : e ∩ conv{s, l, k} ≠ ∅}.
* **Theorem (certified region).** For every set I of H1-free interface edges,
  F(G, X) ⊆ R_I := ∩_{(l,k) ∈ I} Z(l, k).
* **Box lemma.** For a source box Q, ∪_{s ∈ Q} conv{s, l, k} = conv(Q ∪ [l, k]): the join of two convex sets is
  their convex hull. So Q ∩ Z(l, k) = ∅ iff every e ∈ E(l, k) misses conv(Q ∪ [l, k]), a finite segment–polytope
  problem.
* **Multiple occluders and explanations.** These are handled by the union over E(l, k); no explanation is ever
  selected.
* **Finite segments and endpoints.** e is the finite edge segment, and p = k is allowed. Coplanar and tangent
  configurations are the facet and vertex cases above.

**What the proposition does not claim.**
* It does not claim that the transition is explained by a *single* edge along a component: that is H2, which is not
  needed.
* It uses no visibility computation and no receiver beyond l and k.
* It is implied by the exact constraints of l and k. So R_I can add information over a receiver-subset outer bound
  only through interface edges that are not in the subset.

**Where the world geometry enters.** Two places, and both are the global geometry (this is Level B, not Level A):
* the occluder edge list, with each edge's adjacent facet planes;
* the exact H1 test.

## 3. Implementation (`rind3d_posterior/localfirst/certregion.py`) and its validation

**Occluder edges.** All facet pairs sharing at least two enumerated vertices (the `rpl.topology` rule). True edges are
never missed. Spurious pairs are boundary segments of the body and only add explanations.

**Admissibility.** An edge is dropped for (l, k) only when both adjacent facet values have the same strict sign, by
more than a 10⁻⁹ margin, at both l and k.

**Box test.** `seg_misses_hull` decides whether e misses conv(Q ∪ [l, k]).
* *Prefilter.* An AABB test, then a wedge prefilter: the plane through line(e) and s must weakly separate l and k.
  This uses two linear functions on the 8 corners and is sound because the sliver lies inside that double wedge.
* *Separating-axis test.* The candidate axes are the hull facet normals (box axes; planes through l or k and a box
  edge; planes through [l, k] and a corner) and e's direction crossed with all 20 hull edge directions.
* *Margin.* A gap must exceed a 10⁻¹⁰ margin scaled by |u|₁. Any direction with a verified gap is a valid
  separation, so the axes need not be computed exactly.
* *Point test.* `seg_misses_triangle` is the complete 4-axis SAT for a segment against a triangle.

**Octree** (`region_octree`, numba).
* A box is dropped if it lies inside the closed window box, or if some interface edge has no admissible occluder edge
  meeting conv(Q ∪ [l, k]).
* Candidate lists shrink monotonically down the tree.
* If the leaf capacity is reached, refinement stops and the coarser box is kept.

**Tests** (`RIND3D_Posterior/tests/test_certregion.py`, 12 passing).
* *Exact references.* The box and point predicates are checked against an exact rational reference: hull facets
  enumerated in Fractions, segment clipped exactly. 4 × 400 random and near-touching boxes and 4 × 600 triangles, with
  no unsound answer. At most 2% of truly disjoint cases are kept because of the margin.
* *Fixtures.* Proposition A′ is checked on exact fixtures:
  - a rotated box;
  - a 7° knife-edge wedge, where H1 fails on some edges and they are dropped;
  - two near-aligned boxes, which give a narrow F;
  - a source 10⁻⁹ from a facet plane's extension.
* *How F is sampled.* Uniformly, at scales 10⁻⁵–10⁻² around the true source, and by bisection to F's boundary
  (18 halvings) between feasible and infeasible samples.
* *What is asserted.* Every feasible sample lies in R_I for all free edges and for the 64-edge subset, and in a kept
  box of the octree cover.

## 4. E1: information content and safety of the certified interface region (54 tasks)

**Setup.**
* Script: `experiments/localfirst/e12.py`. Full per-task tables: `LOCALFIRST_E123_TABLES.md`.
* Per task, R_I is built from |I| ∈ {8, 32, 128, 512} H1-free interface edges, chosen by:
  - *independent*: components ordered greedily to maximise the smallest singular value of their normals, edges spread
    within each component;
  - *random*.
* Plus all eligible edges, capped at 2,048 by a uniform subsample. The median eligible count is in the tables.
* Volumes:
  - Vol(R ∩ B)/Vol(B): 20,000 uniform samples of B's exact pieces.
  - Vol(R): a depth-10 cover over a stratified subset of the depth-7 cover cells, with Monte Carlo inside it.
* Spatial structure: the depth-7 cover (cells and 26-connected components).

**Safety, measured.**
* 1,620,000 uniform samples of F's exact pieces were tested over all configurations. None fell outside R_I, none
  outside the octree cover, and the true source was always inside. 0 alarms.
* This complements the proof (§2) and the exact fixtures (§3). Inclusion of F's whole volume is checked by samples
  over the pieces, not only by the support cells.

| selection | \|I\| | Vol(R∩B)/Vol(B) p10 / p50 / p90 | Vol(B)/Vol(R∩B) p50 | R cells (depth 7) p50, against B's 136 | R components p50 | build s p50 / max |
|---|---|---|---|---|---|---|
| independent | 8 | 0.93 / 1 / 1 | 1 | 7,191 | 32 | 0.29 / 2.8 |
| independent | 32 | 0.41 / 1 / 1 | 1 | 5,894 | 23 | 0.84 / 6.8 |
| independent | 128 | 0.11 / 0.34 / 1 | 2.9 | 5,669 | 24 | 2.6 / 26 |
| independent | 512 | 0.0062 / **0.090** / 1 | 11 | 5,682 | 23 | 8.4 / 90 |
| random | 8 | 0.68 / 1 / 1 | 1 | 9,433 | 42 | 0.33 / 3.3 |
| random | 32 | 0.32 / 0.84 / 1 | 1.2 | 7,142 | 26 | 0.93 / 7.1 |
| random | 128 | 0.12 / 0.50 / 1 | 2.0 | 6,186 | 26 | 2.7 / 25 |
| random | 512 | 0.0049 / **0.086** / 1 | 12 | 5,941 | 24 | 8.5 / 90 |
| all eligible (≤ 2048) | — | 0.00027 / **0.011** / 1 | 92 | 5,726 | 22 | 28 / 346 |

**The declared rule** (`LOCALFIRST_AUDIT.md` §5.2: median ≤ 0.1, *and* independent < random at the same |I|) is **not
met at any |I|**:
* the volume clause holds from |I| = 512, for both rules;
* the independence clause holds at 128 (0.34 against 0.50) but not at 512 (0.090 against 0.086).

Paired comparisons:

| \|I\| | independent smaller | random smaller | ties |
|---|---|---|---|
| 32 | 10 | 28 | 16 |
| 128 | 27 | 19 | 8 |
| 512 | 23 | 20 | 11 |

Local-plane "independence" therefore gives no robust advantage over random edges. What carries the information is the
*number* of certified interface constraints, not their geometric selection.

**Information, in log-volume.** I(Z) = ln(Vol(domain)/Vol(Z)), over the 45 tasks with Vol(F) > 0:

| region | I p50 (nats) | share of I(F), p10 / p50 / p90 |
|---|---|---|
| F, the exact posterior support | 18.9 | 1 |
| B (exact visibility of 384 receivers) | 15.0 | 0.63 / 0.75 / 1.00 |
| R alone, all edges: interface edges + occluder edge list, no visibility (95% lower bound) | 17.2 | 0.58 / 0.82 / 0.99 |
| B ∩ R, all edges | 18.4 | 0.82 / **0.97** / 1.00 |

B ∩ R gains 4.0 nats over B at the median, up to 8.2 at p90.

**Spatial structure.**
* R alone is thin but spread: a median of 5,700 depth-7 cells in about 22 components. Its cover is 6×10⁻⁵ of the
  domain.
* B is compact (136 cells).
* B ∩ R occupies B's cells but only about 1% of B's volume. This is why a cell- or box-level engine cannot see the
  reduction (§5).

**Where it does not help.** In at least 10% of tasks R ∩ B = B (p90 = 1). These are mainly L64 windows of the
`multiscale_convex`, `occluder_clusters`, `thin_slabs`, `convex_polyhedra` and `prisms` families, where B is already
tight or few interface edges exist. Non-free (H1-failing) edges are a median ≤ 5% per family.

**Small exact configurations** (`experiments/localfirst/fixtures_geometry.py`; principal extents of the accepted
proposals within 0.05 of the source):

| fixture | interface | rank | F extents | R extents | R = F on all proposals |
|---|---|---|---|---|---|
| rotated box | 729 edges | 1 | 2.4e-6 / 7.6e-5 / 1.1e-4 | identical | yes |
| knife-edge wedge (7°) | 148 | 1 | 1.3e-5 / 5.1e-4 / 1.2e-3 | identical | yes |
| two near-aligned boxes | 983 | 2 | 4.8e-7 / 1.2e-6 / 1.1e-5 | identical | yes |
| source 1e-9 from a facet plane | 576 | 1 | 1.9e-4 / 1.6e-3 / 2.2e-3 | 2.0e-4 / 1.8e-3 / 2.3e-3 | R 3% larger |

One plane gives one thin direction; two independent interfaces give a thin tube. These are the plane and line of the
original intuition, certified.

### 4.1 E2 box-level restriction (for completeness)

The exact phase restricted to B ∩ R with a depth-10 box filter gave 1.07× (128 edges) and 1.05× (all edges) at the
median over 45 tasks. That is at the box oracle (1.10× on the same tasks), and below 1 once the refinement cost is
included (0.14× and 0.06×). As predicted (§5.1), the gain must come from polytope-level use.

## 5. E2: computational value

### 5.1 Operation-level profile of the existing exact phase (`experiments/localfirst/e2_profile.py`, 45 exact-phase tasks)

The exact phase (tau-pencil phase 2 on B, outer-bound filter) was instrumented with the engine's own per-root and
per-depth counters, which the production wrapper discards:

| quantity | p50 | p90 | max |
|---|---|---|---|
| stage-1 roots actually evaluated (the B filter kills the rest) | 16% (37 roots) | 65% | 94% |
| item evaluations at BSP depth 0 (the full receiver list at each root node) | 12% | 53% | 100% |
| evaluations created by block expansion (8 × expansions / evaluations) | 42% | 59% | 84% |
| evaluations per BSP node | 11,600 | 131,000 | 271,000 |
| coarse roots, depth 3 instead of 4: exact-phase speed-up | 1.00× | 1.63× | 2.11× |
| coarse roots, depth 2: speed-up | 0.73× | 1.63× | 3.17× |

Label masses agree to 3×10⁻⁹ in every variant, with no UNDET.

**Readings.**
* *The cost is not the per-root re-scan.* Coarser roots do not help.
* *The cost is block expansion.* It happens at the nodes where a face or window split has fixed the occluder aspects.
* *Why blocks expand.* A pure receiver block is decided at its corner receivers only if its τ_e-range does not straddle
  the node's τ_e-range for any silhouette edge e. Near the response interface it does straddle, so the block is
  expanded down to single receivers. This is repeated in every node that contains a piece of F.

**Why box-level restriction cannot help.** Even a perfect box-level region restriction (the piece oracle: F's own pieces
as the outer-bound filter, median 1.10×) leaves this unchanged. A bounding box does not narrow a node's τ-ranges.

### 5.2 Certified root clipping (`tpl/engine_p2xc.py`, `experiments/localfirst/e2_clip.py`)

**The change.** The smallest sound modification that acts on τ-ranges is to clip each root polytope with certified
half-spaces before the BSP starts.

**Source of the clip planes.** Where an interface edge has a *single* admissible explanation e on a root box
(`box_explanations`), Proposition A′ confines F ∩ box to the sliver {s : e meets conv{s, l, k}}. That sliver lies in
one convex wedge through line(e) (`wedge_rows`, with the identity g_l(s) = −tμG, g_k(s) = t(1−μ)G checked on 10⁵ random
instances). In the engine's pencil coordinates the wedge is exactly τ_e ∈ [τ_e(l), τ_e(k)]: the same bound the engine
would otherwise obtain only after expanding the blocks that contain l and k.

**Safety rules.**
* For each explaining edge only the two extreme planes of the pencil are kept; any subset of valid planes is valid.
* Roots where some interface edge has no explanation carry no feasible source and are dropped.
* On capacity overflow the clips are abandoned and the root restarts as a plain box.
* Labels must match the unclipped exact phase in mass (alarm at a relative 10⁻⁶) and tv_bound.

### 5.3 Results of certified root clipping (45 exact-phase tasks; 9 tasks exit with single-cell support)

**Setup.**
* Data: `RIND3D_100K_v3_work/e2_clip_v2`.
* Interface edges: 256 free edges (independent rule).
* Explanations come from the shared octree. Clip rows are built in numba and are identical to the Python reference,
  per `test_root_clips_numba_equals_python`.
* Each variant is timed back to back with the unclipped baseline inside one process.

| measure | p10 | p50 | p90 | max |
|---|---|---|---|---|
| exact-phase speed-up, depth-4 roots + clips | 1.10× | **1.54×** | 2.61× | 4.61× |
| including construction (H1 test, occluder edges, explanations, rows) | 0.26× | 0.95× | 1.71× | 3.09× |
| construction time | 0.18 s | 0.51 s | 0.91 s | 1.28 s |
| BSP nodes, ratio to baseline | 0.34 | 0.50 | 0.79 | 0.97 |
| item evaluations / component evaluations / block expansions, ratios | 0.41 / 0.41 / 0.42 | 0.73 / 0.73 / 0.70 | 0.95 / 0.94 / 0.99 | 0.99 / 0.98 / 1.08 |
| stage-1 roots proven empty by the interface constraints | 3.6% | 12% | 24% | 37% |
| (root, interface edge) pairs with a unique explanation | 33% | 46% | 74% | 95% |

**Certification.**
* Largest relative mass difference to the unclipped exact phase: 1.1×10⁻⁹.
* tv_bound identical (maximum 7.6×10⁻⁸ in both).
* No UNDET record, no overflow, no alarm.
* Soundness rests on Proposition A′ and the tested predicates. It is not inferred from the agreement in mass.

**By cost class**, including construction:

| baseline exact phase | tasks | speed-up p50 | summed net saving |
|---|---|---|---|
| < 1 s | 22 | 0.39× | −6.5 s |
| 1–5 s | 10 | 1.06× | +4.2 s |
| ≥ 5 s | 13 | 1.29× | +50.4 s |

**Summed exact-phase time:** 218.6 s → 147.8 s clipped, or 170.5 s including construction (−22%).

**Whole pathway.** The lf pathway's traces were used with the exact phase scaled by the measured ratio, plus construction:

| pathway | wall time p50 | p95 | max |
|---|---|---|---|
| lf | 2.26 s | 14.9 s | 57.7 s |
| lf + certified clips | 2.61 s | 13.2 s | 32.4 s |
| FastLabel v3.4 (same tasks) | 1.76 s | 2.75 s | 4.1 s |

**Cell-level roots.** Using B's cells as roots, on the 21 tasks where B has ≤ 300 cells: clipped p50 1.35×, unclipped
0.83×. With more cells this was measured at 0.1×, because every root repeats the root-level work.

**Readings.**
* *The box-level oracle was not an upper bound for this.* The 1.10× oracle of the audit bounds restrictions to a set of
  boxes. Certified clipping shrinks the engine's τ-ranges, which no box restriction can do. It halves the BSP nodes and
  removes about 30% of the expansions.
* *End to end it is a tail optimisation, not a median one.* It pays for itself on heavy labels, where the maximum drops
  from 57.7 s to 32.4 s, and loses on light labels through its fixed construction cost.
* *Against FastLabel's flat cost.* The deterministic certificate still has the heavier tail.
* *The decision rule.* The exact-phase speed-up (1.54×) is well above half of the box-oracle savings (1.05×).
  Including construction, the median is 0.95× and the absolute saving is 50 s over the 13 heavy tasks and −6.5 s over
  the 22 light ones.

## 6. E3: the purely local source region in voxel physics

**The setting.**
* Model B is defined in `docs/VOXEL_PHYSICS_DESIGN.md` on `yushu/3d-voxel-physics`.
* *Inputs:* only the Student-visible observation (the window's SOLID occupancy and binary responses, its origin and
  side), the declared physics and the uniform source prior.
* *Never used:* hidden occupancy, the true source, posterior labels, generator internals. The world is used only to
  *produce* the observation and, afterwards, as a check.

**Theorem 10 (exact local region).** Over the class 𝒱 of all complete voxel worlds that agree with the observed window,
the union of the feasible sets equals the set L_B(X) of candidates q (outside the window box, legal with respect to the
observed SOLID) such that:
* (a) no active receiver's segment from q touches an observed SOLID cell;
* (b) every inactive receiver not blocked inside the window has, on its segment from q, a cell outside the window that
  touches no active segment from q and lies farther than ρ from q.

The proof is constructive. "⊆": blockers of unexplained inactive receivers must lie outside W and off every active
segment. "⊇": the maximal world (observed SOLID plus every admissible outside cell) reproduces X from q.

**The three quantifiers, kept apart:**
1. *Fixed known world:* F_B(V, X) is exact; it is what a labeller needs.
2. *Every admissible world consistent with X:* L_B(X) is exact and X-only.
3. *A generator class 𝒢:* L_𝒢 ⊆ L_B is safe only under the assumption 𝒢; it was not computed.

**Verification.**
* *Exact rational, both directions* (32³ grids): q ∈ L_B iff the maximal world reproduces X from q, and the true source
  is in L_B.
* *Random complete hidden worlds* (16³, three occupancy densities, random sources): the source always lies in the
  region computed from the observation it produces. This tests the universal quantifier itself, not one fixed world.
* *GPU pruning is sound:* GPU-pruned implies exactly pruned.

**Measurements on real windows** (37 L64 windows with a direct interface, 19 scenes including voxelized spheres, the
knife edge and v0.1 sub-voxel slabs; `experiments/voxel/e3_local.py`):

| measure | value |
|---|---|
| true source kept / fixed-world feasible sources kept | 37/37 / 255/255; 0 alarms |
| windows in which none of 300 uniform candidates survives (L_B < 1% of the cube at 95%) | 34/37 |
| windows without any observed SOLID: share of the cube pruned by the interface condition (b) alone | 100% (p10 = p50 = p90), 21 windows |
| windows with observed SOLID: share pruned by (a) / (b), p50 | 79% / 21% |
| extent of L_B relative to the fixed-world F_B around the true source, per direction: p10 / p50 / p90 | 1.0 / 1.0 / 27; 67% of directions ≤ 2×, 84% ≤ 10× |
| weak cases | 2 windows with < 100 active receivers (kept 23% and 99%) |
| cost | 2.4 s setup per window, 0.03–0.21 s per candidate (GPU) |

**Readings.**
* *The trivial exclusion is not the result.* Observed-SOLID occlusion, condition (a), does nothing in 21 windows, and
  the response interface prunes the cube there.
* *What (b) encodes.* An inactive receiver seen from q *through* active receivers cannot be blocked anywhere outside
  the window. So (b) is the cone law at voxel resolution: the interface must be ruled by rays from q. This is the
  original local-first intuition, as a theorem with no H1/H2 hypotheses.
* *How tight it is.* Locally it is about as tight as the full-world posterior support along most directions.

## 7. Answers

**A. Information: yes, strongly.**
* *Small exact configurations* (`experiments/localfirst/fixtures_geometry.py`).
  - For a rotated box, the 7° knife edge and two near-aligned boxes, the certified interface region accepts exactly
    the same proposals as F (identical sets and principal extents). For a source 10⁻⁹ from a facet plane it is 3%
    larger.
  - Its dimension follows the plane/line/point intuition. One shadow plane gives one thin direction (extents
    2×10⁻⁶ / 8×10⁻⁵ / 1×10⁻⁴). Two independent interfaces give a tube (5×10⁻⁷ / 1×10⁻⁶ / 1×10⁻⁵).
* *The 54 tasks* (§4), in log-volume information relative to the legal domain:
  - R alone, from interface edges plus the occluder edge list with no visibility computation, carries at least 82%
    of the posterior's information at the median (a 95% lower bound). That is more than the receiver-subset outer
    bound B (75%), which needs exact global visibility.
  - B ∩ R carries 97%.
  - Choosing the edges from local planes is no better than random. The information grows with the number of certified
    constraints.
* *Voxel physics, purely local* (E3, §6): the X-only region is under 1% of the cube in 34 of 37 windows. Locally it is
  about as thin as the fixed-world F.

**B. Certification.**
* *Continuous physics (Proposition A′).*
  - Assumptions: polytope occluders (full-dimensional, convex, compact); closed-segment visibility; H1 decided exactly
    per interface edge.
  - Not assumed: H2, a single explanation, anything about the true source.
  - The region contains every feasible source, with no exceptional set.
  - The box predicates are sound by construction (verified gaps) and verified against exact rational references.
* *When the assumptions fail.*
  - A non-free edge (H1 fails) or an undecidable test contributes nothing.
  - A box whose test is undecided is kept.
  - On curved occluders the polytope edge list is the wrong object, and the region is not certified there. The cone
    version (∂b meets the triangle) is required and was not implemented.
* *Voxel physics (Theorem 10).* No geometric assumption beyond the declared voxel physics. The region is exact over
  all hidden worlds.

**C. Computation.**
* *Box-level restriction* of the existing exact phase cannot exploit the reduction. Even F's own pieces as the filter
  give 1.10×, because the cost is block expansion driven by τ-ranges (§5.1).
* *Polytope-level certified root clipping,* a 30-line engine change, does exploit it: the exact phase is 1.54× faster
  at the median (up to 4.6×), with half the BSP nodes and identical certificates.
* *Including construction* (median 0.5 s) it is neutral at the median (0.95×) and positive on heavy labels: 1.29×
  for tasks above 5 s, the maximum 57.7 s → 32.4 s, and −22% summed exact-phase time.
* *FastLabel cannot use R directly.* Its certificate needs the per-cell volumes of its outer bound, and a union of
  slivers has no computable volume in its pipeline. Its median label (1.6–1.8 s) is also already cheaper than
  building R with many edges.
* *Dominant remaining cost.* Certifying all receivers on the pieces that contain F; certified clips only shrink the
  τ-ranges.

**D. Locality.**
* *E1/E2 (continuous) are not purely local.*
  - The constraints come from the observed interface edges.
  - Their certification needs the global occluder edge list, with facet planes, and the exact free-edge test against
    the world.
  - No visibility computation is needed, and nothing about receivers beyond l and k.
* *The X-only version in continuous physics remains impossible in general.* Theorem 3 gives full measure for v2;
  Theorem 3′ gives positive-probability failure for v3.
* *E3 (voxel physics) is purely local.* It uses only the observation, the window placement and the declared physics.

**E. Generality.**
* *Multiple occluders and multiple explanations* are handled by the union over admissible edges. This is exercised
  by the two-box fixture and every task.
* *Degenerate, coplanar, tangent and endpoint cases* are covered by the proof, and the coplanar fixture tests one.
* *Curved bodies:*
  - not covered by the polytope edge list in Model A (§8);
  - covered by Theorem 10 in Model B, where voxelized spheres behave like polytopes.

## 8. Outcome classification, under the declared rules

* **E1 (information).** The rule from `LOCALFIRST_AUDIT.md` §5.2: median Vol(R ∩ B)/Vol(B) ≤ 0.1, with independent <
  random at equal |I|.
  - **Not met as declared.** The volume clause holds from |I| = 512 for both rules (0.090 and 0.086), and with all
    edges (0.011). The independence clause holds only at |I| = 128, where the volume clause fails.
  - In words: the interface constraints carry strong information, but the hypothesis that local-plane selection
    matters is not supported.
* **E2 (computation).** The rule: a certified speed-up of at least half the pathway's oracle savings.
  - *Exact phase:* 1.54× against the box oracle's 1.10×, which also exceeds the oracle itself.
  - *End to end including construction:* 0.95× median; +50 s over the 13 heavy labels, −6.5 s over the 22 light ones.
* **Classification: conditional result.**
  - Information-positive in volume and in log-volume information, with limited global assistance (the occluder edge
    list). The declared rule fails only on its independence clause.
  - The plane-selection idea itself is not supported.
  - Computation-positive for heavy labels, through polytope-level clipping.
  - Computation-neutral at the median end to end, and negative for light labels.
  - The existing box-filter route is computation-negative.
  - In voxel physics: information-positive and purely local, but without a certified posterior labeller (§9).

## 9. Recommendation for RIND3D-100K production labelling

1. **Keep FastLabel v3.4** as the production labeller: a flat cost of 1.6–2.8 s and an audited certificate. Nothing
   here beats it end to end.
2. **Keep the deterministic "B, then exact on B" pathway** as fallback and audit reference. For it, the certified root
   clipping of §5.2 is a candidate optimisation of the heavy tail, sound by Proposition A′.
   - Before any production use: an independent review of `engine_p2xc`, and a run on a larger task sample.
   - Deploy only with a declared cost rule (for example, clip only when the predicted exact phase is above a few
     seconds), fixed before measuring it.
   - It is not merged.
3. **Do not use any X-only or H2-based local constraint for pruning in continuous physics** (Theorems 3 and 3′;
   §3.4 of the audit).
4. **Voxel physics: research only.** Theorem 10 gives an exact, purely local source region that is strong in practice.
   It is a natural certified target or consistency check for Student studies. Posterior labels in that physics have no
   certified labeller at acceptable cost (`docs/VOXEL_PHYSICS_DESIGN.md` §7).

## 10. Limitations and open problems

* **Curved occluders in continuous physics.** Proposition A′ is for polytopes. The ball version (∂b meets
  conv{s, l, k}, from Corollary 1b) is sound and straightforward, but it was not implemented or measured.
* **Construction cost.** It dominates light labels. The explanation step is 0.3 s at the median.
* **Clipping is partial.** It uses only unique explanations (46% of pairs at the median). Two natural extensions remain
  unexplored, each sound but untested:
  - split roots until explanations become unique;
  - feed the τ-intervals to the engine's per-node bounds directly.
* **Information is not saturated.** R alone has a 95% lower bound of 82% of I(F); its true information is higher,
  and the fixtures suggest R ≈ F in simple configurations. Resolving it needs an exact volume of the sliver
  arrangement, which is not implemented.
* **FastLabel and R.** A FastLabel variant with R would need computable per-cell volumes of B ∩ R, for example by
  representing R's unique-explanation wedges as polytopes, as the clipping does.
* **Open theory.** Whether, under the v3 generator support (inradius ≥ h, bounded dihedral angles), an X-only region
  strictly smaller than the domain exists in continuous physics. Theorem 3′ shows it cannot be built from H2-type
  constraints.

## 11. Provenance

* **Code.**
  - `yushu/3d-localfirst-research`: `localfirst/certregion.py`, `tpl/engine_p2xc.py`, `tests/test_certregion.py`
    (18 tests), and `experiments/localfirst/{e12, e12_report, e1_information, e2_profile, e2_clip, e2_clip_cost,
    fixtures_geometry}.py`.
  - `yushu/3d-voxel-physics`: `rind3d_voxel/*`, `test_voxel_physics.py` (41 tests), and
    `experiments/voxel/{voxel_ab, e3_local, label_feasibility}.py`.
* **Inputs.** `RIND3D_100K_v3_work/cmp_v3/tasks.json`: the 54 tasks, from the v3 canary (contract
  `rind3d-100k-contract-v3`, seed 20261118). Reference labels: `cmp_v3/labels/*_tp.npz`.
* **Seeds.** Selections use `default_rng([20261118, crc32(key), crc32(mode), n])`; the pipeline uses its own seeds.
* **Hardware.** RTX 3060 Laptop (6 GB, driver 620.70.06), Ryzen 7 5800H; one thread per process. 2–4 CPU workers in
  parallel; ratios are timed back to back inside each process.
* **Harness bug, found and fixed.** During E2, a harness bug (a coarse cover leaf mapped to one cell) produced three
  restricted exact phases with too little mass (0.17, 0.31, 0.84). The certified region itself was never at fault:
  every F sample was inside. The filter builder was fixed (regression test `test_filter_from_leaves_covers_coarse_leaves`),
  and the three tasks were rerun. The faulty records are kept in `e12/e12_prefix.jsonl`.
