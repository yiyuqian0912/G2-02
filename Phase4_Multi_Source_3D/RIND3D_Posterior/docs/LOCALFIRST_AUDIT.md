# Audit of the local-first pathway: receiver selection versus source-region reduction

**Scope.** This audit covers the 54 contract-v3 tasks of `LOCALFIRST_COMPARISON.md`.
* Audit script: `experiments/localfirst/audit.py` (`run`, `summary`, `table`).
* Data: `RIND3D_100K_v3_work/lf_audit/`, outside Git.
* Rerun conditions: every stage was recomputed deterministically with the seeds the pipeline uses. Outer-bound
  volumes agree with the comparison traces to a relative 4×10⁻¹⁶.
* Stage times: taken from the clean comparison traces.
* Oracle timings: measured back to back inside one audit process. Four audit workers ran next to other jobs, so use
  only the ratios.

## 1. Two different hypotheses

**(I) Receiver selection.** Local interface geometry chooses which receivers' *exact, global* constraints are
evaluated first. Every choice is safe: dropping receivers only enlarges the outer bound B = F(X′). The candidate region
is always computed from the world geometry.

**(II) Source-region reduction by local inference.** This is the original hypothesis.
* Each local response interface yields a geometric constraint on the driver source. An example is Proposition 4's
  stabbing set U(C): the source lies on a plane that crosses every edge of component C.
* Intersecting several *independent* constraints, from components with different normals (Proposition 5: rank and
  conditioning), gives a region R_loc.
* Global verification then runs only inside R_loc.
* Soundness requires F ⊆ R_loc. The value lies in R_loc being much smaller than what the global search would
  otherwise examine.

The two are logically independent. (I) can fail while (II) holds, and the reverse.

## 2. What the implementation computes

`label_window_localfirst` in `src/rind3d_posterior/localfirst/pipeline.py`:

| step | inputs | output | role of local geometry |
|---|---|---|---|
| local | X only | interface edges, RANSAC planar components, singular values | description only |
| stage 1 (`T.stage1`) | world geometry, sentinel receivers | root boxes at depth 4 (16³): median 239 roots, 5.8% of the cube | none |
| outer (`T.stage2` on X′) | world geometry, 384 receivers | B = F(X′) | **picks X′** (`select_receivers`, line 54) |
| exact (`T.stage2`, `bf = build_filter2(B)`) | world geometry, all receiver items (median 25,600) | the label, deterministic tv_bound | none |

* `local_band_cells`, the only construction of a local *source region*, is never called by the pipeline.
  `compare.py report` uses it only as a diagnostic. (It runs `plane_components` with seed 0 rather than the
  pipeline's seed; the audit uses the pipeline's seed.)
* `conditioning` is recorded and not used.
* No stage intersects source constraints derived from the interfaces.

**Conclusion.** The present implementation tests (I) only. It does not test (II) at all. The negative ablation in
`LOCALFIRST_COMPARISON.md` is a statement about receiver selection. It is not evidence against local inverse geometry.

## 3. Measurements on the 54 tasks

### 3.1 The two strategies select different receivers

| | local selection | random (FastLabel v3.4) |
|---|---|---|
| receivers, p50 (range) | 383 (133–384) | 384 (133–384) |
| active / inactive, p50 | 192 / 181 | 191 / 178 |
| interface receivers, p50 | 256 | 256 |
| on a planar component (≥ 24 edges), p50 / p10 | 248 / 178 | 240 / 70 |
| components touched, p50 | 2 | 2 |

* **Overlap is negligible.** The Jaccard index of the two receiver sets has median 0.006, p90 0.085 and maximum 0.31.
  The median intersection is 4.5 receivers.
* **The selection is not vacuous, but the class mix is the same.** Random interface receivers already land on the
  components because the components hold most interface edges. Local selection differs mainly in the tail (p10: 178
  against 70 receivers on components).

### 3.2 Outer-bound size, constraints and engine work

**Correction to `LOCALFIRST_COMPARISON.md`.** That document said "the same B". This is true only per 128³ cell. In
volume the two outer bounds differ substantially.

| quantity (local / random) | p10 | p50 | p90 | range |
|---|---|---|---|---|
| B volume ratio | 0.025 | **0.42** | 2.6 | 0.0017–33 |
| B cell-count ratio | 0.67 | **1.00** | 1.01 | 0.25–1.17 |

* **Local selection gives a tighter B in volume.** Its B is at least 2× smaller in 29 tasks and at least 2× larger in 8.
* **That has no effect on the exact phase.** The Spearman correlation between the B-volume ratio and the exact-phase
  time ratio is 0.01 (p = 0.93). On the 5 tasks where local B is 10× smaller, the exact phase takes the same time.

**B against the exact feasible set F** (local selection):
* *Volume.* Vol(B)/Vol(F) has median 109, p90 8.5×10⁴ and maximum 2.2×10⁶.
* *Cells.* The cell-count ratio has median 1.02, p90 1.83 and maximum 7.
* B is a thick hull of F, but it occupies almost exactly F's cells.

**Constraints and engine counters**, medians, local selection:

| stage | receiver items | BSP nodes | component evaluations | entry evaluations | ambiguous (exact) evaluations | UNDET / budget / overflow |
|---|---|---|---|---|---|---|
| outer | 383 | 2,030 | 5.4×10⁵ | 4.0×10⁵ | 0 | 0 / 0 / 0 |
| exact on B (45 tasks) | 25,600 (p90 157,000) | 472 | 8.2×10⁶ (max 4.8×10⁸) | 5.9×10⁶ | 6.2×10⁴ | 0 / 0 / 0 |

The exact phase performs about 15× the outer stage's component evaluations (median), on 73× more receiver items.

**Time per stage**, share of the summed wall time over the 54 tasks:

| stage | lf | rand |
|---|---|---|
| local | 5.6% | 5.5% |
| stage 1 | 4.5% | 4.6% |
| outer | 7.7% | 4.6% |
| exact on B | 76% | 79% |

The remaining ~6% is input and setup.

**Fallback and convergence effects.**
* *No fallback or budget events.* No stage of either strategy produced an UNDET record, a node-budget stop or a buffer
  overflow. The largest tv_bound is 7.6×10⁻⁸.
* *Single-cell support.* Local selection reached this early exit (B is one cell, so TV = 0 and the exact phase is
  skipped) in 9 tasks; random selection did in 6. The 3 extra early exits saved 0.2 s, 0.2 s and −0.5 s of wall time:
  the local stage's own cost cancels the saving.

### 3.3 Upper bounds on any region reduction (diagnostic exact runs; never used as labels)

These restrict the exact phase to sub-regions of B. They bound what *any* sound region reduction could save in this
deterministic pathway, whether local or not.

| restriction of the exact phase | runtime speed-up p50 / p90 / max | node ratio p50 | label mass kept |
|---|---|---|---|
| to the 128³ cells of F (cell-level oracle) | 1.07 / 1.84 / 2.82 | 1.00 | 1 |
| to F's own pieces (piece-level oracle: the answer as the filter) | **1.10** / 2.10 / 2.80 | 1.00 | 1 |
| to the heuristic local band | 1.04 / 1.84 / 2.73 | 1.00 | 1 |

**Whole-pipeline bound.** Assume a perfect region is available at zero cost. It replaces the local step, stage 1 and
the outer stage, and the exact phase runs at the piece-level oracle. The speed-up over the lf pipeline would then be
1.55× at the median, 2.9× at p90 and 4.5× at most.

Readings:
* **The deterministic certificate is receiver-bound, not region-bound.** Once B is known, its cost is evaluating all
  receiver items on the BSP pieces that contain F. Even with F itself as the filter, the same nodes are evaluated.
* **The heuristic band contains B entirely** in all 45 exact-phase tasks: inside-band share of Vol(B) = 1.000. As a
  region it is weaker than the receiver-subset outer bound.
* **Consequence.** In this pathway no region reduction, local or otherwise, can give more than about 1.1× on the exact
  phase or about 1.55× end to end at the median. The receiver ablation's null result is therefore expected whatever the
  geometric content of the local interfaces. It measures a pathway that cannot express that content as a speed-up.

### 3.4 Would the local constraints have been safe?

The test covers the 118 planar components (at least 24 edges) of 52 tasks.

**H1: every edge of the component is free.** This was tested exactly against the scene's bodies.
* H1 holds for 84 of 118 components (71%).
* Where H1 fails, the free share of edges has median 0.9992: a few non-free edges spoil a large component.

**H2 at the true source: a plane through the true source crosses every edge (LP).**
* H2 holds for 51 of 118 components (43%), including 36 components that satisfy H1.
* Intersecting every component's exact stabbing set U(C) would therefore **exclude the true source in 32 of 52 tasks.**

**The failures are small but fatal to an exact constraint.**
* An L1-slack LP on 10 failing and passing components of three tasks shows the following:
  - the best plane through the true source differs from the fitted component plane by 0.0–1.1°;
  - 5–63% of the component's edges violate it;
  - the violations are at most 0.75–1.5 voxel.
* The cause is RANSAC's 0.75-voxel tolerance. It absorbs edges of neighbouring interface pieces across creases, which is
  H2 failing *for the decomposition*. The cone law itself holds.

**The thick heuristic band is safe but uninformative.**
* It contained the true source in 54 of 54 tasks and all of the label mass.
* Its median volume is 2.9% of the cube, larger than B in every task.

## 4. Does the present implementation test hypothesis (II)?

No.
* Local geometry enters only receiver selection, hypothesis (I).
* No local source constraint is ever intersected or applied. The only local region, the band, is a diagnostic that is
  always looser than B.
* §3.3 shows that the deterministic pathway cannot reflect a better local region in its runtime anyway: its cost is in
  certifying F against all receivers.

## 5. An experiment that tests (II) soundly

**Design constraints.**
* Theorem 3 (v2) and Theorem 3′ (v3) of `docs/LOCAL_INVERSE_GEOMETRY.md` rule out constraints computed from X alone in
  the current physics (Model A). §3.4 shows that the natural exact version fails at the true source in 32 of 52 tasks.
* A sound test must therefore do two things:
  - certify each constraint from checkable conditions, so that it is never assumed;
  - report the information content and the computational effect separately, against the bounds of §3.3.

### 5.1 A constraint that needs H1 but not H2

**Proposition A (edge-silhouette constraint).** Hypotheses and claim:
* Let G be a finite union of full-dimensional convex polytopes, with edge set E(G).
* Let (l, k) be a direct interface edge of X (l active, k inactive) that is *free*: the closed segment [l, k] misses G.
* Then every feasible source s′ ∈ F(G, X) outside the finite union of the face planes of G satisfies
  > e ∩ conv{s′, l, k} ≠ ∅ for some edge e ∈ E(G).

*Proof.*
1. **A boundary point on the edge.** For s′ ∈ F, l is active and k inactive, because every feasible source reproduces
   the observation. The edge is free, so Proposition 2 gives p ∈ [l, k] ∩ ∂In(s′) with p ∉ G.
2. **The farthest contact.** In(s′) is closed, so [s′, p] meets G. Let c be its point farthest from s′, and take a
   body Cᵢ ∋ c.
3. **Where the contact can lie.**
   - If c were interior to Cᵢ, every segment [s′, p″] with p″ near p would meet Cᵢ. Then p would be interior to
     In(s′), a contradiction.
   - If c were in the relative interior of a facet whose plane does not contain s′, the segment would cross that facet
     transversally. Nearby segments would cross it too, the same contradiction.
   - Hence c lies on an edge e of Cᵢ (vertices lie on edges), or s′ lies in a face plane.
4. **Conclusion.** In the first case c ∈ e ∩ [s′, p] ⊆ e ∩ conv{s′, l, k}. ∎

**What the proposition gives.**
* *Sound for every feasible source.* H1 depends on (G, l, k) only, and the active/inactive pattern is shared by all of
  F. The constraint is not limited to the true source. The excluded set is Lebesgue-null and carries no label mass.
* *No H2.* Each free edge constrains the source on its own. Creases, mixed components and RANSAC errors cannot make the
  constraint unsafe.
* *The intersection is sound.* For any set I of free edges, R_I = ∩_{(l,k)∈I} Z(l, k) ⊇ F up to a null set, where
  Z(l, k) = ∪_e {s : e ∩ conv{s, l, k} ≠ ∅}. Each Z(l, k) is a union of thin wedge-shaped "triangle-sweep" slivers,
  one per world edge e. Their angular width is about h / dist(e, [l, k]). Choosing edges from components with different normals is
  how independence and Proposition 5's conditioning enter.
* *What it uses.* The world's edge list, but no visibility computation. The fully local, X-only version is impossible
  here (Theorems 3 and 3′). This is the closest sound form of (II) in Model A.
* *Its relation to B.* Z(l, k) is implied by the exact constraints of the receivers l and k. R_I can therefore add
  information over B only through interface edges outside X′. That is the point of the test: thousands of edges can
  enter R_I cheaply, against the 384 receivers of B.
* *Curved occluders.* The ball analogue replaces "e meets the triangle" by "∂b meets the triangle", by Corollary 1b.
  It is again sound.

**Exact fallback for ambiguous interfaces.**
* *Edges.* An edge whose exact free test fails, or is numerically undecided, contributes *no* constraint. Dropping a
  constraint only enlarges R_I. Components are never discarded wholesale: their free edges still count.
* *Box tests.* Region computation uses conservative box-against-Z(l, k) tests: a box is removed only if, for some
  admitted edge, it provably misses every sliver.
  - A box misses the wedge relaxation (the plane through line(e) and s lies between l and k) iff two linear functions
    are strictly one-signed with equal signs on its corners.
  - The sliver test is the exact refinement.
  - Both use directed rounding with an exact rational fallback. An undecided box is kept.
* *Alarm.* On every task, the reference label must have mass 1 inside R_I. Any violation means a bug or a wrong lemma.
  It stops the experiment; it is never "fixed" by tolerance.

### 5.2 Protocol (the same 54 tasks, then the tf_T0 / tf_T2 sets)

**E1 — information.** No pruning anywhere.
* *Edge sets.* R_I for |I| ∈ {8, 32, 128, all free edges}, with three selection rules:
  - one edge per planar component, maximizing σ₃ of the stacked normals (the independence hypothesis);
  - random free edges;
  - all free edges.
* *Report:*
  - Vol(R_I)/Vol(Ω \ B_W), the information from interfaces plus edges alone;
  - Vol(R_I ∩ roots)/Vol(roots), against the stage-1 search region;
  - Vol(R_I ∩ B)/Vol(B) and Vol(R_I ∩ B)/Vol(F), the information beyond the current outer bound;
  - the safety check (label mass inside R_I).
* *Purpose.* This decides whether intersecting independent local constraints reduces the region, and whether the
  local-plane choice beats random edges at equal |I|.

**E2 — computation, certified.** The exact phase runs on B ∩ R_I, and on roots ∩ R_I with the outer stage dropped.
* *Correctness.* Labels must equal the tau-pencil reference within tv_bound.
* *Cost.* Time includes building R_I.
* *Comparison.* Against §3.3's bounds: 1.10× on the exact phase and 1.55× end to end at the median.
* *Statistical pathways.* For FastLabel-type certificates the relevant quantity is volume. There Vol(B)/Vol(F) is
  109× at the median, so E2 also reports the acceptance-rate gain Vol(B)/Vol(B ∩ R_I).

**E3 — the purely local version, where it can be certified.** In the deterministic voxel physics under study on
`yushu/3d-voxel-physics` (Model B), the window observation determines the occupancy inside the window exactly. There
the X-only region is provably exact, without H1 or H2:

> L_B(X) = {q ∉ B_W, legal w.r.t. the observed SOLID cells :
>   (a) no active receiver's segment from q touches an observed SOLID cell, and
>   (b) every inactive receiver not already blocked by an observed SOLID cell has, on its segment from q, a cell
>       outside the window that touches no active receiver's segment from q and lies farther than ρ from q}.

* *Exactness.* L_B(X) is exactly the union of F_B over all voxel worlds that agree with the window.
  - "⊆": a blocker of each inactive receiver must lie outside the window and off every active segment.
  - "⊇": the world consisting of the observed SOLID cells plus all admissible cells outside the window reproduces X
    from q.
* *Status.* Derived during this session. It is **not yet numerically verified**.
* *Measures:*
  - Vol(L_B)/Vol(Ω);
  - F_B ⊆ L_B, on Monte Carlo samples of F_B and on the true source;
  - the cost of the membership test (two GPU traversals per candidate).
* *Meaning.* This is the cleanest test of the original hypothesis: local constraints, provably safe, from the
  observation alone.

**Decision rules (fixed in advance).**
* *(II) is supported as information* if Vol(R_I ∩ B) ≤ 0.1·Vol(B) at the median, with the independence rule better than
  random at equal |I|.
* *(II) is supported as computation* in a pathway if its certified E2 speed-up reaches at least half of that pathway's
  oracle bound.
* *(II) has no measurable effect in a pathway* if it reaches neither threshold. That conclusion must name the pathway.

**What this experiment does not do.**
* It applies no uncertified pruning. The band and the H2-based stabbing sets remain diagnostics, reported with their
  measured failure rates.
* It does not reinterpret §3's receiver ablation as evidence about (II).
