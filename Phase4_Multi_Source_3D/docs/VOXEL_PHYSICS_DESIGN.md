# Deterministic voxel physics (Model B): design, theory, A/B evidence and labeller compatibility

Branch `yushu/3d-voxel-physics` (research only; the v3 contract and the production code are untouched).
Code: `RIND3D_Generator/voxel/rind3d_voxel/`. Tests: `RIND3D_Dataset100K/tests/test_voxel_physics.py`, 41 passing.
Experiments: `RIND3D_Posterior/experiments/voxel/` (`voxel_ab.py`, `e3_local.py`, `label_feasibility.py`).
Data: `RIND3D_100K_v3_work/{voxel_ab, e3, voxel_label}`, outside Git.

## 1. The two models

**Model A** (v3, continuous):
* Solid: a union S of closed convex bodies.
* Receivers: the voxel centres (2v + 1)/(2N), N = 1024.
* Visibility: a receiver is response-active iff the closed segment [s, x] misses S.
* SOLID observation: x ∈ S, a centre sample.

**Model B** (authoritative voxel world):
* *World.* The closed supercover V(G) = {cells C_v : C_v ∩ G ≠ ∅} of the same generative geometry G. The physical solid
  is S_B = ∪_{v ∈ V} C_v, with cells closed.
* *Visibility.* A receiver x_v is response-active iff v ∉ V and the closed segment [s, x_v] touches no cell of V.
* *Sources.* Continuous, legal iff dist(s, S_B) > ρ = 2⁻²⁶.
* *Labels.* The posterior target is unchanged: q(c) = Vol(F_B ∩ c)/Vol(F_B) on the 128³ grid.

## 2. Implementation

* **`voxelize.py`.** Box-against-body classification for each candidate cell.
  - Polytopes use a separating-axis test with facet normals, cell axes and their cross products, a 10⁻¹⁰ margin, and
    an exact rational fallback inside the margin.
  - Balls use exact squared distances.
  - Supported rules: `supercover` (Model B) and `center` (Model A's SOLID samples).
* **`world.py`.** Sparse 64³ bricks (a FULL marker or packed bits), with a digest, a dense bitset on demand, crops and
  exact source legality.
* **`visibility.py`.** The exact rational reference set of touched cells; `free_exact`; and `free_fast`, a vectorized
  Amanatides–Woo walk with exact fallback on ties.
* **`cuda_dda.py`.** A CuPy kernel, one thread per receiver, using float64 without fused multiply-add.
  - Crossing parameters are recomputed fresh from the cell index at every step.
  - Steps whose two smallest parameters agree within 10⁻¹² relative are *flagged* and recomputed by the rational
    reference on the host.
  - A variant skips empty 8³ blocks.
  - It runs in a process without the certified polytope CUDA runtime, per the repository rule.
* **`local_region.py`.** Theorem 10 (§3): a rational reference, the maximal-world construction, and sound GPU pruning.

## 3. Theory

**P1. Determinism and exactness.**
* V(G) depends only on G. The order of bodies is irrelevant (the digest is tested).
* Every visibility decision equals the closed-segment/closed-cell rule.
* Across all A/B runs, 0 GPU rays were flagged.

**P2. Sandwich and monotone response.**
* *Statement.* G ∩ [0,1]³ ⊆ S_B ⊆ G ⊕ [−h, h]³, and centre sampling ⊆ supercover. Hence every Model B-legal source is
  Model A-legal, FREE_B ⊆ FREE_A, and on FREE_B a Model B-active receiver is Model A-active.
* *Proof.* Each point of G lies in a closed cell, which meets G. Each point of a meeting cell is within ∞-distance h of
  G. A segment missing S_B misses G. ∎
* *Measured.* 0 receivers active in B and inactive in A, over 145 window channels (§5).

**P3. Connectivity.**
* *Statement.* The closed supercover of a connected set K is 6-connected. In particular every body, however thin, is
  present and connected. The thin-slab rule (T_min) is a Model A artifact.
* *Proof.* Let K_v = K ∩ C_v. These are closed sets covering K. The graph "K_u ∩ K_v ≠ ∅" is connected, otherwise K
  would split into two disjoint closed sets. For an edge u ∼ v of that graph, a common point p lies in every cell
  containing p. Those cells form a 1×1×1 to 2×2×2 block, which is 6-connected and contained in V. ∎

**P4. Complete window observation.** The SOLID mask of a window is exactly V ∩ W. Inside the window box there is no
unobserved geometry. Model A observes only centre samples (the cause of Theorem 3′ in `LOCAL_INVERSE_GEOMETRY.md`).

**P5. The interface law.**
* *H1 always holds.* For 6-adjacent FREE cells u and v, the segment between their centres touches only C_u and C_v
  (tested), so every direct interface edge is free.
* *Staircase law.* By `LOCAL_INVERSE_GEOMETRY.md` Proposition 2 and Corollary 1a applied to the cubes of S_B, for almost
  every source each interface edge is crossed by a plane through the source and an *axis-aligned lattice edge* of
  S_B. So every elementary interface piece lies in a plane containing the source and a coordinate direction.
* *Consequence for H2.* H2 becomes worse. A digitized oblique silhouette is a staircase of such pieces, and a fitted
  component mixes them.
* *Measured.* The exact stabbing LP through the true source holds for 0/103 largest Model B components, against
  38/106 in Model A, and component residuals grow from 0.21 to 0.35 voxel. The plane-fitting route is therefore
  *less* usable in Model B. Theorem 10 does not need it.

**Theorem 10 (the exact local region; the X-only answer).**
* *Admissible worlds.* Let 𝒱 be the class of all complete voxel worlds (any occupancy outside the window). For an
  observation X = (σ, r) on a window W, let 𝒱(X) = {V ∈ 𝒱 : V ∩ W = σ}.
* *The region.* For q ∉ B_W with dist(q, σ) > ρ, define q ∈ L_B(X) iff both:
  - (a) no active receiver's closed segment [q, x] touches a cell of σ;
  - (b) every inactive receiver y whose segment touches no cell of σ has, on [q, y], a cell C ∉ W that touches no
    active segment from q and lies farther than ρ from q.
* *Claim.* L_B(X) = ∪_{V ∈ 𝒱(X)} F_B(V, X).
* *Proof, "⊆".* A feasible q is legal. Active segments miss every cell of V ⊇ σ. An inactive y not blocked inside W is
  blocked by some cell of V \ W; that cell touches no active segment, and its distance from q exceeds ρ because q is
  legal.
* *Proof, "⊇".* The maximal world V′ = σ ∪ {C ∉ W : C touches no active segment, dist(q, C) > ρ} agrees with σ on W.
  It keeps every active receiver active, by (a) and by construction. It blocks every inactive receiver, by σ or by (b).
  And q is legal in it. ∎

**The three quantifiers, kept apart:**
1. *Fixed known world.* F_B(V, X) is exact, and it is what a labeller computes.
2. *Every admissible world.* L_B(X) is exact and computable from X, the window location and size, and the declared
   physics only.
3. *Generator assumptions.* L_𝒢(X) = ∪_{V ∈ 𝒢(X)} F_B(V, X) for a restricted class 𝒢, for example supercovers of v3
   bodies. It satisfies L_𝒢 ⊆ L_B, but it is safe only under the assumption 𝒢, and it is not computed here.

**Verification of Theorem 10.**
* *Exact, both directions* (`test_local_region_is_exact_union_over_worlds`, 32³ grids, rational arithmetic): q ∈ L_B
  iff the maximal world reproduces X from q, and the true source always lies in L_B.
* *The quantifier* (`test_local_region_contains_the_source_for_random_hidden_worlds`, 16³ grids, random complete
  hidden worlds at three densities, random sources): the source is always in the region computed from the observation
  it produces.
* *GPU soundness* (`test_local_region_gpu_pruning_is_sound`): GPU pruning implies exact pruning. For pruning, marking
  of active segments under-approximates and the blocker search over-approximates, both on the safe side.

**What Theorem 10 does not remove.** L_B is the union over hidden worlds, so it is generally larger than the
fixed-world F_B. The posterior label still needs the world (§6).

## 4. Validation summary

All of the following pass in `test_voxel_physics.py`:
* voxelization equals the brute-force exact rule for both rules, on a box, a thin slab, a wedge, a ball and a
  sub-voxel ball;
* the supercover is 6-connected;
* fast traversal equals exact traversal;
* adversarial rays through lattice vertices and edges;
* exact legality;
* determinism and order independence;
* interface edges touch only their two cells;
* Theorem 10 (7 tests).

## 5. Model A against Model B on identical scenes (`voxel_ab.py`)

**Setup.**
* 19 scenes: 12 v3 polytope scenes, 2 v0.1 sub-voxel slab scenes, the knife-edge wedge of Theorem 3′, and 4 curved
  scenes.
* 2 driver sources each (legal in both models), 4 windows each (three L64, one L128).
* 145 window–channel pairs.

| quantity | value |
|---|---|
| occupied cells, B / Model A samples, v3 scenes p10 / p50 / p90 | +2.0% / +4.3% / +6.6% (v0.1 sub-voxel slabs: up to +365%) |
| voxelization (numpy, one process) p10 / p50 / p90 | 1.3 / 8.5 / 32 s per scene |
| sparse storage p10 / p50 / p90 | 1.7 / 6.0 / 19 MB (dense bitset for the GPU: 128 MiB) |
| window rendering, per window channel | A (certified CPU predicates) 1.54 s; B (GPU, exact) 0.094 s; 0 flagged rays |
| source legality disagreement (uniform points) | ≤ 0.33% |
| response differences on commonly FREE receivers p50 / p90 / max | 0.66% / 4.4% / 12%; **all A-active, B-inactive** (P2; 0 violations) |
| channel class changes | 3 of 145 |
| Model A interface edges that are not free (H1 fails) | 2,209 of 723,432 (knife wedge: 273 of 2,924); Model B: 0 by P5 |
| H2 at the true source, largest component | A 38/106; B 0/103 (staircase) |

The per-scene table is the output of `voxel_ab.py report`.

## 6. E3: how much source information the local observation alone carries (`e3_local.py`)

**Setup.**
* 37 windows (L64) with a direct interface, from all 19 A/B scenes: v3 polytopes, v0.1 sub-voxel slabs, the knife edge,
  and voxelized spheres.
* For each window, the Model B observation is rendered once.
* L_B(X) is then evaluated by `LocalRegionGPU`, which reads only the observed SOLID cells, the binary responses, and
  the window origin and side.

| measure (37 windows) | value |
|---|---|
| safety: true source kept / fixed-world feasible sources kept (perturbations reproducing X exactly) | 37/37 / 255/255; 0 alarms |
| uniform candidates of the cube kept, p10 / p50 / p90 / max | 0 / 0 / 0 / 0.993 |
| windows where none of 300 uniform candidates is kept (L_B < 1% of the cube at 95%) | 34 of 37 |
| windows with **no observed SOLID** (21): share pruned by (b), the response interface alone | 100% at p10, p50 and p90 |
| windows with observed SOLID (16): pruned by (a) / by (b), p50 | 79% / 21% |
| extent of L_B relative to F_B around the true source, per random direction, p10 / p50 / p90 | 1.0 / 1.0 / 27; 67% of directions within 2×, 84% within 10× |
| weak cases | the 2 windows with < 100 active receivers (kept 23% and 99%) |
| cost | 2.4 s setup per window (GPU world of the observed SOLID), 0.03–0.21 s per candidate |

**Readings.**
* **The trivial exclusion is not the whole story.** Excluding only the observed SOLID and its shadows, condition (a),
  does nothing in 21 of 37 windows. The binary-response interface, condition (b), prunes everything there.
* **What (b) is geometrically.** An inactive receiver must be blockable outside the window without touching any active
  segment. So every inactive receiver seen from q *through* active receivers is a contradiction. This is the cone law
  at voxel resolution, the original "interfaces point at the source" intuition.
* **How tight it is.** Locally the X-only region is about as thin as the fixed-world feasible set along most
  directions. The exceptions are directions along which hidden worlds can trade one blocker for another (p90 27×).
* **Generality.** Voxelized spheres behave like polytopes: the theorem holds for every voxel world, curved or not.

## 7. Labeller compatibility (Tau-pencil, FastLabel; `label_feasibility.py`, 8 L64 tasks)

**Direct use is impossible.**
* The tau-pencil engine and FastLabel work on at most 62 convex components (a component bitmask) and compress
  receiver constraints per silhouette edge.
* A Model B world is a union of 2×10⁶–10⁸ occupied cubes (§5). Even merged into maximal boxes, it is orders of
  magnitude beyond both limits.

**The sound relaxation with the existing engine is too loose.** By P2, S_B lies between the original bodies (inner)
and the face-offset polytopes {a·x + d ≤ h‖a‖₁} (outer). So F_B lies inside B_dark ∩ B_lit, where:
* B_dark = F(dark receivers; outer bodies);
* B_lit = F(lit receivers; inner bodies).

Both are computable by the existing stage 2 on a receiver subset.

| measure (8 tasks) | p10 / p50 / p90 |
|---|---|
| Vol(B_dark) as a share of the cube | 0.75% / 1.8% / 3.0% (Model A's exact-receiver B is ~10⁻¹¹ of the cube) |
| uniform points of B_dark that are also in B_lit | 0.18% / 1.6% / 9% |
| uniform points of B_dark ∩ B_lit that are truly feasible in Model B | 0 / 0 / 0 (0 of all samples) |
| exact Model B verification per candidate (GPU window render) | 26 / 34 / 67 ms (FastLabel's Model A verifier: 0.25 ms) |
| receivers the relaxation cannot decide for a candidate (need the voxel walk) | 722 / 7,600 / 163,000 |

**Verdict.**
* There is no certified Model B posterior labeller at production cost today.
* A FastLabel-style certificate (≥ 95% of the outer bound feasible) is out of reach with this relaxation.
* An exact Model B engine would need a new representation of voxel occluders. It is not a modification of the
  tau-pencil engine.

## 8. Decision and recommendation

**Options considered.**
* *(A) Adopt Model B as the RIND3D-100K physics.*
  - For: exact determinism, complete window observation, no sub-voxel artifacts.
  - Against: no certified labeller (§7).
  - Not recommended.
* *(B) Model B for a separate research population without posterior labels.* For example Student studies of local
  inference, where Theorem 10's X-only region can serve as an exact target or as a consistency check.
  - Feasible now.
  - Recommended as research only.
* *(C) Keep continuous physics (Model A, v3 contract) for production.*
  - Keep the v3 thin-slab rule and the v3 interface rule.
  - Carry over the validated insights: monotone dilation bounds, the 6-connectivity argument behind T_min, and the
    measured H1/H2 behaviour.
  - **Recommended.**
* *(D) Drop the voxel research.* Not recommended: Theorem 10 is the strongest local-inference result we have (§6).

**Recommendation: (C) for production, with (B) for research.** The deciding fact is labelling cost, not physics
quality. Model B is the better-posed physics, and its local information is strong and certified. But its posterior
labels cannot currently be certified at acceptable cost.

**Open problems.**
* An exact engine for voxel occluders.
* Whether a generator-restricted class 𝒢 (supercovers of v3 bodies) yields a computable L_𝒢 tighter than L_B.
