# Observable thin slabs (contract v3 geometry rule)

**Goal.** Make thin occluders physically observable and spatially coherent: no physically present slab that is
invisible in the voxel-centre SOLID mask. The goal is not to remove difficult shadow geometry. This is a change of
the physical dataset distribution, versioned in contract v3 (`geometry`).
* Contract v2 and its generator are unchanged.
* The frozen 1K release is untouched.

## 1. The rule (`rind3d_100k/geometry_v3.py`)

**Thickness.** Only the `thin` kind changes: every body of `thin_slabs`, and a quarter of the bodies of
`occluder_clusters`.
* v0.1 draws log₂(half-thickness) ~ U(−20, −5), giving a full thickness of 2⁻¹⁹ to 1/16 of the world (0.002 to 64
  voxels at N = 1024).
* v3 draws log₂(half) ~ U(log₂(T_min/(2N)), −5). This is the v0.1 law conditioned on thickness ≥ T_min voxels.
* The default is T_min = 2. It is configurable as `geometry.thin_min_thickness_vox`.

**One source of truth.** One continuous canonical body defines both the SOLID observation and the exact
source-to-receiver visibility.
* Nothing is repaired on the voxel side.
* The draw consumes the same single uniform as v0.1. Every other draw of the scene is therefore the v0.1 draw for the
  same (seed, slot, attempt).
* With T_min = 0 the generator is byte-identical to v0.1 (`test_t0_is_byte_identical_to_v01`).

**Validity.** For every thin slab, the voxel-centre sampling {v : (v + ½)/N ∈ slab} must be nonempty and 6-connected.
* It is decided exactly by `sampled_solid_components`: columns along one lattice axis, each a single run by convexity.
  4-adjacent columns are 6-connected iff their runs overlap. Every run end within 10⁻⁹ voxel of a float bound is
  re-decided in exact rational arithmetic on the canonical rows.
* The checker agrees with dense 6-connected labelling on generated slabs (test).
* **On failure.** The scene attempt is a geometry proposal failure, `thin_slab_invalid`. It is recorded and retried
  under `max_attempts` = 3. No invalid slab is accepted.
* **Validation.** The validator recomputes every recorded slab's sampling from the stored geometry.

## 2. Mathematics: what thickness can and cannot guarantee

**Theorem 1 (unbounded slab).**
* *Claim.* Let n be a unit vector and T ≥ ‖n‖₁, in voxel units (h = 1). The lattice points of any closed slab
  {p : a ≤ n · p ≤ a + T} form a nonempty 6-connected set. Since ‖n‖₁ ≤ √3, T ≥ √3 ≈ 1.73 voxels suffices for every
  orientation, so T_min = 2 does.
* *Reduction.* Reflect the axes so that n ≥ 0, and order them so that n_z = max nᵢ. Reflections preserve
  6-adjacency.

*Proof.*
1. **The standard plane is 6-connected.** Let S = {p : a ≤ n · p < a + ‖n‖₁}. In each column (x, y) the admissible z
   form an interval of length ‖n‖₁ / n_z ≥ 1, so each column holds a nonempty run. Take p ∈ S in column (x, y). Either
   p + e_x ∈ S, or n · p + n_x ≥ a + ‖n‖₁. In the second case both p − e_z and p − e_z + e_x lie in S: their values are
   n · p − n_z ∈ [a + n_y, n · p) and n · p + n_x − n_z ∈ [a + n_x + n_y, n · p], both inside [a, a + ‖n‖₁), using
   n_x, n_y ≤ n_z. So (x, y) and (x + 1, y) are joined by a 6-path
   inside S, and likewise for y. Runs are contiguous and the column lattice is connected, so S is 6-connected.
2. **The slab reduces to it.** Let D be the slab and p ∈ D. While n · p ≥ a + n_z, the point p − e_z stays in D and its
   value drops by n_z. This reaches the bottom layer {a ≤ n · p < a + n_z}, which lies in the standard plane
   S_a ⊆ D. Every point of D is therefore 6-connected within D to S_a, and S_a is connected by step 1. ∎

**Sharpness.**
* *The infinite case.* For n = (1, 1, 1)/√3 and T = 1 voxel, a slab placed so that only one level of x + y + z falls
  inside is *totally* 6-disconnected: every 6-step changes x + y + z by 1
  (`test_one_voxel_diagonal_slab_is_totally_disconnected`). Measured: a generated 1.1-voxel slab with ‖n‖₁ = 1.73
  sampled into 254 components.
* *The finite case.* The bound ‖n‖₁ is sufficient, not tight, for 6-connectivity.

**Finite slabs: thickness alone does not suffice.**
* *Why the proof stops.* Generated slabs are bounded rotated boxes. Theorem 1 covers their interior, but not their rim
  (edges and corners).
* *Observation.* At every thickness, a lattice point within a fraction of a voxel of a rotated box's edge or corner can
  have all six neighbours outside the box. Measured cases had the isolated point 0.0 to 0.26 voxel from the rim, on
  slabs 3.8 to 59 voxels thick (`test_rim_isolation_is_detected_regardless_of_thickness`).
* *Consequence.* No thickness condition alone guarantees a connected sampling of a finite rotated slab. A rounded rim
  (an inner rolling ball of radius ≥ √3h/2) would be a candidate sufficient condition. It is not proved here and not
  adopted, because it would change the occluder shape.
* *Conclusion.* The exact check is necessary, not merely a safeguard.

**Measured rates.** 231 slabs (the first 30 slots of `thin_slabs` and `occluder_clusters`, seed 20261216):

| T_min | invalid slabs | of which empty | cause |
|---|---|---|---|
| 0 (v0.1) | 148 (64%) | 0 | sub-voxel slabs sample into scattered points: nonempty, disconnected |
| 1 | 13 (5.6%) | 0 | interior disconnection where T < ‖n‖₁, plus rim isolation |
| 2 | 2 (0.9%) | 0 | rim isolation only (single voxels) |
| 3 | 3 (1.3%) | 0 | rim isolation only |

**Tests (`tests/test_thin_slabs_v3.py`).**
* the thickness law and paired draws;
* identity with v0.1;
* checker against dense labelling;
* Theorem 1 on random orientations;
* sharpness at 1 voxel;
* rim isolation;
* continuous/discrete SOLID agreement with the certified predicate;
* near-surface visibility against an exact rational segment test.

## 3. Effect on the dataset (canary, laptop; docs/SAMPLING_IMPACT_V3.md)

Thin-slab families only: `thin_slabs` and `occluder_clusters`, 32 scenes per arm. The streams are identical across
T_min because the seed is the same; only the thickness draw changes.

| T_min | rejected attempts / scenes | scenes finally accepted | natural P_A L64 / L128 / L256 | median scene wall (s) |
|---|---|---|---|---|
| 0 | 0 / 32 | 32 | 0.355 / 0.565 / 0.761 | 5.2 |
| 1 | 7 / 32 | 32 | 0.346 / 0.563 / 0.760 | 5.7 |
| 2 | 2 / 32 | 32 | 0.349 / 0.559 / 0.756 | 5.8 |
| 3 | 3 / 32 | 32 | 0.349 / 0.561 / 0.759 | 5.5 |

**Readings.**
* **Diversity.** The thickness distribution moves from 0.002–64 to 2–64 voxels, still log-uniform. In-plane sizes,
  rotations and placements are unchanged (paired).
* **Cost.** The check costs about 7 ms per slab. A retried attempt costs one geometry draw. Wall-time differences
  between arms are within run-to-run variation.
* **Window acceptance.** P_A changes by ≤ 1.7% (relative) at every side.
* **Direct interfaces.** v3's direct-interface rule excludes more windows once slabs are thick (L64: 0.4% → 1.3% of
  v2-admissible proposals). A sub-voxel slab let the two sides of a slab touch as a false FREE–FREE "interface"
  (an occluder interface). A 2-voxel slab is SOLID, and such windows are honestly SOLID-separated.
* **Labelling.** Measured on 36 labelling tasks per arm, drawn from the T_min = 0 and T_min = 2 canaries.
  - The tasks are 2 per (family, L) group over the two thin-slab families. They are only partly paired: windows depend
    on the geometry, and 15 of 36 task keys coincide.
  - Laptop, one process per label, as in `RIND3D_Posterior/docs/LOCALFIRST_COMPARISON.md`.
  - Data: `RIND3D_100K_v3_work/cmp_tfT0`, `cmp_tfT2`.

  | arm | engine | certified | wall s, p50 / p95 / max | non-free interface edges (H1) | channels with a non-free component |
  |---|---|---|---|---|---|
  | T_min = 0 | deterministic (B, then exact on B) | 35/36, 1 timeout at 1800 s | 2.6 / 67 / > 1800 | 4.5% | 30% |
  | T_min = 0 | FastLabel v3.4 | 36/36 (29 statistical, 7 exact) | 1.9 / 2.6 / 10.8 | | |
  | T_min = 2 | deterministic (B, then exact on B) | 36/36 | 2.5 / 21 / 51 | 0.025% | 13% |
  | T_min = 2 | FastLabel v3.4 | 36/36 (28 statistical, 8 exact) | 1.6 / 2.4 / 2.4 | | |

  - *Tails.* Sub-voxel slabs produce the heavy tails. The only deterministic timeout is an `occluder_clusters` L64
    window under T_min = 0, and FastLabel's slowest label falls from 10.8 s to 2.4 s.
  - *Medians.* They are unchanged.
  - *Interface quality.* The H1 numbers (`docs/LOCAL_INVERSE_GEOMETRY.md` §3b) show the thickness rule removing
    almost all occluder-crossing interface edges. The channel-level share stays at 13%, because knife edges and rims
    remain.

**What observability does not remove.** Visible SOLID does not make every response interface source-containing.
Sharp edges and vertices of any convex body (wedges with 7° knife edges, random polyhedra, box and slab rims) have
sub-voxel parts. These can still cross a receiver edge between two FREE receivers
(`Phase4_Multi_Source_3D/docs/LOCAL_INVERSE_GEOMETRY.md` §3b, Theorem 3′).
