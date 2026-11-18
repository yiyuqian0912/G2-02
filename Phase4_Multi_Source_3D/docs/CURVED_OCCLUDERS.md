# Curved occluders: closed balls (research extension)

**Scope.** This note documents a validated sphere-only extension of the physical stage. Closed balls B(c, r) are
composed by union with the certified convex-polytope engine. The polytope engine itself is unchanged: neither the
native C++ nor the CUDA kernel was modified, and no rebuild is needed.
* There is no certified posterior labeller for balls, so curved scenes are not part of T*_v3.
* They are generated under the research contract `rind3d-100k-curved-research-v0`, for physical canaries and
  benchmarks only.

## 1. Semantics

The policy is the generator's `closed-solid-closed-segment-v1`, extended to balls:
* a receiver is SOLID iff it lies in some polytope or some closed ball;
* a receiver is response-active iff the closed segment from the driver source misses every polytope and every closed
  ball.

Tangency counts as a hit. Ball centres and radii are binary64 values, i.e. exact dyadic rationals. Receivers sit at
(2v + 1)/(2N), also exact in binary64.

**Segment test.** For the closed segment [s, x] and the ball B(c, r), let w = s − c and d = x − s. Define

    A = d·d,   B = w·d,   C = w·w − r²,   E = |x − c|² − r²,
    hit  ⇔  C ≤ 0  or  E ≤ 0  or  (B < 0 and A + B > 0 and A·C − B² ≤ 0).

* The last case is the interior minimum of |s + t d − c|² − r² at t* = −B/A ∈ (0, 1), multiplied by A > 0.
* The test is division-free, so it is exact in rational arithmetic.

**Membership.** E ≤ 0.

**Clearance of a source.** |s − c|² > (r + ρ)², with ρ = 2⁻²⁶.

## 2. Certified evaluation (`RIND3D_Generator/curved/rind3d_curved/`)

* **Exact path.** `spheres.segment_hits_ball_exact`, `ball_contains_exact` and `clearance_exact` use Python
  `Fraction`s.
* **CPU fast path.** `segment_hits_ball_interval` uses outward-rounded binary64 interval arithmetic:
  - each operation is round-to-nearest followed by one `nextafter` step outward, so every interval contains the exact
    value;
  - each sign is decided as true, false or UNKNOWN, and the three-valued results are combined in Kleene logic;
  - UNKNOWN cases are recomputed exactly (`BallSet.blocked`, `BallSet.contains`).
* **GPU path.** `cuda_balls.BallTiles` renders 64³ receiver tiles with a CuPy/NVRTC kernel:
  - it uses directed-rounding intrinsics only (`__dadd_rd/_ru`, `__dmul_rd/_ru`); subtraction is addition of the
    exact negation, and the kernel is compiled with `--fmad=false`;
  - UNKNOWN voxels are flagged and recomputed exactly on the host;
  - batched launches cover many tiles at once.
  - *Context isolation.* The certified polytope backend owns a separate CUDA context (`cuCtxCreate`). CuPy's primary
    context is made current around each CuPy call, and the previous context is restored afterwards.
* **All-active certificate.** `ball_misses_hull_exact` searches a float separating direction n by Frank–Wolfe. The
  bound min over hull vertices of n·(c − p) > r|n| is then verified exactly. It certifies that no ball meets
  conv({s} ∪ receiver box). Combined with the existing polytope hull certificate, it proves a channel all-active
  without rendering.
* **Composite scene.** `rind3d_100k/curved_scene.CurvedLocalScene` combines the polytope bits from the unchanged
  certified backends with the ball bits, as SOLID = OR and active = AND NOT.
  - Its `probe` and `_packed` follow the LocalScene interface.
  - The v3 decision machinery (`evaluate_proposal_interface`, tile scans, bisection proofs) therefore applies
    unchanged, and the stored crops are exact.

## 3. Generator (`rind3d_curved.families`, version `curved-spheres-v0`)

| family | bodies |
|---|---|
| `spheres` | 3–10 balls, r ~ log-uniform(0.02, 0.12) (20–123 voxels), centres uniform in [r + 0.01, 1 − r − 0.01]³ |
| `spheres_polyhedra` | the v3 `convex_polyhedra` recipe of the same (seed, slot, attempt), plus 2–6 such balls from an independent stream |

**Draws.** All draws are keyed by (seed, slot, attempt).

**Validity.** A ball must be strictly inside the open world and have a nonempty, 6-connected voxel-centre sampling.
This is checked exactly by column runs with rational re-checks; at these radii it always holds.

**Driver sources.** They follow the generator's uniform law with exact legality:
* polytope legality when polytopes are present;
* exact ball clearance;
* the 2⁻²⁰ anti-duplicate rule.

**Not a mesh approximation.** No ball is replaced by a faceted surrogate.

The 1K study's `faceted_spheres` family (`RIND3D_Dataset1K/docs/GEOMETRY_DIVERSITY.md`) was a polyhedral
approximation. It is not related to this exact extension.

## 4. Validation (`RIND3D_Dataset100K/tests/test_curved_spheres.py`, all passing)

* **Two exact implementations agree.** The exact predicate agrees with the independent canonical oracle
  `rind.ellipsoid3d.ExactEllipsoid3D` (exact rational, different algorithm). The test uses hit-rich, near-tangent
  samples: receivers on rays within ±2% of the tangent-cone angle, 20–80% hits.
* **Fast paths never decide wrongly.** Interval decisions are never wrong where certain, and `BallSet` (interval plus
  exact fallback) equals the exact predicate.
* **Exact tangency.** A segment tangent to the ball at a lattice row is a hit on the exact, interval and canonical
  paths; the next row is a miss.
* **Membership and clearance.** Tested at offsets of 2⁻⁴⁰ and 2⁻²⁷.
* **Sampled connectivity.** Agrees with dense 6-connected labelling.
* **Hull certificate.** Sound on 200 random cases: no sampled hull point lies in the ball when it certifies.
* **Determinism** of generation.
* **GPU against exact CPU.** GPU tiles equal the exact CPU decisions, including tiles placed on the near-tangent shadow
  boundary.

**End to end.** Two research scenes (`spheres`, `spheres_polyhedra`) were generated by `run_scene_curved`. Every
stored direct-interface mask equals the one recomputed from the stored crops. 1,920 stored receivers of the
sphere-only scene equal the canonical oracle. No UNKNOWN voxel occurred: all decisions came from the directed-rounding
path.

## 5. Cost and limitations

**Cost.** See `RIND3D_Dataset100K/docs/GENERATOR_BENCHMARK_V3.md` for measured per-family costs.
* Sphere-only scenes are about an order of magnitude slower than polytope scenes on this laptop.
* The host-side combination of ball and polytope bits costs about 6 ms per tile, and sphere-only scenes have no
  polytope hull certificate to skip channels.

**Limitations.**
* **No certified labeller.** tau-pencil and FastLabel take halfspace rows only, and the FastLabel GPU verifier has no
  ball test. Two possible routes:
  - FastLabel with polytope relaxations. Use a circumscribed polytope for the inactive-receiver constraints and an
    inscribed one for the active-receiver constraints, which keeps B ⊇ F. Then verify sample points exactly with the
    ball predicate.
  - A tau-pencil extension with conical interface surfaces. LOCAL_INVERSE_GEOMETRY.md Corollary 1b gives the
    geometry.
  Either is future work. Until then, curved families must not enter a labelled population.
* **Other primitives.** Ellipsoids are supported by the canonical exact oracle, and the same division-free approach
  extends to them through the affine map. They are not implemented in the fast paths. Cylinders and capsules are not
  implemented.
* **Interface statistics.** The local interface of a ball is a cone with apex at the source (Corollary 1b). Local
  plane fits leave curvature residuals. See §6 once measured.

## 6. Local interface statistics of curved scenes

**Setup.**
* Scenes: 4 per family under the curved research contract (`RIND3D_Posterior/experiments/localfirst/curved_stats.py`).
* Data: `RIND3D_100K_v3_work/curved_stats/`, outside Git.
* No labels exist for balls, so these are X-only diagnostics, plus the containment of the true driver source in the
  HEURISTIC band.

| family | channels | component rms off-plane residual (voxels), p50 L64 / L128 / L256 | p90 | rank of the component normals, 0 / 1 / 2 / 3 | true source in the heuristic band |
|---|---|---|---|---|---|
| spheres | 89 | 0.37 / 0.46 / 0.49 | 0.53 | 3 / 14 / 41 / 31 | 85/89 |
| spheres_polyhedra | 70 | 0.29 / 0.40 / 0.47 | 0.53 | 1 / 15 / 32 / 22 | 69/70 |
| v3 polytope families (54 `cmp_v3` tasks, components ≥ 24 edges) | — | 0.21 overall | 0.27 | (see LOCALFIRST_COMPARISON.md) | 54/54 |

**Readings.**
* **Curvature is visible.**
  - Ball interfaces leave about twice the polytope off-plane residual.
  - The residual grows with the window side, as Estimate 7 of `LOCAL_INVERSE_GEOMETRY.md` predicts: the sagitta grows
    with the square of the patch extent.
  - A planar decomposition still explains almost all edges: the median explained share is 0.999 and 1.0.
* **Rank is higher.** Rank 2 and 3 dominate: 81% of channels for spheres, 77% for the mixed family. A curved patch is
  split into several planes with different normals.
  - This is exactly the conditioning Estimate 7 warns about. The extra "planes" are facets of one cone, not independent
    shadow planes.
  - Their singular values overstate the localization a source-region constraint would deliver.
* **The band is not safe here.** It misses the true source in 4 of 89 spheres channels and 1 of 70 mixed channels.
  This is the expected failure of the planar band (one plane per component, H2) on cone-shaped interfaces.
  - Any source-region use of curved interfaces needs the cone form of Proposition A (`RIND3D_Posterior/docs/LOCALFIRST_AUDIT.md` §5.1):
    "∂b meets the triangle conv{s, l, k}".
  - The band must not be used to prune.
