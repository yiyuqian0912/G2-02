# Curved occluders (closed balls) in RIND3D-100K v3: final assessment

**Date and base.** 2026-11-18. Branch `yushu/3d-v3-curved-assessment`, based on the v3 staging commit `bece73b` (the
certified deterministic labelling commit).

**Scope.** This note assesses closed balls ("spheres") in four respects: physical correctness, cost, scientific value
and certifiability. It ends with a recommendation for the production contract.
* It changes no code, contract or dataset.
* The checks it adds are scripts in `RIND3D_Dataset100K/scripts/curved_assessment/` (§6).

**Measurement conditions.** All new timings were taken on the development laptop **under load**: 4 concurrent
certified-label benchmark processes and other jobs were running, with a load average of 3.3–5.1 on 16 hardware
threads. The GPU was otherwise idle. Each check used one process.

## Verdict in brief

| question | answer |
|---|---|
| Is the ball physics correct? | **Yes.** The 12/12 unit tests pass. The new checks found 0 mismatches: 9,863 stored channel receivers and 2,040 SOLID receivers against the independent canonical oracle, and 6.3 M voxels of whole GPU tiles against the exact CPU path (§1). |
| What does it cost? | **Physical stage: 5.6–7.1× a polytope scene** at the contract's mean source count (29–32 s vs 4.6 s on the laptop). Most of the overhead lies in implementation choices that look avoidable, but no optimized version has been measured (§2.2). Storage and memory are unchanged. **The label cost is undefined:** no labeller exists. |
| Does it add diversity? | **Yes, in interface geometry.** A ball's response interface is a smooth circular cone; every polytope interface is a pencil of planes. The 20–48-plane faceted spheres of the 1K study differ from the ball by 1.7–27 voxels. Whether this helps Students is **unmeasured** (§3). |
| Can sphere posteriors be certified deterministically? | **Not today.** A sound route exists: adaptive polyhedral relaxation inside the existing proof-tree verifier. Its certificate is always a TV bound > 0. It needs a new labeller and a verifier extension, and its cost is unknown. A global inner/outer polyhedral sandwich is sound but hopeless on tightness: it would need 10⁴–10⁷ planes per ball (§4). |
| Contract | **Exclude** balls from the certified production population. **Keep** them as a supported experimental capability, including a label-free OOD evaluation set. **Defer** a certified curved labeller to a gated research pilot (§5). |

---

## 1. Physical correctness

### 1.1 What is implemented

**Representation.** A closed ball B(c, a) has a binary64 centre and radius. Both are read as exact dyadic rationals.
* The solid is a union with the certified convex polytopes.
* A receiver is SOLID iff it lies in a polytope or a ball.
* A receiver is response-active iff the closed segment [s, x] misses every polytope and every ball.
* Tangency is a hit (`closed-solid-closed-segment-v1`; `CURVED_OCCLUDERS.md` §1).
* Receivers (2v + 1)/(2N) and driver sources are binary64, hence exact.

**Implementation.**

| layer | file | method |
|---|---|---|
| exact predicates | `RIND3D_Generator/curved/rind3d_curved/spheres.py` (`segment_hits_ball_exact`, `ball_contains_exact`, `clearance_exact`) | Python `Fraction`s; division-free test: hit ⇔ C ≤ 0 or E ≤ 0 or (B < 0 and A + B > 0 and AC − B² ≤ 0) |
| CPU fast path | same file (`segment_hits_ball_interval`, `BallSet`) | outward-rounded binary64 intervals (round-to-nearest, then one `nextafter` outward); three-valued Kleene logic; UNKNOWN is recomputed exactly |
| GPU path | `rind3d_curved/cuda_balls.py` (`BallTiles`) | CuPy/NVRTC kernel on 64³ tiles; directed-rounding intrinsics `__dadd_rd/ru`, `__dmul_rd/ru`; `--fmad=false`; UNKNOWN voxels are flagged and recomputed exactly on the host |
| all-active certificate | `spheres.ball_misses_hull_exact`; `rind3d_100k/curved_scene._CurvedHull` | Frank–Wolfe float search for a separating direction n, then the exact check min over hull vertices of n·(c − p) > a·\|n\| |
| composite scene | `RIND3D_Dataset100K/rind3d_100k/curved_scene.py` (`CurvedLocalScene`) | polytope bits from the unchanged certified backends, combined as SOLID = OR and active = AND NOT; the v3 decision machinery is reused unchanged |
| generator and physical stage | `rind3d_curved/families.py` (`curved-spheres-v0`); `rind3d_100k/physical_curved.py`; contract `RIND3D_100K_CURVED_RESEARCH_v0.json` | families `spheres` (3–10 balls) and `spheres_polyhedra` (v3 `convex_polyhedra` plus 2–6 balls); r ~ LU(0.02, 0.12); exact source clearance; exact 6-connected nonempty voxel sampling |

### 1.2 Unit tests (`RIND3D_Dataset100K/tests/test_curved_spheres.py`)

All 12 passed in 1.30 s, under load, including the CUDA test.

| test | what it establishes | result |
|---|---|---|
| `test_interval_predicate_matches_exact_and_canonical_oracle[0-3]` | The Fraction predicate equals the independent canonical oracle `rind.ellipsoid3d.ExactEllipsoid3D`, which uses a different algorithm (clamped minimiser). The sample is 4 × 400 near-tangent receivers, 20–80% hits. Interval decisions are never wrong where certain; `BallSet` equals exact. | PASS |
| `test_exact_tangency_is_a_hit_on_every_path` | A segment tangent at a lattice row is a hit on the exact, interval and canonical paths; the next row is a miss. The CPU interval path returns UNKNOWN there, so the exact fallback is exercised. | PASS |
| `test_membership_and_clearance_exact` | Membership at a 2⁻⁴⁰ offset; clearance at 2⁻²⁷ / 2⁻²⁵ around ρ = 2⁻²⁶. | PASS |
| `test_ball_sampling_connectivity_matches_dense[0-2]` | The column-run connectivity check equals dense 6-connected labelling. | PASS |
| `test_hull_certificate_is_sound` | 200 random cases; Monte Carlo check of the hull points. The bound itself is a proof: n·(c − p) is affine in p, so its minimum over the hull is attained at a vertex. | PASS |
| `test_curved_generation_is_deterministic` | Identical balls and polytope ids from (seed, slot, attempt). | PASS |
| `test_gpu_tiles_equal_exact_cpu_including_near_tangent` | GPU tiles on the shadow boundary equal exact CPU decisions on 300 sampled voxels per tile. | PASS |

### 1.3 Additional checks run for this assessment

**C1. Stored containers against the canonical oracle** (`recheck_containers.py`, CPU, 34 s). Data: the 8 curved scene
containers of `RIND3D_100K_v3_work/curved_stats/`.
* *Sphere-only scenes (4).* Every stored bit was recomputed with `ExactEllipsoid3D` on a sample:
  - **9,863 channel receivers**, of which **9,327 lie on a stored direct response interface**, i.e. the near-tangent
    receivers;
  - **2,040 SOLID receivers**, uniform plus the SOLID boundary.
  - Result: **0 mismatches.**
* *Mixed scenes (4).* 5,279 necessary ball conditions were checked: a receiver stored active must miss every ball, and
  a receiver inside a ball must be stored SOLID. **0 failures.**
* This extends the 1,920-receiver end-to-end check of `CURVED_OCCLUDERS.md` §4 to all four sphere-only containers and
  to interface-concentrated samples.

**C2. Whole GPU tiles against the exact CPU path** (`gpu_checks.py full`). The unit test samples 300 voxels per tile;
here every voxel was compared.
* Sample: 24 tiles of 64³ centred on near-tangent receivers of 4 balls, in 2 generated `spheres` scenes with 3 legal
  sources each. That is **6,291,456 voxels**, 1,126,820 of them blocked.
* **0 blocked mismatches and 0 SOLID mismatches.**
* 0 GPU UNKNOWN voxels and 0 CPU exact fallbacks: both paths decided every voxel by intervals.
* The CPU and GPU paths evaluate the same division-free formula. C2 therefore validates the GPU arithmetic, not the
  formula. The formula's independent validation is C1 and the oracle tests.

**C3. Exact tangency on the GPU** (`gpu_checks.py tangent`). The tangency fixture of §1.2 was rendered as a whole tile.
* The GPU decided the tangent receiver as blocked and the next row as unblocked, with **0 UNKNOWN**. With these short
  dyadic inputs every directed-rounding operation is exact, so the interval collapses to D = 0, which is certified ≤ 0.
* The CPU interval path widens by one ulp and needs **64 exact fallbacks** on the tangent row.
* Whole tile against the CPU: 0 mismatches.

### 1.4 Findings and limitations

* **No error found** in the predicates, the interval paths, the composition or the stored data.
* **The exact fallback is not exercised in practice.** In every production-like run it was never needed: 0 UNKNOWN in
  the benchmark (22,257 ball tiles, 0 exact probe fallbacks) and in C2. It is exercised by the CPU unit test. On the
  GPU, exact tangency was decided without it (C3).
* **No floating-point environment self-test.** The CPU interval path assumes round-to-nearest binary64 without
  contraction or extended precision. `certify/ar.py` has a self-test for exactly this; `rind3d_curved` has none. This is
  low risk with numpy on x86-64, but it is a hardening item.
* **Stale docstring.** `physical_curved.py` says "there is no all-active hull certificate". The code enables
  `_CurvedHull` (polytope rows plus one exact separating plane per ball) whenever `hull_certificate` is true, as in the
  research contract. `CURVED_OCCLUDERS.md` §5 is right only literally: sphere-only scenes have no *polytope* hull
  certificate, but they do have the ball certificate.
* **Primitives.** Balls only, in all fast paths.
  - Ellipsoids exist only in the canonical oracle. The same division-free predicate extends to them by an affine map.
  - Cylinders and capsules are not implemented.

**Verdict on correctness.** The ball physics is exact by construction:
* certified intervals, with exact rationals where an interval is inconclusive;
* tested against an independent exact implementation;
* consistent end to end on the stored data.

It is fit for physical datasets, benchmarks and label-free evaluation.

---

## 2. Costs

### 2.1 Measured physical-stage cost

**Source.** The laptop per-family benchmark, `RIND3D_Dataset100K/docs/GENERATOR_BENCHMARK_V3.md`, data in
`RIND3D_100K_v3_work/bench/*.jsonl`.
* Hardware: RTX 3060 Laptop GPU and Ryzen 7 5800H.
* Each family ran in a fresh process with one warm-up scene, on an unloaded machine.
* Samples: 8 scenes per polytope family and 4 per curved family. The curved samples are small.

The source count k matters: cost is close to linear in k. The contract's law κ = (0.15, 0.15, 0.15, 0.55) gives
E[k] = 3.1. The column "at E[k]" uses the least-squares fit wall = a + b·k per group. In brackets is the ratio-of-means
estimate (mean wall / mean k) × E[k].

| group (scenes) | median / p95 / max s | mean s (mean k) | **at E[k] = 3.1** | render per 64³ tile | per probe | geometry stage | rate-stream share | peak RSS / GPU |
|---|---|---|---|---|---|---|---|---|
| 9 v3 polytope families (72) | 4.79 / 7.14 / 13.47 (family medians 3.25–6.96) | 4.71 (3.19) | **4.58 s** (4.57) | 0.34 ms | 1.4 µs | 0.09 s | 0.48 | 1.26–1.32 GB / 124 MB |
| `spheres` (4) | 28.96 / 51.57 / 54.67 | 31.42 (3.00) | **32.5 s** (32.5) | 5.97 ms | 24.8 µs | 3.16 s | 0.65 | 0.94 GB / 122 MB |
| `spheres_polyhedra` (4) | 23.08 / 27.60 / 27.72 | 22.79 (2.25) | **25.5 s** (31.4) | 4.70 ms | 17.4 µs | 2.81 s | 0.67 | 1.09 GB / 244 MB |

The curved/polytope ratio at E[k] is therefore **5.6–7.1×** per scene. The benchmark's own family medians give 5–9×.

### 2.2 Where the curved time goes

The new micro-benchmark is `gpu_checks.py micro`: a 9-ball `spheres` scene, 32 tiles per launch, under CPU load.

| balls | total per 64³ tile | kernel | device-to-host copy | host decode |
|---|---|---|---|---|
| 1 | 3.0 ms | 0.88 ms | 0.05 ms | 2.1 ms |
| 3 | 3.8 ms | 1.79 ms | 0.05 ms | 2.0 ms |
| 9 | 7.0 ms | **4.96 ms** (2.1 ns per voxel-ball test) | 0.05 ms | 2.0 ms |

**Correction to the benchmark's attribution.** `GENERATOR_BENCHMARK_V3.md` attributes "about 6 ms of host work per
64³ tile (transfer and bit combination)". The measurement shows otherwise:
* The certified kernel itself dominates for multi-ball scenes. It costs about 0.55 ms per ball per tile, because FP64
  directed-rounding interval arithmetic is slow on this FP64-weak laptop GPU.
* Host decoding costs about 2 ms per tile. The transfer is negligible.

**The other overheads.**
* The exact voxel-connectivity check costs about 3 s per scene (Python rational re-checks of every column end).
* Probes are 12–18× slower than polytope probes, because the ball path is CPU numpy intervals looped over balls.

**Available optimizations** (none implemented or measured):
* *Per-tile ball culling* with the existing exact separation test. Only balls meeting conv({s} ∪ tile box) can affect
  a tile.
* *Early exit* once a voxel is certainly blocked.
* *A certified FP32 filter* in front of the FP64 intervals.
* *GPU-side bit combination.*
* *A vectorized exact-integer connectivity check.*

Most of the physical overhead therefore looks like an implementation property, but it is not free. Until these optimizations are
measured, the measured 5.6–7.1× is the planning number.

### 2.3 Storage and memory

Container sizes were measured with `cost_stats.py`.

| containers | n | mean size | at k = 4 |
|---|---|---|---|
| v3 polytope (`sampling_impact/v3`) | 72 | 347 KB (121–1,003) | 389 KB |
| curved (`curved_stats`) | 8 | 247 KB (107–449) | 320 KB |

* A ball costs 32 bytes of geometry. The responses dominate, and they are stored in the same format.
* **Storage is unchanged** within sampling noise.
* **Memory.** Peak RSS is 0.94–1.09 GB, against 1.26–1.32 GB for polytope families. GPU memory is 122–244 MB: CuPy
  adds its own context, about 120 MB.
* **Labels.** A curved label would use the same per-cell format (47 KB per label, `LABEL_COST_100K.md`). Its
  certificate would be larger than a polytope certificate (§4.3), by an amount that is unknown.

### 2.4 Extrapolation to a 100K campaign

The campaign has N_s = 10⁵ scenes, 12 windows each and 1–4 sources (E[k] = 3.1). Let f be the curved fraction.

**Physical stage.**

    T_phys(f) = N_s · [(1 − f) · t_P + f · t_C] / 3600   laptop-GPU-worker-hours
    t_P = 4.58 s (polytope, at E[k]);  t_C = 29.0–31.9 s (equal mix of the two curved families; range of the two k-normalisations)

| f | T_phys | relative to f = 0 |
|---|---|---|
| 0% | 127 h | — |
| 10% | 195–203 h | +53–60% |
| 20% | 263–279 h | +107–120% |

**Storage.** S(f) = N_s · [(1 − f) · 347 KB + f · 247 KB]. That is 34.7 GB at 0%, 33.7 GB at 10% and 32.7 GB at 20%,
i.e. unchanged.

**Labels.**

    Λ(f) = Λ_P · [1 + f · (ρ − 1)]

* Λ_P is the polyhedral label work, and ρ is the cost ratio of a curved label to a polyhedral one.
* **ρ is undefined today, because no curved labeller exists.** With the pilot gate of §4.5 (ρ ≤ 3), f = 10% would add
  ≤ 20% and f = 20% would add ≤ 40%.
* With ρ = 30, which is plausible for elongated posteriors (§4.3), f = 10% adds 290% and f = 20% adds 580%.

**Scale.** The physical stage, about 127 laptop-GPU-hours, is small next to label work: the historical TPE basis was
about 33,400 laptop-thread-hours (`LABEL_COST_100K.md`), and FastLabel's production cost is measured in the canary.
**The decisive cost of curved scenes is the label, not the physics.**

**Informative channels per scene** (these samples): 19.9 for curved scenes (m = 1.66 per window) and 22.1 for polytope
scenes (m = 1.84).

**A label-free curved OOD set is cheap.** 1,000–5,000 curved scenes cost about 8–44 laptop-GPU-hours at 29–32 s per
scene.

---

## 3. Scientific value

### 3.1 What the v3 population already contains

Contract v3 has nine polyhedral families: `rotated_boxes`, `thin_slabs`, `prisms`, `wedges`, `convex_polyhedra`,
`multiscale_convex`, `corridors`, `barriers` and `occluder_clusters`.
* **The faceted families are not in v3.** `faceted_spheres` (Fibonacci polyhedra, 20/32/48 planes),
  `faceted_cylinders`, `rounded_boxes` and `curved_walls` exist only in `RIND3D_Dataset1K/rind3d_1k/families.py`.
* They were excluded from the 1K release on label cost (`GEOMETRY_DIVERSITY.md` §4).
* So the production population has no rounded body at all. The most rounded v3 body is a `convex_polyhedra` body with
  8–32 random-normal planes.

### 3.2 Faceted spheres are not observationally spheres at 1024³

**Setup** (`facet_gap.py`). The 1K `faceted_spheres` recipe puts every plane tangent to the ball, with the same radius
law LU(0.02, 0.12) as the curved family. The polyhedron therefore contains the ball. Its farthest vertex lies δ beyond
the ball, with δ/r ≈ 3.7/n for n planes (n ≥ 100).

| planes | max gap δ/r | δ in voxels at r = 20 / 50 / 123 voxels (min / median / max radius) |
|---|---|---|
| 20 | 0.221 | 4.5 / 11.1 / 27.2 |
| 32 | 0.129 | 2.6 / 6.5 / 15.9 |
| 48 | 0.083 | 1.7 / 4.2 / 10.2 |
| 200 | 0.019 | 0.4 / 0.9 / 2.3 |
| 800 | 0.0047 | 0.1 / 0.2 / 0.6 |

**Readings.**
* *At the catalogue facet counts the faceting is resolved.* The 20–48 planes deviate by 1.7–27 voxels at the body, and
  the shadow magnifies this for receivers beyond it.
* *A sub-voxel gap (≤ 0.5 voxel) needs 160, 380 and 920 planes* at r = 20, 50 and 123 voxels.
* *Even a sub-voxel gap is not observational equivalence.* Any shift of the interface flips the receivers inside the
  shift, and posterior supports are thin: their thinnest extents are 5×10⁻⁷–2×10⁻⁴ in the polytope fixtures (§4.3). A faceted body is therefore a different world, not an
  approximation. The 1K study states the same for its families ("the polyhedron IS the canonical object").

### 3.3 Measured differences of the response interface

| quantity | curved (`spheres` / `spheres_polyhedra`) | v3 polytopes | source |
|---|---|---|---|
| off-plane residual of interface components, p50 at L64 / L128 / L256 (voxels) | 0.37 / 0.46 / 0.49 and 0.29 / 0.40 / 0.47 | 0.21 overall (p90 0.27) | `CURVED_OCCLUDERS.md` §6 |
| rank of component normals 2 or 3 | 81% / 77% of channels | — | same |
| true source inside the heuristic planar band | 85/89, 69/70 | 54/54 | same |
| response-active fraction of informative channels, p10 / p50 / p90 | 0.12 / 0.90 / 1.00 (159 channels) | 0.05 / 0.76 / 1.00 (1,586 channels) | `cost_stats.py` |
| informative channels per window | 1.66 | 1.84 | same |

**Theory behind the numbers** (`LOCAL_INVERSE_GEOMETRY.md`).
* *Polytopes (Cor. 1a).* The interface consists of planes through the source and occluder edges. Creases and the ends
  of ruled pieces lie on lines through the source (Lemma 10).
* *Balls (Cor. 1b).* The interface is a right circular cone with apex at the source, half-angle asin(a/|s − c|), and no
  creases.
* *Curvature carries less apex information (Estimate 7).* It is second order: σ₃ = O(Δφ²) for an arc of angular
  extent Δφ.
* *Small patches look planar.* Curvature is invisible when ℓ² < 4hD tan α, for example for ℓ < 19 voxels at D = 0.5
  and tan α = 0.17. The residual growing with window side in the table above is this effect.

### 3.4 What Students could learn from curved vs faceted silhouettes

* **Faceted bodies offer first-order cues that smooth balls do not.**
  - Every facet edge gives an exact plane through the source.
  - Every crease gives a line through the source.
  - A Student trained only on polyhedra can learn to rely on creases and planar facets. Neither exists on a ball.
* **Smooth cones require a different inference: fitting the apex of a cone.**
  - The apex information is weaker and grows with patch size.
  - Planar decompositions of a cone inflate the rank: 77–81% of channels decompose into rank 2–3 normals. Treating
    those "planes" as independent overstates localization.
  - The planar band misses the true source in 4 of 89 and 1 of 70 channels.
  - A polyhedra-only Student is therefore at risk of **overconfidence on curved interfaces**. This is a calibration
    failure, not only a loss of accuracy.
* **Less shadow per channel.** Informative curved channels are more response-active (p50 0.90 vs 0.76), so each
  carries less evidence.
* **Prior evidence on rounded polyhedra is modest.** In the 1K study, multimodal posteriors stayed at 3–9% in every
  family, and the gain was described as "visual and geometric, not structural".

### 3.5 Verdict on scientific value

* **Balls add a genuinely new interface geometry,** which neither the v3 families nor low-facet surrogates provide:
  smooth developable cones, no creases, second-order apex information.
* **Its value for Student performance is unmeasured.** The cheapest way to measure it needs no posterior labels:
  - Score Students on a curved OOD set with the held-out one-hot NLL of the true source relative to the prior. This is
    the strictly proper primary OOD metric of `RIND3D_StudentResearch/F_OOD_BENCHMARK.md`.
  - Also score HPD coverage of the true source. That tests the overconfidence hypothesis directly.
* **Training on curved scenes is a different matter.** One-hot supervision is 2.2–2.6 nats worse than posterior
  supervision at 1K scale (`E_STUDENT_PILOT.md` §2.2), so useful training data would need certified labels (§4).

---

## 4. Certification

### 4.1 The predicate and its geometry

Write d = x − s and u = c − s. For endpoints outside the ball, closed-segment contact is

    0 < d·u < d·d   and   |d × u|² ≤ a² |d|²        (ASTRA_MIXED_INTERFACE.md)

This is identical to the generator's division-free test.
* With w = −u: B = −d·u, A + B = d·d − d·u, and AC − B² = |d|²(|u|² − a²) − (d·u)².
* By Lagrange's identity, the last expression is |d × u|² − a²|d|².

**Degrees.**
* For a fixed receiver x, put s = x + y. Then d × u = (c − x) × y, which is linear in y. The contact condition
  |(c − x) × y|² ≤ a²|y|² is quadratic in the source.
* Symmetrically, it is quadratic in x for a fixed s.
* Jointly it is degree 4, because AC and B² are products of two quadratics.

**The inverse shadow of a receiver.** C_x(B) = {s : [x, s] ∩ B ≠ ∅} is bounded by two pieces:
* the right circular cone with apex x, axis x → c and half-angle asin(a/|c − x|), beyond its contact circle;
* the far cap of the sphere.

Its boundary is not a polyhedron. So the posterior support F(G, X), and with it every cell mass, is bounded by pieces of
quadric cones and spheres, not by planes.

**Lemma (convexity).** For convex B and any point x, C_x(B) is convex.
* *Proof.* Let s₁, s₂ ∈ C_x(B) with bᵢ = x + tᵢ(sᵢ − x) ∈ B and tᵢ ∈ (0, 1]. (If some tᵢ = 0, then x ∈ B and
  C_x(B) = ℝ³.)
* For s_λ = (1 − λ)s₁ + λs₂, the point x + μ(s_λ − x) with μ = 1/((1 − λ)/t₁ + λ/t₂) ∈ (0, 1] equals
  (1 − ν)b₁ + νb₂ with ν = λt₁/((1 − λ)t₂ + λt₁) ∈ [0, 1].
* So [x, s_λ] meets [b₁, b₂] ⊆ B. ∎

**Corollary (box predicates for balls).** Let R be a polytope region and Q a receiver box.
* *COVER* — every segment from R to Q meets B — holds iff every (vertex of R, corner of Q) pair hits B.
  - Fix a vertex v. C_v(B) is convex in x, so it contains Q.
  - Then for every x ∈ Q, C_x(B) is convex and contains every vertex of R, so it contains R.
  - This is the same biconvexity that underlies the polytope verifier's "bilinear vertex × corner form". Each pair is
    a degree-4 sign test.
* *NOCOVER* — no segment from R to Q meets B — holds iff B ∩ conv(R ∪ Q) = ∅.
  - A separating direction checked exactly proves it (`ball_misses_hull_exact`, already implemented).
* *Clearance and legality.* The same two tests apply to the inflated ball B(c, a + 2⁻²⁶).

### 4.2 What a certified ball solver needs

**What transfers unchanged from the polyhedral verifier** (`RIND3D_Posterior/src/rind3d_posterior/certify/`):
* the proof-tree architecture, in which the labeller is untrusted and every claim is re-proved;
* exact plane-based polytope regions (`poly.py`) with rigorous volume enclosures;
* the 128³ cell split and the TV bound (`tvb.py`). It needs exactly lo_c ≤ v_c ≤ hi_c, plus a pooled undetermined
  mass;
* the component bitmask (≤ 62 components). The curved families have ≤ 15 components.

**New, in the verifier:**
1. *Ball item predicates.*
   - COVER by vertex × corner degree-4 sign tests, evaluated by interval arithmetic on the vertex enclosures.
     Undecided pairs mean UNDEC. That is sound: undecided volume never becomes certified volume.
   - NOCOVER by exact separation.
2. *Ball clearance in the legal-domain shell.* Use an outer polytope of the inflated ball within the region.
3. *Geometry decoding.* Decode `balls` from the container, and decode SOLID by exact membership.
4. *Labeller-named rational split planes* in the certificate. These are approximate tangent planes of the cones C_x(B).
   - Soundness does not depend on tangency, because any plane is a valid split. Only efficiency does.
   - The tree already materializes named planes with error bounds.

**New, in the labeller:** a curved-aware refinement strategy.
* The tau-pencil structure does not exist for balls. Constraint cones from different receivers have different apexes
  and share no axis. There is no τ-ordering along a pencil, and no "two extremes per edge".
* The labeller must bracket each curved boundary piece of F:
  - a tangent plane of the cone: C_x(B) is convex and lies on one side of it, so the outer side is NOCOVER for this
    ball and receiver;
  - an inward parallel plane, offset by more than the sagitta over the region: the inner piece is COVERed, proved by
    the vertex test, using convexity;
  - the slab between the two stays undetermined. Split it across the rulings until its mass fits the TV budget.

**Not needed:** volumes of regions bounded by quadrics. Every certified or undetermined leaf is a polytope.

### 4.3 Why the certificate is always a bound, and how tight it must be

**Exact masses are not rational.** Cell masses v_c = Vol(F ∩ c) for F bounded by circular cones are in general not
rational. They involve π and inverse trigonometric terms already for one cone cut by planes.
* So, unlike the polyhedral case, no finite exact certificate exists.
* Polytope leaves can never tile F. A boundary layer of undetermined volume remains at every depth.
* The certificate is the TV bound of `tvb.py`, which is > 0 except for single-cell supports. This is acceptable: the
  contract certifies at TV ≤ 0.05.

**Tightness.** Let F be locally a plate of width w, the thinnest extent of the support. A curved boundary piece of
lateral extent E, bracketed by a tangent plane and its offset, leaves a slab of thickness σ = E²/(8ρ_c) per face.
* ρ_c = ℓ·tan α is the cone's radius of curvature across its rulings, at distance ℓ from the apex.
* The cone is developable: it is curved only across the rulings.
* The bracketing gap gives TV ≈ 2σ/w. Meeting ε requires pieces of extent

      E_p ≤ √(4 ε w ρ_c),   with ε = 0.05.

**No curved F has ever been computed.** The polytope fixtures' F extents (`LOCALFIRST_E123.md` §4) serve as stand-ins,
with ρ_c = 0.05 (ℓ ≈ 0.3, tan α ≈ 0.17).

| stand-in support | w | lateral extents | E_p | pieces across rulings |
|---|---|---|---|---|
| rotated box (one interface plane) | 2.4e-6 | 7.6e-5 / 1.1e-4 | 1.5e-4 | 1 |
| two near-aligned boxes (tube) | 4.8e-7 | 1.2e-6 / 1.1e-5 | 6.9e-5 | 1 |
| knife-edge wedge | 1.3e-5 | 5.1e-4 / 1.2e-3 | 3.6e-4 | ≈ 4 |
| weak window (source 10⁻⁹ from a facet) | 1.9e-4 | 1.6e-3 / 2.2e-3 | 1.4e-3 | ≈ 2 |
| elongated support (hypothetical: spans ≈ 6 cells across rulings) | 1e-6 | 5e-2 | 1e-4 | ≈ 500 |

**Readings.**
* *Thanks to quadratic convergence, compact curved posteriors should need little refinement.*
* *Elongated supports drive the tail.* Examples are long slivers and curved tubes along intersections of two cones.
* *A second, unquantified driver.* F's boundary is the envelope of many receivers' cones, a "digital conic preimage".
  It has more pieces than the two τ-extremes of a straight edge. **This is the main unknown of the cost.**
* These figures are scaling estimates under stated assumptions, not measurements.

### 4.4 Alternatives compared

**(a) Global inner/outer polyhedral sandwich.** Use fixed polytopes P_in ⊆ B ⊆ P_out.
* *It is sound by monotonicity.* Define:
  - F_outer = {s legal for P_in : DARK receivers blocked by P_out, LIT receivers clear of P_in};
  - F_inner = {s legal for P_out : DARK blocked by P_in, LIT clear of P_out}.
  - The scene's polytopes enter both sets unchanged.
  - Then F_inner ⊆ F_B ⊆ F_outer, and `tvb.py` applies verbatim.
* *Its gap is far too large.* The gap is δ ≈ 3.7 r/n (§3.2). The required δ ≤ εw/2 means n ≥ 7.4 r/(ε w):
  - 3.8 × 10⁴ planes per ball for the weak window;
  - 3 × 10⁶ for the rotated-box support;
  - 1.5 × 10⁷ for the tube.
* *Such polytopes are also too expensive to verify.* The verifier's exact segment–polytope test evaluates all
  front/back pair rows, about n²/4 per (region, box) test (`check.comp_status`).
* *Measured on an analogous case.* The Model B relaxation has a gap of about one voxel. Its outer bound was 1.8% of the
  cube, against about 10⁻¹¹ for the exact bound, and 0 of its samples were truly feasible (`VOXEL_PHYSICS_DESIGN.md`
  §7, branch `yushu/3d-voxel-physics`).
* **Ruled out.**

**(a′) Adaptive local relaxation** (§4.2–4.3). The same sandwich idea, applied per region and per receiver constraint
with tangent and offset planes, refined until the TV budget is met.
* *Sound and deterministic.* It reuses the verifier architecture and the TV bound.
* *Needs new code:* the ball predicates in the verifier and a new labeller strategy.
* **It is the only credible deterministic route.**

**(a″) FastLabel with relaxations** (outer bound B ⊇ F, exact ball membership of samples). This is a statistical
certificate (`certified_stat`).
* It does not meet the deterministic production goal.
* Its acceptance ratio would be about w/δ for any global relaxation.
* **Not recommended.**

**(b) Faceted spheres as canonical physics.**
* *For:*
  - exact with the existing certified pipeline;
  - already implemented (1K `faceted_spheres`);
  - no approximation term, because the polyhedron is the world.
* *Against:*
  - not a sphere: a 1.7–27-voxel gap at 20–48 planes, with creases (§3.2);
  - label cost about 5× the v0.1 mean at 128³ for 32.7 planes on average (6.57 s vs about 1.3 s,
    `GEOMETRY_DIVERSITY.md`); not measured at 1024³ or with the deterministic verifier;
  - a sub-voxel gap needs 160–920 planes. That is about 6×10³–2×10⁵ pair rows per test, against ≤ 9 for a box,
    which is likely prohibitive (not measured).
* **A separate decision about rounded *polyhedra*, not a substitute for balls.**

**(c) Exclude balls from the production population.**
* *Cost:* none.
* *Loss:* the smooth-cone regime is absent from the training population. Curved scenes can still be evaluated
  label-free.

**(d) Voxel physics (Model B).** Voxelized balls behave like polytopes, and Theorem 10 gives an exact X-only region.
* There is no certified posterior labeller at production cost.
* Research only (`VOXEL_PHYSICS_DESIGN.md` §8).

| route | deterministic and sound | faithful to the smooth ball | new engineering | label-cost risk | status |
|---|---|---|---|---|---|
| (a) global sandwich | yes | yes, but TV ≈ 1 | small | infeasible (n ≥ 10⁴–10⁷) | ruled out |
| (a′) adaptive relaxation | yes | yes (TV ≤ 0.05 by refinement) | verifier extension + new labeller | unknown; tail from elongated supports | **pilot candidate** |
| (a″) FastLabel + relaxation | no (statistical) | yes | moderate | high (low acceptance) | not recommended |
| (b) faceted canonical | yes (exact) | no (a different world) | none (family exists) | ≈ 5× at 128³; prohibitive at sub-voxel facets | separate decision |
| (c) exclude | — | — | none | none | **recommended now** |
| (d) voxel physics | — | no (a different physics) | new engine | no labeller | research only |

### 4.5 Work and expected cost of route (a′)

These are estimates, not measurements. For scale, the polyhedral verifier package plus the certificate engine is
about 4,100 lines (`bece73b`).

| step | content | effort |
|---|---|---|
| P1 | Verifier ball items (COVER by vertex × corner, NOCOVER by separation, clearance shell, SOLID decode, mixed with polytopes); tests against `ExactEllipsoid3D` and Fraction oracles, including near-tangent regions | 1–2 engineer-weeks |
| P2 | Certificate format: labeller-named rational planes. Labeller: curved bracketing, splits across rulings, a TV-budget-driven stopping rule | 2–4 engineer-weeks |
| P3 | Pilot on 50–100 curved tasks, stratified by side and family (both curved families): TV bound distribution, label + verify seconds, node counts, tail | ≈ 1–40 CPU-hours, depending on ρ |
| P4 | Independent review of the verifier extension | ≈ 1 week |

**Gate, to be declared before P3 is measured:**
* ≥ 95% of tasks certified at TV ≤ 0.05 within the contract's timeouts (1800 / 3600 / 7200 s);
* a median cost ratio ρ ≤ 3 against polyhedral labels of the same side.

**Total.** About 5–8 engineer-weeks to a go/no-go. If the gate passes, the campaign label cost scales as
Λ(f) = Λ_P · [1 + f(ρ − 1)] (§2.4).

---

## 5. Recommendation for the production contract

1. **Exclude** continuous balls from the certified production population of RIND3D-100K v3 (T*_v3).
   * This is the status quo: contract v3 lists nine polyhedral families, and its labeller has no ball support.
   * No curved scene receives a production label, and no curved scene is mixed into T*.
   * The decisive reason is that no deterministic posterior certificate exists (§4). Physics quality and physical cost
     are not the reasons.
2. **Keep, as a supported experimental capability:**
   * the `rind3d_curved` package, `CurvedLocalScene`, `physical_curved`, and the research contract
     `rind3d-100k-curved-research-v0`, with the 12 curved tests kept in the test suite;
   * physical canaries and benchmarks, and X-only interface statistics;
   * **a label-free curved OOD evaluation set**, for example 1,000–5,000 scenes for 8–44 laptop-GPU-hours, scored by the
     true source's one-hot NLL relative to the prior and by HPD coverage. It tests the overconfidence hypothesis of
     §3.4 without labels.
   * the fixture set for a future labeller pilot.

   Not supported: posterior labels, mixing into T*, and planar-band pruning on curved interfaces.
3. **Defer** a certified curved labeller to the gated pilot of §4.5. Start it only after the deterministic polyhedral
   pipeline is in production, and only if the OOD evaluation shows that curved interfaces matter for Students.
4. **Do not adopt a global polyhedral sandwich** (§4.4 a), and do not use a statistical certificate for curved labels
   (a″).
5. **Treat faceted spheres and cylinders as a separate decision about polyhedral families.**
   * They are not a substitute for balls.
   * If they are considered, measure their label cost at 1024³ with the deterministic verifier before any contract
     change. The 1K study already recommended repeating its measurement at 1024³ before adoption.
6. **Housekeeping follow-ups** (no code was changed here):
   * correct the `physical_curved.py` docstring about the hull certificate;
   * correct the per-tile cost attribution in `GENERATOR_BENCHMARK_V3.md` (§2.2);
   * add the whole-tile GPU comparison (C2) as a test;
   * add a floating-point environment self-test to `rind3d_curved`, like `certify/ar.py`'s.

---

## 6. Provenance and reproduction

**Code read.**
* `RIND3D_Generator/curved/rind3d_curved/` (all modules);
* `RIND3D_Dataset100K/rind3d_100k/{curved_scene,physical_curved,local_eval}.py`;
* `RIND3D_Posterior/src/rind3d_posterior/certify/` (`check.py`, `tree.py`, `tvb.py`, `verify.py`);
* `RIND3D_Dataset1K/rind3d_1k/families.py`.

**Documents.**
* `docs/CURVED_OCCLUDERS.md`, `docs/LOCAL_INVERSE_GEOMETRY.md`, `RIND3D_Dataset100K/docs/GENERATOR_BENCHMARK_V3.md`,
  `LABEL_COST_100K.md`;
* `RIND3D_Posterior/docs/{LOCALFIRST_E123,LOCALFIRST_AUDIT,ASTRA_MIXED_INTERFACE}.md`;
* `RIND3D_Dataset1K/docs/GEOMETRY_DIVERSITY.md`;
* `RIND3D_StudentResearch/{E_STUDENT_PILOT,F_OOD_BENCHMARK}.md`;
* `VOXEL_PHYSICS_DESIGN.md` (branch `yushu/3d-voxel-physics`).

**Commits.**
* v3 curved work: `423e25e` (research layer), `caaf9cb` (note and tests), `0c60cb4` (benchmark), `1846525`
  (interface statistics);
* `bbac6ce` (the quadratic sphere constraint);
* `bece73b` (certified verifier).

**Data.** `/home/yushu/Projects/RIND/RIND3D_100K_v3_work/{bench, curved_stats, sampling_impact/v3}`, outside Git.

**Reproduce.** Use the environment of `RIND3D_Posterior/experiments/certify/env.sh`, with one process.

    python -m pytest -q RIND3D_Dataset100K/tests/test_curved_spheres.py                          # §1.2
    python RIND3D_Dataset100K/scripts/curved_assessment/recheck_containers.py $WORK/curved_stats   # C1
    python RIND3D_Dataset100K/scripts/curved_assessment/gpu_checks.py full                        # C2 (GPU, ~25 s)
    python RIND3D_Dataset100K/scripts/curved_assessment/gpu_checks.py tangent                     # C3
    python RIND3D_Dataset100K/scripts/curved_assessment/gpu_checks.py micro                       # §2.2
    python RIND3D_Dataset100K/scripts/curved_assessment/cost_stats.py $WORK                       # §2.1, 2.3, 2.4, 3.3
    python RIND3D_Dataset100K/scripts/curved_assessment/facet_gap.py                              # §3.2
