# RIND3D production decision: backbone, backend strategy, lifecycle cost and canary

**Status.**
- The contract `rind3d-production-contract-v0.2` is a candidate, frozen for the canary. It is not READY and not a release.
- RIND3D v0.1, the producer checkout, the Gaussian studies and the Teacher branch are unchanged.
- Nothing was committed or pushed. No 1K, 10K or 100K production was launched.

**Labels.** Every number is marked **[M]** measured on this machine (RTX 3060 Laptop 6 GiB, 16 logical CPUs, 31 GiB RAM), **[X]** extrapolated, or **[H]** hypothesis.

## 1. Decisions

1. **Core backbone.** The certified CSG convex-union core: the nine v0.1 generator families with exact binary64 convex predicates. It is published under two explicitly versioned profiles:
   - `rind3d-csg-single-binary-v0.2` (SB);
   - `rind3d-csg-multi-additive-v0.2` (MA).

   Gaussian readiness is not a prerequisite.
2. **Backend.** The labels stay with the certified analytic convex oracle:
   - the CUDA directed-binary64 raster for fields and dense windows;
   - certified CPU clipping for arbitrary source–receiver pairs;
   - an independent rational shadow projection for verification.

   Directly triangulated meshes plus BVH form an optional screening layer, never truth (§4).
3. **Materialization (strategy D, hybrid).**
   - Complete 1024³ fields per source are the canonical lossless observation.
   - Windows are index records into them.
   - Teacher hypothetical-source queries are always lazy and certified.
   - Only the multi-source public observations are materialized per window (§5).
4. **Observation policy.** A natural, unfiltered base distribution plus explicit, re-weightable informative, extreme, rare and nonlocal subsets (§7).
5. **Gaussian-derived geometry.** Deferred from the initial 100K core. It becomes a bounded supplementary or difficult-evaluation family only after a preregistered v3 calibration passes on fresh held-out seeds (§8).
6. **Scale.** The bounded canary was **accepted**: 216/216 scenes, all eight frozen gates passed. The next step, 1K, is prepared but not launched; it needs explicit authorisation (§11).

## 2. Stage A — architecture audit

**Already working, reused unchanged** (producer checkout `G2-02-3d/RIND3D_Generator`, build certificate `outputs/integration/scientific-certification.json`, scientific identity `e19b3004…`, re-validated at campaign creation):
- nine certified convex-union generators;
- certified CPU and CUDA predicates (parametric clipping with GMP fallback; independent Fourier–Motzkin shadow projection);
- source legality (exact FREE test, clearance > 2⁻²⁶);
- RINDPack v1 (64³ bricks, CRC and SHA), the information index and exact CountIndex;
- the two-replay physical verifier;
- v0.1's resumable campaign protocol.

v0.1 itself measured 332 READY scenes/h with complete fields.

**Missing for the new goals** (implemented here in `rind3d_production/`):
- a multi-source additive profile with private per-source channels and public per-window observations that do not leak a global palette;
- a natural-distribution observation policy that admits all-lit, all-shadow, SOLID-only and source-containing windows, plus explicit difficult subsets;
- shard-level publication with SHA256 manifests;
- a slot-level idempotent runner with worker-failure isolation and bounded retries;
- an independent validator;
- access-profile loaders;
- a Teacher physics interface with receiver-sampling probabilities;
- a lifecycle cost model.

**Judged scientifically unnecessary:**
- a new renderer: the certified CUDA raster renders a 64³ window in 0.4–2 ms and a complete 1024³ field in 3.2 s **[M]**;
- a voxel or marching-cubes path for CSG: direct face triangulation takes 0.8 ms **[M]**, where it is needed at all.

## 3. Production trade-off: core, supplementary, Gaussian

| | CSG certified core (this canary) | Gaussian mesh, pilot generator | Gaussian mesh, additive generator |
|---|---|---|---|
| Truth | exact binary64 convex union | float32 mesh (extracted from the field) | float32 mesh |
| Complete 1024³ field per source | **3.2 s** (p10–p90 2.6–3.5) **[M]** | 851–949 s **[M, devcal]** | not measured (≥ pilot **[H]**) |
| Mesh/BVH build | 0.8 ms triangulation + 0.17 ms Embree, optional **[M]** | 26 s mesh stage **[M]** | 120 s mesh stage, 1.3 M faces **[M]** |
| Complete scene | 1024³ fields + 102 windows + verification: SB 8.8 s, MA 23–38 s (K = 2–4) **[M, canary]** | 256³ only: 48.8 s **[M]** | 256³ only: ~150–175 s **[X]** |
| Peak memory | ≤ 2 GiB producer (v0.1) **[M]** | ≤ 9.8 GiB **[M]** | 17.4 GiB in one 1023 rung **[M]** |
| Yield | v0.1: 1000/1000 (3 source resamples) **[M]**; canary 216/216 **[M]** | 26/28 development, 28/28 held-out **[M]** | 26/28 development **[M]** |
| Occupancy | family-defined: SOLID median 0.0087, p90 0.057, max 0.18 **[M, canary]** | 0.63 / 0.15 of target at 0.08 / 0.15 **[M]** | 1.01 / 1.03 of target **[M]** |
| Components / genus | 1–24 convex components | median 2, genus 0 **[M]** | median 5, genus > 0 in 5/42 **[M]** |
| Grazing label risk | exact, two independent predicates | probes catch the mesh family's common-mode errors; band detector not admissible **[M]** | same |

**Scientific utility.**
- **v0.1 nonlocal audit:** first blockers lay outside the window for 1,753 of 1,999 sampled shadow receivers; 1,535 were external-necessary **[M, v0.1]**.
- **Canary:** 2,515 nonlocal windows were found across all sides; 97% of the nonlocal quota was filled (§10).
- **Diversity:** CSG covers thin slabs (half-thickness down to 2⁻²⁰), corridors, barriers, clusters and multiscale unions.
- **What Gaussian geometry adds:** smooth curved, multi-component and toroidal shapes. This is the difficult-subset or OOD value. It costs 250–300× more per complete 1024³ field.
- **Conclusion:** a visually richer scene is not automatically more valuable for nonlocal, inverse-source reasoning.

## 4. Shared Mesh/BVH acceleration layer: when it pays, when it must not be used

**Three layers are kept separate:**
1. generator (`lite_generators`);
2. canonical mathematical geometry (`CanonicalBundle`);
3. physical-query acceleration (certified convex predicates; optional float32 triangulation plus BVH).

**Medians over 18 fresh scenes,** 2 per family, benchmark seed 20261201, 64³ windows; first figure random windows, second near-geometry windows. All **[M]**:

| Backend (same rays) | Setup | Throughput | Disagreement with certified A |
|---|---|---|---|
| A: certified CPU clipping | context 7 ms, per-source bind 0.26 ms | 1.5 / 1.2 M rays/s per thread; Teacher batches 9.1 / 8.2 M pairs/s on 16 threads | reference |
| B: independent projection | – | 3 sources × 262 k rays in 0.66–0.83 s | 0 |
| R: certified CPU raster, 64³ | – | 0.8 / 36 ms per window | 0 |
| G: certified CUDA raster, 64³ | 8 ms per source bind | 0.4 / 2.0 ms per window | 0 |
| E: triangulated mesh, Embree | 0.8 ms triangulation + 0.17 ms build | 73 / 22 M rays/s | 1 ray in 4.7 M |
| W: triangulated mesh, Warp GPU | 0.33 ms build | 258 / 198 M rays/s | 1 ray in 4.7 M |
| H: E + W + ±2⁻²⁰ probes + certified arbitration | +45 / 35 ms per 262 k rays | ≈ 5 M rays/s | 0 |

**Grazing** (26,735 silhouette rays at signed gaps of 10⁻¹²–10⁻⁵, nine families) **[M]**:
- the float32 mesh disagrees with canonical truth on 16% of rays (Embree 4,251; Warp 4,181; common-mode 2,375);
- every disagreement lies below a 10⁻⁷ gap, the float32 vertex-rounding scale (maximum vertex plane residual 3.2·10⁻⁸);
- the hybrid H was exact on all 26,735 rays.

**Findings.**
- **BVH setup:** amortizes immediately for CSG (under 1 ms), so setup cost is not the obstacle.
- **Exactness:** float32 triangulation changes the geometry, so mesh answers are not labels.
- **The exact hybrid:** costs about 5 M rays/s, slower than certified CSG on 16 threads (8–9 M pairs/s).

**Decision.**
- Certified analytic CSG stays the label path.
- The mesh/BVH layer is useful only as an opt-in screen, for example Warp at about 2·10⁸ rays/s, roughly 25× certified CPU, for ranking Teacher candidates. Finalists must be re-evaluated with certified queries.

**Further cautions.**
- **Overlapping components:** their surfaces are valid for any-hit occlusion between FREE points, never for inside/outside parity.
- **Gaussian family:** the mesh is the truth by definition of that family. Labels there need dual engines plus the full translation probes. The band detector remains inadmissible.

## 5. Dataset responses versus Teacher physics: materialization strategies A–D

**Window access** (nine fields, 8–4 windows per side, medians) **[M]**:

| Window | A: read stored field (cold / warm) | C: lazy certified CUDA render (warm) | B: read materialized window | B bytes |
|---|---|---|---|---|
| 64³ | 3.0 / 0.85 ms | 0.76 ms | 0.06 ms | 99 B |
| 128³ | 8.1 / 3.8 ms | 3.0 ms | 0.30 ms | 562 B |
| 256³ | 38 / 23 ms | 27 ms | 4.0 ms | 10 KB |

- Every lazily rendered window was bit-identical to the stored field (27 of 27).
- Complete field: 3.2 s and 0.51 MB response + 0.32 MB SOLID + 0.08 MB information index + 0.05 MB counts per source.

**Crossover.**
- **Compute:** for CSG, lazy rendering costs about the same as decoding a stored window. Compute therefore never forces materialization.
- **Storage:** small, about 1 MB per source.
- **Why still store the fields:** they buy GPU-free, producer-free, verifiable reading and exact window statistics (CountIndex 0.1 ms per query).
- **Teacher hypothetical sources:** need lazy physics regardless of what is stored. Certified CPU gives 8–9 M pairs/s; a dense 64³ full-window cost is 8 ms per candidate on CUDA against 0.22 s on one CPU thread.
- **Multi-source public windows:** are materialized (B) because their public form must not expose the private per-source channels. This dominates MA storage (§10).
- **When full materialization would not pay [H]:** if Student training ran on GPU producer nodes and storage were scarce, C would replace A. At 100K scenes A costs on the order of 100–400 GB (§11), so no such pressure exists.

## 6. Data contract and leakage

**Specification.** `RIND3D_PRODUCTION_CONTRACT.json` (frozen copy and SHA256 in the canary root) separates:
- canonical geometry, optional acceleration geometry and true sources;
- the response semantics of each profile and the local-observation definition;
- Student-visible and privileged Teacher fields;
- sampling metadata, the oracle version and numerical precision.

**Profiles.** SB (binary, unit strength; v0.1-compatible) and MA (K ∈ {2,3,4}, strengths U[0.1,1], R = Σ a_k lit_k in channel order) are never merged. Derived variants carry their own names.

**Student inputs never contain:**
- sources, strengths or K;
- per-source channels;
- a scene-level palette;
- subset tags, counts or nonlocal estimates;
- other windows;
- candidates or seeds.

**Fixes for the leaks the Gaussian study identified:**
- MA public observations use a local palette per window (validated: no value absent from the window).
- No global multi-source response is public.
- Student sparse views are response-independent (uniform) only.
- Response-dependent stratified banks are Teacher-only.

**Declared limitation.** The SB public `response.rindb` is the complete field, as in v0.1. A strictly local release must export windows only.

## 7. Observation policy

Per scene and per side (64, 128, 256):

| Subset | Windows | Rule |
|---|---|---|
| natural | 16 | uniform, unfiltered |
| informative | 8 | mixed, no source inside the window |
| extreme | 4 | minority ≤ 1% |
| rare | 2 | minority ≤ 64 receivers |
| nonlocal | 4 | ≥ 90% of sampled shadow receivers have their first blocker outside the window |

- All windows are index records with proposal indices, so inclusion designs are reproducible.
- Shortfalls are recorded, never filled.
- The complete local response stays available. Sparse receivers are generated with stored coordinates, inclusion probabilities and Horvitz–Thompson weights.
- The FREE-receiver mismatch is unchanged by any sampling design.

## 8. Gaussian generator: bounded production investigation

Cap: about 30 min CPU, no new meshing. Evidence: `outputs/gaussian/gaussian-investigation.json`.

**Dominant costs** (devcal additive, first rung, medians) **[M]**:
- fp64 sub-cell fidelity probe: 35 s (pilot: 10 s);
- facet diagnostic: 12.3 s;
- self-intersection gate: 6.8 s;
- combinatorics: 5.0 s;
- field sampling: 4.6 s;
- generation: 4.9 s;
- faces: 0.74 M; RSS: 4.4 GiB.

**Local support** **[M]**: an exact per-kernel cutoff below binary64 resolution of τ (2⁻⁵³ τ) leaves only 11–55% of kernel evaluations. A culled sampler would cut the dominant probe 2–9× **[X, not implemented]**, giving an estimated 30–40 s mesh stage. Even then a complete 1024³ field costs 850–950 s **[M]**, against 3.2 s for CSG.

**The anisotropy metric.**
- **What E5 penalised:** a multi-component space-filling layout. Per-component anisotropy was kept (0.94×); components, genus and shadow complexity rose.
- **Is global anisotropy needed:** not obviously. Because this rationale comes from already-observed data, a replacement metric is preregistered (`configs/gaussian_v3_preregistration.json`):
  - per-component aspect;
  - a KS distance between component-aspect distributions;
  - global aspect only within densities where the reference meets occupancy;
  - shadow complexity.

  It is to be judged on fresh seeds. devcal-v1's decision and failed H4 stand.

**Decision.**
- Gaussian is not in the initial 100K backbone.
- It is a candidate supplementary or difficult-evaluation family (proposed ≤ 5% of scenes) after a preregistered v3 calibration passes on fresh held-out seeds.
- It needs the culled sampler and a versioned mesher fix for the Lewiner interior-vertex exception.

## 9. Production infrastructure

**Identity and splits.**
- Slots are deterministic (family = slot mod 9; profile and K by schedule); every random stream is keyed by (seed, slot, stream).
- `scene_id` binds profile, geometry and source bytes.
- Splits are geometry-group (v0.1 hash).

**Jobs.**
- **Isolation:** a worker subprocess handles one shard; a dead worker or a per-scene timeout costs one attempt (at most 2). FAILED slots are recorded, never replaced.
- **Journal:** fsynced.
- **Resume:** moves partial attempts aside and re-verifies every sealed scene by SHA256 before continuing.
- **Freeze:** campaign config and contract are frozen at `create`; a changed SHA refuses to run.

**Publication.**
- Scenes are sealed with SHA256 of every file.
- Shards (24 scenes) are published by atomic rename with `MANIFEST.json`; published shards are immutable.

**Validation and accounting.**
- An independent validator (V1–V11) runs as a separate process.
- Accounting covers wall/CPU per stage, worker RSS and GPU memory sampled every 5 s.

**File counts.** About 19 files per SB scene and 33 per MA scene; RINDPack files are 0.2–0.8 MB. At 100K scenes that is 2–3 M files in about 4,200 shard directories **[X]**. For distribution, tar or zip each published shard (a single object per shard). A shard-container RINDPack reader with offsets is a possible later optimisation, not needed at canary scale.

**Execution resources.**
- **Local RTX 3060 laptop:** one producer, measured below.
- **CPU-only workers:** the certified CPU route takes 8.2 s per field on 4 threads **[M]**, so high-core CPU nodes are viable without GPUs.
- **Compatible GPU workers (sm_70+, NVRTC):** need a native build and their own certification.
- **Colab:** suitable for bounded experiments only. Production integrity cannot depend on an interactive runtime, because campaigns need durable storage, journals and leases.

No cloud production was started.

## 10. Canary (Stage D)

**Verdict: canary ACCEPTED.** The frozen gates C1–C8 were evaluated mechanically by `canary_analysis.py` against the frozen contract (`outputs/canary/contract.json`).

| Gate | Result | Evidence |
|---|---|---|
| C1 yield | pass | 216/216 slots published; unpublished: none |
| C2 validation | pass | independent validator: 216 scenes, all checks {'V3_geometry': 216, 'V4_sources': 216, 'V5_fields': 216, 'V6_windows': 216, 'V7_independent_rerender': 216, 'V8_multi_observations': 108, 'V9_public_private': 216} |
| C3 physics | pass | 132188 bricks replayed by both independent predicates at generation, all bits equal; validation-time independent re-render of 4 random bricks per source equal |
| C4 windows | pass | natural 100% filled: True; informative shortfall 0; natural all-shadow SB windows: 733; per side {"64": {"all_lit": 1108, "mixed": 847, "all_shadow": 334, "extreme": 864, "rare": 432, "nonlocal": 833}, "128": {"all_lit": 930, "mixed": 1490, "all_shadow": 256, "extreme": 864, "rare": 432, "nonlocal": 854}, "256": {"all_lit": 532, "mixed": 2425, "all_shadow": 143, "extreme": 864, "rare": 432, "nonlocal": 828}} |
| C5 leakage | pass | tests: 9 passed in 1.17s; public/private separation True; MA local palettes 108/108 scenes |
| C6 recovery | pass | SIGKILL of runner + worker at 2026-11-09T22:51:50 during slot 50; partial attempt moved aside: ['failed-attempts/interrupted-000050-0']; duplicate sealed slots: none; determinism re-runs identical: [True, True] |
| C7 resources | pass | max scene wall 94.3 s; max worker RSS 721 MiB; max GPU memory in use 684 MiB; canary wall 4.69e+03 s |
| C8 usability | pass | SB 64^3 student random access 156 windows/s; DataLoader {'windows': 256, 'seconds': 10.641192480001337, 'torch': '2.15.0+cpu'}; Teacher queries equal stored: True |

**Scenes.** 216 published: families {'barriers': 24, 'prisms': 24, 'convex_polyhedra': 24, 'rotated_boxes': 24, 'thin_slabs': 24, 'multiscale_convex': 24, 'wedges': 24, 'corridors': 24, 'occluder_clusters': 24}; profiles {'SB-K1': 108, 'MA-K2': 36, 'MA-K3': 36, 'MA-K4': 36}; splits {'train': 177, 'val': 20, 'test': 19}; 21955 window records; 0.78 GB.

| Profile | Scenes | Wall s median (p95) | Bytes median | Stage medians (s) |
|---|---|---|---|---|
| MA-K2 | 36 | 23 (42.5) | 4.34 MB | count_index 0.19, fields 6.2, geometry_and_acceptance 0.048, independent_verification 6.8, publication_files 7.7, sources 0.00052, windows 2 |
| MA-K3 | 36 | 30.9 (58.3) | 6.36 MB | count_index 0.28, fields 9.6, geometry_and_acceptance 0.051, independent_verification 9.1, publication_files 9.6, sources 0.00079, windows 2.1 |
| MA-K4 | 36 | 37.7 (60.8) | 8.12 MB | count_index 0.39, fields 12, geometry_and_acceptance 0.039, independent_verification 10, publication_files 12, sources 0.00097, windows 2.5 |
| SB | 108 | 8.77 (16.3) | 1.10 MB | count_index 0.093, fields 3.1, geometry_and_acceptance 0.049, independent_verification 3.5, publication_files 0.0044, sources 0.00031, windows 1.7 |

**Natural observation distribution** [M]: {'SB/all_shadow': 733, 'SB/mixed': 1871, 'SB/all_lit': 2570, 'MA/constant_response': 2282, 'MA/mixed': 2891, 'SB/no_free': 10, 'MA/no_free': 11}. SB natural lit-ratio quantiles {'0.1': 0.0, '0.25': 0.3960522562265396, '0.5': 0.9999940394197635, '0.75': 1.0, '0.9': 1.0}. Subset shortfalls {'extreme': 0, 'informative': 0, 'natural': 0, 'nonlocal': 77, 'rare': 0} of quotas {'extreme': 2592, 'informative': 5184, 'natural': 10368, 'nonlocal': 2592, 'rare': 1296}. MA non-natural windows with ≥1/≥2/≥3/all informative sources: {'informative_any': 5791, 'informative_ge2': 3210, 'informative_ge3': 1303, 'informative_all': 1878}.

**Diversity** [M]: SOLID fraction quantiles {'0.1': 0.0014507784508168697, '0.5': 0.00871308147907257, '0.9': 0.05687275202944875} (max 0.184); per family medians: barriers solid 0.0127, components 8.5; prisms solid 0.00548, components 4.5; convex_polyhedra solid 0.0037, components 1; rotated_boxes solid 0.0252, components 6; thin_slabs solid 0.00117, components 6; multiscale_convex solid 0.0944, components 12; wedges solid 0.00574, components 3; corridors solid 0.00815, components 6; occluder_clusters solid 0.0131, components 10.5.

**Throughput** [M]: 166 published scenes/hour over all sessions (wall 4.69e+03 s, killed session counted to its last journal event — a lower bound); median scene 17.8 s.

## 11. Scale decision (Stage E)

All figures below are **[X]** extrapolations from the measured canary rate and mix on one RTX 3060 laptop; they assume linear scaling.

| Scale | Producer hours (1 laptop GPU) | ~4 GPU workers | Storage | Files | Window records | Validation CPU-hours (1 process) |
|---|---|---|---|---|---|---|
| 1,000 | 6.03 | 1.51 | 3.62 GB | 28,000 | 101,643 | 2.17 |
| 10,000 | 60.3 | 15.1 | 36.2 GB | 280,000 | 1,016,435 | 21.7 |
| 100,000 | 603 | 151 | 362 GB | 2,800,000 | 10,164,351 | 217 |

**Failure risks.**
- Yield and recovery are measured only at 216 scenes.
- Multi-source cost grows roughly linearly in K.
- Storage is dominated by the multi-source public observation payloads and the private per-source fields.
- One producer GPU is a single point of throughput, not of integrity: a crash costs at most the in-flight scene.
- **Before any public release:**
  - the master seed must be private, otherwise hidden labels are regenerable from code plus seed. `seed_file` support was added to the runner after the canary (non-scientific code) and smoke-tested: the frozen record stores only the file's SHA256, and runs are refused if the file is missing or changed;
  - a release must not distribute the private `src*/` channels.

**Next approved scale: 1K. Commands are provided, not executed:**

```bash
cd /home/yushu/Projects/RIND/G2-02-gaussian/RIND3D_Production
# 1. write a private 256-bit master seed (never committed or published) and set its path as seed_file in the config
python3 -c "import secrets; print(secrets.token_hex(32))" > /secure/rind3d-v0.2-1k.seed && chmod 600 /secure/rind3d-v0.2-1k.seed
./run.sh rind3d_production.runner create /data/rind3d-v0.2-1k configs/production-1k-v0.2.json
./run.sh rind3d_production.runner run /data/rind3d-v0.2-1k --max-runtime 43200
./run.sh rind3d_production.runner status /data/rind3d-v0.2-1k
./run.sh rind3d_production.validate /data/rind3d-v0.2-1k --determinism 7 501
RIND3D_TEST_ROOTS=/data/rind3d-v0.2-1k ./run.sh pytest -q -p no:cacheprovider tests
```

10K and 100K would use the same commands with a new versioned root and `slots`. They must not be launched without a separate authorization and an approved storage and compute budget.

## 12. What remains a hypothesis

- Linear scaling to 100K: no catalogue-growth, disk or thermal effects were measured beyond the canary.
- Equivalent-source enrichment for rare-mode studies: not part of the canary.
- The value of Gaussian geometry as an OOD benchmark: argued, not measured.
- Unverified bricks: the complete field is certified by the primary renderer, while independent replays cover the verification bricks and the validation sample. Unverified bricks rely on the primary certification, as in v0.1's coverage of window bricks only.
