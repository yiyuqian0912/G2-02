# RIND3D implementation facts (source-grounded reference for the Chinese handoff document)

Branch `yushu/3d`, consolidated on 2026-11-13. Paths are relative to the repository root. Numbers are measurements unless
marked **[proj]** (projection) or **[derived]** (read-only aggregation of release files done for this sheet).

**Status codes:**
* **V**: implemented and validated;
* **P**: implemented, partially validated;
* **D**: designed or proposed, not implemented;
* **U**: unresolved.

**Abbreviations:**
* `G` = `RIND3D_Generator/src/rind3d`
* `N` = `RIND3D_Generator/native`
* `D1K` = `RIND3D_Dataset1K/rind3d_1k`
* `FL` = `TPE/fastlabel` (FastLabel core; until 2026-11-13 `D1K/fastlabel`, which is now a compatibility layer; see `RIND3D_Posterior/docs/FASTLABEL_CORE.md`)
* `TPE` = `RIND3D_Posterior/src/rind3d_posterior`
* `SR` = `RIND3D_StudentResearch`
* `100K` = `RIND3D_Dataset100K/rind3d_100k` (on `yushu/3d` since 2026-11-14; historical branch `yushu/3d-100k-h20`). Facts marked "100K" below describe contract v1 unless they say v2; the v2 law is in `100K/sampling.py` and `RIND3D_Dataset100K/contract/RIND3D_100K_CONTRACT_v2.json`
* `REL` = `RIND3D_1K_data/release_1k_v1` (outside Git)

**Hardware of every measurement:** one laptop with an RTX 3060 Laptop GPU (6 GB, sm_86), a Ryzen 7 5800H (8C/16T) and 31 GB RAM. **Nothing has been measured on the 16×H20 target.**

---

## 0. Answers to the key questions

| question | answer | status | evidence |
|---|---|---|---|
| Does the final 100K pipeline generate full 1024³ responses? | **No.** The 100K renders only local evidence:<br>• exact probe points;<br>• exact rational hull certificates (all-lit proofs);<br>• certified 64³ CUDA tiles with early stop during selection;<br>• the full crops of accepted windows.<br>Geometry is global. The 1K reference did render a full 1024³ field per source (3,110 fields, 2.53 s each), and those stored fields are why the derived NAT-1K windows could be generated later | V (local pipeline); equivalence to full-field selection documented on 12 scenes, harness not committed | `100K/local_eval.py:3-7,30,167-261`; `RIND3D_Dataset100K/docs/LOCAL_RENDERING_100K.md:32-37`; `G/rendering/stream.py:26-101`; smoke check 1 (`RIND3D_HANDOFF/VALIDATION_RECORD.md`) |
| What geometry representation does forward visibility use? | A finite union of closed, bounded convex polyhedra (CanonicalBundle; leaves `affine_box_v1` or `convex_halfspaces_v1`). Coefficients are binary64 and interpreted **exactly** as dyadic rationals. The renderer works with exact face rows a·x + d ≤ 0, held as binary64 interval enclosures on the GPU and GMP rationals on the CPU. The labeller receives float64-rounded rows (`geometry_rows.npz`) | V | `G/rendering/polyhedra.py:38-67`; `N/convex.cpp:15,74-99`; `D1K/physical.py:92-102` |
| Window proposal and acceptance? | Per side L, i.i.d. uniform origins o ∈ {0..N−L}³. Admissible iff A = no source in the closed box ∧ FREE > 0 ∧ m ≥ 1. Accepted iff A ∧ U < c(m, L, k) (1K and 100K contract v1: m-balanced c-tables from an adaptive controller). **100K contract v2:** natural acceptance (c = 1); optional coverage stream (L64, K = 4, c(m) = 1/8, 1/4, 1/2, 1). Quota 16/24/24 (1K) or 4/4/4 (100K), cap 6000. A separate rate stream estimates Λ_L (§3.2) | V | `D1K/physical.py:121-175`; `100K/physical.py:46-101`; `D1K/controller.py` |
| Canonical storage? | **1K:** one directory per scene. Packed-bit zstd crops in `public/observations.bin` (SOLID crop once plus one crop per source), RINDPack full fields, sha256 `SEAL.json`, one npz per label task.<br>**100K:** one uncompressed npz container per scene (same crop encoding) plus an append-only label journal (R3LJ) with per-record sha256 | V | §5 |
| Measured costs? | **1K physical stage:** 24.6 s per scene (full fields, sequential).<br>**100K physical:** 3.7 s per scene (local).<br>**FastLabel:** mean 1.44 s worker-wall per label, 10.2 labels/s on the laptop.<br>**TPE on the 100K canary:** ≈ 41 warm CPU-s per label.<br>**Storage:** 1K ≈ 4.4 MB physical + ≈ 15.5 MB labels per scene; 100K ≈ 0.36 MB per scene + 47 KB per TPE label | V (laptop) | §5, §6, §8 |
| How are posterior probabilities discretised? | Cell masses on the 128³ grid, cell = (ix·128 + iy)·128 + iz with i = min(⌊128 s⌋, 127). The target is q_F(c) = vol(F ∩ c) / vol(F) | V | `TPE/api.py:9,27-39`; `D1K/loader.py:cell_of` |
| What does the TV certificate guarantee? | **`certified`:** deterministic TV(q̂, q_F) ≤ tv_bound ≤ 0.05.<br>**`certified_stat`:** TV(q_B, q_F) ≤ 0.05 except with probability 6.26·10⁻⁸ per label, union-bounded to ≤ 0.01 over all 159,802 planned 1K labels.<br>Both assume the engine's exclusions are sound (F ⊆ B) | V (math audited) | §6.3 |
| Why do solid-separated windows have a long labelling tail? | **Measured:** about 13 s vs about 1 s per label; 2.5% of the labels take ≈ 23% of the compute.<br>**Code facts:** no lit/dark boundary receivers exist, so phase 1 starts from 128 interior receivers only, and the single-cell exit never applies.<br>**[derived] mechanism:** 72–78% of the time goes to phase 1 (BSP and emission) over a very broad B (6–8·10⁵ pieces, 3–6·10⁵ cells), not to extra certification rounds.<br>These windows **do** carry decisive information (2–4× prior mass on the feasible region in the pilot). The difficulty is narrowing and emitting their large feasible region efficiently | V (measured), mechanism [derived] | §4.3 |
| All-lit / all-dark support? | Physically valid channels, stored, not labelled in production.<br>**All-lit:** certifies with v3.4 (180/180), about 25× the cost of a mixed label.<br>**All-dark:** fails in v3.4 (9/18) because B keeps solid interiors. The S6 research prototype certifies 18/18 and 180/180, at about 80–100× the cost | V (measurement); S6 = P | §4 |
| Loader training population and weights? | `flatten(ds, population=...)` requires an explicit declaration: B1K, Tstar or Tstar_mis. The default estimator is ω̂ = combined_rb. Every certified status is accepted. The one-hot fallback is explicit | V | §7.1 |
| Student loss? | Forward cross-entropy of the certified label field: Σ_c q_G(c) log p_θ(c). Examples are drawn ∝ the T* balance-heuristic weight. One-hot NLL is the baseline | V (pilot scale) | §7.2 |
| Is a neural Teacher still needed? | **No neural Teacher is in the current pipeline.** Direct certified-posterior supervision is the preferred approach. The 2026-11-10 study found direct NLL better than the Gibbs-Teacher targets it tested. The 13-arm pilot shows certified fields beat one-hot by 2.2–2.6 nats. A learned Teacher would only matter as a label-cost amortiser or for channels FastLabel cannot label affordably (extremes). That is an open option, not a component | V (decision evidence); Teacher role = U | §7.3 |
| What is unverified on 16×H20? | Everything at scale:<br>• environment and NVRTC build;<br>• cross-device bit-exactness;<br>• FP64 throughput of the directed-rounding renderer ("H20 FP64 reported low");<br>• TPE per-thread speed;<br>• multi-GPU scaling, RAM per worker, storage I/O;<br>• the real label tail;<br>• FastLabel at all: it is not integrated into the 100K branch | U | `RIND3D_Dataset100K/handoff/README_H20.md:3`; `docs/GPU_ARCHITECTURE_100K.md:109-112`; `100K_READINESS.md` |

---

## 1. Research objective

**Primary task T\*** (`SR/A_TASK_DEFINITION.md`).
* Worlds: natural multi-source physical worlds. G ~ π_f over 9 families; K ~ κ independent of G and S; S_i i.i.d. Uniform(D(G)); one source-separated binary response field per source.
* Windows: natural admissible windows.
* Examples: every mixed channel of such a window is one example.
* Input X = (SOLID ∩ W, r_j = R_j | FREE(W), L, origin).
* Output: P(S_j ∈ cell | X) on the 128³ grid, outside the closed window box.

**Optimum.** The population-optimal Student is E_T*[q_G | X], a mixture over the hidden worlds compatible with X. Its loss floor is H(S | X) > 0 (`RIND3D_Audit/04_LEARNING_THEORY.md`).

**Multi-source structure** (`A_TASK_DEFINITION.md` §3).
* Under natural window sampling, a single-channel target changes by ≤ 1.0% TV against one-source worlds; the only factor is the exclusion of the other sources from the box.
* The 1K's m-balancing changes it through a hidden factor spanning 100×.

**Labels.** Uniform(F) labels are exact for T* (Theorem A1).

---

## 2. Dataset specification and physical definitions

| item | definition | code | status |
|---|---|---|---|
| world | Ω = (0,1)³, N = 1024 voxels per axis, receivers at voxel centres (i+½)/N | contract `RIND3D_Dataset1K/contract/RIND3D_1K_CONTRACT_v1.json:6-7`; `N/convex.cpp:836-848` | V |
| solid | closed union of bounded, full-dimensional convex polyhedra, each strictly inside Ω with a gap | `G/lite_acceptance.py:41,60-61` | V |
| visibility | V_G(s, x) = 1 iff the **closed** segment [s, x] misses the **closed** solid. Contact and tangency block; SOLID receivers are 0 | `RIND3D_Generator/README.md:11`; `docs/scientific/cpu-cuda-certification.md:22-29` | V |
| legal source domain D(G) | open cube, not in the closed solid, certified clearance > 2⁻²⁶ (exact separating-plane proof, or the exact canonical `clearance_decide`) | `D1K/physical.py:54-64`; `N/convex.cpp:582-603`; `G/rendering/context.py:87-105` | V |
| sources | K ∈ {1..4}; i.i.d. rejection sampling from the uniform cube (keyed stream 51001+97·attempt); anti-duplicate ‖·‖∞ ≥ 2⁻²⁰ (TV effect < 10⁻¹⁴); no minimum separation | `D1K/physical.py:67-83` | V (per-scene uniformity PIT-tested, audit E2) |
| source count | 1K: adaptive controller (15% forced floor per k, else m-balance optimisation); realised 154/135/158/553. 100K v1: same rule per 1000-slot epoch. **100K v2:** declared κ = (0.15, 0.15, 0.15, 0.55), provisional, keyed draw per slot | `D1K/controller.py`; `100K/controller.py`; `100K/sampling.py` | V (v2 implemented; κ provisional until the canary) |
| families | the 9 v0.1 families: rotated_boxes, thin_slabs, prisms, wedges, convex_polyhedra, multiscale_convex, corridors, barriers, occluder_clusters; family = slot mod 9. Five extra polyhedral families are implemented but excluded for label cost | `G/lite_generators.py`; `D1K/families.py:23-130`; `docs/GEOMETRY_DIVERSITY.md:71-77` | V |
| acceptance gate | Fraction-verified native proof (interior ball, vertices inside, LP duals for containment), codec round trip, membership probes, segment reversal, two independent receiver predicates on 16 receivers | `G/lite_acceptance.py:78-112` | V (1000/1000 accepted at attempt 0) |
| windows | sides L ∈ {64, 128, 256}; closed box B̄(W) = [o/N, (o+L)/N]³; 1K 16/24/24 windows per side (64 per scene); 100K 4/4/4 (12 per scene) | `D1K/physical.py`; `100K/physical.py` | V |
| response types | per channel: mixed (≥ 1 lit and ≥ 1 dark FREE receiver), all-lit, all-dark. Within mixed: solid-separated (no 6-adjacent lit/dark FREE pair) vs boundary-mixed. All are X-measurable | `SR/pilot/featlib.py`; `RIND3D_Audit/06_EXTREME_RESPONSES.md` | V |
| posterior target | q_F on 128³ cells, F_W^j = {s ∈ D(G) \ B̄(W) : V_G(s, x) = r_j(x) ∀ FREE x ∈ W} | `TPE/README.md:12-19`; `D1K/labels.py:12-15` | V |

---

## 3. Scene generation and window selection

### 3.1 Physical stage (1K reference)

**Entry point:** `RIND3D_Dataset1K/scripts/run_physical.py` → `D1K/physical.run_scene`.

**Per scene:**
1. geometry from `families.generate` with acceptance;
2. sources;
3. full fields per source via `render_stream`, backend `cuda_certified`;
4. CountIndex (exact 16³ prefix sums);
5. `select_side` for each side;
6. `materialize` of the crops;
7. staged directory, `os.replace` publish, and a `SEAL.json`.

**Cost** (mean per scene; 6.84 h for 1000 scenes sequentially) [derived from `REL/manifest.jsonl`]:

| stage | mean |
|---|---|
| geometry + acceptance | 0.12 s |
| fields | 7.9 s |
| count index | 0.29 s |
| selection | 4.5 s |
| materialize | 11.8 s |
| **total** | **24.6 s** (p95 39.3 s) |

### 3.2 Selection law (1K, and the current 100K draft)

`D1K/physical.py:121-175`; derivation in `RIND3D_Dataset1K/docs/SELECTION_LAW_1K.md`.

**`evaluate`** in order:
1. any source in the closed box → reject;
2. FREE = L³ − SOLID > 0 (exact CountIndex);
3. channel j informative ⇔ lit > 0 ∧ dark FREE > 0; m = Σ_j I_j, mask = bits; requires m ≥ 1.

**Acceptance.** A ∧ U < c(m, L, k).
* The 1K controller sets c ∝ deficit(m)/p̂(m | L, k), clipped to [0.01, 1]: effective-layer balancing.
* 1K outcome: m histogram of windows 16,071 / 16,008 / 15,969 / 15,952.

**Rate stream.**
* 1024 separate natural proposals per side (100K: R = 256/128/64).
* Λ̂ = accepts/R, plus the m histogram of admissible proposals (`rate_m`; 100K: only the first `rate_exact` = 8).

**Training weight ω.**
* ω̂ estimates Λ_L(G, S), the per-proposal acceptance probability; the 1K/100K quota sampling needs it.
* Estimators (`D1K/loader.py:Scene.weights`), all unbiased:
  * `rate`;
  * `quota` = (r−1)/(N−1);
  * `combined`;
  * `rb` = Σ_m rate_m c(m)/R;
  * `combined_rb` (default).
* 100K port caveat: `rb` needs full rate-stream m counts.

### 3.3 Natural windows and the preferred sampling design

**NAT-1K** (`SR/natural/natural_supplement.py`; data `RIND3D_1K_data/natural_supplement_v1`).
* Natural windows from the stored 1K fields: stream 57000+L, n_p fixed proposals per side, no thinning.
* 907 non-test scenes, 30,349 mixed channels labelled.
* Regeneration is byte-identical (VALIDATION_RECORD).
* Status: V.

**T\* weights** (loader `population=`; derivation `SR/B_SAMPLING_WEIGHTING.md`):

| population | weight (per natural proposal) |
|---|---|
| B1K | ω̂ |
| Tstar, release only | ω̂/(n_L c) |
| natural windows | 1/n_p |
| union (balance heuristic) | 1/(n_L c/ω̂ + n_p) ≤ 1/n_p |

Measured Kish ESS of the pilot-train pool: B1K 69%, T* release-only 10.8%, T*-MIS 33% (`SR/experiments/out/pilot_pool_ess.json`).

**100K recommendation** (D, not implemented; `SR/H_100K_RECOMMENDATIONS.md`, `SR/J_COMPLEXITY_COVERAGE.md`):
* natural windows, c ≡ 1, a declared fixed κ;
* if κ has < 50% K = 4: one coverage window of the four at L64 in K = 4 scenes, with c(m) = (1/8, 1/4, 1/2, 1), combined by the balance heuristic;
* predicted to meet the declared minimum-coverage targets. It is a prediction from the 1K worlds; it needs canary verification.

---

## 4. Four response types and their difficulties

**Frequencies** [measured, 1K release accepted windows; `RIND3D_Audit/02_SELECTION_AUDIT.md` §7, `D_BACKFILL_REPORT.md` §3]:
* 159,802 mixed channels, of which 3,924 are solid-separated in the pilot scenes (L64 6.7%, L128 2.5%, L256 0.4%);
* 25,795 all-lit and 13,443 all-dark channels, stored and unlabelled.
* In natural windows (NAT-1K pilot-eval), all-lit is the most common channel type at L64.

| type | information content | current label path | cost | status |
|---|---|---|---|---|
| boundary-mixed | high: T* labels carry 10.3 nats over the prior on average (`pilot_info_recovered.json`) | FastLabel v3.4: single-cell support or a statistical certificate | 0.8 s median per label | V, production at 1K scale |
| solid-separated mixed | decisive: full-world labels carry 1.9–2.0 nats; mixed-trained Students put 2.6–4× the prior's mass on F | FastLabel v3.4, 100% statistical | 13.4 s mean per label (10–13× boundary-mixed); 2.5% of labels, ≈ 23% of 1K label compute | V. **Open:** optimise phase 1 for broad B; boundary items adjacent to SOLID (audit O5) are an untested idea |
| all-lit | weak: 0.36 nats on average even with the full world (posteriors cover 7–95% of the cube) | FastLabel v3.4 certifies (180/180 natural OOD sample) | 24–35 s median per label, 1.5–1.8·10⁶ cells | P (research); not labelled in production (cost gate) |
| all-dark | moderate: 1.63 nats | v3.4 fails (9/18): a source inside a closed solid is consistent with "all dark", so B keeps solid interiors and sampling cannot certify. **S6 prototype** (`SR/s6/fastlabel_s6.py`): sample B′ = B \ solids by exact rejection, label q_B′ with float half-space volumes plus a deterministic η ≤ 2.8·10⁻⁶ | S6: 82–116 s median | P (S6: 18/18 audit tasks, 180/180 OOD references, 0 of 53,590 feasible points outside the support). Not production-reviewed |

**Cost gate** (`SR/G_GENERALIZATION_AND_EXTREMES.md` §3): extremes fail criteria 3–4 (cost and scalability). They stay evaluation-only (OOD), and their channels stay stored with status `not_computed`.

**Audit fixtures.** Seven exact N = 32 fixtures confirmed the engine against independent Monte Carlo for every type (`RIND3D_Audit/06_EXTREME_RESPONSES.md` §3).

---

## 5. Local response computation and storage

### 5.1 Visibility backends (`G/rendering/backends.py`)

| backend | algorithm | guarantee | measured |
|---|---|---|---|
| `cuda_certified` (production) | NVRTC kernels per 64³ brick. Region labels at levels 64/32/16/8, then per-voxel tests in **directed-rounding binary64 intervals** (`__d*_rd/ru`, `--fmad=false`) of exact projected shadow planes. UNKNOWN bits are recomputed on the CPU by native interval/GMP clipping. Device arithmetic is self-tested at start-up | every bit decided exactly; no lit SOLID (asserted) | full 1024³ field: 2.53 s per source mean (p95 3.78 s; 3,110 fields; 0 UNKNOWN rays in the release) [derived]. 64³ tile 0.15 ms at batch 32 (`LOCAL_RENDERING_100K.md:12-19`) |
| `cpu_native_certified` | parametric segment clipping in outward-rounded intervals, falling back to exact GMP; region certificates on an octree | exact | full-field CPU time not measured |
| `cpu_reference` | canonical CGAL oracle with adaptive region certificates | exact (independent) | slow (reference only) |

**Batch / local use.**
* `fac.source(bundle, s, "cuda_certified", POLICY).packed_batch(origins, 64, N)` renders arbitrary 64³ tiles. This is what 100K `local_eval` and the FastLabel tests use.
* Re-rendering 191 M stored window bits from geometry and true sources matched exactly (VALIDATION_RECORD, check 1).

**Point queries.**
* `context.bind(p).native.points(...)`: exact GMP membership and visibility for receiver lists.
* FastLabel's batched verifier (§6.2) is a separate certified GPU point-source tester.

### 5.2 Storage formats

**RINDPack v1** (`G/pack.py`, `N/pack.cpp`).
* 256-byte header with CRC and identity/policy hashes.
* 64³ bricks, bit-packed little-endian ZYX; codecs zero / one / raw / zstd-3; per-brick CRC and popcount.
* Transactional writer (fsync checkpoints, journal, atomic rename); random access via `read_window`.
* Status: V (tests: `test_pack.py`).

**1K scene directory** (`REL/scenes/<slot>/`):

| path | contents |
|---|---|
| `scene.json` | scene_id = sha256(contract ‖ geometry ‖ sources), k, c_table, per-side statistics, Λ̂ |
| `public/windows.npy` | window, x, y, z, L, Lp, m, mask, k. **m / mask / k are privileged for single-channel Students** |
| `public/observations.bin` + `observations_index.npy` | per window one SOLID crop plus one crop per source; `np.packbits` big-endian, ZYX, zstd-3 |
| `privileged/fields/src{j}/{response,solid}.rindb` | full fields (1K only) |
| `privileged/geometry.npz`, `geometry_rows.npz` | canonical bundle; float64 rows for the labeller |
| `privileged/selection_log.npz`, `sources.json`, `windows_full.json`, `geometry_acceptance.json` | provenance and diagnostics |
| `labels/tasks/w{w:03d}_s{j}.npz` | cell, lo, hi, kind, tv_bound, meta (status, certificate, engine version, config sha256); sha256 logged in `REL/labels_log.jsonl` |
| `labels/LABELS.json` | per-scene index |
| `SEAL.json` | sha256 of every physical file at publication |

* Sizes per scene [derived]: public ≈ 1.7 MB, fields ≈ 2.7 MB, labels ≈ 15.5 MB.
* Release totals: public 1.6 GiB, privileged 2.7 GiB, labels 15.5 GiB, so `scenes/` = 20 GB.

**100K container** (`100K/storage.py`, `100K/physical.py:172-185`).
* One `np.savez` per scene: meta JSON, windows, packed crops of all channels, geometry, P/K rows, logs.
* Atomic write; container sha256 in the manifest.
* Labels in an append-only `R3LJ` journal (per-record sha256, torn-tail detection), consolidated per scene.
* Sizes: 0.36 MB per scene; 47 KB per TPE label. Projection ≈ 0.2 TB for 100K [proj].
* Known gaps: the manifest is appended without fsync; consolidated label files have no recorded sha256.

---

## 6. Global posterior estimation

### 6.1 Tau-pencil engine (TPE): exact, deterministic

**Entry point:** `TPE/api.py: label_window(P, K, cls, origin, N, engine="cpu"|"gpu")`, version `tpe-b4/p2-cpu+g2-gpu-pass-v3.1`.

**Algorithm** (`TPE/tpl/engine_p2.py:6-125, 581-1143`).
* On source regions with fixed face signs, each receiver's shadow condition is an orthant in τ_e = f_b/f_f coordinates.
* Dark receivers coverable by one component, and lit receivers with one undecided edge, collapse to half-spaces.
* Stage 1: a coarse octree refuted by sentinel receivers.
* Stage 2: exact BSP of convex polytopes, emitted per 128³ cell (`emit_region`).

**Records.**
* IN: lo = vol − clearance shell, hi = vol.
* UNDET: lo = 0, hi = vol.
* TV bound by exact greedy over the [L_c, U_c] intervals (`TPE/rpl/bound.py`).
* `certified` if ≤ 0.05; `certified_loose` if ≤ 0.10; else `uncertified`.

**GPU variant.** Stage-2 pass segments run on the GPU, bit-identical to the CPU reference (software equivalence: `RIND3D_Posterior/tests/test_regression.py`).

**Cost.**
* 1K first pass: 7.5 / 14 / 139 mean CPU-s at L64/128/256, which projected to ≈ 141 h for the 1K; stopped after 333 labels.
* 100K canary: ≈ 41 warm CPU-s per label, 98.6% tier-1 coverage, ≈ 33,400 laptop-thread-hours for 2.94 M labels [proj].

**Role now.**
* The exact reference.
* FastLabel's phase-1 engine.
* The current 100K labeller.

### 6.2 FastLabel v3.4: production at 1K

**Entry points.**
* `FL/pipeline.label_window_fast(task, **cfg)`, with `task` from `FL/inputs.window_from_1k` or `window_from_100k`; the 1K entry point `D1K/fastlabel.label_task(scene_dir, w, s, **cfg)` wraps it.
* `RIND3D_Dataset1K/scripts/run_labels.py <root> 15 0 --engine fastlabel --scene-order` (one GPU semaphore).
* `backfill/run_backfill.sh`.

**Algorithm** (`FL/pipeline.py:127-260`).
* Phase 1 builds an outer bound B ⊇ F on a receiver subset X′: 256 boundary + 128 interior items; rounds ≥ 2 add violated receivers through `engine_p2x` plus an outer-bound filter.
* If B is a single cell: done (`certified`, support certificate).
* Otherwise, sample uniformly in B (pieces ∝ volume, Delaunay tetrahedra, barycentric folding) and test each point **exactly** for membership in F with the batched GPU verifier (`FL/verify.cu`). The verifier uses:
  * an FP32 filter with a proved error bound (2⁻¹⁸);
  * FP64 directed intervals;
  * a GMP host predicate for UNKNOWN voxels and an exact clearance certificate.
* Failing points contribute their violated receivers to the next round.
* Up to R = 12 rounds.

**Gate A** (`RIND3D_Posterior/experiments/fastlabel/GPUVERIFY.md:62-84`; research harness `validate_verify.py`; production port `FL/gate_a.py`, same candidates and checks).
* 27 windows, 2,684 candidates.
* The verifier's dump fields equal the certified renderer's on all of them.
* Required on any new GPU type.

### 6.3 Certificates (math: `RIND3D_Posterior/experiments/fastlabel/CERTIFICATE_AUDIT.md`, `RIND3D_Audit/07_TV_CERTIFICATE.md`)

**Identity.** If F ⊆ B, then TV(Uniform(B), Uniform(F)) = 1 − vol F / vol B. So TV(q_B, q_F) ≤ 1 − p, with p = P_{s∼U_B}(s ∈ F).

**Two-stage exact binomial test per round r.**
* δ_r = δ·2⁻ʳ.
* Stage 1: n₁ = 512, certify iff K ≤ a₁.
* Stage 2: n₂ = 1024, certify iff K ≤ a₂.
* The exact α is computed by convolution and is monotone in p.
* Round 1: a₁ = 3, a₂ = 17, α = 2.92·10⁻⁸.

**Per-label δ.** δ = family_delta / n_tasks = 0.01 / 159,802 = 6.26·10⁻⁸. Union bound: P(any false statistical certificate among the planned labels) ≤ 0.01.
* NAT-1K: its own family, 0.01 / 30,349.
* OOD references: 0.01 / 360.

**Statistical label.** One record per cell, lo = hi = vol(B ∩ cell), kind 4, tv_bound = 0.05, full certificate in the meta.

**Assumptions:**
* **A1:** the engine's OUT decisions are sound (F ⊆ B).
* **A2:** the filter boxes contain the pieces.
* Keyed i.i.d. sampling.
* No re-rolls after a failed certificate.

Sampling cannot detect a violation of A1. Evidence for it:
* the true source lies in the support for 159,802 / 159,802 + 30,349 / 30,349 labels;
* audit E12: 0 of 106,170 prior-feasible points fall outside broad supports.

**Error budget:** TV ≤ ε + η_samp (≈ 10⁻¹¹) + η_vol (≈ 10⁻¹²).

### 6.4 Measured (1K backfill; `SR/D_BACKFILL_REPORT.md`)

**Run.**
* 159,469 labels in 4.33 h (15,600 s), 10.2 labels/s, 15 workers, 0 failures.
* Per label: mean 1.44 s, median 0.82 s, p99 17 s, max 72 s.

**Outcome by side** (deterministic = single-cell support):

| side | certified (single-cell) | certified_stat | median cells |
|---|---|---|---|
| L64 | 2,171 | 36,460 | 2,997 |
| L128 | 14,993 | 45,239 | 121 |
| L256 | 35,359 | 25,580 | 1 |

**Accuracy against 447 exact references:** max TV 0.0164 (`REPORT.md`).

**Determinism.** Eight labels recomputed after a driver update were bit-identical.

**Verification.** `RIND3D_Dataset1K/backfill/verify_backfill.py` recomputes the certificate thresholds and checks sha256 and soundness. It does not redraw samples.

---

## 7. Student supervision and the role of a Teacher

### 7.1 Loader (`D1K/loader.py`, tests `RIND3D_Dataset1K/tests/test_loader.py`; `SR/C_LOADER.md`)

* **Canonical access.** `RIND3D1K(root, split)` / `RIND3DNatural(root)` are multi-channel window records. Flattening is training-time only.
* **`flatten(ds, population, *, estimator="combined_rb", accept=("certified","certified_stat"), max_tv=0.05, fallback=None|"onehot", missing="yield"|"drop", supervision="posterior"|"onehot", ...)`.** Each example carries:
  * `supervision` ∈ {posterior, onehot, onehot_fallback, missing};
  * `target` {cells, q};
  * status, tv_bound, certificate, provenance (engine, version, config sha256, file sha256);
  * population, weight;
  * `privileged` (source, source_cell) only on request.
* **Splits** by sha256(scene_id) mod 100: < 80 train, < 90 val, else test. The pilot never opened the test split.

### 7.2 Student reference implementation (pilot scale; `SR/pilot/`, results `SR/E_STUDENT_PILOT.md`)

**Features** (`build_features.py`, `featlib.py`).
* 4 × 32³ uint8 pooled fractions per channel: lit-FREE, dark-FREE, SOLID, lit/dark boundary.
* Plus origin and side; no privileged inputs.

**Model** (`student.py`, ≈ 2.0 M parameters).
* 3D-CNN encoder to 8³ tokens with world-position encodings.
* 4096 coarse-cell queries with window-relative features, 3 cross-attention blocks (d = 128).
* Hierarchical, exactly normalised density: P(c) over 16³ coarse cells × P(f | c) over 8³ fine cells, weighted by the legal-volume fractions of the cube minus the window box. The fine grid equals the label grid.

**Loss** (`target_ce`). CE(T, p) = −Σ T(c) log p(c), exact on the sparse 128³ target. T is the certified label (posterior) or the true-source cell (one-hot).

**Training** (`train.py`).
* AdamW 4·10⁻⁴, 8,000 steps, batch 32, bf16, cross-attention recomputation.
* Batches drawn ∝ population weight, with equal side shares.
* Boundary-mixed channels only; extremes held out.
* Early stopping on pilot-select one-hot NLL.

**Measured** (13 arms; scene-bootstrap CIs; 1K val split as evaluation):
* **Posterior vs one-hot:** +2.2 to +2.6 nats held-out information gain (+3.7 with 25% of scenes). Calibration MAE 0.03 vs 0.25–0.28. One-hot memorises (best checkpoint at 1–2k steps); no augmentation was used.
* **T\*-MIS vs B1K weighting:** +0.14 nats (two seeds). Seed spread 0.06–0.14. The balancing tilt is below resolution.
* **Scenes vs channels:** 25% of the channels over all scenes matches the full data; 25% of the scenes loses 0.77 nats.
* **OOD, never trained on:** solid-separated +0.7 to +1.0 nats (passes all criteria); all-dark +0.20–0.26 (partial); all-lit ≈ prior. One-hot Students are worse than the prior on all three (`SR/G_GENERALIZATION_AND_EXTREMES.md` §1–2).

**Status.** V at pilot scale (684 training scenes, 2 seeds for the main arms). Not a production training stack.

### 7.3 Teacher

**Current pipeline: no neural Teacher.** Supervision is certified posterior fields.

**History.**
* The 2026-11-10 study compared a Gibbs/energy Teacher against direct NLL, and direct NLL won (`RIND3D_Audit/08_EXPERIMENTS.md` §4).
* `Phase3_Single_Source_3D` holds earlier teacher interfaces, plus uncommitted adaptive-teacher work by another session. It is not part of the RIND3D-1K/100K pipeline.

**Open question (U).** Whether a learned Teacher is worth building as a cheap amortised label source. Candidates are the extremes, or a 100K label budget if FastLabel cannot be ported. No evidence yet.

---

## 8. Pipeline, performance and quality verification

| stage | 1K (reference; measured) | 100K (production branch; measured on the laptop unless [proj]) |
|---|---|---|
| physical | 24.6 s per scene, full fields, 6.84 h / 1000 scenes | 3.7 s per scene, local; ≈ 6.4 h on 16 GPUs [proj] |
| labels | FastLabel 4.33 h for 159,469 labels | TPE ≈ 41 CPU-s per label, ≈ 33.4 k thread-h for 2.94 M labels [proj]; FastLabel not integrated |
| storage | ≈ 20 MB per scene including labels | 0.36 MB per scene + 47 KB per label; ≈ 0.2 TB [proj] |
| verification | `verify_backfill.py`: 0 problems; seals 25,550/25,550; true source in support 159,802/159,802 | `scripts/smoke_test.py` (6 steps), `validate.py` / `verify_root.py`; the 100K package has no pytest suite |

**Tests on this branch** (results in `VALIDATION_RECORD.md`):
* `RIND3D_Generator/tests` (337 pass, 18 skip);
* `RIND3D_Posterior/tests` (9 pass);
* `RIND3D_Dataset1K/tests` (9 pass);
* `RIND3D_HANDOFF/smoke_checks.py`.

**Known untested parts:**
* `D1K/physical.py`, `controller.py`, `families.py` and `validate.py` have no unit tests;
* release-wide independent physics spot checks were never recorded (canary only);
* the 100K equivalence harness against full-field selection is not committed.

**Inconsistencies left in code, documented and not changed:**
* `TPE/tpl/engine_p2x.py:1` calls itself an EXPERIMENTAL copy, but FastLabel uses it for rounds ≥ 2;
* `REL/STOP` (a control file from the stopped first label pass) remains in the frozen release root.
