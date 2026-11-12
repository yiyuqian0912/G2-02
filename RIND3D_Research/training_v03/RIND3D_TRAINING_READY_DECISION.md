# RIND3D Training-Ready v0.3: decision

Date: 2026-11-10.

**Candidate.** `rind3d-training-contract-v0.3`, in `G2-02-gaussian/RIND3D_Training_v03/`. It is separately versioned. The accepted v0.2 canary, the frozen contract v0.2, RIND3D v0.1, the Phase 3 tree and all prior research were read-only. SHA256 trees taken before and after the work are identical (`outputs/preservation/compare.json`). Nothing was committed, pushed or launched.

Labels used below: [M] measured on this laptop (RTX 3060 Laptop, 16 CPU threads); [X] linear extrapolation.

## Verdicts

| Question | Verdict |
|---|---|
| **SB-focused 1K training candidate** | **GO (built).** `v01-sb-v03`: 1000 certified v0.1 SB scenes, re-indexed (obs-index-v0.3 P256, private HMAC index key) and exported in two views. 255,712 windows; 2.0 GB of student shards; 21 min of export; **no physics regenerated**. The leakage audit passes on the exported files. |
| **Initial window profile** | **P256 fixed index**, an exact superset of the v0.2 P102 index (prefix identical in 216/216 canary scenes). Hybrid dynamic refresh shards are the extension pathway (§3). |
| **10K SB rollout** | **GO to propose, not launched.** `configs/next-campaign-sb-10k-v0.3.json`: ≈ 26 GPU-h of certified generation, ≈ 32 GB in total [X]. Needs explicit authorization and a budget. |
| **100K** | **Defined as a separate SB release** (`rind3d-sb-100k`), with MA as its own later release. **CONDITIONAL GO**, after 10K acceptance and a 10K training result. Needs ≈ 263 GPU-h of generation on one laptop GPU (or ≈ 66 h on 4 GPUs) and ≈ 323 GB [X]. Training at 100K needs several GPUs or a per-epoch scene-balanced sub-sample (§6). |
| **Final question** | **Yes, with the stated conditions:** the pipeline is efficient, statistically well defined and leakage-safe for local-only probabilistic inverse-source training at 100K. See §10. |

## 1. What was built and tested

`rind3d_training/` imports v0.2 read-only:

| Module | Role |
|---|---|
| `index` | obs-index-v0.3. The v0.2 proposal streams are continued, so P102 is reproduced bit-for-bit as the prefix. Every rule indicator is computed for every record and proposal, and a 1024-per-side uniform rate stream is added. |
| `selection` | inclusion law, P(A_s) estimation, population weights |
| `export` | Student view (opaque ids, packed zstd shards, private interleaving) and privileged view (labels, flags, weights, scene references) |
| `student` | NumPy/zstd-only reader, scene-balanced deterministic sampler, size-bucketed batches, on-GPU unpacking |
| `teacher` | TeacherPhysics v0.3 (fixes in §8) |
| `phase3_adapter` | Phase 3 Teacher adapter |
| `leakage` | file-level audit L1–L10 |
| `study` | the 102/256/dynamic/hybrid study |
| `bench_*`, `blocksparse`, `interop_audit` | benchmarks and audits |
| `scale_cost`, `report`, `evidence`, `preservation` | reports and evidence |

Exports produced [M]:

| Export | Scenes | Samples | Student bytes | Index wall | Export wall |
|---|---|---|---|---|---|
| `canary-v03` | 216 (108 SB + 108 MA) | 55,184 (P102 ⊂: 21,955 = v0.2) | 798 MB | 683 s | 1040 s |
| `v01-sb-v03` | 1000 SB (776 / 116 / 108) | 255,712 (P102 ⊂: 101,743) | 2.01 GB | 2993 s | 1271 s |

Both index and export ran on 12 CPU workers.

**Tests and audits.** `tests/test_training_ready.py` has 21 tests and **all pass** (`outputs/tests/pytest.txt`). They include the 10-step Teacher interop sequence (§8). The leakage audit L1–L10 passes for both exports (`outputs/tests/leakage_*.json`).

## 2. Two supervision routes (§2 of the brief)

- **Route A (direct).** Student shards provide the inputs. The privileged `join.npy` provides the targets: sources, strengths, K, and the optional population weight. They are joined by opaque `sample_id` in the trusted trainer.
- **Route B (selective Teacher).** TeacherPhysics v0.3 or the Phase 3 oracle is called on demand for the windows an experiment chooses.

Nothing forces Gibbs precomputation. Sampled HT mismatch costs under 1 ms per candidate per window [M], so on-demand Teacher work is cheap. Precomputing every window would not be: a full L256 CPU mismatch costs 1.35 s per candidate.

## 3. Window profiles: 102 vs 256 vs dynamic vs hybrid

Full study: `RIND3D_102_VS_256_WINDOW_STUDY.md`. It uses the stored canary fields only.

**P256 vs P102.** The category mix is unchanged and the natural part equals the rate-stream population. P256 adds:
- 2.5× the rare, extreme, nonlocal and informative windows per scene;
- world coverage 38.7% → 60.1%;
- shadow-boundary leaf coverage 52.7% → 75.0%;
- SOLID-boundary leaf coverage 67.9% → 94.8%.

39% of the extra volume is new. The mean L256 pairwise IoU is 0.017. P256 stays out of the redundancy regime (multiplicity 2.5, against 7–19 at 1024–3072 windows per scene).

**Intra-scene correlation.** For target-like window statistics the ICC is 0.23–0.29 on the canary and 0.13–0.21 on the 1000 v0.1 scenes. A scene therefore yields only about 3–7 effective windows per side, and going from 16 to 40 natural windows per side gains only 4–11% effective samples on these statistics. Observation fine structure is nearly independent within a scene (log-minority ICC ≈ 0.01), so extra windows do add rare and boundary diversity.

**Conclusion:** scale through scenes; use P256 to cover the observation space.

**Dynamic sampling straight from fields** is 10–100× slower than the shards for MA and at L256. It is not Student-safe: it needs the complete fields, and for MA the private channels.

**Hybrid** means P256 plus refresh shards. The trusted exporter materialises extra natural windows from a private HMAC stream. Their weights are trivial and their containment flags exact, and no physics is rerun. This is the extension pathway. It is not needed for the 1K candidate (§6).

**Cost.** SB student export is 1.89 MB per scene for P256 against 0.76 MB for P102 [M], i.e. 189 GB against 76 GB per 100K SB scenes [X]. P102 stays selectable through `in_P102`.

## 4. Source domains and task profiles

See `docs/SOURCE_DOMAINS.md` and the contract.

**Legal domain.** D(G) is the open world, outside the closed solid, with certified clearance > 2⁻²⁶.

**Priors.**
- v0.2 SB: Uniform(D(G)).
- v0.2 MA: K ∈ {2, 3, 4} cycled; i.i.d. legal sources with separation ≥ 1/32; a ~ U[0.1, 1].
- v0.1 SB: Uniform(D(G)) conditioned on coverage eligibility. Only 3 of 1003 trials were rejected, all on one geometry that was dropped. This is declared as the prior tag `v0.1-coverage-eligible`.

**Task profiles.**
- `sb-global-v0.3`: the source may lie in the window; target p(S | O, W).
- `sb-source-excluded-v0.3`: a separately declared conditional task p(S | O, W, S ∉ closed box); source-containing windows are excluded.
- `ma-additive-v0.3`: an unordered set target with K as part of the target.
- evaluation strata.

Containment uses the closed box and is privileged. The Student is told which task it is solving, never whether the window contains a source.

**No-FREE windows** (69 in the canary, 0.4% of L64 natural windows) stay in the natural population. Teacher mismatch on them is `no_free`/NaN, never zero.

## 5. Safe Student export (§5 of the brief)

Schema: `RIND3D_STUDENT_TARGET_SCHEMA.md`.

**Student features.** The local response, the local SOLID mask, the window origin, the side and the modality. Local coordinates follow from (o + i + ½)/1024. There is also an opaque HMAC `sample_id`, which is a join key only and is returned outside the feature dict, and an opaque per-scene `group` token, which is sampler-only and never returned.

**Privileged view** (mode 0700):
- geometry and oracle references;
- true sources and strengths in canonical set order;
- prior and legality rules;
- subset, rank and rule indicators; containment; no-FREE; category;
- nonlocal estimates; population weights per task;
- Teacher metadata via TeacherPhysics.

**Audit on the actual files** (L1–L10):
- the inventory and the exact index dtype;
- no scene or geometry ids (hex or raw), no true-source bytes and no seeds in any index or manifest;
- every payload byte accounted for, with exact lengths and CRCs;
- a sample of decoded payloads equal to the observation re-derived from the certified fields;
- MA palettes local;
- ids unique and uncorrelated with scenes;
- shard order independent of the hidden subset and containment flags (max |corr| < 0.15);
- loader keys exact.

**Secrets.** Index, export and dynamic keys live in `private/` (0600). Release seeds stay private. The canary seed is public and the canary is integration data only.

## 6. Trainable at scale (§6 of the brief)

Details: `RIND3D_LOADER_BENCHMARKS.json`.

| [M] | L64 | L128 | L256 |
|---|---|---|---|
| v0.2 per-scene reader, random cross-scene, cold, windows/s (SB / MA) | 134 / 227 | 56 / 81 | 13 / 15 |
| v0.3 shards, packed read + zstd, cold, one process (SB / MA) | 7677 / 7306 | 3981 / 3070 | 698 / 479 |
| v0.3 + CPU unpack to bool/float (SB / MA) | 5519 / 1675 | 276 / 225 | 84 / 17 |
| CPU→GPU incl. GPU unpack: packed (pinned) vs CPU-unpacked fp16, per SB batch (32 / 8 / 2) | 1.6 vs 55 ms | 3.6 vs 112 ms | 7.4 vs 223 ms |
| training step, GPU-resident, 0.86 M-parameter 3D CNN, AMP (windows/s) | 287 | 253 | 65 |
| training, loader-fed (4 shared-memory workers, cold; canary) | 277 (97%) | 215 (85%) | 49 (76%) |

**DataLoader over a full canary epoch** (45,207 windows, size-bucketed packed batches) [M]:
- 0 workers: 756 / 839 windows/s, cold / warm. On a bounded 13,487-window pass with MA code planes trimmed to the batch maximum: 1112 / 1409.
- Workers with pipe-pickled NumPy batches: 212–422/s. This is IPC-bound at about 1 GB/s.
- Workers with **shared-memory tensors** (returned from the worker): 803 / 903 / 948–964 windows/s with 2 / 4 / 8 workers.
- The v0.2-style mixed-size unpacked path: 92/s with 0 workers and **20/s with 4 workers**.

Cold and warm page caches differ by only 10–20% on NVMe for the sharded format. The v0.2 format pays about 4 ms per scene open on every random cross-scene access.

**L256 memory.**
- Space-to-depth (lossless, a 64³ grid with 2·(L/64)³ channels) makes every side share the same trunk. It was verified bit-exact.
- Block-sparse 8³ coding (exact) leaves only 1.8% (median; p95 10.5%) of L256 blocks mixed. GPU fp16 memory per window falls from 64 MiB to 1.2 MiB median, and the zstd size is 42% smaller.
- Lossy pooling is ruled out: a 2× majority pool erases minority receivers in 96 of 120 L256 windows, and SOLID voxels in 73.

**Deterministic scene-balanced sampling.** `SceneBalancedSampler(index, batch_size_per_L, seed, per_scene)` is keyed by (seed, epoch). It takes a per-scene quota per side, shuffles across scenes and buckets by (modality, L). This is tested.

**The earlier v0.2 numbers are not contradictory** [M re-measured]:
- *156 SB-L64/s:* random cross-scene access where every access re-opens a scene (3.7 ms) and then reads it (3.8 ms). Re-measured as 134 cold and 164 warm, and 281/s when scene-local.
- *10.64 s for 256 windows:* the pass mixes all sides and both profiles. One third are L256, and those return 33.5 MB (SB) or 151 MB (MA) of unpacked arrays each, 7.5 GB in total. With 2 workers it is IPC-bound: re-measured 9.8 s with 2 workers against 7.7 s with 0.

**Epoch arithmetic** (SB, this model on one RTX 3060 [X]; `RIND3D_TRAINING_SCALE_COST.json`). Each cell is hours per full epoch over every stored window, with the steps per epoch in brackets:

| Scenes | P256 | P102 |
|---|---|---|
| 1K | 0.68 h (55,812 steps) | 0.27 h |
| 10K | 6.8 h | 2.7 h |
| 100K | 68 h (5.6 M steps; 54 h if purely compute-bound) | 27 h |

**Availability vs consumption.** A 1K SB P256 release makes 256K unique windows available. Two ways to consume it:
- an epoch over all of them takes 0.68 hours;
- a scene-balanced epoch of, e.g., 24 windows per scene per side takes about 0.3× of that.

The hybrid adds unique windows per refresh, but **not scenes**.

## 7. Distributions and selection law (§7 of the brief)

Details: `docs/SELECTION_LAW.md`.

**Design and weights.** Each (scene, side) is a mixture of uniform proposal streams with exact rules. The density is q(w) = Σ_s (n_s/n) p_nat(w) A_s(w)/P(A_s), and the weights are 1/(n_nat/n + Σ_s (n_s/n) A_s/P(A_s)). The source-excluded task has its own variant.

**Validation [M].** The weighted mixture reproduces the natural rate-stream frequencies on both the canary and v0.1, for example:

| Statistic (v0.1, L64) | Unweighted | Weighted | Population |
|---|---|---|---|
| mixed | 56.1% | 16.4% | 16.2% |
| rare | 12.8% | 0.79% | 0.80% |

**ESS and weight range.** The population ESS is 64–79% of the records, and the median maximum weight is about 1.9.

**Clean population target.** The natural subset with weight 1.

**What may train which task.**
- Selected subsets train the global task **only with weights**. They are unweighted-consistent for the declared source-excluded task (covariate shift only).
- The nonlocal subset is an evaluation stratum only.
- No response-dependent sampling quantity reaches the Student.

## 8. Teacher interoperability (§9 of the brief)

Details: `RIND3D_DATASET_TEACHER_INTEROP.md`.

All 10 steps pass on exported files:
1. both views name the same scene;
2. the true label is retrieved;
3. visibility of the true source matches the observation;
4. CPU and CUDA reconstruction are bit-exact;
5. uniform and stratified banks behave correctly;
6. HT estimates are unbiased and the population weights validate;
7. full and sampled mismatch work, including against Phase 3 `full_window_cost`;
8. candidate legality agrees with Phase 3;
9. source-containing windows are handled;
10. the Student inputs are clean.

Audit fixes, measured:

| Item | v0.2 / Phase 3 behaviour | v0.3 fix |
|---|---|---|
| CUDA window | v0.2: `AttributeError` without a factory; SOLID = None | owns its factory; returns the certified SOLID |
| Stratified banks | v0.2: 68 of 120 banks with k ≠ requested | exact k |
| No-FREE windows | v0.2: `ZeroDivisionError`; Phase 3 `full_window_cost` silently returns NaN | explicit `no_free` status |
| Probe design | Phase 3 falls back to a non-uniform packed-byte scan (≈ 0.1% of natural SB windows), so HT weights are invalid | adapter flags it; v0.3 banks are always uniform |
| Sample identity | Phase 3 dicts carry the real `scene_id`, a commitment to the label in v0.2 | adapter uses the opaque id |
| MA | Phase 3 is single-source | `additive_response` |
| Window box | closed vs half-open | declared in the contract |
| Policy | different policy hash | identical certified answers |

## 9. SB and MA, and what "100K" means (§8 of the brief)

**SB 1K** is v0.1, reused read-only. **Expanded SB** is a new versioned release produced by the v0.2 runner with `profile_schedule ["SB"]` and a private seed file, then indexed and exported by v0.3. MA keeps permutation-invariant set semantics: the canonical stored order carries no meaning, and the tests check the response is invariant.

Costs per definition, from `RIND3D_TRAINING_SCALE_COST.json` [X]:

| 100K definition | Scenes | Generation (1 laptop GPU) | Storage, P256 | Note |
|---|---|---|---|---|
| SB-only | 100K | ≈ 263 GPU-h | ≈ 323 GB (P102: ≈ 197 GB) | **recommended first** |
| combined 50/50 | 100K | ≈ 601 GPU-h | ≈ 755 GB (P102: ≈ 520 GB) | one prior per release becomes a mixture; ties SB to MA |
| separate SB-100K + MA-100K | 200K | ≈ 1201 GPU-h | ≈ 1.51 TB (P102: ≈ 1.04 TB) | MA dominates; size MA after a Student study |

## 10. Answer to the final question

**Can the pipeline provide an efficient, statistically well-defined, leakage-safe foundation for local-only probabilistic inverse-source training at 100K scale? Yes, conditionally.**

- **Efficient [M].**
  - Windows are index records over certified fields, and v0.3 adds only the index and the packed two-view export.
  - The Student loader delivers 480–7700 windows/s per process from packed shards (cold, by side and profile), 30–60× the v0.2 reader. In-process batched loading over the 1K SB release runs at 1887 / 3187 windows/s (cold / warm), so training is GPU-bound.
  - Windows of every side move to the GPU packed and are unpacked there, losslessly.
  - Teacher calls are under 1 ms per candidate with sampled banks.
- **Statistically well defined [M].**
  - Declared priors per release, task profiles with explicit prediction domains, and a separate source-excluded task.
  - An exact inclusion design whose weights reproduce the natural population.
  - A clean natural target, and no-FREE handling that never scores zero.
  - Known intra-scene correlation, which says scale must come from scenes.
- **Leakage-safe [M].**
  - Two views and opaque HMAC ids.
  - Private keys and seeds, with the public-seed replay attack closed.
  - A file-level audit of the actual exports.
- **Conditions** for 100K:
  1. Authorise and fund generation: ≈ 263 GPU-h SB-only, ideally on 4 or more GPUs, plus ≈ 1000 CPU-h of indexing (≈ 83 h wall on this laptop's 12 workers) that can overlap with GPU generation.
  2. Approve ≈ 323 GB of storage for SB P256, or ≈ 197 GB with P102.
  3. Train on several GPUs or with scene-balanced sub-sampled epochs.
  4. Stage it as 10K first, with the frozen T1–T6 gates (`docs/NEXT_CAMPAIGN_PROPOSAL.md`).
  5. Keep MA a separate release.
  6. Treat the public v0.1 test split as an internal evaluation split, not a blind test: its complete response fields are public.

## 11. Commands, evidence, risks

**Commands.** All commands are in `docs/REPRODUCE.md`.

**Evidence.** `EVIDENCE_MANIFEST.json` and `outputs/RIND3D_TRAINING_READY_REVIEW.zip`. The archive excludes private keys, privileged views and payload shards.

**Remaining risks.**
- Index CPU cost: about 33 s per scene under 12-way load. It is dominated by the nonlocal and rate streams, and parallel and overlappable.
- The MA L256 student payload is 58 KB per window.
- The worker scaling of the DataLoader is bounded by the shared-memory hand-off: about 950 windows/s with 8 workers, against 840/s for one in-process reader.
- The ICC means 100K scenes give about 0.3–0.7 M effective windows per side for target-like statistics, not 25 M.
- The benchmark model is small; larger models make training even more GPU-bound.
