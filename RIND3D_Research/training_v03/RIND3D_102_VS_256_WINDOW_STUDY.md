# 102 vs 256 vs dynamic vs hybrid windows: study on the stored canary fields

**Data.**
- All 216 accepted v0.2 canary scenes: 108 SB, 108 MA (K = 2/3/4), nine families.
- No physics was regenerated. Every window was evaluated exactly from the stored certified fields: CountIndex for counts, stored packed fields for responses, and the stored 16³ count leaves for the boundary analysis.
- Raw output: `outputs/study/window_study_canary-v0.2.json`. Code: `rind3d_training/study.py`, `rind3d_training/index.py`.
- [M] = measured. [X] = extrapolated.

## Profiles compared

| Profile | Per side L = 64/128/256 | Windows/scene | Construction |
|---|---|---|---|
| **P102** | natural 16, informative 8, extreme 4, rare 2, nonlocal 4 | 102 | = obs-policy-v0.2. **Reproduced bit-for-bit as the prefix of P256 in 216/216 scenes** [M] |
| **P256** | natural 41/40/40, informative 20, extreme 11, rare 6, nonlocal 8 | 256 | the same deterministic proposal streams, continued (caps × 2.5) |
| **dynamic** | natural only, drawn on demand from the stored fields | unbounded | the rate stream (1024 per side) stands in for it here |
| **hybrid** | P102 or P256 fixed index + deterministic dynamic natural windows | fixed + per refresh | dynamic origins from HMAC(private key, scene, refresh, j) |

## 1. Category distribution (fractions of windows) [M]

The natural population is identical in both profiles and matches the 1024-per-side rate stream to within sampling error. For example, SB natural L64 is 64% all-lit in both P102 and P256, and 65% in the rate stream. The selected subsets shift the mix towards mixed and difficult windows in the same proportion in both profiles.

| SB, all windows | L64 P102 / P256 | L128 P102 / P256 | L256 P102 / P256 | natural L64 / L128 / L256 (rate stream) |
|---|---|---|---|---|
| no-FREE | 0.2% / 0.4% | 0.0% / 0.1% | 0 / 0 | 0.42% / 0.08% / 0 |
| all-lit | 30.3% / 30.6% | 25.4% / 25.6% | 14.5% / 14.5% | 64.7% / 54.1% / 30.7% |
| all-shadow | 9.1% / 9.5% | 7.0% / 7.0% | 3.9% / 3.9% | 19.7% / 15.1% / 8.3% |
| mixed | 60.3% / 59.6% | 67.6% / 67.3% | 81.6% / 81.6% | 15.1% / 30.8% / 61.0% |
| rare (mixed, minority ≤ 64) | 11.1% / 12.2% | 8.4% / 9.7% | 7.2% / 8.3% | 0.71% / 0.77% / 0.63% |
| extreme (mixed, ≤ 1%) | 26.3% / 27.9% | 28.7% / 30.4% | 31.3% / 33.8% | 3.1% / 6.6% / 12.9% |
| nonlocal flag | 49.8% / 48.2% | 47.6% / 46.3% | 37.2% / 35.4% | natural records: 12% / 20% / 21% |
| source-containing | 0.0% / 0.0% | 0.1% / 0.1% | 1.1% / 0.9% | 0.03% / 0.20% / 1.8% |

MA follows the same pattern. Mixed is 68% / 79% / 90% of all windows at L64 / L128 / L256 against 33% / 55% / 80% natural, and source-containing natural windows are 0.08% / 0.6% / 4.4%. "All-lit" is replaced by constant-lit (every channel constant over FREE).

**Per scene.** P256 holds 2.5× as many rare, extreme, nonlocal and informative windows per scene as P102. For example, there are 6 instead of 2 rare windows per side. Difficult-stratum availability (shortfalls) [M]:
- informative, extreme and rare: no shortfall in either profile;
- nonlocal at L64 / L128 / L256: 18 / 6 / 15 scenes short in P102 (3 with none at L256), and 11 / 6 / 13 in P256 (1 with none).

## 2. New spatial information, overlap and redundancy [M]

Values are medians per scene. Coverage is measured on an 8-voxel lattice of cell centres. Boundary coverage is the fraction of the scene's mixed 16³ leaves (shadow boundary) or partially-solid leaves (SOLID boundary) that lie inside at least one window.

| Variant | Windows | World coverage | Multiplicity (covered) | Shadow-boundary leaves | SOLID-boundary leaves | L256 pair IoU | L256 pairs overlapping |
|---|---|---|---|---|---|---|---|
| P102 (SB) | 102 | 38.7% | 1.57 | 52.7% | 67.9% | 0.017 | 17.8% |
| P256 (SB) | 256 | 60.1% | 2.52 | 75.0% | 94.8% | 0.017 | 18.2% |
| P102 natural only | 48 | 23.3% | 1.23 | 27.3% | 42.1% | 0.015 | 16.7% |
| P256 natural only | 121 | 44.0% | 1.62 | 50.9% | 75.7% | 0.016 | 16.7% |
| dynamic 256 | 256 | 63.0% | 2.41 | 70.7% | 94.6% | 0.016 | 17.2% |
| dynamic 1024 | 1023 | 85.5% | 7.11 | 90.4% | 100% | 0.016 | 17.0% |
| dynamic 3072 | 3072 | 93.8% | 19.4 | 96.3% | 100% | 0.016 | 17.0% |

- **Overlap.** The pairwise overlap within a side is small: the mean IoU of two L256 windows is 0.017, and L64 windows almost never overlap (0.4% of pairs). The fixed P256 index is not redundant at the voxel level.
- **New volume.** The median fraction of the 154 extra P256 windows' volume that no P102 window covers is **38.8%**.
- **Redundancy grows fast beyond P256.** At 1024 windows per scene the mean covered voxel is seen 7 times; at 3072, 19 times. Spatial information saturates at the scene's own field: more windows re-read the same geometry and the same source.

## 3. L distribution

Both profiles split windows evenly across sides: 86/85/85 for P256 and 34/34/34 for P102.

Only L256 windows are spatially dense: 85 L256 windows cover 56% of the world, while 85 L64 windows cover 2%. L256 also dominates storage, I/O and compute:
- **Student bytes per window** [M]: SB 456 B / 2.3 KB / 19.5 KB and MA 0.8 KB / 5.8 KB / 58 KB at L64 / L128 / L256.
- **Training throughput** for the benchmark CNN [M]: 287 / 253 / 65 windows/s.

The even split is retained because the Student must be scale-aware and the natural population is defined per side. A per-side reweighting is available in the sampler (`per_scene={64: m64, 128: m128, 256: m256}`).

## 4. Intra-scene correlation [M]

The ICC is a one-way random-effects estimate over 216 scenes. The design effect is 1+(m−1)·ICC, and the effective number of windows per scene per side is m/(1+(m−1)·ICC).

| Window statistic (natural) | ICC L64 / L128 / L256 | eff/side at m = 16 (P102) | eff/side at m = 40 (P256) | gain |
|---|---|---|---|---|
| source distance / L (target proxy) | 0.23 / 0.26 / 0.29 | 3.6 / 3.4 / 3.1 | 4.0 / 3.6 / 3.2 | +4–11% |
| lit fraction | 0.25 / 0.28 / 0.37 | 3.4 / 3.1 / 2.5 | 3.8 / 3.4 / 2.6 | +5–12% |
| mixed indicator | 0.11 / 0.17 / 0.26 | 6.2 / 4.4 / 3.2 | 7.8 / 5.3 / 3.6 | +12–26% |
| SOLID fraction | 0.10 / 0.14 / 0.31 | 6.4 / 5.0 / 2.8 | 8.2 / 6.2 / 3.1 | +11–28% |
| log₁₀ minority (mixed windows) | 0.011 / 0.017 / 0.075 | ≈ m | ≈ m | ≈ 2.5× |

**Interpretation.** Every window of a scene shares one geometry draw and one source draw. For statistics tied to the target (where the source is relative to the window) or to the scene's overall lighting, a scene supplies only about 3–4 effective windows per side whether it has 16 or 40 natural windows. For the fine structure of the observation (the minority receivers and shadow-boundary shape), windows are nearly independent, so more windows add nearly proportional diversity.

More windows per scene therefore enrich **observation diversity** (rare and thin structure, boundary coverage). They do **not** substitute for **more scenes** for the target distribution. Scale must come from scenes.

## 5. Selected mixtures: population effective sample size [M]

Population importance weights (`docs/SELECTION_LAW.md`) are bounded; the median maximum weight is about 1.9 per (scene, side). The weighted mixture keeps 64–79% of its records as population-effective samples:

| Task | P102 ESS/side (of 30) | P256 ESS/side (of 77) |
|---|---|---|
| global | 19.3 / 21.5 / 24.1 | 49.4 / 53.8 / 60.5 |
| source-excluded | 19.3 / 21.4 / 23.8 | 49.4 / 53.7 / 59.7 |

## 6. Throughput and extra storage / generation cost [M]

| | P102 | P256 | dynamic (trusted, from fields) | hybrid |
|---|---|---|---|---|
| Student export, SB per scene | 0.76 MB | 1.89 MB | 0 extra | fixed + refresh shards |
| Student export, MA per scene | 2.21 MB | 5.52 MB | 0 extra | fixed + refresh shards |
| Index (privileged) per scene | ≈ 40% of P256 | ~0.2 MB (records + rate stream + report) | 0 | as fixed |
| Index + export CPU per scene | ⊂ P256 | canary: index 683 s + export 1040 s wall for 216 scenes, 12 workers | per-window cost at train time | refresh ≈ export cost of the new windows |
| Loader rate, packed shards, cold [M] | as P256 | SB 7677 / 3981 / 698 per s; MA 7306 / 3070 / 479 per s (L64/128/256, one process) | SB 38 / 30 / 11 per s cold, 582 / 120 / 15 per s warm; MA 14 / 10 / 3 per s cold, 187 / 32 / 3.6 per s warm | as fixed |
| Physics regenerated | none | none | none | none |
| Student-safe | yes | yes | **no** (needs full fields; MA needs private per-source channels) | yes, if refreshes are materialised by the trusted exporter |

Dynamic sampling straight from the fields is 10–100× slower than the packed shards for MA and at L256. It is also not Student-safe: it needs the complete fields, which are non-local information, and for MA the private per-source channels. Dynamic windows must therefore be **materialised by the trusted exporter** into new shards (the hybrid route), not read by DataLoader workers.

## 7. Decision: initial training profile

**Initial profile: P256 fixed index**, exported in the two-view format, for SB (the 1K candidate from v0.1) and for the canary SB and MA. Training uses either:
- (a) natural windows with weight 1, the clean population; or
- (b) the natural + informative + extreme + rare mixture with `w_global_mixture`, or `w_srcexcl_mixture` for the declared source-excluded task.

Nonlocal windows are an evaluation stratum only.

**Rationale.**
1. P256 costs no physics. It is an exact superset of the validated v0.2 index (P102 = prefix, flag `in_P102`), so every v0.2 result remains reproducible inside v0.3.
2. Per scene it gives 2.5× the difficult strata (rare, extreme, nonlocal), +55% world coverage, +42% shadow-boundary and +40% SOLID-boundary coverage. 39% of the extra volume is new. It stays far below the redundancy regime (multiplicity 2.5 against 7–19 for 1024–3072).
3. The extra cost is small and linear: +1.1 MB per SB scene, i.e. 110 GB per 100K SB scenes [X], and roughly 2.5× index/export CPU. The Student loader is not the bottleneck (§6 and `RIND3D_LOADER_BENCHMARKS.json`).
4. The ICC study shows that neither profile fixes scene scarcity. Beyond P256, scale must come from **more scenes**, not more windows per scene.

**Extensible pathway.**
- (i) Raise the quotas: continue the same streams. Existing records are unchanged and the new ones are appended.
- (ii) The hybrid: the trusted exporter materialises extra natural windows from the private dynamic stream into refresh shards (e.g. one refresh per K epochs). Their weights are trivial (natural), their containment flags are computed exactly, and no physics is rerun.
- (iii) P102 stays selectable through `in_P102` for a cheaper 100K release if storage is constrained: 76 GB instead of 189 GB of SB student shards per 100K scenes [X].
