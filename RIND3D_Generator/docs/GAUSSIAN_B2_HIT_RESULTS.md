# B2 fixed-witness HIT experiment

The experiment reuses the scalar primary's strict witness at its existing ray parameter. A signed-gradient bound and a direct full-sum lower bound were first compared on continuous axial intervals. The measured incremental coverage justified a bounded primary-only cache. The independent F_J build supplies a separate physical result. [Proof and reproduction](GAUSSIAN_B2_HIT.md)

## Witness audit before cache integration

The audit sampled 2,304 original scalar HITs in the cheap, middle and expensive strata of six frozen tiles, plus 384 extra HITs along sampled FREE-visibility transitions. It included the three original hotspots 14-05, 14-13 and 14-14, as well as 7-hot, 60-hot and 63-hot. All 2,688 sampled decisions came from the original directed point comparison and retained a positive strict margin without a new exponential. Each sample's rerun agreed with its unchanged scalar decision, reason, node count and work counters. The sample gives no frequency estimate for analytic or exact-contact shortcuts in other geometries.

The median strict margin as a fraction of the threshold was 0.076, 0.156 and 0.116 for the most expensive quartile of 14-05, 14-13 and 14-14 respectively. For the added shadow-transition samples the medians were 0.0086, 0.0106 and 0.0132. The audit's accepted point parameters were interior. Endpoint, exact contact, near-source inner-ellipsoid, one-ULP and subnormal paths have separate numerical controls.

In the table, one geometrically eligible four-step direction per regular sample was selected by the number of future centers, without using their labels or costs. Only future centers with an original scalar HIT path count as potential savings. Counts are deduplicated across the three regular strata within each tile. Construction time covers successes and failures, and the full renderer separately measures actual skips.

| Tile | A / B successful trials out of 384 | A / B distinct potential scalar HITs | A / B construction, ms |
|---|---:|---:|---:|
| 14-05 | 276 / 281 | 1,057 / 1,076 | 4.54 / 4.61 |
| 14-13 | 286 / 294 | 1,091 / 1,121 | 4.55 / 4.59 |
| 14-14 | 352 / 344 | 1,224 / 1,194 | 4.58 / 4.52 |

On 14-13 the direct proof required 4.59 ms across the selected trials. The original scalar time represented by their covered centers was about 85 ms when counted by stratum. For one step the same block yielded about 22.5 ms of potential original work for 4.0 ms of construction. These audit timings do not deduct overlapping future seeds or substitute for complete-tile measurements.

The old region controls were measured on 1,862 matched one- and four-step axial intervals. The warm inner-box shadow and single-proxy tests succeeded 7 and 0 times. The old searched common-section test succeeded 1,212 times in 250.8 ms. A and B succeeded 1,514 and 1,468 times, in 24.6 and 24.2 ms. B certified 313 intervals missed by all three old controls. The two new bounds each use a scalar witness already available at dispatch. The end-to-end timings include discovering that witness.

The previous angular study improved expressiveness through shadow composition but gave a small primary cold gain and essentially unchanged independent time. The measured fixed-witness gain comes from using one already established local depth after the original region traversal has declined the receiver. [Previous angular study](GAUSSIAN_B2_ANGULAR_RESULTS.md)

## Complete-tile measurements

The primary measurements use fresh single-threaded sessions on the same Ryzen 7 5800H host with GCC 16.2.1 and MPFR 4.2.2. The independent cold time for the original workload comes from the unchanged archived F_J renderer on the identical inputs. F_J was rerun on all ten new tiles. Cold wall and CPU include source-bound cover preparation and writing and checking the traced tile archives.

### Original 24 tiles

| Primary | Cold wall / CPU, s | Primary + F_J cold wall, s | Peak RSS, MiB | HIT marks / actual skips | Original HIT seconds avoided | Top-quartile HIT skips / seconds avoided |
|---|---:|---:|---:|---:|---:|---:|
| P_J | 84.03 / 84.02 | 233.95 | 150.41 | 0 / 0 | 0.000 | 0 / 0.000 |
| P_JH1 | 77.38 / 77.36 | 227.31 | 152.96 | 147,127 / 145,201 | 11.018 | 31,732 / 3.493 |
| P_JH4 | 71.95 / 71.93 | 221.88 | 152.86 | 217,052 / 207,804 | 15.008 | 42,009 / 4.061 |
| P_JHA | 71.26 / 71.24 | 221.19 | 153.43 | 229,332 / 219,441 | 16.375 | 47,314 / 4.981 |

Independent F_J cold wall and CPU time were 149.93 s and 149.89 s on this workload.

| Tile | P_J | H1 | H4 | HA | HA actual HIT skips | HA original HIT seconds avoided |
|---|---:|---:|---:|---:|---:|---:|
| 14-00 | 0.006 | 0.007 | 0.007 | 0.007 | 0 | 0.000 |
| 14-01 | 0.004 | 0.005 | 0.005 | 0.005 | 0 | 0.000 |
| 14-02 | 0.002 | 0.002 | 0.002 | 0.002 | 0 | 0.000 |
| 14-03 | 0.002 | 0.002 | 0.002 | 0.002 | 0 | 0.000 |
| 14-04 | 3.546 | 3.415 | 3.416 | 3.379 | 2,252 | 0.243 |
| 14-05 | 16.171 | 14.738 | 13.815 | 13.413 | 51,571 | 3.485 |
| 14-06 | 0.002 | 0.002 | 0.002 | 0.002 | 0 | 0.000 |
| 14-07 | 0.002 | 0.002 | 0.002 | 0.002 | 0 | 0.000 |
| 14-08 | 0.002 | 0.002 | 0.002 | 0.002 | 0 | 0.000 |
| 14-09 | 0.002 | 0.002 | 0.002 | 0.002 | 0 | 0.000 |
| 14-10 | 0.007 | 0.007 | 0.007 | 0.007 | 0 | 0.000 |
| 14-11 | 0.109 | 0.106 | 0.106 | 0.106 | 0 | 0.000 |
| 14-12 | 4.484 | 4.114 | 3.774 | 3.766 | 12,340 | 0.920 |
| 14-13 | 13.843 | 11.772 | 10.189 | 9.882 | 62,749 | 4.880 |
| 14-14 | 14.794 | 13.696 | 12.542 | 12.543 | 40,481 | 2.853 |
| 14-15 | 0.002 | 0.002 | 0.002 | 0.002 | 0 | 0.000 |
| 7-ordinary | 2.222 | 2.003 | 1.789 | 1.788 | 7,098 | 0.545 |
| 7-hot | 4.252 | 3.884 | 3.697 | 3.642 | 6,366 | 0.747 |
| 60-ordinary | 1.032 | 0.943 | 0.899 | 0.876 | 1,475 | 0.181 |
| 60-hot | 8.297 | 7.977 | 7.711 | 7.840 | 3,561 | 0.533 |
| 63-ordinary | 1.620 | 1.440 | 1.320 | 1.303 | 6,366 | 0.412 |
| 63-hot | 3.301 | 2.988 | 2.739 | 2.728 | 6,857 | 0.675 |
| 72-ordinary | 1.309 | 1.213 | 1.116 | 1.109 | 4,822 | 0.277 |
| 72-hot | 1.907 | 1.715 | 1.504 | 1.479 | 13,503 | 0.625 |

### Ten new held-out tiles

| Primary | Cold wall / CPU, s | Primary + F_J cold wall, s | Peak RSS, MiB | HIT marks / actual skips | Original HIT seconds avoided | Top-quartile HIT skips / seconds avoided |
|---|---:|---:|---:|---:|---:|---:|
| P_J | 20.54 / 20.54 | 75.36 | 150.39 | 0 / 0 | 0.000 | 0 / 0.000 |
| P_JH1 | 20.02 / 20.02 | 74.84 | 150.58 | 15,389 / 13,961 | 1.200 | 2,910 / 0.322 |
| P_JH4 | 19.50 / 19.49 | 74.31 | 151.16 | 23,945 / 19,797 | 1.656 | 3,922 / 0.416 |
| P_JHA | 19.35 / 19.35 | 74.17 | 150.21 | 25,599 / 21,045 | 1.786 | 4,410 / 0.479 |

Independent F_J cold wall and CPU time were 54.82 s and 54.81 s on this workload.

| Tile | P_J | H1 | H4 | HA | HA actual HIT skips | HA original HIT seconds avoided |
|---|---:|---:|---:|---:|---:|---:|
| new-0-uniform | 0.001 | 0.001 | 0.001 | 0.001 | 0 | 0.000 |
| new-0-boundary | 3.023 | 2.984 | 3.018 | 2.979 | 1,147 | 0.097 |
| new-21-uniform | 0.048 | 0.048 | 0.048 | 0.048 | 0 | 0.000 |
| new-21-boundary | 3.681 | 3.322 | 3.023 | 3.004 | 11,280 | 0.877 |
| new-62-uniform | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0.000 |
| new-62-boundary | 3.242 | 3.075 | 2.914 | 2.901 | 4,124 | 0.431 |
| new-64-uniform | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0.000 |
| new-64-boundary | 0.938 | 0.898 | 0.888 | 0.870 | 1,498 | 0.109 |
| new-75-uniform | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0.000 |
| new-75-boundary | 2.224 | 2.151 | 2.079 | 2.036 | 2,996 | 0.273 |

All primary variants match the independent response and solid bits on every completed tile. The original workload covers 4,456,448 voxels and the new one covers 327,680. The original MISS seed arrays match P_J exactly for every mode. The 69,002 original-workload MISS skips therefore remain in place, representing 21.49 s of original uncached P MISS work in the previous matched trace. The HIT changes add zero MISS skips.

The new plan was frozen before the HIT audit or any live tuning. Cases 0, 21, 62, 64 and 75 are disjoint from the original 24-tile geometries. Each supplied a uniform origin and a different boundary-near origin selected from an archived coarse physical field. Uniform tiles include easy, mostly region-certified regions. Boundary tiles exercise different kernel counts, aspect ratios, occupancies and source distances. Full identities and selection inputs are retained in heldout-plan.json.

### Construction and cache accounting

| Mode, original 24 | HIT proofs passed / failed | HIT test time / failed-test time, s | HIT planning / indexing, s | Combined cache lookup, s | Original-path / MISS-extension / HIT-extension exp endpoints, M |
|---|---:|---:|---:|---:|---:|
| P_J | 0 / 0 | 0.000 / 0.000 | 0.000 / 0.000 | 0.647 | 66.28 / 4.22 / 0.00 |
| P_JH1 | 147,127 / 14,472 | 1.923 / 0.177 | 0.238 / 0.213 | 0.660 | 56.40 / 4.22 / 1.76 |
| P_JH4 | 58,782 / 36,855 | 1.145 / 0.447 | 0.162 / 0.097 | 0.659 | 52.95 / 4.22 / 1.04 |
| P_JHA | 73,085 / 18,981 | 1.101 / 0.230 | 0.153 / 0.118 | 0.659 | 51.65 / 4.22 / 1.00 |

Original-path exp counts include the existing region and scalar routines. Cache lookup time covers both label arrays when HIT reuse is active. Advancing the original J planner is also charged in the live cache-return path. A 64³ tile requires one additional 1 MiB HIT seed array, or 2 MiB with provenance tracing, plus up to 3 KiB of history in HA.

The capture-only control retained the witness on the same three hot tiles and did no neighborhood tests. Their combined warm times were 44.808 s for P_J and 44.600 s for capture only. The paired scalar capture audit and this replay show no resolved capture penalty at the observed timing variability.

| Hot tile | Four-step gradient A | Four-step direct B | Adaptive direct B |
|---|---:|---:|---:|
| 14-05 | 13.728 | 13.815 | 13.413 |
| 14-13 | 10.173 | 10.189 | 9.882 |
| 14-14 | 12.382 | 12.542 | 12.543 |

The gradient won narrowly in this matched fixed-length hotspot replay. The direct bound was stronger on some audited strata, especially 7-hot and 14-13, while the gradient certified more of the 60-hot and 63-hot samples. There is no universal bound winner. The best measured scheduling variant for the original octant maxima was adaptive direct B.

In the new holdout, scene 21's boundary tile had the largest measured gain. Setup and region filling dominate the millisecond uniform tiles. HA added about 1.06 ms on 14-00 and 0.33 ms on new-21-uniform, both with zero HIT skips. On every measured tile that built HIT extensions, HA's summed construction cost stayed below the original scalar HIT time actually avoided. The reported construction time charges failed long and short tests.

## Original 256³ admission

| Backend | Probe wall / CPU, s | Guarded estimate, s | Cap, s | Peak RSS, MiB | Admitted |
|---|---:|---:|---:|---:|---|
| Previous P_J | 54.31 / 54.30 | 429.34 | 360 | 129.80 | Fail |
| Previous F_J | 85.86 / 85.84 | 613.59 | 900 | 136.02 | Pass |
| P_JHA | 45.28 / 45.27 | 349.66 | 360 | 130.26 | Pass |
| F_J | 85.90 / 85.88 | 613.67 | 900 | 135.98 | Pass |

| Octant in probe order | P_JHA slowest tile / s | F_J slowest tile / s |
|---|---:|---:|
| 1 | 14-00 / 0.005 | 14-00 / 0.005 |
| 2 | 14-02 / 0.004 | 14-02 / 0.004 |
| 3 | 14-05 / 13.483 | 14-05 / 19.643 |
| 4 | 14-06 / 0.004 | 14-07 / 0.005 |
| 5 | 14-09 / 0.004 | 14-08 / 0.004 |
| 6 | 14-11 / 0.108 | 14-11 / 0.449 |
| 7 | 14-13 / 9.841 | 14-13 / 14.979 |
| 8 | 14-14 / 12.550 | 14-14 / 27.357 |

The primary's sum of observed octant maxima fell from 44.305 s to 35.997 s, a 8.308 s reduction under the unchanged estimator. The original paired probe matched on 4,194,304 valid voxels. Both resource gates passed, admitting the single complete-field attempt.

## Complete 256³ dual field

After admission, each backend completed the remaining 48 tiles using its own source-bound session. The checker streamed all 64 tile pairs and verified each archive SHA256, valid mask, response mask and solid mask. All 16,777,216 centers were valid in each backend, with zero physical bit mismatches and zero unresolved centers.

| Worker | Tiles | Total wall, s | Total CPU, s | Peak RSS, MiB | Tile archives, MiB | Slowest tile, s |
|---|---:|---:|---:|---:|---:|---:|
| primary | 64 | 155.37 | 155.35 | 130.26 | 0.15 | 13.48 |
| independent | 64 | 282.76 | 282.71 | 141.91 | 0.15 | 27.36 |

The two worker phases per backend include fresh source clearance, cover construction, rendering and archive writes. Their sequential wall times sum to 438.14 s. The complete primary run returned 681,882 cached HITs and 193,567 cached MISSes. The independent run returned 257,009 of its own cached MISSes. The original guarded-out P_J baseline has no matched complete-field timing.

## Decision and remaining cost

HA is the preferred optional primary research policy for follow-up work. Both A and B remain useful controls. The observed gain is substantial on the original hot tiles and smaller across the five newly held-out geometries. Keep P_J and F_J as scientific controls and preserve the published production paths.

On the three traced primary hotspots the remaining scalar MISS paths took 22.43 s, the remaining scalar HIT paths took 7.87 s, and region traversal took 2.74 s. These are the original trace categories after deducting actual cache returns. Remaining scalar timings include the respective proof/dispatch work. In the complete dual run the independent verifier consumed most of the combined worker time. The strongest next scaling test is a bounded spatially stratified 512-grid trial on multiple frozen scenes, profiling F_J's remaining MISS enclosures and both backends' costly octants before considering a full 1024³ field.

The full result is one B2 research geometry/source case. The geometry-family and READY publication gates retain their existing status.

## Validation and identities

The focused suite passed 26 tests including analytic interior and endpoint contacts, strict fusion-only witnesses, source-close inner ellipsoids, one-ULP and subnormal bounds, thin anisotropy, zero-length rays, unaligned deterministic tile replay and conflicting cache labels. The independent Arb path confirmed all 48 sampled actual cached centers and interior axial points as HIT.

The full regression passed 693 tests. Seven tests requiring restricted pilot or campaign artifacts were skipped.

The ignored measurement directory is outputs/gaussian-b2-hit-v1. It contains the frozen plans, build hashes, witness audit, paired traces, admission calculations, phase logs and the per-tile archive and physical digests. The compact field-verification record fingerprints the sorted list of tile origins and physical digests.

- Original plan SHA256 `07beb76045ea27372277038a9bb21fd5612fb39ebf597fc09a97d3dffa86e8fa`
- New holdout plan SHA256 `e256c9650257d6990012d06be6288831ce013d8b73944587e5efd0476ec8c1d1`
- Audit plan SHA256 `1ac75cc8f43e067798011499df8b7f2def37481e09c2a6d4f70c946853892146`
- Ordered complete physical tile fingerprint SHA256 `61efb04fd4e669ab156d7a7d27ccac7701e839a03cfb01fb6cd319d2dfe7376e`
- Matched measurements SHA256 `0ca439c787b3100b6ac06660b0b83061d30671a9760e2cb53655e50cb28d1881`
