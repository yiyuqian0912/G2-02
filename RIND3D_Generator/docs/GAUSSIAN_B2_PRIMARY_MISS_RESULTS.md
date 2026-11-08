# B2 primary MISS reuse and adaptive distance results

The primary can reuse a closed scalar MISS proof. Adaptive axial extension reduced cold wall time by 15.2% on the primary and by 30.7% on the independent verifier across the matched 24 tiles. The original primary resource cap still determines the dual-field decision. [Proof, policy and commands](GAUSSIAN_B2_PRIMARY_MISS.md)

## Primary audit

The frozen case-14 sequence uses sixteen 64³ tiles on the 256³ grid. The three slowest original primary tiles are 14-05 at (64, 192, 0), 14-13 at (192, 128, 64), and 14-14 at (128, 192, 128). The other eight matched tiles are the previously frozen unaligned 32³ evaluation tiles from cases 7, 60, 63, and 72. J's constants were frozen before the current propagation runs. The previously identified 60-hot and 63-hot failures serve as regression checks. The original development and evaluation membership and the historical controls are recorded in the tile plan.

The fresh primary's sixteen traced tiles took 66.63 s, with 42.15 s in scalar MISS, 19.06 s in scalar HIT, and 3.33 s in region work. Separate cover construction was paid once for the source. The top three tiles consumed 55.68 s, with 34.37 s in MISS and 17.03 s in HIT. In 14-13, HIT alone cost 7.15 s, compared with 6.63 s in MISS. This limits the potential of a MISS-only change on that tile.

| Tile | Wall, s | Scalar MISS count / s | Scalar HIT count / s | Region, s | Largest hot component / hot rays |
|---|---:|---:|---:|---:|---:|
| 14-05 | 24.37 | 37,311 / 17.42 | 71,745 / 5.72 | 0.82 | 9,318 / 9,328 |
| 14-13 | 15.44 | 21,246 / 6.63 | 87,316 / 7.15 | 1.02 | 4,985 / 5,313 |
| 14-14 | 15.87 | 22,506 / 10.33 | 56,847 / 4.15 | 0.87 | 4,914 / 5,628 |

The hot sets are each tile's most expensive quartile of MISS receivers. On 14-05 the 20,493 face-adjacent hot pairs were 3.17 times the expectation of a uniform shuffle over the same MISS support. On 14-13 and 14-14 the ratios were 2.44 and 3.16.

The audit build uses a calibrated invariant cycle counter on a deterministic one-in-128 scalar sample for the MPFR exp time estimate. All scalar queries contribute to the per-label counts. The same instrumentation with capture enabled took 66.47 s versus 65.78 s without capture, a 0.69 s difference for the sixteen tiles. Both traversals agreed with the original bits and evaluation path and visited exactly 1,518,721 scalar parameter nodes. Capture retained usable closed partitions for all 114,244 original scalar MISS queries. The difference measures the audit capture trial including its instrumentation. The live render measurements also charge affine enclosure retention and the extension/cache path.

The case-14 primary cover took 2.03 s to construct in the audit. Scalar exp work dominated the three hotspots. The exp seconds below scale sampled Gaussian MPFR call time by the counted calls in each label class. Natural failures are intervals for which the directed sum left both decisions open. Taylor nonclosures include midpoint HIT witnesses.

| Tile | Scalar MISS / HIT exp endpoints, M | Sampled exp estimate MISS / HIT, s | Inconclusive natural MISS / HIT | Taylor nonclosures |
|---|---:|---:|---:|---:|
| 14-05 | 17.61 / 5.41 | 14.49 / 4.42 | 239,749 / 104,161 | 87,113 |
| 14-13 | 6.66 / 6.95 | 5.47 / 5.68 | 86,009 / 104,451 | 35,027 |
| 14-14 | 10.37 / 3.95 | 8.52 / 3.27 | 140,594 / 63,058 | 48,266 |

These sampled time estimates are instrumentation results. The untraced admission uses ordinary wall time. Scalar exact fallback exp calls were zero in these three profiles.

## Matched fields

All eight methods agree on 4,456,448 response and solid voxel pairs across 24 completed tiles. Cold totals include session and cover construction and writing and checking the trace archives. F, F_H and F_I are the archived matched runs on the same host, numerical policy, inputs, and tile order. P and the four new propagation runs were measured in this iteration. The small primary difference between I and J is one timing comparison.

Runs used one CPU thread on the Ryzen 7 5800H host with GCC 16.2.1 and MPFR 4.2.2. The frozen build policy disables fast math and floating-point contraction.

| Backend / policy | Cold wall / CPU, s | Eight unaligned tiles, s | Peak RSS, MiB | Actual MISS skipped | Original top-quartile skipped | Original MISS time avoided, s |
|---|---:|---:|---:|---:|---:|---:|
| P | 100.93 / 100.91 | 26.94 | 146.34 | 0 | 0 | 0 |
| P_H | 91.45 / 91.44 | 24.95 | 151.18 | 39,754 | 9,781 | 14.26 |
| P_I | 86.73 / 86.71 | 24.64 | 151.06 | 58,879 | 10,984 | 18.12 |
| P_J | 85.55 / 85.54 | 24.22 | 151.35 | 69,002 | 12,788 | 21.49 |
| F | 216.27 / 216.21 | 67.74 | 155.38 | 0 | 0 | 0 |
| F_H | 181.76 / 181.72 | 57.78 | 159.24 | 62,066 | 12,558 | 52.07 |
| F_I | 160.32 / 160.29 | 52.28 | 159.42 | 86,136 | 15,767 | 68.80 |
| F_J | 149.93 / 149.89 | 48.06 | 158.75 | 95,604 | 17,744 | 78.22 |

Actual savings join a cache return's exact receiver index to the original scalar trace. All methods retain the original scalar HIT counts within their own backend, 309,738 for primary and 60,247 for independent. The evaluation-only 21-tile wall times are 64.30, 59.79, 59.51 and 58.01 s for P, P_H, P_I and P_J. They are 141.85, 119.26, 108.65 and 99.73 s for F, F_H, F_I and F_J.

| Method | Counted original-path exp endpoints, M | Extra local exp endpoints, M | Total extension time, s |
|---|---:|---:|---:|
| P | 87.80 | 0 | 0 |
| P_H | 73.49 | 4.67 | 4.65 |
| P_I | 69.63 | 3.64 | 3.62 |
| P_J | 66.28 | 4.22 | 4.19 |
| F | 105.05 | 0 | 0 |
| F_H | 72.97 | 6.90 | 17.14 |
| F_I | 62.33 | 4.50 | 11.66 |
| F_J | 56.69 | 4.84 | 11.91 |

The primary original-path counter includes scalar, region, and membership exp calls during the tiles. F's original-path column counts its precision scalar exp calls. Cover setup is measured separately. Local extension counters include failures.

On 14-05, P_J avoided 23,345 actual MISS queries and 9.54 s of original scalar MISS work, including 4,831 top-quartile queries. The primary retained a 1,056-byte affine line payload per prospective seed at this kernel count. The bounded interval array has 256 entries. On this tile, J's long successes and failures cost 0.751 and 0.074 s respectively. The short successes and failures cost 0.185 and 0.079 s. Cache allocation took 0.097 ms. Lookup, direction dispatch, and marking took 0.151, 0.138, and 0.014 s. Of 25,651 marks, 23,345 reached the cache path. The untraced tile cache and history have bounded payloads of 1 MiB and 3 KiB respectively.

On the primary, I was faster than H on 14-05 and 14-13. H was faster on 14-14, where I's 24,951 failed long checks erased the benefit. J retained most of I's gains in the first two and cut the failures in 14-14 to 3,642. On 7-ordinary, P_J added about 0.19 s while skipping just 178 MISS rays. That is the largest absolute P_J regression in the matched tiles.

## Per-scene and per-tile wall time

The tables give warm tile wall seconds. Setup and archive costs appear in the cold totals.

| Scene | P | P_H | P_I | P_J | F | F_H | F_I | F_J |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gaussian_sum-007 | 8.061 | 7.080 | 6.634 | 6.568 | 27.217 | 22.039 | 16.140 | 15.442 |
| gaussian_sum-014 | 66.631 | 59.255 | 54.760 | 53.945 | 125.591 | 101.055 | 84.959 | 79.077 |
| gaussian_sum-060 | 10.004 | 9.296 | 9.750 | 9.397 | 19.628 | 17.533 | 19.268 | 16.812 |
| gaussian_sum-063 | 5.233 | 5.005 | 5.010 | 4.993 | 12.453 | 11.165 | 11.204 | 10.334 |
| gaussian_sum-072 | 3.641 | 3.568 | 3.248 | 3.266 | 8.445 | 7.047 | 5.670 | 5.478 |

| Tile | P | P_H | P_I | P_J | F | F_H | F_I | F_J | P_J / F_J actual skips |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 14-00 | 0.005 | 0.005 | 0.006 | 0.006 | 0.006 | 0.006 | 0.007 | 0.007 | 0 / 0 |
| 14-01 | 0.004 | 0.004 | 0.004 | 0.004 | 0.004 | 0.005 | 0.004 | 0.004 | 0 / 0 |
| 14-02 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0 / 0 |
| 14-03 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0 / 0 |
| 14-04 | 4.893 | 4.101 | 3.620 | 3.646 | 12.725 | 10.406 | 7.809 | 7.548 | 8,957 / 11,265 |
| 14-05 | 24.371 | 20.309 | 16.262 | 16.510 | 38.750 | 29.172 | 20.774 | 19.859 | 23,345 / 25,215 |
| 14-06 | 0.001 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0 / 0 |
| 14-07 | 0.001 | 0.002 | 0.002 | 0.002 | 0.004 | 0.003 | 0.004 | 0.004 | 0 / 0 |
| 14-08 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0 / 0 |
| 14-09 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0 / 0 |
| 14-10 | 0.006 | 0.007 | 0.006 | 0.007 | 0.047 | 0.047 | 0.046 | 0.042 | 4 / 22 |
| 14-11 | 0.098 | 0.107 | 0.100 | 0.108 | 0.633 | 0.598 | 0.462 | 0.453 | 245 / 575 |
| 14-12 | 5.932 | 5.324 | 4.575 | 4.545 | 15.228 | 12.186 | 8.981 | 8.769 | 7,720 / 9,410 |
| 14-13 | 15.444 | 14.709 | 13.934 | 14.115 | 24.617 | 20.575 | 15.607 | 15.001 | 11,411 / 14,047 |
| 14-14 | 15.867 | 14.677 | 16.240 | 14.993 | 33.565 | 28.046 | 31.252 | 27.378 | 4,665 / 8,080 |
| 14-15 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.003 | 0 / 0 |
| 7-ordinary | 2.076 | 2.206 | 2.140 | 2.261 | 12.540 | 10.426 | 6.889 | 6.583 | 178 / 6,052 |
| 7-hot | 5.986 | 4.874 | 4.493 | 4.307 | 14.678 | 11.614 | 9.251 | 8.859 | 4,211 / 4,888 |
| 60-ordinary | 1.184 | 1.081 | 1.015 | 1.040 | 3.623 | 3.178 | 2.370 | 2.351 | 915 / 1,586 |
| 60-hot | 8.819 | 8.215 | 8.735 | 8.356 | 16.005 | 14.355 | 16.898 | 14.461 | 692 / 1,975 |
| 63-ordinary | 1.859 | 1.709 | 1.656 | 1.637 | 4.089 | 3.265 | 2.501 | 2.449 | 2,501 / 4,811 |
| 63-hot | 3.374 | 3.296 | 3.354 | 3.356 | 8.364 | 7.899 | 8.703 | 7.885 | 232 / 1,251 |
| 72-ordinary | 1.761 | 1.636 | 1.336 | 1.318 | 3.430 | 2.981 | 2.290 | 2.202 | 3,208 / 3,309 |
| 72-hot | 1.880 | 1.933 | 1.912 | 1.948 | 5.014 | 4.066 | 3.380 | 3.276 | 718 / 3,118 |

## Long-attempt regressions

| Backend and tile | I long failures | J long failures | I / J long successes | I / J actual MISS skipped | H / I / J wall, s |
|---|---:|---:|---:|---:|---:|
| P 60-hot | 14,484 | 1,686 | 0 / 0 | 0 / 692 | 8.215 / 8.735 / 8.356 |
| P 63-hot | 6,545 | 1,218 | 0 / 0 | 0 / 232 | 3.296 / 3.354 / 3.356 |
| F 60-hot | 18,513 | 1,722 | 8 / 2 | 31 / 1,975 | 14.355 / 16.898 / 14.461 |
| F 63-hot | 14,718 | 1,670 | 28 / 2 | 107 / 1,251 | 7.899 / 8.703 / 7.885 |

The failed long tests are predictable from recent nearby failures. J reduces their frequency and recovers short opportunities. In these two regimes its elapsed cost is close to H. Across the eight unaligned tiles, J reduced warm wall time by 10.1% on the primary and 29.0% on the independent backend. The study includes different kernel counts and the prior difficult cases. The primary gain varies by tile, including the measured 7-ordinary regression.

## Original 256³ admission

The same paired-octant probe and the original 20% planning margin were used with fresh untraced sessions. The estimator charges two session builds, then 1.2 times eight times the sum of the slower sampled tile in each octant. It is a planning guard for work outside the observed sample.

| Backend / control | Probe wall, s | Guarded time, s | Cap, s | Peak RSS, MiB | Admission |
|---|---:|---:|---:|---:|---:|---|
| Original P | 67.59 | 530.64 | 360 | 130.07 | Fail |
| Original F | 128.53 | 918.26 | 900 | 134.84 | Fail |
| Previous F_I | 95.15 | 689.43 | 900 | 135.40 | Pass |
| P_J | 54.31 | 429.34 | 360 | 129.80 | Fail |
| F_J | 85.86 | 613.59 | 900 | 136.02 | Pass |

| Octant in probe order | Primary slowest tile / s | Independent slowest tile / s |
|---|---:|---:|
| 1 | 14-00 / 0.004 | 14-00 / 0.005 |
| 2 | 14-02 / 0.004 | 14-02 / 0.004 |
| 3 | 14-05 / 15.987 | 14-05 / 19.651 |
| 4 | 14-07 / 0.004 | 14-07 / 0.005 |
| 5 | 14-08 / 0.004 | 14-09 / 0.004 |
| 6 | 14-11 / 0.106 | 14-11 / 0.450 |
| 7 | 14-13 / 13.591 | 14-13 / 14.952 |
| 8 | 14-14 / 14.607 | 14-14 / 27.366 |

The fresh probes finished in 140.17 s across the two passes. Both independently rendered and validated 4,194,304 voxel pairs with zero response or solid mismatches. This is exactly 25% of the grid. Primary admission failed, so the 48 remaining tiles are outside the measured support.

## Decision and remaining cost

**Keep H, I and J as optional B2 research modes, with J the preferred next benchmark candidate.** J has the best observed cold aggregate on both backends and mitigates the known long-distance failure regime. Its small primary lead over I and the per-tile regressions support keeping P and F as the default controls.

**The complete 256³ dual-field experiment remains outside the original resource gate.** At the observed octant maxima, the primary needs roughly 7.22 s less summed maximum-tile time to clear the guard with the same setup cost. The three original hottest P_J tiles still spent about 22.93 s on remaining scalar MISS paths including extension work and 18.26 s on scalar HIT paths including dispatch. The original scalar profile locates the dominant numerical expense in directed Gaussian exp endpoints. A targeted next experiment should measure whether strict primary HIT witnesses can amortize the high HIT workload on 14-13, alongside the remaining scalar exp cost. The same independent checks and 256³ gate should precede any larger-grid attempt.

## Evidence and checks

The focused suite passed 24 tests. It exercises closed endpoint contact, a rotated narrow bridge with opposite adjacent labels, source-close clearance, a strict exact-fallback leaf, subnormal margin behavior, budget exhaustion and deterministic unaligned cache replay for both backends.

The full regression passed 667 tests. Seven tests requiring restricted pilot or campaign artifacts were skipped.

The raw measurement directory is outputs/gaussian-b2-primary-miss-v1. The input plan, source and binary fingerprints, profile and capture runs, per-tile traces, original scalar costs, exact bit hashes, admission decision and regression logs are retained there.

- Frozen tile plan SHA256 `07beb76045ea27372277038a9bb21fd5612fb39ebf597fc09a97d3dffa86e8fa`
- Primary audit binary SHA256 `d746067e81057ee367345478bfcaff91085c40f4e39df4c2f2f27bda37651d02`
- Primary and adaptive prototype binary SHA256 `c4099d6ac6619f906e4f0b982b780d0e005d03e0d73d6f242e911a57150884e7`
- Matched summary SHA256 `336977045bd051d603bc997e8e23b288a4d125b42647166e1ae32bf04ed926b5`
