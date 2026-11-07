# B2 MISS neighborhood experiment

The independent renderer can reuse strict scalar MISS proofs. The local axial certificate produced a measurable speedup on the frozen workload, with smaller gains or losses on particular geometries. The global gradient bound covered almost no useful neighbors. [Proof, implementation and commands](GAUSSIAN_B2_MISS.md)

## Frozen workload

The experiment used the original case-14 source, geometry, and sixteen 64³ tiles from the 256³ admission probe. Three of these tiles were designated for development before the gradient experiments. The remaining thirteen and eight new unaligned 32³ tiles form the evaluation set. The new tiles use the same 256³ world lattice and fixed lower-half and upper-quartile trace draws from cases 7, 60, 63, and 72. The five scenes contain 11, 16, 19, 10, and 8 kernels respectively. Selection used previous trace strata, with a frozen seed and exact input and archive hashes in `tile-plan.json`. No receiver labels or previous timing enter the live dispatch.

All traced runs use one CPU thread, the same source-bound cover policy, and the frozen per-ray limits on the Ryzen 7 5800H host with MPFR 4.2.2. Tile time includes traversal and bit assembly. Cold totals include constructing all five sessions, building their covers, writing compressed arrays and summaries, and checking matched bitmaps. These are elapsed measurements on this machine. Millisecond tiles should be interpreted at that scale.

## Hotspot and proof margins

The F reproduction matched the old 16-probe bits. Its case-14 tile at `(64,192,0)` took 38.75 s with tracing, compared with 37.53 s in the old untraced probe. Existing region certificates filled 218,744 of 262,144 voxels. The residual contained 37,311 MISS rays taking 33.25 s and 6,089 HIT rays taking 2.38 s. Median and P95 MISS times were 0.863 and 1.328 ms. The solver visited 608,644 parameter intervals with 288,368 splits, 19.57 million 64-bit exp endpoints, and 2.49 million 256-bit endpoints.

The expensive MISS rays were strongly clustered. The top quartile contained 9,328 rays and 32.6% of MISS time. The largest face-connected component contained 8,446 of those rays and 91.6% of their cost. There were 18,637 hot-to-hot face pairs versus 6,463 expected from uniformly shuffling hot labels over the fixed MISS support. A MISS face neighbor of a hot ray was hot 76.8% of the time. Median angular separation between hot face neighbors was 0.239° and P95 was 0.296°. The moderately costly development tile at `(0,128,64)` had a largest hot component of 4,255 out of 4,405 rays.

The hot tile spent 2.93 s in region tests. It visited 9,657 field cells and 549,197 cover-volume nodes, with 1,334 whole-volume FREE closures and no cached inner-SHADOW closure. The latter is specific to the cover pass. The existing cross-section certificates also closed shadow regions. The case-14 cover took 7.33 s to build cold, with 92,865 nodes and a 15.1 MiB payload. Of its 81,257 leaves, 35,120 were UNKNOWN at the configured spatial limit. UNKNOWN cover boxes can block a whole-volume proof even when the corresponding individual ray is FREE. This count measures inconclusive cover boxes under the configured spatial bound. The reused tile calls made no new cover-build queries.

The global bound audit used separate samples of residual MISS cost strata. A receiver count of one means that the weighted stencil certified only the seed.

| Group | Proofs | Seed only | 2–8 receivers | Median cube radius, voxels | P95, voxels |
|---|---:|---:|---:|---:|---:|
| Moderate development, remaining | 128 | 128 | 0 | 0.0441 | 0.145 |
| Moderate development, most expensive quartile | 128 | 128 | 0 | 0.0337 | 0.131 |
| Hot development, remaining | 512 | 502 | 10 | 0.0827 | 0.276 |
| Hot development, most expensive quartile | 512 | 512 | 0 | 0.0319 | 0.191 |
| All evaluation strata | 3,144 | 3,133 | 11 | — | — |

No sampled weighted stencil certified more than eight centers. The largest single-axis radius in the hot expensive quartile was 0.989 voxels. The Spearman correlation of cube radius with scalar time there was −0.575. Its median minimum strict certified margin lower bound was 0.0162 in field units. The corresponding median upper bound on the true margin from F's witnessed lower values was 0.0600. These margins bracket the continuous-ray maximum via available enclosures. The median limiting upper parameter endpoint was 0.5625, and the median retained partition had nine intervals. The gap and the narrow anisotropic kernels justified measuring a tighter local derivative.

## Development coverage and construction economics

The local test used up to 96 fixed flat-order seeds from each sampled development stratum and tested each legal axial direction for one or four voxel steps. In the hot expensive stratum, 320 of 566 one-step attempts and 231 of 529 four-step attempts were certified. Under the unchanged traversal, four-step attempts with a future center cost 0.064 s in total and covered 704 distinct future residual MISS rays representing 0.655 s of original work. The same calculation for one step gave 162 future MISS rays and 0.172 s of original work for 0.047 s of construction. These coverage figures replay the seed extensions independently. The live measurements account for skipped seeds and overlaps between their candidate extensions.

The old from-scratch mode-2 volume control was measured on 96 identical sampled axial boxes. Its dispatch includes the old fallback. The measured comparison is conditional on F already having solved the seed, and the live end-to-end experiment charges that seed solve.

The old volume method certified 30 of the 96 sampled boxes. The local extension certified 69, including 24 also certified by the old method. Median per-box times were 1.720 ms and 0.180 ms. Total costs were 0.184 s and 0.0144 s. This comparison uses the old method's complete dispatch and the local method's incremental extension cost.

## Matched rendering

All methods finished the same 24 tiles and agreed on every response and solid bit across 4,456,448 evaluated voxels. Cold totals include all sessions, including those whose tiles were nearly free. Exp endpoints below count the precision scalar solver and local extension. The unchanged region and point phases can perform additional exponentials.

| Method | Warm tile wall / CPU, s | Cold wall / CPU, s | 16 case-14 tiles, s | Other 8 tiles, s | Original MISS avoided, s | Scalar / local exp endpoints, M | Peak RSS, MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| F | 193.33 / 193.29 | 216.27 / 216.21 | 125.59 | 67.74 | 0 | 105.05 / 0 | 155.38 |
| G | 194.33 / 194.28 | 217.32 / 217.25 | 126.04 | 68.30 | 0.95 | 104.46 / 0 | 160.34 |
| H | 158.84 / 158.81 | 181.76 / 181.72 | 101.06 | 57.78 | 52.07 | 72.97 / 6.90 | 159.24 |
| I | 137.24 / 137.21 | 160.32 / 160.29 | 84.96 | 52.28 | 68.80 | 62.33 / 4.50 | 159.42 |

The three extensions avoided 1,325, 62,066, and 86,136 actual original MISS queries respectively. Of these, 31, 12,558, and 15,767 were in their tile's most expensive quartile. Original-time sums map actual skipped center indices to F's recorded scalar nanoseconds. On the 21 evaluation tiles alone, F, G, H, and I took 141.85, 143.61, 119.26, and 108.65 seconds respectively. I reduced total cold independent wall time by 25.9%. Its net scalar-plus-extension exp count fell by 36.4%.

| Hot tile method | Tile wall / CPU, s | Cold session + tile wall, s | Actual MISS skipped | Original MISS seconds skipped | Original top-quartile skipped | Scalar 64-bit / 256-bit / local exp endpoints, M |
|---|---:|---:|---:|---:|---:|---:|
| F | 38.75 / 38.74 | 46.08 | 0 | 0 | 0 | 19.57 / 2.49 / 0 |
| G | 38.03 / 38.02 | 45.10 | 863 | 0.72 | 26 | 19.15 / 2.45 / 0 |
| H | 29.17 / 29.17 | 36.26 | 16,075 | 13.91 | 3,443 | 11.83 / 1.55 / 1.74 |
| I | 20.77 / 20.77 | 28.05 | 24,567 | 21.04 | 4,842 | 7.77 / 1.12 / 1.00 |

On the hot tile, I avoided 63.3% of the 33.25 s of original MISS work and reduced measured tile wall time by 46.4%, including its new work. The region phase changed from 2.93 to 2.89 s. F's original scalar HIT phase was 2.38 s and the HIT count remained 6,089. I's remaining scalar MISS path, including capture and extension, took 15.13 s. Its 9,105 successful and 6,761 failed extensions took 2.48 s combined. It marked 31,885 future or as-yet-unvisited centers, of which 24,567 actually reached the cache before their otherwise necessary scalar solve. The 1,074 marks in G led to 863 actual skips, mostly outside the expensive quartile.

I spent 0.063 s in timed cache lookup, 0.031 s selecting directions, and 0.015 s marking cache entries. Allocating the traced cache took 0.13 ms and 2 MiB, including trace indices. The untraced cache uses 1 MiB per 64³ tile. The closure-capture timing sample recorded 0.775 ms for 465 leaves, about 0.19 s when scaled to the 113,818 captured leaves. This is a sampled estimate of the closure hook only. Remaining enclosure copies, setup, timers, and dispatch are charged in the end-to-end measurements. The retained interval array peaked at 768 bytes on this tile. G's cold gradient preparation took 0.11 ms and its global stencil tests took 0.073 s. Proof capture and unsuccessful lookups still carry a cost when almost no rays can be saved.

### All tile comparisons

Times are wall seconds with tracing. The last two columns report actual I savings against F's original scalar trace.

| Tile | Origin | Split | F | G | H | I | I MISS skipped | Original MISS time avoided |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 14-00 | (64, 0, 0) | development | 0.006 | 0.006 | 0.006 | 0.007 | 0 | 0.0% |
| 14-01 | (64, 64, 0) | evaluation | 0.004 | 0.004 | 0.005 | 0.004 | 0 | 0.0% |
| 14-02 | (64, 0, 128) | evaluation | 0.002 | 0.002 | 0.002 | 0.002 | 0 | 0.0% |
| 14-03 | (64, 64, 128) | evaluation | 0.002 | 0.002 | 0.002 | 0.002 | 0 | 0.0% |
| 14-04 | (0, 128, 64) | development | 12.725 | 12.686 | 10.406 | 7.809 | 10,968 | 63.2% |
| 14-05 | (64, 192, 0) | development | 38.750 | 38.028 | 29.172 | 20.774 | 24,567 | 63.3% |
| 14-06 | (0, 128, 192) | evaluation | 0.002 | 0.002 | 0.002 | 0.002 | 0 | 0.0% |
| 14-07 | (64, 192, 192) | evaluation | 0.004 | 0.004 | 0.003 | 0.004 | 0 | 0.0% |
| 14-08 | (128, 0, 0) | evaluation | 0.002 | 0.002 | 0.002 | 0.002 | 0 | 0.0% |
| 14-09 | (192, 0, 64) | evaluation | 0.002 | 0.002 | 0.002 | 0.002 | 0 | 0.0% |
| 14-10 | (192, 64, 128) | evaluation | 0.047 | 0.048 | 0.047 | 0.046 | 16 | 23.8% |
| 14-11 | (128, 64, 128) | evaluation | 0.633 | 0.652 | 0.598 | 0.462 | 558 | 50.4% |
| 14-12 | (192, 128, 0) | evaluation | 15.228 | 15.303 | 12.186 | 8.981 | 9,247 | 61.5% |
| 14-13 | (192, 128, 64) | evaluation | 24.617 | 25.072 | 20.575 | 15.607 | 13,483 | 60.6% |
| 14-14 | (128, 192, 128) | evaluation | 33.565 | 34.221 | 28.046 | 31.252 | 4,361 | 17.6% |
| 14-15 | (192, 128, 192) | evaluation | 0.002 | 0.002 | 0.002 | 0.002 | 0 | 0.0% |
| 7-ordinary | (66, 212, 90) | evaluation | 12.540 | 12.693 | 10.426 | 6.889 | 5,719 | 60.8% |
| 7-hot | (52, 164, 224) | evaluation | 14.678 | 14.756 | 11.614 | 9.251 | 4,561 | 53.9% |
| 60-ordinary | (137, 89, 91) | evaluation | 3.623 | 3.688 | 3.178 | 2.370 | 1,565 | 53.2% |
| 60-hot | (165, 197, 67) | evaluation | 16.005 | 16.287 | 14.355 | 16.898 | 31 | 0.3% |
| 63-ordinary | (189, 57, 3) | evaluation | 4.089 | 3.972 | 3.265 | 2.501 | 4,676 | 59.9% |
| 63-hot | (85, 201, 27) | evaluation | 8.364 | 8.324 | 7.899 | 8.703 | 107 | 1.1% |
| 72-ordinary | (13, 141, 103) | evaluation | 3.430 | 3.522 | 2.981 | 2.290 | 3,241 | 60.3% |
| 72-hot | (73, 0, 43) | evaluation | 5.014 | 5.056 | 4.066 | 3.380 | 3,036 | 54.2% |

The four-voxel certificate regressed two evaluation hotspots. On case 60 it certified eight extensions after 18,521 attempts and avoided 31 MISS rays, worth 0.038 s of original work. Construction took 0.892 s. On case 63 it certified 28 extensions after 14,746 attempts and avoided 107 MISS rays, worth 0.051 s. Construction took 0.422 s. H reduced both tile times, to 14.36 s and 7.90 s respectively. H also beat I on case-14 tile 14-14. The signed local bounds were sufficiently sharp to save time across most tiles, with a measurable distance-dependent failure mode.

## Independent checks and 256³ admission

The primary also rerendered the eight unaligned evaluation tiles. Their 262,144 response and solid bits matched F and all three extension modes. The 16-tile untraced probe below separately matched the original frozen bits and the newly rendered primary and independent fields. The focused suite checks all 17 previous adversaries with A, D, F and Arb. It also checks a rotated narrow bridge between adjacent rays, source clearance close to a boundary, an exact receiver-end contact, subnormal margin rounding, analytic and aborted rays, and deterministic replay on an unaligned tile. For the exact summed-Gaussian endpoint contact, Arb reached its depth budget and the proposed extension was rejected. With a strictly blocked adjacent center the canonical controls return HIT. All 23 focused tests passed.

The exact same 16-tile order, two observations per octant, estimator, memory limit and per-backend time caps were used for the new admission probe. The guard uses the slower observed tile per octant with the original 20% planning margin. This is the same planning criterion for the unseen runtime.

| Backend | Completed probe wall / CPU, s | Unguarded projected wall, s | Guarded planning time, s | Cap, s | Peak RSS, MiB | Admission |
|---|---:|---:|---:|---:|---:|---|
| Unchanged primary P | 67.59 / 67.57 | 266.09 | 530.64 | 360 | 130.07 | Fail |
| Independent F + I | 95.15 / 95.11 | 365.26 | 689.43 | 900 | 135.40 | Pass |

The paired sample contains 4,194,304 valid voxels, exactly 25% of the 256³ world, with zero response or solid mismatches. The fresh dual probe took 162.74 s across the two passes. The historical untraced P + F probe took 194.69 s on the same support. The independent guard fell from 918.26 s to 689.43 s. The primary's guarded estimate remains over its cap, so the full dual field is not admitted. All remaining tiles are outside the measured support.

The primary spent 54.74 of its 65.50 tile seconds on three sampled tiles. The respective times were 24.11, 15.43, and 15.21 s. This concentration drives the primary planning guard. The regression run passed 547 cases, and the 96 old independent controls passed with their frozen A/D build selected explicitly. Seven restricted-artifact tests were skipped.

## Decision

**GO for optional independent local propagation in the B2 research renderer.** The hot-tile speedup exceeded the 20% milestone and the evaluation tiles improved in aggregate. The single-step H mode is the shorter-range control for cases where I's four-voxel proposals repeatedly fail. F stays the default and the unchanged matched control. G has insufficient useful radius and negligible savings at the tested spacing. I has the best aggregate timing and measured regressions on two evaluation hotspots, so automatic use across all geometries needs more evidence.

**NO-GO for the full 256³ dual-field run under the current caps.** The primary's concentrated expensive tiles now determine admission. On the independent side, residual scalar intervals still dominate and near-boundary four-voxel extensions sometimes repay almost none of their cost.

Raw run identities, timings, traces, masks, audit samples and the admission decision are under `outputs/gaussian-b2-miss-v1`. The aggregate is `matched-summary.json`. Frozen source/geometry identity and original archive checksums are in `tile-plan.json` with SHA256 `07beb76045ea27372277038a9bb21fd5612fb39ebf597fc09a97d3dffa86e8fa`. The matched native extension has SHA256 `416cf52a58f973b82f81769536f4437feba2fc001c35146b668a2c8932723932`. Results and builds use research-only Gaussian inputs.
