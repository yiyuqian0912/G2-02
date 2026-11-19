# Local-first deterministic certification on v3 channels

The default configuration certified 11 of 12 real channels at TV ≤ 0.05 on a 10-second soft solver budget. Frozen CBC certified the same 11, with identical probability bytes on all 11. The new code certified every one of 14 archived exact comparison cases, including all four multicell cases. The exact higher-order patch test worked on real observations but added time on this panel. Keep spread adjacent endpoints as the research default. Treat the implementation guarantee as conditional on the numerical and software assumptions in the [mathematical specification](ASTRA_XLF_CERTIFIED_INTERFACE.md).

## Matched measurements

The fixed panel spans all nine polyhedral families and L = 64, 128, 256. It deliberately includes previously slow windows and both known legacy Tau-pencil regressions. The input is the same geometry, crop, placement and channel for each arm. A Ryzen 7 5800H ran one low-priority CPU worker with one numerical-library thread. New measurements include input decoding, search, conditional certificate construction, file write and readback. One warm-up per worker is reported separately. All attempts, including abstentions, count towards total time. This is a diagnostic panel of 12 channels on 11 geometries, not a random sample of the planned release.

| 12-channel arm | Certified | Total wall, s | Median, s | Slowest, s |
|---|---:|---:|---:|---:|
| Frozen CBC v3.0.1 | 11 | 34.92 | 1.100 | 11.79 |
| Input validation and exact predicates, before layout correction | 8 | 60.62 | 3.131 | 12.52 |
| Validated solver, layout-preserving copies and crop hashing | 11 | 33.55 | 0.473 | 10.52 |
| Final, homogeneous rational volume and adjacent endpoints | 11 | 29.46 | 0.433 | 11.23 |
| Final, plus projective patches to receiver depth 1 | 11 | 29.61 | 0.449 | 11.22 |
| Final, plus projective patches to receiver depth 2 | 11 | 30.07 | 0.502 | 11.22 |

The frozen numerical source is `2b05802`, with identical solver source in documentation commit `4ff88c7`. Its historical default native binary SHA256 was `abfdf4c6876f2536629c6b53c469391745bc40da0bdee6c167d1ac489fc3d437`. The final numerical build is `545ac17`. Its default uses Python integer and rational integration. A previous warm run of this build took 30.31 seconds for the default arm. The repeat archived here took 29.46 seconds, a 15.6% reduction against the frozen run. Only one complete archived run per arm is used for these comparisons. Quantiles on 12 observations are imprecise; the nearest-rank p95 here equals the maximum.

Worker warm-up was 4.28 s for frozen CBC and 3.59 s for the final three-arm worker. Total CPU for the final default arm was 29.46 s. The final three-arm process reached 1,130 MiB peak RSS, compared with 765 MiB for the frozen single-arm process. These figures include interpreter, compiled screens, temporary arrays and allocator retention across attempts. They do not identify isolated per-case or per-feature memory use.

For ten channels the stored v3.4 FastLabel record matched both input fingerprints, origin, size and channel. Historical statistical-label time totaled 22.95 s and all ten had statistical labels. The final deterministic arm took 28.81 s and certified nine of those ten at the short budget. Among the nine jointly successful cases, the corresponding totals were 17.59 and 17.34 s. The largest individual ratio among these nine was 9.54× on a wedge. The historical timing was not rerun under identical warm-up and CPU/GPU conditions. These comparisons do not establish a tenfold per-channel service guarantee or a production certification rate.

| Window | Side | Released cells | Total wall, s | TV upper bound |
|---|---:|---:|---:|---:|
| 000004_w00_s0, polyhedra | 64 | 214 | 0.26 | 5.851×10⁻¹⁵ |
| 000060_w11_s2, corridors | 256 | 1 | 0.50 | 0 |
| 000038_w06_s2, prisms | 128 | 82 | 0.16 | 0.009717 |
| 000007_w08_s2, barriers | 256 | 3,019 | 1.63 | 0.042205 |
| 000007_w09_s2, barriers | 256 | 3,410 | 2.86 | 0.046754 |
| 000000_w00_s0, rotated boxes | 64 | 81 | 0.12 | 2.364×10⁻¹⁵ |
| 000001_w02_s0, thin slabs | 64 | abstained | 11.23 | best diagnostic 0.592 |
| 000002_w04_s2, prisms | 128 | 121 | 0.27 | 3.586×10⁻¹⁵ |
| 000003_w09_s0, wedges | 256 | 1,166 | 8.93 | 0.045826 |
| 000005_w04_s1, multiscale union | 128 | 6,993 | 3.01 | 0.039320 |
| 000006_w04_s0, corridors | 128 | 31 | 0.13 | 8.543×10⁻¹⁶ |
| 000008_w00_s1, clusters | 64 | 121 | 0.37 | 3.574×10⁻¹⁵ |

Displayed successful bounds round outward. Each label stores its exact rational bound. Released cells count nonzero mass in p. Cells omitted from p may have feasible mass covered by its error allowance.

## What Local-first contributes

**Reduction of source space.** Holding spread and count fixed lets us vary adjacency. At 24 spread pairs, the 214-cell case retained 0.0001323 of the world after endpoint constraints, versus 0.02859 for spread opposite-class pairs. Times were 0.251 versus 0.841 s. The corridor case retained 1.54×10⁻¹¹ versus 6.57×10⁻⁴; times were 0.491 versus 4.482 s. At 24 randomly ordered pairs the 214-cell times were 0.233 s for adjacent endpoints and 1.453 s for independently sampled opposite classes. These seed volumes are floating summaries of the remaining convex pieces, before window exclusion. Counts of 8 versus 24, and each policy's complete timings, are in the control records. The unspread adjacent sample actually retained less volume on the 214-cell case than the spread adjacent sample. There is no universal best spread or count in this experiment.

**Compression of receiver constraints.** At depth two, two prism patches covering 51,350 FREE receiver obligations reduced to four extremal halfspaces. Three wedge patches covering 786,432 reduced to six. An obligation is counted per source piece and receiver; these are not necessarily distinct voxel indices across pieces. The highest mixed-patch direction rank observed per piece was one. No tested root completed the entire visibility obligation through the optional patch pass. The reversed orientation had an exact synthetic positive test but supplied no additional real chart in this panel. In the geometric control, three distinct rotated blockers supplied three independent pairs of cuts that successively reduced volume to 31/60, 43937/162000 and 491/3400. Duplicate cuts produced no reduction. Ambiguous blockers forced abstention from the single-blocker explanation.

**Output-oriented certification.** The closed-room 64³ fixture yields a certified single-cell posterior with zero TV. The corridor task likewise certifies unit mass in one cell. Their within-cell source position can remain uncertain. The volume ledger also lets the multicell solver stop with residual obligations if their contribution fits the reported TV allowance. All comparative arms used the same stopping theorem. This study therefore does not attribute a separate runtime multiplier to it.

The data support a local-to-global gain from using the locations of measured opposite-response neighbors. There is a stronger projective mechanism and its compression is real. On this panel the extra search for a collective explanation did not pay for itself. Its cuts only apply to particular source pieces; checking full-scene geometry and all remaining receiver obligations can still dominate. The current fallback conservatively rescans partially covered blocks. A useful future comparison would retain a compact mask of *proved* receiver blocks on descendants and charge both its maintenance and failed chart construction. A rank-three synthetic example alone is insufficient justification to enable the pass.

## Where the time goes

On the slow corridor input in the first repaired build, profiling attributed 3.05 s to array ravel operations, 1.48 s to conversion to bytes and about 1.54 s to copies and contiguous conversions. Preserving [z,y,x] storage through detached hints and hashing the identical [x,y,z] byte sequence in cache-sized slices removed this avoidable cost. The later integer-determinant fan retained the Cartesian Fraction result in differential tests. The table separates the layout change from the subsequent volume change. End-to-end improvements combine these changes with exact validation and higher build assurance; they are not an estimate of speedup attributable to the projective theorem.

The thin-slab channel contains **one inactive FREE receiver**, 261,124 active FREE receivers and 1,019 SOLID receivers. Only two adjacent pairs are available. After exact endpoint constraints there are 109 pieces with summed volume about 0.212 of the world. With a 45 s soft budget, the solve still abstained. It spent 26.48 s integrating grid mass and 47.51 s end to end. At that checkpoint the summed feasible lower credit was about 0.16360 and live volume about 0.03382. This gives a concrete obstruction to any proposed source localization trick: feasible lower volume of that size already requires over 343,000 source-grid cells of capacity 1/128³ in aggregate. Exact positive volume cannot be discarded to narrow the search. By contrast, a 45 s budget on the wedge took 9.15 s, of which 6.52 s went to receiver screening. The two tails have different physical and computational causes.

A second matched deadline check used the thin-slab FastLabel time of 5.611 s. With a 52 s soft limit the deterministic attempt took 54.79 s end to end, inside the 56.11 s engineering target. Its best diagnostic TV upper bound was 0.147003. An extended attempt consumed 122.42 s on a 120 s soft budget and still stopped at 0.094318, with about 0.17400 world-volume credited feasible and 0.01812 still live. Neither attempt issued a distribution. Those numbers are bounds on possible error, not measurements of actual error. This family has other easy windows in the exact-reference panel; the difficulty here is tied to the observation and wide support. A practical next experiment is to batch grid-credit aggregation and profile repeated snapshots on diffuse posteriors while comparing the same frontier and outward bounds. Run it on further minority-class windows before changing default search.

## Verification and handoff

The kernel, adversarial, archive and split tests passed **562/562**. The exact inverse-geometry, mixed-patch and ledger suites passed **38/38**. The existing CPU package suite passed 20 tests; two optional tests skipped and four GPU cases were deselected. The 16 existing kernel arithmetic lemmas, six fixed-point screen lemmas and four additional projective lemmas all passed Z3. The proof obligations are narrower than verification of Python or of Numba's generated machine code.

All 14 available fully resolved rational references from the two frozen rational-reference archives were compared after input-identity and artifact checks. Four were multicell, with 2, 4, 31 and 214 support cells. Every exact rational difference met its issued bound. On the 214-cell example the observed TV was 5.850004512461103×10⁻¹⁵, below the issued 231742554111251/39614081257132168796771975168. Seven successful timing-panel cases also agreed within their combined bounds with the archived comparison labels; their stored floating estimates do not constitute exact ground truth. The rational archives are conditioned on successful prior resolution and share some geometry primitives with this solver. They do not certify the solver on arbitrary inputs.

The [real example](../experiments/credit_solver/study/results/example_214.json) includes the actual sparse bytes, the exact TV bound, numerical build identity and input fingerprints. Its wire SHA256 is `6fa36134ded366874036cc619281c4532c9d8af0fedbe5dedcd54e0cbff183cd`. The scientific status is **conditional pass for a controlled CPU research canary**. Production adoption still needs review of the trusted implementation and build, memory measurement with independent workers, and held-out measurements including highly imbalanced windows and wide posterior supports. For a student, these labels describe its teacher's hidden-geometry-conditional posterior. Testing transfer to unseen families or counterfactual hidden obstacles is a separate learning experiment.

### Reproduction

Use the environment described in the [mathematical specification](ASTRA_XLF_CERTIFIED_INTERFACE.md). From `experiments/credit_solver`, set `CHECKOUT` and `FROZEN_CHECKOUT` to the two repository roots, `V3_WORK` to the v3 observation and reference directory, and `OUT` to a new local directory.

```sh
mkdir -p "$OUT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
python study/run.py "$FROZEN_CHECKOUT" "$V3_WORK" "$OUT/frozen.jsonl" --arms baseline
python study/run.py "$CHECKOUT" "$V3_WORK" "$OUT/final.jsonl" --arms patch
python study/run.py "$CHECKOUT" "$V3_WORK" "$OUT/controls.jsonl" --arms factorial
python study/run.py "$CHECKOUT" "$V3_WORK" "$OUT/tail.jsonl" --arms tail
python study/tail_budget.py "$V3_WORK" "$OUT/final.jsonl" "$OUT/large_support.jsonl"
python study/check_exact.py "$V3_WORK" "$OUT/exact_references.jsonl"
python study/summarize.py --results "$OUT"
```

Omitting the summary command’s directory argument selects the checked-in records. Each benchmark refuses to overwrite an existing output. The source coordinates are absent from the benchmark inputs used by the solver and from the released label example. Family, channel and crop identities allow an independent collaborator to repeat the comparisons.
