# B2 independent verifier optimization results

This study starts from the frozen `9ac08749130eaadb6b8e2b57f6e7cb72a4e57281`
reference and uses only continuous positive Gaussian-sum research geometry.
The [method derivation](GAUSSIAN_B2_INDEPENDENT.md) and
[reproduction commands](../experiments/gaussian_geometry/b2_independent/README.md)
describe the proof conditions and independent search.

## Comparison protocol

Timings were collected on an AMD Ryzen 7 5800H with Linux x86-64, Python
3.12.14 and GCC 16.2.1, with `-fno-fast-math` and `-ffp-contract=off`.
Runs use one active measurement process, a single native evaluation thread
and single-threaded numerical library settings. Three ray trials rotate
method order. Complete field timings are single fresh-process measurements
per method and workload. N³ denotes a full unit-world sampling resolution.
Voxel centers, source coordinates, region cover settings, proof budgets and
canonical geometry are fixed within each comparison. All coordinates come from synthetic B2 research inputs.

A is the optimized, original 256-bit independent verifier. B changes only
the natural prefix to 64-bit directed MPFR, with 256-bit continuation.
C tries signed local curvature at 256 bits after natural, midpoint and
global Taylor tests. It also avoids repeated full-exp point checks when
the saved midpoint enclosure already proves that the midpoint is below
threshold. D combines B and C. Optional 128-bit and stationary-localization
ablations use the same inputs and budgets. The development split of the
original hard corpus was evaluated first. The implementation and default
dispatch were frozen before any held-out geometry timings.

## Complete-field baseline audit

The unchanged 256-bit implementation already uses both natural interval
sides, signed midpoint derivatives, exact GMP derivative coefficients,
a global absolute curvature bound and an independent breadth-first search.
It caches exact inverse enclosures and uses conservative positive tails.
Its region proof tree and inherited membership or visibility decisions
discharge most receivers. The original Taylor test proves 17,933
intervals in the four complete 64³ runs. The missing techniques are a
signed **local** enclosure of the full curvature sum, a concave quadratic
maximum and a validated signed derivative or stationary test.

The complete original support of 73,368 residual rays shows where time
goes. The fast and slow MISS halves use the measured per-scene latency
median, with equal populations up to rounding. They include every MISS.
Seconds below come from the final instrumented run.

| Case at 64³ | Region-filled voxels | Faster half of MISS s | Slower half of MISS s | HIT s |
|---|---:|---:|---:|---:|
| 0 | 252,872 | 2.720 | 8.747 | 2.861 |
| 7 | 240,224 | 9.981 | 29.399 | 7.643 |
| 14 | 241,336 | 5.203 | 16.705 | 2.941 |
| 21 | 230,896 | 6.473 | 21.998 | 8.043 |

| Case | Scalar ray s | Natural interval s | Midpoint bound s | Taylor s | Line s | Exact derivative setup s | Other ray s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 14.307 | 5.750 | 3.387 | 1.071 | 1.113 | 2.666 | 0.321 |
| 7 | 46.979 | 23.263 | 12.017 | 5.257 | 1.889 | 3.772 | 0.781 |
| 14 | 24.815 | 12.324 | 6.288 | 2.684 | 1.025 | 1.909 | 0.586 |
| 21 | 36.450 | 16.208 | 9.019 | 2.958 | 2.589 | 4.785 | 0.889 |
| Total | 122.551 | 57.545 | 30.711 | 11.970 | 6.616 | 13.131 | 2.578 |

Other ray work includes queue traversal, 451,819 splits, termination checks
and measurement overhead. The 36.18 million ray-only exp endpoints consume
70.17 s, or 57.3% of scalar time. They are nested inside the bound columns.
Likewise 41.59 s is time inside inconclusive bounds, and 12.32 s is time
inside the 158,577 full-exp retries. The two latter diagnostics overlap
the bound columns and can overlap each other. This locates most cost in
repeated Gaussian enclosure, especially for slower MISS rays. Exact GMP
ray derivative preparation is material but smaller. The initial 256-bit
baseline has a fixed precision, so precision escalation cannot explain
its long tail.

The final profiler uses a calibrated invariant cycle counter on one
logical CPU. Its nested totals match independently measured per-ray
wall time to within 0.18% in each case. Field wall times are within 3%
of the uninstrumented runs. Sparse timer extrapolation was inconsistent
on this host. Direct HPET reads cost about 1.5 μs apiece and materially
distort per-exp timings. Unpinned cycle timing also failed the nested
time check. Those pilot records are retained separately. The headline
A/B/C/D timings use the uninstrumented build.

## Scalar rays

There are 1,398 distinct exact frozen queries across the cost-stratified,
cost-unselected residual and uniform FREE corpora. These are all on the
four previously studied geometries. Twenty-four hash-selected queries also
agree with the unchanged, independently searching Arb evaluator. Each
method certified every corpus query. All certified labels agree. The invalid and unresolved counts are zero.

Times below are the sum of per-query median native seconds across three
trials. The selected expensive MISS column is an overlapping subset of the original
374-query corpus.

| Method | Original 374 | Expensive MISS 96 | Cost-unselected residual 512 | Ordinary FREE 512 |
|---|---:|---:|---:|---:|
| A | 2.62367 | 1.80368 | 0.96327 | 0.20322 |
| B | 2.50973 | 1.75882 | 0.81675 | 0.14480 |
| B128 | 2.55611 | 1.77564 | 0.88158 | 0.16804 |
| C | 0.70576 | 0.33208 | 0.58992 | 0.18348 |
| C_Newton | 0.71956 | 0.33681 | 0.58962 | 0.18462 |
| D | 0.60449 | 0.29616 | 0.44668 | 0.12497 |
| D128 | 0.65423 | 0.31375 | 0.51093 | 0.14635 |
| D_Newton | 0.62068 | 0.30282 | 0.44583 | 0.12616 |

On cost-unselected residuals, D cuts elapsed native time by **53.6%**.
On ordinary FREE receivers it cuts time by **38.5%**. The stationary
variant is nearly tied on residuals and loses on the expensive and
ordinary corpora. The expensive subset's 6.09× result measures a selected
stratum only. Original A on all 314 stratified MISS rays reproduces the
old cost (2.3765 s versus 2.386 s reported previously).

| Population | Method | MISS seconds | HIT seconds | MISS P50 / P95 / P99 / max single ms | MISS interval visits | MISS directed exp endpoints |
|---|---|---:|---:|---|---:|---:|
| Stratified | A | 2.3765 | 0.2472 | 3.080 / 27.687 / 35.503 / 40.763 | 48,054 | 2,452,230 |
| Stratified | D | 0.4944 | 0.1101 | 1.271 / 4.540 / 5.744 / 6.228 | 14,484 | 552,300 |
| Cost-unselected residual | A | 0.8579 | 0.1054 | 1.782 / 6.171 / 9.988 / 14.219 | 19,584 | 778,488 |
| Cost-unselected residual | D | 0.3876 | 0.0591 | 0.931 / 2.187 / 2.563 / 3.602 | 13,734 | 476,160 |
| Ordinary FREE | A | 0.1787 | 0.0245 | 0.194 / 1.154 / 2.640 / 6.270 | 4,617 | 133,572 |
| Ordinary FREE | D | 0.1104 | 0.0146 | 0.146 / 0.737 / 1.265 / 2.499 | 4,353 | 120,330 |

Work counts sum the three trials, with the same query count on each row.
The cost-unselected MISS work drops by 29.9% in visited intervals and
38.8% in exp endpoints. On ordinary rays, B uses the same number of
intervals and exp endpoints as A; most of its gain is lower cost per
operation. B and D reuse the same immutable inverse preparation as A.
The cold native geometry constructors measured 0.26–0.52 ms per scene.
Each precision escalation rebuilds that ray's line intervals and is
charged in the timing. C/D avoid exact per-ray derivative-coefficient
recomputation; the one-time geometry inverses remain exact GMP.

The full evidence records CPU seconds, per-label latency, total
derivative work, successful and failed added tests and fallback counts.
A high exp or node reduction can coexist with a small timing gain when
setup and region work dominate. The field comparison measures this
directly.

## Complete frozen fields

Each of the 40 complete records covers the same 32³ or 64³ sample centers
for its geometry and source. Primary P, original independent A, and B/C/D
agree on all response and solid bits. B/C/D also have exactly the same
independent receiver evaluation paths as A, including which points are
certified by regions and which need a scalar ray. Each method covers
1,179,648 valid voxels across the eight supports.

The following cold seconds include source clearance, engine construction,
field traversal, bit and diagnostic array assembly, and trace output.
Fresh-process wall time, CPU time and peak RSS are recorded separately.

| Case | Grid | Original P | Original A | B | C | D | Independent reduction | P + independent reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 32³ | 0.788 | 3.546 | 3.257 | 2.880 | 2.606 | 26.5% | 21.7% |
| 7 | 32³ | 2.382 | 12.155 | 10.724 | 10.107 | 8.633 | 29.0% | 24.2% |
| 14 | 32³ | 2.727 | 11.641 | 10.781 | 10.398 | 9.626 | 17.3% | 14.0% |
| 21 | 32³ | 1.996 | 9.097 | 8.216 | 7.611 | 6.642 | 27.0% | 22.1% |
| 0 | 64³ | 2.143 | 16.988 | 15.344 | 11.244 | 9.607 | 43.5% | 38.6% |
| 7 | 64³ | 9.023 | 55.534 | 46.369 | 37.066 | 28.728 | 48.3% | 41.5% |
| 14 | 64³ | 6.429 | 33.884 | 29.287 | 24.757 | 20.383 | 39.8% | 33.5% |
| 21 | 64³ | 6.510 | 42.459 | 36.504 | 29.711 | 24.153 | 43.1% | 37.4% |

At 32³, A totals 36.44 s and D 27.51 s, a 24.5% reduction. The required
P + independent total goes from 44.33 s to 35.40 s, down 20.1%.
Case 14's cover and region work alone takes about 7.8 s at 32³ and limits
its benefit. At 64³ the totals are 148.87 → 82.87 s for independent
rendering (**44.3% lower**) and 172.97 → 106.98 s for P + independent
(**38.2% lower**). Charging imports and verification of checkpoint artifacts
too gives 179.16 → 112.98 s for the 64³ paired total. Peak process RSS
across these records was 139.61 MiB.

| Case at 64³ | Region-filled voxels | MISS count | A → D MISS s | HIT count | A → D HIT s | A → D cover + region s |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 252,872 | 6,396 | 11.715 → 5.735 | 2,608 | 2.928 → 1.614 | 2.152 → 2.072 |
| 7 | 240,224 | 11,528 | 39.795 → 16.807 | 8,345 | 7.742 → 4.262 | 7.415 → 7.093 |
| 14 | 241,336 | 10,093 | 21.096 → 8.943 | 4,937 | 2.834 → 1.600 | 9.130 → 9.016 |
| 21 | 230,896 | 17,413 | 28.224 → 13.438 | 12,048 | 7.961 → 4.626 | 5.792 → 5.622 |

The original residual MISS times total 100.83 s versus 44.92 s in D. HIT
times total 21.46 s versus 12.10 s. On the same four complete 64³ runs,
A visits 943,138 ray intervals and D 709,550. Field-level exp calls
include the common cover and region cost. They fall from 41.88 million
directed endpoints to 30.00 million, down 28.4%. B retains A's interval
and exp counts. D performs 504,918 low-precision node visits and 41,618
ray precision escalations out of 73,368 residual queries. Escalation
increases the number of line/kernel preparations from 1,114,297 to
1,751,863, and the speedup already includes this extra work.

D performs 31,861 interval coefficient preparations in place of 33,340
GMP ray derivative preparations in A. It attempts 95,867 signed local
bounds. There are 12,260 concave quadratic proofs, 40,486 additional
local Taylor proofs, 2,971 endpoint decisions and 40,150 failed local
tests. The global Taylor pretest never succeeds on these D fields.
The full-field D path invokes zero final old-ray fallbacks. These counts
show where proofs are discharged. C and D change both the bound and
the handling of redundant midpoint retries, so the total gain cannot be
attributed to curvature alone.

## Numerical adversaries

The 17 fixed adversaries include exact and near tangency, one-ULP threshold
margins, analytic endpoint contact, multiple peaks, an ill-conditioned
transform, thin rotated fused kernels, mixed scales, subnormal amplitudes
and thresholds, and a zero-length ray. Every certified result agrees
with its expected mathematical label. Arb retains its original distinct
budget-exhaustion status. The following remain UNKNOWN in A and D under
2,048 visits and 0.4 s per native query.

| Case | Limiting proof step |
|---|---|
| Exact non-analytic transcendental contact | The threshold interval overlaps the exact maximum at the depth limit |
| Quartic degenerate contact | The stationary point has zero curvature and interval subdivision reaches the node budget |
| Tail upper bound equal to minimum subnormal threshold | Strict comparison fails under the positive tail remainder and breadth-first search runs out of nodes |
| Subnormal-weight closed receiver endpoint | Midpoint subdivision runs out of depth before it certifies the endpoint |

The optional bounded stationary path independently checks that last
endpoint and certifies HIT. Its separate performance results favor keeping
it optional. The shared-kernel analytic closed-contact control resolves
through the original rational discriminant.

## Independent third implementation and held-out geometries

The unchanged rational/Arb evaluator completes the four original 16³
fields. Its response and solid masks agree with both A and D over 16,384
distinct sample centers. Its full-field times were 2.36, 6.90, 4.55 and
6.35 s for cases 0, 7, 14 and 21, respectively. The separate ray and
adversarial tests exercise exact contacts and extreme numeric ranges.

The four held-out geometry/source pairs were frozen from a disjoint seed
before evaluating the new algorithm. The frozen recipe accepts every
predeclared ordinal that passes the existing continuous containment,
nonempty interior and source-clearance checks. All four passed. The
generator's occupancy targets are proposals. The column below reports
actual exact solid occupancy on the 64³ sample grid. A/P/D complete both
32³ and 64³ on all four pairs with identical response and solid bits,
and A/D have identical scalar receiver support.

| Case | Layout | Kernels | 64³ solid occupancy | 64³ P / A / D cold s | A → D reduction |
|---|---|---:|---:|---|---:|
| 60 | Clustered | 19 | 1.000% | 11.884 / 66.593 / 35.473 | 46.7% |
| 62 | Branching | 16 | 2.628% | 5.396 / 39.466 / 23.940 | 39.3% |
| 63 | Elongated | 10 | 0.819% | 4.693 / 24.581 / 14.756 | 40.0% |
| 64 | Multiscale | 15 | 4.524% | 8.329 / 76.046 / 40.501 | 46.7% |

The 64³ held-out total is 206.69 → 114.67 s for A → D, down **44.5%**.
P + independent is 236.99 → 144.97 s, down **38.8%**. At 32³ the
independent total drops 24.7% and P + independent drops 20.4%. The
held-out RSS peak is 152.20 MiB. All source coordinates and workload
identities are frozen, with no filtering by runtime. These four layouts
support a generalization result within the measured positive B2 family.

All four held-out fields also provide an independently certified fusion
control. A small dyadic cube around one 64³ sample center is SOLID for
the sum and FREE for every individual kernel at the same threshold.
Floating-point margins only select the candidate cube. The original
independent box predicates certify all labels. Kernel singular-value
estimates span approximately 0.00761 to 0.28148 in world units. The cube
checks establish local overlap effects; the occupancy measurements count
sampled solid voxels.

## Bounded 128³ follow-up

After the 32³/64³, Arb and held-out gates, case 14 completed before case
7 began. Each of the six runs completed 2,097,152 valid voxels within
its 480 s cap. The new P/A/D fields agree with each other and with the
historical complete independent fields. D and A use the same scalar
receiver supports.

| Case | P cold s | A cold s | D cold s | A → D MISS s | A → D HIT s | A → D reduction | P + independent reduction |
|---|---:|---:|---:|---|---|---:|---:|
| 14 | 33.363 | 184.031 | 87.895 | 150.681 → 60.705 | 14.631 → 8.302 | 52.2% | 44.2% |
| 7 | 56.935 | 336.889 | 159.448 | 276.286 → 116.338 | 40.467 → 23.002 | 52.7% | 45.1% |

Together these reduce the independent time from 520.92 to 247.34 s
(**52.5%**). P + independent goes from 611.22 to 337.64 s (**44.8%**).
A/D wall and CPU times agree closely. The peak observed RSS was
199.03 MiB including diagnostic array materialization. Field-level
exp endpoints fall from 60.84M to 38.62M in case 14 and from 112.52M
to 71.69M in case 7. Both improved runs complete without a final
original-ray fallback.

## Decision

Use D as the preferred configuration for further **B2 research** on
these workloads. It uses the 64-bit prefix through depth three,
256-bit signed-local tests from depth four and the existing proof
budgets. Keep the optional stationary stage off for the measured
default. B's standalone improvement, C's standalone improvement and
the additional D gain each hold on complete fields. The results
support the simpler cheap-first dispatch with adaptive precision.

This decision is supported by genuine independent rendering. The
prototype discovers its own partition and numerical decisions from
the canonical inputs. The original independent and the unchanged
Arb implementations remain reference comparisons. The 128³ gate
is a finite result for two synthetic scenes. Publication authority
remains `ready_capable=False` for B2. The evidence supports adopting
the independent optimization as a research configuration, so a
replacement algorithm is unnecessary for this decision. Repeated
exponential enclosure remains the principal measured cost.

The [machine-readable measurements](../experiments/gaussian_geometry/results/b2-independent-measured-v1.json)
contain 86 complete field records over 22 geometry/source/grid supports,
per-label ray percentiles, CPU and cold wall time, work counters and
build/workload fingerprints. The [acceptance record](../experiments/gaussian_geometry/results/b2-independent-acceptance-v1.json)
records the numerical adversaries, regression and preservation checks.

The combined generator and Gaussian regression suite passed **598 tests**,
including the enabled real-CUDA controls. Seven existing checks were
skipped because their optional private pilot or campaign artifacts were
absent. Preservation was checked against the starting branch files, the
integration worktree and its teacher WIP, the historical experiment
outputs, all 13,013 public dataset files, and both release archive copies.
