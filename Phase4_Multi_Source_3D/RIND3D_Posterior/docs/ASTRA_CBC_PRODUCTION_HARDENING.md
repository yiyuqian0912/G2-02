# Deterministic posterior certification on the full v3 canary

**Decision: CONDITIONAL GO for bounded, single-host, audited batches with numerical build acc21857771fb64b0c012b0236d0d4b979d2751f.** All 807 original eligible channels received a deterministic certificate on their actual saved binary64 probabilities. No statistical fallback, observation substitutions, private source queries, new worlds or 100K production were used. The full implementation and toolchain have not been mechanically proved. Multi-host unattended production and unauthenticated public issuance remain outside this acceptance.

The [contract](../experiments/credit_solver/assurance/THEOREM.md), [obligation map](../experiments/credit_solver/assurance/OBLIGATIONS.md), [integration, accounting and wire proofs](../experiments/credit_solver/assurance/PROOFS.md), [reproduction commands](../experiments/credit_solver/assurance/OPERATIONS.md) and [measured results](../experiments/credit_solver/assurance/results/summary.json) accompany this report.

## What changed at the certification boundary

The previous public certifier accepted caller tolerances up to one. Its reader verified the identity fields enumerated by a statement without enforcing a complete physical field set. Those interfaces could not justify a universal 0.05 release ceiling or a complete input binding. The candidate enforces both. In addition it checks round-trip canonical geometry arrays, rejects duplicate container entries and trailing compressed crop data, validates serialized dyadic ticks without making a Fraction per record, checks the full expected build at write-back, and fsyncs the directory after replacement. The release reader can authenticate a file against a separately protected issuance digest. The forgery regression deliberately changes the reported TV and recomputes every self-hash; the independently pinned digest detects it. Old envelopes should retain their historical readers and provenance.

The numerical CPU backend uses Python integers and Fractions for decisions about polygon clipping, geometric volume and probability credit. The approved fixed-width scan is range restricted. Reusing the summed ceilings of unresolved pieces speeds broad-frontier scheduling. An authoritative snapshot recomputes those ceilings, independently checks the accumulated lower total and taints on discrepancy. Encoding large sparse distributions uses exact integer quotas and fewer copies. The optional projective mixed patch and the C++ comparison integrator are outside the canary's issue path. The default physics conditioning still uses adjacent opposite-response FREE endpoints.

## Full population and cost

36 frozen geometries, nine families, four natural v3 windows per size including the coverage slot; every eligible source-separated channel retained. No certification result was consulted in constructing the input manifest. All 807 task IDs, geometry and crop hashes match the corresponding historical inputs. The same numerical contract applied to all three queues.

| Soft seconds / live piece cap | Workers | Attempts | Newly certified | Cumulative | Worker CPU s | Attempt wall P50 / P95 / P99 / max, s |
|---|---:|---:|---:|---:|---:|---|
| 10 / 4,096 | 5 | 807 | 795 | 795 | 854.09 | 0.65 / 3.92 / 10.42 / 11.77 |
| 45 / 8,192 | 3 | 12 | 8 | 803 | 346.63 | 30.24 / 47.88 / 47.91 / 47.92 |
| 360 / 32,768 | 2 | 4 | 4 | 807 | 505.50 | 130.07 / 205.19 / 205.72 / 205.85 |

823 total attempts, 16 durable INCOMPLETE attempts, zero terminal incompletes, invalid inputs, integrity failures, worker errors or hard watchdog terminations in the production canary. Twelve initial time limits; three retry time limits and one retry fragment limit. The 11 first-pass completions whose search stopped with an unresolved frontier were authorized by the kernel's final exact bound. Failure recovery has its own separate experiment.

From first campaign lock creation through final receipt verification, elapsed time was **612.24 s** by filesystem timestamps, including 21.41 s to construct the initial input manifest. Active dispatch sessions measured 191.05 + 127.58 + 263.90 s. Total summed worker wall was 1,730.30 s, worker CPU 1,706.22 s (2.11 CPU s per final certificate), and timed coordinator CPU 23.77 s. The original manifest build and a few pre-dispatch operations were outside those CPU counters. A separately metered identical clean manifest build consumed 27.30 CPU s; the independent final receipt verification consumed 2.70 CPU s. These separate measurements must not be double-counted as dispatch time. Across all attempts worker wall P50/P95/P99/max were 0.65/5.12/34.26/205.85 s. Per **channel** summed across its retries they were 0.65/3.92/28.55/264.84 s. Peak independently measured worker RSS was 2,014,464 KiB (1.92 GiB); maxima at the other two tiers were 892,740 and 1,544,580 KiB. No assertion about the physical aggregate footprint follows from summing shared pages. The local campaign occupies 74,801,911 file bytes, including unsuccessful attempts and metadata.

| Family | Certified at 10 s / all | Channel accumulated wall P95, s |
|---|---:|---:|
| Rotated boxes | 111/111 | 2.22 |
| Thin slabs | 112/116 | 4.67 |
| Prisms | 86/86 | 1.14 |
| Wedges | 72/73 | 2.22 |
| Convex polyhedra | 62/62 | 1.82 |
| Multiscale convex | 76/79 | 5.23 |
| Corridors | 90/91 | 4.89 |
| Barriers | 81/81 | 3.10 |
| Occluder clusters | 105/108 | 5.44 |

| Window side | Certified at 10 s / all | Median support / entropy (nats) |
|---|---:|---:|
| 64 | 200/202 | 2,864 / 7.46 |
| 128 | 251/256 | 128 / 4.30 |
| 256 | 344/349 | 1 / 0 |

Across all channels support ranged from 1 to 829,044 cells (median 82), and entropy from 0 to 13.61 nats (median 3.91). The final reported TV bound had median 3.34×10⁻¹⁵, P95 0.047793 and maximum 0.04998755343. These are bounds, not measured errors. There were 295 one-cell wires; a grid cell of positive mass can contain a continuum of sources. Minority fraction is the smaller FREE class divided by the total FREE count. Separating out single-voxel minorities first, the initial outcomes were 1/3 for a single minority voxel, 154/163 for other channels below 1%, 224/225 from 1% to below 10%, and 416/416 from 10% through 50%. All groups reached 100% on retry. The three single-voxel cases alone used 530.95 worker CPU s. This supports a separate compute queue while retaining those observations in the population.

## Comparators and long tails

The historical CBC v3.0.1 also completed 795/807 within its first soft budget. One channel moved in each direction, consistent with sensitivity of a deadline to execution conditions. After retries the new probability wires were **byte-identical on all 795 historically certified channels**. This comparison is an implementation diagnostic. On the 14 archived *independently integrated exact rational* references, every exact decoded TV comparison was inside its own reported bound. Maximum measured exact-reference error was 5.86×10⁻¹⁵ or less; these 14 do not span every diffuse geometry. The 214-cell result is included. Against FastLabel there were 295 deterministic historical labels (zero comparison TV) and 512 statistical ones; maximum comparison TV over all 807 was 0.059479, with no numerical discrepancy beyond the sum of the two reported bounds to a diagnostic 10⁻¹² tolerance. This tolerance applies only to the floating cross-product comparison table. Issuance and exact-reference validation use rational comparisons without an epsilon.

Archive FastLabel reports 731.83 CPU s and 772.47 summed worker wall seconds for this population. All new deterministic attempts together used **2.33×** that reported CPU or **2.24×** summed worker wall. The historical CBC reported 1,413.86 CPU s and 2,201.32 summed process wall seconds at 10 s. Historical launch/JIT amortization and load differ. These are same-input cost comparisons rather than matched contemporaneous throughput speedups.

| Same-machine control and frozen candidate | Released cells | Old wall s | New wall s | Wall reduction | Same wire and bound |
|---|---:|---:|---:|---:|---|
| Thin slab 000001_w02_s0, cap 32,768 | 404,212 | 255.48 | 190.85 | 25.3% | yes |
| Multiscale 000023_w07_s0, cap 4,096 | 829,044 | 54.49 | 44.47 | 18.4% | yes |

The contemporary control is detached bf5738c. Budgets, physical inputs, screening and benchmark readback are matched per row. Other low-concurrency work shared the host. One paired run per case does not yield a stable population-wide speedup interval. Historical times were 230.01 and 44.59 s; the new multiscale result is essentially tied with the historical time even though it beats this run's matched control. A separately measured new cap-32,768 multiscale run took 42.34 s with the same bytes.

The thin-slab candidate spent 69.63 s integrating grid credits, 38.86 s screening, 31.34 s in state commits and 27.49 s enforcing exact receiver cuts on 24,714 transitions. Timers can nest. Integration itself did not become faster in this pair. The avoided full-frontier volume scans between commits, cheaper wire validation and fewer allocation-heavy passes are relevant mechanisms; the paired measurements identify their combined effect, not a unique percentage for each. Huge-output validation is an inherent linear cost. In the 829k-cell matched run, peak benchmark-process RSS fell from 2,349,196 to 2,164,076 KiB. Benchmark RSS includes the reference harness's optional materialized reader; the campaign uses the streaming validator.

## Final soundness and release decisions

1. **Target existence.** Feasible volume is measurable for all supported tasks. Normalization is defined only for positive volume. Every issued artifact proves a positive lower mass. A valid observation can remain incomplete even if generated by a real source.
2. **Input binding.** The scoped canonical decoder, binary64-to-rational map, SOLID recheck, FREE response crop hash, full problem identity, approved numerical build and independently protected issuance digest bind one physical problem to one artifact. The ordinary low-level Python constructor alone is not the population ingestion protocol.
3. **Inverse geometry.** The exact ray elimination, conservative envelopes, clearance sandwich and obligation induction justify their decisions mathematically. Their implementation has review, directed arithmetic lemmas and differential tests. There is no machine proof of every execution path.
4. **Analytical integrator.** Sign reflection, zero coefficients, exact affine extrema and inclusion–exclusion establish the single-facet formula *and when it applies*. Multiple active rows require exact clipping.
5. **Rounding.** Subject to the polytope and arithmetic invariants, no lower over-credit or upward-charge deficit follows from either integrator. Conservation alone could not establish a correct per-cell split.
6. **Incremental ledger.** Copy-before-commit, immutable consumed ownership, directed tick deltas and a final rescan support the state induction. Stale handles, mid-commit interruption and deliberately incorrect cached live totals have regressions.
7. **Checkpoint skipping.** The necessary-condition inequality is exact. It cannot authorize a label. Fixed-state certification is invariant; identical wall-time trajectories, best failing diagnostics or byte-identical outputs across arbitrary timed runs are not guaranteed.
8. **Bytes and precision.** The bound covers authenticated, decoded, normalized dyadic52 binary64 records. Binary32 conversion needs an additional allowance; the independent conversion helper tests this, including a case where 0.05 would no longer suffice.
9. **Failure paths.** No false CERTIFIED found on the tested error paths. Timeout can return a previously valid certificate, otherwise INCOMPLETE. Arithmetic guard failure, inconsistent build or artifact integrity failure cannot qualify as a release. The conclusion assumes the reviewed implementation and runtime contracts.
10. **Implementation assurance.** Six new SMT implications and 26 pre-existing arithmetic lemmas passed. 673 solver tests plus 128 related tests passed, including optional GMP controls and both archived Tau-pencil counterexamples. General clipping topology, the state transition correspondence, machine code, CPython and dependency execution retain explicit trusted assumptions.
11. **Independence from replay.** No per-label physical replay or statistical validation is required. The release readback independently enforces manifest completeness and the external digest of every issued file. It verifies issuance/integrity, not geometry a second time.
12. **Broad performance.** Two matched cases improved by 25.3% and 18.4% with identical probability bytes. The cost and memory tails remain materially larger than typical cases.
13. **Coverage.** 795/807 initially; 807/807 with all retries charged. This establishes coverage on these 36 known worlds only.
14. **Bottleneck.** Broad outputs combine exact grid integration, large frontier management and expensive encoding/readback. Some smaller-support hard cases instead spend most time constructing receiver envelopes and screening. Fixed balance thresholds would incorrectly change the population.
15. **Reproducibility and isolation.** Detached clean checkout reconstructed a byte-identical physical/build manifest; a fresh CPU venv reproduced all 14 reference wires and rational bounds. An actual interrupted worker, exclusive owner, recovery, corruption refusal and idempotent resume were tested. Full sweep outputs, attempt receipts, checksums and grouped measurements are preserved.
16. **Before general deployment.** Bind the input manifest and issuance index to an organizational trust root, deploy the pinned CPU runtime with controlled upgrades, enforce bounded campaign sharding/coordinator ingress for larger datasets, and measure difficult unseen scenes on the deployment hardware. There is no proof of termination, universal 10 s coverage, a portable verified compiler, adversarial same-user process isolation or bounded 100K tail cost. A future family or physics change requires a new input and proof audit.

On all 807 actual wires, binary32 rounding yielded 507 vectors whose exact sum differed from one. The largest absolute normalization drift was 4.35×10⁻⁸. When the rounded weights are **exactly** renormalized, the largest measured exact TV displacement was 1.77×10⁻⁸ and the worst combined bound was 0.04998755523. This is a separate conversion audit, not a promise for arbitrary GPU reductions or training transforms.

A forced one-second hard watchdog on a disposable hard channel exited with signal 9, recorded INTERRUPTED and produced no label. A second freshly reconstructed environment reproduced the complete pinned input manifest as well as the 14 independent reference wires.

The independently checked scientific files are retained locally under /home/yushu/Projects/RIND/reports/astra-cbc-production-proof/canary. The research branch stores manifests, source/build identities, raw attempt summaries and compact comparison results. No production branch change is required to inspect them.
