# XLF CBC labeller: results

Machine: Ryzen 7 5800H (8 cores / 16 threads), single-threaded solver processes, 4 in parallel; laptop
measurements only. Raw records are in `/home/yushu/Projects/RIND/RIND3D_100K_v3_work/xlf_cbc/` (not in git):

* `baseline_astra/`, `v2_*/`: runs;
* `ref_cert/`: exact references;
* `audit_*/`: exact audit;
* `canary_*/`: canary.

Benchmarks run from frozen detached worktrees, so edits during a run cannot contaminate it.

## 1. Tasks and references

**Tasks.** These are the 90 frozen tasks of Astra's blind global protocol (`global_inverse/tasks.json` holdout 27 +
archive 54, `fresh.json` 9). They cover v3-distribution scenes from all 9 families, with 30 tasks each at
L = 64, 128 and 256. The solver reads only the canonical geometry and one channel's observation.

**References.** Each task was re-certified with the independent exact-kernel verifier (`yushu/3d-v3-staging`,
`certify/`, Tau-pencil labeller + exact verifier).

* 87/90 certified, with TV ≤ 0.0069 (median 2.7e-7).
* 3 timed out at 1800 s.
* Label + verify cost per task: median 26.6 s, maximum 650 s.

**Legacy references are not used.** Archived legacy Tau-pencil labels (`cmp_v3/labels/*_tp`, and Astra's
`records/reference*`, same engine) were wrong by TV 0.0193 on `000038_w06_s2` while claiming 0. The credit solver
exposed this; the exact verifier confirmed it. FastLabel v3.4's label for that task was correct (1.7e-7).

## 2. Certification rate and correctness

| version | change | certified, 8 s | certified, 20 s | inconsistent vs exact refs |
|---|---|---|---|---|
| Astra credit solver (`8d4e3cf`) | baseline | 34/90 | | 0 |
| v2 (`611e3a5`) | split transition, inherited obligations, obstacle exclusion, support planes, policy | 54/90 (cold JIT) | | 0/25 |
| v2.2 (`3123f0e`) | candidate-body masks, float sign filter, JIT warm-up | 65/90 | | 0/62 |
| v2.4 (`67e1eea`) | exact octree cell credit, support-plane axes | **69/90** | **79/90** | **0/77** |

By window size, v2.4 at 20 s: L64 28/30, L128 25/30, L256 26/30.

* **Consistency.** For every certified output with an exact reference, TV(p, q_ref) ≤ tv_bound + ref_bound. The
  largest observed distance is 0.0454, against a bound of 0.0483 (`000025_w04_s0`).
* **Output shapes.**
  * 54 of the 79 certified outputs are multi-cell: median 94 cells, maximum 118,578.
  * 32% have bound exactly 0, i.e. all feasible mass is proved in single cells.
  * Median bound 1.2e-2.
* **Byte identity across optimizations.** The float sign filter, body masks and octree credit leave the certified
  payload sha256 and transition count unchanged on 5 regression tasks. The 214-cell task's payload equals the wire
  encoding of Astra's archived exact solution (`6fa36134…`).

## 3. Exact differential audit (`audit_exact.py`)

Exact rational sample points are taken from every credited IN piece; each must be legal and reproduce every FREE
receiver (exact oracle). Points discarded by each transition kind are sampled the same way; none may be feasible.

PENDING: L64/L128 audit of v2.4 in progress (first 3 tasks: 87 IN checks, 62 discarded points, 0 violations).

## 4. Cost

| comparison (tasks certified by both, 20 s budget) | CBC v2.4 | other |
|---|---|---|
| deterministic: exact Tau-pencil labeller + independent verifier (78 tasks) | median 2.7 s, total 302 s | median 34.5 s, total 9,493 s (**18× slower**) |
| statistical: FastLabel v3.4 (`cmp_v3/fl.jsonl`, 47 archive tasks; FastLabel certifies all 54) | median 2.8 s | median 1.5 s |

* **Latency of certified CBC runs at 20 s:** median 2.7 s, p90 10.3 s, maximum 21.9 s.
* **Memory:** peak RSS median 472 MiB, maximum 2.6 GiB, on a 128³ posterior with 118K cells.
* **One-time JIT load** of about 1.6 s per process; a long-lived worker pays it once.
* **Unit cost:** exact Python polytope clipping (vertices of about 500-bit integers) dominates the remaining time.

**Against the 10× FastLabel target.** The certified CBC runs are within 2× of FastLabel's median. The certification
rate is not yet at FastLabel's level: 79/90 against 54/54 on the archive.

## 5. Remaining failures (v2.4, 20 s)

There are 11 failures. They are also the slowest tasks for the exact reference: Tau-pencil labelling took 4–110 s
plus 8–542 s verification, and two timed out at 1800 s.

**Common pattern.** A broad, fragmented frontier: 50–750 live pieces of similar volume. Much of the live volume lies
in pieces touching an obstacle, where both envelopes degenerate. A facet-parallel contact split (search only, after
v2.4) certifies 2 more of 5 such tasks at 20 s and reduces the bound of two others from 0.80 to 0.17 and from 0.70 to
0.17.

## 6. Representative workload: release canary (production contract v3)

PENDING: 72 FastLabel-labelled channels sampled from the 36-scene release canary (24 per L, all families; seed
20261118), CBC at a 16 s budget, consistency against the production FastLabel labels.
