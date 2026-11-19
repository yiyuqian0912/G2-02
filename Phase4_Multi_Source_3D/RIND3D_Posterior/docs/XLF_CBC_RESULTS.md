# XLF CBC labeller: results

Machine: Ryzen 7 5800H (8 cores / 16 threads), single-threaded solver processes, 4–6 in parallel; laptop
measurements only. Raw records are in `/home/yushu/Projects/RIND/RIND3D_100K_v3_work/xlf_cbc/` (not in git).
Benchmarks run from frozen detached worktrees, so edits during a run cannot contaminate it. Each process warms its
cached JIT (about 1.6 s) before the solver budget starts.

## 1. Tasks and references

**Tasks.** These are the 90 frozen tasks of Astra's blind global protocol (`global_inverse/tasks.json` holdout 27 +
archive 54, `fresh.json` 9). They cover all 9 families of the v3 distribution, with 30 tasks each at L = 64, 128
and 256. The solver reads only the canonical geometry and one channel's observation.

**Exact references.** Each task was re-certified with the independent exact-kernel verifier (`yushu/3d-v3-staging`,
Tau-pencil labeller + exact verifier).

* 87/90 certified, with TV ≤ 0.0069 (median 2.7e-7).
* 3 timed out at 1800 s.
* Label + verify cost: median 36 s, maximum 650 s.

**Fallbacks.** Where no exact reference exists, FastLabel v3.4 (statistical TV ≤ 0.05) is the comparison. Legacy
Tau-pencil labels (`cmp_v3/labels/*_tp`, Astra's `records/reference*`) are informational only. Two defects are
**exactly confirmed**, in both cases while the legacy engine claimed bound 0:

| task | legacy TV to exact | CBC TV to exact (bound) | FastLabel TV to exact |
|---|---|---|---|
| `000038_w06_s2` | 0.019 | 0.0097 (bound 0.0189) | 1.7e-7 |
| `000007_w08_s2` (exact reference after 1,984 s, bound 1.7e-5) | 0.069 | 0.024 (bound 0.049) | 3e-4 |

The CBC solver flagged both while being itself within its certified bound. FastLabel is unaffected.

**Exact reference set after re-certification:** 88/90.

## 2. Certification rate and correctness

| version | main change | 8 s | 20 s | inconsistent |
|---|---|---|---|---|
| Astra credit solver (`8d4e3cf`) | baseline | 34/90 | | 0 |
| v2 (`611e3a5`) | split, inherited obligations, obstacle exclusion, support planes | 54/90 (cold JIT) | | 0/25 |
| v2.2 (`3123f0e`) | candidate-body masks, float sign filter | 65/90 | | 0/62 |
| v2.4 (`67e1eea`) | exact octree cell credit, support-plane axes | 69/90 | 79/90 | 0/77 |
| v2.7 (`7621fcc`) | body-subset screens, facet contact splits, integer encoder, heap scheduling, budgeted parking | 79/90 | 87/90 | 0/85 exact + 0/1 FastLabel |
| **v2.8 (`70522ab`)** | exact GMP cell credit (native, differential-tested), cache-friendly pair proposals | **84/90** | **88/90** | **0/87 exact** |

v2.8 at 20 s:

* by window size: L64 30/30, L128 30/30, L256 28/30;
* by split: holdout 27/27, fresh 9/9, archive 52/54.

At 8 s: L64 29/30, L128 29/30, L256 26/30.

* **Consistency.** For every certified output, TV(p, q_ref) ≤ tv_bound + ref_bound. The largest observed distance
  is 0.0413, against a bound of 0.0494.
* **Output shapes (20 s).**
  * 62 of 87 certified outputs are multi-cell, up to 118,578 cells.
  * 25 have bound exactly 0.
  * Median bound 1.1e-2.
* **Byte identity across optimizations.** These optimizations leave the certified payload sha256 and the transition
  count unchanged on the regression tasks:
  * float sign filter;
  * body masks;
  * octree credit;
  * per-edge intersection cache.

  The 214-cell task's payload equals the wire encoding of Astra's archived exact solution (`6fa36134…`).

## 3. Exact differential audit (`audit_exact.py`)

Exact rational points are sampled from credited IN pieces; each must be legal and reproduce every FREE receiver
(independent exact oracle, `exactcheck.py`). Points discarded by each transition kind are sampled the same way;
none may be feasible.

**v2.8 (`70522ab`), all 60 L64/L128 tasks (8 s solver budget; 47 certified, 13 incomplete; at most 12 IN pieces
× 2 points and 30 discarded points per task):**

| check | sampled | violations |
|---|---|---|
| exact points in credited IN pieces must be legal and reproduce every FREE receiver | 604 | **0** |
| exact points discarded by a transition must not be feasible | 958 | **0** |

* The discarded points came from:

  | transition | points |
  |---|---|
  | pair conditioning | 458 |
  | exact cuts | 409 |
  | screens (contradictions and clearance settlement) | 90 |
  | window/obstacle exclusion | 1 |

* 417 receiver decisions needed the oracle's exact rational fallback.
* An earlier partial audit of v2.4 (7 tasks: 126 + 150 points) also found 0 violations.
* **L = 256 spot check:** `000007_w08_s2` at a 20 s budget (certified, bound 0.049):
  * 48 points from 16 of its 30 IN pieces, all feasible;
  * 29 discarded points, all infeasible: 8 from pair conditioning, 9 from cuts, 12 from screens;
  * 0 violations, after 40 minutes of exact oracle work.

  The other L = 256 tasks are not audited; the exact oracle costs about 12 s per point there.

This is sampled differential evidence, independent of the kernel's code. It does not prove the transitions.

## 4. Cost (v2.8)

| comparison | CBC v2.8 | other |
|---|---|---|
| deterministic: exact Tau-pencil labeller + independent verifier (83 tasks both certify, 8 s budget) | median 1.60 s, total 164 s | median 36.4 s, total 10,520 s (**30× median, 64× total**) |
| statistical: FastLabel v3.4 (archive; FastLabel certifies 54/54, CBC 49/54 at 8 s, 52/54 at 20 s) | median 1.55 s | median 1.59 s (**on par for the channels CBC certifies**) |

* **Latency of certified runs:**
  * 8 s budget: median 1.6 s, p90 4.7 s, maximum 7.9 s;
  * 20 s budget: median 1.6 s, p90 5.1 s, maximum 22.1 s.
* **Memory:** peak RSS median 480 MiB, maximum 1.2 GiB.
* **One-time JIT load:** about 1.6 s per process, outside the budget. A long-lived worker pays it once.
* **Where the time goes** (share of total wall, 8 s run, kernel wall-clock counters):

  | component | share |
  |---|---|
  | Numba screens | 34% |
  | Python envelopes | 10% |
  | exact cuts | 9% |
  | cell integration (native) | 8% |
  | pair proposals | 7% |
  | pair conditioning | 4% |
  | bound snapshots | 4% |

  Before the native credit, integration was 28% and pair proposals 13%.
* **Measurement note.** cProfile overstated pair conditioning by about 20×. All numbers here come from wall-clock
  counters.

## 5. Remaining failures (v2.8, 20 s)

* **`000007_w08_s2` (L = 256 barriers)** certifies at 20 s (bound 0.049). Its exact reference was finally obtained
  after 1,984 s, and the certified output is 0.024 from it, which is consistent.
* **Two uncertified L = 256 tasks** (barriers / occluder clusters). The exact reference failed on them too (1800 s
  timeouts).
  * Pattern: a broad fragmented frontier with obstacle contact.
  * They stay INCOMPLETE, with no misleading certificate.

## 6. Representative workload: release canary (production contract v3)

72 FastLabel-labelled channels were sampled from the 36-scene release canary: 24 per L, all 9 families, seed
20261118. FastLabel labels: 48 `certified_stat`, 24 `certified`.

| run | certified | L64 / L128 / L256 wall median | inconsistent with production labels |
|---|---|---|---|
| `73bc0ad`, 16 s, heavily loaded machine | 65/72 | 7.4 / 5.5 / 12.3 s | 0 |
| **v2.7, 16 s, clean** | **71/72 (99%)** | **2.3 / 1.5 / 4.0 s** | **0** |

* TV(p_CBC, q_FastLabel): median 2.0e-3, maximum 0.0444.
* On the 24 channels with deterministic FastLabel labels, the CBC output is identical (TV 0).
* By family, all channels certify in every family except barriers (8/9).

**Full canary population.** All 807 FastLabel-labelled channels (512 `certified_stat`, 295 `certified`) were run
with v2.7 at a **10 s** budget, 6 parallel jobs.

| L | certified | wall median / p90 |
|---|---|---|
| 64 | 184/202 | 3.3 / 11.0 s |
| 128 | 226/256 | 2.2 / 11.4 s |
| 256 | 329/349 | 4.7 / 8.1 s |
| **all** | **739/807 (91.6%)** | certified runs: median 4.4 s, p90 6.3 s, p99 10.7 s |

* By family:

  | family | certified |
  |---|---|
  | convex polyhedra | 62/62 |
  | prisms | 85/86 |
  | wedges | 71/73 |
  | rotated boxes | 106/111 |
  | thin slabs | 104/116 |
  | occluder clusters | 96/108 |
  | barriers | 72/81 |
  | corridors | 79/91 |
  | multiscale | 64/79 |

* **0 inconsistent with the production labels.**
  * TV(p_CBC, q_FastLabel): median 3.6e-4, maximum 0.0585 (`000001_w06_s3`, CBC bound 0.0496, statistical FastLabel
    label).
  * On all **295** channels where FastLabel's label is deterministic, the CBC output is identical (TV 0).
* CBC bounds: median 5.6e-4; 294 of 739 are exactly 0.
* All 68 uncertified channels are ones where FastLabel had to use its statistical path. They stay INCOMPLETE, with
  no certificate.
* A certified bound near the tolerance means a truncated support. Example: 4,120 cells against FastLabel's 5,060 at
  bound 0.0496. The certificate accounts for the omitted mass.

