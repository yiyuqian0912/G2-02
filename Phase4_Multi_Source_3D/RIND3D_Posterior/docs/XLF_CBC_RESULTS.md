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
Tau-pencil labels (`cmp_v3/labels/*_tp`, Astra's `records/reference*`) are informational only. They were wrong by
TV 0.019 on `000038_w06_s2` (exact-verified) while claiming 0, and they are 0.070 from FastLabel on `000007_w08_s2`.

## 2. Certification rate and correctness

| version | main change | 8 s | 20 s | inconsistent |
|---|---|---|---|---|
| Astra credit solver (`8d4e3cf`) | baseline | 34/90 | | 0 |
| v2 (`611e3a5`) | split, inherited obligations, obstacle exclusion, support planes | 54/90 (cold JIT) | | 0/25 |
| v2.2 (`3123f0e`) | candidate-body masks, float sign filter | 65/90 | | 0/62 |
| v2.4 (`67e1eea`) | exact octree cell credit, support-plane axes | 69/90 | 79/90 | 0/77 |
| **v2.7 (`7621fcc`)** | body-subset screens, facet contact splits, integer encoder, heap scheduling, budgeted parking | **79/90** | **87/90** | **0/85 exact + 0/1 FastLabel** |

v2.7 by window size at 20 s: L64 30/30, L128 30/30, L256 27/30. By split: holdout 27/27, fresh 9/9,
archive 51/54.

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

**v2.4, 7 tasks (L64/L128):**

* 126 IN points, all feasible.
* 150 discarded points, all infeasible:
  * 78 from pair conditioning;
  * 60 from exact cuts;
  * 12 from screens and clearance settlement.
* 0 violations. The full run was paused for clean timing.

## 4. Cost

| comparison | CBC v2.7, 20 s | other |
|---|---|---|
| deterministic: exact Tau-pencil labeller + independent verifier (85 tasks both certify) | median 2.7 s, total 268 s | median 36.4 s, total 10,760 s (**40× in total**) |
| statistical: FastLabel v3.4 (archive; FastLabel certifies 54/54, CBC 51/54) | median 2.7 s | median 1.6 s (**1.7×**) |

* **Latency of certified runs at 20 s:** median 2.7 s, p90 8.3 s, maximum 16.5 s.
* **At 8 s:** median 2.5 s, p90 4.8 s.
* **Memory:** peak RSS median 472 MiB, maximum 1.2 GiB.
* **Where the time goes (kernel counters):**
  * exact Python polytope clipping (homogeneous vertices of about 500-bit integers) in exact cuts;
  * cell integration of large IN pieces;
  * screen envelopes;
  * Numba screens are minor.
* **Measurement note.** cProfile overstated pair conditioning by about 20× (10.8 s against 0.5 s measured). All
  numbers here come from wall-clock counters.

## 5. Remaining failures (v2.7, 20 s)

* **`000007_w08_s2`.** It certifies at 20 s, but its exact reference timed out. It agrees with FastLabel (TV 0.024);
  exact re-certification (4 h budget) and a targeted exact audit are running.
* **The three uncertified tasks** are L = 256 barriers / occluder-cluster windows. The exact reference also failed
  (1800 s timeout) on two of them.
  * Pattern: a broad fragmented frontier with obstacle contact.
  * They remain INCOMPLETE, with no misleading certificate.

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

PENDING: the full canary population (807 FastLabel-labelled channels) at 10 s.
