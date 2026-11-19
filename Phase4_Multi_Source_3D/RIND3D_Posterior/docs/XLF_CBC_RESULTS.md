# XLF CBC labeller: results (v2.9)

**Machine and method.**
* Ryzen 7 5800H (8 cores, 16 threads); single-threaded solver processes, 4–6 in parallel; laptop only.
* Raw records: `/home/yushu/Projects/RIND/RIND3D_100K_v3_work/xlf_cbc/` (not in git).
* Every benchmark runs from a frozen detached worktree. The final build is v2.9e (`067d746`, worktree
  `G2-02-xlf-cbc-final`, native `.so` sha256 `abfdf4c6…`).
* **Certified mode** (`CBC_VERIFY_SPLITS=1`, the default) certifies every polytope split at run time.
  **Speed mode** (`=0`) does not; both modes keep the structural checks of credited and charged polytopes.
* "In-process wall" covers `cbc_run.main`: decoding the channel, JIT warm-up (≈ 0.15 s) and the solve.
  "Process wall" adds interpreter start-up and imports (≈ 0.9 s).

## 1. Tasks and references

* **Benchmark.** The 90 frozen tasks of Astra's blind protocol (holdout 27, fresh 9, archive 54): all 9 families,
  30 each at L = 64/128/256.
* **Exact references.** These are 88 of 90 tasks, re-certified with the independent exact-kernel verifier
  (`yushu/3d-v3-staging`); the missing two are `000000_w00_s0` and `000008_w11_s2`.
* **Release canary.** All 807 FastLabel-labelled channels of the production canary (512 `certified_stat`,
  295 deterministic `certified`).
* **Legacy Tau-pencil labels are informational only.** Two defects are exactly confirmed:

  | task | legacy TV to exact | CBC TV to exact (bound) | FastLabel TV to exact |
  |---|---|---|---|
  | `000038_w06_s2` | 0.019 | 0.0097 (bound 0.0189) | 1.7e-7 |
  | `000007_w08_s2` | 0.069 | 0.024 (bound 0.049) | 3e-4 |

## 2. Certification rate and correctness (90-task benchmark)

| version | main change | 8 s | 20 s | inconsistent vs exact refs |
|---|---|---|---|---|
| Astra (`8d4e3cf`) | baseline | 34/90 | | 0 |
| v2.4 (`67e1eea`) | exact octree credit | 69/90 | 79/90 | 0/77 |
| v2.7 (`7621fcc`) | subset screens, contact splits, integer encoder | 79/90 | 87/90 | 0/85 |
| v2.8 (`70522ab`) | native GMP credit | 84/90 | 88/90 | 0/87 |
| v2.9a (`94ec247`) | hardening: release discipline, structure checks, read-only evidence | 84/90 (payloads byte-identical to v2.8) | | 0 |
| v2.9b (`285bf69`) | exact mixed facets of the possible shadow | 85/90 | | 0 |
| v2.9c (`a0533a8`) | hull step (receiver convexity), non-contact guaranteed shadow | 85/90 | | 0 |
| v2.9d (`8dfcab7`), speed mode | per-split certificate available; fragmentation cap | **87/90** | | 0 |
| **v2.9e (`067d746`), certified mode** | final build | **84/90** | **87/90** | **0/83 (8 s), 0/86 (20 s)** |

**Final build, certified mode, 20 s.**
* By L: 30/30, 29/30, 28/30. By split: holdout 27/27, fresh 9/9, archive 51/54.
* Uncertified (all INCOMPLETE): `000008_w11_s2` (no exact reference exists either), `000037_w06_s0` (bound
  0.082), `000059_w11_s3` (0.070).
* Against v2.8 at 20 s: gained `000007_w09_s2`; lost `000037_w06_s0` and `000059_w11_s3`, which certify in
  speed mode.
* The largest observed TV to an exact reference is 0.0459, against a bound of 0.0487.
* Certified bounds: median 6.6e-15; 25 exactly 0.
* **Latency:** in-process median 1.28 s (v2.8: 1.6 s); total in-process wall 228 s for 90 tasks.

**Speed mode, 8 s (v2.9d):** 87/90, with a median in-process wall of 1.05 s (v2.8: 1.6 s) and no task lost
against v2.8.

## 3. Independent validation

### 3.1 Reference consistency

For every certified output, TV(decoded payload, q_ref) ≤ bound + ref_bound. Results:
* benchmark: 0 inconsistent of 83 (8 s) and 86 (20 s);
* canary: 0 inconsistent of 786 / 796;
* identical (TV 0) to all 295 deterministic FastLabel labels.

### 3.2 Chaos search (soundness independent of the search policy; `chaos.py`)

The chaos search drives the real kernel through random legal API sequences, interleaved with the greedy policy:
* degenerate split planes: through three vertices, voxel and cell planes, small-integer normals;
* random receiver subsets, possibly non-pending;
* random non-interface pairs;
* any order.

It also attempts misuse:
* writes through returned arrays;
* stale handles;
* constant rows;
* SOLID and out-of-window receivers;
* screening before exclusion.

Every snapshot is compared with an exact-certified reference.

| campaign | runs (tasks) | snapshot checks | violations | misuse attempts rejected (state unchanged) |
|---|---|---|---|---|
| v2.9a, 48 referenced tasks × 3 seeds | 127 (43) | 9,600 | **0** | 2,904 |
| v2.9e certified mode (every split certified), 48 tasks × 3 seeds | 144 (48) | 1,421 | **0** | 498 (+1 oversize split row) |

In the certified-mode campaign no kernel was tainted. Every split, including the deliberately degenerate chaos
planes through vertices, passed its run-time certificate, so the checker raised no false alarm. 123 of 137 final
releases reached bound ≤ 0.05; the check covers bounds of every size. Fewer snapshots fit the budget than in the
v2.9a campaign because each split is certified.

The v2.9a campaign also exposed the native decimal-key defect (IMPLEMENTATION §2, defect 4): a crash, not a wrong
certificate.

### 3.3 Exact differential audit (`audit_exact.py`)

Exact rational points are sampled in credited IN pieces (they must be legal and reproduce every FREE receiver) and
among points discarded by each transition kind (they must be infeasible). An independent exact oracle
(`exactcheck.py`) decides each point.

| build | tasks | IN points | discarded points | violations |
|---|---|---|---|---|
| v2.8, all 60 L64/L128 | 60 | 604 | 958 | **0** |
| v2.8, L = 256 spot check `000007_w08_s2` | 1 | 48 | 29 | **0** |
| v2.9e certified mode, all 60 L64/L128 (8 s; 51 certified under audit load) | 60 | 414 | 1,215 (pairs 615, cuts 512, screens 87, exclusion 1) | **0** |
| v2.9e, 11 hard canary channels newly certified by v2.9 (5 × L64 mixed-facet cases, 6 × L256 hull-step cases; reduced sampling) | 11 (9 certified under audit load; `000007_w09_s2`, `000007_w10_s1` incomplete there, so only their discarded points were sampled) | 70 | 127 (pairs 65, cuts 62) | **0** |

This is sampled evidence and proves nothing over continuous regions; the per-run certificates are §3.4.

### 3.4 Run-time certificates

| check | final canary, certified mode | failures |
|---|---|---|
| polytope splits certified (`verify_split`) | 494,099 | **0** |
| structural certificates of credited and charged polytopes | 23,834 | **0** |
| ERROR (tainted) releases | — | **0** of 807 |

### 3.5 Regression fixtures (`test_regression.py`, exact references, 11 s)

All five pass: the 214-cell posterior, both legacy Tau-pencil defects, the hull-only L = 256 task, and the kernel
test task. Astra's missing-sibling counterexample is in `xlf_assurance/test_assurance.py` (passes).

## 4. Hard cases: what made them hard and what fixed them

The tail set is the 68 canary channels v2.7 could not certify at 10 s. Number certified at 10 s:

| build | certified of 68 |
|---|---|
| v2.8 / v2.9a | 22 |
| v2.9b (mixed facets) | 35 |
| v2.9c (hull step + contact vertices) | 50 |
| v2.9d (fragmentation cap), speed mode | 58 |

Each mechanism was diagnosed with exact instruments before it was fixed.

1. **Loose possible shadow** (stuck L64/L128 pieces).
   * Diagnosis: 10⁵ lit receivers stayed ambiguous on a piece while 531 of 537 exact cut batches showed they did
     not cut it. The possible shadow was missing the mixed facets of O = K + cone(K − P) (design §4d).
   * Effect of adding them, validated exactly: ambiguous receivers on 12 captured pieces 101,001–159,566 →
     10–36,542 (≥ 99.7% fewer on 11 of 12). Five channels that failed at 10 s certify in ≈ 3 s.
2. **Overflowing obligation lists at L = 256.**
   * Diagnosis: 31–49 of the full-window screens per run found 3–15 × 10⁶ ambiguous receivers (cap 2.1 × 10⁶),
     each one split again and rescanned in full. Almost all were dark receivers with a single possible blocker,
     and genuinely ambiguous (exact cuts did cut, 55 of 60 sampled on one piece and 60 of 60 on the others).
   * Fix: receiver convexity (design §4e). 1.6–9.9 × 10⁶ receivers have convex-hull vertex sets of 54–78. After
     enforcing those, 3,245,475 ambiguous receivers → 14, and 3,798,238 → 1,287.
   * The one case that stayed ambiguous (2.07 M) was traced to the contact-vertex over-constraint, which is now
     removed (design §4f).
3. **Fragmentation after a hull step.**
   * Diagnosis: a 96-receiver batch of two-blocker dark receivers split one piece into 1,860.
   * Fix: at most 16 fragmenting receivers per batch. All four regressions recovered.
4. **What was not the cause.**
   * Grid structure: the exact per-cell decomposition of all live pieces gives the same bound on all 54 tail tasks
     with positive mass (design §6). The tail consists of spread posteriors over 10³–2 × 10⁵ cells.
   * Weak lower bounds: the lower credits are exact up to 2⁻¹⁶⁰ per boundary cell.

**Remaining failures (certified mode, canary at 10 s; 21 of 807).** All are INCOMPLETE, none is a misleading
certificate. Classes:
* **Huge spread posteriors** (`000023_w07_s0/s1`): exact integration of 2 × 10⁵ cells.
* **Thin slabs** (`000001_w02_*`, `000001_w05_*`).
* **L = 256 multi-blocker penumbras** (`000008_w11_s2/s3`, `000005_w11_s0`, `000006_w11_s0`): 63–93% of the
  ambiguous receivers have two or more possible blockers.
* **Near-threshold channels** (bounds 0.054–0.19) that certify in speed mode or with a longer budget.

## 5. Cost and production economics (final build; canary, 10 s budget, 6 parallel processes)

| | CBC certified mode | CBC speed mode | FastLabel v3.4 (production canary run, laptop GPU) |
|---|---|---|---|
| certified | **786/807 (97.4%)** | **796/807 (98.6%)** | 807/807 (512 statistical, 295 deterministic) |
| semantics | deterministic TV bound on the output bytes | same | 295 deterministic; 512 statistical (`certified_stat`, TV ≤ 0.05 w.h.p.) |
| per-channel wall, median | 1.67 s in-process (2.73 s process) | 1.27 s (2.36 s) | 0.68 s |
| p90 | 4.37 s process | 3.66 s process | 1.76 s |
| maximum | 13.5 s process (budget 10 s + decoding/start-up) | 14.2 s | 6.1 s |
| total, 807 channels | 1,563 s in-process / 2,435 s process | 1,271 s / 2,151 s | 772 s |
| ratio of totals to FastLabel | **2.0× (3.2× with start-up)** | **1.65× (2.8×)** | 1 |
| peak RSS | median 444 MiB, max 771 MiB | median 443, max 794 | (GPU worker) |
| failures | 0 crashes, 0 ERROR, 21 INCOMPLETE | 0 crashes, 11 INCOMPLETE | — |

Notes on the comparison:
* FastLabel's per-channel cost is measured inside a warm scene worker. CBC's in-process wall still includes
  decoding the channel (`inputs.load`), measured on a lightly loaded machine at median 0.01 / 0.03 / 1.66 s for
  L = 64 / 128 / 256.
  * The L = 256 cost is the shared input path: it re-reads the scene's whole observation blob per channel.
  * Over the canary this is about 590 s (349 × 1.7 s).
  * Net of decoding, CBC's total is ≈ 970 s in certified mode (≈ 1.3× FastLabel) and ≈ 680 s in speed mode
    (≈ 0.9×), an approximation from medians.
  * A per-scene worker that decodes once would realize this.
* CBC uses one CPU core per channel and FastLabel the laptop GPU.

**Latency of certified labels (certified mode):** in-process median 1.76 s, p90 3.08 s, p99 6.90 s, max 9.36 s.

**By output shape (certified mode, in-process wall median / p90):**

| output | channels | median / p90 |
|---|---|---|
| single cell | 295 | 2.31 / 2.74 s (225 are L = 256, where decoding takes ≈ 1.7 s of it) |
| 2–100 cells | 129 | 0.96 / 2.78 s |
| 101–10⁴ cells | 326 | 1.16 / 3.56 s |
| > 10⁴ cells (max 29,247) | 36 | 2.95 / 5.46 s |

**By family (certified mode):**

| family | certified |
|---|---|
| convex polyhedra | 62/62 |
| prisms | 86/86 |
| rotated boxes | 111/111 |
| wedges | 72/73 |
| barriers | 79/81 |
| corridors | 87/91 |
| thin slabs | 112/116 |
| multiscale | 75/79 |
| occluder clusters | 102/108 |

**Geometric work, whole canary, certified mode:**

| quantity | count |
|---|---|
| transitions | 46,890 |
| pair conditionings | 19,158 |
| exact receiver cuts | 268,005 |
| full-window screens | 2,725 |
| list screens | 12,534 |
| geometric splits | 2,335 |
| hull steps | 741 |
| IN settlements | 3,306 |
| OUT drops | 783 |

**Time shares:** pair conditioning 16.6%, screens 12.8%, exact cuts 9.9%, integration 8.9%, envelopes 7.4%,
structure checks 0.5%. Per-split certification sits inside the pair-conditioning and cut shares; its total cost is
+23% wall against speed mode on the same 786 channels.

**Trusted-kernel overhead.**
* Structural checks: 0.5% of wall.
* Per-split certificates: +23% total, +28% median (pure Python).
* Release discipline: negligible.

**Against previous versions.**
* On the 738 channels that both v2.7 and v2.9e certify, the median in-process wall falls from 3.23 s to 1.62 s.
* Versus the exact Tau-pencil labeller plus verifier on the benchmark (v2.8 measurement): about 30× faster by
  median and 64× by total.

**Relation to FastLabel.** XLF-CBC does not fall back to FastLabel; INCOMPLETE is its only non-certified
outcome. A pipeline that labels INCOMPLETE channels with FastLabel would be a separate decision, with the two
assurance models kept distinct in the envelope.
