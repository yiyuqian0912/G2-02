# RIND3D 1K v1 — production log

* **Root:** `/home/yushu/Projects/RIND/RIND3D_1K_data/release_1k_v1` (outside git).
* **Contract:** `contract/RIND3D_1K_CONTRACT_v1.json`.
* **Driver:** `scripts/production_loop.sh`.

## 1. Physical stage (complete)

**Run:** 2026-11-12 03:12 → 10:02 EST (6 h 50 min). Sequential, controller-driven, slots 0–999 on one RTX 3060 Laptop.

| quantity | value |
|---|---|
| attempts / accepted / rejected | 1000 / 1000 / 0 (every slot accepted at attempt 0) |
| proposals per side: mean, p95, max (cap 6000) | L64: 388, 1068, 4658 · L128: 149, 383, 2678 · L256: 68, 169, 979 |
| windows | 64,000 (exactly 16/24/24 per scene) |
| m histogram, m = 1..4 (target ¼ each) | 16071 / 16008 / 15969 / 15952, i.e. 25.1 / 25.0 / 25.0 / 24.9 % |
| m histogram, ω-weighted | 30.7 / 24.6 / 23.1 / 21.6 % |
| m by side, 64 | 4414 / 4055 / 4017 / 3514 |
| m by side, 128 | 5896 / 6059 / 5962 / 6083 |
| m by side, 256 | 5761 / 5894 / 5990 / 6355 |
| k histogram, k = 1..4 | 154 / 135 / 158 / 553 |
| families | 111 or 112 scenes each (slot mod 9) |
| informative channels (= labels to compute) | 159,802 |
| physical wall per scene | mean 24.6 s, p50 25.8 s, p95 39.3 s, max 63.8 s; total 6.8 h |
| storage | 4.4 GB (du): public 1.6 GB, privileged 2.7 GB (certified fields, selection logs) |
| scene-rejection bound (SELECTION_LAW §3, §8) | r̂ = 0; r/(1 − r) ≤ 0.0046 at 99 % confidence |

Validation results are in `VALIDATION.json` (section 4).

## 2. Label stage: incident and fix (2026-11-12)

**What happened.**
* The first label pass ran with `multiprocessing.Pool`.
* At 03:42–04:27, four workers died in native code: three SIGABRT in numba `NRT_Free` (heap corruption) and one SIGSEGV. All were on scene 000001 (thin_slabs, k = 3), source 0, windows 7, 15, 27, 33 and 38. A fifth worker was still busy after 8 h.
* `Pool` waits forever for lost tasks, so the label stage stalled from 03:43 to 11:33. 271 labels had been written by then.

**Root cause** (from the systemd core dumps):
* The SIGSEGV hit a page boundary inside `engine_g2.emit_region`, with a register equal to 2^22 = `max_records`.
* `emit_region` appended output records without a capacity check. A posterior that needs more than 2^22 cell pieces wrote past the record arrays.
* The CPU reference (`engine_p2`) has the identical code.

**Fixes:**

| commit | change |
|---|---|
| `31bdcbb` | **Supervised label pool** (`rind3d_1k/supervise.py`). One task per worker over a pipe. A dead worker → `kind=crash`; a per-side wall-clock limit (1800 / 3600 / 7200 s) → `kind=timeout`. Failures are logged to `labels_failures.jsonl` and quarantined. The slot status becomes `engine_failure` / `timeout`, never a fabricated label. Tested with abort, segfault and hang. |
| `deb1bac` | **Engine v3.1** (`tpe-b4/p2-cpu+g2-gpu-pass-v3.1`). Records are written only below capacity, and the exact need is counted. `run_window` reruns with the next power of two (limit 2^26). The engine is deterministic, so the rerun is exactly the unbounded-buffer result. Regression digests are unchanged on CPU and GPU, and a forced 16-record buffer reproduces the reference digest. |

**Exposure check.**
* Of the 5682 labels computed before the fix (release, canaries, geometry study), the largest has 3,755,613 records, below 2^22.
* No stored label was truncated or produced by an overflowing run.
* An overflowing run could only have corrupted later work in the same worker. Every overflowing worker died (or was killed) before writing another label.

**Restart.**
* Production restarted 2026-11-12 11:48 EST on engine v3.1: 12 workers (5 GPU-assisted), nice 10, supervised.
* The five affected tasks were rerun separately on v3.1 (section 3).

## 3. Status of the five scene-000001 tasks

The rerun on v3.1 runs with per-side limits of 7200 / 14400 / 14400 s. Results are recorded in `labels_log.jsonl`; any failure goes to `labels_failures.jsonl`.

## 4. Validation

`scripts/validate_release.py <root> --physics 24` (producer environment). Results are in `VALIDATION.json`.

## 5. Label throughput and completion estimate

* **Projection.** 159,802 informative labels at the Stage-B costs (6.2 / 24.3 / 106.7 worker-s per label for L = 64 / 128 / 256) come to ≈ 141 h on the full machine (16 hardware threads, 6 GPU-assisted).
* **Resuming.** Every pass skips finished task files. Stop with `touch <root>/STOP` (between passes) or SIGTERM to the runner. To resume, rerun `production_loop.sh` with the same arguments.
* **Quarantined tasks.** They are retried only with `run_labels.py ... --retry-failed`.

## 6. Stopped (2026-11-12, ≈ 12:10 EST, on request)

> **Superseded on 2026-11-13.** The labels were completed by the approved FastLabel backfill: 159,469 labels in 4.33 h,
> 0 failures, all 159,802 informative channels certified (`../../RIND3D_StudentResearch/D_BACKFILL_REPORT.md`). The
> `STOP` file in the release root is a leftover of this stop. The cost projections in this section are historical.

* **What was stopped.** The user stopped the campaign to rethink the labelling cost. The production loop, both label runners and the release validator were stopped. `<root>/STOP` is present.
* **State preserved.**
  * Physical stage: complete, all 1000 scenes.
  * Labels: 333 certified (333/333) in scenes 000000–000003.
  * Scene 000001: five `engine_failure` slots. Their v3.1 reruns were still busy after 16 min and were stopped.
  * `VALIDATION.json` was not produced, because the validator was stopped. `RELEASE_STATS.json` was produced.
* **Measured cost of the release labels so far (CPU-s per label):**

| side | labels | mean | median | p95 | max |
|---|---|---|---|---|---|
| L = 64 | 66 | 7.5 | 4.3 | 17 | 54 |
| L = 128 | 102 | 14.0 | 4.1 | 75 | 98 |
| L = 256 | 165 | 139 | 114 | 440 | 726 |

* **Where the cost goes.**
  * L = 256 windows are 92% of the CPU.
  * The most expensive 10% of labels are 50% of the CPU.
* **Resuming.** Remove `STOP` and rerun `production_loop.sh` with the same arguments. Finished task files are kept and skipped.
