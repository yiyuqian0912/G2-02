# FastLabel tier of the 1K labeller (`rind3d_1k.fastlabel`, engine `fastlabel`)

**Status:** integrated and tested (`tests/test_fastlabel.py`). The 1K backfill was approved and launched on 2026-11-13
(`backfill/run_backfill.sh`; report: RIND3D_StudentResearch/D_BACKFILL_REPORT.md).

## What it computes

The target is unchanged: q_F, the cell masses on the 128³ grid of Uniform(F_W^j) (CONTRACT_1K.md, SELECTION_LAW_1K.md).
FastLabel is a cheaper way to compute a label with a guarantee relative to this target.

The research record is in `RIND3D_Posterior/experiments/fastlabel/`:
* `REPORT.md`: methods, benchmarks, cost frontier;
* `CERTIFICATE_AUDIT.md`: the mathematics of the guarantee;
* `GPUVERIFY.md`: the exact GPU point verifier and its validation (Gate A).

**Label statuses written by the `fastlabel` engine:**

| status | meaning | task-file records |
|---|---|---|
| `certified` | deterministic TV bound ≤ 0.05: single-cell support (TV 0) or an exact path | engine records (lo, hi per piece), `tv_bound` = deterministic bound |
| `certified_stat` | **TV(label, q_F) ≤ 0.05 except with a stated probability.** Per-label confidence 1 − δ with δ = 0.01 / 159,802, so by the union bound P(any false statistical certificate in the whole 1K) ≤ 0.01 | one record per cell, lo = hi = cell volume of the certified outer bound B (the loader's q = lo / Σ lo is q_B), kind = 4, `tv_bound` = 0.05 |
| `certified_loose` / `uncertified` | exact fallback with a weaker deterministic bound (not expected; none in the benchmarks) | engine records |

**Certificate fields** (`meta["certificate"]` of statistical labels):
* eps, delta_label;
* the round and its budget δ_r;
* the stage (1 or 2);
* n tested, K failures, thresholds a1 / a2, the exact α of the round;
* the Clopper–Pearson p_lower;
* the policy string;
* the keyed seed `[base_seed, scene, window, source]`.

A rerun reproduces the same label: the streams are keyed, and the verifier's decisions and records are deterministic.

**Rule:** never rerun a label with a different seed after a failed certificate. That would be a "re-roll" and is not covered by δ.

## Loader

`flatten(ds, population=..., accept=("certified", "certified_stat"), fallback=None | "onehot", ...)`. Changed on
2026-11-13 by audit M2 (RIND3D_StudentResearch/C_LOADER.md):
* every certified status with tv_bound <= 0.05 is accepted by default, deterministic and statistical;
* the target population must be declared;
* missing labels are yielded as supervision "missing" (or counted when dropped);
* the true-source one-hot fallback is explicit and labelled.

Every example carries `status`, `certificate`, `provenance` (engine, version, config hash, file sha256) and `supervision`.

## Running

```bash
# labeller environment with CUDA; scene-ordered tasks, 15 workers (one GPU semaphore is created automatically)
export RIND_NVRTC_LIBRARY=<path to libnvrtc.so.12> OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
python scripts/run_labels.py /home/yushu/Projects/RIND/RIND3D_1K_data/release_1k_v1 15 0 --engine fastlabel --scene-order
```

* Existing task files (the 333 certified release labels) are skipped, never overwritten.
* Every finished task is appended to `labels_log.jsonl`.
* Crashes and timeouts are quarantined by the supervisor, as for the tau-pencil engines.
* `LABELS.json` is rebuilt per touched scene.

## Go / no-go

Gates C/D of the brief: performance projection on held-out scenes, accuracy against certified references, and the
user's explicit approval.

The measured numbers are in REPORT.md, "Checkpoint 6":
* HOLDOUT-18: 12.23 labels/s, 3.63 h.
* Pooled 36 scenes: 4.27 h (90% [3.53, 5.03] h, 99% upper bound 5.36 h).
* Accuracy vs 447 certified references: max TV 0.0164.
* 0 errors, 0 fallbacks.
