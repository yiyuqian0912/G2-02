# FastLabel for RIND3D-100K: adapter, integration plan, durability review (2026-11-13)

**Scope.** This document prepares the work; it changes nothing on the 100K side.
* Branch `yushu/3d-100k-h20` (`5560a20`) was **not modified, merged or launched**.
* Both patches below were developed and tested on a scratch export of `5560a20`. `git apply --check` passes on the branch.
* Everything H20-specific is **pending** until it runs on the target hardware.

| item | state |
|---|---|
| FastLabel core with scene-format adapters (`rind3d_posterior.fastlabel`, `window_from_100k`) | done on `yushu/3d-fastlabel-core`; byte-identical to the 1K backfill; tested on a real 100K container |
| `patches/100k_durability_5560a20.patch` | prepared and tested in scratch; **not applied** |
| `patches/100k_fastlabel_engine_5560a20.patch` (on top of the durability patch) | prepared and tested in scratch; **not applied** |
| H20 validation (Gate A, cross-device bit-exactness, throughput, canary) | **pending** |

## 1. Adapter

```python
from rind3d_posterior.fastlabel import label_window_fast, window_from_100k
lab = label_window_fast(window_from_100k(scene_path(root, slot), w, s), n_tasks=N_planned, base_seed=SEED_100K)
```

**What the adapter reads.** `window_from_100k` reads the container exactly as `rind3d_100k.storage.SceneFile` does:
* `meta.resolution` and `meta.slot`;
* `windows`, `obs` and `obs_index`, i.e. packed bits of bool [Z, Y, X], zstd, with s = −1 for SOLID;
* `geometry` (canonical bundle bytes), `P`, `K` and `sources_hex`.

The RNG scene index is the slot. Both stores use the same crop encoding, so a physical window gives the same task record in either format; `tests/test_fastlabel_core.py::test_window_from_100k_layout` checks this.

**Tested on a real container** (slot 0 of contract v1, rendered by `run_scene` into scratch; windows equal the handoff fixture):
* 14/14 mixed channels certified, with byte-identical reruns and the true source in the support of every label;
* Gate A on 2 windows: 188/188 identical fields, 0 batch disagreements, 0 false violations;
* FastLabel vs exact tau-pencil on 2 L64 channels: TV 0.0000 and 0.0007.

## 2. Engine patch (`patches/100k_fastlabel_engine_5560a20.patch`)

Minimal wiring. The tau-pencil path stays the default and is unchanged.

| file | change |
|---|---|
| `labels.py` | `engine == "fastlabel"` → `_fastlabel_task`. Journal arrays stay per-cell (`dcell`, `lo`, `hi`). **Status and `tv_bound` come from the certificate**: statistical labels have lo = hi, so the tau-pencil per-cell TV recomputation would wrongly give 0. Meta keeps the certificate, path, rounds, verifier counters and `config_sha256`. `n_tasks` is mandatory |
| `pool.py` | optional worker `init = (fn_path, args)`, run before the task loop (the 1K `supervise` hook) |
| `coordinator.py` | `label_engine = "fastlabel"`: every label worker becomes a FastLabel worker with CUDA, round-robin over the GPUs, with **one semaphore per GPU** (`set_gpu_lock`). No worker mixes FastLabel with the CuPy tau-pencil engine (CUDA-context rule, `RIND3D_Posterior/docs/FASTLABEL_CORE.md`) |
| `validate.py` | `certified_stat` requires a statistical certificate with ε ≤ 0.05, tv_bound = ε, 0 < δ < 1 and K ≤ a2 |
| `loader.py` | `flatten(certified_only=True)` keeps `certified_stat`. Without this, every statistical label (≈ 67% of the 1K) would be **silently dropped** |
| `scripts/run_node.py` | `--label-engine {tpe,fastlabel}`, `--fastlabel-n-tasks`, `--fastlabel-base-seed` |

**Out of scope of the patch:**
* the M2 loader (populations T* / T*_mis / B1K, `combined_rb`, explicit fallback);
* contract fields for κ and the coverage stream (`J_COMPLEXITY_COVERAGE.md`);
* the S6 all-dark prototype.

## 3. Integration order (for the 100K branch owner)

1. **Code.** Bring the FastLabel core into the 100K branch: merge `yushu/3d`, or cherry-pick `RIND3D_Posterior/src/rind3d_posterior/fastlabel/` with `pyproject.toml`, `docs/` and `tests/test_fastlabel_core.py`. The shared files (`RIND3D_Generator`, `RIND3D_Posterior` tau-pencil, `rind3d_1k.families` / `physical`) are identical on both branches, so the merge is textually conflict-free.
2. **Apply the durability patch, then the engine patch** (`git apply`, review, commit).
3. **Contract.** Do these before the canary:
   * set `labels.engine` to `fastlabel-v3.4` plus the core commit;
   * set `n_tasks` = N_planned, the planned informative-channel count of the campaign. δ = 0.01 / N_planned; it must not be changed after launch;
   * choose a new `base_seed`;
   * record the `DEFAULTS` sha256 that the labels will carry.
4. **Worker layout** (to be benchmarked on H20):
   * Phase 1 (tau-pencil, numba) is CPU-bound. The GPU verifier is short and serialised by the per-GPU semaphore.
   * The 1K ran 15 workers on one laptop GPU at 10.2 labels/s.
   * On H20, measure labels/s per GPU against workers per GPU (for example 4/8/12) before fixing `label_cpu`.
5. **H20 gates** (§5).

## 4. Durability review (`patches/100k_durability_5560a20.patch`)

| # | finding (at `5560a20`) | consequence | fix in the patch |
|---|---|---|---|
| D1 | `Coordinator.record` appends to `manifest/physical.jsonl` without flush+fsync; `failures.jsonl` likewise | a crash can lose the last records (benign: deterministic attempts are recomputed and re-written atomically), **or leave a torn last line: `_jsonl` then raises `JSONDecodeError` and the coordinator cannot resume** without manual repair | `append_line`: write, flush, fsync, plus a directory fsync on creation. `read_jsonl` ignores only a torn **final** line; interior corruption still raises |
| D2 | consolidated label files `labels/scenes/*/<slot>.labels.npz`: `atomic_write` returns a sha256, but `consolidate` discards it; neither `validate_labels` nor `verify_root` can detect corruption at rest | silent bit rot or a truncated copy of a consolidated label file passes verification as long as it parses | `labels/consolidated.jsonl` (slot, path, sha256, bytes, statuses; fsynced; last record wins). `validate_labels(..., cons_rec)` fails on a missing record or a sha256 mismatch; `verify_root --labels` passes the records |
| D3 | `atomic_write` fsyncs the file but not the directory after `os.replace` | after a power loss the rename may be lost while a later fsynced manifest record claims the container. The validator then reports a missing container | `fsync_dir(parent)` after the rename. Order kept: container durable, **then** the manifest record |
| D4 | `journal_index` accepts any 3-field `.idx` line; the `.idx` is flushed but not fsynced | a torn idx line can parse with a truncated length, so `read_record` raises during consolidation | only newline-terminated lines are used, and offset + length must lie within the journal file. A lost line only causes a harmless recomputation (first record wins) |

**Recommendation: apply before the H20 canary.** The changes are small (+71 / −19 lines), affect the write path only, and keep every format.

**Tests** (scratch):
* unit checks: torn tail ignored, interior corruption raises, unterminated complete record kept, durable append, idx with a torn line and an out-of-range entry, atomic_write sha256;
* end-to-end on slot 0: `run_node` with labels, then `verify_root --labels`, a resume with an injected torn manifest line, and a corrupted consolidated file detected (§6).

## 5. H20 validation, all **pending**

These preserve the checks of the 1K record; none may be skipped on a new device type or driver family.

1. **Environment.**
   * NVRTC matches the driver;
   * the FP64 / FP32 throughput probe (`scripts/probe_hardware.py`);
   * `RIND3D_Generator/tests`, `RIND3D_Posterior/tests` (tau-pencil digests, FastLabel core) and `RIND3D_Dataset1K/tests`.
2. **Directed rounding and the verifier.** Gate A with `python -m rind3d_posterior.fastlabel.gate_a`, run three times:
   * the default build on the 27 GPUVERIFY windows, or an equivalent 100K window set;
   * `FASTLABEL_VERIFY_DEFINES="E32=1e-3f"`, the FP64 interval path;
   * `"UNK_BAND=2e-4f"`, the GMP path.

   Pass iff field_diff = batch_disagree = bad_violation = 0 everywhere.
3. **Cross-device bit-exactness.**
   * `scripts/smoke_test.py`: label digests against the 1K references, and physical fixtures for slots 0–3, compared bit for bit;
   * FastLabel on the 47-task equivalence set of `FASTLABEL_CORE.md`, byte-identical to the laptop. RNG streams are keyed and integer, so any difference localises a numeric divergence.
4. **Accuracy on the 100K mix.** At least 200 exact tau-pencil references, stratified by (L, m, family) and including solid-separated channels. Report the max and quantiles of TV(FastLabel, exact) (1K: 447 references, max 0.0164).
5. **Throughput and tail.**
   * labels/s per GPU against the worker count;
   * the solid-separated share of time;
   * timeouts;
   * RSS per worker.
6. **Canary of at least 256 scenes** under the final contract (κ, coverage stream), with the checks of `100K_READINESS.md` §5. Then a crash test: `kill -9` the coordinator mid-run, resume, and require `verify_root --labels` to be clean.

## 6. Scratch test record (2026-11-13, laptop RTX 3060; scratch copies of `5560a20`, never the branch)

**Durability patch.**

| check | result |
|---|---|
| unit checks (6) | all pass: torn final line ignored, interior corruption raises, unterminated complete record kept, durable append, idx with torn and out-of-range lines ignored, `atomic_write` sha256 = file sha256 |
| `run_node --slots 0:1` with tau-pencil labels | 1 scene accepted, 32 labels, 0 failures; `labels/consolidated.jsonl` written; `verify_root --labels`: ok, physics 0 mismatches |
| torn manifest line injected | unpatched reader: `JSONDecodeError`. Patched coordinator: resumes, exit 0 |
| one bit flipped in the consolidated label file | `validate_labels`: "consolidated labels sha256 mismatch"; `verify_root`: ok = false. The intact root still validates |

**Engine patch** (applied on top of the durability patch; core from `yushu/3d-fastlabel-core`).

| check | result |
|---|---|
| `run_node --slots 0:2 --label-engine fastlabel --fastlabel-n-tasks 4800000 --fastlabel-base-seed 20261114` | 2 scenes, 52 labels (43 `certified_stat`, 9 `certified`), 0 failures, 0.57 labels/s with 2 laptop workers |
| `verify_root --labels` | ok: 2 scenes, physics 0 mismatches, statistical certificates valid, sha256 records present |
| journal records vs direct `label_window_fast` | 6/6 byte-identical (cells, masses, status, tv_bound) |
| FastLabel vs exact tau-pencil, all 32 channels of slot 0 | max TV 0.0147, all within the combined certified bounds |
| 100K loader `flatten` | unpatched: 9 of 52 labels kept (every `certified_stat` silently dropped). Patched: 52 of 52 |
