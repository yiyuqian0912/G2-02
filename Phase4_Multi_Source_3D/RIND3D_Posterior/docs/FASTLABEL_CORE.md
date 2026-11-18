# FastLabel core (`rind3d_posterior.fastlabel`)

**FastLabel v3.4** computes certified posterior labels quickly: a tau-pencil outer bound, followed by exact GPU verification and statistical certification.

On 2026-11-13 it moved from `RIND3D_Dataset1K/rind3d_1k/fastlabel/` into this package, on branch `yushu/3d-fastlabel-core`:
* Every module was moved with `git mv`, so history is kept.
* The algorithm is unchanged.
* The scene format now enters only through adapters.
* Recovery point: tag `archive/pre-fastlabel-core-2026-11-13`, which points to `b4f264a`, the last commit with the 1K-local package.

Research record (method, benchmarks, certificate audit, kernel validation): `experiments/fastlabel/`, i.e. `REPORT.md`, `CERTIFICATE_AUDIT.md` and `GPUVERIFY.md`.

## Layout

| module | role |
|---|---|
| `inputs.py` | `FastLabelWindow` (the task record), `legal_source` (exact source domain D(G)), `POLICY`, adapters `window_from_1k`, `window_from_100k` |
| `pipeline.py` | `label_window_fast(task, **cfg)`, `DEFAULTS`, `VERSION = "fastlabel-v3.4"`, `STAT_KIND = 4`, certificate design (`design2`, `kstar`, `cp_lower`), keyed RNG (`label_rng`), per-process CUDA factory and scene cache, `set_gpu_lock` |
| `engine.py`, `filters.py` | tau-pencil phase 1 (stage 1 octree, stage 2 BSP with the outer-bound filter), receiver items |
| `sampler.py` | `BSampler`: i.i.d. Uniform(B) over the stage-2 pieces |
| `gpuverify.py`, `verify.cu`, `fastplanes.py` | exact batched GPU verifier, `WindowVerifier(fac, ctx, task, cls)`: FP32 filter (E32 = 2⁻¹⁸), then FP64 directed intervals, then the native GMP predicate |
| `window.py` | class tensor from packed crops (`cls_from_packed`, `window_classes`), `LazyWindow2` (numba boundary mask and top items) |
| `gate_a.py` | Gate A: verifier against the certified CUDA renderer (CLI and `check_window`) |

## API

```python
from rind3d_posterior.fastlabel import label_window_fast, window_from_1k, window_from_100k, DEFAULTS

task = window_from_1k("<release>/scenes/000007", w, s)       # or window_from_100k("<root>/scenes/0000/000123.npz", w, s)
lab = label_window_fast(task, n_tasks=N_planned)              # dict(cell, lo, hi, kind, tv_bound, status, meta)
```

**Task record.** A `FastLabelWindow` holds:
* the geometry rows `P` and `K` (float64, for the engine) and the canonical `bundle` (exact predicates);
* the window `origin`, `L` and `N`;
* the stored crops `lit_bits` and `solid_bits`. Both are `np.packbits` of bool [Z, Y, X], big-endian, which is the encoding of both the 1K and the 100K stores;
* `key = (scene_index, w, s)`;
* `scene_key`, the cache key of per-scene objects;
* `bind_point`, a legal source used only to bind the native evaluator for the SOLID membership check, which is source-free;
* `policy`.

**Determinism.** Every random stream is keyed by `(base_seed, *key)`. The label is a deterministic function of the task record and the configuration. The 1K uses `scene_index = int(scene directory name)`; the 100K adapter uses the container's slot.

**Configuration.**
* `DEFAULTS` is byte-identical to the 1K production configuration: sha256 `d4eb92d9…` of the JSON, recorded as `config_sha256` in every 1K FastLabel label.
* `delta = family_delta / n_tasks`. For any new campaign, set `n_tasks` to its planned label count (the union bound over the campaign), and preferably a new `base_seed`.

**Statuses.**
* `certified`: deterministic TV ≤ 0.05; single-cell support or an exact path.
* `certified_stat`: TV ≤ ε = 0.05 with the stated confidence. Stored as lo = hi = vol(B ∩ cell), kind 4, with the certificate in `meta`.
* `certified_loose` and `uncertified`: exact fallback with a weaker bound.

**1K compatibility.** `rind3d_1k.fastlabel` re-exports the core and keeps the old 1K signatures:
* `label_task(scene_dir, w, s)`;
* the SceneObs-based `scene_cache`;
* `fast_cls3(S, w, s)`;
* `WindowVerifier(fac, ctx, S, w, s, cls)`;
* `set_gpu_lock`; `pipeline.GPU_LOCK` reads the core's live value.

Unchanged callers: `run_labels.py --engine fastlabel`, `labels._run_fastlabel`, `tests/test_fastlabel.py`, `smoke_checks.py` and the S6 / OOD research tools.

## Equivalence record of the move (2026-11-13, RTX 3060 Laptop)

**Tasks.** 47 train-split 1K tasks from 7 scenes, covering every (path, L, certificate stage, round) stratum of the backfill:
* 6 `single_cell_support`;
* 36 `stat_certified`, stages 1 and 2, rounds 1 and ≥ 2;
* 3 forced `exact_small_B` (config `small_records=1e9`);
* 2 with the `validate_n=256` empirical-validation branch.

**Runs.** Old code (export of `b4f264a`) and new code, each in its own process, twice: old, new, old, new.
* Cell masses, support, kinds, tv_bound, status, certificate metadata, rounds and verifier counters were compared, excluding timing fields.
* All 47 are byte-identical across the four runs, for the task-file arrays, the task-file meta and the full `label_task` meta.
* All 42 production-config tasks are byte-identical to the stored backfill labels, certificate included.

**Runtime.** 55.4 s (old) vs 55.6 s (new) warm. The median per-task ratio is 1.003. The first old run (107.8 s) included numba compilation in a fresh cache.

**100K container** (slot 0 rendered by `rind3d_100k.physical.run_scene` into a scratch directory; its windows equal the 100K handoff fixture):
* 14 mixed channels (L = 64/128/256) certified: 13 `certified_stat`, 1 single-cell. Reruns are byte-identical, and the true source is in the support of every label.
* Gate A on 2 windows: 188/188 fields identical, 0 batch disagreements, 0 false violations.
* FastLabel vs the exact tau-pencil label on 2 L64 channels: TV 0.0000 and 0.0007.

**Tests.**
* `RIND3D_Posterior/tests`: 16 passed (7 new FastLabel-core tests, 9 regression).
* `RIND3D_Dataset1K/tests`: 9 passed.
* `RIND3D_Generator/tests`: 337 passed, 18 skipped.
* `RIND3D_HANDOFF/smoke_checks.py`: all OK.

## Gate A (production port)

```bash
python -m rind3d_posterior.fastlabel.gate_a out.jsonl 1k:<scene_dir>:<w>:<s> 100k:<container.npz>:<w>:<s> ...
FASTLABEL_VERIFY_DEFINES="E32=1e-3f" python -m rind3d_posterior.fastlabel.gate_a ...    # forces the FP64 interval path
FASTLABEL_VERIFY_DEFINES="UNK_BAND=2e-4f" python -m rind3d_posterior.fastlabel.gate_a ... # forces the native GMP path
```

**Candidates and checks.** The candidate kinds and the checks are those of `experiments/fastlabel/validate_verify.py` (GPUVERIFY.md).
* The reference is the certified CUDA renderer (`packed_batch`) over every 64³ tile of the window.
* The seed per window is `crc32(spec)`, which makes runs reproducible; the research harness used Python's salted `hash`.
* A window passes iff `field_diff = batch_disagree = bad_violation = 0`.

**Required on every new GPU type or driver family** (H20: **pending**):
* the 27-window default build;
* the two forced-path builds;
* `RIND3D_Dataset1K/tests/test_fastlabel.py`;
* `tests/test_fastlabel_core.py`.

## Operational constraint: CUDA contexts

The rind3d CUDA runtime creates its own non-primary context (`cuCtxCreate`) and leaves it current on the thread. Running CuPy code afterwards in the same process, such as the tau-pencil `engine="gpu"` pass, fails with `CUDA_ERROR_INVALID_HANDLE`. This is pre-existing behaviour, identical before and after the move.

**Rule:** FastLabel workers and CuPy tau-pencil workers must be separate processes. The 1K already runs them that way. The CUDA tests in `tests/test_fastlabel_core.py` run in a spawned subprocess for the same reason.

## Solid-separated windows: cost tail and low-risk optimisations (documented, not implemented)

**Mechanism** (measured at 1K scale; `RIND3D_HANDOFF/RIND3D_IMPLEMENTATION_FACTS.md` §1):
* A solid-separated mixed channel has no 6-adjacent active/inactive FREE pair (no direct response interface), so `boundary_mask` is empty.
* Phase 1 then starts from the 128 interior receivers alone; `point_items` draws `nbd = 256` boundary and `nin = 128` interior receivers.
* Stage 1's sentinels are only the `n_class` active and inactive interior picks.
* The single-cell exit never applies.
* B stays broad (6–8·10⁵ pieces, 3–6·10⁵ cells), and 72–78% of the 13.4 s mean goes to stage-2 BSP and cell emission, not to certification rounds.

**Output-preserving options** (labels stay byte-identical; only cost changes):
1. **Scheduling.** The class is known from the stored crops before labelling (empty boundary mask, an O(L³) numba pass). Dispatching these tasks first, longest-processing-time-first, and to a dedicated heavy queue shortens the campaign's makespan tail. It does not change the total compute.
2. **Profiling-guided engineering inside stage 2** (emission buffers, cell accumulation). This is admissible only with the bit-equality harness of this document. Any change of floating-point summation order changes the bytes and is therefore a new engine version.

**Output-changing options** (new version; certificate semantics unchanged because B ⊇ F holds for any receiver subset; requires Gate A, the 447-reference accuracy check and a new `config_sha256`):
1. **SOLID-adjacent receivers as boundary items.** These are FREE receivers 6-adjacent to SOLID, whose rays graze the occluder's silhouette (audit O5). Untested.
2. **More interior receivers** (`nin`) when `boundary_mask` is empty, trading phase-1 items against BSP width.

Neither weakens the guarantee. Both need a fresh equivalence and accuracy record before production use.
