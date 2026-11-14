# Validation record: consolidated `yushu/3d` (2026-11-13)

All checks ran on the development laptop: RTX 3060 Laptop GPU, NVIDIA driver 620.70, Ryzen 7 5800H, Python 3.12 (`RIND3D_HANDOFF/env.sh`).

## 0. Git

| item | value |
|---|---|
| archive | tag `archive/rind3d-research-2026-11-13` (object f0271b6) → `0270712`; branch `archive/yushu-3d-research-2026-11-13` → `0270712`; both on origin (`git ls-remote` verified) |
| consolidation commits | `52ac18c` removals, `6697c6d` configuration, `c26f230` documentation, and the handoff commit (this file) |
| archive recovery | `git show archive/rind3d-research-2026-11-13:RIND3D_Research/README.md` works. The tag holds 505 files under the removed paths |

## 1. Automated tests

| suite | command | result |
|---|---|---|
| `RIND3D_Dataset1K/tests` (loader ×4, FastLabel ×4 incl. GPU verifier vs the original renderer and the label determinism test, supervisor ×1) | `pytest -q tests` | **9 passed** (re-run after every commit) |
| `RIND3D_Posterior/tests` (tau-pencil CPU digest, GPU pass vs reference, normalisation, record-buffer rerun) | `pytest -q tests` | **9 passed**, 1 warning |
| `RIND3D_Generator/tests` (geometry, certified rendering, RINDPack, CountIndex, campaign, selection) | `OMP_NUM_THREADS=1 pytest -q tests` | **337 passed, 18 skipped** in 247 s |

**Generator skips.** They need a completed real Lite pilot or READY campaign root, or the separately built legacy comparison extensions (`RIND3D_Generator/README.md`).

**Import path of the Generator run.** The `rind3d` package was imported from the identical copy in the `yushu/3d-100k-h20` worktree, through the venv's `.pth` path. Its `src/` is byte-identical to this branch's apart from built extensions, and this worktree has no built `rind3d` native module.

## 2. Smoke checks (`RIND3D_HANDOFF/smoke_checks.py`, scenes 000010 / 000500 / 000999)

| check | result |
|---|---|
| local response: certified-CUDA re-render of stored windows (L = 64, 128, 256; SOLID + every channel) from geometry and the true sources, tile by tile with **no full-world field** | 76.5 M + 57.4 M + 57.4 M bits, **0 mismatches** |
| certificates: `verify_backfill.check_scene` on all labels (sha256 vs log, arrays, certificate fields and thresholds, true source inside the support) | 233 + 132 + 138 labels, **0 problems** |
| seals | 0 mismatches |
| loader: `flatten(population="Tstar")` | every label yielded as `posterior` (certified 78 + certified_stat 155 in scene 000010) |

## 3. Behaviour unchanged by the consolidation

| check | result |
|---|---|
| analyses E13, E14, E18, E19 re-run with the new configurable paths | output JSON **byte-identical** to the committed outputs |
| NAT-1K generator re-run for scene 000010 (`natural_supplement.py`) | observations.bin, observations_index.npy, windows.npy and natural_log.npz **byte-identical**; natural.json equal apart from wall time |
| Student reference (`pilot/student.py`): trained FL_MIS_s0 checkpoint on 16 pilot-eval rows | cell masses sum to 1 ± 4·10⁻⁷; NLL 10.84 (evaluation set mean 10.6–10.8); sparse CE equals the full log-mass to within 10⁻⁶ |
| diff review | the 28 modified code files contain only path, import and docstring changes. No algorithm, distribution or label-format line changed |

## 4. Data untouched

| check | result |
|---|---|
| `integrity_snapshot.py` over all 1000 release scenes | 25,550 sealed files, **0 seal mismatches**; 159,802 label files; 0 temporary files |
| label files against the post-backfill sha256 manifest (`RIND3D_1K_data/backfill_verification_v1/post_backfill_integrity.json`) | **0 scenes with any label hash change** |

## 5. Not verified here (and why)

* **Full physical-stage regeneration of a scene** (`run_physical.py`). It writes a new release root and takes about 25 s per scene. Covered indirectly by check 2.1 (bit-exact re-render) and the Generator test suite.
* **A fresh FastLabel run on a new window.** Covered by `test_label_task_vs_release_and_determinism` (in the 9 passing tests). No new campaign was started.
* **Student training.** Not re-run (no training runs in this task); the forward and normalisation smoke check passed.
* **The 100K branch.** Not run or modified. Its state is in `100K_READINESS.md`. Its own `smoke_test.py` was not executed.
* **Gate A** (`validate_verify.py`). Not re-run: same GPU and driver family as the recorded run. It is required on any new device type (H20).
* **CPU-only hosts.** Not tested. The CUDA tests skip there by design.

## 6. FastLabel core promotion (2026-11-13, branch `yushu/3d-fastlabel-core`)

FastLabel moved from `RIND3D_Dataset1K/rind3d_1k/fastlabel/` (`git mv`) into `rind3d_posterior.fastlabel`, with scene-format adapters and a 1K compatibility layer. Recovery point: tag `archive/pre-fastlabel-core-2026-11-13` → `b4f264a`. Details: `RIND3D_Posterior/docs/FASTLABEL_CORE.md`.

| check | result |
|---|---|
| old (`b4f264a` export) vs new, 47 train-split tasks × 2 runs each, every backfill (path, L, stage, round) stratum plus forced exact and validation branches | **byte-identical** task-file arrays and meta, and full `label_task` meta (timing excluded), in all 4 runs |
| new vs stored backfill labels, 42 production-config tasks | **byte-identical**, certificate metadata included |
| `DEFAULTS` sha256 | unchanged (`d4eb92d9…`) |
| runtime, warm | 55.4 s old vs 55.6 s new (median per-task ratio 1.003) |
| 100K container (slot 0, rendered into scratch) | 14/14 channels certified and deterministic; true source in every support; Gate A 188/188; TV vs exact 0.0000 / 0.0007 |
| tests | Posterior 16 passed (7 new), Dataset1K 9 passed, Generator 337 passed / 18 skipped, `smoke_checks.py` all OK |
| Gate A, production port (`gate_a.py`) | 1K smoke windows (test) and 2 windows of the 100K container: 0 field differences, 0 batch disagreements, 0 false violations. The full 27-window run is not repeated (same GPU and driver); it is **required on H20** |

## 7. RIND3D-100K integration (2026-11-14, laptop RTX 3060; branch `yushu/3d-100k-integration` → `yushu/3d`)

**What changed.**
* `yushu/3d-100k-h20` (`5560a20`) was merged into `yushu/3d`. The merge is textually conflict-free; that branch only added `RIND3D_Dataset100K/`.
* The two reviewed patches were applied: durability, and FastLabel as the engine.
* Contract v2 was implemented (`rind3d_100k/sampling.py`, `physical.select_side_v2`):
  * declared κ, natural T* sampling, the L64 coverage slot;
  * exact-m rate streams;
  * FastLabel as the default labeller.
* The M2 loader was ported.

| check | result |
|---|---|
| `RIND3D_Posterior/tests` | 16 passed |
| `RIND3D_Dataset1K/tests` | 9 passed |
| `RIND3D_Dataset100K/tests` (new) | 21 passed, including the GPU reproduction of a v2 fixture scene and its FastLabel labels |
| `RIND3D_Generator/tests` | 306 passed, 17 skipped in the integration worktree. The 34 tests of `test_cpu_cuda_experiment.py` / `test_shadow_experiment.py` need locally built experiment natives (untracked build artifacts): 31 passed, 3 skipped in the primary worktree. The Generator tree is identical (`cea09c8`) |
| `RIND3D_HANDOFF/smoke_checks.py` (1K) | 10/10 OK |
| `RIND3D_Dataset100K/scripts/smoke_test.py` | **SMOKE PASS**, all 9 steps: environment; Posterior + 100K pytest; TPE label fixtures (CPU / GPU); v1 physical fixtures; v2 physical fixtures; FastLabel v2 fixtures (12/12 digests); Gate A smoke (196 legal candidates, 0 differences); node run under v2 with FastLabel labels and label validation; journal recovery |
| v2 scenes, slots 0–3 | accepted, 4.1–4.5 s per scene (v1: 3.7 s); `validate_scene` ok; the exact-m rate stream dominates the tile count |
| κ draw, 20,000 slots | frequencies 0.153 / 0.151 / 0.148 / 0.549; lag-1 correlation −0.012 |
| crash test (`run_node` slots 0–8, SIGKILL of the coordinator at 2 scenes / 31 labels, resume) | resume completed 8 scenes and 203 labels, 0 failures, no duplicate accepted slot; `verify_root --labels` ok (physics 0 mismatches; certificates, true source in support and n_tasks valid) |
| `coverage_report.py` (8 scenes) | k histogram 1 / 0 / 0 / 7; T* Kish ESS fractions 0.92 (L64), 0.98 (L128), 0.99 (L256); max / median weight ≤ 1.35; coverage share of the L64 weight 0.29 (small sample) |
| `compare_engines.py` (26 tasks, L ≤ 128) | all within bounds; max TV 0.0067 |
| `student_smoke.py` | 48 loader examples (natural and coverage, both certified statuses) → pilot features → Student (2.0 M parameters). Weighted CE 14.55 (≈ ln 128³ untrained), finite; gradient norm 1.91 |
| `report.py --costs` + `estimate.py` | run end to end. Laptop projection: 25.4 labels per scene, 7.0 labels/s with 8 workers. Not an H20 number |

**Not executed (H20 pending):** every section of `RIND3D_Dataset100K/docs/H20_CANARY_RUNBOOK.md`.
