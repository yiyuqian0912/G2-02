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
