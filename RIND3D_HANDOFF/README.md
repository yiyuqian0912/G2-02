# RIND3D handoff: repository map, setup, entry points

This branch (`yushu/3d`) belongs to the shared G2-02 repository. The 2D project (`Phase1_Single_Source_Single_Intensity`,
`Phase2_Multi_Source`) and the early 3D Phase 3 code (`Phase3_Single_Source_3D`) come from `main` and are not part of
the RIND3D pipeline described here. The RIND3D code lives in the `RIND3D_*` directories.

| directory | role | status |
|---|---|---|
| `RIND3D_Generator/` | package `rind3d` plus native C++. Canonical convex-polyhedra geometry (vendor `rind`), certified visibility (CPU native, CPU reference, `cuda_certified`), RINDPack storage, exact CountIndex, the v0.1 geometry families and acceptance gate. This is an imported, hash-tracked source tree (`provenance/ORIGINAL_SOURCE.json`): keep it byte-stable. `experiments/{shadow_volumes,cpu_cuda,lite_quality}` are acceptance-test dependencies | production (used by 1K and 100K) |
| `RIND3D_Posterior/` | package `rind3d_posterior`: the exact tau-pencil labeller (`label_window`), CPU and GPU-pass variants, TV bound. `experiments/fastlabel/` keeps the FastLabel research record (REPORT, CERTIFICATE_AUDIT, GPUVERIFY) and the Gate A verifier harness (`validate_verify.py`); `experiments/twophase/twophase.py` is a harness dependency | production (exact labels; current 100K labeller) |
| `RIND3D_Dataset1K/` | package `rind3d_1k`: physical stage, selection law and controller, label orchestration (`labels.py`, `supervise.py`), **FastLabel v3.4** (`rind3d_1k/fastlabel/`), the **loader** (`loader.py`), release validation; `scripts/`, `backfill/` (run, monitor, verify), `docs/`, `tests/` | production for the 1K (validated at 1K scale) |
| `RIND3D_StudentResearch/` | scientific decisions A–J for the current design, plus their tools: NAT-1K natural-window generator (`natural/`), the Student reference implementation (`pilot/`), the OOD benchmark (`ood/`), the all-dark S6 prototype (`s6/`), analyses E13–E19 (`experiments/`) | reference / research prototypes |
| `RIND3D_Audit/` | the scientific foundations audit (01–10) with its toy-world and fixture experiments | reference |
| `RIND3D_HANDOFF/` | this directory: implementation facts, cleanup manifest, validation record, 100K readiness, `env.sh`, `smoke_checks.py` | handoff |
| `docs/RIND3D*`, `docs/RIND3D_INTEGRATION.md` | the 2026-11-02 integration of the RIND3D v0.1 release into G2-02 (Phase 3 reader) | historical context |

**Related branches:**
* `yushu/3d-100k-h20`: the RIND3D-100K production implementation (`RIND3D_Dataset100K/`, local-only rendering, multi-GPU coordinator, 16×H20 handoff). It is **not merged**; see `100K_READINESS.md`.
* `yushu/gaussian-b2-native`: the Gaussian-B2 geometry research line (local branch).
* `main`: the shared 2D project.

**Archive of the full pre-consolidation research state:**
* tag `archive/rind3d-research-2026-11-13`, commit `0270712`;
* branch `archive/yushu-3d-research-2026-11-13`.

```bash
git fetch origin --tags
git show archive/rind3d-research-2026-11-13:RIND3D_Research/README.md          # read a removed file
git checkout archive/rind3d-research-2026-11-13 -- RIND3D_Research              # restore a removed tree
```

## Setup (Linux x86_64, Python 3.12)

**1. Labeller / physics environment.**
* Build the generator as in `RIND3D_Generator/README.md` (C++20, CMake, GMP/MPFR, zstd; `scripts/build.sh`).
* Add `numba`, `scipy`, `zstandard` and an NVRTC 12 library (e.g. the `nvidia-cuda-nvrtc-cu12` wheel) for the CUDA backend and FastLabel.
* CUDA is optional for physics on the CPU backends. It is **required** for FastLabel.

**2. Student environment.** Python 3.12 with CUDA PyTorch, numpy and scipy.

**3. Paths.** `source RIND3D_HANDOFF/env.sh`. It sets `RIND3D_REPO`, `RIND3D_1K_DATA`, `RIND3D_LABEL_PY`, `RIND3D_TORCH_PY`, `RIND_NVRTC_LIBRARY` and `PYTHONPATH`. Override any variable before sourcing.

## Main entry points

| task | command |
|---|---|
| physical scenes (1K reference) | `$RIND3D_LABEL_PY RIND3D_Dataset1K/scripts/run_physical.py ...` (see `RIND3D_Dataset1K/docs/REPRODUCE.md`) |
| exact labels (tau-pencil) | `RIND3D_Dataset1K/scripts/run_labels.py <root> <workers> <gpu_workers>`; API `rind3d_posterior.label_window` |
| FastLabel labels | `RIND3D_Dataset1K/backfill/run_backfill.sh`, i.e. `run_labels.py <root> 15 0 --engine fastlabel --scene-order --retry-failed`; set `n_tasks` for any new campaign |
| verify labels and seals | `RIND3D_Dataset1K/backfill/verify_backfill.py <root> <pre_integrity.json> <out.json> [workers] [n_tasks]` |
| natural-window supplement | `RIND3D_StudentResearch/natural/natural_supplement.py <release> <out> a:b`, then `run_nat_labels.sh` |
| training examples | `rind3d_1k.loader.flatten(ds, population="Tstar"|"Tstar_mis"|"B1K", ...)` |
| Student pilot | `RIND3D_StudentResearch/pilot/build_features.py`, `run_arms.py`, `final_analysis.sh` |
| smoke checks | `$RIND3D_LABEL_PY RIND3D_HANDOFF/smoke_checks.py` |

## Tests

```bash
source RIND3D_HANDOFF/env.sh
(cd RIND3D_Dataset1K && $RIND3D_LABEL_PY -m pytest -q tests)                 # loader, FastLabel (GPU), supervisor
(cd RIND3D_Posterior && $RIND3D_LABEL_PY -m pytest -q tests)                 # tau-pencil regression digests
(cd RIND3D_Generator && OMP_NUM_THREADS=1 $RIND3D_LABEL_PY -m pytest -q tests) # geometry, rendering, RINDPack, counts
$RIND3D_LABEL_PY RIND3D_HANDOFF/smoke_checks.py                              # re-render, certificates, loader, seals
```

## Data outside Git

**Location:** `RIND3D_1K_data/`, about 60 GB, never committed. It is inventoried in `CLEANUP_MANIFEST.md` §4.

**Frozen.** The release `release_1k_v1` (physical data sealed by `SEAL.json`, plus 159,802 certified labels) must not be modified.
