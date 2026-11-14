# RIND3D handoff: repository map, setup, entry points

This is Phase4 of the shared G2-02 repository. **All paths in this directory are relative to `Phase4_Multi_Source_3D/`**, the Phase4 root.

**Branch.** `yushu/3d` is the canonical RIND3D-100K pre-production branch. It holds:
* the shared physics and labelling modules;
* the 1K reference pipeline, kept for regression;
* the 100K production pipeline (`RIND3D_Dataset100K/`, integrated 2026-11-14).

**Other phases.** Phase1 (`Phase1_Single_Source_Single_Intensity/`) and Phase2 (`Phase2_Multi_Source/`) are the 2D project, maintained on `main`. Phase3 (`Phase3_Single_Source_3D/`) is the RIND3D v0.1 single-source reader, viewer and teacher experiments. All three sit at the repository root and are not part of the pipeline described here. Phase3 depends on `RIND3D_Generator/` through its uv path sources.

**Layout.** The RIND3D components moved from the repository root into `Phase4_Multi_Source_3D/` on 2026-11-14; their internal layout is unchanged.

| directory | role | status |
|---|---|---|
| `RIND3D_Generator/` | package `rind3d` plus native C++. Canonical convex-polyhedra geometry (vendor `rind`), certified visibility (CPU native, CPU reference, `cuda_certified`), RINDPack storage, exact CountIndex, the v0.1 geometry families and acceptance gate. This is an imported, hash-tracked source tree (`provenance/ORIGINAL_SOURCE.json`): keep it byte-stable. `experiments/{shadow_volumes,cpu_cuda,lite_quality}` are acceptance-test dependencies | production (used by 1K and 100K) |
| `RIND3D_Posterior/` | package `rind3d_posterior`: the exact tau-pencil labeller (`label_window`), CPU and GPU-pass variants, TV bound; **FastLabel v3.4 core** (`rind3d_posterior.fastlabel`: `label_window_fast`, adapters `window_from_1k` / `window_from_100k`, exact GPU verifier, Gate A `gate_a.py`; `docs/FASTLABEL_CORE.md`). `experiments/fastlabel/` keeps the FastLabel research record (REPORT, CERTIFICATE_AUDIT, GPUVERIFY) and the research Gate A harness (`validate_verify.py`) | production (exact labels; current 100K labeller; FastLabel core) |
| `RIND3D_Dataset1K/` | package `rind3d_1k`: physical stage, selection law and controller, label orchestration (`labels.py`, `supervise.py`), the 1K FastLabel entry points (`rind3d_1k/fastlabel/`, a compatibility layer over the core), the **loader** (`loader.py`), release validation; `scripts/`, `backfill/` (run, monitor, verify), `docs/`, `tests/` | production for the 1K (validated at 1K scale) |
| `RIND3D_Dataset100K/` | package `rind3d_100k`: **the 100K production entry point**. Contract v2 (`contract/RIND3D_100K_CONTRACT_v2.json`, provisional): declared κ, natural T* sampling, optional L64 coverage slot, exact-m rate streams. Also: FastLabel labels (tau-pencil as reference), the multi-GPU coordinator, durable storage, validators, the M2-style loader, fixtures, the smoke test, coverage / engine / Student smoke tools, and the H20 runbook `docs/H20_CANARY_RUNBOOK.md` | pre-production (validated on the laptop; the H20 gate is pending) |
| `RIND3D_StudentResearch/` | scientific decisions A–J for the current design, plus their tools: NAT-1K natural-window generator (`natural/`), the Student reference implementation (`pilot/`), the OOD benchmark (`ood/`), the all-dark S6 prototype (`s6/`), analyses E13–E19 (`experiments/`) | reference / research prototypes |
| `RIND3D_Audit/` | the scientific foundations audit (01–10) with its toy-world and fixture experiments | reference |
| `RIND3D_HANDOFF/` | this directory: implementation facts, cleanup manifest, validation record, 100K readiness and FastLabel integration plan (`100K_FASTLABEL_INTEGRATION.md`, `patches/`), figure sources of the Chinese design document (`figures/`), its review (`DOC_REVIEW_2026-11-13.md`), `env.sh`, `smoke_checks.py` | handoff |
| `docs/RIND3D_数据集设计与技术说明.md` | the Chinese design and technical document, with figures in `docs/assets/` | handoff |
| `docs/RIND3D/`, `docs/RIND3D_INTEGRATION.md` | RIND3D goals, semantics and history; the 2026-11-02 integration of the RIND3D v0.1 release into G2-02 (Phase 3 reader) | historical context |

**Related branches:**
* `yushu/3d-100k-h20` (`5560a20`): the historical development branch of the 100K v1 profile. It was merged into `yushu/3d` on 2026-11-14 and is not developed further.
* `yushu/gaussian-b2-native`: the Gaussian-B2 geometry research line (local branch).
* `main`: the shared 2D project.

**Archives** (pre-Phase4 layout: their paths start at the repository root, e.g. `RIND3D_Research/`, `RIND3D_HANDOFF/`):
* the full pre-consolidation research state: tag `archive/rind3d-research-2026-11-13` (commit `0270712`) and branch `archive/yushu-3d-research-2026-11-13`;
* the former 100K branch: tag `archive/rind3d-100k-h20-2026-11-14`;
* `yushu/3d` before the Phase4 reorganization: tag `archive/yushu-3d-pre-phase4-2026-11-14` (`e55e7d9`).

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
| FastLabel API (any scene format) | `rind3d_posterior.fastlabel.label_window_fast(window_from_1k(scene_dir, w, s))` or `window_from_100k(container, w, s)` |
| Gate A (verifier vs certified renderer) | `$RIND3D_LABEL_PY -m rind3d_posterior.fastlabel.gate_a out.jsonl 1k:<scene_dir>:<w>:<s> ...` |
| FastLabel labels | `RIND3D_Dataset1K/backfill/run_backfill.sh`, i.e. `run_labels.py <root> 15 0 --engine fastlabel --scene-order --retry-failed`; set `n_tasks` for any new campaign |
| verify labels and seals | `RIND3D_Dataset1K/backfill/verify_backfill.py <root> <pre_integrity.json> <out.json> [workers] [n_tasks]` |
| natural-window supplement | `RIND3D_StudentResearch/natural/natural_supplement.py <release> <out> a:b`, then `run_nat_labels.sh` |
| training examples | `rind3d_1k.loader.flatten(ds, population="Tstar"|"Tstar_mis"|"B1K", ...)` |
| Student pilot | `RIND3D_StudentResearch/pilot/build_features.py`, `run_arms.py`, `final_analysis.sh` |
| smoke checks | `$RIND3D_LABEL_PY RIND3D_HANDOFF/smoke_checks.py` |
| **100K node** (contract v2, FastLabel) | `$RIND3D_LABEL_PY RIND3D_Dataset100K/scripts/run_node.py --root <root> --slots a:b --gpus all`; then `verify_root.py <root> --labels` and `coverage_report.py <root>` |
| **100K H20 gate** | `RIND3D_Dataset100K/docs/H20_CANARY_RUNBOOK.md`; `RIND3D_Dataset100K/scripts/smoke_test.py --work <dir>` |

## Tests

```bash
cd Phase4_Multi_Source_3D                                                  # from the repository root
source RIND3D_HANDOFF/env.sh
(cd RIND3D_Dataset1K && $RIND3D_LABEL_PY -m pytest -q tests)                 # loader, FastLabel (GPU), supervisor
(cd RIND3D_Posterior && $RIND3D_LABEL_PY -m pytest -q tests)                 # tau-pencil digests, FastLabel core (GPU)
(cd RIND3D_Generator && OMP_NUM_THREADS=1 $RIND3D_LABEL_PY -m pytest -q tests) # geometry, rendering, RINDPack, counts
(cd RIND3D_Dataset100K && $RIND3D_LABEL_PY -m pytest -q tests)               # 100K sampling, weights, storage, loader, GPU fixtures
$RIND3D_LABEL_PY RIND3D_HANDOFF/smoke_checks.py                              # re-render, certificates, loader, seals
```

## Data outside Git

**Location:** `RIND3D_1K_data/`, about 60 GB, never committed. It is inventoried in `CLEANUP_MANIFEST.md` §4.

**Frozen.** The release `release_1k_v1` (physical data sealed by `SEAL.json`, plus 159,802 certified labels) must not be modified.
