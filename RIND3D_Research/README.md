# RIND3D research archive

This directory preserves the RIND3D research that was done outside git before 2026-11-12:
* source code;
* reports and documentation;
* tests;
* small evidence files.

Each subdirectory is a faithful copy of the original working directory with these removed:
* generated outputs, virtual environments and caches;
* large arrays and private keys.

The maintained implementations derived from this research live in:
* `../RIND3D_Posterior/`: the posterior labeller package;
* `../RIND3D_Dataset1K/`: the 1K production pipeline.

Consolidated documentation is in `../docs/RIND3D/`.

| Path | Origin (local) | What it is | Status |
|---|---|---|---|
| `posterior_labels/study1_cepl/` | `RIND3D_PosteriorLabels/` | First exact-label study. CEPL engine: octree, exact polytope leaves, TV bound. | Correct on controlled cases. On real 1024³ windows the certificate is vacuous (0/23 windows at TV ≤ 0.05). Superseded. |
| `posterior_labels/phase1_tau_pencil/` | `RIND3D_PosteriorLabels2/` | Phase 1: the tau-pencil engine (TPE), Stage A at 128³ and Stage B at 1024³. | **Frozen.** `PHASE1_FREEZE_MANIFEST.json` lists the SHA-256 of every frozen file. The review ZIP is kept outside git. |
| `posterior_labels/phase2_gpu/` | `RIND3D_PosteriorLabels3_Phase2/` | Phase 2: source-excluded target, repeated-evaluation study, exact GPU evaluation pass, multicore comparison, P64 batches. | Final engine: `code/tpl/engine_g2.py` + `gpu_pass2.py` / `gpu_pass3.py` + `run_g1.py`. See `docs/PHASE2_STEP3_GPU_PASS.md`. Evidence JSON is in `evidence/`. |
| `posterior_labels/rpl_shared/` | `RIND3D_PosteriorLabels2/code/rpl/` | Shared predicates, TV bound and data access (`rpl`). One copy stands for all three studies, which held identical copies. | Frozen copy (`README_FROZEN.txt`, `SOURCE_SHA256.txt`). |
| `production_v02/` | `G2-02-gaussian/RIND3D_Production/` | Production-readiness v0.2: certified CSG core, contract v0.2, 216-scene canary. | The canary passed all gates. 1K was prepared but not launched. |
| `training_v03/` | `G2-02-gaussian/RIND3D_Training_v03/` | Training-ready v0.3: two-view export, P256 index, selection-law weights, loaders. | SB 1K was built from v0.1. HMAC release keys (`private/`) are **not** included. |

**Reproduction notes.**
* The archived code uses absolute paths of the original machine (`/home/yushu/Projects/RIND/...`). Where available, environment overrides exist, e.g. `RIND3D_ROOT` in `rpl/rindenv.py`.
* To run the Phase 1/2 code from this archive, put `posterior_labels/rpl_shared` on `PYTHONPATH` as package `rpl`.
* The large experiment outputs (npz, logs, shards) are not in git. They stay in the original working directories.
