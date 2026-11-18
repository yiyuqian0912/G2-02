# RIND3D documentation index

> **2026-11-13:** paths under `RIND3D_Research/` refer to the archive tag `archive/rind3d-research-2026-11-13` (removed from the active branch). Current RIND3D overview: `RIND3D_HANDOFF/README.md`.

RIND3D is the 3D inverse-ray-geometry dataset and research line of G2-02. Its task is inverse inference of a hidden
driver source from one local window of binary responses produced by occlusion. Terminology, and the boundary between
full-world (teacher) information and local-only Student inputs: [GLOSSARY.md](GLOSSARY.md).

Each entry below has a status:
* **proven**: a mathematical statement with proof;
* **tested**: empirical behaviour on stated samples;
* **estimate**: a projection from measurements;
* **open**: an unresolved research question.

| # | Topic | Authoritative document(s) | Status |
|---|---|---|---|
| 0 | Terminology and information boundaries | [GLOSSARY.md](GLOSSARY.md) | definitions |
| 1 | Scientific goals | [GOALS.md](GOALS.md) | design |
| 2 | Canonical geometry and physical response semantics | [SEMANTICS.md](SEMANTICS.md); `RIND3D_Generator/docs/` | implemented; certified renderer tested |
| 3 | Local windows and Student-visible inputs | `RIND3D_Dataset1K/docs/SELECTION_LAW_1K.md` §1; `RIND3D_Dataset1K/rind3d_1k/loader.py` | implemented |
| 4 | Single-source inverse geometry | `RIND3D_Research/posterior_labels/phase1_tau_pencil/RIND3D_POSTERIOR_LABELING_REPORT.md` | proven lemmas, tested engine |
| 5 | Tau-pencil derivation (Lemmas 1, 2, 2W; quasi-linearity; exact collapses) | `RIND3D_Research/posterior_labels/phase1_tau_pencil/docs/METHOD.md` | proven |
| 6 | Posterior-label semantics and error guarantees | `RIND3D_Research/posterior_labels/phase2_gpu/docs/TARGET_V03_SOURCE_EXCLUDED.md`; `RIND3D_Dataset1K/docs/SELECTION_LAW_1K.md` §2–5 | proven (per-channel theorem); bounded terms stated |
| 7 | Phase 1 experiments | `RIND3D_Research/posterior_labels/phase1_tau_pencil/` (report, docs/RESULTS_TABLES.md, freeze manifest) | tested (frozen) |
| 8 | Phase 2 and Phase 3 optimization | `RIND3D_Research/posterior_labels/phase2_gpu/docs/PHASE2_STEP3_GPU_PASS.md`; `RIND3D_Dataset1K/docs/OPTIMIZATION_REPORT.md` | tested, including negative results |
| 9 | Scene-level dataset production | `RIND3D_Dataset1K/README.md`; `RIND3D_Dataset1K/docs/CONTRACT_1K.md` | implemented |
| 10 | Multi-source channel storage and flattened Student training | `RIND3D_Dataset1K/docs/CONTRACT_1K.md` §storage; `rind3d_1k/loader.py` | implemented |
| 11 | Geometry diversity and generator limitations | `RIND3D_Dataset1K/docs/GEOMETRY_DIVERSITY.md` | tested at 128³ |
| 12 | The 1K dataset contract | `RIND3D_Dataset1K/docs/CONTRACT_1K.md` | frozen v1 |
| 13 | Validation and reproducibility | `RIND3D_Dataset1K/docs/REPRODUCE.md` | implemented |

## Not proven, by design

* **Bit-identity of the GPU pass with the CPU reference** is a software equivalence on the tested windows. It is not a proof of real-arithmetic correctness.
* **The engine's exactness** relies on the decision margins of the floating-point predicates (Phase-1 METHOD.md).
* **Statistical terms** in the label error budget are estimates with stated confidence (SELECTION_LAW_1K §3):
  * the scene-rejection bound;
  * the training-weight estimation.

## RIND3D-100K (`RIND3D_Dataset100K/` on this branch)

The local-only 100K production pipeline for a 16 × H20 server, integrated into `yushu/3d` on 2026-11-14 (`RIND3D_Dataset100K/README.md`). The former branch `yushu/3d-100k-h20` is historical.
* **Contract v2 (provisional):**
  * 100K scenes at 1024³, 4 / 4 / 4 windows, 1–4 sources;
  * declared κ and natural T* sampling, with an optional L64 coverage slot;
  * FastLabel posteriors.
* **Pipeline:** exact local rendering, proven equal to the full-field pipeline.
* **Node coordinator:** multi-GPU, supervised, with durable storage.
* **Production gate:** `RIND3D_Dataset100K/docs/H20_CANARY_RUNBOOK.md`. Everything on the H20 is pending.
