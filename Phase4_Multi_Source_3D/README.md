# Phase4: multi-source 3D source localization (RIND3D-100K)

**The question.** Phase4 studies global driver-source localization from limited local observations in three-dimensional multi-source worlds: inverse inference of a hidden driver source from one local window of binary responses produced by occlusion.
* **The worlds:** 1024³ worlds of closed convex polyhedra with 1–4 point driver sources and exact closed-segment
  source-to-receiver visibility.
* **What a Student sees:** one local window and one source-specific response channel: the SOLID crop, that source's binary response, the side and the origin.
* **What it predicts:** that source's posterior on a global 128³ grid.
* **Supervision:** certified posterior labels computed from the full geometry, which the Student never sees.

**Status (release 2026-11-18/19).** `yushu/3d` is the authoritative RIND3D-100K v3 production branch.
* **Production profile:** contract v3 (`RIND3D_100K_V3_CONTRACT.md`).
  * Direct FREE–FREE response-interface eligibility.
  * Observable thin slabs (≥ 2 voxels, 6-connected sampling).
  * Nine convex-polyhedral families.
  * **FastLabel v3.4** statistical certification (`RIND3D_Posterior/docs/FASTLABEL_V34_CERTIFICATION.md`).
* **Validated on a laptop:** tests, smoke test, and a 36-scene canary with crash/resume. See
  `RELEASE_CANARY_2026-11-18.md`.
* **Not yet run:** the 16 × H20 production gate (`RIND3D_Dataset100K/docs/H20_CANARY_RUNBOOK.md`). The 100K campaign
  has not been launched.
* **Start here:**
  * `HANDOFF.md`: what is done, what is next;
  * `PRODUCTION_RUNBOOK.md`: commands;
  * `NEXT.md`: open actions;
  * `RESEARCH_ARCHIVE.md`: XLF, deterministic certification, local-first, Astra's results.
* **Research, not on the production path:**
  * local inverse geometry and the local-first studies (`docs/LOCAL_INVERSE_GEOMETRY.md`,
    `RIND3D_Posterior/docs/LOCALFIRST_COMPARISON.md`);
  * curved occluders: closed balls, physical stage only, no certified labels, refused by the production profile
    (`docs/CURVED_ASSESSMENT.md`, `docs/CURVED_OCCLUDERS.md`);
  * the per-family generator benchmark (`RIND3D_Dataset100K/docs/GENERATOR_BENCHMARK_V3.md`).

**Terminology.**
* RIND3D is presented as a general driver-source / binary-response inference framework. A receiver is
  *response-active* or *response-inactive*, and the two meet at a *response interface*.
* The implemented physics is still a visibility-based binary response: active iff the closed segment from the driver
  source misses the closed solid. The general vocabulary does not mean other response mechanisms are supported.
* The 1K formats and internal arrays keep their legacy names (`lit`, `dark`, `cls` 0/1/2: 0 inactive, 1 active,
  2 SOLID). The v3 container metadata uses `active_frac`, and the loader exposes `response`/`responses`,
  `informative` and `driver_sources()`. No 100K data existed before v3, so no 100K compatibility mode is kept.
* Definitions, legacy names, and the boundary between full-world (teacher) information and local-only Student inputs:
  `docs/RIND3D/GLOSSARY.md`.

## Where things are

| path | content |
|---|---|
| `RIND3D_Dataset100K/` | **the 100K dataset pipeline**: contract v3 (production), local-only physical stage, FastLabel labels, multi-GPU coordinator, durable storage, validators, the training loader, the smoke test (`README.md`) |
| `RIND3D_Posterior/` | posterior labellers: exact tau-pencil, and **FastLabel** (`src/rind3d_posterior/fastlabel/`, `docs/FASTLABEL_CORE.md`) |
| `RIND3D_Generator/` | **the generator**: canonical geometry, certified CPU / CUDA visibility, storage. A provenance-tracked source tree, also used by Phase3 |
| `RIND3D_Dataset1K/` | the 1K reference pipeline and its release tools, kept for regression and reproducibility |
| `RIND3D_StudentResearch/` | task definition, Student reference (`pilot/`), coverage and generalization studies |
| `RIND3D_Audit/` | scientific foundations audit |
| `RIND3D_HANDOFF/` | environment (`env.sh`), entry points, implementation facts, validation record, figure sources |
| `docs/` | **the Chinese dataset document** `RIND3D_数据集设计与技术说明.md` (nine figures in `docs/assets/`), RIND3D goals, semantics, glossary and history |

## Set up and run the lightweight tests

```bash
cd Phase4_Multi_Source_3D
source RIND3D_HANDOFF/env.sh            # interpreters, data root, NVRTC; override any variable before sourcing
(cd RIND3D_Posterior   && $RIND3D_LABEL_PY -m pytest -q tests)
(cd RIND3D_Dataset1K   && $RIND3D_LABEL_PY -m pytest -q tests)
(cd RIND3D_Dataset100K && $RIND3D_LABEL_PY -m pytest -q tests)
$RIND3D_LABEL_PY RIND3D_Dataset100K/scripts/smoke_test.py --work /tmp/r3d_smoke   # GPU; includes Gate A
```

**Requirements.**
* The environment build is described in `RIND3D_Generator/README.md` (natives) and `RIND3D_HANDOFF/README.md` (Python, PyTorch, paths).
* CUDA tests skip without a GPU.
* The 1K data tests skip without the 1K release.

## Further reading

* **H20 production gate:** `RIND3D_Dataset100K/docs/H20_CANARY_RUNBOOK.md`.
* **Implementation facts and validation:** `RIND3D_HANDOFF/RIND3D_IMPLEMENTATION_FACTS.md`, `RIND3D_HANDOFF/VALIDATION_RECORD.md`.
* **Historical research** is archived in Git:
  * `archive/rind3d-research-2026-11-13`: the pre-consolidation research tree;
  * `archive/rind3d-100k-h20-2026-11-14`: the former 100K branch;
  * `archive/yushu-3d-pre-phase4-2026-11-14`: `yushu/3d` before this directory existed.

  The tags keep the pre-Phase4 paths (components at the repository root).
