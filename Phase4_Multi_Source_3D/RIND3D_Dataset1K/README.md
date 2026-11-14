# RIND3D 1K (contract `rind3d-1k-contract-v1`): source-separated multi-source scenes with certified per-source posteriors

> **Status (2026-11-13).** Physical stage: 1000/1000 scenes, sealed. Labels: all 159,802 informative channels are certified
> (FastLabel v3.4 backfill: 52,190 single-cell + 107,279 statistical; plus 333 tau-pencil labels), verified in
> `../RIND3D_StudentResearch/D_BACKFILL_REPORT.md`. The m-balanced window law below is the release's *selection law*,
> not the Student's target: training must declare a population (`loader.flatten(population=...)`, see
> `../RIND3D_StudentResearch/A_TASK_DEFINITION.md` and `C_LOADER.md`). Handoff overview: `../RIND3D_HANDOFF/`.

**Each scene has:**
* 1 to 4 physical sources;
* exactly 64 local windows: 16 of side 64, 24 of side 128 and 24 of side 256, in physical voxels at N = 1024;
* every source outside every closed window box;
* one binary response channel per source, kept separate.

**Each window keeps the source axis explicitly.** It stores the local SOLID mask once, one response per source, and one posterior slot per source with its certification status.

**Balance.** The dataset-wide counts of windows with m = 1, 2, 3 and 4 effective informative layers are balanced by `rind3d_1k/controller.py`.

**Flattening into single-source examples** is a downstream adapter (`rind3d_1k/loader.py`).

**Two stages, both resumable:**
1. **Physical stage** (producer venv with the certified rind3d natives): `scripts/run_physical.py`. It writes geometry, sources, certified per-source full fields, count indices, quota selection with thinning, the rate streams and the window observations.
2. **Label stage** (labeller venv: numba, CuPy): `scripts/run_labels.py`. It writes one certified posterior per (window, source): `--engine fastlabel` (FastLabel v3.4, the engine of the 1K backfill; core `rind3d_posterior.fastlabel`, 1K entry points `rind3d_1k/fastlabel`) or the exact tau-pencil engine of `../RIND3D_Posterior`.

**Read these first:**
* `docs/SELECTION_LAW_1K.md`: the generative process, the per-channel posterior theorem, the weights and the error budget.
* `docs/CONTRACT_1K.md`: the frozen contract.
