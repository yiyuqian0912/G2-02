# RIND3D Student lab (`yushu/3d-experiment`)

This is research on Students that infer the global 128³ source posterior from one window channel. It is separate from
the validated production code. Nothing here changes the 1K release, its labels, the 100K contract or the frozen pilot
(`../RIND3D_StudentResearch/`).

| file | content |
|---|---|
| `docs/A_RESEARCH_PLAN.md` | questions, protocol, order, risks; the Stage A audit (facet law) |
| `docs/B_RESULTS.md` | results and conclusions |
| `slab/` | data layer (splits, sets, facets), flat density, models, training, evaluation, analysis |
| `scripts/build_facets.py` | X-measurable interface facets for every row of the feature table |
| `scripts/validate_facets.py` | facet-law check |
| `configs/` | arm configurations |
| `tests/test_lab.py` | normalisation, plane factor, synthetic facet law |

**Environment.**

    uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt pytest

**Data.**
* Inputs: `$RIND3D_1K_DATA` (default `~/Projects/RIND/RIND3D_1K_data`): `release_1k_v1`, `natural_supplement_v1`, `pilot_features_v1`.
* Outputs: `$RIND3D_LAB_OUT` (default `$RIND3D_1K_DATA/studentlab_v1`): facet table, runs, evaluations. They are outside Git.

**Workflow.**

    .venv/bin/python scripts/build_facets.py                     # once, ~1-2 h on 12 cores
    .venv/bin/python -m slab.train configs/<arm>.json            # resumable; writes runs/<name>/final.pt
    .venv/bin/python -m slab.evaluate $RIND3D_LAB_OUT/runs/<name> # dev split; --split confirm --confirm only for declared runs
    .venv/bin/python -m slab.analyze A=<eval.npz> B=<eval.npz> --pairs A:B

**Splits.**

| split | scenes | role |
|---|---|---|
| train | pilot-train | training |
| dev | pilot-select | every exploratory comparison |
| confirm | 1K val | used once, for declared comparisons, logged |
| test | 1K test | never read |
