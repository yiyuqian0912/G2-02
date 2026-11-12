# RIND3D_PosteriorLabels: exact, error-controlled source-posterior labels (research prototype)

This is an isolated study directory. It contains no git repository and nothing in it is committed. The canonical
RIND3D v0.1 data and code are only read: they are imported from their checkouts with `PYTHONDONTWRITEBYTECODE=1`,
so nothing is written into those trees.

## Start here

| File | Content |
|---|---|
| `RIND3D_POSTERIOR_LABELING_REPORT.md` | The research report: target, method, verification, integration, scaling, recommendation (A–F) |
| `docs/METHOD.md` | Definitions, lemmas with proofs, algorithm and certificate |
| `docs/RESULTS_TABLES.md` | All result tables, generated from `outputs/*.json` by `code/make_tables.py` |
| `outputs/COST_MODEL.json` | Measured costs and declared extrapolations (`code/cost_model.py`) |

## Code

| Path | Role |
|---|---|
| `code/rpl/engine3.py` | **CEPL v3**, the main engine: cover stage, leaf stage with incremental clipping, exact polytope leaves, inclusion–exclusion |
| `code/rpl/engine2.py` | v2 (earlier version, kept for the record) and the shared certified predicates (shadow rows, block certificates) |
| `code/rpl/engine.py` | v1 (first attempt, kept for the record) |
| `code/rpl/polytope.py` | Convex polytope clipping and volume |
| `code/rpl/predicates.py` | Filtered float64 predicates, full Fourier–Motzkin shadow rows, float renderer |
| `code/rpl/topology.py` | Face adjacency, used to build silhouette-edge shadow rows |
| `code/rpl/bound.py` | Rigorous TV bound from per-cell volume intervals |
| `code/rpl/labels.py` | Side-car label format and conversion to the hierarchical 16³ × 8³ Student target |
| `code/rpl/data.py`, `code/rpl/rindenv.py` | Read-only access to v0.1 scenes, windows and splits |
| `code/rpl/synth.py`, `code/synth_cases.py`, `code/run_synth.py` | Controlled reference cases and the brute-force reference |
| `code/batch.py`, `code/analyze_windows.py` | Real-window batches (validation split) and their certified per-window analysis |
| `code/bank_vs_full.py` | Exact bank-observation posterior vs the full window |
| `code/overlap.py` | Scene-level sharing measurement |
| `code/check_render.py` | Float renderer vs stored v0.1 windows |
| `tests/` | Tests: polytope clipper vs qhull; shadow rows vs Fourier–Motzkin vs canonical exact oracle; TV bound vs exhaustive enumeration |

## Reproduce

The environment is `uv venv` with Python 3.12 and numpy, numba, scipy, pyarrow, python-flint, zstandard and pyyaml.
The canonical `rind3d` and `rind` packages are imported from `/home/yushu/Projects/RIND/RIND3D/src` and
`/home/yushu/Projects/RIND/dataset/src` (see `code/rpl/rindenv.py`).

```bash
export PYTHONDONTWRITEBYTECODE=1
.venv/bin/python tests/test_polytope.py; .venv/bin/python tests/test_shadow.py; .venv/bin/python tests/test_bound.py
.venv/bin/python code/check_render.py
RPL_ENGINE=rpl.engine3 RPL_KW='{"d_full": 7, "d_max": 12}' RPL_OUT=outputs/synth_T1_engine3.json .venv/bin/python code/run_synth.py T1_edge_slab
.venv/bin/python code/batch.py --scenes 12 --per-scene 2 --sides 64 --workers 12 \
   --kw '{"d_full": 10, "d_max": 13, "max_terms": 1024, "max_nodes": 150000}' --out outputs/batch_val64_v3
.venv/bin/python code/analyze_windows.py outputs/batch_val64_v3
.venv/bin/python code/bank_vs_full.py <scene_id> 82 256
.venv/bin/python code/overlap.py
.venv/bin/python code/cost_model.py outputs/batch_val64_v3 outputs/COST_MODEL.json
.venv/bin/python code/make_tables.py
```

The T2 and T3 results in `outputs/synth_results.json` were produced with engine v2 (`d_full=7`, `d_max=12`).

## Compute used

Everything ran locally on CPU, using at most 12 cores, with a declared node budget on every run. No GPU and no cloud
compute were used.
