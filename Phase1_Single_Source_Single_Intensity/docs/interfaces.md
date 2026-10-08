# Phase I Data and Module Contracts

[English README](../README.md) · [中文 README](../README.zh-CN.md) · [Method](method.md) · [Teacher usage](teacher-baseline.md)

Data access, physical response costs, uniform search, teacher records, and teacher diagnostics are implemented. Student/training/evaluation contracts below describe the intended handoff to later modules.

## 1. Observation — `data.py`

`Phase1Dataset.get_observation(scene_id,view_id)` and `ds[i]` return:

| Field | Shape / type | Meaning |
|---|---|---|
| `scene_id`, `view_id` | Python integers | Stable record IDs; not student features |
| `response` | `float32 [L,L]` | Binary local response |
| `obstacle` | `bool [L,L]` | Local mask; true means obstacle |
| `window` | `int64 [3]` | World-coordinate `(x,y,L)` |

`make_dataloader` groups equal-size windows; PyTorch batches include response `[B,L,L]`, boolean obstacle `[B,L,L]`, window `[B,3]`, and IDs. No resizing/padding is required. Source truth, references, region rendering, and continuous geometry are separate methods, teacher/evaluation-only. A scene's observations and targets inherit its partition; formal scene splits remain unresolved.

Array indexing is `[row,col]`; coordinates point right/down and pixels sample world centers. The local mask is an allowed student feature; masks outside that window and full geometry are not.

## 2. Physical evaluation — `physics.py`

Bind `PhysicalEvaluator(dataset,scene_id,view_id,backend="auto")`, then call `.evaluate(xy)` for one finite world-coordinate `[2]` candidate. It returns a `CandidateEvaluation`:

| Field | Type | Meaning |
|---|---|---|
| `valid` | Boolean | Inside world and accepted by continuous scene geometry |
| `physical_cost` | Float | Mean absolute response disagreement; infinity if invalid |
| `num_physics_evaluations` | 0 or 1 | New physical renders for this query |
| `cache_hit` | Boolean | Exact coordinate previously evaluated in this scene/window |

Physics does not decide parent/window exclusions. The search domain owns those restrictions. The physical cost is `mismatched_pixels/L²` for the binary unit-intensity release. One evaluator supports all six stored sizes at original resolution; no fixed response width or height is assumed. It uses all sampled pixels equally, without boundary weighting or an edge term.

`CandidateEvaluator` is a small protocol: `.evaluate(xy) -> CandidateEvaluation`. Synthetic evaluators can test search/teacher without a renderer. `response_disagreement(rendered,observed)` exposes the same scalar cost. `check_reference_solutions` independently checks all ten references and returns their coordinates/costs and the separate diagnostic render count.

## 3. Domain and search — `search.py`

`CandidateDomain(window,world_size,quadtree_prior=False)` defaults to world-minus-window: the task studies sources outside the observation window. Set the prior to true only for the optional parent-minus-window ablation. It validates alignment and exposes bounds, membership, and metadata. Split-line ownership is right/down; terminal world edges are included.

`uniform_grid(domain,spacing)` returns `UniformGrid`: candidate coordinates, ascending x/y axes, flattened raster `grid_index`, spacing, and domain. The global half-spacing lattice is deterministic; order is ascending y then x. `grid.as_map(values)` restores a source-space raster with excluded-window entries as NaN. `deduplicate_candidates` removes exact duplicates in first-occurrence order.

`uniform_search(domain,evaluator,spacing=...,max_evaluations=None)` returns:

| Field | Shape / type | Meaning |
|---|---|---|
| `grid` | `UniformGrid` | Declared support and ordering |
| `valid` | `bool [N]` | Physical geometric validity for evaluated candidates |
| `evaluated` | `bool [N]` | Candidate visited; false for budget-unfinished entries |
| `physical_cost` | `float64 [N]` | Finite valid cost; infinity if invalid; NaN if unfinished |
| `complete` | Boolean | Every domain candidate evaluated |
| `num_physics_evaluations` | Integer | Actual new renders; cache hits and invalid queries count zero |
| `num_cache_hits` | Integer | Reused evaluations |
| `elapsed_seconds` | Float | Search/evaluation elapsed time |

The optional budget stops visits conservatively once the new-render limit is reached. Incomplete results cannot become teacher targets. This guard and cache prepare later adaptive work; no adaptive pruning is implemented.

## 4. Teacher records — `teacher.py`

`generate_uniform_teacher(ds,scene_id,view_id,spacing=...,temperature=...,quadtree_prior=False)` requires explicit spacing and temperature. It never reads reference positions or uses true-source labels to construct support or probabilities.

Core candidate-aligned fields are:

```text
candidate_xy       float64 [N,2]
valid, evaluated   bool [N]
physical_cost      float64 [N]
teacher_prob       float64 [N]
search_level       uint8 [N] (zero for this uniform baseline)
```

The record also holds `scene_id`, `view_id`, `window`, `temperature`, `run_id`, `config_id`, `grid_index`, x/y axes, and `metadata`. Metadata preserves dataset manifest/archive identity, code/reader hashes, actual domain and prior, spacing/anchor/order, cost settings, tau, search mode, completeness, rendering count, cache hits, and timing.

`target_from_search(result,temperature=...)` rejects incomplete/nonuniform support. `teacher_probabilities(cost,valid,temperature=...)` normalizes stably over valid entries in the original order. Invalid probabilities are zero; no valid candidate is an explicit failure. No area weighting, reference augmentation, or true-source one-hot target is used.

`save_teacher_record(record,path)` writes compressed NPZ with numeric/string arrays and `metadata_json`. `load_teacher_record(path)` uses `allow_pickle=False`, restores JSON metadata, and converts scalar arrays to Python values. Training can reload these records without rerendering. Keep raw costs available for absolute compatibility diagnostics.

## 5. Diagnostic reports — `diagnostics.py`

`inspect_teacher(ds,record,low_cost_threshold=None)` independently rerenders all ten references and requires exactly zero costs. It reports sampled zero/low-cost fractions, four-neighbor components and approximate areas, entropy/effective candidate count, witness/domain coverage, and nearest-grid witness costs. Witnesses remain outside teacher normalization.

`grid_from_record(record)` restores map indexing. `plot_teacher(ds,record,report,path)` writes a static six-panel PNG; install the optional `diagnostics` extra. `representative_view_ids(ds,scene_id)` selects the first stored observation of each size, small to large.

Components and area estimates describe the sampled lattice, not continuous topology. Report spacing with every ambiguity comparison and use common spacing for quantitative comparisons. Grid and diagnostic witness renders have separate counts.

## 6. Student / training / evaluation handoff (not implemented)

The intended energy inputs are `response`, local `obstacle`, `window`, and queried `candidate_xy`. `valid` is used after scoring for probability normalization and is not an encoder feature. IDs, truth, reference positions, full geometry, outside-window masks, physical costs, and the complete quadtree are excluded from student features.

`train.py` should match observations and teacher records by IDs, preserve candidate ordering, and compare distributions on the same valid support. Save actual architecture/optimizer settings and checkpoints; choose models using validation scenes. Padding, if introduced later, needs a separate valid-pixel mask.

`evaluate.py` should use matching candidate support, record window size/spacing/domain, and compare adaptive results against a common uniform reference grid. Teacher-supported geometric validity uses hidden scene information and must be distinguished from geometry-free deployment on a declared public candidate set.

## 7. Configuration and runnable checks

`configs/phase1.json` explicitly disables the parent prior and selects the world-minus-window response-only uniform baseline (`alpha=1,beta=0,boundary_lambda=0`). Spacing, temperature, scene split, adaptive settings, and training hyperparameters remain unresolved `null` fields. CLI spacing/tau/prior overrides are reflected in saved target settings. A richer physical-cost/adaptive configuration is rejected by the current CLI.

```bash
uv run --no-sync python -m unittest discover -s tests -v
NUMBA_NUM_THREADS=2 uv run --no-sync python -m rind_phase1.checks --data-only
NUMBA_NUM_THREADS=2 uv run --no-sync python -m rind_phase1.checks --teacher
```

Data checks include the generation prior for every installed view, local masks, subsets/batching/tree, reference rerendering, and serialization. Teacher checks cover six window sizes, sixty exact witnesses, coordinate/cost alignment, prior/world domains, geometric rejection, physical caching/counts, normalization, and NPZ round trips. They do not claim student training or adaptive-search completion.
