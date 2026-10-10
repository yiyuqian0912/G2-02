# Multi-source interface reference

[English guide](../README.md) · [中文指南](../README.zh-CN.md)

## Implemented dataset interfaces

```python
from rind_phase1_multi.data import Phase1MultiDataset, make_dataloader
```

| Call | Result | Intended use |
|---|---|---|
| `Phase1MultiDataset(root=None, scene_ids=None)` | Lazy subset of complete scenes | Default data path or `RIND_MULTI_DATA_ROOT` |
| `ds[i]` / `get_observation(scene,view)` | IDs, response float32 `[L,L]`, obstacle bool `[L,L]`, window int64 `[3]` | Student receives response/local mask/window |
| `get_source_params(scene)` | float64 `[K,3]` `(x,y,strength)` | Teacher/evaluation truth |
| `get_channels(scene,view)` | float64 `[K,L,L]` | Exact separate generating contributions |
| `get_region(scene,x,y,width,height=None)` | float64 channels/response plus bool obstacle mask | Rectangle diagnostics; no student access outside its window |
| `get_scene(scene)` | Continuous obstacle geometry, sources, strengths and local-view metadata | Teacher/evaluation |
| `get_candidates(scene,view)` | List of float64 `[K_candidate,3]` witnesses | Renderer checks and overlays; not targets |
| `rerender(scene,window,sources)` | float64 `[L,L]` | Explicit nonempty `[K,3]` parameters, unit pixel-center sampling |
| `rerender(...,return_channels=True)` | `channels`, `response`, `obstacle` dictionary | Separate hypothetical contributions |
| `get_view_tree(scene)` | Nested split/view/occupied nodes | Tree reconstruction, teacher/diagnostics |
| `iter_view_leaves(scene,include_occupied=False)` | Iterator of view leaves | Read each leaf with its `view_id` |
| `make_dataloader(ds,...)` | Equal-L batches: response/mask `[B,L,L]`, window `[B,3]`, IDs `[B]` | Optional PyTorch; K labels not included |

Window metadata uses integer `(x,y,L)` in world coordinates. Source positions and strengths are float64; sources are not snapped to the sample grid. Tree nodes use `x`, `y`, `size`, `kind`, and view leaves also have `view_id`; split nodes have `children`.

## Implemented physical interface

```python
from rind_phase1_multi.physics import evaluate_sources
result = evaluate_sources(ds, scene_id, view_id, sources, backend="auto")
```

`sources` is an explicit nonempty `[K,3]` array. Each row must be finite, lie in world bounds/outside the window/outside obstacles, and have intensity in `[0,1]`. The forward interface permits diagnostic source counts above the generation cap. Source locations are never restricted to the window parent.

Outputs are `valid` (bool), `physical_cost` (float64 mean absolute intensity error or infinity), and `num_source_renders` (actual unique position-visibility renders in this fresh evaluator). Co-located sources reuse visibility, including diagnostic strength splits. The NumPy/Numba renderer is not differentiable.

Stored total response and rerendered loss targets are float64. Student observations are float32. Reference sets can contain different source counts and co-located sources; physical sanity uses `REFERENCE_ATOL=1e-12`.

## Joint search and teacher interfaces

| Call | Result / semantics |
|---|---|
| `JointResponseEvaluator(ds,s,v,backend="auto",max_source_renders=None)` | One float64 observation and teacher geometry; shared visibility/cost caches |
| `evaluator.positions_valid(xy)` | Boolean `[N]`, including world/window/continuous obstacles |
| `evaluator.evaluate(sources)` | `CandidateEvaluation(valid,physical_cost,num_source_renders,new_joint_evaluations)` |
| `CandidateDomain(window,world_size)` | Explicit world-outside-window domain, no quadtree prior |
| `uniform_source_grid(domain,evaluator,spacing=...)` | Stable y/x grid and separate geometric validity mask |
| `sample_joint_support(grid,source_counts=...,samples_per_count=...,count_prior=...,seed=...)` | Canonical, deduplicated padded complete sets with empirical prior mass |
| `make_joint_support(sets,base_mass=...,metadata=...)` | Caller-declared support; exact permutations deduplicated with mass aggregation |
| `evaluate_joint_support(support,evaluator,max_evaluations=None)` | Aligned costs/validity/evaluated mask, completion, actual work counts |
| `target_from_search(result,temperature=...)` | Stable `[N]` joint probabilities; incomplete/no-valid support fails |
| `generate_joint_teacher(ds,s,v,spacing=...,temperature=...,source_counts=...,samples_per_count=...,count_prior=...,seed=...)` | Full sampled teacher record; no reference/true-source access |
| `save_teacher_record(record,path)` / `load_teacher_record(path)` | Compressed NPZ with JSON provenance, no pickle |
| `inspect_teacher(ds,record,low_cost_threshold=None)` | Separate witness sanity checks, costs, ambiguity, count distribution, top sets |
| `source_space_projections(record)` | 2D presence, expected count/intensity, sampled minimum joint cost and coverage |

`SourceSetEvaluator` is a small protocol: `positions_valid([N,2]) -> bool[N]` and `evaluate([K,3]) -> CandidateEvaluation`. Synthetic evaluators can exercise search and normalization independently of dataset physics. Optional `is_cached` allows zero-new-evaluation reuse under budgets.

Record arrays are `sources[N,K_max,3]`, `source_counts[N]`, `base_mass[N]`, `multiplicity[N]`, `physical_cost[N]`, `valid[N]`, `evaluated[N]`, `teacher_prob[N]`, `window[3]`, `position_xy[P,2]`, `position_valid[P]`, `position_grid_index[P,2]` (row/column), and the two grid axes. Read only `sources[i,:source_counts[i]]`; inactive rows are zero padding. Array index i always identifies the same complete set, raw cost and probability. Metadata distinguishes completed samples from exhaustive joint coverage and records dataset, geometry/cost, sampler, temperature, code/config/run identity and render counts.

`base_mass` is required for custom support: declaring it explicitly avoids silently inventing a source-count/proposal measure. The sampler supplies `pi_K/N_K` and aggregates duplicate masses. `num_joint_evaluations` counts new complete-set cost calculations; `num_source_renders` counts new unit-visibility renders at unique positions. Diagnostic reference renders are counted separately. Neither count is the number of source rows in all proposals.

## Remaining research modules

`model.py`, `train.py`, and `evaluate.py` remain future work. Joint teacher generation and inspection are usable now; adaptive search and student interfaces need their own research decisions while keeping student conditioning local.

`configs/phase1.json` records current data semantics and leaves research settings unresolved. Source count and strength ranges describe generation, not an automatic inverse-domain restriction. `quadtree_prior=false` and `source_location_scope=outside_observation_window` describe the main task.
