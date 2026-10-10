# Uniform physical teacher: use and interpretation

[English README](../README.md) · [中文 README](../README.zh-CN.md) · [Contracts](interfaces.md)

The current runnable chain is **observation → uniform candidate positions → physical costs → teacher probabilities**. It preserves multiple compatible positions. This document describes the low-level unweighted teacher baseline. Formal experiments now implement student training, boundary weighting and optional edge costs; adaptive retention remains an independent comparison. See [current method](method.md) and [training instructions](experiments.zh-CN.md).

## What the teacher represents

An observation has `response [L,L] float32`, `obstacle [L,L] bool`, and `window [x,y,L] int64`. The obstacle mask contains only the observed window. `scene_id` and `view_id` identify records. Source truth, reference solutions, and geometry outside the window are teacher/diagnostic data.

The task studies **sources outside the observation window**. The default domain is `world \ W`: quadtree provides dataset windows and does not restrict the main teacher to the parent. Candidates need only satisfy the declared spatial domain and continuous geometric validity; they need not regenerate the same quadtree leaf.

`teacher.quadtree_prior=false` is the default in configuration and Python APIs. Explicit `quadtree_prior=True` or CLI `--prior parent` enables an optional generation-prior ablation:

```text
parent_origin = (floor(x / (2L)) * 2L, floor(y / (2L)) * 2L)
optional prior domain = parent cell of side 2L minus the observed cell of side L
```

Split-line ownership is right/down; the outer world edges belong to terminal cells. Optional parent construction validates quadtree alignment. Metadata records `source_location_scope="outside_observation_window"`, the actual domain, and whether the parent prior was enabled.

Candidates are world-coordinate lattice points `((kx+0.5)*spacing, (ky+0.5)*spacing)`, globally anchored at `(0,0)`, ordered by ascending y then ascending x. Domain exclusion happens before geometry evaluation. This common anchor makes the prior support a subset of the world support at the same spacing. Response pixels continue to sample world positions at one-unit pixel centers; candidate spacing changes source-space sampling only.

The response-only physical cost is:

```text
C(s) = mean(abs(rerendered_response(s) - observed_response))
     = number of disagreeing sampled pixels / L²   (binary unit-source data)
```

All window pixels have equal weight, including obstacle pixels, which are zero in both responses. Costs range from 0 to 1. A zero cost means equality on the sampled observation, not equality everywhere in the continuous scene.

On the completed valid support, `q_i ∝ exp(-C_i/tau)`. The implementation subtracts the minimum valid cost before dividing by positive `tau`. Invalid probabilities are zero; no valid candidates or incomplete evaluation is an explicit failure. Raw costs remain available. No area factors or true-source one-hot labels are used.

## Generate and inspect one observation

The following spacing, temperature, and low-cost threshold are **illustrative settings**, not an agreed research protocol. The shared config leaves spacing and temperature `null` until supplied explicitly.

```bash
# Install optional plotting dependencies; preserve any existing extras.
uv sync --locked --inexact --extra diagnostics

# Scene 0, view 4 is a 16x16 window. Default: world outside the window.
NUMBA_NUM_THREADS=2 uv run --no-sync rind-teacher \
  --scene-id 0 --view-id 4 --spacing 64 --temperature 0.05 \
  --low-cost-threshold 0.01 --plot \
  --output outputs/teachers/example-world.npz

# Optional parent-prior ablation; finer here because the parent is small.
# Use the SAME spacing in a formal prior-on/off comparison.
NUMBA_NUM_THREADS=2 uv run --no-sync rind-teacher \
  --scene-id 0 --view-id 4 --prior parent --spacing 2 --temperature 0.05 \
  --plot --output outputs/teachers/example-parent.npz
```

`NUMBA_NUM_THREADS` is optional; rendering thread count affects runtime, not the candidate grid. `uv run --extra diagnostics rind-teacher ...` is an alternative that syncs dependencies automatically. `python -m rind_phase1.teacher` invokes the same CLI. The root follows `Phase1Dataset`, `RIND_DATA_ROOT`, or `--data-root`; the config supplies the project-relative data path otherwise.

Each run writes:

- `.npz`: teacher candidates, validity, evaluated flags, raw costs, probabilities, window, and grid coordinates/indexing;
- `.diagnostics.json`: ambiguity metrics and independent witness checks;
- `.png` when `--plot` is requested: local response/mask, physical-cost map, zero-cost map, teacher mass, and a witness zoom.

For a requested `--max-evaluations` budget, unfinished search raises an error and produces no final teacher. To inspect an unfinished result in Python, call `uniform_search` directly; unevaluated costs are `NaN`, evaluated invalid costs are infinity, and `complete=False`. This is a completion guard, not an adaptive pruning algorithm.

## Read records or use the pipeline in Python

```python
from rind_phase1.data import Phase1Dataset
from rind_phase1.teacher import (
    generate_uniform_teacher, load_teacher_record, save_teacher_record,
)
from rind_phase1.diagnostics import grid_from_record, inspect_teacher

ds = Phase1Dataset()
sample = ds.get_observation(0, 4)
record = generate_uniform_teacher(
    ds, 0, 4, spacing=64, temperature=0.05,  # default: world minus window
)
save_teacher_record(record, "outputs/teachers/example.npz")
record = load_teacher_record("outputs/teachers/example.npz")

xy = record["candidate_xy"]          # float64 [N,2], stable candidate order
cost = record["physical_cost"]       # float64 [N], infinity for invalid positions
valid = record["valid"]              # bool [N]
q = record["teacher_prob"]           # float64 [N], invalid entries exactly zero
assert abs(q.sum() - 1) < 1e-12
cost_map = grid_from_record(record).as_map(cost)
report = inspect_teacher(ds, record)  # independently checks all ten witnesses
```

NPZ records use numeric and string arrays, so ordinary `np.load(path, allow_pickle=False)` also works. `metadata_json` holds JSON. `scene_id`, `view_id`, `temperature`, `config_id`, and `run_id` are scalar arrays on disk; the helper returns their ordinary Python values.

Metadata preserves dataset manifest/archive identity, reader and pipeline source hashes, support/ordering/spacing, renderer/cost settings, temperature, completeness, new render count, cache hits, elapsed search time, and run/config IDs. The saved settings describe the actual CLI overrides; `config_id` hashes those settings. Uniform render counts exclude the ten additional diagnostic witness renders, which have a separate count.

`PhysicalEvaluator` caches costs by exact coordinates within one scene/window. `CandidateEvaluator.evaluate(xy) -> CandidateEvaluation` can also be supplied by a synthetic evaluator. Search owns spatial domain selection; physics owns continuous geometric validity and response comparison; teacher owns final normalization/storage.

## Understand ambiguity diagnostics

Diagnostics include exact-match counts, zero/low-cost fractions, four-neighbor zero-cell components and approximate areas/bounding boxes, entropy in nats, normalized entropy, effective candidate count, and reference overlays.

- `zero_fraction_of_domain_grid` divides by all spatial-domain lattice points, including geometrically invalid points. `zero_fraction_of_valid_grid` divides by valid points only.
- A component area is `number_of_sampled_zero_cells * spacing²`. Connectivity and area describe the sampled lattice; they do not recover exact continuous topology or area.
- `low_cost_threshold` is optional and explicit. It does not change the teacher costs or probabilities.
- All ten supplied reference positions must independently rerender with **exactly zero** cost. They are witnesses, never grid additions or teacher labels.
- `reference_domain_coverage_fraction` checks whether the declared domain retains those witnesses. `reference_nearest_grid_zero_fraction` checks their nearest sampled grid cells; it is a spacing-dependent diagnostic, not exhaustive inverse-set recall.
- A coarse grid may have no zero-cost points while all ten off-grid witnesses have zero cost. Refine the declared spacing before drawing conclusions about inverse geometry. Do not insert witnesses into softmax to repair this sampling issue.

The same teacher supports all six stored sizes: 16, 32, 64, 128, 256, and 512. Each candidate is rendered at the original `L×L` resolution; no resizing or fixed-size teacher is needed. Costs divide by `L²`, so their unit remains a disagreement fraction across sizes. Larger windows require more ray/pixel computation.

Previously generated parent-prior records retain their own support metadata and are still valid ablation records. Generate new world-support targets for the main experiment; loading an old record does not reinterpret its support.

To explore window sizes, `representative_view_ids(ds, scene_id)` returns the first stored view of each size, ordered small to large. Generate records for those IDs, using a declared common world spacing for quantitative comparisons. `inspect_teacher` exposes size/spacing together with ambiguity metrics. A quick smoke experiment may use `spacing=L/4` to bound runtime, but must label its changing spacing rather than attributing every difference to window size.

## Validation and next decisions

```bash
uv run --no-sync python -m unittest discover -s tests -v
NUMBA_NUM_THREADS=2 uv run --no-sync python -m rind_phase1.checks --data-only
NUMBA_NUM_THREADS=2 uv run --no-sync python -m rind_phase1.checks --teacher
```

The teacher check uses explicitly reported smoke settings (`spacing=world_size/8`, or 128 in this dataset; `tau=.05`) on the default world domain across six sizes and checks all 60 witnesses. It also checks the optional parent-prior ablation, geometry rejection, caching/counts, physical-cost alignment, and portable record loading. Unit tests cover domain boundaries, deterministic ordering, deduplication, synthetic evaluator alignment, completion failures, stable softmax, empty support, and label-free target generation.

Still unresolved: the formal scene split, candidate spacing, temperature, near-zero analysis threshold, boundary/edge conventions, and adaptive refinement policy/budget. The model and training implementations are untouched. Later adaptive work must separate exploratory trace points from a completed final support at a common spacing, preserve multiple regions, deduplicate/cache evaluations, and compare against this uniform reference.
