# Phase I Data and Module Contracts

[Project responsibilities](../README.md) · [中文 README](../README.zh-CN.md) · [Method](method.md)

These are proposed handoff contracts for the current flat module layout. Implementations are pending; field names match the current README.

## 1. Observation — `data.py`

| Field | Shape / type | Meaning |
|---|---|---|
| `scene_id`, `view_id` | Stable identifiers | Trace the sample; never student features |
| `response` | `[size, size]` float | Single-source local response |
| `window` | `[3]` | World-coordinate `(x, y, size)` |

Source coordinates, intensity, and scene geometry may be retrieved separately for teacher supervision and evaluation. They are not student features. World coordinates span [0,1024]; arrays use [row,col]. Preserve original window extent when batching different sizes. If padding is used, distinguish valid image pixels from padded entries.

Split manifests in `data/splits/` contain disjoint scene-ID lists. Every derived sample and target inherits its scene's partition.

## 2. Physical evaluation — `physics.py`

Input: scene geometry, observed response, window, candidate coordinates, fixed intensity, and cost settings.

| Output | Shape / type | Meaning |
|---|---|---|
| `candidate_xy` | `[2]` per query | World-coordinate source candidate |
| `candidate_response` | `[size,size]`, or absent for invalid candidates | Rerendered response |
| `L_resp` | Scalar | Boundary-weighted response disagreement |
| `L_edge` | Scalar or absent when disabled | Symmetric boundary disagreement |
| `physical_cost` | Scalar | Consistently combined candidate cost |
| `valid` | Boolean | Candidate satisfies the declared domain restrictions |

Reject invalid candidates before rendering. Their costs do not enter probability normalization. Record whether edge costs were computed; disabled and measured-zero edge costs are different cases. Geometry access is confined to the teacher/evaluation pathway.

## 3. Candidate set — `search.py`

| Field | Shape | Meaning |
|---|---|---|
| `candidate_xy` | `[N,2]` | Candidate world coordinates |
| `cell_bounds` | `[N,4]` | World-coordinate `[x_min,y_min,x_max,y_max]` for each represented cell |
| `valid` | `[N]` boolean | Validity under the recorded support policy |
| `area_weight` | `[N]` | Area represented by each candidate cell |
| `physical_cost` | `[N]` | Final costs, available after physical evaluation |
| `search_level` | `[N]` | Refinement depth or level |
| `num_physics_evaluations` | Scalar | Actual physical evaluation count |

Final cell areas must not double-count parent/child coverage. `area_weight` represents valid-domain area or a documented approximation; handle cells crossing window/obstacle boundaries rather than assuming center validity implies full-cell validity. Store the area estimation convention in configuration metadata. Valid candidates have positive areas. All fields share candidate order. Partial search results must be marked incomplete and cannot be passed to target construction as final costs. Retain the configured domain, coverage, and cost definition with search metadata.

## 4. Teacher record — `teacher.py`

Required fields (`temperature` is a scalar; `config_id` is a record identifier, not a candidate array):

```text
scene_id, view_id
candidate_xy       [N,2]
cell_bounds        [N,4]
valid              [N]
area_weight        [N]
physical_cost      [N]
teacher_prob       [N]
temperature        scalar > 0
config_id          reference to saved experiment settings
```

Save records under `outputs/teachers/`. Retain renderer/generation version and support policy in the referenced settings. Targets sum to one over valid candidates; invalid probabilities are zero. No valid candidates means explicit failure. Uniform candidate cells may use equal positive weights consistently.

The same coordinates, ordering, validity, and weights must be used for student normalization. Stored physical costs remain available for absolute compatibility diagnostics.

## 5. Student — `model.py`

Energy inputs: `response`, `window`, and `candidate_xy`.

Probability-normalization inputs: `energy`, `valid`, and `area_weight`. The last two are not energy-encoder features.

```text
candidate_xy       [N,2]
energy             [N]
probability        [N]
```

Candidate permutation must only reorder the associated scores. Different window sizes must be supported. Keep model/configuration identifiers in run records. A hidden-geometry-derived validity mask is used during teacher-supported training and explicitly labeled teacher-supported evaluation; it is not available automatically at deployment. Geometry-free prediction must use a predeclared public candidate set, not a set selected through teacher geometry or physical costs. Scores outside training support are not guaranteed to be suppressed.

## 6. Training, evaluation, and checks

`train.py` matches observations to teacher records by identifiers and candidate ordering. Save model weights, optimizer state, histories, and configuration in `outputs/checkpoints/`. Use validation scenes for model selection.

`evaluate.py` records per-observation and aggregate metrics, window size, support policy, and timing. Produce observation / physical-cost / teacher / student figures in `outputs/figures/` and metrics and summaries in `outputs/reports/`. Compare search strategies on common evaluation support rather than comparing unequal probability vectors directly.

`checks.py` verifies scene separation, source-free windows, generating-source reconstruction, consistent candidate arrays, probability normalization, variable-size model inputs, and the small end-to-end handoff. Report pass/fail and identify the failing stage. Clearly label untrained outputs and incomplete stages.

## 7. Input routing and use

The Chinese README includes a field-by-field input guide for all eight modules. The following rules specify how those inputs are used together.

| Module | Inputs and their use |
|---|---|
| `data.py` | RIND location supplies the generator/reader; scene count controls dataset size; fixed intensity defines the shared response scale; seed controls reproducibility; quadtree settings define observations; split settings assign whole scenes to partitions. |
| `physics.py` | Geometry determines occlusion; observed response supplies the comparison target and observed-boundary weights; window fixes the render region; candidate coordinates reposition the source; fixed intensity must match data generation; cost settings define response/edge evaluation. |
| `search.py` | World bounds and the window define public spatial restrictions; `candidate_spacing` is the initial grid step in world units; `adaptive_budget` limits physical candidate evaluations; the evaluator is bound to the current scene, response, intensity, and cost settings; geometry may be passed directly or encapsulated in that teacher-only evaluator. |
| `teacher.py` | Coordinates and cell bounds identify hypotheses and coverage; final costs establish compatibility; validity excludes candidates; area weights convert density-like scores to cell masses; positive temperature controls sharpness; sample IDs match targets to observations; `config_id` resolves to the settings and provenance used to generate targets. |
| `model.py` | Response conveys local structure; window supplies world location and scale; candidate coordinates specify queries. Normalize window and candidate coordinates using the same `coordinate_scale` before Fourier encoding. `valid` and `area_weight` are used only after energy scoring to normalize probabilities. Initialization reads Fourier frequencies and recorded architecture settings. |
| `train.py` | Observations supply model conditioning and record IDs; teacher candidates determine queries; `teacher_prob` supplies the soft target; shared validity and area ensure consistent normalization; the model supplies trainable parameters; training settings govern resources, updates, and validation-based checkpoint selection. |
| `evaluate.py` | Test observations supply held-out inputs; teacher records supply reference costs and distributions on matching candidates; a fixed trained checkpoint supplies predictions; evaluation configuration defines support, metrics, comparisons, and provenance. |
| `checks.py` | A few real scenes provide verifiable physical examples; the actual Phase I configuration keeps all module settings consistent and identifies incomplete required settings. |

Configuration notes:

- `candidate_spacing` replaces the ambiguous former name `candidate_resolution`; it is an initial step in world-coordinate units, not an image resolution or candidate count.
- `adaptive_budget` counts physical candidate evaluations, not batch API calls. Report reevaluations and optional edge-computation cost separately; record actual elapsed time as well.
- `learning_rate` and `checkpoint_selection_metric` remain unresolved until training is configured. Save the optimizer and architecture settings actually used along with each checkpoint.
- `teacher_prob` already includes area weighting. Do not multiply it by `area_weight` again in the cross-entropy target.
- If teacher records use different adaptive candidates, obtain reference teacher evaluations on the common evaluation grid before comparing distributions. Do not directly compare vectors with different coordinate meanings.
- Zero padding is not an observed zero response. If batching uses padding, carry a pixel-valid mask and exclude padding from response feature aggregation and any pixel-domain loss.
