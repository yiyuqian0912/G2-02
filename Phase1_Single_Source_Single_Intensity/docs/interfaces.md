# Phase I module contracts

[README](../README.md) · [Method](method.md)

The NumPy/checkpoint handoff identifier remains `phase1-v1` for backward compatibility. **It does not imply an identical experiment.** New experiments declare `protocol_version=2`; checkpoint `model_config` records input flags and normalization support. Missing flags in older checkpoints load as `use_obstacle=true`, `pixel_coordinates=false`, `normalization_support=geometry`. The new default configuration explicitly chooses false/true/world respectively.

## Observation: `Phase1Dataset.get_observation(scene_id, view_id)`

| Field | Shape/type | Role and use |
|---|---|---|
| `scene_id`, `view_id` | integers | Join observation, teacher and prediction; group scene splits. Never model features. |
| `response` | float32 `[size,size]` | Native unit-strength response; primary observation feature and physical cost reference. |
| `obstacle` | bool `[size,size]` | Local data validation and optional mask-input ablation; ignored by the new default model. |
| `window` | integer `[3] = [x,y,size]` | World origin/extent, candidate-domain exclusion, size batching and learned window encoding. |

`interfaces.validate_observation` requires all fields; response must be finite, aligned to size and zero inside obstacle pixels. Carrying a mask in the data record does not authorize using it as a feature. `SourceEnergyField.use_obstacle` determines that choice. Source truth, references and full geometry are accessible only through separate dataset methods for teacher/evaluation.

## Physical evaluator and candidate domain

`PhysicalEvaluator(dataset,scene_id,view_id,alpha,beta,boundary_lambda,boundary_sigma,backend)` binds one observation and hidden scene. `evaluate(candidate_xy)` returns `CandidateEvaluation(valid,physical_cost,num_physics_evaluations,cache_hit)`. World/obstacle invalidity yields `valid=false`, cost infinity and zero renders. Exact repeated coordinates reuse the cache. Boundary weights derive from the observed response only; nonnegative alpha/beta, positive sigma and intensity 1 are required.

`CandidateDomain(window,world_size,quadtree_prior=False)` owns window exclusion and optional parent constraint. `uniform_grid(domain,spacing)` returns world-coordinate candidates, row-major `grid_index`, axes and domain metadata. Sources on split lines belong right/down; the terminal world boundary is included. Image pixels use half-offset centers.

`uniform_search` tracks `evaluated`, `valid`, cost, render/cache counts and completeness. Unknown entries are not silently assigned zero probability. A budget-unfinished search cannot become a completed teacher. `part_e/adaptive.py` separately tracks explored indices and completed retained blocks on a common final lattice; its result is conditional on that retained support, without area weighting.

## Teacher: `generate_uniform_teacher` / `load_teacher_record`

| Field | Shape/type | Role and use |
|---|---|---|
| `scene_id`, `view_id`, `window` | as above | Exact join identity; mismatches reject training. |
| `candidate_xy` | float64 `[N,2]` | Ordered, unique world-coordinate source queries outside the window. |
| `valid` | bool `[N]` | Physical geometry validity, not the primary student's normalization mask. |
| `evaluated` | bool `[N]` | Every entry must be true for training; metadata must also declare complete. |
| `physical_cost` | float64 `[N]` | Finite nonnegative valid costs; infinity on invalid entries. |
| `teacher_prob` | float64 `[N]` | Equal-candidate Gibbs probability; invalid entries zero; sum one. |
| `temperature` | positive scalar | Converts teacher costs to relative target weights, not a second student temperature. |
| `grid_index`, `x_axis`, `y_axis` | arrays | Reconstruct sampled raster without spatial interpolation. |
| `search_level` | uint8 `[N]` | Zero in uniform records; not a model feature. |
| `config_id`, `run_id` | strings | Identify teacher settings and record provenance. |
| `metadata` | mapping | Data/code hashes, support, physics settings, timing, complete flag and render counts. |

On disk, metadata is JSON text in the NPZ; `load_teacher_record` restores the mapping. Record validation rejects `area_weight`, duplicated candidates, unfinished evaluation and q inconsistent with costs. Reference positions never augment support.

`interfaces.training_record(teacher,observation)` joins these two objects. `train.TeacherRecords` reads small IDs/window metadata initially and decodes an observation only when requested. Size metadata supports batching without eagerly loading all images.

## Training batch and model

| Field | Shape/type | Role and use |
|---|---|---|
| `response`, `obstacle` | `[B,size,size]` | Same native size in a batch; the mask is zeroed internally when disabled. |
| `window` | float32 `[B,3]` | World position and size; divided by coordinate scale internally. |
| `candidate_xy` | float32 `[B,Nmax,2]` | World candidates; only model code normalizes them. |
| `valid` | bool `[B,Nmax]` | Teacher physical validity; padded entries false. |
| `support` | bool `[B,Nmax]` | Actual student normalization mask; padded entries always false. In world mode, all real candidates are true. |
| `teacher_prob` | float32 `[B,Nmax]` | Fixed supervision; padded/invalid entries zero. |

The loss uses `support`, never blindly `valid`. Candidate chunks produce energies that are concatenated before one global softmax. The model's `encode_observation` can be reused for all candidates of a view, including dense inference. Added pixel coordinates are deterministic positions within the observation, not geometry or truth labels.

`model_config` saves `hidden`, `fourier_frequencies`, `coordinate_scale`, `use_obstacle`, `pixel_coordinates`, `normalization_support`. `load_model` constructs the declared architecture before loading weights. Altering flags requires a new trained model. Low-level tensor `model.predict` expects real, unpadded candidate sets in world mode; use the canonical record predictor for ordinary inference.

## Prediction: `predict(model,teacher,observation,chunk,normalization_support=None)`

Returns identity fields, ordered candidate coordinates, unchanged teacher `valid` and `config_id`, plus:

| Field | Role |
|---|---|
| `energy[N]` | Finite unnormalized scores at actual candidate positions. |
| `student_prob[N]` | One globally normalized distribution on declared support. |
| `normalization_support` | `world` (public world-minus-window candidates) or `geometry` (hidden-validity assisted). Defaults to the checkpoint setting. |
| `support[N]` | Explicit mask: all real candidate entries for world, exactly `valid` for geometry. |
| `interface_version` | Record/checkpoint handoff identifier. |

The candidate set/order and physical-validity fields remain identical to the teacher for KL/JS/L1. Distribution support may differ: teacher is zero on physical invalids; primary student can place erroneous mass there. Old predictions without a support declaration retain geometry semantics. `part_e.checks.prediction_support` enforces the declared policy.

Formal metrics live in `part_e/evaluate.py`. `student_invalid_mass` measures mass on physical invalids. `student_expected_valid_cost` conditions on valid positions; `student_expected_physical_cost` is null with an infinity flag when invalid mass is positive. `student_expected_penalized_cost` instead uses the explicit finite penalty `alpha+2beta`. Do not compare these as interchangeable metrics.

## Protocol, recovery and result files

`experiments.plan` freezes config/data/source hashes, stable scene lists and observation IDs in `protocol.json`. The split manifest has its own digest. Existing output roots reject changes in these values. `prepare` saves teacher records and a hashed manifest. `train` selects on validation only and saves optimizer, RNG, counters and model state in `last.pt`; `best.pt` holds the selected model. Resume restores the last completed epoch, not an interrupted mid-epoch batch. Formal test evaluation requires completed training/early stopping and matching protocol/target signatures.

`outputs/experiments/<name>/<variant>/seed_<seed>/test_metrics.json` contains per-observation metrics, scene-balanced aggregates, size/response groups, uniform baseline and separately labelled geometry-assisted results. NPZ predictions preserve exact candidates and probability support. No source truth is used to select the checkpoint.

## Dense queries and results browser

`sampling.score_grid` queries actual centers at a requested integer spacing dividing the world; it returns energy raster and flat world coordinates. `probability_from_energy` normalizes only once on the requested mask. It does not correct for area. `sampling_report` explains the shortest Fourier period relative to training spacing.

`results_browser` discovers only experiment directories with `protocol.json`, variant `records.json` and seed `best.pt`. The selector is built from the saved test list and verified against test scene IDs; it does not silently display arbitrary release windows as test data. Jobs run one at a time, and recent results remain in bounded memory. Runs and record paths are validated. The server listens only on localhost.

The dense display contains model energies and probabilities at new query points. Teacher samples remain exactly those saved during the experiment. Matched mode reuses the saved teacher candidates, globally renormalizes student scores on those candidates and shares the probability color range. Metrics always use this same-candidate comparison, even when the displayed student raster is finer.

A downloaded NPZ contains `probability`, `energy`, `support`, `response`, `window`, `source_xy` (display/evaluation only), `teacher_candidate_xy`, `teacher_probability`, `teacher_cost`, `teacher_valid`, `student_on_teacher`, and JSON `metadata`. Probabilities are float64 and sum to one over their own support. Browser payloads also carry float64 values; PNG is a colored view, not a numeric substitute. Server inference does not modify original teacher/checkpoint/report files.
