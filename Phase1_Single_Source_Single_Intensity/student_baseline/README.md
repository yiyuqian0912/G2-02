> Legacy pilot implementation (`legacy-relative-v0`). New project work uses `src/rind_phase1/{interfaces,model,train,predict}.py` and the [canonical contract](../docs/interfaces.md). Legacy input names and normalization below apply only to old checkpoints.

# G2-02 Phase I student baseline

Independent student-model project implementing Thomas’s confirmed assignment: Section 16.2, “Model and optimization,” of `G2-02_Conditional_Energy_Source_Field_Model.pdf`, with the detailed contract in `CODEX_G2-02_Phase1_Student_Model_Instructions.md`. Dataset construction, teacher physics/search, analytic diagnostic scenes, and profiling belong to other workstreams. Real teacher records are not supplied in this workspace.

## Integration status

This directory is Thomas’s standalone student baseline built from the supplied Section 16.2 model specification. It is independently verified with mocks and is **not an implementation of the current shared `rind_phase1` interface**. The parent project’s data reader, placeholders, configuration, dependencies and lockfile remain separate.

| Contract | This baseline | Current shared Phase I interface |
|---|---|---|
| Observation sizes | One fixed response size across a dataset | Variable-size quadtree windows |
| Model conditioning | Response and candidate coordinates only | Response, window position/size, and candidate coordinates |
| Coordinate frame | `(candidate_xy - window_origin) / window_side` | Window and candidate world coordinates scaled by `coordinate_scale` (currently 1024) |
| Window metadata | Preprocessing only; never an encoder feature | Explicit model conditioning |
| Teacher normalization | Optional represented-area weights for cost-derived targets; supplied probabilities used directly | Equal weight per retained candidate; no area factors |
| Validity field | `valid_mask` | `valid` |

See the [shared interface contract](../docs/interfaces.md). Connecting this baseline to that pipeline requires an explicit scientific/interface decision and an adapter; it must not be treated as a drop-in replacement. Real teacher integration and physical validation remain pending. The reference PDF was supplied during development and is not bundled here; the detailed [student instructions](CODEX_G2-02_Phase1_Student_Model_Instructions.md) and independent reference-parity tests are included.

Run all commands below **from this `student_baseline/` directory**, using its own environment. Do not replace the parent project’s `pyproject.toml`, `uv.lock`, or dataset module with these files. Dependencies are local to this baseline. Generated datasets, checkpoints, plots and logs are intentionally excluded from Git; the [handoff report](IMPLEMENTATION_REPORT.md) contains recorded results and artifact-generation commands.

## Run

Create an isolated environment in this directory:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python verify_pipeline.py --output outputs/verification
.venv/bin/python train.py --output outputs/train
```

`requirements-lock.txt` records the exact installed environment (Python 3.14 on macOS ARM64). `requirements.txt` provides broader version bounds for other environments. CPU with one thread is the tested default. GPU devices may be selected in configuration but have not been verified.

Use `--config config.json` to override any `TrainConfig` fields. For example:

```json
{"epochs": 150, "batch_size": 4, "learning_rate": 0.001, "random_seed": 0}
```

Resume with `--resume outputs/train/last.pt --output outputs/train --config config.json`. Set `epochs` to a total greater than the saved epoch. Resume from the latest checkpoint into its existing directory to retain history and the best checkpoint. Fresh runs require an empty output directory. Without a config override, the saved configuration is reused. Only total epochs, device, thread count, and validation/checkpoint intervals may change on resume. All other differences are rejected. Checkpoints restore optimizer, optional scheduler, progress, best validation loss, configuration and random states. New run checkpoints also contain fingerprints of every split; changed records or teacher targets are rejected on resume. Legacy checkpoints without fingerprints remain loadable, but their data identity cannot be verified. Checkpoint and record files must be trusted: the loader supports Python/NumPy checkpoint state using unrestricted PyTorch deserialization.

## Exact teacher record schema

Each record is a Python dictionary. Tensor-like values are converted to float32 during preprocessing except `valid_mask`, which must be boolean.

| Field | Shape/type | Contract |
|---|---|---|
| `response` | `[1,H,W]` | Finite response image; fixed size across the dataset |
| `candidate_xy` | `[K,2]` | Finite source coordinates, `K > 0`; ordering is preserved |
| `coordinate_space` | `"world"` or `"normalized"` | Defaults to world; normalized coordinates must be explicitly marked |
| `window` | `{x0: float, y0: float, L: float}` | Required for world coordinates; finite values and `L > 0` |
| `teacher_prob` | `[K]`, optional | Finite, nonnegative, sums to one within `atol=rtol=1e-5`; zero on invalid candidates |
| `physical_cost` | `[K]`, optional | Required if probabilities are absent; finite on valid candidates |
| `area_weight` | `[K]`, optional | Finite and strictly positive on valid candidates; omitted means equal area |
| `valid_mask` | bool `[K]`, optional | Defaults to all valid; at least one valid candidate |
| `scene_id` | string or integer | Required; cannot occur in more than one split |
| `view_id` | string or integer | Required for traceability |

Preprocessing computes `(candidate_xy - [x0,y0]) / L` for world coordinates and marks the result normalized. It never clips coordinates to the observation window. The dataset owner is responsible for source-free windows and the other upstream physical assumptions.

`collate_records(records, teacher_temperature=0.1)` pads candidates to the largest K in a batch, emits `[B,1,H,W]` responses and `[B,K,2]` coordinates, constructs a `[B,K]` teacher distribution per record, and preserves scene/view IDs. Mixed cost-only and probability-supplied records are supported. Padding gets a false mask and zero probability. Unavailable physical costs are NaN placeholders for diagnostics only. They are not model inputs or used to reconstruct an already supplied target.

For external data, save a trusted torch dictionary containing nonempty `train`, `validation`, and `test` record lists and pass `--records records.pt`. The full dataset is validated before optimization. Set explicit non-mock values for `dataset_version`, `scene_split_version`, `candidate_search_config`, `teacher_cost_config`, and `area_weight_convention` in the config. This in-memory format is the baseline integration adapter; large-scale streaming is not implemented.

## Architecture and objective

`SourceEnergyField(hidden=128, fourier_frequencies=6)` implements the prescribed baseline: three CNN layers retain spatial tokens; Fourier-encoded source coordinates produce queries; single-head cross-attention retrieves image context; an MLP maps concatenated query/context to scalar energy. There is no spatial pooling before attention, explicit edge extraction, geometry input, or extra positional encoding.

```python
energy = model(response, candidate_xy)  # [B,K]
outputs = model.predict(response, candidate_xy, valid_mask)
probability = outputs['student_probability']
```

`predict` does not itself disable gradients or switch to eval mode. `student_distribution` returns `(log_probability, probability)`. Each candidate's energy is independent of other candidate queries; its probability depends on the complete candidate set.

For valid candidates, cost-derived teacher mass is `q = softmax(log(area_weight) - physical_cost / temperature)`. Equal weights are used if area is omitted. A supplied `teacher_prob` is validated and used directly, without applying area correction again. Teacher metadata never enters the model.

Student probabilities are `p = softmax(-energy)` over valid candidates. The only loss is batch-mean teacher-to-student cross entropy:

`loss = mean_b[-sum_{i valid} q[b,i] * log(p[b,i])]`.

Masked log probabilities are replaced by zero before multiplication, avoiding `0 * -inf`. Teacher probabilities are detached in the loss. No coordinate loss, analytic constraint, or auxiliary objective is added. Forward KL is reported as cross entropy minus teacher entropy.

## Training and evaluation interfaces

`train.py` exposes `run_training(config, splits, output_dir, resume=None)`, `build_model`, `normalize_teacher_distribution`, `teacher_student_loss`, `train_step`, `train_epoch`, `validate`, `save_checkpoint`, `load_checkpoint`, and `evaluate_source_grid`. `run_training` accepts a `TrainConfig` (or equivalent dictionary) and raw record split lists, validates and normalizes them, and returns a summary of final validation metrics, parameter count, elapsed run time, progress and split fingerprints. Lower-level training helpers accept already normalized, batched records; `collate_records` is the record-level adapter. Validation restores the model's previous train/eval state and reports sample-weighted means of all six required metrics.

`evaluate_source_grid(model, response, xy_grid, chunk=4096)` accepts `[1,1,H,W]` and normalized `[N,2]`, returning CPU energies `[N]`. Apply softmax once across all N energies, never per chunk. `save_dense_grid` demonstrates that normalization and saves a 64×64 grid over `[-1,2]²`. Its bounds are a mock diagnostic default; evaluation callers can supply any normalized grid.

Each validation interval saves three fixed records as raw `.pt` tensors and PNG comparisons. Raw outputs include response, candidate positions, available physical costs, teacher probabilities, student energies/probabilities, mask, IDs, and all metrics. Complete Cartesian candidate sets use heatmaps; other sets use scatter plots. Training produces `config.json`, `metrics.jsonl`, `summary.json`, `best.pt`, and `last.pt`. Built-in CLI mock data is cost-only, so changing temperature changes the constructed targets. Direct use of supplied probabilities retains their authoritative values.

## Section 16.2 experiments

Run the supplied two-trial mock example with:

```sh
.venv/bin/python experiments.py configs/mock_trials.json --output outputs/my-experiment
```

The example trains hidden dimensions 64 and 128 for 120 epochs each on the same 64-candidate mock targets. It exercises the runner; it is not a real-data hyperparameter recommendation. The default architecture remains hidden dimension 128.

Manifests list explicit trials rather than generating an implicit search:

```json
{
  "base_config": {"epochs": 120, "validation_interval": 40, "checkpoint_interval": 40},
  "trials": [
    {"name": "hidden64", "mock": true, "config": {"hidden_dim": 64}},
    {"name": "hidden128", "mock": true, "config": {"hidden_dim": 128}}
  ]
}
```

Each trial requires a unique `name` and exactly one source: `"mock": true` or `"records": "path/to/teacher.pt"`. Names are case-insensitively unique, 1–64 characters, begin with a letter/digit and otherwise contain letters, digits, underscores or hyphens. Trial configuration overrides the base configuration by field; nested metadata dictionaries are replaced, not deep-merged. Record paths resolve relative to the manifest file. All trial configurations and datasets are validated before training starts. Output directories must be empty; existing runs are not overwritten.

Experiment record files use an upstream metadata envelope:

```python
torch.save({
    "splits": {"train": train_records, "validation": validation_records, "test": test_records},
    "metadata": {
        "dataset_version": "upstream-v1",
        "scene_split_version": "scene-splits-v1",
        "candidate_count": 64,
        "candidate_search_config": {"method": "upstream-search", "budget": 64},
        "teacher_cost_config": {"alpha": 1.0, "beta": 2.0},
        "area_weight_convention": "represented source-cell area"
    }
}, "teacher.pt")
```

The example metadata describes the integration contract, not data generated here. `candidate_count` is the upstream nominal count/budget; adaptive records may contain different valid counts, which are reported separately. The runner imports all six metadata fields and rejects any conflicting config override. Metadata describes already generated supervision, never a command to rerender or resample. The single-run CLI still accepts legacy bare split dictionaries with explicit metadata in its config, as well as envelopes whose metadata agrees with that config.

Tuning behavior:

- **Model size:** change `hidden_dim`, retaining the prescribed architecture.
- **Temperature:** set `teacher_temperature` on cost-only records. Mock experiment records are cost-only automatically. An explicit temperature setting anywhere in an upstream experiment manifest is rejected if any record supplies `teacher_prob`; upstream probabilities are never discarded or retempered silently.
- **Candidate count:** use separately generated upstream record files with the intended nominal counts. Student code never truncates or resamples these candidates. Mock mode can generate another candidate count solely to test the pipeline.
- **α/β:** use upstream files with the intended coefficients in `teacher_cost_config`. If either coefficient is recorded, both must be finite, nonnegative, and not both zero. Student code does not calculate or recombine physical cost terms. Mock metadata cannot be relabeled as physical α/β experiments.

Each trial directory contains its config, `trial.json` provenance, epoch metrics, validation diagnostics, checkpoints and `summary.json`. The experiment root contains the original manifest and `results.json`/`results.csv`, with forward KL, L1 distance, entropy mismatch, parameter count, runtime and comparison group. Results always evaluate the **last checkpoint** at the configured total epoch count; `best.pt` is preserved for separate evaluation. Runtime includes the epoch loop, validation, checkpointing and plots; it excludes preflight/model setup and is not a controlled throughput benchmark. Resumed summaries report only the resumed segment’s elapsed time.

Rankings are local to a comparison group. The grouping fingerprint includes the exact ordered validation observations, normalized candidates, masks, scene/view identifiers and effective teacher probabilities, plus dataset/search/teacher/area metadata and temperature. Changed targets, candidate sets or teacher configurations therefore remain separate. Singleton groups and failed trials have no rank. Within a group, lower forward KL ranks first, followed by L1 and entropy mismatch; exact ties share a rank. No test-set metrics participate in selection. Failed execution is recorded alongside completed results and raises an error; remaining trials are not launched.

The Python APIs are `prepare_trials(manifest, base_dir)`, `run_experiments(manifest, output_dir, base_dir)` and `run_training(config, splits, output_dir, resume=None)`. Experiment directories preserve per-trial provenance; individual interrupted trials can be resumed through `run_training` or the training CLI. Rerun the experiment into a new directory for a fresh combined ranking; resuming a single trial does not automatically rewrite its parent experiment table.

## Verification and remaining integration work

`verify_pipeline.py` overfits 12 distinct mock records, four per family: localized Gaussian, two Gaussian peaks, and elongated ridge. Small target shifts have distinct response intensities; this intentionally learnable association is not a physical simulator. It uses the unmodified default architecture and optimizer, checks every 25 steps, and stops at mean forward KL < 0.02 and mean L1 < 0.1 or fails after 3,000 steps. No full training is launched by this script. The unit suite also runs this acceptance experiment. It additionally checks numerical output, cross-entropy, input-gradient and parameter-gradient equivalence against an independent transcription of PDF Section 5.4 at hidden dimensions 16 and 128, as well as manifest validation, target/provenance handling, reproducible trials and exact epoch resume.

The verifier writes `outputs/verification/report.json`, history, an accepted checkpoint, teacher/student comparison figures, and dense-grid plots. It checks exact checkpoint output equivalence, exact agreement after one resumed optimization step, and dense chunk equivalence. Mock train/validation/test scene IDs are disjoint, but their synthetic target families are shared, so their scores are not evidence of generalization to unseen geometry.

Dense interpolation is diagnostic only. Sparse candidate overfitting does not constrain every coordinate; the prescribed Fourier baseline may show oscillations between supervised grid points. No smoothness loss or architectural changes are added to suppress them.

Real record integration, 5–20-record real overfitting, small real-subset training, held-out geometry validation, and full training remain pending. Confirm represented cell areas with the teacher/search owner before interpreting adaptive-candidate probabilities. In particular, the prescribed student uses counting-measure softmax, while the teacher may include area mass; consistent sampling semantics must be agreed upstream. No real-data or physical-geometry success is claimed.
