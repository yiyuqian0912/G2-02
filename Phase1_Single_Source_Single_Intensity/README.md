# RIND Phase I

[完整中文 README](README.zh-CN.md) · [Method](docs/method.md) · [Interfaces](docs/interfaces.md) · [Training](docs/experiments.zh-CN.md) · [Results explorer](docs/dense_probability.zh-CN.md)

Learn a conditional source-location energy field from **local response and window position/size**. A RIND physics teacher rerenders candidate sources in the hidden scene, produces an equal-per-candidate Gibbs target, and supervises the student's distribution. No single-coordinate regression or area weighting is used.

The current implementation includes native quadtree observations, boundary-weighted response cost, optional symmetric edge cost, uniform physical targets, lazy size-bucketed training, resumable frozen experiments, evaluation, and a test-set explorer. Adaptive retention is an independent coverage/cost comparison; the full proposed uncertainty-driven, response-to-edge cascade and controlled one/two-boundary experiments remain research work.

```bash
uv sync --locked --extra train --extra browser --extra diagnostics
./scripts/install_data.sh # first installation, with the single-source ZIP in this directory
./view_results.sh
```

Open **http://127.0.0.1:8767**. Select a saved test scene/view to compare the local observation, actual dense model predictions, and original teacher samples. Query spacing is adjustable down to one source pixel. The matched-candidate mode uses identical coordinates and a shared probability color scale. Hover for values; export PNG or full-precision NPZ.

Start a new 5,000-observation training run:

```bash
./run_training.sh --run-name consistent_5000 \
  --scene-counts 1250 125 125 --window-sizes 16 32 64 128 \
  --set views_per_size=1 --seeds 0 --epochs 30 --patience 8 \
  --spacing 64 --fourier-frequencies 4 --device cpu --threads 4
```

The default template is `configs/experiment_consistent.json`. Outputs go to `outputs/experiments/<run-name>/`. Use `--check` to inspect settings. Change the run name when changing parameters. Checkpoints resume only under the same frozen configuration, dataset, selection, and code.

The new protocol fixes scene pools, uses response/window inputs by default, retains explicit pixel coordinates inside the encoder, checks Fourier frequency against teacher spacing, and normalizes the primary student prediction over world-minus-window without a hidden obstacle mask. Geometry-assisted results are labelled separately; invalid probability mass is measured rather than silently discarded.

Legacy `train5000` and `round1_cpu` checkpoints remain readable. They used local masks and geometry-assisted normalization; some frequency/spacing choices are under-resolved. They are not v2 results and require retraining for the new model. The fixed v2 held-out pool reuses previously inspected `train5000` scenes for continuity, not a fresh final blind test.

Core ownership: `src/rind_phase1/` for data, physics, model and orchestration; `part_e/` for authoritative metrics and adaptive comparisons; `configs/` for frozen choices; `scripts/` for launch/export; `tests/` for regression checks. The Chinese README maps every core file to its purpose and deliverable.

See [integration scope](docs/integration.md) and [module responsibilities](docs/modules.zh-CN.md). Runtime data, checkpoints, reports and personal run configurations are local-only.
