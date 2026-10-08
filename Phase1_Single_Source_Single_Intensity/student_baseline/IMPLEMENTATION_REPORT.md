# Thomas’s Section 16.2 implementation handoff

This independent project implements the model and optimization workstream in `G2-02_Conditional_Energy_Source_Field_Model.pdf`, following `CODEX_G2-02_Phase1_Student_Model_Instructions.md`. Dataset/teacher construction, analytic evaluation scenes, and profiling remain outside Thomas’s assignment.

Publication verification (October 3, 2026): all **54 tests passed in 18.92 seconds** from this copied `student_baseline/` directory, using the development Python environment. All eight student-module imports resolved to the copied source. Virtual environments and generated artifacts are excluded from Git; artifact paths below refer to the recorded development runs and can be regenerated with the documented commands. The baseline differs from the newer shared interface as detailed in its README.

## 1. Files created or modified

- `model.py`: Fourier features, spatial CNN, candidate attention, energy field and probability outputs.
- `train.py` (updated): shared `run_training(config, splits, output_dir, resume=None)` entry point, CLI, configuration validation, metrics, dense inference and checkpoint/resume helpers. New run checkpoints record split fingerprints.
- `data.py`: exact teacher schema validation, normalization, padding and scene-split checks.
- `mock_data.py`: deterministic random and structured synthetic teachers.
- `diagnostics.py`: raw diagnostic tensors, grid/scatter comparison plots and dense-grid plots.
- `verify_pipeline.py`: bounded 12-record overfit and integration acceptance experiment.
- `experiments.py` (new): manifest preflight, isolated trial execution, provenance checks, target-safe comparison groups and JSON/CSV result tables.
- `run_data.py` (new): cost-only mock splits, trusted upstream-record loading, provenance and target fingerprints.
- `configs/mock_trials.json` (new): reproducible two-trial hidden-dimension smoke experiment.
- `tests/test_reference.py`, `tests/test_experiments.py` (new): PDF parity, experiment and shared-training tests.
- `tests/test_pipeline.py`, `pytest.ini`: retained baseline tests.
- `requirements.txt`, `requirements-lock.txt`: dependencies and exact installed versions.
- `.gitignore`, `README.md`, `IMPLEMENTATION_REPORT.md`: local artifact exclusions, usage/schema documentation and this handoff.
- Local artifacts: `.venv/`, `outputs/verification/`, and `outputs/thomas-smoke/` (excluded from version control).

No upstream physics modules were used or modified. The model architecture, preprocessing contract, sole training objective and baseline acceptance experiment were preserved.

## 2. Architecture

The required baseline is implemented without extensions: 1→32→64→128 CNN channels, spatial feature tokens, six-frequency Fourier coordinates, an MLP query encoder, scaled dot-product candidate-to-image attention, and an MLP energy head. Only response and normalized candidate coordinates enter the model. Energy output is `[B,K]`; `model.predict` also exposes `student_probability` and `student_log_probability`. New tests compare independent PDF-reference and implementation outputs/loss at `rtol=atol=1e-12`, and input/parameter gradients at `rtol=1e-10, atol=1e-12`, using float64 and hidden dimensions 16 and 128. Both cases passed.

## 3. Exact teacher record schema

Required: finite float32-compatible `response [1,H,W]`, `candidate_xy [K,2]`, string/integer `scene_id`, string/integer `view_id`, and either `teacher_prob [K]` or `physical_cost [K]`.

Optional: bool `valid_mask [K]` (all valid by default), positive `area_weight [K]` (equal by default), and `coordinate_space` (`world` by default, or explicitly `normalized`). World coordinates additionally require `window={x0,y0,L}` with finite values and positive L. Preprocessing computes `(xy-[x0,y0])/L` and preserves ordering. Every sample needs at least one valid candidate. Probabilities must be finite/nonnegative, sum to one within `atol=rtol=1e-5`, and assign no invalid mass. Cost and area values must be finite on valid entries. See README for batching and on-disk split format. Experiment record files additionally contain `{splits, metadata}`. Required upstream metadata fields are dataset version, split version, candidate-search config, teacher-cost config, area convention and nominal candidate count. Candidate count and α/β overrides must agree with the upstream file. An explicit temperature trial rejects supplied probabilities instead of silently replacing them.

## 4. Exact loss

For valid candidates, `p = softmax(-energy)` and `loss = mean_b[-sum_i q[b,i] log p[b,i]]`. Invalid log probabilities are replaced with zero before multiplication, and teacher targets are detached. No other losses are present.

When only cost is supplied, `q = softmax(log(area_weight) - physical_cost / temperature)` with invalid candidates excluded. Supplied probabilities are validated and used directly. Student logits are not area-adjusted, as specified by the baseline.

## 5. Tests added

Tests cover model shapes including 128×128 response inputs, candidate permutation consistency, probability normalization, masked loss/gradients, area-weighted targets, malformed records, coordinate normalization, variable-count padding, target precedence, scene leakage, fixed response size, deterministic random mocks, cost-only training, dense chunk equivalence, empty dense grids, diagnostic scatter rendering with absent costs, exact checkpoint/resume (including scheduler and RNG states), the 12-record overfit gate, and CLI training followed by epoch resume. Added tests cover reference output/loss/gradient parity, manifest errors, source exclusivity, name/path validation, complete preflight before training, temperature effects and rejection of supplied targets, metadata preservation, prevention of independent candidate resampling, fingerprint sensitivity, comparison-group isolation, independent trial outputs, repeated-seed equality and uninterrupted-versus-resumed epoch equality.

## 6. Test results

Executed in the isolated environment using Python 3.14.6, PyTorch 2.14.1, NumPy 2.5.3, Matplotlib 3.11.2 and pytest 9.1.1 on CPU.

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q model.py data.py mock_data.py train.py diagnostics.py verify_pipeline.py experiments.py run_data.py tests
.venv/bin/python experiments.py configs/mock_trials.json --output outputs/thomas-smoke
```

Final pytest result: **54 passed in 18.42 seconds**, including the retained overfit acceptance experiment. Compilation passed. GPU execution was not tested. The saved test log is `outputs/thomas-smoke/test-results.txt`.

The two-trial smoke experiment completed 120 epochs / 360 steps per trial on identical six-record mock validation targets. Both used 64 candidates, temperature 0.1 and seed 0. The table reports the **last checkpoint**, not the best validation-selected checkpoint:

| Hidden dimension | Forward KL | L1 | Entropy mismatch | Parameters | Run time (s) | Group rank |
|---|---:|---:|---:|---:|---:|---:|
| 64 | 0.013474 | 0.095683 | 0.014669 | 80,321 | 3.765 | 1 |
| 128 | 0.016891 | 0.113879 | 0.039974 | 186,881 | 3.677 | 2 |

Runtime includes training, validation, checkpointing and plots and is not a controlled throughput benchmark. These mock results do not change the required 128-unit baseline or select a real-data configuration. The 128-unit smoke run’s L1 exceeds 0.1; this is a separate fixed-epoch experiment, not the 12-record acceptance gate in Section 7.

The full results and provenance are in `outputs/thomas-smoke/results.json`, `results.csv`, and each trial’s `trial.json`/`summary.json`. Only identical effective validation targets and metadata share a ranking; different candidate sets, temperatures and teacher α/β configurations are kept separate. Upstream-style record-envelope tests used synthetic fixtures, not real teacher data.

## 7. Mock overfit result

Twelve distinct records, four per family, 64 candidates per record, default hidden dimension/frequencies, AdamW learning rate 0.001, batch size 4 and seed 0. The threshold is a mean over all records, not a per-record or per-family threshold.

| Target family | Mean forward KL | Mean L1 distance |
|---|---:|---:|
| One peak | 0.006973 | 0.077002 |
| Two peaks | 0.011712 | 0.099688 |
| Ridge | 0.018131 | 0.104223 |
| All 12 records | **0.012272** | **0.093637** |

The overall acceptance gate (KL < 0.02 and L1 < 0.1) passed at **325 steps**, below the 3,000-step limit. Initial mean KL was 1.713487 and L1 was 1.490248. The ridge family's mean L1 is slightly above 0.1; the gate required the overall mean.

Full numerical results and learning history are in `outputs/verification/report.json` and `outputs/verification/history.json`. The model checkpoint is `outputs/verification/mock-student.pt`.

## 8. Checkpoint round trip

Validation outputs matched exactly after loading into a fresh model/optimizer. One subsequent training step produced exactly matching state dictionaries. Unit tests additionally verify restored scheduler state and Python/NumPy/PyTorch random streams. The CLI completed epoch 1, resumed, and completed epoch 2 with global step increasing from 3 to 6. The new shared training API also matched uninterrupted two-epoch training exactly, including model state and validation results. It rejects changed split fingerprints on resume. Independent repeated-seed trial execution produced identical model state and validation metrics.

## 9. Dense-grid and diagnostic status

Generated and visually inspected teacher/student comparison images for all three target families under `outputs/verification/diagnostics/record-{0,1,2}.png`. Candidate-grid plots reproduce localized, bimodal and elongated structures.

Generated 64×64 dense grids for all three families under `outputs/verification/dense/{one_peak,two_peaks,ridge}/`. Inspected the dense ridge plot: it retains a left-side high-probability region but shows oscillations between supervised candidate points. Sparse overfitting does not establish smooth dense interpolation.

Chunked versus unchunked maximum energy difference: **9.54e-7**. Whole-grid probability sum: **0.99999988**. Raw energies, positions and probabilities accompany the plots as `.pt` files.

The Section 16.2 smoke experiment additionally produced six 64×64 dense grids under `outputs/thomas-smoke/{mock-hidden64,mock-hidden128}/dense/{one_peak,two_peaks,ridge}/`. Inspected the final 64-unit bimodal comparison, 128-unit ridge comparison and dense ridge plots for both models. The candidate-grid plots retain the intended target structure; dense ridge plots retain visible high-frequency artifacts between supervised coordinates. Maximum chunked/unchunked difference across these six grids was **3.81e-6**; maximum whole-grid probability normalization error was **5.96e-7**. Numerical checks are saved in `outputs/thomas-smoke/dense-checks.json`.

## 10. Remaining integration assumptions/blockers

Real teacher records are absent. The experiment runner is ready for separately generated upstream candidate-count and α/β record sets; it deliberately does not recompute physical costs, resample adaptive candidates or claim that real tuning has been performed. Real-record integration, real-record overfit, small real-subset training, held-out geometry tests and full-scale training have not been run. Synthetic fixtures encode deliberately learnable response/target associations, not RIND physics; their distinct scene IDs do not establish held-out geometry generalization.

The teacher/search owner must confirm the relationship between adaptive candidate cells and physical area mass. Supplied teacher probabilities must already embody the intended area convention. The specified student softmax depends on the candidate set, so calibration across changing adaptive candidate sets requires agreement on sampling semantics. No upstream physics computation, analytic constraint loss, or architectural complexity was added.
