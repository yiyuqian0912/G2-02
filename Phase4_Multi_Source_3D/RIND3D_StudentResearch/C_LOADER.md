# C — Corrected training loader (M2): every certified status by default, explicit one-hot fallback, preserved provenance

**Files:**
* code: `RIND3D_Dataset1K/rind3d_1k/loader.py`;
* tests: `RIND3D_Dataset1K/tests/test_loader.py` (4 tests, passing).

The canonical multi-channel access (`RIND3D1K`, `Scene.window`) is unchanged.

## What changed

| item | before (b7c8dee) | now |
|---|---|---|
| accepted label statuses | `certified` only (`accept_statistical=False`). After a FastLabel backfill this keeps 5.8% / 25.4% / 57.4% of L64 / L128 / L256 labels, all single-cell: a selection on hidden geometry (audit M2) | `accept=("certified", "certified_stat")` with `tv_bound <= max_tv = 0.05`: every status whose certificate meets the accuracy requirement. `certified_loose` (0.05 < TV ≤ 0.10) and `uncertified` are not certified posteriors and are not accepted by default |
| target population | implicit: ω-weighted B1K | **required argument** `population=` ∈ {"B1K", "Tstar", "Tstar_mis"}. There is no default; the choice defines what the Student learns (`A_TASK_DEFINITION.md`) |
| ω estimator | "rate" (one scene-side has ω = 0) | `estimator=` with five unbiased estimators; default `combined_rb` (strictly positive, lowest variance; `B_SAMPLING_WEIGHTING.md` §2) |
| missing or not-accepted label | silently dropped | `missing="yield"` (default): the example is yielded with `supervision="missing"` and `target=None`. `missing="drop"` skips it but counts it in `report` |
| one-hot fallback | none | `fallback="onehot"` (explicit opt-in): the true-source cell, with `supervision="onehot_fallback"`. The original status ("timeout", "uncertified", ...) is kept in `status` |
| one-hot baseline | — | `supervision="onehot"`: the true-source cell for every example (One-hot arm) |
| provenance | status, certificate, engine | additionally: engine_version, path, config_sha256, sha256 of the task file, dataset ("release" / "natural"), population, estimator |
| natural supplement | — | `RIND3DNatural(root)` reads NAT-1K. Its windows get T* weights 1/n_p, or balance-heuristic weights with the release |
| privileged | — | `with_source=True` adds `privileged = {source, source_cell}` (targets and evaluation only). The cell convention, (ix·128 + iy)·128 + iz, was checked against the certified supports |

## Per-example record (`flatten`)

**Ids:** scene_id, scene, dataset, window, source, L, Lp, origin.

**Inputs:** `solid`, `response` (one channel). They are omitted with `with_inputs=False`.

**Supervision:**
* `supervision` ∈ {"posterior", "onehot", "onehot_fallback", "missing"};
* `target` = {cells, q} or None;
* `label` (the full posterior dict or the status slot).

**Certification:** `status`, `tv_bound`, `certificate` (statistical labels: eps, δ, n, K, thresholds, seed, n_tasks, policy).

**Provenance:** `provenance`.

**Weights:** `population`, `estimator`, `weight` (a loss weight; never a model input).

## Rules this enforces

1. A one-hot fallback is never presented as a certified posterior: its `supervision` and `status` say what it is.
2. Posterior-reference metrics (label CE, KL) must be computed only on `supervision == "posterior"`. The pilot's evaluation (`pilot/evaluate.py`) computes them only where a certified label exists.
3. Every non-posterior example is reported per side, with its count and weight mass (`report`). Nothing is dropped silently.
4. `m`, `mask`, `k`, the weights and the status are not model inputs. They encode hidden channels or label paths.

## Missing labels and the training distribution (brief §3)

* After the 1K backfill, the number of informative tasks without an accepted label is reported in `D_BACKFILL_REPORT.md`.
* If labels go missing for reasons that depend on (G, W, r_j), such as timeouts or loose bounds, dropping those examples is selection on hidden geometry: MNAR (audit 02 §6). The label-or-one-hot target removes the bias:
  * for each such example it uses the one-hot target, whose conditional expectation given V is the same posterior (audit 04 Cor. 1);
  * the population optimum is therefore unchanged, and only variance increases.
* So `fallback="onehot"` is the correct choice whenever the missing fraction is not zero. Report the fallback count with every result.
