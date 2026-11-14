# RIND3D-100K: local-only production pipeline for a 16 × H20 server

**Branch.** `yushu/3d` is the canonical RIND3D-100K pre-production branch. It holds the shared physics and labelling modules, the 1K reference pipeline (kept for regression) and this package. `yushu/3d-100k-h20` is the historical development branch of the v1 profile; it is not developed further.

**The profile** (contract v2, `contract/RIND3D_100K_CONTRACT_v2.json`, **provisional**):
* **Scenes:** 100,000 at 1024³, from nine geometry families.
* **Windows:** exactly 12 per scene, 4 / 4 / 4 of sides 64 / 128 / 256.
* **Sources:** K ~ κ = (0.15, 0.15, 0.15, 0.55) declared per slot (provisional default). Responses are source-separated.
* **Target population T\*:**
  * natural acceptance of every admissible window, with no m-dependent thinning;
  * an optional coverage slot: L64 only, in K = 4 scenes, 3 natural + 1 coverage window, c(m) = (1/8, 1/4, 1/2, 1);
  * the provenance needed for balance-heuristic training weights: stream, proposal index, U, c-row, and the exact-m rate-stream histograms.
* **Labels:** certified global 128³ posteriors of every informative channel by **FastLabel v3.4** (`rind3d_posterior.fastlabel`):
  * `certified` (deterministic) or `certified_stat` (TV ≤ 0.05 with δ = 0.01 / 4.8 M);
  * the exact tau-pencil engine stays available (`--label-engine tpe`) as the reference and for regression.
* **Storage:** no global response field is computed or stored.

**The provisional κ and coverage table are not a validated final population.** They are frozen only after the H20 canary review (`docs/H20_CANARY_RUNBOOK.md` §7).

**Contract v1** (`RIND3D_100K_CONTRACT_v1.json`) is the earlier profile: an adaptive source-count controller, m-balanced thinning and TPE labels. Its code paths and fixtures are kept for regression only.

| topic | document |
|---|---|
| **H20 production gate and canary runbook** | `docs/H20_CANARY_RUNBOOK.md` |
| contract | `contract/RIND3D_100K_CONTRACT_v2.json`; `docs/CONTRACT_100K.md` (v1 text plus the v2 changes) |
| sampling law and training weights | `rind3d_100k/sampling.py`; `docs/SELECTION_LAW_100K.md` (v2 section first); `RIND3D_StudentResearch/A_TASK_DEFINITION.md`, `J_COMPLEXITY_COVERAGE.md` |
| local-only rendering | `docs/LOCAL_RENDERING_100K.md` |
| GPU architecture | `docs/GPU_ARCHITECTURE_100K.md` |
| label cost | `docs/LABEL_COST_100K.md` (TPE, v1); FastLabel: `../RIND3D_Posterior/docs/FASTLABEL_CORE.md` |
| operator guide and environment | `handoff/README_H20.md`, `handoff/env/` |
| earlier go / no-go (v1, TPE) | `docs/GO_NO_GO_100K.md` (historical) |

## Code (`rind3d_100k/`)

| module | role |
|---|---|
| `sampling.py` | contract v2 law: κ draw, side streams (natural, coverage), outcome-independent plans, balance-heuristic T* weights |
| `local_eval.py` | exact local-only window decisions: exact probes, rational all-lit hull certificate, lazy decisions, progressive certified CUDA tiles |
| `physical.py` | one scene: geometry, sources, natural / coverage streams (v2) or thinned selection (v1), exact-m rate streams, one atomic container |
| `controller.py` | `plan_epoch`: declared plans (v2) or the epoch-frozen balancing controller (v1) |
| `pool.py`, `coordinator.py`, `worker.py` | supervised multi-GPU node: per-worker GPU binding, per-GPU FastLabel semaphores, timeouts, RSS limits, retries, quarantine, journals, stop / resume, contract digest check |
| `labels.py` | label tasks: FastLabel (default; certificate status and tv_bound preserved) or tau-pencil (CPU / GPU pass) |
| `storage.py` | atomic containers (fsync + directory fsync), fsync'd JSONL appends with torn-tail tolerant readers, checksummed journals, consolidated labels with a sha256 record |
| `validate.py` | scene, sampling-law and label validation (certificates, true source in support, δ policy) |
| `loader.py` | canonical loader plus `flatten(population=...)`: Tstar / Tstar_natural / stored, weight estimators, certified and certified_stat labels, explicit one-hot fallback, `student_inputs` |
| `fixtures.py` | reference fixtures (TPE label digests) |

**Process rule.** One rind3d CUDA runtime per process; FastLabel workers never share a process with the CuPy tau-pencil engine. Each runtime owns its CUDA context, so mixing them invalidates the other's handles. The coordinator, the smoke test and the tests follow this rule.

## Scripts (`scripts/`)

| script | role |
|---|---|
| `run_node.py` | launch and resume. Defaults: contract v2, FastLabel |
| `verify_root.py` | integrity: containers, sampling law, physics spot checks, journals, labels |
| `coverage_report.py` | κ, per-(L, m, family) coverage, T* ESS, label statuses |
| `compare_engines.py` | FastLabel vs exact tau-pencil accuracy |
| `student_smoke.py` | loader → pilot features → one Student forward / backward step |
| `smoke_test.py` | server bring-up |
| `make_fixtures.py` | v1, v2 and FastLabel references |
| `probe_hardware.py`, `bench_labels.py`, `bench_scaling.py`, `report.py`, `estimate.py`, `make_handoff.py` | probes, benchmarks, reports and packaging |

## Quick start (development machine)

```bash
bash RIND3D_Dataset100K/handoff/env/setup_env.sh && source r3d_env.sh
python -m pytest -q RIND3D_Dataset100K/tests
python RIND3D_Dataset100K/scripts/smoke_test.py --work /tmp/r3d_smoke
python RIND3D_Dataset100K/scripts/run_node.py --root /tmp/r3d_dev --slots 0:8 --gpus 0 --label-cpu 6 --rss-gb 3
python RIND3D_Dataset100K/scripts/verify_root.py /tmp/r3d_dev --labels
python RIND3D_Dataset100K/scripts/coverage_report.py /tmp/r3d_dev
```

**Do not start the 100K campaign before every section of `docs/H20_CANARY_RUNBOOK.md` has passed on the target.**
