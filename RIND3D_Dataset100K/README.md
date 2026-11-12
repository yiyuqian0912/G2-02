# RIND3D-100K: local-only production profile for a 16 × H20 server

**Branch:** `yushu/3d-100k-h20`, based on `yushu/3d`. The RIND3D 1K pipeline and its data are untouched.

**The profile:**
* 100,000 scenes at 1024³;
* exactly 12 windows per scene (4 / 4 / 4 of sides 64 / 128 / 256);
* 1–4 sources;
* source-separated channels;
* globally balanced effective layers;
* exact tau-pencil global posteriors for ≈ 2.9M informative channels.

**No global response field is computed or stored.**

| deliverable | document |
|---|---|
| A. Local-only rendering feasibility | `docs/LOCAL_RENDERING_100K.md` |
| B. 100K dataset contract | `docs/CONTRACT_100K.md`, `contract/RIND3D_100K_CONTRACT_v1.json` |
| Selection law and training weights | `docs/SELECTION_LAW_100K.md` |
| C. GPU architecture and scalability | `docs/GPU_ARCHITECTURE_100K.md` |
| D. Posterior-label cost | `docs/LABEL_COST_100K.md` |
| E. Handoff package | `handoff/README_H20.md`, `scripts/make_handoff.py`, `handoff/env/`, `handoff/fixtures/` |
| F. Go / no-go (provisional) | `docs/GO_NO_GO_100K.md` |

## Code (`rind3d_100k/`)

| module | role |
|---|---|
| `local_eval.py` | exact local-only window decisions: exact probes, rational all-lit hull certificate, lazy decisions, progressive certified CUDA tiles |
| `physical.py` | one scene: geometry, sources, 4/4/4 selection with rate streams, one atomic container |
| `controller.py` | epoch-frozen effective-layer balancing (the 1K rule, parallel-safe) |
| `pool.py`, `coordinator.py`, `worker.py` | supervised multi-GPU node: per-worker GPU binding, timeouts, RSS limits, retries, quarantine, journals, stop / resume |
| `labels.py` | tau-pencil label tasks on local crops plus global geometry; per-cell label format; GPU-error CPU fallback |
| `storage.py` | containers, checksummed journals, consolidated labels |
| `validate.py`, `loader.py`, `fixtures.py` | validation; canonical loader plus flatten adapter with ω; reference fixtures |

## Scripts (`scripts/`)

`run_node.py` (launch and resume), `probe_hardware.py`, `smoke_test.py`, `bench_labels.py`, `bench_scaling.py`,
`report.py`, `estimate.py`, `verify_root.py`, `make_fixtures.py`, `make_handoff.py`.

## Quick start (development machine)

```bash
bash RIND3D_Dataset100K/handoff/env/setup_env.sh && source r3d_env.sh
python RIND3D_Dataset100K/scripts/smoke_test.py --work /tmp/r3d_smoke
python RIND3D_Dataset100K/scripts/run_node.py --root /tmp/r3d_dev --slots 0:16 --epoch-scenes 8 --label-slots 0:1 \
       --prior-manifest RIND3D_Dataset100K/handoff/prior/release_1k_rate_stats.jsonl
```
