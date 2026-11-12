# RIND3D-100K on a 16 × H20 server: operator guide

**Status:** a validated development package. Nothing here has run on an H20 yet.
* Every number measured on the development laptop is labelled *[laptop]*: RTX 3060 Laptop 6 GB, Ryzen 7 5800H 8C/16T, 31 GB.
* Every projection is labelled *[projection]*.

**Before any production run, the steps in §4 must pass in order.**

## 1. What RIND3D-100K is

* **Scenes.** RIND3D scenes are canonical polyhedral (CSG) worlds at 1024³. There are 1–4 point sources, with exact
  closed-segment ray visibility.
* **Windows.** Each scene has exactly 12 local observation windows: 4 each of sides 64, 128 and 256 voxels. No source
  lies in any closed window box.
* **Responses.** Each window stores the SOLID crop once, and one binary response channel per physical source.
* **Effective layers.** m(W) is the number of channels with both lit and dark FREE receivers. A balancing controller
  targets 25% of windows for each m = 1..4 over the 1.2M windows.
* **Posteriors.** Every informative channel (≈ 29 per scene, ≈ 2.9M in total) gets a *global* single-source posterior.
  * The target is Uniform(F_W^j) over the legal source domain minus the closed window box, given the GLOBAL geometry.
  * It is computed by the exact tau-pencil engine (TPE v3.1) as certified cell-mass intervals on the 128³ grid, with a
    rigorous TV bound.
* **Storage.** The canonical storage keeps the source axis: scene → window → source → (response, posterior slot).
  Flattening to single-source examples is only `rind3d_100k.loader.flatten`.

## 2. Why no global response fields

**The 1K pipeline** rendered a certified 1024³ field per source (≈ 2 s per source *[laptop]*) and answered every window
proposal from exact count indices.

**The 100K pipeline** computes only what the selection law needs (`rind3d_100k/local_eval.py`). Every decision equals
the exact rule:
* exact CPU receiver probes (≈ 0.5 µs each *[laptop]*) can *prove* a channel informative;
* an exact rational hull certificate can *prove* a channel all-lit;
* a proposal is decided without rendering when U < c(m) is settled for every m still possible;
* otherwise certified CUDA 64³ tiles are rendered progressively (≈ 0.15 ms per tile *[laptop]*), stopping as soon as
  the channel is decided;
* accepted windows are rendered completely for every source.

**Measured *[laptop]*:**
* Equivalence with the full-field pipeline on 12 scenes (k = 1..4, all families, realistic thinning): identical
  windows, identical Λ̂, and 0 mismatches in 2.9×10⁹ response bits.
* Cost: 3.7 s per scene (p95 6.1 s), against 3.3–14.3 s full-field.

## 3. Package contents

| path | purpose |
|---|---|
| `RIND3D_Generator/` | certified geometry and renderer (rind3d, canonical-geometry natives with CGAL) |
| `RIND3D_Posterior/` | the tau-pencil labeller (CPU reference plus exact GPU pass), with bit-exact regression tests |
| `RIND3D_Dataset1K/` | reused library code: families, source rule, 1K controller logic |
| `RIND3D_Dataset100K/rind3d_100k/` | 100K pipeline: local_eval, physical, controller, pool, coordinator, labels, storage, validate, loader, fixtures |
| `RIND3D_Dataset100K/contract/RIND3D_100K_CONTRACT_v1.json` | the frozen production contract |
| `RIND3D_Dataset100K/handoff/env/` | `setup_env.sh`, `requirements-lock.txt`, `Dockerfile`, `rind3d_100k.def` (Apptainer) |
| `RIND3D_Dataset100K/handoff/fixtures/` | 14 label fixtures with reference digests (1K release, train split) and 4 physical fixtures (100K contract) |
| `RIND3D_Dataset100K/handoff/prior/` | controller prior: the rate statistics of the independent 1K release |
| `RIND3D_Dataset100K/scripts/` | `probe_hardware`, `smoke_test`, `bench_labels`, `bench_scaling`, `run_node`, `report`, `estimate`, `verify_root`, `make_handoff` |
| `RIND3D_Dataset100K/docs/` | contract, selection law, rendering study, GPU architecture, label cost, go / no-go |
| `HANDOFF_MANIFEST.json` | commit, sha256 of every file, contract, fixture and lock hashes |
| `deps/` (optional) | CGAL 6.2.1 and Boost 1.92 source archives, for offline builds |

## 4. Bring-up sequence on the server (in order; stop at the first failure)

```bash
# 0. unpack and check integrity
sha256sum -c rind3d_100k_handoff_<commit>.sha256
tar xzf rind3d_100k_handoff_<commit>.tar.gz && cd rind3d_100k_<commit>
python3 - <<'EOF'
import json, hashlib, pathlib
m = json.load(open("HANDOFF_MANIFEST.json"))
bad = [f for f, v in m["files"].items() if not f.startswith("deps/") and hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest() != v["sha256"]]
print("files", m["n_files"], "bad", bad[:5])
EOF
git init -q && git add -A && git -c user.email=op@localhost -c user.name=op commit -qm "handoff <commit>"   # smoke_test records a revision

# 1. environment (or build the container: handoff/env/Dockerfile, handoff/env/rind3d_100k.def)
bash RIND3D_Dataset100K/handoff/env/setup_env.sh          # offline: R3D_DEPS=$PWD/deps R3D_WHEELS=/path/wheelhouse
source r3d_env.sh

# 2. hardware probe (CPU, NUMA, RAM, GPUs, topology, storage, schedulers; FP64/FP32/bandwidth/renderer micro-benchmarks)
python RIND3D_Dataset100K/scripts/probe_hardware.py --out probe.json --paths /data/r3d,/local_scratch --io-mb 4096 --bench

# 3. smoke test: engine regression, label fixtures CPU+GPU (bit-exact digests), physical fixtures (bit-exact), node run, journal
python RIND3D_Dataset100K/scripts/smoke_test.py --work /local_scratch/r3d_smoke --gpu 0
#    add --full-fixtures for all 14 label fixtures (heavy tails take up to ~10 min each)

# 4. single-GPU label benchmark (production layout; adjust CPU workers to the real core count / 16)
python RIND3D_Dataset100K/scripts/bench_labels.py --gpus 0 --cpu-workers 8 --gpu-workers-per-gpu 2 --set all --repeat 1 --out bench_1gpu.json

# 5. multi-GPU scaling (labels and physical scenes; 1,2,4,8,16 GPUs)
python RIND3D_Dataset100K/scripts/bench_scaling.py --out-dir /local_scratch/r3d_scaling --gpu-counts 1,2,4,8,16 --cpu-per-gpu 8

# 6. representative canary on the real machine: 256 scenes and a stratified label sample
python RIND3D_Dataset100K/scripts/run_node.py --root /data/r3d_canary --slots 0:256 --gpus all --no-labels \
       --prior-manifest RIND3D_Dataset100K/handoff/prior/release_1k_rate_stats.jsonl
python RIND3D_Dataset100K/scripts/run_node.py --root /data/r3d_canary --slots 0:256 --gpus all --no-physical --label-sample 200,200,200,7
python RIND3D_Dataset100K/scripts/report.py /data/r3d_canary
python RIND3D_Dataset100K/scripts/verify_root.py /data/r3d_canary --labels

# 7. projection (go / no-go input)
python - <<'EOF'   # export the canary's WARM label metas (each worker's first task includes the one-time JIT compile)
import sys, json; sys.path[:0]=["RIND3D_Dataset100K"]
from rind3d_100k.storage import journal_index, read_record, decode_payload
ms = [decode_payload(read_record(loc)[1])["meta"] for k, loc in sorted(journal_index("/data/r3d_canary").items(), key=lambda kv: kv[1][1])]
first = {}
for m in ms: first.setdefault(m["pid"], m["key"])
with open("canary_label_costs.jsonl", "w") as f:
    for m in ms:
        if m["key"] not in first.values():
            f.write(json.dumps(dict(L=m["L"], cpu=m["cpu"], wall=m["wall"], status=m["status"])) + "\n")
EOF
# N = label workers in the canary label run (see its "start" event in logs/progress.jsonl)
python RIND3D_Dataset100K/scripts/estimate.py --report /data/r3d_canary/REPORT.json --cost-samples canary_label_costs.jsonl \
       --workers N --scaling /local_scratch/r3d_scaling/SCALING.json --out ESTIMATE.json
```

**Do not start production until** `docs/GO_NO_GO_100K.md` has been updated with ESTIMATE.json and the decision has been
made.

## 5. Production

```bash
# use a separate root per campaign; slots can be split in disjoint ranges (one coordinator per root)
python RIND3D_Dataset100K/scripts/run_node.py --root /data/r3d100k --slots 0:100000 --gpus all \
       --label-cpu auto --rss-gb 6 --label-gpu-per-gpu 2 \
       --prior-manifest RIND3D_Dataset100K/handoff/prior/release_1k_rate_stats.jsonl  2>&1 | tee -a /data/r3d100k/node.log
```

**Behaviour:**
* **Physical stage.** One worker per GPU (`--phys-per-gpu`). Epochs of 1000 slots: epoch e is planned when every slot
  of epochs ≤ e − 1 is final. Physical failures retry the same attempt; quota-unfilled scenes are rejected and redrawn
  at attempt + 1, up to 3 attempts, all recorded.
* **Label stage, concurrent.**
  * L256 tasks go first to GPU-assisted workers (`--label-gpu-per-gpu` per device; GPU pass with `gpu_cap` queue
    entries; raise `--gpu-cap` on 96 GB devices after benchmarking).
  * L64 and L128 tasks go first to CPU workers.
* **Worker count.** `--label-cpu auto` uses the CPUs in the process affinity / cgroup, minus GPU and physical workers,
  minus `--reserve-cpus` (2), capped by MemAvailable / `--rss-gb`.
* **Per-task limits.**
  * Wall clock: 1800 / 3600 / 7200 s for L64 / L128 / L256 (contract `labels.timeouts_s`).
  * RSS: `--rss-gb` per worker.
  * A crash, timeout or memory overrun quarantines the task in `labels/failures.jsonl`. The slot keeps an explicit
    status; no label is ever invented. Retry later with `--retry-failed` and, if needed, more memory.
* **GPU errors.** Any GPU error, including out-of-memory, falls back to the exact CPU engine, recorded as `fallback`
  in the label metadata.

**Stopping and resuming:**
* **Stop safely:** `touch /data/r3d100k/STOP`. No new tasks are started; running tasks get `--stop-grace` (600 s);
  then the coordinator exits.
* **Emergency stop:** `kill -TERM <coordinator pid>`. It is safe; interrupted tasks are recomputed.
* **Resume:** `rm /data/r3d100k/STOP` and rerun the same command. The manifest, plans and journals are authoritative;
  finished scenes and labels are never recomputed.

**Monitoring:**
* `tail -f /data/r3d100k/logs/progress.jsonl` shows scenes/h, labels/s, queue sizes, busy workers and RSS.
* `python RIND3D_Dataset100K/scripts/report.py /data/r3d100k` shows coverage, layer balance (raw and ω-weighted),
  costs and failures.
* `python RIND3D_Dataset100K/scripts/verify_root.py /data/r3d100k --labels` is a read-only integrity check.

**Per-scene completion:**
* `manifest/physical.jsonl`: status accepted / rejected / failed, with `final`.
* `labels/scenes/<shard>/<slot>.labels.npz` exists once every informative slot is resolved. Its `status` maps every
  (window, source) to certified / certified_loose / uncertified / engine_failure / timeout / memory_limit /
  not_computed.

**Multiple nodes:** give each node a disjoint `--slots` range and its own root, with separate controller histories.
Do not run two coordinators on one root.

## 6. Root layout and certification metadata

The layout is specified in `rind3d_100k/storage.py`:
* `scenes/<shard>/<slot>.npz`: one atomic container per scene. It holds the windows, packed responses and SOLID,
  geometry, rows, sources and selection logs.
* `labels/journal/*.r3lj`, `*.idx`: append-only, checksummed label journals.
* `labels/scenes/...`: consolidated labels per scene.

**Each label's meta** records:
* `tv_bound`, a rigorous bound for the declared target, given the engine's float-filtered predicates;
* `status`: certified if tv_bound ≤ 0.05, certified_loose if ≤ 0.10, uncertified otherwise;
* engine version, engine (`cpu` / `gpu`), fallback, CPU and wall seconds, records, engine statistics and queue
  capacities.

**Training must weight every example by ω** (`loader`: `weight`, `weights`), as explained in docs/SELECTION_LAW_100K.md.

## 7. Troubleshooting

| symptom | cause / action |
|---|---|
| `NVRTC not found` | `source r3d_env.sh` (sets RIND_NVRTC_LIBRARY) |
| `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` | NVRTC newer than the driver: rerun setup_env.sh (it pins NVRTC to the driver's CUDA) |
| `Failed to find CUDA headers` (cupy) | the environment lacks the `cupy-cuda12x[ctk]` / cuda-pathfinder wheels: reinstall from requirements-lock.txt |
| GPU OOM in label workers | handled (CPU fallback); lower `--label-gpu-per-gpu` or `--gpu-cap` |
| workers killed with kind=memory | raise `--rss-gb` or lower the worker count; then `--retry-failed` |
| many `timeout` failures at L256 | expected heavy tail; leave them quarantined; later retry with `--retry-failed` and fewer, larger workers |
| `root holds a different contract` | wrong `--root` for this contract |
| numba compile storms at start | the first tasks compile the engine (≈ 30–60 s per process; cached under `<root>/.cache/numba`) |
| disk filling up | scenes ≈ 0.36 MB each, labels ≈ see REPORT.json `bytes_per_label` *[laptop canary]* |
