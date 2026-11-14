# Report C — 16 × H20 architecture, scheduling, memory, failure handling, scaling

**Status of the numbers:**
* The design is implemented and tested on the laptop (1 GPU).
* Nothing has been measured on an H20 yet. Every H20 statement below is a **projection**, or a **test to run**
  (`bench_scaling.py`).

## 1. Workload

| item | count (100K) | per item [laptop] |
|---|---|---|
| physical scenes | 100,000 | 3.7 s on one GPU worker (≈ 0.6 s GPU tiles, ≈ 1.3 s CPU serialisation, plus probes, hull and Python) |
| windows | 1,200,000 | |
| informative posterior labels | ≈ 2.94 M (29.4 per scene, dev canary) | heavy-tailed: see Report D |
| source slots (incl. uninformative, kept as `not_computed`) | ≈ 3.7 M | 0 |

Labels dominate by two to three orders of magnitude. The architecture is built around keeping CPU cores and GPU-pass
workers saturated with independent label tasks, while the GPUs render scenes alongside.

## 2. Process model (`rind3d_100k/coordinator.py`, `pool.py`)

```
run_node.py (1 coordinator per node and root)
 ├─ physical workers: phys_per_gpu × G     CUDA_VISIBLE_DEVICES=g; certified renderer; writes its own scene container
 ├─ GPU-assisted label workers: label_gpu_per_gpu × G   TPE with the exact GPU pass (cupy) on device g
 └─ CPU label workers: auto = CPUs(affinity, cgroup) − GPU-bound workers − reserve, capped by RAM / rss_gb
```

* **Coordinator.** It is the only writer of the manifest, journals, failures and progress files. Workers communicate
  over private pipes, one task at a time, so there is no unbounded buffering.
* **No cross-GPU communication.** Every task is independent. Dynamic dispatch through idle workers gives natural work
  stealing, and long tails never block other tasks.
* **Queues and routing.**
  * L256 tasks go first to GPU-assisted workers, where the GPU pass pays off for large item fronts.
  * L64 and L128 tasks go first to CPU workers.
  * Any idle worker takes any task.
* **Epochs.** Physical tasks run epoch by epoch, 1000 slots each. The plan for epoch e needs every slot of epochs ≤ e − 1
  to be final (barrier). The barrier only delays physical work: label workers keep draining earlier scenes, so the
  barrier costs little.
* **CUDA contexts.** Each worker process has one context. Geometry is resident per scene in physical workers (DeviceGeometry
  cache of 2 geometries). Label workers allocate per window and free the cupy pool after each window.
  * Option C (a persistent GPU service fed by CPU workers) was prototyped in Phase 2: a threaded GPU server, correct,
    2× with 6 threads. It is not used, because process-per-worker is simpler and crash-isolated.

## 3. Memory budgets

| resource | budget | control |
|---|---|---|
| label worker RSS | ≈ 1.0–1.8 GB typical, up to ≈ 4 GB on heavy L256 tasks [laptop] | `--rss-gb` kills and quarantines an overrunning task (kind=memory) |
| physical worker RSS | ≈ 0.6 GB | |
| GPU memory per GPU-assisted label worker | ≤ gpu_cap × (bytes per entry) plus scratch; ≈ 0.5–1.5 GB at gpu_cap = 2²² [laptop] | `--gpu-cap`, `--label-gpu-per-gpu`; OOM falls back to the exact CPU engine |
| GPU memory per physical worker | ≈ 0.6 GB | |
| coordinator RAM | O(#scenes) manifest records plus the journal index (≈ 100 B per label) | |
| disk | scenes ≈ 0.36 MB each (≈ 36 GB); labels: see Report D | atomic writes; checksummed journals |

**96 GB per H20 [projection].** VRAM is not the constraint. Even 8 GPU-assisted workers per GPU at gpu_cap = 2²⁴ would
fit. The constraints are CPU cores (TPE stage control is CPU-side) and host RAM (≈ 2 GB per label worker). Raise
`--label-gpu-per-gpu` and `--gpu-cap` only as far as `bench_scaling.py` shows a gain.

## 4. CPU-aware allocation

**Do not assume the laptop layout** (laptop: 11 CPU plus 4 GPU-assisted workers on 8C/16T).

**Rule at start-up:**
* CPU label workers = available CPUs − (physical + GPU-assisted workers) − 2.
* This is capped by MemAvailable / rss_gb.
* Hyper-threads count as CPUs. On the laptop, 15 workers on 16 threads was the measured best (1K, Phase 2).

**If CPUs are scarce** (for example 64 cores for 16 GPUs):
* GPU-assisted workers become the main producers.
* GPU-assisted workers are themselves CPU-bound in stage control: Phase 2 measured machine-level 1.55× from GPU help on
  the laptop.
* Labels/s then scales with CPU cores, not GPUs. `bench_scaling.py` shows this directly: efficiency drops as G grows
  while the CPU budget stays fixed.

## 5. Failure handling (lessons of the 1K campaign)

| failure | handling |
|---|---|
| native crash (abort, segfault) | the worker is replaced; the task is quarantined (`labels/failures.jsonl`, kind=crash); its slot status is `engine_failure` |
| hang or extreme tail | per-side wall limits of 1800 / 3600 / 7200 s; kind=timeout; slot status `timeout` |
| memory overrun | RSS polling every 5 s; kind=memory; slot status `memory_limit` |
| GPU OOM or any CUDA error | exact CPU fallback in the same task; recorded in meta.fallback |
| record-buffer overflow | engine v3.1: capacity-checked writes plus exact rerun with a larger buffer |
| physical failure | retry the same attempt (identical streams) up to 2 times; then final failed, counted in the rejection bound |
| coordinator crash or power loss | manifest, plans and journals are authoritative; journal tails are checksummed; scene files are atomic; rerun resumes |
| duplicate work | task keys; the journal index; the first valid record wins in consolidation |

## 6. Storage (`rind3d_100k/storage.py`)

* **One container per scene** (≈ 0.36 MB), 1000 per shard directory: about 10⁵ files in total.
* **Labels go to append-only journals** (one per coordinator run), not to 3M tiny files. They are consolidated per
  scene into one file when the scene completes.
* **Recommended placement:**
  * keep the root on local NVMe during production;
  * periodically sync to durable storage — whole shard directories, plus journals rotated by run;
  * `verify_root.py` checks every container's sha256 and every journal checksum before and after a transfer.

## 7. Scaling expectations [projection, to be measured]

* **Physical stage.** It is GPU- and CPU-light and embarrassingly parallel per scene. Expected near-linear in G up to
  the CPU budget (≈ 1.3–2 s CPU per scene for serialisation and probes).
* **Labels.** These are independent tasks, so throughput is Σ over workers of 1 / mean task time. Efficiency loss
  comes from:
  1. CPU oversubscription;
  2. memory bandwidth contention in the CPU engine;
  3. the GPU pass's per-segment launch overhead with many workers per GPU;
  4. long-tail L256 tasks at the end of the campaign.
* **H20 FP64.** H20 FP64 is reported to be low relative to H100. Our kernels use value-aware FP32 with FP64 only for
  uncertain or contributor edges (Phase 2: 99.99% FP32). The renderer uses FP64 directed rounding.
  `probe_hardware.py --bench` measures FP64 and FP32 FMA throughput and renderer tile throughput directly. Do not
  extrapolate from datasheets.
