# Report D — Posterior-label cost under the 100K workflow (measured on the laptop)

> **Historical cost basis (TPE, contract v1).** The production labeller under contract v2 is FastLabel (`../RIND3D_Posterior/docs/FASTLABEL_CORE.md`). Its cost is measured in the canary (`docs/H20_CANARY_RUNBOOK.md` §9).

**Setup:**
* engine: TPE v3.1, tier 1 (nq_cap 2²⁵, ne_cap 2²⁶, gpu_cap 2²²);
* workers: 11 CPU + 4 GPU-assisted on the laptop (8C/16T, RTX 3060 6 GB); this is the 1K production layout;
* inputs: exact local responses from the 100K physical stage plus global geometry rows;
* sample: a stratified random label sample of the 96-scene dev canary (`run_node --label-sample 48,48,48,7`), 144
  tasks;
* plus the 32 labels of canary scene 0 and the 14 reference fixtures (bench_labels).

CPU seconds are the worker process's CPU time under full machine load; hyper-threads are shared. **Warm costs**
exclude each worker's first task: in a fresh worker, that task includes the one-time numba JIT compile, ≈ 40–60 s.
The coordinator and bench_labels now pre-compile both engines before starting the pool.

## 1. Results of the 144-task stratified sample

| side | completed | certified | warm CPU-s mean | p50 | p95 | max | incl. first-task compile: mean |
|---|---|---|---|---|---|---|---|
| 64 | 47 of 48 | 47 | **10.8** (n = 36) | 5.7 | 33.5 | ≈ 144 | 24.6 |
| 128 | 47 of 48 | 47 | **36.7** (n = 47) | 27.1 | 110.9 | 184.9 | 36.7 |
| 256 | 48 of 48 | 48 | **75.2** (n = 44) | 26.6 | 318.8 | 503.3 | 74.7 |

**Engines at L256:** GPU-assisted, 28 tasks: CPU-s mean 70.8. CPU, 20 tasks: 80.2. The GPU pass offloads item
evaluation, but stage control stays on the CPU.

**Tail:**
* 070/03/1 (L64, barriers, m = 4): **timed out at its 1800 s limit**; quarantined as kind=timeout.
* 089/06/0 (L128, occluder_clusters, m = 1): **timed out at its 3600 s limit**; quarantined as kind=timeout.
* Completed labels: 142 / 142 certified (tv_bound ≤ 0.05). Tier-1 coverage is 142 / 144 = 98.6%. No GPU fallbacks, crashes or memory overruns. The run ended cleanly: supervision released both workers at their limits.

**Storage:**
* 47 KB per label (per-cell, delta-encoded; was 234 KB with the raw polytope pieces).
* Scenes: 0.36 MB each.

## 2. Per scene and for 100K

| quantity | value |
|---|---|
| informative labels per scene | 29.4 (canary: 2822 / 96); side shares 0.334 / 0.329 / 0.337 |
| mean warm CPU-s per label (production mix, completed tasks only) | **≈ 41** (cost shares: L64 9%, L128 29.5%, L256 62%) |
| CPU-s per scene | ≈ 1,200 (≈ 20 CPU-min) + tail |
| 100K labels | 2.94 M |
| 100K label work | **≈ 33,400 laptop-thread-hours**, completed tasks only |
| tail sensitivity | if 1.4% of tasks run to their limits (1800 / 3600 / 7200 s), the mean rises to ≈ 75 s → ≈ 61,000 thread-hours; with the limits, the cost is bounded and those slots are quarantined (coverage ≈ 98.6% in tier 1) |
| laptop throughput | 15 workers / ≈ 43 s wall = 0.35 labels/s, i.e. ≈ 97 days for 100K on this laptop |

## 3. What the labels need from a server [projection]

The machine needs ≈ 33,400 laptop-thread-hours of labeller work.
* **Required sustained rates:** 11.6 labels/s for 72 h, or 34.7 labels/s for 24 h. That is 33× and 99× the laptop.
* **Scaling.** The engine is CPU-bound: numba stage control. GPUs help, but at the machine level on the laptop the
  GPU pass gave 1.55× (Phase 2).
* So throughput scales mainly with **host CPU threads**, assuming laptop-equivalent speed per thread:

| host threads (laptop-equivalent) | labels/s | 100K labelling, completed-task mean | with tail at limits |
|---|---|---|---|
| 128 | 3.1 | 11 days | 20 days |
| 256 | 6.2 | 5.4 days | 10 days |
| 384 | 9.4 | 3.6 days | 6.6 days |
| 512 | 12.5 | 2.7 days | 5.0 days |

The real per-thread speed and the GPU-assist factor on the H20 server are unknown. `bench_labels.py` plus
`estimate.py` on the server replace this table with measurements.

## 4. Heavy tail

**The tail is the main uncertainty.** In the 1K release, 5 tasks on one thin-slab scene never finished (record-buffer
crash, then > 16 min). Here 2 of 144 tasks exceeded 30 min.

**Recommended production policy:**
* keep the tier-1 limits;
* quarantine;
* run a **tier-2 retry pass after the main campaign**, with fewer, larger workers and longer limits;
* report the unresolved fraction explicitly.

Never extend the limits for the main pass: one task must not block a worker for hours while thousands wait.

## 5. Algorithmic headroom (not a prerequisite, but decisive for the timeline)

The exact two-phase labeller (`RIND3D_Posterior/experiments/twophase`, research checkpoint) works in two steps:
1. phase 1 on a point subset of receivers gives an exact outer bound B ⊇ F;
2. phase 2 runs the full engine restricted to B.

On 36 stratified 1K windows it is **exact** (TV vs reference ≤ 3e-16), and it is:
* 5.4× faster at L256 (20–35× on concentrated posteriors);
* 2.5× at L128;
* 0.9× at L64.

**Weighted by the 100K cost mix** (warm cost shares L64 9%, L128 29.5%, L256 62% from §2), this is **≈ 3×**
(4.6× with the 1K mix, where L256 was 92% of the cost). That is ≈ 5.4 → ≈ 1.8 days on a 256-thread host
[projection].

**Reaching 24–72 h for 2.9M labels also needs:**
* a broad-posterior tier: the statistical certificate on q_B in research notes §4;
* the GPU pass in phase 2;
* lower fixed overhead per label (stage 1 and Python item construction, 0.3–1 s per label).

These are the next research steps (RESEARCH_NOTES.md).
