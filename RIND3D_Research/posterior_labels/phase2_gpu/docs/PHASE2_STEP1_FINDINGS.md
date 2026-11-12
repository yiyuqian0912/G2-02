# Phase 2, step 1 — compute environment, profiling and GPU PoC (2026-11-11, evening)

## Compute environment

* **Colab attempt.** A T4 runtime was allocated (Tesla T4 15 GB, 2 vCPU Xeon @ 2.00 GHz, 13.6 GB RAM, Python
  3.13, CUDA 13.0) and the source/bundle pushed (43 files, verified).
  - The VM-side Drive mount failed twice. The helper's consent URL was first hidden by output buffering, the
    helper then hit a KernelConnectionError, and after the user's approval it reported
    `mount_authorization_failed`.
  - The helper also held the kernel, which blocked process starts (`request_not_submitted`).
  - At the user's request the runtime was released, after about 20 min of T4 time with no workload run. Phase 2
    continues **locally**.
* **Local machine.** 16 logical CPUs, 31 GiB RAM, NVIDIA GeForce RTX 3060 Laptop GPU (6 GiB, 30 SMs, driver
  615.78, CUDA UMD 13.4).
* **Phase 2 environment.** Isolated `.venv`: numba 0.68.1, numpy 2.5, scipy 1.18, CuPy 14.2 + CUDA 12.9 pip
  wheels. Raw float64 kernels were verified.
* **Phase 1 stragglers.** They keep running, reniced to 19.

## Profiling the frozen engine b4 (12 declared validation windows; `outputs/profile_b4_local`, `outputs/depth_profile`)

* **Where the time goes.** Cost is dominated by entry evaluations, at 145–270 ns per component evaluation in the
  engine (70 ns in a tight loop of the same predicate, so the engine overhead is about 2–3×).
* **Where in the tree.** Evaluations are spread over the first levels of every root's subtree:
  - depth 0 (root regions) holds only 0–15%;
  - 44–100% of evaluations happen by depth 5, and up to 100% by depth 10.

  Re-evaluating surviving receivers after each split is therefore the bottleneck, not the first pass over the
  item list.
* **Size.** Side-256 windows have up to 1.3×10⁵ top items and up to 3.5×10⁵ evaluations per region.

## GPU PoC (`code/gpu_eval_poc.py`, `outputs/GPU_EVAL_POC.json`)

* **What was ported.** The exact pencil predicate: same float64 operation order, `--fmad=false`, same margins.
* **Correctness.** On 1.43×10⁶ real (root region, item, component) evaluations, GPU decisions are
  **identical** to the CPU engine: 0 status and 0 undecided-count mismatches.
* **Throughput.** 7.7–9.2×10⁷ evaluations/s on the RTX 3060 Laptop GPU, against 1.4×10⁷/s on one CPU core, i.e.
  **5–6.5× one core**. Consumer GPUs run FP64, and FP64 division, at 1/64 of their FP32 rate.

## Consequences for the design

1. **No speedup from a straight FP64 port on this GPU class.** Exact FP64 evaluation on a consumer GPU is not
   worth it alone. Two routes remain:
   * a **filtered FP32 predicate** with a rigorous forward-error bound, falling back to FP64 when the margin is
     too small (the classic static-filter technique; it can stay certifiable);
   * data-centre GPUs with full FP64 (A100: about 30× the laptop GPU's FP64).
2. **Batching matters more than kernel speed.** Per-region batches are small (10³–10⁵). Level-synchronous
   processing, i.e. all regions of one tree depth evaluated in one batch, is needed for any GPU to pay off.
3. **Most promising CPU-side change, measured next.** Cache each item's pencil ranges per window-side component.
   They depend only on the item and the component, not on the region. Each re-evaluation then reduces to range
   comparisons; the GPU can precompute the cache for all items in one batch.
4. **Target fixes are required before any v0.3 numbers:**
   * remove the source-outside-window exclusion for `sb-global-v0.3`;
   * add explicit solid/prior-support exclusion for windows without lit receivers (all-dark, no-FREE);
   * add natural-window handling (no-FREE label = prior).
