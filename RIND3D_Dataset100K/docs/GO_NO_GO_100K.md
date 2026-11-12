# Report F — RIND3D-100K go / no-go (provisional: the H20 server has not been accessed)

**Every H20 number here is a projection from laptop measurements.** It must be replaced by the server bring-up
measurements (README_H20 §4 → ESTIMATE.json) before any production decision.

## Answers

| question | answer |
|---|---|
| **Is local-only response production faster?** | **Yes.** Exact local rendering is 2–3× faster than full-field on the laptop: 3.7 s per scene vs 3.3–14.3 s, with bit-identical outputs (0 mismatches in 2.9×10⁹ bits). No global field is computed or stored. Physical 100K ≈ 103 laptop-GPU-worker-hours, ≈ 6 h on 16 GPUs at one worker each and laptop-equal speed [projection]. **Physical production is not the bottleneck.** |
| **Can the current tau-pencil solver be scaled sufficiently on H20?** | **Not to 24–72 h, unless the host has ≳ 500 laptop-equivalent CPU threads.** The engine is CPU-bound: stage control in numba. The GPU pass helps (1.55× machine-level on the laptop) but does not make the labeller GPU-bound, so 16 × 96 GB of GPU memory is mostly idle. |
| **Is the 100K target practical?** | **Physically yes; label-wise it is a multi-day campaign.** ≈ 33,400 laptop-thread-hours of label work for 2.94M labels: ≈ 5.4 days on a 256-thread host (≈ 10 days if the tail runs to its limits), ≈ 3.6 days on 384 threads [projection]. |
| **Estimated end-to-end time** | ≈ max(physical ≈ 6 h, labels ≈ 3.6–10 days). The stages run concurrently. |
| **RAM** | label workers: 1–2 GB typical, up to ≈ 5 GB on heavy tasks. Plan for ≈ 2 GB × workers + headroom (e.g. ≥ 512 GB for 256 workers), or cap with `--rss-gb`. |
| **Storage** | scenes ≈ 36 GB (0.36 MB each) + labels ≈ 140 GB (47 KB each, per-cell format) + journals; **≈ 0.2 TB total** [projection from canary samples] |
| **Expected coverage** | ≥ 98.6% of informative labels certified in tier 1 (canary: 142 / 144 completed, all 142 certified; the other 2, an L64 and an L128, hit their 1800 s / 3600 s limits and were quarantined); the rest stay explicit (`timeout` / `engine_failure`) for a tier-2 pass |
| **Dominant remaining bottleneck** | posterior labelling: CPU-bound TPE with a heavy tail. ≈ 1–2% of tasks take ≥ 30 min, and a few take hours, mostly thin-slab, barrier and cluster scenes. |
| **Is more algorithmic work needed before full production?** | **Yes, for any timeline under ≈ 1 week.** The exact two-phase labeller is validated exact on 36 windows and gives ≈ 3× on the 100K mix. A broad-posterior statistical tier and GPU-native phase 2 are the next steps. |

## Recommended decision path

1. **Now.** Ship the handoff package (tested: clean extract → setup_env → SMOKE PASS).
2. **On server access.** Run README_H20 §4 steps 1–7: probe, smoke, single-GPU bench, 1/2/4/8/16-GPU scaling, a
   256-scene canary with a 600-label stratified sample, and `estimate.py`. Replace every projection here with the
   measured ESTIMATE.json.
3. **Decision rule:**
   * if the measured label projection is ≤ the acceptable campaign length, GO with staged production:
     1. 2,000 scenes;
     2. verify and report;
     3. continue in 10,000-scene stages, with verify_root after each;
   * otherwise run only the physical stage at full scale (cheap, ≈ hours) and label in stages, while the two-phase
     engine is integrated.
   Never weaken the contract or the certificate to meet a date.
4. **Physical-only first is safe.** Labels can be added later without touching the scene containers: they live in
   separate journals.

## What is measured versus projected

**Measured on the laptop:**
* local / full-field equivalence;
* physical cost per scene;
* controller balance (raw 27.3 / 24.6 / 24.1 / 24.1%; ω-weighted 30.5 / 25.3 / 23.1 / 21.1%);
* coarse-screening bias;
* label costs per side and tail;
* storage per scene and per label;
* the handoff build and smoke test in a clean directory.

**Projected:** every H20 throughput and every duration in hours or days for 16 GPUs.
