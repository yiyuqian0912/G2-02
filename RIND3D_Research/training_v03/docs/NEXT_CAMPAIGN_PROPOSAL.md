# Next-campaign proposal: `rind3d-production-sb-10k-v0.3` (NOT LAUNCHED)

**Status.** This is a proposal only. It needs explicit authorization and an approved compute and storage budget. Config: `configs/next-campaign-sb-10k-v0.3.json`. No 1K, 10K or 100K production has been launched.

## What 100K means (decision)

**Recommendation: separate releases, SB first.** "100K" means `rind3d-sb-100k`, a single-source binary release of 100,000 scenes, built in versioned stages (1K → 10K → 100K). MA is a separate, independently sized release, `rind3d-ma-*`, decided after a Student study on the MA canary data and the 1K SB results.

The reasons:
1. The two task profiles have different targets: a single point versus an unordered set with K.
2. MA costs 3.6× the GPU time and 5.4× the field storage per scene [M canary].
3. A combined release would tie the SB timeline to MA risks.
4. Separate releases keep each one's prior, selection law and splits simple.

Costs for all three definitions (SB-only, combined 50/50, separate SB-N + MA-N) are tabulated in `RIND3D_TRAINING_SCALE_COST.json`.

## Stages

1. **Already done in this assignment (no physics):** the SB 1K training candidate from v0.1. Export `v01-sb-v03` (1000 scenes × 256 windows; `configs/training-sb-1k-from-v01-v0.3.json`).
2. **10K SB (this proposal).**
   - Runner: the v0.2 runner with the frozen v0.2 physics contract, `profile_schedule = ["SB"]`, a private 256-bit `seed_file`, and resumable sessions.
   - Gate: the first 216 slots form a canary. The v0.2 gates C1–C8 apply, plus the v0.3 tests: leakage audit, interop tests, P102 prefix and weight validation.
   - After the canary gate: bulk generation, then validation V1–V11, the obs-index-v0.3 P256 index with a private index key, the two-view export, the leakage audit and the loader benchmark.
3. **100K SB.** Only after the 10K release is accepted and a 10K training result shows that more scenes help (the ICC study predicts that scenes, not windows, are the limiting factor). It needs a multi-GPU or cloud budget.

## Estimated cost of 10K SB

Measured on this laptop, extrapolated linearly [X]. Full table: `RIND3D_TRAINING_SCALE_COST.json` → `SB-only/10000/P256`.

| Item | Cost |
|---|---|
| Certified physics generation | ≈ 26 GPU-hours (9.5 s/scene mean, one RTX 3060 Laptop) |
| Validation V1–V10 | ≈ 3 CPU-hours |
| obs-index-v0.3 P256 | ≈ 91 CPU-hours (≈ 8 h wall on 12 CPU workers); overlappable with GPU generation |
| Two-view export | ≈ 3.5 h on 12 workers (the 1000 v0.1 SB scenes took 21 min) |
| Storage | ≈ 11 GB fields + ≈ 19 GB student shards + ≈ 1.6 GB index |

## Acceptance (frozen before launch)

- The v0.2 canary gates C1–C8 apply unchanged.
- Training-layer gates:
  - T1: leakage audit L1–L10 passes on every split;
  - T2: integration tests 1–10 pass;
  - T3: P102 prefix identical;
  - T4: weighted mixture statistics within 25% (relative) of the rate-stream population per side;
  - T5: no-FREE windows are flagged, and their Teacher mismatch is NaN or `no_free`;
  - T6: loader-fed training throughput is ≥ 90% of the GPU-resident rate at L64.
- No gate may be weakened after the results are seen.
