# RIND3D Phase 3 summary (2026-11-12): production optimization, consolidation, geometry, 1K v1

## What was built

| Component | Location | Status |
|---|---|---|
| Maintained labeller | `RIND3D_Posterior/` (`rind3d_posterior` 0.2) | Bit-exact regression tests pass on CPU and GPU |
| 1K pipeline | `RIND3D_Dataset1K/` (`rind3d_1k`) | Physical stage, label stage, balancing controller, validators, loader, release statistics, production driver |
| Frozen contract | `RIND3D_Dataset1K/contract/RIND3D_1K_CONTRACT_v1.json`, `docs/CONTRACT_1K.md` | Frozen v1 |
| Sampling-law audit | `RIND3D_Dataset1K/docs/SELECTION_LAW_1K.md` | Per-channel posterior theorem under ω-weighting; bounded terms stated |
| Studies | `docs/OPTIMIZATION_REPORT.md`, `docs/RENDERING_BENCHMARK.md`, `docs/GEOMETRY_DIVERSITY.md` | Measured, including negative results |
| Research archive | `RIND3D_Research/` | Phase 1/2, v0.2, v0.3 code, docs and evidence |

## Key measured facts

* **Physical stage at 1024³** costs ≈ 23–40 s per scene. Strategy A (certified CUDA fields + exact count index) beats local rendering once rate streams and thinning are counted.
* **Labels at 1024³** cost 6 / 24 / 107 worker-s on average for sides 64 / 128 / 256. L256 has p95 439 s and max 555 s.
* **The balanced 1K is projected at ≈ 141 h** of labels on this machine. This is the only blocker to completion within the session.
* **Queue capacity is a certification issue.**
  * At 2^24, 3.4% of early L256 labels were uncertified.
  * At 2^25 (tier 1), the 96-label stratified sample was 96/96 certified.
  * At 2^26 (tier 2), 2 of the 3 failures were recovered.
* **GPU, exact pass.**
  * Single window: 7.1× faster.
  * Machine level: 1.55× with 6 GPU + 10 CPU workers.
  * The value-aware FP32 kernel is exact but only 1.4–1.7× faster at kernel level.
  * MPS is unavailable on this GPU.
  * The single-process threaded GPU server is correct, but not yet benchmarked on an idle machine.
* **Early stopping by node budgets brings no speedup.** The relaxed-set bound is not useful for tiny feasible sets.
* **Balancing controller (simulated on canary statistics).** It reaches an exact 25/25/25/25 m-histogram, ≈ 25% for each m on every side, with k = 1..4 all present (15% floor each). Informative labels: ≈ 160k.
* **Geometry.** Five polyhedral curved/irregular families cost 2.7–5× more per label at 128³, and curved walls add an extreme tail. They are excluded from 1K v1 and kept for future releases.

## Open questions

1. **Heavy-tail L256 cost.** Dual-tree expansion fronts reach 10⁷ entries. Candidates: receiver-block ordering, front-bounded source subdivision.
2. **Uninformative-channel posteriors.** Their feasible sets are large, at 8–500× the cost. They are deferred.
3. **A formal real-arithmetic proof of the engine predicates.** Today's guarantee rests on margins plus software equivalence.
4. **Training with ω weights.** Reported bias of unweighted training: 0.02–0.11 TV (v0.1 analogue).
