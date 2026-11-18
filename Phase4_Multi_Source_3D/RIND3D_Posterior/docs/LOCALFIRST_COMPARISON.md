# Tau-pencil, FastLabel and local-first labelling, compared on v3 windows

**Setup.**
* **Tasks.** 54 labelling tasks of contract v3. Each is one (scene, window, interface channel): two per
  (family, L) group, over 9 families × L64/L128/L256. They come from the 72-scene v3 canary (thin slabs ≥ 2 voxels).
* **Hardware.** A laptop: RTX 3060 Laptop GPU 6 GB, Ryzen 7 5800H, one thread per labelling process.
* **Same target everywhere.** Every method labels the same world, window and channel, on the same 128³ grid, with
  the same target: the uniform law on F(G, X), source-excluded.
* **Measurement.** Each (task, method) runs in a fresh process, so wall time and peak RSS belong to that label.
  Six CPU processes ran in parallel; FastLabel ran alone on the GPU.
* **Code.** `experiments/localfirst/compare.py` and `summarize.py`. Data: `RIND3D_100K_v3_work/cmp_v3/`, outside Git.

| method | certificate | certified (TV ≤ 0.05) | wall s, p50 / p95 / max | peak RSS above import baseline, MB p50 / max | TV of its point estimate to tau-pencil's, p50 / max |
|---|---|---|---|---|---|
| **tp** tau-pencil reference (`label_window`, CPU) | deterministic | 54/54 | 12.4 / 140.0 / 172.3 | 194 / 1687 | — |
| **lf** local-first: outer bound from local-geometry receivers, then exact tau-pencil on B | deterministic, the same engine certificate | 54/54 | 2.2 / 14.4 / 57.7 | 184 / 1783 | 0.0000 / 0.0000 |
| **rand** as lf, with FastLabel v3.4's receiver choice | deterministic | 54/54 | 1.8 / 14.2 / 55.6 | 193 / 1789 | 0.0000 / 0.0000 |
| **fl** FastLabel v3.4 (`label_window_fast`) | statistical, ε = 0.05, δ = 0.01 / 4.8 M | 54/54 | 1.6 / 2.7 / 4.1 | 354 / 504 | 0.0000 / 0.0141 |

| L | tasks | tp wall p50 (s) | lf | rand | fl | lf speed-up over tp (median of per-task ratios) |
|---|---|---|---|---|---|---|
| 64 | 18 | 7.0 | 1.7 | 1.7 | 1.9 | 2.4× |
| 128 | 18 | 27.5 | 3.2 | 3.2 | 1.6 | 4.5× |
| 256 | 18 | 31.4 | 2.2 | 1.7 | 1.4 | 7.9× |

## Readings

* **A deterministic certificate at a fraction of the reference cost.** The pathway "receiver-subset outer bound B, then
  the exact tau-pencil phase 2 restricted to B" certified all 54 tasks.
  - It gives the same deterministic tv_bound as the reference, and point estimates identical to tau-pencil's to
    1e-4.
  - It is 2.4–7.9× faster at the median, and more so at L256, where tau-pencil spends most of its time refuting
    candidate boxes that B already excludes.
  - Correctness rests on Proposition 8 of `docs/LOCAL_INVERSE_GEOMETRY.md`: dropping receivers is safe, and
    restricting to B ⊇ F loses no feasible source. The assumptions are the reference engine's own: A1, sound
    float-margin decisions; A2, the outer-bound filter boxes contain the pieces.
* **Choosing receivers from the local planes adds nothing measurable to speed.** This is a statement about receiver
  selection (hypothesis I of `LOCALFIRST_AUDIT.md`) only. It does not test source-region reduction by local
  constraints (hypothesis II), which this pathway never applies.
  - The two strategies pick almost disjoint receiver sets (Jaccard median 0.006).
  - Their outer bounds B cover the same 128³ cells (median cell-count ratio 1.00).
  - In volume, local selection's B is smaller at the median (ratio 0.42; at least 2× smaller in 29 tasks).
  - The exact phase's time does not depend on that volume (Spearman 0.01). It is set by certifying F against all
    receivers: even F itself as the filter gives only 1.10× at the median (audit §3.3).
  - The speed-up comes from computing exactly inside B, not from local geometry.
* **Against FastLabel.** The deterministic pathway matches FastLabel at the median (2.2 vs 1.6 s), but its tail is
  about 5× longer (p95 14 vs 2.7 s). FastLabel's statistical certificate stops after about 1,500 exact point tests,
  whereas the exact phase 2 must resolve every BSP region of B.
  - Measured on these tasks, FastLabel's point estimates are within TV 0.014 of the exact reference (bound 0.05).
  - A production choice between the two is therefore a trade: a deterministic certificate with a heavier tail,
    against a statistical certificate with a flat cost.
* **Memory.** About 0.2 GB above the import baseline at the median for the CPU engines (up to about 1.8 GB on the
  largest L256 tasks), and about 0.35 GB for FastLabel with its GPU verifier.
* **The heuristic local band.** This is the X-only band of `localfirst.local.local_band_cells`, never used for pruning.
  - It contained the whole label and the true source in all 54 tasks. But it is wide: a median 177× the label's
    support.
  - In the 9-task smoke run it excluded the true source once (a thin_slabs L256 window under the v0.1 slab law).
  - Measured H1 violations occur in 43% of v3 channels (LOCAL_INVERSE_GEOMETRY.md §3b), so 54/54 here is not a safety
    argument.
* **Local planar structure.** Rank of the local planar components: rank 0 in 2 tasks, rank 1 in 22, rank 2 in 19, and
  rank 3 in 11. Point-like localization from one window is the minority case.

## Recommendation

* **Keep FastLabel as the production engine.** It has the flat cost tail, and its certificate is already audited.
* **Offer the deterministic "B, then exact on B" pathway.**
  - As a certified fallback, and as a replacement for the full tau-pencil reference in regression and audit work,
    where a deterministic certificate is wanted. It costs about 1/5 of the reference.
  - The function is `rind3d_posterior.localfirst.label_window_localfirst`, with `select="random"` or `"local"`. Both
    give the same runtime.
* **Testing source-region reduction requires a different experiment.** `LOCALFIRST_AUDIT.md` §5 designed it, and
  `LOCALFIRST_E123.md` reports it: certified edge-silhouette constraints under exactly checked H1 (no H2), their
  polytope-level use in the exact phase, and the exact X-only region of voxel physics.
* **Do not use local-only interface bands for pruning in any certified path** (LOCAL_INVERSE_GEOMETRY.md
  Theorems 3 and 3′).
