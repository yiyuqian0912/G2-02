# RIND3D scientific goals

## The learning problem: local-to-global inverse ray inference

**What the Student sees** (local-only inputs, [GLOSSARY.md](GLOSSARY.md) §2):
* the geometry inside a local window W of a 3D world, sampled at voxel centres (the SOLID crop);
* one source-specific binary response channel there: R(x) = 1 (response-active) iff the closed segment from a point
  driver source s to the FREE receiver x misses the closed solid, and R(x) = 0 (response-inactive) otherwise.

**What it must infer:** the global distribution of the driver-source location.

**What it is never given:** the global geometry, the other sources' channels, or the source coordinate as an input.

The target is the **posterior over source locations**, given the local observation, under the declared prior and the sampling law.
* The training labels are full-world posteriors: they condition on the complete geometry, which only the labeller sees. On the 128³ source-space grid they are represented as certified per-cell mass intervals.
* The Student's optimum averages these labels over all worlds compatible with the local observation ([GLOSSARY.md](GLOSSARY.md) §2–3).

The point of the task is to reward **nonlocal geometric reasoning**. The response interfaces inside W (legacy: shadow boundaries) are produced by occluders that may lie outside W, and several global source regions can explain the same local pattern.

## Not substitutes for the posterior objective

These are useful as baselines only:
* direct true-source NLL;
* one-hot source labels;
* coordinate regression.

## Resolutions

| Resolution | Role |
|---|---|
| **1024³** | Physical world: geometry and receiver grid, in production. |
| **128³** | Source-space output grid of the posterior labels. |
| 128³ physical world | Low-cost development and validation environment only. |
| 512³ | Not used. It is a fallback that would need measured justification and approval. |

## Accuracy target

* TV(label, true posterior) ≤ 0.05, with ≤ 0.10 as the secondary target.
* A deterministic certificate per label is preferred.
* Statistical terms of the sampling law are stated with their confidence.
