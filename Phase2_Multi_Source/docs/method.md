# Multi-source task and physical semantics

[English guide](../README.md) · [中文指南](../README.zh-CN.md) · [Interfaces](interfaces.md)

## Observation

Each scene has a continuous world, continuous opaque obstacles, and K generating sources `(x_k,y_k,a_k)`. Generation uses K=1–4 and uniform strengths `0 <= a_k < 1`. At each pixel center, source k contributes `a_k` if visible and zero otherwise:

```text
I_k(u) = a_k * visibility_k(u)
R(u) = sum_k I_k(u)
```

The sum is not clipped or binarized. A scene with one random-strength source is a different distribution from the unit-strength single-source release. Stored binary visibility and float64 strength preserve separate contributions without storing K full float images.

Source-driven quadtree cells are subdivided into four until source-free or side length 16. Record source-free cells; omit occupied terminal cells. Partition edges belong to right/bottom children, including the world edge in terminal cells. Observation counts are variable. Sampling uses original world-coordinate pixel centers with step 1.

Student conditioning is `O=(R,M,(x,y,L))`, where M is only the local obstacle mask. Scene/view IDs identify samples; they are not features. Full geometry, outside masks, source labels/channels and the full view tree belong to teacher/evaluation. Scene partitions must be disjoint.

## Fixed-window sources outside the observation

For every source in a candidate configuration, the domain is **world minus the observation window**, further excluding obstacle interiors through teacher-side geometry. No quadtree parent restriction is used. Quadtree is the dataset's window-generation policy; a candidate is not required to regenerate the same leaf.

The target concerns compatibility with a fixed local observation. It is not a calibrated posterior that models the likelihood of the source-driven window selection. Keep this distinction in experiment reports.

## Physical comparison

An inverse hypothesis is a complete unordered source set, not K independent source-location labels. Rerender that set in the original geometry and original window:

```text
C(sources;O,S) = mean_u(abs(R_hat(u)-R(u)))
```

This is mean absolute **intensity** error, not a binary mismatch fraction. Values are un-clipped and compared in float64. The mean normalizes pixel count across window sizes. The student receives float32 features; do not introduce float32 rounding into renderer sanity comparisons. The evaluator accepts all recorded sizes without resizing. Invalid candidates have infinite cost and zero source renders.

Reference witnesses include truth, source-order reversal when K>1, and four co-located strength splits. Splits can use K+1 sources. These witnesses preserve the ideal field, up to float64 summation tolerance `1e-12`. They are diagnostic witnesses only, not an exhaustive inverse set, proposal support or probability target.

## Joint teacher baseline

For fixed known K, unrestricted positions and strengths form a 3K-dimensional space. Unknown K adds alternatives of different dimensions. The implemented target retains complete source sets and their probabilities. A spatial presence marginal is provided for inspection; it cannot encode correlations between complete source sets.

The baseline uses a deterministic global position grid with centers `(spacing/2 + j*spacing, spacing/2 + i*spacing)`, ordered y then x. Exclude the observation and continuous-geometry obstacles. Conditional on a declared K, independently draw K valid grid positions uniformly **with replacement**, and K independent strengths uniformly in `[0,1)`. Repeat `N_K` times using a recorded PCG64 seed/substream. This is finite Monte Carlo sampling of complete configurations, not exhaustive enumeration of the grid's combinatorial joint space or the continuous world.

Canonicalize source rows by `(x,y,strength)`. Deduplicate exact permutation-equivalent sets in first-occurrence order and add their sampling mass. Co-located strength splits remain distinct hypotheses, especially across K. Drawing ordered iid sources then canonicalizing preserves the induced sampling multiplicities; uniformly enumerating unordered combinations would define a different prior.

Known K uses a point count prior. Unknown K requires explicit positive probabilities `pi_K` summing to one. Each draw receives base mass `pi_K/N_K`. After duplicate aggregation, normalize valid candidates as:

```text
q_i = base_mass_i * exp(-C_i/tau) / sum_j(valid_j * base_mass_j * exp(-C_j/tau))
```

Fixed-K equal-mass support reduces to ordinary cost softmax. Count stratification does not apply spatial area weighting: grid positions already have equal mass. It prevents the number of proposals per K from silently replacing the declared count prior. Normalize across all K together, so response evidence can change count probabilities. This is a compatibility distribution under a declared empirical prior, not a calibrated physical posterior.

Normalization subtracts the minimum cost before division by temperature and then uses log-mass centering. Invalid candidates receive zero probability; empty valid support fails. Every final support point must be evaluated before a target is constructed. Budget exhaustion returns an incomplete search result and target construction fails. Completing the declared samples does **not** establish exhaustive coverage. No reference set or true source parameters enter proposal generation or probability normalization.

The physical evaluator caches bit-packed unit-strength visibility by source position and costs by canonical complete set. Report unique visibility renders separately from unique joint-cost evaluations. Different strengths and configurations can reuse visibility without repeating geometry calculations.

Records retain padded source parameters, active counts, sampling mass/multiplicity, validity, raw costs, probabilities, position-grid ordering, window metadata, count/strength policies, spacing, temperature, dataset archive/manifest identity, code hashes and run identity. Full-geometry teacher evaluation and geometry-free student deployment remain separate settings. Student conditioning stays local; teacher geometry/masks and reference diagnostics are not student features.

## Diagnostics and remaining decisions

Diagnostics compare observed/best-candidate responses, show local obstacles, spatial projections with truth/reference overlays, count probabilities, joint entropy, effective support size, and sampled zero/low-cost prior mass. The 2D minimum-cost projection takes the minimum **complete-set** cost among sampled configurations containing a point; it is not a single-source physical cost map. Joint region connectivity and continuous inverse volume cannot be recovered from this sparse high-dimensional support and are not claimed.

Sampling spacing, samples per count, temperature, candidate counts and their prior are explicit unresolved experiment settings. Demonstration/check commands choose values openly. Compare larger sample sizes, finer grids and multiple seeds before using the baseline as a research target; entropy comparisons need comparable support policies. Adaptive search, alternative strength priors/fitting, student architecture and training remain future work. See the [teacher guide](teacher.md) for runnable examples.
