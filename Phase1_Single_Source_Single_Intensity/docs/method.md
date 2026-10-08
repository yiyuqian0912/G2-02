# Phase I Method Reference

[English README](../README.md) · [中文 README](../README.zh-CN.md) · [Contracts](interfaces.md) · [Run the teacher](teacher-baseline.md)

The implemented baseline is observation → uniform physical cost map → soft teacher target. Adaptive search, richer boundary losses, and student modeling/training remain future work.

## 1. Observation and inverse problem

Use the supplied single-source release: 10,000 scenes, 180,000 observations, intensity 1, and a continuous 1024 × 1024 occlusion world. Phase I excludes distance decay, reflections, noise, and material differences. Keep all observations and derived targets of a scene in the same split; the formal split ratios are still unresolved.

Let `R` be the sampled local response, `M` the local obstacle mask, `v=(x,y,L)` the window, and `O=(R,M,v)`. The student is intended to predict source-space compatibility for candidate `s=(sx,sy)`. It receives no outside-window obstacle information or full scene geometry. IDs identify records and are not model features.

Windows use the source-driven quadtree. Split source-containing cells into four; immediately retain source-free cells; stop at side length 16 and exclude occupied terminal leaves. A single-source scene has 18 observations. Split-line sources belong to right/lower children; the outer world edge belongs to terminal cells.

Coordinates point right in x and down in y. Window pixel `[row,col]` samples `(x+col+0.5,y+row+0.5)`. Image resolution describes sampling precision; source and obstacle geometry remain continuous.

## 2. Sources outside the observation window

For a window of side `L`, its parent is the aligned cell of side `2L`:

```text
parent_origin = (floor(x/(2L))*2L, floor(y/(2L))*2L)
main domain = world \ W
optional prior ablation = quadtree_parent(W) \ W
```

The task explicitly studies sources outside the observation window. The generator supplies quadtree windows, but the main teacher conditions on a fixed window and evaluates response compatibility without requiring candidates to generate the same leaf. The default is `teacher.quadtree_prior=false`, world-minus-window. Although the generating source is inside the parent, that restriction is used only when explicitly enabling the optional parent-prior ablation. Both domains remain subject to world bounds and continuous teacher-side geometric validity. Domain construction belongs to `search.py`; physics owns obstacle/source validity.

The main target is physical compatibility for window-external sources on the chosen world support. It is not a calibrated generative posterior: the quadtree window-selection likelihood and a continuous source density are not inferred.

## 3. Response-only physical baseline

The teacher knows hidden scene geometry `S`. At each valid source position it rerenders the same window at intensity 1 and compares with `R`:

```text
C(s) = mean(abs(R_hat(s) - R))
     = disagreeing sampled pixels / L²  (binary data)
```

The same evaluator handles all six stored window sizes (16–512) at their original resolution. Dividing by `L²` gives the same disagreement-fraction interpretation across sizes, rather than larger windows automatically receiving larger costs.

Every sampled pixel has equal weight. Obstacle pixels are included and zero in both responses. Thus `C` lies in `[0,1]`; `C=0` means exact equality on this sampled observation. Invalid candidates are not rendered and have infinite cost. This CPU NumPy/Numba evaluator is not differentiable in source coordinates.

The inverse set is conceptually `{s:C(s)=0}` and can have continuous, disconnected, elongated, or large regions. A finite grid only samples that set. No zero-cost grid point does not imply that the continuous inverse set is empty.

The ten dataset source positions, including truth, are incomplete compatibility witnesses. Each must independently rerender exactly and have zero response cost. They are used for checks, overlays, and limited coverage diagnostics. They never augment the grid or enter teacher normalization as extra samples or labels.

### Later boundary-loss experiments

The proposal's boundary-weighted response cost and optional symmetric edge distance are not implemented in this baseline:

```text
L_resp = sum_u w(u) abs(R_hat(u)-R(u)) / sum_u w(u)
w(u) = 1 + lambda exp(-d(u)²/(2 sigma²))
C = alpha L_resp + beta L_edge
```

Current settings are `alpha=1`, `lambda=0`, `beta=0`; no edge cost is computed. Before enabling richer costs, define boundary extraction, distance units, crop-border behavior, and empty/missing/extra boundary handling. All candidates of a target must use the same physical cost definition.

## 4. Uniform support and teacher distribution

Use globally anchored world-coordinate lattice points `((kx+0.5)*spacing,(ky+0.5)*spacing)`. Filter the declared domain, then evaluate geometric validity and physical cost. Order is ascending y, then ascending x. At equal spacing the prior support is a subset of the world support. Candidate spacing is independent of image sampling resolution.

For a completed uniform result and valid candidate set `V`:

```text
a_i = exp(-(C_i - min_(j in V) C_j)/tau)
q_i = a_i / sum_(j in V) a_j
```

Require finite positive `tau`. Invalid probabilities are exactly zero; an empty valid set or incomplete search is an explicit failure. Retain raw costs, positions, validity, ordering, support metadata, and provenance. No one-hot truth label or area weighting is applied. `q` is a discrete approximation of the compatibility landscape on this support, not a continuous density. Compare distributions on the same coordinates.

Spacing and temperature have no formal defaults in `configs/phase1.json`. CLI overrides are stored with the actual generated target; configuration and source hashes identify the run.

## 5. Diagnostics before adaptive search

`diagnostics.py` shows local response/mask, cost map, sampled zero-cost map, teacher mass, truth, all ten witnesses, and a separate witness zoom. It reports zero/low-cost fractions, four-neighbor sampled components, approximate component areas/bounding boxes, entropy, and effective candidate count. Low-cost thresholds are explicit optional analysis settings.

Component connectivity and area describe sampled cells, not proven continuous topology. Witness coverage is only a lower-bound diagnostic; nearest-grid witness matches depend on spacing. Record spacing and use a common world spacing when studying changes with window size.

The uniform baseline is the reference for future adaptive search. Later work must preserve multiple regions, separate trace points from final support, deduplicate/cache coordinates, count actual unique renders, reach a consistent declared final spacing, and fail explicitly when the budget prevents completion. Coarse and fine trace points must not be mixed blindly into the teacher softmax. A pruning rule remains undecided.

## 6. Student and evaluation: intended later stages

The intended energy input is `(R,M,v,s)` with local `M` only. Full geometry, outside-window masks, truth, scene IDs, physical costs, and the complete quadtree are excluded. Fourier source features and the model architecture remain student responsibilities; no model/training implementation was changed for this baseline.

On the same valid teacher support:

```text
p_i = exp(-E_i) / sum_(j in V) exp(-E_j)
L_CE = -sum_(i in V) q_i log(p_i)
```

Teacher-to-student KL has the same optimization objective up to a teacher-only constant. Energy need not equal physical cost numerically. Validity is normalization metadata, not an encoder feature.

Evaluate held-out scenes, ambiguity retention, physical compatibility, window-size effects, boundary-loss ablations, and uniform/adaptive quality and rendering cost on shared support. Single-coordinate error alone cannot describe this inverse problem.

The teacher knows `S`, while the student sees only `O`; identical observable inputs can have different scene-specific teacher maps. Agreement on teacher-provided geometric validity must be labeled teacher-supported evaluation. Geometry-free deployment needs a declared public candidate set and cannot reuse support selected by hidden geometry or physical costs. Training on retained candidates does not automatically suppress untrained locations.
