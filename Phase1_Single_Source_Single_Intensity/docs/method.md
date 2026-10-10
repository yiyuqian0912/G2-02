# Phase I method reference

[README](../README.md) · [Interfaces](interfaces.md) · [Experiment instructions](experiments.zh-CN.md)

## Scope and observation

Phase I uses one unit-strength source in a continuous 1024×1024 hard-occlusion world. There is no distance decay, reflection, noise or material variation. The later agreed source-driven quadtree supersedes the proposal's initial fixed-size window: split occupied cells into four, retain source-free cells immediately, stop at size 16, and exclude occupied terminal leaves. Each scene yields 18 observations, three at each size 16–512.

The primary student's information is `O=(R,v)`, where `R` is the native local response and `v=(x,y,size)`. A pixel `[row,col]` samples the world at `(x+col+0.5,y+row+0.5)`. The candidate `s=(sx,sy)` is a world coordinate. IDs, true sources, the full tree and hidden geometry are not model inputs. A local obstacle mask remains in the data contract for checking and an explicit `use_obstacle=true` ablation; it is disabled in the new default configuration.

## Source domain and the quadtree prior

The default domain is the continuous world minus the observed window. The renderer's valid-source test additionally excludes closed obstacle interiors/boundaries. We distinguish this hidden physical validity from the public domain available to the student.

The source-driven selection process also implies a parent-cell constraint for the original generating source. The default experiment deliberately evaluates *fixed-window counterfactual compatibility*, without conditioning on that selection event. It may retain sources outside the parent if they reproduce the fixed local response. `quadtree_prior=true` is a separately labelled parent-minus-window ablation, not the default. Consequently the default target is a compatibility distribution, not a fully generative Bayesian posterior for the entire window-selection process.

## Physical teacher

For hidden scene `S`, rerender `R(S,s;v)` in the same window. A candidate inside an obstacle is invalid, is not rendered, has infinite physical cost, and receives zero teacher probability.

For valid candidates:

\[
L_{resp}(s)=\frac{\sum_u w(u)|R(S,s;v)(u)-R(u)|}{\sum_u w(u)},
\quad w(u)=1+\lambda\exp[-d(u)^2/(2\sigma^2)].
\]

Weights depend only on the observed response boundary, never the candidate response. The implementation marks both sides of 4-neighbor binary transitions; the image frame is not a response boundary. With no observed boundary, weights are uniform. The formal v2 default uses `lambda=2`, `sigma=1` pixel, `alpha=1`. Direct low-level calls retain their documented unweighted baseline default `lambda=0` unless overridden.

The optional edge term is a symmetric mean nearest-edge distance. Distances are divided by the window diagonal so sizes are comparable. Both empty edge sets have cost zero; exactly one empty set has cost 1 (empty-to-nonempty contributes zero; nonempty-to-empty contributes one). This finite convention makes `0 <= L_edge <= 2`. The exact convention is tested in `physics.py`.

\[
C(s;O,S)=\alpha L_{resp}(s)+\beta L_{edge}(s).
\]

Response-only uses `beta=0`; the default edge ablation uses `beta=0.1`. Boundary weighting and explicit edge distance are distinct but can overlap in information. Their practical value requires a paired experiment, not an assumption of independent gain.

At half-spacing centers of a globally anchored uniform grid, keep candidates outside the observation window in row-major order. Define

\[
q_i=\frac{\exp(-C_i/\tau)}{\sum_{j:valid_j}\exp(-C_j/\tau)},\qquad q_i=0\text{ for invalid }i.
\]

Each actual candidate has equal base weight. **No cell area, retained block area, source label, or supplied reference witness is multiplied into q.** A valid but uniformly poor candidate set still sums to one; report costs, minimum cost and compatible-grid coverage alongside distributions. Supplied reference solutions are separate renderer checks, not inserted training candidates.

## Student and optimization

The response encoder operates at native size. The window encoder provides position and scale. In the new model, deterministic local x/y channels preserve pixel position within the response features. Candidate Fourier features are computed from world coordinates divided by world size, using `sin(pi*2^k*s/world)` and cosine plus the normalized coordinates. Candidate-conditioned attention combines the observed feature map and candidate query to produce `E_theta(O,s)`.

For the primary protocol, the student softmax includes **all declared world-minus-window candidates**, including physically invalid ones. Physical invalidity is hidden from the student. Teacher q is zero there, but those locations still affect the softmax denominator and therefore receive training gradients. A geometry-assisted ablation masks them out and is labelled separately.

\[
p_i=\operatorname{softmax}(-E)_i,\qquad
\mathcal L=-\sum_i q_i\log p_i.
\]

This is forward-KL minimization up to teacher entropy; the energy need not equal the physical cost. Teacher temperature is already present in q and is not applied to the student a second time. Candidate scoring can be chunked, but concatenated energies are normalized **once globally**. Size buckets avoid resizing/padding image observations. Training records load lazily; candidate attention uses activation recomputation to reduce peak memory.

The shortest Fourier period is `2*world_size/2^(K-1)` for K frequency bands. New configurations require at least four candidate spacings per shortest period. The default `K=4, spacing=64` satisfies this guard; the legacy `K=6, spacing=64` does not. The guard reduces a specific aliasing risk; it does not prove accurate interpolation or prevent all learned high-frequency structure. A finer validation-grid physics probe measures off-grid agreement without tuning on test observations.

## Search and experimental scope

Uniform exhaustive evaluation on a declared grid is the primary training reference. `part_e/adaptive.py` performs broad block scouting followed by complete refinement of promising and randomly explored blocks on the same final grid. Final targets exclude scouting-only and unfinished blocks. Their normalization is conditional on retained support; omitted points are unknown, not known to have zero physical compatibility.

The comparison records rendering cost and the compatible/teacher mass omitted relative to the uniform reference. It does **not yet implement** the proposal's calibrated uncertainty selection or cheap-response/expensive-edge cascade, and is not silently substituted for primary training. This separation prevents search pruning errors from being presented as exact teacher uncertainty.

## Evaluation and interpretation

Freeze data identity, source code, candidate settings, scene split and windows before training. v2 uses fixed disjoint scene pools, with window choices stable under changes to training count. Select best epoch using validation cross-entropy, then evaluate held-out scenes against the same teacher coordinates. Report KL, JS, L1, compatible mass, entropy, invalid mass, size groups, constant/nonconstant responses, and a uniform baseline over the same student support. Edge ablations also compare against the common response teacher.

If student mass on invalid positions is positive, the true expected infinite-cost physical objective is infinite, encoded as `null` plus an explicit flag. Report valid-conditional cost separately. A finite auxiliary cost penalizes invalid mass by `alpha+2*beta`, the upper bound on valid costs for this binary setup; it must not be called the original expected physical cost. Geometry-assisted secondary results normalize the same energies after applying hidden validity.

Dense visualizations directly query every selected grid center and normalize there. The saved teacher remains at its measured resolution; no interpolated physical teacher is invented. Different grid sizes imply different per-point probabilities. Matched-grid displays use identical candidates and shared color scales; dense displays state their independent normalization and scales. Hover and NPZ expose numerical values.

The teacher has access to S while the student does not. Identical local observations from different scenes can require different scene-conditioned teacher distributions. The expected cross-entropy optimum averages their teacher targets conditional on the student's information; exact per-scene recovery is not generally identifiable. Preserving uncertainty and generalization are empirical goals, not guaranteed by the energy formulation.

Constant/nonconstant response is not the number of independent boundaries. Obstacle-type multisets are a limited grouping diagnostic, not all geometric combinations. Controlled one-boundary/two-boundary paired tests and broader unseen-combination studies remain necessary before claiming the corresponding proposal outcomes. Existing inspected test scenes are development evidence; reserve a new untouched pool for a final scientific claim.
