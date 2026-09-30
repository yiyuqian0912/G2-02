# Phase I Method Reference

[Project responsibilities](../README.md) · [中文 README](../README.zh-CN.md) · [Data contracts](interfaces.md)

This document defines the current method, not completed software or experimental results. File responsibilities remain in the root README.

## 1. Observation and inverse problem

Generate new single-source scenes with a shared fixed intensity in a continuous 1024 × 1024 occlusion environment. Phase I excludes distance decay, reflections, noise, and material differences. Split data by scene ID before training or target selection.

Let R be the local response, v = (x, y, size) the window metadata, and O = (R, v). The student predicts a distribution of candidate source locations s = (s_x, s_y), rather than one uniquely labeled coordinate.

Windows follow the confirmed source-driven quadtree: subdivide source-containing nodes into four, retain source-free nodes, and exclude source-containing leaves at window size 16 × 16. A single-source scene yields 18 source-free windows. This replaces the original proposal's fixed-size window assumption.

Assign partition-line sources to right/lower children and the outer world boundary to terminal children. Reuse this ownership convention when excluding source-containing windows.

Coordinates point rightward in x and downward in y. Pixel [row, col] in a window samples (x + col + 0.5, y + row + 0.5). Preserve window extent when handling variable-size observations.

## 2. Physical compatibility

The teacher knows hidden scene geometry S. For a candidate s, rerender the same window at the fixed intensity to obtain the candidate response R_hat(s). The generating source should reproduce its stored observation within a declared numerical tolerance.

Boundary-weighted response consistency is:

```text
L_resp(s) = sum_u w(u) |R_hat(s)(u) - R(u)| / sum_u w(u)
w(u) = 1 + lambda exp(-d(u)^2 / (2 sigma^2))
```

Here d(u) is distance to the nearest observed response boundary. Weights depend only on the observed response. Use uniform weights when no valid boundary exists.

The optional edge term directly compares boundary sets:

```text
L_edge(s) = D(E_R, E_s) + D(E_s, E_R)
C(s; O) = alpha L_resp(s) + beta L_edge(s)
```

A mean nearest-boundary distance is a possible definition of D. Declare distance units, crop-border treatment, and empty-set behavior before creating targets. These implementation choices are not fixed by the proposal. Start with beta = 0 and evaluate the added edge term through ablation. Require positive fixed intensity accepted by RIND (at most 1), alpha > 0, beta >= 0, lambda >= 0, and sigma > 0. Zero intensity makes the inverse response task degenerate.

## 3. Candidate search and spatial weights

Start with uniform candidates, then implement coarse-to-fine adaptive search. Screen broadly with response cost and refine promising or explicitly defined uncertain regions. Retain exploration coverage and compare missed feasible regions and rendering cost against a uniform reference.

Candidates are inside world bounds and outside the observation window; the teacher also rejects obstacle interiors using geometry. The additional restriction to the window's parent node is disabled in the current configuration. Consequently, the target is fixed-window response compatibility, not the full posterior conditioned on generation of that particular quadtree leaf. Reproducing the leaf would require the source inside its parent and outside the leaf; activating that interpretation requires new teacher targets.

The current spatial convention is a uniform prior with nonoverlapping candidate cells. Each candidate represents an area a_i. When a cell is subdivided, its parent and children must not both contribute overlapping area. Save cell bounds with the candidates. At window/obstacle boundaries, clip, subdivide, or document an effective-area approximation. A valid cell center does not establish that the entire cell is valid. Equal-area valid cells reduce to the original equal-weight formula; clipped regular-grid cells need not have equal weights. If sampling changes to a different scheme, its integration weights must be redefined.

All final costs within a teacher record use the same definition. Do not mix response-only and response-plus-edge costs without an explicit approximation policy. If only part of the domain is retained, report that coverage; area weights alone cannot recover omitted modes.

## 4. Teacher distribution

For valid candidates V and temperature tau > 0:

```text
q_i = a_i exp(-C_i / tau) / sum_(j in V) a_j exp(-C_j / tau)
```

Invalid candidates receive zero probability. Valid areas must be positive. An empty valid set is a failure, not a uniform fallback. Preserve raw costs: a normalized distribution does not establish that any candidate explains the observation well.

This is a Gibbs compatibility target, not a calibrated generative posterior. This distribution is spatial probability mass represented by the selected cells, not a pointwise continuous density. Store coordinates, areas, validity, and candidate ordering alongside the target.

## 5. Conditional energy student and training

The energy function receives response, window metadata, and candidate coordinates only. It does not receive source truth, scene IDs, obstacle masks, full geometry, teacher cost, or the complete quadtree.

Fourier coordinate features are:

```text
 gamma(s) = [s, sin(omega_0 s), cos(omega_0 s), ..., sin(omega_K s), cos(omega_K s)]
```

They improve coordinate representation within the student and do not alter physical rendering. The model outputs E_theta(O, s), with lower energy indicating greater compatibility. On the same valid support and areas as the teacher:

```text
p_i = a_i exp(-E_i) / sum_(j in V) a_j exp(-E_j)
L_CE = -sum_(i in V) q_i log(p_i)
```

Soft-target cross-entropy and teacher-to-student KL yield the same optimization objective up to a teacher-only constant. Energy need not numerically equal physical cost. Validity and area weights are distribution-normalization metadata, not encoder features.

Fit a small subset first, then scale. Select models using validation scenes; keep test scenes out of tuning. At inference, the student scores new coordinates without rerendering. A normalized map still requires a declared candidate support and spatial measure.

## 6. Evaluation and interpretation

Evaluate on common candidate support and a consistent area convention. A shared uniform reference grid is preferred for comparing different search strategies. For variable-area heatmaps, plot cell probability divided by represented area as piecewise density; sum cell masses for region probabilities.

Required evidence includes teacher/student distribution agreement, physical cost of student-supported locations, ambiguity retention, unseen scenes and obstacle combinations, results by window size, response-only versus added-edge ablation, and uniform/adaptive search quality and cost. Inspect one-boundary and independent-two-boundary observations without assuming concentration is guaranteed in every example.

Three limits must remain explicit:

- Source-driven window selection conveys source information through window location and size.
- The teacher target is scene-conditioned, q(s | O, S), while the student sees only O. Identical student inputs may have different scene-specific teacher targets; exact recovery is not always identifiable.
- Evaluation on a teacher-provided obstacle-validity mask uses hidden geometry to define support. Label this separately from geometry-free deployment.

Geometry-free deployment must also use a geometry-free candidate set, not just an encoder without geometry. Report teacher-supported agreement separately from predictions on a predeclared full public domain. Training on masked or adaptively retained candidates does not constrain excluded locations automatically. A good teacher-supported score does not establish good deployment behavior on those locations.
