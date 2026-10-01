# Phase I Method Reference

[Project responsibilities](../README.md) · [中文 README](../README.zh-CN.md) · [Data contracts](interfaces.md)

This document defines the current method, not completed software or experimental results. File responsibilities remain in the root README.

## 1. Observation and inverse problem

Use the supplied single-source release (10,000 scenes, intensity 1) in a continuous 1024 × 1024 occlusion environment. Generate additional scenes separately when an evaluation protocol requires them. Phase I excludes distance decay, reflections, noise, and material differences. Split data by scene ID before training or target selection.

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

## 3. Candidate search and retained support

Start with uniform candidates, then implement coarse-to-fine adaptive search. Screen broadly with response cost and refine promising or explicitly defined uncertain regions. Retain exploration coverage and compare missed feasible regions and rendering cost against a uniform reference.

Candidates are inside world bounds and outside the observation window; the teacher also rejects obstacle interiors using geometry. The additional restriction to the window's parent node is disabled in the current configuration. Consequently, the target is fixed-window response compatibility, not the full posterior conditioned on generation of that particular quadtree leaf. Reproducing the leaf would require the source inside its parent and outside the leaf; activating that interpretation requires new teacher targets.

The current convention is a discrete distribution over retained valid candidate positions, without area weighting. Coarse-to-fine search excludes clearly incompatible regions and refines the retained regions. Aim for the same final spacing across retained regions and deduplicate coordinates; otherwise sampling density still affects aggregate candidate mass. Region bounds may be retained internally for search bookkeeping.

All final costs within a teacher record use the same definition. Do not mix response-only and response-plus-edge costs without an explicit approximation policy. If only part of the domain is retained, report that coverage; subsequent softmax cannot recover omitted modes.

## 4. Teacher distribution

For valid candidates V and temperature tau > 0:

```text
q_i = exp(-C_i / tau) / sum_(j in V) exp(-C_j / tau)
```

Invalid candidates receive zero probability. An empty valid set is a failure, not a uniform fallback. Preserve raw costs: a normalized distribution does not establish that any candidate explains the observation well.

This is a Gibbs compatibility target over the retained candidates, not a calibrated full-domain posterior or continuous density. Store coordinates, validity, and candidate ordering alongside the target. Candidate-set changes change the normalization; evaluate distributions on shared support.

## 5. Conditional energy student and training

The energy function receives response, window metadata, and candidate coordinates only. It does not receive source truth, scene IDs, obstacle masks, full geometry, teacher cost, or the complete quadtree.

Fourier coordinate features are:

```text
 gamma(s) = [s, sin(omega_0 s), cos(omega_0 s), ..., sin(omega_K s), cos(omega_K s)]
```

They improve coordinate representation within the student and do not alter physical rendering. The model outputs E_theta(O, s), with lower energy indicating greater compatibility. On the same valid candidate support as the teacher:

```text
p_i = exp(-E_i) / sum_(j in V) exp(-E_j)
L_CE = -sum_(i in V) q_i log(p_i)
```

Soft-target cross-entropy and teacher-to-student KL yield the same optimization objective up to a teacher-only constant. Energy need not numerically equal physical cost. Validity is distribution-normalization metadata, not an encoder feature.

Fit a small subset first, then scale. Select models using validation scenes; keep test scenes out of tuning. At inference, the student scores new coordinates without rerendering. A normalized map still requires a declared candidate support and spatial measure.

## 6. Evaluation and interpretation

Evaluate teacher and student on the same candidate coordinates. Use a shared uniform reference grid when comparing search strategies. Label plot spacing and do not interpret adaptive candidate mass as continuous density. For search recall, count pruned reference locations as uncovered; comparing only retained regions hides missed modes. Retained-region softmax is conditional on that support and does not quantify omitted probability mass.


Required evidence includes teacher/student distribution agreement, physical cost of student-supported locations, ambiguity retention, unseen scenes and obstacle combinations, results by window size, response-only versus added-edge ablation, and uniform/adaptive search quality and cost. Inspect one-boundary and independent-two-boundary observations without assuming concentration is guaranteed in every example.

Three limits must remain explicit:

- Source-driven window selection conveys source information through window location and size.
- The teacher target is scene-conditioned, q(s | O, S), while the student sees only O. Identical student inputs may have different scene-specific teacher targets; exact recovery is not always identifiable.
- Evaluation on a teacher-provided obstacle-validity mask uses hidden geometry to define support. Label this separately from geometry-free deployment.

Geometry-free deployment must also use a geometry-free candidate set, not just an encoder without geometry. Report teacher-supported agreement separately from predictions on a predeclared full public domain. Training on masked or adaptively retained candidates does not constrain excluded locations automatically. A good teacher-supported score does not establish good deployment behavior on those locations.
