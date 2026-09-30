# Phase I Method Reference

Detailed formulation and implementation considerations. For folder responsibilities and expected deliverables, see [the project README](../README.md).


### 3.1 Observation generation

Start with the entire world. A node containing the source is divided into four equal children unless its side length is already 16. A source-free node becomes an observation immediately and is not subdivided further. A source-containing 16 × 16 node is occupied and excluded from observations. Obstacles do not trigger subdivision.

Children are ordered top-left, top-right, bottom-left, bottom-right. Sources on partition lines belong to the right or lower child; the outer world boundary belongs to the terminal child. For one source, this yields **18 observations per scene**, with three source-free leaves at each side length 512, 256, 128, 64, 32, and 16. Verify this property in generated data.

Coordinates increase rightward in x and downward in y. Arrays use `[row, column]`; a window pixel samples the world at `(x + column + 0.5, y + row + 0.5)`.

Do not resize every observation and discard its extent. Use size-bucketed batches or padding with valid-pixel masks, and supply the original `(x, y, side)` metadata. Normalize spatial coordinates by 1024 for network inputs.

### 3.2 Inverse target

Let O be the response, v = `(x, y, side)` its window metadata, s a candidate source position, and S the hidden scene geometry. The student estimates:

```text
pθ(s | O, v)
```

Different source positions can explain the same response. A distribution retains this ambiguity; one coordinate prediction cannot represent it reliably.

The source-driven crop policy also contains information. A particular source-free leaf implies that the source is outside that leaf but inside its parent. The proposed baseline applies this **selection-consistency mask** to candidate support. Treat it separately from geometric validity, such as excluding obstacle interiors. Do not feed the complete tree into the student: its occupied leaf would reveal the source region directly.

### 3.3 Physical teacher

Place a candidate source into the original scene at the fixed intensity and rerender the same window:

```text
Ôs = R(S, s, v)
```

Use the RIND `rerender_region` interface for new source parameters. `get_region` only decodes existing responses and cannot substitute for rerendering. Verify the adapter by reconstructing an observation with its generating source.

First compute boundary-weighted response consistency:

```text
Lresp(s) = Σu w(u) |Ôs(u) − O(u)| / Σu w(u)
w(u)     = 1 + λ exp[−d(u)² / (2σ²)]
```

Here d(u) is distance to the nearest observed response boundary. Weights depend only on O. If no valid boundary exists, use uniform weights. This loss measures response disagreement, with added emphasis near observed boundaries.

An optional geometric term compares observed and candidate boundary sets directly:

```text
Ledge(s) = D(EO, Es) + D(Es, EO)
D(A, B) = mean over a ∈ A of the distance to the nearest point in B
C(s)    = α Lresp(s) + β Ledge(s)
```

The mean nearest-distance definition is a proposed implementation. Normalize edge distances by the window diagonal and document empty-boundary behavior. A reasonable initial convention is zero if both sets are empty and a declared maximum penalty if exactly one is empty. Define whether crop borders count as edges; avoid artificial crop-border edges by default.

Start with **β = 0**. Hard occlusion may make the edge term redundant; its benefit must be tested rather than assumed.

### 3.4 Teacher distribution

For uniformly weighted candidates, convert physical costs to soft targets:

```text
q*i = exp(−C(si) / τ) / Σj exp(−C(sj) / τ),    τ > 0
```

Temperature controls concentration. Use validation scenes to select it. This is a relative distribution over the selected candidate set: normalization does not prove that any candidate explains the observation well. Also report minimum physical cost.

Nonuniform adaptive samples require a declared base measure. For cell-based sampling, spatial mass can be formed as `q*i ∝ ai π(si) exp(−C(si)/τ)`, where ai is cell area and π is the source prior. Do not let sampling density silently change the intended spatial distribution. Use the same weights and support for teacher and student normalization.

The dataset's stored equivalent candidates mainly reverse source order or split intensity across colocated sources. They do not enumerate alternative source locations and cannot replace our spatial candidate search.

### 3.5 Conditional energy student

A proposed initial model combines:

1. A variable-size response encoder with masked or adaptive pooling.
2. An embedding of normalized window metadata `(x, y, side)`.
3. Fourier features of candidate world coordinates.
4. A scalar energy head; lower energy means higher compatibility.

```text
γ(s) = [s, sin(ω0 s), cos(ω0 s), …, sin(ωK s), cos(ωK s)]
pθi  = exp(−Eθ(O, v, si)) / Σj exp(−Eθ(O, v, sj))
LCE  = −Σi q*i log pθi
```

Sine and cosine apply componentwise. Frequency bands are configurable. Fourier features affect only the student representation, never the renderer. Cross-entropy is equivalent to teacher-to-student KL minimization up to a teacher-only constant. Energies need not numerically equal physical costs.

At inference, encode the observation and score candidate locations without rerendering. A normalized spatial map still needs a chosen candidate grid or integration scheme.

**Information limit:** the teacher conditions on `(O, v, S)`, while the student conditions on `(O, v)`. Different hidden scenes can share the same observable input but have different teacher maps. The student cannot always recover the exact map of a hidden scene; under cross-entropy it learns an input-conditioned average of teacher targets. Report this limitation separately from optimization errors.

### 3.6 Candidate search

Begin with a uniform grid and a dense reference on a small subset. Adaptive search is a later cost-saving extension: refine promising regions while retaining exploration coverage. Compare mode recall and rendering cost against the reference. Define any uncertainty criterion explicitly and retain area or proposal metadata.

Use a consistent final cost across candidates within a target distribution. Do not mix response-only scores and response-plus-edge scores without a documented approximation policy.

