# Module contracts — proposed initial API

These contracts are coordination targets. Functions and tensor classes are not implemented yet. Owners may revise them together before integration.

## Observation record

| Field | Shape / type | Meaning |
|---|---|---|
| `scene_id`, `view_id` | integer | Stable identifiers; not student features |
| `response` | `[H, W]` float | Single-source response; H = W = side |
| `window` | `[3]` | World-coordinate `(x, y, side)` |
| `source_xy` | `[2]` float | Generating source; supervision / diagnostics only |
| `source_intensity` | scalar | Same fixed value across Phase I |
| `obstacle` | `[H, W]` bool | Teacher / diagnostic field; not student input |

A batch uses size buckets or padding. Padded batches additionally carry `pixel_valid[B, Hmax, Wmax]`; masked pixels must not affect feature pooling or losses. Keep world extent independent of padded shape.

Split scene IDs before deriving observations or teacher caches. New scene geometry and source configurations must not leak across splits through derived identifiers.

## Rendering and costs

```python
render_candidate(scene_id, window, source_xy, fixed_intensity) -> response[H, W]
response_cost(observed, rendered, observed_weights) -> scalar
edge_cost(observed_edges, rendered_edges, window_diagonal) -> scalar
```

The adapter wraps the actual RIND `rerender_region` API after checking its implementation. Reject sources outside the world or inside obstacles. Apply the separate window-selection support rule before scoring. Keep intensity and window metadata unchanged during candidate rendering.

## Candidate / teacher record

```text
scene_id, view_id
candidate_xy       [N, 2]   world coordinates
candidate_valid    [N]     valid physical and selection-consistent support
base_weight        [N]     declared area / prior weight, or equal weights
response_cost      [N]
edge_cost          [N]     optional; record whether computed
physical_cost      [N]
teacher_mass       [N]
config_id, renderer_version, coordinate_convention
```

Use stable log-sum-exp normalization. Invalid candidates have zero mass. A sample with no valid candidates is an explicit failure, not a uniform fallback. Adaptive candidates require metadata sufficient to recover their base measure. Preserve a uniform evaluation grid for comparable distributions.

Obstacle validity uses hidden geometry. Record whether student evaluation is normalized on teacher-provided valid support; do not present such evaluation as geometry-free deployment. A deployable candidate generator can use world bounds and window-selection metadata alone, or learn validity implicitly. Distinguish these settings in reports.

## Student

```python
encode_observation(response, window, pixel_valid=None) -> embedding
score_candidates(embedding, candidate_xy) -> energy[..., N]
```

The encoder receives response and window metadata only. It must not consume source truth, scene IDs, obstacle masks, or the complete tree. Normalize coordinates by world size. Candidate order must not change the associated scores.

## Training and evaluation

Training consumes observations and teacher records with matching identifiers, candidate order, masks, and base weights. Apply the same normalization measure to teacher and student. Record checkpoint configuration, seed, and split version.

Evaluation exports one record per observation with distribution metrics, physical-cost diagnostics, window size, and runtime. Figures should show the response, window location, teacher map, student map, and generating source as a reference marker. Label every conceptual illustration and untrained example explicitly.
