# Rendering and selection strategies for the physical stage (Section 10)

* **Measured on:** RTX 3060 Laptop, using the certified renderers of `rind3d` (CUDA directed-rounding intervals with exact CPU recomputation of UNKNOWN bits).
* **Scenes:** fresh 1024³ canary scenes (`canary1024_A`).
* **Script:** `scripts/render_benchmark.py`.

## Strategies

| | Strategy | What is computed |
|---|---|---|
| A | Full global rendering (adopted) | k certified full 1024³ fields (≈ 2.5 s per source) → exact count index (< 0.1 s per source) → every proposal, rate stream included, answered by O(1) exact count queries (≈ 0.1–1.4 ms) → accepted windows read from the fields |
| B | Independent local rendering | Every proposal window rendered per source with the certified CUDA 64³ tile renderer, one call per tile (L = 256: 64 tiles) |
| C | Batched local rendering | As B, with 32 tiles per call and one bound source per channel |
| D | Hybrid (coarse global → exact local) | Not applicable without changing the law: the acceptance A and the layer count m need exact counts, so a coarse field can only reorder work, never skip it. See below. |

## Measurements (k = 4 scenes, no thinning; 81 quota proposals per scene)

| scene | A: fields | A: count index | A: selection (81 quota + 3072 rate proposals) | A: materialise | A total | B (quota proposals only) | C (quota proposals only) |
|---|---|---|---|---|---|---|---|
| 000003 (wedges) | 8.9 s | 0.36 s | 5.7 s | 15.0 s | 29.9 s | 3.3 s | 1.6 s |
| 000015 (corridors) | 12.0 s | 0.55 s | 8.2 s | 21.2 s | 41.9 s | 4.4 s | 2.4 s |

**Exactness.** The local CUDA responses equal the full-field crops bit for bit: 0 mismatches in 2 × 60.8M bits.

**Physics validation of A** (`validate.physics_spot_check`): 0 mismatches against the canonical certified CPU point predicate (239,616 receivers) and the independent rational predicate (2,016 receivers).

## Reading

1. **Selection really does need exact per-channel information for every proposal.** The rate streams (3 × 1024 proposals per scene, needed for the training weights ω and for the balancing statistics) and the balancing thinning (a simulated mean of 490 quota proposals per scene) turn local rendering into 3,500+ window renders per scene. With C that is ≈ 60–90 s for the rate streams alone, versus a 9–12 s fixed cost for A.
2. **Local rendering would only win** for a design with few proposals and no rate stream, which the 1K contract does not have.
3. **The fixed costs of A are small.** Full-field rendering costs about one L = 256 window label every ≈ 100 scenes. Fields are ≈ 0.8 MB per source and are kept as canonical privileged data.
4. **The remaining physical-stage cost is materialisation (15–21 s).** It is CPU work: reading crops, packing and zstd, overlap and diversity statistics. It is not rendering. It is 4–8% of a scene's label cost, so it is not on the critical path.
5. **The sampling law is unchanged by any of these strategies.** The same proposal streams and the same exact decisions are used, so B and C would produce identical windows.

**Decision:** strategy A, full certified fields with exact count indices.
