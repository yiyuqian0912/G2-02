# Report B — RIND3D-100K dataset contract v1

* **Machine-readable form:** `contract/RIND3D_100K_CONTRACT_v1.json`, id `rind3d-100k-contract-v1`.
* **Status:** frozen draft. Development canaries only; production requires the go / no-go of GO_NO_GO_100K.md.
* **Relation to 1K:** a new production profile. RIND3D 1K (contract v1, 16/24/24) is unchanged and remains its own
  release.

| item | rule |
|---|---|
| scenes | 100,000 accepted independent scenes, slots 0..99,999, seed 20261215, scene_id = sha256(contract ‖ geometry_id ‖ sources) |
| splits | sha256(scene_id) mod 100: < 80 train, < 90 val, else test (scene-level isolation) |
| physical world | canonical polyhedral / CSG geometry at **1024³**; continuous coordinates; rind3d closed-solid / closed-segment visibility; receivers at voxel centres (i + ½)/1024; SOLID receivers unlit |
| geometry families | the nine validated v0.1 families, by slot mod 9 (balanced): rotated_boxes, thin_slabs, prisms, wedges, convex_polyhedra, multiscale_convex, corridors, barriers, occluder_clusters |
| windows | **exactly 12 per scene: 4 × side 64, 4 × side 128, 4 × side 256** (sides in 1024³ voxels) |
| sources | k ∈ {1..4}, chosen by the controller before the sources are drawn; i.i.d. uniform on D(G) (FREE, certified clearance > 2⁻²⁶); anti-duplicate 2⁻²⁰ (L∞) |
| source exclusion | every source outside every accepted **closed** window box [o/N, (o + L)/N]³, including sources with uninformative channels |
| informative channel | lit FREE and dark FREE receivers both present for that source; m(W) = number of informative channels; every accepted window has m ≥ 1 |
| balancing | global target 25% per m = 1..4 over 1.2M windows; epoch-frozen controller (1000-slot epochs, lag 1, k floor 15%, c_min 0.01); raw and ω-weighted histograms reported |
| proposals | i.i.d. uniform origins per side; one independent U per proposal; accept iff A ∧ U < c(m, L, k); cap 6000 per side; repeats skipped |
| rate stream | R = 256 / 128 / 64 per side; Λ̂_L = accepts / R; training weight ω (contract default) |
| scene acceptance | all quotas within caps, otherwise attempt + 1 (≤ 3 attempts, all recorded); failures retry the same attempt |
| responses | exact LOCAL renderings only; **no global response field is computed or stored**; per window: SOLID once plus one lit channel per physical source; packed bits (Z, Y, X), zstd |
| posterior target | Uniform(F_W^j) over D(G) minus the closed window box, with GLOBAL geometry; cell masses on the 128³ grid; TPE v3.1 (`rind3d_posterior`) |
| certification | rigorous `tv_bound` per label; certified ≤ 0.05; certified_loose ≤ 0.10; uncertified > 0.10 (stored, never presented as qualified) |
| label coverage | every informative slot: a certified label, or an explicit status `uncertified` / `engine_failure` / `timeout` / `memory_limit` / `unavailable`; uninformative slots: `not_computed` |
| label tiers | tier 1 nq_cap 2²⁵, ne_cap 2²⁶, gpu_cap 2²²; tier 2 retry 2²⁶ / 2²⁷; wall limits 1800 / 3600 / 7200 s (L = 64 / 128 / 256) |
| storage | per-scene atomic container; append-only checksummed label journals; consolidated per-scene label files; manifests (storage.py) |
| canonical structure | scene → window → source → (response, posterior slot, certification metadata); **never pre-flattened**; flattening only in `loader.flatten` |
| training interface | `loader.RIND3D100K` (window records with all channels); `flatten(…, weight=…)` attaches ω (a loss weight, not an input) |
| validation | `validate.py` / `verify_root.py`: quotas, exclusion, crops, counts, m/mask, physics spot checks (certified and independent predicates), labels |

**Selection law and theorem:** SELECTION_LAW_100K.md. **Local rendering study:** LOCAL_RENDERING_100K.md.
