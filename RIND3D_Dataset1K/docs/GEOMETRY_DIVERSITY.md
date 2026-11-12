# Geometry diversity: audit of the obstacle distribution, and curved and irregular families

**Data:** `GEOMETRY_STUDY_128.json`, produced by `scripts/geometry_study.py` over campaigns `geom128/<family>`.

**Design:**
* 128³ physical world;
* 4 matched scenes per family, each with k = 2 sources, seed 20261114;
* 16/24/24 windows, no thinning;
* every informative channel labelled with the CPU reference, 16 workers.

## 1. The existing (v0.1) generator

Every family is a CSG union of bounded convex polyhedra with uniform SO(3) orientations:

| Family | Bodies | Planes per body | Shapes |
|---|---|---|---|
| rotated_boxes | 2–10 | 6 | boxes, log-uniform half-sizes 0.025–0.22 |
| thin_slabs | 2–8 | 6 | plates (thickness 2⁻²⁰–2⁻⁵), i.e. thin walls, partial barriers, slits between slabs |
| prisms | 2–6 | 5–12 | 3–10-gon prisms |
| wedges | 2–6 | 5 | wedges |
| convex_polyhedra | 1–5 | 8–32 | random-normal polyhedra |
| multiscale_convex | 4–24 | mixed | small boxes, large boxes, prisms, polyhedra (scale range ≈ 30×) |
| corridors | 4–8 | 6 | parallel walls in one frame: corridors, narrow gaps, tunnels |
| barriers | 4–12 | 6 | baffles: partial barriers, corners |
| occluder_clusters | 4–16 | 6 | overlapping clusters, giving nonconvex unions |

**Coverage.** The audit covers boxes, oblique polyhedra, compound and nonconvex unions, overlaps, disconnected structures, thin walls, slits, corridors and clusters at several scales.

**Not covered:**
* curved or rounded surfaces;
* very large tall occluders spanning the world;
* holes that are not formed by unions.

## 2. Candidate families (all polyhedral; case 1: the polyhedron IS the canonical object)

| Family | Construction |
|---|---|
| faceted_cylinders | 12/16/24-gon prisms: pillars and disks |
| faceted_spheres | Fibonacci polyhedra with 20/32/48 planes |
| rounded_boxes | 26 planes, chamfered edges and corners |
| curved_walls | arcs of 4–8 thin slabs, i.e. curved partial barriers |
| mixed_irregular | a mixture of all kinds |

The posterior engine stays exact relative to these declared worlds. No smooth object is approximated, so there is no approximation term.

## 3. Results (128³; labels at k = 2; CPU-s per informative label)

| family | planes/body mean (max) | bodies | physical stage s/scene | label mean | p95 | max | certified | multimodal labels | lit fraction p10–p90 |
|---|---|---|---|---|---|---|---|---|---|
| wedges | 5 (5) | 4.2 | 0.25 | 0.41 | 1.6 | 3.5 | 100% | 4% | 0.09–1.00 |
| prisms | 8.4 (12) | 4.2 | 0.26 | 0.55 | 2.3 | 9.6 | 100% | 4% | 0.24–1.00 |
| rotated_boxes | 6 (6) | 7.0 | 0.37 | 0.68 | 2.3 | 11 | 100% | 3% | 0.09–0.99 |
| convex_polyhedra | 14.5 (32) | 3.2 | 0.29 | 1.00 | 3.1 | 7.2 | 100% | 4% | 0.08–0.99 |
| barriers | 6 (6) | 9.0 | 0.28 | 1.01 | 3.7 | 8.1 | 100% | 7% | 0.04–0.99 |
| occluder_clusters | 6 (6) | 11.5 | 0.28 | 1.04 | 3.9 | 8.4 | 100% | 7% | 0.05–0.99 |
| multiscale_convex | 8.1 (32) | 16.2 | 0.37 | 1.44 | 4.7 | 10 | 100% | 9% | 0.02–0.98 |
| corridors | 6 (6) | 6.2 | 0.26 | 2.00 | 5.5 | 201 | 100% | 9% | 0.10–1.00 |
| thin_slabs | 6 (6) | 6.0 | 0.26 | 3.32 | 3.2 | 422 | 100% | 5% | 0.14–0.99 |
| **rounded_boxes** | 26 (26) | 4.2 | 0.45 | **3.59** | 9.7 | 149 | 100% | 4% | 0.19–1.00 |
| **faceted_cylinders** | 18.5 (26) | 4.2 | 0.35 | **3.60** | 5.5 | 402 | 100% | 6% | 0.59–1.00 |
| **curved_walls** | 6 (6) | 12.2 | 0.28 | **5.97** | 31 | 227 | 100%, plus 1 label unfinished after > 21 min | 6% | 0.28–1.00 |
| **faceted_spheres** | 32.7 (48) | 4.2 | 0.60 | **6.57** | 19 | 35 | 100% | 4% | 0.31–1.00 |
| mixed_irregular | — | — | — | not labelled; it is a mixture of the families above | | | | | |

**Readings:**
* **Cost scales with the number of planes and silhouette edges.** It also scales with many thin bodies, which give long dual-tree expansion fronts. The new families cost **2.7–5× the v0.1 mean label** (≈ 1.3 s). Curved walls add an extreme tail.
* **Posterior structure barely changes.** The fraction of multimodal posteriors stays at 3–9% in every family.
* **Rounded and curved obstacles are less informative.** Their lit fractions are higher, i.e. there is less shadow per channel.
* **The observational diversity gained is visual and geometric, not structural.**

## 4. Decision for the 1K release

1. **The five new families are excluded.**
   * The 1K campaign is labelling-bound: about 140 h on this machine with the v0.1 families (OPTIMIZATION_REPORT §A).
   * Adding any of the new families raises that cost in proportion to its share times 2.7–5×.
   * Curved walls also violate the heavy-tail criterion.
2. **They stay in `rind3d_1k/families.py`.** The generators produce certified geometry and pass the v0.1 acceptance gate, ready for a future release once the labeller is faster.
3. **Recommended future work:**
   * low-facet variants: 8–12-gon cylinders, 14-plane rounded boxes;
   * a cost-aware family mix;
   * repeating this study at 1024³ before adoption.
