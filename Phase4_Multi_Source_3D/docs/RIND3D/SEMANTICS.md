# Canonical geometry and physical response semantics

Terms (driver source, binary response, response-active / response-inactive, response interface, source-specific
response channel) and their legacy names are defined in [GLOSSARY.md](GLOSSARY.md).

## Geometry

* **The solid is a finite union of bounded closed convex polyhedra strictly inside the open world (0,1)³.** Leaves are `affine_box_v1` or `convex_halfspaces_v1` under one CSG UNION. There are no differences.
* **Parameters are binary64 numbers read as exact rationals.** `rind3d.rendering.polyhedra.decode_convex` gives exact halfspace rows a·x + d ≤ 0.
* **The posterior engine uses the binary64 roundings of these rows,** renormalised. Decisions are made with explicit margins (Phase-1 METHOD.md).
* **Families.**
  * The nine v0.1 families of `rind3d.lite_generators`.
  * The polyhedral families of `RIND3D_Dataset1K/rind3d_1k/families.py` (faceted cylinders and spheres, rounded boxes, curved walls, mixed). These are canonical polyhedra, not approximations of smooth bodies.

## Driver sources

* **Legal domain.** D(G) = {p ∈ (0,1)³ : p is outside the closed solid, with certified clearance > 2⁻²⁶}.
* **Intensity.** Driver sources have unit intensity.
* **Source-specific response channels.** Each driver source has its own binary response channel. Channels of different
  sources are stored separately and never superposed.

## Binary response by visibility (`closed-solid-closed-segment-v1`)

* **Rule.** R(s, x) = 1 iff the closed segment [s, x] does not meet the closed solid. Contact and tangency block.
  * A FREE receiver with R = 1 is *response-active* (stored as `lit`); one with R = 0 is *response-inactive* (legacy:
    dark).
  * Visibility is the only response mechanism implemented.
* **Receivers.** Receivers are voxel centres (i + ½)/N. SOLID receivers have no response: they are never
  response-active, and they are not counted as response-inactive.
* **Certification of the renderers.**
  * The CPU point predicate uses outward-rounded interval clipping with a GMP rational fallback.
  * The independent predicate uses a rational Fourier–Motzkin projection of each convex body's inactive set, i.e. the
    receivers that body occludes from the source (a "shadow volume" in graphics terms).
  * The CUDA field renderer uses directed-rounding intervals, and recomputes unknown bits exactly on the CPU.
* **Validation.** The 1K validator compares stored crops with the CPU and independent predicates at random receivers.

## Windows

* **Shape.** Windows are axis-aligned cubes [o, o + L)³ of voxels.
* **Exclusion box.** Source exclusion uses the **closed** continuous box [o/N, (o + L)/N]³.
* **Precedents.** This follows the v0.2/v0.3 convention. v0.1 used a half-open box in voxel units.
