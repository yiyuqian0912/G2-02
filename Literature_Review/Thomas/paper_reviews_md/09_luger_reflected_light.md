# Luger et al. (2022): Analytic Light Curves in Reflected Light

## What methodology the paper introduces

This work extends analytic light-curve calculations to **reflected light**, where the measured signal depends on more than simple visibility. A point on a spherical body contributes only if it is both visible to the observer and illuminated by the source.

The observed flux can therefore be viewed schematically as an integral over

\[
\mathcal V(\Theta)\cap \mathcal I(\Theta),
\]

the intersection of the visible region and the illuminated region.

This creates several moving geometric boundaries:

- the occultation boundary;
- the illumination terminator;
- intersections between these boundaries.

The paper develops analytic calculations for phase curves, occultations, and more general scattering models. The methodology shows how a physical response can remain smooth inside geometric regions while its derivatives change when the active integration domain changes.

## What can be borrowed for our project

This is particularly useful because our project can also contain **multiple interacting boundaries**.

For one source, the relevant state may depend on:

- whether the source-to-query ray is clear;
- which obstacle is first along the ray;
- whether the query lies inside a source-specific illuminated region;
- whether multiple sources contribute simultaneously.

The reflected-light formulation suggests that we should not always represent visibility with one binary variable. Instead, we can explicitly model the intersection of several physical conditions.

It is also a warning for later stages of the project: event surfaces can **intersect**. At those intersections, one normal and one simple jump may no longer be sufficient.

## What the application would look like technically

For each source \(L_k\), define a contribution

\[
I_k(q)
=
V_k(S,q,d_k)\,
a_k A(L_k,q).
\]

Instead of only predicting total illumination, maintain source-specific geometric states where useful:

\[
Y_k(q)
\in
\{\text{clear},\text{blocked by }j,\ldots\}.
\]

For each candidate structural event, learn a margin

\[
g_{k,j}(S,q,d_k).
\]

The event surface is

\[
\Gamma_{k,j}
=
\{d_k:g_{k,j}=0\}.
\]

If multiple conditions are active, the model can combine them through a structured mechanism state

\[
\sigma_q
=
(\sigma_{q,1},\ldots,\sigma_{q,K}).
\]

An important extension is to detect **event intersections**:

\[
g_i(d)=0,
\qquad
g_j(d)=0.
\]

Near such a point, the inverse solver should not assume that one unique event normal explains all changes. A later mathematical extension could use multiple candidate normals or a generalized/set-valued sensitivity.

The reflected-light analogy also suggests a clean architecture:

1. predict smooth source-specific attenuation/intensity;
2. predict structured geometric masks/events separately;
3. combine them only at the final physical response stage.

That mirrors the proposal's separation between geometric visibility and continuous attenuation, while making the boundary structure richer and more explicit.

## Main limitation for our setting

The paper assumes spherical celestial bodies and highly structured illumination geometry. Our scenes are much more general. The useful transfer is therefore the **decomposition into intersecting geometric domains**, not the exact analytic formulas.

## Nature-style citation

Luger, R., Agol, E., Bartolić, F. & Foreman-Mackey, D. Analytic light curves in reflected light: phase curves, occultations, and non-Lambertian scattering for spherical planets and moons. *Astron. J.* **164**, 4 (2022). https://doi.org/10.3847/1538-3881/ac4017
