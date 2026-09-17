# Luger et al. (2019): `starry` — Analytic Occultation Light Curves

## What methodology the paper introduces

`starry` studies an inverse problem in which part of a luminous spherical object is hidden by another object. The observed signal is integrated light rather than a direct image of every surface point.

The method represents the surface-intensity map using spherical harmonics and derives analytic, closed-form expressions for the total visible flux during occultation. Because the integration boundary is handled analytically, the calculation is both fast and differentiable with respect to physical parameters.

The paper also emphasizes **probabilistic inference**. Different maps or geometric configurations can explain similar light curves, so the goal is often a posterior distribution rather than one forced point estimate.

This is relevant to our project because both problems infer hidden structure from incomplete illumination information affected by occlusion.

## What can be borrowed for our project

Three ideas are especially useful.

### 1. Treat boundaries as physical geometry

`starry` does not simply blur away occultation boundaries. It integrates over the correct visible domain. This reinforces the idea that our project should explicitly represent visibility events rather than rely only on a soft approximation.

### 2. Use structured basis representations where possible

The exact spherical-harmonic basis is specific to spheres, but the broader idea is useful: if part of the field can be expressed in a compact basis, inference may become more stable and efficient.

For our model this could mean:

- low-dimensional basis functions for smooth illumination inside a mechanism;
- learned spatial basis functions;
- coarse-to-fine voxel/grid bases;
- mode-specific decoders rather than one unrestricted field network.

### 3. Keep inverse solutions probabilistic

The proposal already predicts a source probability field rather than only one source point. `starry` supports this philosophy: partial observations can be genuinely ambiguous, and a good inverse model should preserve multiple plausible solutions.

## What the application would look like technically

Our forward illumination model is

\[
I(q)
=
\sum_k
V(S,q,L_k)\,
a_k A(L_k,q).
\]

The direct analogue to `starry` is not to copy spherical harmonics, but to separate:

1. a **smooth response law** inside a visibility mechanism; and
2. an **explicit visibility domain** determining where that response contributes.

For mechanism \(m\), write a smooth field decoder

\[
I_m(x)
=
\sum_{r=1}^{R} c_{m,r}\,\psi_r(x),
\]

where \(\psi_r\) are fixed or learned basis functions. Visibility/event geometry determines which mechanism is active.

For inversion, instead of returning only

\[
\hat d=\arg\min_d \mathcal L(d),
\]

maintain a distribution such as

\[
p(d\mid S,I_{\text{obs}},M_R).
\]

This can be represented by the proposal's source-probability field or by samples from an approximate posterior. Event-aware gradients can be used within each smooth mechanism, while mechanism transitions are handled separately.

A useful experiment borrowed from astronomical inversion would be to compare:

- single best-source prediction;
- multimodal source probability;
- posterior samples conditioned on the same local observation.

If two distant source configurations produce nearly identical local illumination, the model should retain both rather than average them into an unphysical source location.

## Main limitation for our setting

`starry` obtains its analytic efficiency from very strong spherical geometry and harmonic assumptions. Arbitrary urban or obstacle scenes do not have this structure. We should borrow its **boundary-aware and probabilistic inversion principles**, not its exact solver.

## Nature-style citation

Luger, R., Agol, E., Foreman-Mackey, D., Fleming, D. P., Lustig-Yaeger, J. & Deitrick, R. `starry`: analytic occultation light curves. *Astron. J.* **157**, 64 (2019). https://doi.org/10.3847/1538-3881/aae8e5
