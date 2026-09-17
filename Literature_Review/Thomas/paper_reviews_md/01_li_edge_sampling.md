# Li et al. (2018): Differentiable Monte Carlo Ray Tracing through Edge Sampling

## What methodology the paper introduces

Li et al. address a basic problem in differentiable rendering: **visibility is not smooth**. If a light ray moves slightly, it may suddenly change from hitting one object to hitting another, or from being blocked to being clear. Standard automatic differentiation works well for smooth quantities, but it misses the special derivative contribution created at these visibility boundaries.

The paper handles this by separating the derivative of the rendering integral into two parts:

1. a normal smooth derivative inside regions where visibility does not change; and
2. a **boundary term** concentrated on occlusion edges.

For triangle meshes, these discontinuities are tied to projected geometric edges. The renderer therefore samples both ordinary light transport and relevant visibility edges. In simple terms, instead of pretending the shadow edge is smooth, the method explicitly asks: **where is the edge, and how much does moving it change the rendered result?**

This is important for inverse rendering because gradients with respect to light position, camera parameters, geometry, or materials become physically meaningful even when occlusion is present.

## What can be borrowed for our project

Our project also has hard visibility switches. A source direction or position can move smoothly while the illumination at a query point changes abruptly because an obstacle starts or stops blocking the source.

The most useful idea to borrow is **not necessarily the full renderer**, but the treatment of visibility boundaries as first-class objects. The method can serve as a **physics teacher** for the learned event model:

- detect where a ray's first-hit object changes;
- identify which geometric primitive caused the change;
- estimate a physically correct boundary contribution;
- generate supervision around hard shadow transitions;
- verify whether the learned event normal and jump direction agree with true ray geometry.

This is especially valuable because the project already has synthetic scene geometry and ray-intersection information. The simulator can therefore provide much more than binary visible/blocked labels: it can also provide event locations and local boundary behavior.

## What the application would look like technically

Let the scene be \(S\), a query point be \(q\), and the source or drive parameter be \(d\). Suppose the learned system predicts an event function for candidate obstacle \(j\),

\[
g_j(S,q,d).
\]

The zero set

\[
g_j(S,q,d)=0
\]

should correspond to the source configuration at which the relevant visibility mechanism changes.

A practical training pipeline would be:

1. **Ray-trace nearby source configurations.** For a sampled \(d\), perturb it to \(d+\epsilon v\) in several directions.
2. **Detect event crossings.** If the first-hit obstacle or visibility state changes between the two configurations, bracket an event.
3. **Refine the boundary location.** Use bisection or another root search in drive space to estimate \(d^*\), the source configuration where the switch occurs.
4. **Record attribution.** Save the obstacle or edge responsible for the transition.
5. **Generate local supervision.** Train \(g_j(d^*)\approx 0\) and require opposite signs on the two sides of the event.
6. **Compare normals.** The learned normal
   \[
   n_j=\frac{\nabla_d g_j}{\|\nabla_d g_j\|}
   \]
   can be checked against a ray-traced boundary direction or finite-difference event geometry.
7. **Use the renderer as a verifier during inversion.** When the inverse solver proposes crossing an event, the exact ray tracer confirms the post-event visibility and illumination.

A possible event loss is

\[
\mathcal L_{\text{event}}
=
\lambda_0 |g_j(d^*)|
+
\lambda_s \mathcal L_{\text{sign}}
+
\lambda_n \mathcal L_{\text{normal}}.
\]

The key role of this paper in the project is therefore **high-quality physical supervision for discontinuity geometry**, rather than being the final learned representation itself.

## Main limitation for our setting

Edge sampling gives excellent local derivatives around known visibility boundaries, but it does not automatically construct a reusable global map of where future events lie in source-parameter space. Our event-field model must still learn that structure.

## Nature-style citation

Li, T.-M., Aittala, M., Durand, F. & Lehtinen, J. Differentiable Monte Carlo ray tracing through edge sampling. *ACM Trans. Graph.* **37**, 222:1–11 (2018). https://doi.org/10.1145/3272127.3275109
