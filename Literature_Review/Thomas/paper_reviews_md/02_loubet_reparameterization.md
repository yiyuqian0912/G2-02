# Loubet, Holzschuch & Jakob (2019): Reparameterizing Discontinuous Integrands for Differentiable Rendering

## What methodology the paper introduces

Loubet, Holzschuch and Jakob solve a related differentiable-rendering problem in a different way. Instead of explicitly sampling every discontinuity, they **change coordinates around the discontinuity**.

Suppose a rendering integral contains a boundary that moves when a scene parameter \(\theta\) changes. Direct differentiation is difficult because the discontinuity itself moves. Their method introduces a parameter-dependent transformation

\[
x=T(y,\theta),
\]

chosen so that, in the new coordinate system, the moving boundary becomes approximately stationary. Standard automatic differentiation can then be applied more effectively.

A simple intuition is a moving step function. If a step occurs at \(x=\theta\), changing coordinates to \(y=x-\theta\) places the step at \(y=0\). The discontinuity still exists physically, but the numerical coordinate system now moves with it.

The important methodological lesson is therefore: **do not necessarily smooth the event; instead, choose coordinates that are aligned with it.**

## What can be borrowed for our project

Our inverse problem may need to optimize source position or direction across shadow boundaries. Ordinary gradient descent can behave badly when the current parameterization cuts across a hard visibility event.

The reparameterization idea suggests that, once an event surface has been learned, we can construct **event-aligned coordinates**. One coordinate measures motion normal to the event, while the remaining coordinates move tangentially along it.

This could make local source refinement more stable because the optimizer knows which direction approaches or crosses the event and which directions remain within the same visibility mechanism.

It also provides a useful interpretation of the learned event field \(g_j\): rather than treating \(g_j\) only as a classifier, we can use it as a coordinate measuring signed distance or signed margin from a visibility transition.

## What the application would look like technically

Assume the event network predicts

\[
g_j(S,q,d),
\qquad
\Gamma_j=\{d:g_j(S,q,d)=0\}.
\]

Near a predicted event point \(d^*\), compute the normal

\[
n_j
=
\frac{\nabla_d g_j(d^*)}
{\|\nabla_d g_j(d^*)\|}.
\]

Construct a local orthonormal basis

\[
B=[n_j,\;t_2,\ldots,t_p],
\]

where \(t_2,\ldots,t_p\) span directions tangent to the event surface. Define local coordinates \(z\) by

\[
d=d^*+Bz.
\]

Then:

- \(z_1\) moves approximately **across** the event;
- \(z_2,\ldots,z_p\) move approximately **along** the event.

An inverse optimizer could first optimize tangential coordinates while remaining in the same mechanism, then deliberately change \(z_1\) if evidence suggests that the solution lies in the neighboring regime.

A more learned version could define

\[
z=T_\theta(S,q,d),
\]

with the first coordinate trained to approximate the event margin:

\[
z_1 \approx g_j(S,q,d).
\]

The local illumination or likelihood model could then be conditioned on the event-aligned representation \(z\), rather than on raw global source coordinates.

This does **not** replace the event model. Instead, it uses the event model to construct a better coordinate system for optimization.

## Main limitation for our setting

The paper's reparameterization is designed to make rendering derivatives tractable near discontinuities. It does not itself discover a global event graph, identify all future boundaries, or assign every event to an obstacle. Those pieces still need to come from our structure-conditioned event model.

## Nature-style citation

Loubet, G., Holzschuch, N. & Jakob, W. Reparameterizing discontinuous integrands for differentiable rendering. *ACM Trans. Graph.* **38**, 228:1–14 (2019). https://doi.org/10.1145/3355089.3356510
