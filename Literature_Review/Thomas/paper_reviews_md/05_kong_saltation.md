# Kong et al. (2024): Saltation Matrices for Cross-Event Sensitivity

## What methodology the paper introduces

Hybrid systems have two kinds of behavior:

- smooth evolution inside a mode; and
- discrete changes when an event occurs.

A normal Jacobian describes how small perturbations behave while the system stays inside one smooth mode. The problem is that a perturbation can also change **where or when the event happens**. That effect is missed if one differentiates only the discrete reset map.

The saltation matrix corrects this. For a transition from mode \(I\) to mode \(J\), with event guard

\[
g(t,x)=0
\]

and reset map

\[
x^+=R(t,x^-),
\]

the saltation matrix gives the first-order sensitivity update across the event. Its extra correction term depends on the gradient of the event guard and the dynamics on the two sides.

In simple language, it answers:

> if I perturb the system slightly before a discontinuity, what is the correct first-order perturbation after the discontinuity?

## What can be borrowed for our project

This is the most direct mathematical answer to the phrase **"discontinuous Jacobian."**

For our project, however, saltation should be used **after** an event surface has been found. It does not discover the visibility boundary by itself.

A useful separation is:

- **ordinary Jacobian**: sensitivity while the source remains inside one visibility mechanism;
- **event function \(g=0\)**: location and normal of the mechanism boundary;
- **saltation-style update**: sensitivity immediately after the boundary is crossed.

That division prevents us from forcing one ordinary Jacobian to describe two fundamentally different effects.

## What the application would look like technically

The original problem is a static map from source parameters \(d\) to field response \(u\). To use saltation language, introduce a continuation parameter \(s\) and a path through source space,

\[
d=d(s),
\qquad
\frac{dd}{ds}=v(s).
\]

Define an augmented state

\[
z(s)=(d(s),u(s)).
\]

Inside one visibility mechanism \(k\), \(u\) changes smoothly according to a mode-specific model. An event occurs when

\[
g_j(S,q,d(s))=0.
\]

The inverse solver would then work as follows:

1. Start at source estimate \(d_0\).
2. Use the ordinary within-mode Jacobian \(J_k\) to improve the fit while all relevant \(g_j\) remain away from zero.
3. If a candidate event margin approaches zero, root-find the crossing \(s^*\).
4. Evaluate the event normal \(n_j=\nabla_d g_j(d^*)\).
5. Evaluate the predicted field behavior immediately before and after the event.
6. Apply a saltation-style sensitivity update to the perturbation/covariance/gradient carried by the inverse solver.
7. Continue optimization in the neighboring mechanism.

For a standard hybrid transition the saltation matrix is

\[
\Xi
=
D_xR
+
\frac{
(F^+-D_xR\,F^- -D_tR)D_xg
}{
D_tg+D_xgF^-
}.
\]

Our exact static adaptation may not use this formula unchanged. The key borrowed structure is the explicit correction for **event displacement**. A project-specific version would use the learned event normal, continuation direction, and side-conditioned illumination models.

This method is therefore most useful in the **solver**, not necessarily in the neural architecture.

## Main limitation for our setting

Saltation assumes a known event guard and a well-defined crossing. Tangential/grazing events and simultaneous intersections of several event surfaces are harder. Also, the paper is formulated for hybrid dynamics, so applying it to a static ray problem requires the continuation reformulation described above.

## Nature-style citation

Kong, N. J., Payne, J. J., Zhu, J. & Johnson, A. M. Saltation matrices: the essential tool for linearizing hybrid dynamical systems. *Proc. IEEE* **112**, 585–608 (2024). https://doi.org/10.1109/JPROC.2024.3440211
