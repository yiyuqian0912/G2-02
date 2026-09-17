# Sethian: Level-Set Methods for Implicit Event Geometry

## Source note

The research report cites Sethian's **technical explanation of level-set methods**, not a single journal paper. This review treats that source as the methodology reference. A formal book reference is provided at the end.

## What methodology the source introduces

A level-set method represents a boundary **indirectly**. Instead of storing a curve or surface as an explicit list of points, it defines a smooth scalar function

\[
\phi(z)
\]

and represents the boundary as its zero set:

\[
\Gamma=\{z:\phi(z)=0\}.
\]

The sign of \(\phi\) tells which side of the boundary a point lies on. If \(\phi\) behaves like a signed distance function, its magnitude also tells approximately how far the point is from the boundary.

A major advantage is that the boundary normal is immediately available from the gradient:

\[
n=
\frac{\nabla \phi}{\|\nabla\phi\|}.
\]

Classical level-set methods also describe how interfaces move through Hamilton–Jacobi equations. For our current static inverse problem, that time-evolution machinery is not necessarily needed. The most important part is the **implicit representation of the boundary**.

## What can be borrowed for our project

This is probably the cleanest representation for the project's learned visibility events.

Instead of asking a neural network to directly predict a discontinuous visible/blocked response, give each relevant obstacle \(j\) a smooth event-margin function

\[
g_j(S,q,d).
\]

Then:

\[
\Gamma_j=\{d:g_j(S,q,d)=0\}
\]

is the visibility-event surface in source/drive space,

\[
\text{sign}(g_j)
\]

identifies the side of the event, and

\[
n_j=
\frac{\nabla_dg_j}{\|\nabla_dg_j\|}
\]

gives the local direction normal to the event.

This representation converts a difficult discontinuous-learning problem into a smoother geometric-learning problem.

## What the application would look like technically

For each scene \(S\), query point \(q\), source configuration \(d\), and candidate obstacle \(j\), the event network predicts

\[
g_j=g_\theta(S,q,d,j).
\]

Useful training labels from the ray tracer include:

- samples known to lie on an event boundary;
- samples known to lie on either side;
- first-hit obstacle identity;
- approximate ray-clearance values;
- finite-difference event normals.

A basic loss could be

\[
\mathcal L_g
=
\lambda_0\mathcal L_{\text{zero}}
+
\lambda_s\mathcal L_{\text{sign}}
+
\lambda_n\mathcal L_{\text{normal}}
+
\lambda_e\mathcal L_{\text{eikonal}}.
\]

Possible terms are:

\[
\mathcal L_{\text{zero}}
=
|g_j(d^*)|,
\]

for known event samples \(d^*\);

\[
\mathcal L_{\text{sign}}
=
\max(0,m-s\,g_j(d)),
\]

where \(s\in\{-1,+1\}\) is the correct side label; and, if a signed-distance-like field is desired,

\[
\mathcal L_{\text{eikonal}}
=
\left(\|\nabla_dg_j\|-1\right)^2.
\]

At inference time, the system can use \(g_j\) for three different tasks:

1. **event detection:** small \(|g_j|\) means a transition is close;
2. **event navigation:** \(\nabla_dg_j\) gives a direction toward or away from the event;
3. **attribution:** obstacle-indexed event heads tell which structural element owns the boundary.

This can become the central geometric representation that later methods—aspect graphs, root finding, and saltation—operate on.

## Main limitation for our setting

A level-set representation does not automatically know which obstacle matters or what illumination jump occurs after crossing the event. The scene encoder, candidate-occluder retrieval, and side-conditioned field model must supply those pieces.

## Nature-style references

**Source cited in the research report (technical web source, not a paper):**

Sethian, J. A. Technical explanation of level set methods. *University of California, Berkeley*. https://math.berkeley.edu/~sethian/2006/Semiconductors/ieee_level_set_explain_technical.html (accessed 17 September 2026).

**Stable formal reference for the methodology:**

Sethian, J. A. *Level Set Methods and Fast Marching Methods: Evolving Interfaces in Computational Geometry, Fluid Mechanics, Computer Vision, and Materials Science*, 2nd edn (Cambridge University Press, 1999).
