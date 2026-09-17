# Paper-by-Paper Literature Reviews for the Event-Geometry Project

These notes expand the references cited in the earlier brief literature review. Each file explains:

- the paper's relevant methodology in simple language;
- what can be borrowed for the project;
- a technical sketch of how the borrowed idea could be implemented; and
- a complete Nature-style reference.

The technical application sections are **project adaptations/inferences**, not claims that the original papers implemented the proposed RIND/G2-02 architecture.

## Files

1. [Li et al. — Edge-sampling differentiable ray tracing](01_li_edge_sampling.md)
2. [Loubet et al. — Reparameterized differentiable rendering](02_loubet_reparameterization.md)
3. [Gigus, Canny & Seidel — Aspect graphs](03_gigus_aspect_graphs.md)
4. [Sethian — Level-set methodology note](04_sethian_level_sets.md)
5. [Kong et al. — Saltation matrices](05_kong_saltation.md)
6. [Chen, Amos & Nickel — Neural Event ODEs](06_chen_neural_event_odes.md)
7. [Pfrommer, Halm & Posa — ContactNets](07_pfrommer_contactnets.md)
8. [Luger et al. — `starry`](08_luger_starry.md)
9. [Luger et al. — Analytic reflected-light curves](09_luger_reflected_light.md)

## How the methods fit together

A useful division of labor is:

\[
\text{ray tracing}
\rightarrow
\text{event supervision}
\rightarrow
\text{implicit event fields}
\rightarrow
\text{event graph}
\rightarrow
\text{cross-event solver}.
\]

- **Li et al.**: physics-aware gradients and boundary supervision.
- **Loubet et al.**: event-aligned coordinates for optimization.
- **Gigus et al.**: partition parameter space into stable mechanisms and transitions.
- **Sethian**: represent each event boundary as a zero level set.
- **Kong et al.**: update sensitivity when a boundary is crossed.
- **Chen et al.**: learn event functions and differentiate through event roots.
- **ContactNets**: learn smooth geometric causes of discontinuous behavior.
- **`starry`**: preserve exact boundary structure and probabilistic inverse solutions.
- **Reflected-light model**: handle multiple interacting visibility/illumination boundaries.


---

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


---

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


---

# Gigus, Canny & Seidel: Aspect Graphs and Visual-Event Partitions

## What methodology the paper introduces

The aspect-graph approach represents how the visible structure of an object changes as the viewpoint moves. Instead of storing every possible image independently, the method divides viewpoint space into **regions with the same qualitative visibility structure**.

Inside one region, the topology of the visible line drawing does not change. When the viewpoint crosses a boundary, a **visual event** occurs and the visible structure changes. These regions can be represented as nodes in a graph, while event crossings become edges between neighboring nodes.

The main idea is simple:

> parameter space can be divided into stable mechanism regions, and the boundaries between those regions are meaningful geometric events.

This is very close to what our project needs, except that our changing parameter is usually a light-source position or direction rather than a camera viewpoint.

The original exact construction is computationally expensive for complex geometry, so the method is more valuable to us as a **representation idea** than as an algorithm to reproduce exactly.

## What can be borrowed for our project

We can treat source or drive space in the same way aspect graphs treat viewpoint space.

For a query point \(q\), define two source configurations \(d_1\) and \(d_2\) as belonging to the same mechanism if the relevant ray/occlusion structure is the same:

\[
d_1 \sim_q d_2
\quad\Longleftrightarrow\quad
\sigma_q(S,d_1)=\sigma_q(S,d_2).
\]

Then the source-parameter domain is divided into cells

\[
\mathcal D
=
\bigcup_k D_{q,k}\cup\Gamma_q,
\]

where each \(D_{q,k}\) is a stable visibility regime and \(\Gamma_q\) is the union of event boundaries.

This immediately gives a useful inverse-search strategy. Instead of performing one continuous optimization over all source parameters, the solver can:

1. optimize inside the current mechanism;
2. identify nearby event boundaries;
3. cross one event;
4. enter a neighboring mechanism;
5. continue optimization there.

The graph also provides a natural place to store **attribution**: an edge can record which obstacle or geometric relation caused the transition.

## What the application would look like technically

A practical learned version should be **sparse and local**, not a complete exact aspect graph.

For training scenes:

1. Sample source configurations \(d\) over a coarse grid or adaptive set.
2. For each \(d\), ray-trace a mechanism signature, for example:
   - visible/blocked state for selected query points;
   - first-hit obstacle identity;
   - active set of occluders;
   - source-specific visibility pattern.
3. Group nearby configurations with the same signature into provisional mechanism cells.
4. Detect transitions between neighboring samples.
5. Label each transition with the obstacle or primitive that changed the mechanism.

The learned representation could then store

\[
k
\xrightarrow[\Gamma_j]{a_j}
\ell,
\]

where:

- \(k\) is the current mechanism;
- \(\ell\) is the neighboring mechanism;
- \(\Gamma_j\) is the predicted event surface;
- \(a_j\) is the responsible obstacle or structural relation.

At inference time, the model does not need the whole global graph. Given the current \(d\), it can retrieve a small set of candidate neighboring events and predict their margins \(g_j(S,q,d)\). The nearest roots of these functions define the locally reachable transitions.

This gives the project a structured answer to the question, **"If the current source explanation is wrong, what qualitatively different explanation should the solver try next?"**

## Main limitation for our setting

Classical aspect graphs assume explicit geometry and can have severe combinatorial growth. Our project should therefore borrow the **cell-and-event representation**, not attempt a complete exact construction for every scene.

## Nature-style citation

The report cites the 1988 Berkeley technical report. A peer-reviewed journal version was later published:

Gigus, Z., Canny, J. & Seidel, R. Efficiently computing and representing aspect graphs of polyhedral objects. *IEEE Trans. Pattern Anal. Mach. Intell.* **13**, 542–551 (1991). https://doi.org/10.1109/34.87341

Original technical report: Gigus, Z., Canny, J. F. & Seidel, R. *Efficiently Computing and Representing Aspect Graphs of Polyhedral Objects*. Technical Report UCB/CSD-88-432 (University of California, Berkeley, 1988).


---

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


---

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


---

# Chen, Amos & Nickel (2021): Learning Neural Event Functions for Ordinary Differential Equations

## What methodology the paper introduces

Neural ODEs normally evolve a continuous state until a known end time. Chen, Amos and Nickel extend this idea by allowing a neural network to learn **when an event should occur**.

The state follows continuous dynamics

\[
\dot z(t)=f_\theta(t,z(t)),
\]

while a learned scalar event function

\[
g_\phi(t,z(t))
\]

is monitored. When

\[
g_\phi(t,z(t))=0,
\]

an event is triggered. The event may terminate the trajectory or cause an instantaneous update before integration continues.

The important idea is that the event time is treated as an **implicit root** rather than as a fixed label. Because the root condition is differentiable through the implicit-function theorem, the event detector can be trained end to end.

In simple language, the system learns a smooth function whose zero crossing means **"something discrete happens here."**

## What can be borrowed for our project

The full ODE framework is probably unnecessary for the current static illumination problem. The useful part is the learned event function itself.

We can directly transfer

\[
g_\phi(t,z)=0
\]

to

\[
g_{\phi,j}(S,q,d)=0,
\]

where the zero crossing indicates a visibility change caused by candidate obstacle \(j\).

This is helpful when exact event positions are not always provided as labels. The model can learn event surfaces indirectly from before/after observations, ray states, or inverse-task losses.

The paper also gives a principled way to differentiate **through the event location**, which is useful if the source estimate is optimized end to end.

## What the application would look like technically

Suppose the current source configuration is \(d_0\) and we want to search along direction \(v\). Define a one-dimensional continuation path

\[
d(s)=d_0+s v.
\]

For candidate event \(j\), define

\[
h_j(s)=g_{\phi,j}(S,q,d_0+s v).
\]

The next event is a root

\[
h_j(s^*)=0.
\]

A practical pipeline is:

1. The network predicts several candidate event functions \(g_{\phi,j}\).
2. For each candidate, evaluate \(h_j(s)\) over a short forward interval.
3. If the sign changes, bracket a root.
4. Use bisection, secant, or differentiable root finding to estimate \(s^*\).
5. Compare the predicted pre-event and post-event field responses.
6. Backpropagate through the root condition so the event-function parameters improve.

Training need not always use exact event coordinates. A weaker supervision pair could contain

\[
(d^-,Y^-),\qquad(d^+,Y^+)
\]

known to be on different sides of a visibility switch. The model can be required to place at least one zero of \(g_j\) between them.

A useful loss could combine:

- sign consistency on both sides;
- correct event ordering;
- agreement with first-hit ray identity;
- reconstruction error after crossing;
- regularization on event-surface smoothness.

If the project later includes time-dependent source trajectories or moving geometry, the complete Neural Event ODE machinery becomes more directly applicable.

## Main limitation for our setting

Neural Event ODEs do not inherently model visibility, occlusion, or structural attribution. The event function must be conditioned on the scene geometry, and obstacle identity must be added explicitly.

## Nature-style citation

Chen, R. T. Q., Amos, B. & Nickel, M. Learning neural event functions for ordinary differential equations. In *International Conference on Learning Representations (ICLR)* (2021). https://arxiv.org/abs/2011.03902


---

# Pfrommer, Halm & Posa (2021): ContactNets

## What methodology the paper introduces

Contact dynamics are difficult to learn because the motion can change suddenly at collision, sticking, or sliding events. ContactNets avoids directly fitting the discontinuous state transition. Instead, it learns **smooth geometric quantities that cause the discontinuity**.

For each potential contact, the method learns a signed inter-body distance

\[
\phi_i(q).
\]

The surface

\[
\phi_i(q)=0
\]

represents contact, and its gradient gives the contact normal/Jacobian. Known contact physics is then applied around that learned geometry.

The main lesson is extremely relevant to our project:

> when the output is discontinuous, learn a smooth latent geometry whose zero crossing triggers the discontinuity.

This is much easier to generalize than trying to make a neural network directly memorize every possible jump.

## What can be borrowed for our project

Replace **signed contact distance** with a **signed visibility-event margin**.

For candidate obstacle \(j\), learn

\[
g_j(S,q,d),
\]

where \(g_j=0\) means that the ray/source configuration is exactly at the event where obstacle \(j\) changes the visibility mechanism.

The analogy is:

\[
\text{ContactNets signed distance}
\longleftrightarrow
\text{ray-clearance/event margin},
\]

\[
\phi_i=0
\longleftrightarrow
g_j=0,
\]

\[
\nabla_q\phi_i
\longleftrightarrow
\nabla_dg_j.
\]

If each event field is indexed by an obstacle or primitive, **attribution comes naturally**. The model does not merely say that "some boundary exists"; it says which structural candidate owns that boundary.

## What the application would look like technically

A useful architecture could contain three stages.

### 1. Candidate-occluder retrieval

A scene encoder receives geometry \(S\), query point \(q\), observation region \(R\), and current source estimate \(d\). It retrieves a small set

\[
\mathcal A_q=\{a_1,\ldots,a_M\},
\qquad M\ll |S|.
\]

This avoids predicting an event function for every obstacle in a large scene.

### 2. Obstacle-conditioned event fields

For each candidate \(a_j\), predict

\[
g_j=g_\theta(\psi(a_j,q),d),
\]

where \(\psi\) contains relative geometric features such as:

- obstacle/query relative position;
- angular relation to the source;
- distance;
- surface orientation;
- projected edge information;
- ray-depth ordering.

The field should be smooth even though the visible/blocked output is not.

### 3. Visibility-specific physical constraints

ContactNets uses complementarity and dissipation constraints. Our version would replace those with ray/visibility constraints, for example:

- \(g_j=0\) should coincide with a ray tangency or first-hit ordering exchange;
- the sign of \(g_j\) should agree with the side of the event;
- the attributed obstacle should agree with the ray tracer;
- a blocking obstacle should satisfy correct depth ordering;
- the predicted post-event mechanism should match a ray trace just across the boundary.

The blocking probability already present in the proposal can help choose candidates:

\[
P_{\text{block}}(o_j)
\rightarrow
\text{candidate event head }g_j.
\]

A side-conditioned decoder can then predict

\[
u_j^-,
\qquad
u_j^+,
\qquad
\Delta u_j=u_j^+-u_j^-.
\]

This gives the project a full event object: location, normal, attribution, and response change.

## Main limitation for our setting

Contact constraints do not transfer literally to light visibility. The project must design its own ray-based structural losses. Still, the architectural principle is one of the strongest matches in the literature.

## Nature-style citation

Pfrommer, S., Halm, M. & Posa, M. ContactNets: learning discontinuous contact dynamics with smooth, implicit representations. In *Proceedings of the 2020 Conference on Robot Learning* (eds Kober, J., Ramos, F. & Tomlin, C.) **155**, 2279–2291 (PMLR, 2021).


---

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


---

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
