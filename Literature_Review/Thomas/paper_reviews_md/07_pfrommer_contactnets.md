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
