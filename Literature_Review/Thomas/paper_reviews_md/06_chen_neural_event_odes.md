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
