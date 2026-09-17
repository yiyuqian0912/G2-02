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
