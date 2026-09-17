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
