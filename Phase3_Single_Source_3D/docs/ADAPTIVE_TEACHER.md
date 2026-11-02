# Adaptive physical compatibility in 3D source space

An observation O contains a sampled local response, local solid mask and window coordinates. The teacher additionally knows canonical full-scene geometry G. For a proposed FREE source s outside the observation, physical cost C_O(s) compares its rerendered response with the observation. The current transparent cost is disagreement fraction over FREE receivers. With all local voxel centers it is a full-window cost; with 64 selected FREE centers it is only a sampled estimate of that full cost. Exact labels at the probes do not certify unobserved voxels.

The teacher represents compatibility, not a single recovered coordinate. The zero-cost inverse set can be disconnected, thin, extended or continuously ambiguous. Truth and stored equivalent-source witnesses are not the support or labels of this search. The public data withholds them, and this scaffold does not need them. Precise enumeration of all compatible lights is not required to query physics, regress costs or investigate ranking.

## Four different derivatives/variations

1. **Physical candidate-source cost** C_O(s) is computed from certified visibility. For finite binary observations it is generally piecewise constant with discontinuities when rays encounter obstacle boundaries.
2. **Source-space finite differences** compare C_O(s+h) and C_O(s). A zero difference at a few samples does not show that the surrounding volume is homogeneous, smooth or safe to prune. A narrow low-cost pocket can lie between all samples.
3. **Student energy gradient** is a derivative of a learned continuous energy E_theta(s,O) with respect to source position. It belongs to the surrogate, not to exact binary physics. Smoothness of E_theta is a modeling choice.
4. **Neural parameter gradient** differentiates a chosen training loss with respect to theta. Backpropagating an energy/regression/distillation objective is not differentiating the canonical geometry oracle.

No smooth physical gradient or source uniqueness follows from any one of these objects. A single stored reference is never a uniqueness proof. Likewise, multiple observed transitions are sampled inequalities, not automatically independent exact continuous silhouette-plane equations.

## Minimal source-space octree

Start with the 1024³ source domain and split it once into **eight** 512³ cubes: offsets are 0 or 512 along each axis. This is not an initial 8×8×8 grid. A selected 512 cube splits into eight 256 cubes, then later 128, 64 and finer cubes as an experiment chooses. These aligned source-space regions do not constrain the arbitrary integer origins of observation windows.

Each visited region samples a constant k=4 source positions uniformly within its volume, independently of observation probes. Exact source validity rejects solid/boundary, insufficient-clearance, out-of-world and inside-observation proposals. The teacher's candidates can be arbitrarily close; the producer's reference-label separation 32/1024 does not apply. The same reproducible FREE observation probes are used to compare costs across candidates.

Every cell records candidate coordinates, exact source statuses, sampled costs, range/std, valid counts and a descriptive uncertainty proxy. That proxy is not a confidence certificate. All regions remain `UNRESOLVED`; four equal costs, four invalid positions or a zero finite difference never certifies the whole region. No region is removed on that basis. Query-budget exhaustion leaves explicit pending cells and incomplete execution, rather than an invented completed teacher target.

`UniformRefinement` is a coordinate-stable breadth-first baseline. `VariationRefinement` prefers observed variation plus a low-sample uncertainty term and reserves explicit random exploration (default 25%). This is only a working heuristic. Its `choose(records,rng)` method can be replaced by later mass/uncertainty decisions. Neither heuristic is claimed optimal. A source-space node's geometrical volume is `(side/1024)^3`, not the number of candidate samples. Parents remain in the evaluated trace after subdivision; they must not be counted again as disjoint final support.

Defaults in `configs/octree-smoke.json` are bounded demonstration settings: 64 candidate attempts, 4,096 pair queries and one refinement after the eight initial cubes. These are not settled research hyperparameters. Geometry validity work and native setup also cost CPU time and must be reported alongside pair counts. Native calls are not assumed instantly preemptible.

## Probability mass and future learning objectives

If later modeling a continuous Gibbs density, `q(s) ∝ exp(-C_O(s)/tau)` is with respect to a declared source-space measure. A cell's mass is the integral of that density over its allowed volume. Under uniform whole-cell samples an elementary estimate is `volume(cell) × mean(1_valid(s) exp(-C(s)/tau))`, normalized over a disjoint leaf partition. Averaging only valid samples and discarding the acceptance fraction would misestimate allowed volume. UNKNOWN mass must remain unresolved, rather than being assigned zero without proof.

Constant-k samples in differently sized cells are not equal-area support. Equal probability per cell would overrepresent refined regions. Coarse parent traces and fine child support must not be blindly mixed in one softmax. A later uniform final grid can use its consistent measure; an adaptive density approximation needs volume/importance weighting and an explicit completeness/uncertainty policy. This scaffold intentionally exports **no probability target**.

Energy regression learns a numerical cost/energy, ranking learns relative compatibility, and probability distillation learns a normalized measure-dependent target. They require different losses and calibration. No choice among them is fixed here. The permitted local-only student also lacks outside geometry that the teacher knows, so learning the teacher landscape across scenes is a privileged-information research question, not a claim of deterministic identifiability from every local window.

Future experiments should compare refinement rules under the same total setup/query budget and a common reference support, report narrow-region misses and invalid-source fractions, reserve exploration, and use full-window independent verification for any claimed equivalent light. Ambiguity bounds, inverse-cone analysis and reference enrichment remain independent optional research, never prerequisites for this data interface or mandatory Core production.
