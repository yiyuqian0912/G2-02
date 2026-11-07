# Certified MISS neighborhood reuse

This experiment extends the interval-local independent F renderer in a separate native build. Its numerical inputs are the frozen positive Gaussian sums and binary64 sources from the B2 precision study. It uses the same region traversal, continuous cover, exact source offsets, and budgets as F. The primary renderer, original independent A, D, standalone E and Arb remain controls.

## Proof

Write $R_i=M_i^{-1}$, $u_i=R_i(z-c_i)$, $g_i=w_i\exp(-\|u_i\|^2/2)$, and $F=\sum_i g_i$. The exact inverse of the canonical binary64 matrix is rational. For a component $k$,

$$
\partial_k g_i=-g_i (R_i[:,k]\cdot u_i).
$$

Cauchy–Schwarz gives a bound by $w_i\|R_i[:,k]\| \cdot r\exp(-r^2/2)$, where $r=\|u_i\|$. The derivative of $r\exp(-r^2/2)$ vanishes at 1 and changes from positive to negative there. Consequently $G_k=\exp(-1/2)\sum_i w_i\|R_i[:,k]\|$ bounds $|\partial_k F|$ globally. The column norms use exact rational sums of squares, followed by upward MPFR square roots, exponential, products, and sums. [MPFR rounding conventions](https://mpfr.org/mpfr-current/mpfr.html#Rounding)

For a source $s$ and a receiver $x$, let $r_x(t)=s+t(x-s)$. F closes a dyadic partition of the entire closed interval $[0,1]$ with upper bounds $U_j<\tau$. The experiment captures the successful natural, Taylor, concavity, monotone-endpoint, or terminal upper enclosure as each closed interval $I_j$ is accepted. It retains a downward bound on $\delta_j=\tau-U_j$ using the lower threshold enclosure. Any earlier exact special-case return, original-A fallback, HIT, incomplete partition or UNKNOWN supplies no reusable proof. F's integer length check and recursive bisection establish full cover even at shared endpoints.

For any other receiver $y$, the difference along the two rays is $t(y-x)$. The mean-value theorem certifies a box $B$ about $x$ as FREE when every interval satisfies

$$
  t_{j,\mathrm{hi}} \sum_k G_k \max_{y\in B}|y_k-x_k| < \delta_j .
$$

G computes the downward capacity $m=\min_j\delta_j/t_{j,\mathrm{hi}}$. For world grid size $N$, its strict weighted test for integer voxel-center offset $d$ is $\mathrm{upper}(\sum_k G_k |d_k|)<\mathrm{lower}(Nm)$. Its displayed symmetric cube radius is the downward $Nm/\sum_k G_k$, in global voxel units. Equality is insufficient. The evaluated stencil is clipped to four voxels per axis and to world bounds. The audit also brackets the true minimum margin using the maximum of F's observed lower enclosures as a lower bound on the true ray maximum.

H and I use an axial closed receiver box from $x$ to $x+\sigma h e_k$. The degenerate box still certifies every continuous segment for all receivers along its length. Each F interval's existing 64-bit transformed ray line is inflated in direction $\sigma R_i[:,k] h$ by every displacement multiplier in $[0,t_{j,\mathrm{hi}}]$. A directed interval calculation over those transformed coordinates bounds $q_i$ and the signed factor $-\sigma R_i[:,k]\cdot u_i$. Multiply the factor's upper bound by the upper density if it is positive and by the lower density if it is negative. The original tail rule supplies a positive upper density and a zero lower density when applied. Sum signed upper products before clamping the total at zero. One directed exp endpoint per non-tail kernel suffices. The extension passes only if this upper derivative times $h t_{j,\mathrm{hi}}$ is strictly below the recorded downward margin for **every** closed interval. Nonfinite arithmetic fails closed. A failing limiting interval ends an attempt early.

## Experiment modes

| Mode | Decision |
|---|---|
| F | Original source-bound cover and interval-local scalar rays. |
| G | For every applicable F residual ray, try the global weighted stencil on its complete strict MISS proof. |
| H | Try one one-voxel axial extension, using the first direction with an unmarked future center in order −y, −z, −x, +y, +z, +x. |
| I | After at least five proved partition intervals, try up to two negative-axis four-voxel extensions with at least two unmarked future centers. Rank them by potential new centers, with ties in order −y, −z, −x. Stop at the first certificate. |

The H/I policy uses only integer traversal order, current marks, and work performed by the same solver. The frozen development measurements inform its constants. Held-out trace timings and physical labels are used for reporting, never for dispatch. The local experiment retains at most 256 accepted intervals per ray. Truncation disables local propagation.

The cache is one signed seed index per center in one tile of side at most 64, plus a second index array when tracing. It records only dyadic centers that passed the strict continuous test. It is consulted after the existing region and membership paths reach a FREE voxel that still needs a scalar visibility query. Source clearance and membership retain their original gates. The cache belongs to one tile of one source-bound independent engine. The separate primary run supplies the response and solid mask comparison.

Earlier whole-ray-volume certificates traverse precomputed FREE/UNKNOWN geometry boxes or bound a swept volume starting from the receiver region. Those methods can prune early without any scalar solve. This experiment instead starts with the successful scalar partition and spends additional work only on extending it. The bounded comparison uses the old mode-2 region function on exactly the same axial boxes, with its existing fallback.

## Reproduce

Run from the generator root with the existing B2 numerical dependencies, frozen generated A/D sources, precision tile build and prior traces. `OUT` below denotes a new study-output directory. Preserve the historical outputs as read-only controls.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD:$PWD/outputs/gaussian-geometry-v1/deps"
OUT=outputs/gaussian-b2-miss-replay
mkdir -p "$OUT"
python -m experiments.gaussian_geometry.b2_miss.bench freeze --plan "$OUT/tile-plan.json"
python -m experiments.gaussian_geometry.b2_miss.build --output "$OUT/native"
python -m experiments.gaussian_geometry.b2_miss.bench run --plan "$OUT/tile-plan.json" --output "$OUT/baseline"
for split in development evaluation; do
  python -m experiments.gaussian_geometry.b2_miss.audit --split "$split" --plan "$OUT/tile-plan.json" \
    --baseline "$OUT/baseline" --native "$OUT/native" --output "$OUT/global-$split.json"
done
python -m experiments.gaussian_geometry.b2_miss.local_audit --plan "$OUT/tile-plan.json" \
  --baseline "$OUT/baseline" --native "$OUT/native" --output "$OUT/local-development.json"
for spec in I:3 H:2 G:1; do
  python -m experiments.gaussian_geometry.b2_miss.bench run --plan "$OUT/tile-plan.json" \
    --native "$OUT/native" --mode "${spec#*:}" --reference "$OUT/baseline" --output "$OUT/${spec%:*}"
done
RIND_B2_MISS_BUILD="$OUT/native" python -m pytest experiments/gaussian_geometry/tests/test_b2_miss.py
python -m experiments.gaussian_geometry.b2_miss.gate --plan "$OUT/tile-plan.json" \
  --native "$OUT/native" --output "$OUT/admission"
python -m experiments.gaussian_geometry.b2_miss.bench run --plan "$OUT/tile-plan.json" \
  --primary --ids 7-ordinary 7-hot 60-ordinary 60-hot 63-ordinary 63-hot 72-ordinary 72-hot \
  --reference "$OUT/baseline" --output "$OUT/P-heldout"
python -m experiments.gaussian_geometry.b2_miss.metrics --run "$OUT" --output "$OUT/matched-summary.json"
```

The gate command measures the original paired-octant probe only. Its result includes the capped projection, paired physical coverage, and admission decision. Machine-readable measurements and the interpretation are described in [the results](GAUSSIAN_B2_MISS_RESULTS.md).
