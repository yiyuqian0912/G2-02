# B2 primary fixed-witness HIT reuse

The primary prototype captures a fixed-depth HIT witness from the scalar search already used by P_J. The independent F_J build renders its own field. Both operate on the same canonical positive Gaussian sum with closed-segment semantics.

## Witness capture

| Scalar HIT mechanism | Parameter and strict lower bound |
|---|---|
| Directed point or endpoint | Reuse the parameter and field lower enclosure from the successful comparison. |
| Entire SOLID parameter interval | Take its dyadic midpoint and reuse the lower enclosure for the whole interval. |
| Rational quadratic and directed MPFR fallback | Retain the successful point or interval lower sum, converted downward. |
| Contained inner ellipsoid | Reuse the kernel's evaluated parameter and quadratic upper bound. A strict lower enclosure for its positive contribution follows from the original log-radius bound. |
| Analytic common-shape intersection | The minimizer is rational. The audit reports it when representable as binary64 and distinguishes exact contact from strict intersection. The prototype leaves this fast path out of propagation. |

For the inner-ellipsoid case, let $\rho$ be the stored lower bound on $2\log(w_i/\tau)$, and let $\overline q$ be the already computed upper bound at the parameter. If $\overline q\le\rho$, then $w_i\exp(-q_i/2)\ge\tau\exp((\rho-\overline q)/2)\ge\tau(1+(\rho-\overline q)/2)$. The implementation encloses the last expression downward with the lower threshold enclosure. The other positive contributions have a conservative zero lower bound for this seed proof. Both extension methods enclose the full sum.

A witness is eligible for extension only if its numerical lower bound $L$ exceeds the upper threshold enclosure. The signed-gradient test uses the downward margin $m\le L-\overline\tau$. Exact contact and unresolved intervals retain their original scalar semantics.

## Continuous certificates

For an axial receiver interval $B=[x,x+\sigma h e_k]$, fix the accepted parameter $t_*\in[0,1]$. For each kernel, let $b_i=\sigma M_i^{-1}[:,k]$ and $z_i=M_i^{-1}(s+t_*(x-s)-c_i)$. The transformed witness locations for the entire interval have the form

$$
  z_i+t_*\alpha b_i,\qquad 0\le\alpha\le h .
$$

**A, signed gradient.** Enclose the upper value of $\sum_i g_i b_i\cdot(z_i+t_*\alpha b_i)$ on this interval, where $g_i=w_i\exp(-q_i/2)$. This is the field's downward derivative with respect to displacement in the signed axis direction. Choose a density upper or lower endpoint according to the sign of the upper factor bound. Sum signed upper products and clamp the sum below at zero. Its upward product with $t_*h$ must be at most the retained downward margin.

**B, direct lower bound.** Each squared norm is convex in $\alpha$. The maximum of the two endpoint quadratic upper bounds therefore encloses it from above everywhere on the interval. Evaluate one downward density endpoint per kernel and sum the weighted lower contributions. A sum at least $\overline\tau$ proves the full interval shadowed at the fixed parameter. This uses the same sufficient condition as the existing common-section method with a parameter supplied by the scalar proof. The existing common-section control searches multiple parameters from scratch.

Both calculations use the primary's outward binary64 inverse and affine enclosures and directed MPFR endpoints. Tail lower enclosures may be zero even when the exact tail is positive. Gradient upper tails remain positive when needed. Nonfinite arithmetic or a failed test leaves the receiver to its normal physical path. [MPFR rounding conventions](https://mpfr.org/mpfr-current/mpfr.html#Rounding)

## Tile policies

- **H1** tries a one-voxel direct certificate in the first direction with a future unmarked center.
- **H4** tries a four-voxel direct certificate on the best negative axis with at least two future unmarked centers.
- **HA** uses the same long choice when the margin divided by the upper threshold and depth is at least 0.008. Two failed long attempts in the same tile-local 8³ block and axis trigger a 12-opportunity cooldown. The one-step attempt is the fallback.
- **HG4** substitutes the signed gradient for the direct lower bound, as a matched three-hot-tile control.

Only the proof decides a label. The axial cache records exact dyadic center indices. Cached HIT returns advance the old MISS planner's scheduling state. The matched tests require identical MISS seed indices and region paths for every mode. Separate per-tile HIT and MISS marks are checked for conflicts with each other and with resolved field bits. A conflict aborts the experiment. F_J receives only the same canonical inputs and renders in an independent session.

## Reproduce

Use the same frozen B2 numerical builds and Python environment from the generator root. These commands reserve new output directories.

    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    export PYTHONPATH="$PWD:$PWD/outputs/gaussian-geometry-v1/deps"
    OUT=outputs/gaussian-b2-hit-replay
    OLD=outputs/gaussian-b2-primary-miss-v1
    mkdir -p "$OUT/compiler-tmp"
    python -m experiments.gaussian_geometry.b2_hit.freeze --output "$OUT"
    env TMPDIR="$PWD/$OUT/compiler-tmp" python -m experiments.gaussian_geometry.b2_hit.build \
      --output "$OUT/audit-native"
    python -m experiments.gaussian_geometry.b2_hit.audit --plan "$OUT/audit-plan.json" \
      --native "$OUT/audit-native" --output "$OUT/witness-audit.json"
    env TMPDIR="$PWD/$OUT/compiler-tmp" python -m experiments.gaussian_geometry.b2_hit.live \
      --output "$OUT/live-native"
    python -m experiments.gaussian_geometry.b2_hit.bench P_J --output "$OUT/P_J"
    for method in P_JHA P_JH4 P_JH1; do
      python -m experiments.gaussian_geometry.b2_hit.bench "$method" --native "$OUT/live-native" \
        --baseline "$OUT/P_J" --output "$OUT/$method"
    done
    for method in P_JHC P_JHG4; do
      python -m experiments.gaussian_geometry.b2_hit.bench "$method" --native "$OUT/live-native" \
        --ids 14-05 14-13 14-14 --baseline "$OUT/P_J" --output "$OUT/$method"
    done
    python -m experiments.gaussian_geometry.b2_miss.bench run --native "$OLD/propagation-native" \
      --mode 4 --plan "$OUT/heldout-plan.json" --output "$OUT/F_J-heldout"
    python -m experiments.gaussian_geometry.b2_hit.bench P_J --plan "$OUT/heldout-plan.json" \
      --reference "$OUT/F_J-heldout" --output "$OUT/P_J-heldout"
    for method in P_JHA P_JH4 P_JH1; do
      python -m experiments.gaussian_geometry.b2_hit.bench "$method" --native "$OUT/live-native" \
        --plan "$OUT/heldout-plan.json" --reference "$OUT/F_J-heldout" \
        --baseline "$OUT/P_J-heldout" --output "$OUT/$method-heldout"
    done
    RIND_B2_HIT_BUILD="$OUT/live-native" python -m pytest experiments/gaussian_geometry/tests/test_b2_hit.py
    python -m experiments.gaussian_geometry.b2_hit.arb_check --output "$OUT/arb-controls.json"
    python -m experiments.gaussian_geometry.b2_hit.gate --native "$OUT/live-native" \
      --primary P_JHA --output "$OUT/admission"
    python -m experiments.gaussian_geometry.b2_hit.summarize --run "$OUT" \
      --output "$OUT/matched-summary.json"

The admission driver uses the original ordered 16-tile probe, paired octants, 20% planning margin and 360 s / 900 s and 512 MiB limits. Only a paired passing probe can dispatch the remaining 48 tiles. The checker streams tile pairs and verifies each archive and both physical masks. [Measured results](GAUSSIAN_B2_HIT_RESULTS.md)
