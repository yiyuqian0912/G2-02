# Primary MISS reuse and adaptive axial extension

This B2 experiment instruments the primary and captures proofs from its own scalar solver. Independent F builds and checks its own proofs. The scheduler shares integer traversal rules and attempt policy across the two backends. A tile cache belongs to exactly one backend, geometry, binary64 source, world grid and tile.

## Capture and proof

The primary's depth-first bisection works on closed dyadic intervals of parameter $t$. It can close a FREE leaf with its outward binary64 quadratic range and density sum, a signed Taylor enclosure, or the existing exact rational quadratic range with a directed MPFR density sum. The grouped retry retains the enclosure actually compared with the threshold. Capture stores the dyadic interval and a positive downward margin $m_j \le \tau-U_j$. An exact fallback converts its margin to binary64 downward. The integer dyadic lengths must sum to $2^{50}$ before propagation. The closure capacity is 256 leaves. Insufficient capacity, nonpositive retained margins, outer-filter and analytic early exits, HIT, and unresolved results supply no neighborhood evidence.

Let $R=M^{-1}$ be the exact rational inverse of a canonical kernel matrix and $u=R(s-c), v=R(x-s)$. The primary keeps outward binary64 enclosures of $u,v$ as it computes its usual coefficients. For an axial displacement of sign $\sigma$, length $h$, and axis $k$, set $b=\sigma R[:,k]$. Along the rays to receivers between $x$ and $x+\sigma h e_k$, the transformed coordinate lies in

$$
  z \in u+t v + [0,h\,\max(I_j)]\,b,\qquad t\in I_j .
$$

Outward binary64 squares and sums enclose $\|z\|^2$. The signed derivative factor is $-b\cdot z$. A positive factor upper bound uses the upward density at the lower squared-norm bound. A negative factor upper bound uses the downward density at the upper squared-norm bound. The implementation uses a single directed MPFR endpoint per such kernel. For a tail with argument $q\ge128$, $2^{-\min(1074,\lfloor q/2\rfloor)}$ encloses the positive upper density because $e>2$, and zero encloses the lower density. The test sums signed upper products, then clamps the sum at zero. A directed upper bound on $h\,\max(I_j)$ times that derivative bound must be strictly smaller than $m_j$ on every leaf. This certifies every complete closed segment in the axial receiver interval. A failed or nonfinite check leaves the original query path available. [MPFR directed rounding](https://mpfr.org/mpfr-current/mpfr.html#Rounding)

F uses its previous MPFR transformed lines, directed local gradient check, and full closed partition. J schedules additional checks on that existing F proof. Each backend captures its own numerical evidence. The cache only answers a center whose original region and membership path has reached scalar visibility.

## Distance policies

- **H** checks one step in the first direction with a future unmarked center. The direction order is −y, −z, −x, +y, +z, +x.
- **I** ranks negative-axis four-step options by their number of future unmarked centers. An option needs at least two such centers and a seed with at least five strict leaves. It checks up to two axes and ends on the first successful proof.
- **J** uses the same bounded long search, with an independent failure history for each negative axis in each tile-local 8³ block. Two failures suppress the next 12 eligible dispatch opportunities. A successful retry clears the history. J falls back to the H attempt when no long proof is accepted. The work threshold and history select which exact checks to run. A check starts with the limiting retained leaf. Every accepted extension still checks all leaves.

The J parameters were fixed before running propagation on the previous 24-tile plan. The policy observes cache occupancy, proof sizes, and its own extension outcomes. The traced cache requires at most 2 MiB on a 64³ tile, and the untraced cache requires 1 MiB. J history adds at most 3 KiB. Candidate marks are exact dyadic center indices. The comparison counts a saved ray only on an actual cache return.

## Reproduce

Run from the generator root with the existing B2 dependencies and prior frozen builds. Choose a new output directory to preserve the measured artifacts.

    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    export PYTHONPATH="$PWD:$PWD/outputs/gaussian-geometry-v1/deps"
    OLD=outputs/gaussian-b2-miss-v1
    OUT=outputs/gaussian-b2-primary-miss-replay
    mkdir -p "$OUT"
    python -m experiments.gaussian_geometry.b2_miss.bench run --primary \
      --plan "$OLD/tile-plan.json" --reference "$OLD/baseline" --output "$OUT/P"
    python -m experiments.gaussian_geometry.b2_primary_miss.build --output "$OUT/audit-native"
    for method in P_profile P_capture; do
      python -m experiments.gaussian_geometry.b2_primary_miss.bench "$method" --case14 \
        --native "$OUT/audit-native" --reference "$OUT/P" --output "$OUT/$method"
    done
    python -m experiments.gaussian_geometry.b2_primary_miss.propagation --output "$OUT/propagation-native"
    for method in P_J P_H P_I; do
      python -m experiments.gaussian_geometry.b2_primary_miss.bench "$method" \
        --native "$OUT/propagation-native" --reference "$OUT/P" --output "$OUT/$method"
    done
    python -m experiments.gaussian_geometry.b2_primary_miss.bench F_J \
      --native "$OUT/propagation-native" --reference "$OLD/baseline" --output "$OUT/F_J"
    RIND_B2_PRIMARY_MISS_BUILD="$OUT/propagation-native" \
      python -m pytest experiments/gaussian_geometry/tests/test_b2_primary_miss.py
    python -m experiments.gaussian_geometry.b2_primary_miss.gate \
      --native "$OUT/propagation-native" --primary P_J --independent F_J --output "$OUT/admission"
    python -m experiments.gaussian_geometry.b2_primary_miss.summarize --run "$OUT" \
      --output "$OUT/matched-summary.json"

The gate uses the original 16 probe tiles and the existing 360 s / 900 s and 512 MiB limits. It launches the remaining 48 tiles only after both guarded estimates pass and all probe bits agree. Each backend renders its own archive. The validator streams exact tile pairs and verifies their checksums and physical masks. [Measured results](GAUSSIAN_B2_PRIMARY_MISS_RESULTS.md)
