# XLF CBC: next steps (prioritized, grounded in the measurements)

Production stays FastLabel v3.4 on `yushu/3d`. None of this is a production change.

## 1. Throughput of the exact kernel

This is the largest lever. Exact Python clipping of about 500-bit homogeneous polytopes dominates every remaining
failure; screens are Numba integer code and already cheap.

Options, in order of expected gain per effort:

1. **Avoid fragmentation by receiver choice.**
   * Enforcing a dark receiver with a single candidate body is one intersection.
   * An active receiver splits the piece into up to one piece per cone row.
   * Prefer single-candidate dark receivers. Implemented behind `--prefer-single-dark`; to be evaluated.
2. **Compiled exact split.** C++ with GMP/boost cpp_int under the same face-cycle representation, keeping the
   Python version as the differential reference. The byte-identity regression and the run-time volume-conservation
   guards carry over unchanged.
3. **Lazy H-representation of live pieces.** Append rows and enumerate vertices only at a screen. Many pieces are
   cut several times between screens.

## 2. Broad fragmented frontiers (the 11 failures at 20 s)

* Make contact-layer handling quantitative: stop refining contact layers once their charged volume is below the
  tolerance slack, instead of the fixed 1/2000 parking rule.
* Use a proper rational centroid as the search anchor. The homogeneous vertex sum is biased toward high-denominator
  vertices.
* A per-piece budget, so a breadth-first sweep over hundreds of equal pieces cannot starve the pieces that are
  nearly resolved.

## 3. Assurance

1. Finish the exact differential audit on all 90 tasks. L = 256 needs a chunked GPU or Numba oracle to be
   affordable.
2. Mechanize the smallest proof targets where it pays:
   * the frontier accounting theorem and the integer wire bound (a short Lean/Isabelle development, or keep the
     300-instance exact property test);
   * the directed-rounding lemma of the screen (an SMT bit-vector check of the interval endpoint formula over the
     verified ranges).
3. Pin the Numba build: compiler, flags, cache keys.

## 4. Population evidence

* Run the release-canary comparison with every informative channel (807), not a sample.
* Measure H20 or server throughput only after the kernel speed-up.
* A 10× FastLabel production target needs a certification rate close to FastLabel's 100%. Until then the realistic
  deployment is a deterministic first pass with FastLabel as a fallback, which still yields deterministic labels
  for most channels.
