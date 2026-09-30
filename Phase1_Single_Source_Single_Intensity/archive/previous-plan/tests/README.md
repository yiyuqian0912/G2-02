# Planned scientific and integration checks

Add checks when the corresponding module is implemented:

1. Single-source quadtree produces 18 source-free views: three at each side length 512, 256, 128, 64, 32, and 16; test partition-line and world-boundary ownership.
2. Scene-ID splits are disjoint; padding entries never become samples.
3. Rerendering the generating source reproduces the stored response within a declared tolerance.
4. Response weights depend only on the observation; no observed boundary gives uniform weights.
5. Edge cost has declared behavior for empty sets, translations, missing and extra boundaries.
6. Teacher and student distributions share support and base measure, sum to one, and handle invalid candidates consistently.
7. Candidate permutation changes only output ordering; padded pixels do not affect valid pooled features or losses.
8. Model scoring supports unequal window sizes; a tiny dataset can be fit.

No tests or experimental results are present yet.
