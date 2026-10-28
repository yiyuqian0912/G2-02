# Multisource peeling: proof and verification

Read [PROOF.md](PROOF.md) for the mathematical results and their assumptions.

The main distinction is between a probability-zero **exact shared shadow boundary** and a **false connection on the sampled grid**. The second event need not become unlikely as pixels get smaller. The note proves both a decaying strip probability and a physically realizable crossing counterexample.

No external packages, dataset download, training, or GitHub connection are required to run the checks:

```sh
# Run from this proof directory.
python3 -m unittest -v
python3 verify.py --output results.json
```

`verify.py` checks the strip benchmark, crossing scenes, finite-mask peeling, and the difference between full-mask and local subtraction. It writes its seed, sample counts, measured errors, and pass/fail results. Tests and simulations support the proofs; they do not establish a dataset-wide failure rate.

The note includes a [standalone figure](figures/ambiguity.png), also available as [SVG](figures/ambiguity.svg). Regenerate it with `python3 plot_figures.py` from this directory, using an environment with NumPy and Matplotlib installed. Only this optional plotting script needs those packages; the checks above use the standard library.

Recorded verification: **15 tests passed**, all five diagnostic groups passed, maximum fixed-strip probability error below `0.000079`, and no mismatches between direct rectangle-ray visibility and the affine crossing model in the sampled checks. See [results.json](results.json) for counts and seeds. These are synthetic verification results, not measured dataset probabilities.

GitHub was checked on October 27, 2026, America/New_York. The relevant branch heads were unchanged from the planning discussion:

| Branch | Commit |
|---|---|
| `feat/phase2-unit-random-data` | `77c03a031b923953197184ba16cf6dd6e55c5e48` |
| `main` | `1591783558e04e1188e97ee008932b11421db52b` |

The note contains pinned links to the inspected implementation. No recurring monitor was created and no existing training code was changed.
