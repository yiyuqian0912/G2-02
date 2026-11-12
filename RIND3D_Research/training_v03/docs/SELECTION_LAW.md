# Selection law and inclusion design (obs-index-v0.3)

Code: `rind3d_training/index.py` (design), `rind3d_training/selection.py` (probabilities, weights, validation).

## Design

Each scene and side L ∈ {64, 128, 256} has five deterministic uniform proposal streams, one per subset s. These are exactly the v0.2 streams: `keyed_rng(seed, slot, STREAM[s]·8 + side_index)`.

Every proposal w is an origin drawn uniformly from {0..1024−L}³, independently of (G, S). It is accepted when its rule A_s(w) holds:

| Subset | Rule A_s (privileged: it uses the true sources) |
|---|---|
| natural | 1 (no filter: all-lit, all-shadow, no-FREE and source-containing windows are all kept) |
| informative | mixed over FREE ∧ no source in the closed box |
| extreme | informative ∧ minority fraction ≤ 0.01 |
| rare | informative ∧ minority count ≤ 64 |
| nonlocal | informative ∧ ≥ 90% of 32 sampled shadow-FREE receivers are blocked outside the box (evaluation only) |

A stream stops at its quota or at its proposal cap. Shortfalls are recorded and never filled.

**Profiles as prefixes.** A profile is a set of quotas, and P256 continues the same streams. P102 is the prefix with the v0.2 quotas and caps. It was verified bit-identical to the stored v0.2 records for all 216 canary scenes (`v02_prefix_identical: 216/216`).

**Extending a release** therefore needs no new physics and no change to existing records. It has two options:
- raise the quotas, which continues the streams;
- add the hybrid dynamic stream (natural, keyed by a private key).

Every record carries the exact counts and every count-based rule indicator. The rules are evaluated for **all** records and all proposals, so membership in every subset is known for every window. A further fixed-size uniform **rate stream** (1024 proposals per side, stream 3051) gives unbiased estimates of P(A_s) and of the natural category distribution.

## Density and weights

Inside a (scene, L) group with n_s records of subset s and n = Σ n_s:

q(w) = Σ_s (n_s/n) · p_nat(w) · A_s(w) / P(A_s)

The weights recover each population:

- **Global natural population:** w(w) = p_nat/q = 1 / (n_nat/n + Σ_{s≠nat} (n_s/n) · A_s(w)/P(A_s)).
- **Source-excluded population** p_nat(· | S ∉ B(W)). Only records with no source in the box count, so n′ excludes source-containing natural records: w(w) = 1 / (n′_nat/n′ + Σ_s (n_s/n′) · A_s(w) · (1−P(cs))/P(A_s)).

Weights are self-normalised to mean 1 inside each group, so every scene has equal mass per side.

P(A_s) is the inverse-variance combination of two estimates:
- the fixed-design estimate (rate stream plus natural records);
- the unbiased inverse-binomial estimate (r−1)/(n−1) from the subset's own stream.

Validation is `selection.check_weights`, used in `test_06`. The weighted mixture reproduces the rate-stream population frequencies of mixed, informative, extreme and source-containing windows per side (see the study for the numbers).

## Which subsets may train what

| Subset | sb-global / ma-additive | sb-source-excluded | evaluation |
|---|---|---|---|
| natural | weight 1 (clean) | if no source in box | yes |
| informative, extreme, rare | **only with weights** (the rule conditions on hidden S) | yes; weights optional (the rule is a function of O_W and the declared condition, i.e. covariate shift only) | yes |
| nonlocal | **no** | **no** (the rule depends on the hidden first-blocker position) | yes (diagnostic stratum) |

## What must stay private

The following are privileged:
- the rule indicators, `contains_source`, proposal indices, acceptance counts, shortfalls, weights and nonlocal estimates;
- the index seed of a release.

**Why the seed must be private.** With a public seed, anyone could replay the informative stream. A *mixed* proposal that was rejected can only have been rejected for containing a source, which localises the source to that box.

The canary uses its public integration seed and is integration data only. Releases use HMAC(private 256-bit index key, policy, release, scene_id); the v0.1 overlay is built this way.

## Hidden response-dependent sampling is never a Student feature

The Student sees no subset tag, no weight and no shard position that correlates with them (audit L9). Banks that depend on the response are Teacher-only: stratified receiver banks reject any call without the observed response and are not offered to Student views. Student sparse views (v0.2 `ProductionDataset(sparse=...)`) are uniform only.
