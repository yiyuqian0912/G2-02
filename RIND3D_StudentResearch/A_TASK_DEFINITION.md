# A — Revised scientific task definition: multi-source physical worlds, single-channel posterior prediction

Status: **declared** (2026-11-13). It supersedes the "natural single-source worlds" wording of the audit's M1
recommendation (`RIND3D_Audit/09_CONTRACT_vNEXT.md`, `10_EXECUTIVE_DECISION.md`). Nothing in the physical 1K release,
its labels, or the canonical format changes.

**Evidence classes** (as in the audit):
* **A**: proved here;
* **B**: verified in code;
* **C**: measured, with the script under `experiments/`;
* **E**: a recommendation or hypothesis.

## 1. The physical-world population is multi-source (mandatory)

A RIND3D world is (G, K, S_1..S_K) with:

| element | definition | 1K implementation |
|---|---|---|
| geometry | G ~ π_f: a finite union of closed convex polyhedra; family f = slot mod 9 | `families.generate` |
| source count | K ~ κ on {1, 2, 3, 4}, **independent of G and S** | 1K: the controller's k_t, chosen before the scene from earlier scenes only; realised histogram 15.4 / 13.5 / 15.8 / 55.3 % |
| sources | S_i i.i.d. Uniform(D(G)), D(G) = legal FREE domain (open cube, outside solids, clearance > 2⁻²⁶); anti-duplicate rule ‖s_i − s_j‖_∞ ≥ 2⁻²⁰ (TV effect < 10⁻¹⁴) | `physical.sample_sources` |
| responses | R_j = V_G(S_j, ·) ∈ {0,1}^{N³}: one **separate** binary visibility field per source (closed solids, closed segments, SOLID receivers unlit). There is no superposition across sources. | `privileged/fields/src{j}` |
| windows | closed boxes B̄(W), W = (o, L), L ∈ {64, 128, 256} voxels of N = 1024. They are physical objects shared by all channels | `public/windows.npy` |

**Canonical storage stays source-aligned** and is unchanged. One record per physical window holds:
* the SOLID crop once;
* responses[K] (one channel per source);
* posteriors[K] (one slot per channel: a certified label or an explicit status).

A scene with 4 sources and 12 windows is 12 multi-channel window records, never 48 files. Flattening into
single-channel examples is a training-time adapter only (`rind3d_1k.loader.flatten`).

## 2. The primary learning problem

**Single-channel Student.** The input is one source-specific local response plus local geometry:

    X = (SOLID ∩ W, r_j = R_j|FREE(W), L, o)

The output is a probability field over the global source domain: the 128³ cells, outside the window box.

**Not inputs** (privileged for this task):
* the other channels;
* K, m, mask (they encode the other channels);
* ids, weights, label status.

A single-channel Student does not imply a single-source world. The worlds stay multi-source. Only the Student's view
is one channel.

A multi-channel Student, which sees all K channels and K, is a legitimate secondary task. It does not replace this one.

## 3. Four populations that must not be confused (brief §1)

Notation:
* the window-box mass β(G, W) = vol(D(G) ∩ B̄(W)) / vol(D(G)): the chance that one Uniform(D(G)) source falls inside the closed box;
* I_j(W) = 1 iff channel j has ≥ 1 lit and ≥ 1 dark FREE receiver in W ("mixed"; the generator's "informative");
* M(W) = Σ_j I_j;
* A(W, S) = 1[no S_i ∈ B̄(W)] · 1[FREE(W) ≠ ∅];
* p_nat(W) = uniform origins.

Densities of a flattened example (G, K, S, L, W, j), each up to normalisation:

| | population | example density | what the single-channel optimum P(S_j \| X) becomes |
|---|---|---|---|
| (1) | **N1: physical worlds with exactly one source** | π(G) u_D(s) ρ(L) p_nat(W) 1[s ∉ B̄] 1[FREE ≠ ∅] I(W) | P_N1(G \| X) ∝ π(G) vol F_W(G) / vol D(G). **Not a RIND3D population** (it violates §1); it serves only as a reference point. |
| (2) | **T*_sel: multi-source worlds, one channel chosen by a declared rule** (e.g. J uniform on 1..K, kept if mixed) | π(G) κ(K) Π_i u_D(s_i) ρ(L) p_nat(W) A(W,S) · (1/K) · I_J(W) | the N1 expression times g⁽¹⁾_κ(β) = E_κ[(1 − β)^{K−1}] |
| (3) | **T\*: multi-source worlds, every mixed channel is one example** (primary) | π(G) κ(K) Π_i u_D(s_i) ρ(L) p_nat(W) A(W,S) · I_j(W), j = 1..K | the N1 expression times g_κ(β) = E_κ[K (1 − β)^{K−1}] / E_κ[K] |
| (4) | **B1K: what the 1K release induces when flattened and ω-weighted** | (3) × c_t(M(W), L, k) × [slot mixture of κ_t] | the N1 expression times H_t(G, W) = Σ_k κ_t(k) k Σ_{m'} C(k−1, m') a^{m'} b^{k−1−m'} c_t(1 + m', L, k), where a, b are the chances that a hidden source is mixed / not mixed in W (audit 03 §3) |

Derivations, all class A:
* integrate the hidden sources s_{−j} out of each density;
* given G, they are i.i.d. Uniform(D(G));
* their only constraint is the closed-box exclusion, probability (1 − β) each;
* their responses never enter channel j's response (separate fields).

So the multi-source structure of a world enters a natural single-channel example through exactly two routes:
1. **The exclusion of the other sources** from the window box: the factor g_κ(β). It is bounded below.
2. **The selection rule**:
   * natural in (2) and (3);
   * m-balanced in (4), with c spanning 0.01–1 (100×) within a scene. Its factor depends on how often *hidden* sources are mixed in W, so it tilts P(G | X).

**A natural distribution of one-source worlds is *not* the same population as natural channels from multi-source
worlds.** The example-level source count of T* is size-biased: P_ex(K = k) = k κ(k) / E[K].
* With the 1K κ, 71.1% of T* examples come from K = 4 scenes, against 55.3% of scenes.
* The joint law of examples within a scene also differs: up to K correlated channels share one G.

For the **single-channel conditional** P(S_j | X), however, the difference reduces to g_κ, and it is small [A + C,
`experiments/e14_exclusion_factor.py`]:

| side | box volume | g_κ at a fully FREE box (vol D = 1) | max TV between the T* and N1 optima, over the solid fractions of any family (1K κ) |
|---|---|---|---|
| L64 | 2.4·10⁻⁴ | 0.99938 | **1.6·10⁻⁴** |
| L128 | 2.0·10⁻³ | 0.99508 | **1.3·10⁻³** |
| L256 | 1.6·10⁻² | 0.96110 | **1.0·10⁻²** |

* **Bound used:** TV ≤ 1 − g(β_max)/g(β_min) (SELECTION_LAW_1K §4 lemma). β varies across compatible worlds only through vol D(G), since the window's own FREE volume is observed.
* **Solid fractions:** 2·10⁻⁶ to 0.20 over the 1000 scenes.
* **(2) vs (3):** they differ by g⁽¹⁾ vs g, a smaller difference again.

**Conclusion (A + C):** under natural window sampling, the multi-source world changes the single-channel posterior
by at most 1% TV, and only at L256. The m-balanced release (4) changes it through a factor of up to 100×. This
measured difference, not the multi-source nature of the worlds, is the scientific problem behind M1.

## 4. Declared target population T* (primary Student task)

**T\*(κ, π, ρ).** A world (G, K, S) from §1, with κ declared. Natural windows: o uniform on {0..N−L}³, side L ~ ρ
(equal thirds), admissible iff no source lies in the closed box and FREE ≠ ∅. Every mixed channel j of an
admissible window is one example.

The training stratum may additionally be restricted by any **X-measurable** rule (§6). Such a restriction leaves
P(S | X) unchanged on the retained X [A; Zadrozny 2004].

For the 1K, κ = κ_1K: the realised histogram (154, 135, 158, 553) / 1000. k is drawn independently of the scene, so
the pooled slots realise π(G) × κ_1K(k). k is not associated with family (χ² p = 0.15, audit E1).

**Target of the Student:** p*(s | X) = P_T*(S_j ∈ cell s | X) = E_T*[q_G(s) | X].

**Theorem A1 (labels are exact for T\*) [A].** Under T*, given (G, W, r_j):

    P(S_j | G, W, r_j) = Uniform(F_W^j(G)),   F_W^j = {s ∈ D(G) \ B̄(W) : V_G(s, x) = r_j(x) ∀ x ∈ FREE(W)}

*Proof.* Integrating s_{−j} gives π(G) u_D(s_j) 1[s_j ∉ B̄] 1[R_G(s_j)|W = r_j] · g_κ(β(G, W)) · (terms in G, W only). g
does not depend on s_j. ∎

The FastLabel and tau-pencil labels target exactly this set and need no change.

**Theorem A2 (population optimum) [A].** For any loss weight w_e with E[w_e | G, K, S, W, j] ∝ dT*/dP_data, the
minimiser of E[w · CE(T_e, p(·|X))] is p*:
* with T_e the certified label q_G (up to its TV ≤ 0.05 certificate);
* or with T_e the one-hot e_{S_j}.

These are the same functional (audit 04, Cor. 1). One-hot and posterior supervision therefore estimate the same p*,
with different variance.

**The Student is not expected to reproduce any individual full-world posterior.**
* q_G conditions on the complete geometry G; X sees only a window.
* The optimum mixes q_G over the worlds compatible with X.
* The loss floor is H(S | X) = H(S | X, G) + I(S; G | X) > 0. A training loss below it on held-out scenes is impossible; on training scenes it indicates memorised worlds.

## 5. The release (B1K) and the natural supplement as estimators of T*

| data | windows per scene and side | density relative to T* | T* weight (per natural-proposal unit) |
|---|---|---|---|
| release 1K (accepted windows) | n_L = 16 / 24 / 24 | n_L c_t(m, L) / Λ_L(G, S) | ω̂ / (n_L c(m, L)) |
| NAT-1K supplement (`natural/natural_supplement.py`; fixed n_p natural proposals, no thinning) | n_p = 8 (pilot-train), 16 (pilot-select, pilot-eval), 0 (test) | n_p | 1 / n_p |
| union (balance heuristic, Veach 1995) | both | n_L c/Λ + n_p | 1 / (n_L c(m, L)/ω̂ + n_p) ≤ 1/n_p |

* Each weight is in the same units across scenes, so the scene-level weighting of T* holds as well.
* The B1K population itself is the release with weight ω̂: the contract's "training law".
* Choice of estimators, effective sample sizes and the bias/variance trade-off: see `B_SAMPLING_WEIGHTING.md`.

## 6. Response-type strata (X-measurable)

For channel j in window W, computed from X alone:

| stratum | definition | role in the first Student study |
|---|---|---|
| boundary-mixed | mixed, with ≥ 1 pair of 6-adjacent lit-FREE / dark-FREE voxels | **training** |
| solid-separated | mixed, every 6-connected FREE component monochromatic (lit and dark regions separated by SOLID) | held out: OOD stratum C (certified labels exist: they are part of the backfill) |
| all-lit | every FREE receiver lit | held out: OOD stratum A (never trained on, never used for model selection) |
| all-dark | every FREE receiver dark | held out: OOD stratum B |

**Validity of the split.** Restricting training to boundary-mixed X is an X-measurable selection. It leaves the
target on boundary-mixed X unchanged and only defines the domain the Student is trained on.

**Population of the held-out strata.**
* The held-out strata are evaluated on **natural** windows (NAT-1K), where their frequencies are natural.
* The release's all-lit and all-dark channels are not natural examples: they exist only next to a hidden mixed channel (audit M4). They are not used.

**Why all-lit and all-dark are not uniform.** Both observations constrain the source:
* all-lit excludes every source position from which some solid would shadow a FREE receiver of W;
* all-dark requires every segment from the source to every FREE receiver to meet a solid.

Neither posterior is uniform in general. The audit measured:
* fixtures: all-lit excludes 18–38% of the prior, all-dark keeps 4%;
* real windows: all-lit posteriors cover 7–95% of the cube (audit E8, E11).

No uniform reference is assumed anywhere (`F_OOD_BENCHMARK.md`).

## 7. Evaluation populations

| population | data | weights |
|---|---|---|
| **T\*** (primary) | NAT-1K windows of pilot-eval scenes (the 1K val split) | equal within side; sides pooled with equal shares |
| B1K (reported for comparison) | release windows of pilot-eval scenes | ω̂ (combined_rb) |
| OOD strata A/B/C | NAT-1K windows of pilot-eval scenes, by stratum; plus analytic fixtures | equal |

**Scene split** (declared in `natural/natural_supplement.py`, by hash):

| role | source | size | use |
|---|---|---|---|
| pilot-train | the 1K train split minus pilot-select | ≈ 700 scenes | training |
| pilot-select | sha256(id + ":pilot-select") mod 8 = 0 | ≈ 100 scenes | early stopping, model selection, exploratory OOD-dev |
| pilot-eval | the 1K val split | ≈ 100 scenes | final in-distribution and OOD evaluation, used once per arm |

**The 1K test split is not opened.**
