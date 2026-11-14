# 01 — Scientific Model Specification: the generative process that the 1K code actually implements

Evidence classes: **A** proved, **B** verified in code (file:function), **C** measured, **D** literature, **E** hypothesis/recommendation.
Code paths are relative to `/home/yushu/Projects/RIND/G2-02-3d/`. Release data: `/home/yushu/Projects/RIND/RIND3D_1K_data/release_1k_v1/`.

## 1. What each step of the code does (brief §2, items 1–18)

| # | Question | What the code does | Evidence |
|---|---|---|---|
| 1 | How G is sampled | `family = FAMILIES[slot % 9]`; `RIND3D_Dataset1K/rind3d_1k/families.py:generate` delegates v0.1 families to `RIND3D_Generator/src/rind3d/lite_generators.py:generate` (keyed RNG `(seed, slot, 8101+97·attempt)`): object count uniform in a family range, uniform SO(3) rotations, log-uniform half-sizes, centres uniform in the world-fitting interval (or anchored/clustered layouts for `occluder_clusters`, `corridors`, `barriers`). A proposal failure rejects the attempt (`physical.py:run_scene`). The acceptance gate `families.py:acceptance` → `lite_acceptance.py:geometry_acceptance` raises on failure. Neither step looks at sources. | B |
| 2 | How K is selected | `scripts/run_physical.py:main` → `rind3d_1k/controller.py:Controller.choose(u, floor=0.15)`. A keyed uniform `u` of the slot (stream 56000) forces k = 1..4 with prob. 0.15 each; otherwise k minimises the expected squared distance of the cumulative m-histogram from (¼,¼,¼,¼), using the m-histograms and rate-stream statistics of **earlier** scenes only. Realised k histogram 154/135/158/553. | B, C |
| 3 | How S_j is sampled | `physical.py:sample_sources`: i.i.d. `rng.random(3)` proposals, accepted iff `legal_source` (strictly inside (0,1)³, not in a solid, certified clearance > 2⁻²⁶) and ‖p − s_i‖_∞ ≥ 2⁻²⁰ for earlier accepted sources. | B |
| 4 | Uniform in legal FREE? | Yes, per scene: rejection sampling from the uniform cube proposal gives Uniform(D(G)). Empirical PIT test on 3110 sources: no axis or radial statistic significant after multiplicity correction (min p = 0.018 over 57 tests; E2). | B, C |
| 5 | Independent given G? | Yes up to the anti-duplicate rule (TV effect < 1e-14, SELECTION_LAW_1K §4). | B, A |
| 6 | Spacing/clearance constraints | Only the 2⁻²⁰ anti-duplicate rule and the 2⁻²⁶ clearance. No MIN_SEPARATION. | B |
| 7 | Sources resampled for window feasibility? | No. A scene whose quotas cannot be filled is rejected as a whole and the next attempt redraws G and S (`run_physical.py`, `max_attempts=3`). In the release every slot was accepted at attempt 0 (`manifest.jsonl`: 1000 accepted, 0 rejected, 0 failed). | B, C |
| 8 | How windows are proposed | `physical.py:select_side`: origins i.i.d. uniform on {0..N−L}³ (stream `52000+L`), one U per proposal (stream `53000+L`); repeated origins skipped (2 repeats in the whole release). | B, C |
| 9 | Do proposals depend on S? | No: streams keyed by (seed, slot, L, attempt) only. | B |
| 10 | Source exclusion | `physical.py:contains_any`: a window is rejected if **any** source lies in the closed box [o/N, (o+L)/N]³. The labeller excludes the same closed box (`engine_p2.stage2`, "window box" test with 1e-12 tolerance). | B |
| 11 | Mixed-window acceptance | `physical.py:evaluate`: A = [no source in box] ∧ [FREE > 0] ∧ [m ≥ 1]; accepted iff A ∧ U < c(m, L) where c is the scene's table. | B |
| 12 | Effective layers | I_j = [channel j has ≥ 1 lit and ≥ 1 dark FREE receiver in W]; m = Σ_j I_j (exact counts from `CountIndex`). Solid-separated windows count as informative. | B |
| 13 | Effect of balancing | c(m, L) is chosen per scene by the controller (`plan_k`): c ∝ deficit(m)/p̂(m|L,k), clipped to [0.01, 1]. For k ≥ 2 the median c(1) is the floor 0.01 at L = 64 and 128 (E1). | B, C |
| 14 | Exhaustion and rejection | Cap 6000 proposals per side; any unfilled quota rejects the scene. Max proposals used: 4658 (L64), 2678, 979. | B, C |
| 15 | Family balancing | Deterministic: `slot mod 9` → 111 or 112 scenes per family. k is not significantly associated with family (χ² = 31.2, 24 dof, p = 0.15; E1). | B, C |
| 16 | Sample weights | ω = Λ̂_L(G,S) per (scene, side): `rate` = accepts/1024 on a separate stream (contract default); `quota` = (r−1)/(N−1); `combined` = (1−a_L)·rate + a_L·quota (`loader.py:Scene.weights`). | B |
| 17 | What the Student receives | `loader.py:flatten` yields `solid` (L³ crop), `response` (L³ crop of channel j), `L`, `origin`, plus ids, `label`, `weight`. The contract's Student view: SOLID crop, response channel(s), side, origin optional. | B |
| 18 | What is privileged | G outside W, all source positions, k, the other channels, the c-table, U values and the selection log, Λ̂, label status. **Caveat:** `public/windows.npy` also stores `m`, `mask` (I_j of every channel) and `k`. These are privileged for a single-channel Student (they encode the hidden channels), although the directory is named `public`. | B |

## 2. Random variables

* t ∈ {0,…,999}: campaign slot (deterministic order). f(t) = FAMILIES[t mod 9].
* Z_t: controller state before slot t (a deterministic function of accepted scenes < t); u_t keyed uniform. (k_t, c_t) = plan(Z_t, u_t).
* G ~ π_{f(t)}(G): global geometry (finite union of closed convex polyhedra). D(G) ⊂ (0,1)³: legal FREE domain.
* K = k_t; S = (S_1,…,S_k) source positions.
* R_j = V_G(S_j, ·) ∈ {0,1}^{N³}: source-separated binary field (closed solid, closed segment; SOLID receivers unlit). N = 1024.
* For side L: proposals (o_i, U_i); W = closed box B̄(o, L); FREE(W) = non-SOLID receivers in W.
* I_j(W) = informative indicator; M(W) = Σ_j I_j; A(W,S) = acceptance indicator.
* E: scene acceptance (all quotas filled).
* Λ_L(G,S) = E_{o∼p_nat}[A(W,S) · c_t(M(W,S), L)]; Λ̂_L its estimate (ω).
* Flattened example e = (t, W, j) with I_j(W) = 1; Student input X_e = (K_W, r_j, L, o) with K_W the SOLID crop and r_j = R_j|FREE(W).
* Label q_e = Uniform(F_W^j(G)) cell masses, F_W^j = {s ∈ D(G) \ B̄(W) : V_G(s,x) = r_j(x) ∀x ∈ FREE(W)}.
* Λ_e = label-availability indicator (label computed, status accepted by the loader).

## 3. Factorisation

**Scene level** (slot t, attempt 0) [A, from the code in §1]:

    p(G, S | t, Z_t, u_t) = π_{f(t)}(G) · Π_{i=1..k_t} u_{D(G)}(s_i) · Δ(S)            Δ = anti-duplicate indicator (≈ 1)

**Window level** (side L, quota n_L, i.i.d. proposals, infinite cap): the n_L accepted windows are i.i.d. from

    q(W | G, S) = p_nat(W) · A(W,S) · c_t(M(W,S), L) / Λ_L(G,S)                       [A]

(skipping repeated origins makes them a without-replacement sample; with ≤ 4658 proposals among ≥ 5.4·10⁸ origins the effect is negligible, 2 repeats were observed.)

**Accepted (unweighted) population of windows** (expected count density):

    π(G) κ_t(k) Π u_D(s_i) · n_L · p_nat(W) A(W,S) c_t(M, L) / Λ_L(G,S) · P_acc(G,S)

**ω-weighted population** (the contract's training law; E[ω̂ | G,S, windows] = Λ_L):

    P_ω(G, k, S, W) ∝ π(G) κ_t(k) Π u_D(s_i) · n_L · p_nat(W) A(W,S) c_t(M(W,S), L) · P_acc(G,S)        (∗)

**Flattened single-channel population** (what `flatten` yields, default arguments):

    P_flat(G, k, S, W, j) ∝ (∗) · I_j(W) · Λ_e(G, W, r_j)                                                  (∗∗)

where Λ_e = 1[label exists and is accepted by the loader filter] (default: status == "certified", TV ≤ 0.05; FastLabel statistical labels are excluded unless `accept_statistical=True`).

**Graphical model** (single slot):

    t ──► f ──► G ──► D(G) ──► S_1..S_k ──► R_1..R_k
    Z_t, u_t ──► (k, c_t) ─────────┘                │
    p_nat ──► W ──────────────────────────────────► I_1..I_k ──► M ──► A·[U < c_t(M,L)] ──► accepted W
                                                     │
    X_e = (SOLID∩W, R_j|W, L, o)  ◄──────────────────┘          label q_e = Uniform(F_W^j(G)) (needs G)

## 4. The four distributions that must not be confused

| Distribution | Definition | Where it appears |
|---|---|---|
| Proposal | π κ Π u_D × p_nat(W) | before any acceptance |
| Accepted dataset | the unweighted accepted density above | what a loader iterates over |
| Training (ω-weighted, flattened) | (∗∗) | what a correctly weighted loss averages over |
| Intended evaluation | **not declared anywhere** in CONTRACT_1K / SELECTION_LAW_1K | must be fixed before Student evaluation (see 09) |

Natural reference populations that a scientific claim could target (E):
* **N1, natural single source:** G ∼ π, one S ∼ Uniform(D(G)), W ∼ p_nat restricted to S ∉ B̄(W) (optionally conditioned on an X-measurable response type).
* **Nk, natural multi-source:** as N1 with k sources and the exclusion of all sources, without thinning (c ≡ 1).
* **B1K, the release's training population (∗∗)**, which contains the controller's c_t and κ_t.

## 5. What the Student must learn

Given X only, the target is the conditional p_T(s | X) under a declared target population T. Under forward KL / cross-entropy with labels q_G and weights ω (04, Theorem 1) the population optimum is

    p*(s | X) = E_T[q_G(s) | X],     T = the ω-tilted flattened population (∗∗).

Section 03 shows that (∗∗) ≠ N1 for the single-channel Student, through the balancing factor c_t(I_j + M_{−j}) that depends on the hidden channels, and through the label filter Λ_e.
