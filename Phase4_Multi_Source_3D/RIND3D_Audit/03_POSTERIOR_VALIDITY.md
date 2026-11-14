# 03 — Posterior Validity: is Uniform(F) the right label, and for which population?

Short answer:

| Level | Question | Verdict |
|---|---|---|
| Label, given G | Is q_G = Uniform(F_W^j) the posterior of S_j given (G, W, r_j) in the ω-weighted 1K population? | **Yes** (A, B; confirmed by C). It holds under hypotheses H1–H8 below and needs ω. Without ω it is wrong: toy mean TV 0.022, max 0.34. |
| Student, given X, flattened single channel | Is E[q_G \| X] under the training population the posterior of a natural single-source world? | **No.** It is the posterior of the balanced multi-source population, which carries the hidden factor c(I_j + M_{−j}), plus the label-availability filter. Toy: mean TV 0.031, max 0.087 from balancing alone; mean TV 0.22 from a single-cell-only filter. |
| Student, given all channels and k | Same question for a multi-channel Student | The balancing factor is observable and cancels (A). Remaining: the slot mixture and the label filter. |

## 1. Bayes' rule under the implemented law

From 01 (∗), for a flattened example with channel j informative, write s₋ⱼ for the other sources and
M₋ⱼ = Σ_{i≠j} I_i(W). The ω-weighted joint is

    P_ω(G, k, s_j, s₋ⱼ, W) ∝ π(G) κ_t(k) p_nat(W) · u_D(s_j) Π_{i≠j} u_D(s_i) · 1[s_j ∉ B̄] Π_{i≠j} 1[s_i ∉ B̄]
                              · 1[FREE(W) ≠ ∅] · c_t(I_j + M₋ⱼ, L, k) · P_acc(G, S).

Condition on V = (t, G, W, r_j). Then I_j = I(r_j, SOLID∩W) is fixed, R_G(s_j)|W = r_j is required, and integrating over
(k, s₋ⱼ) gives

    P_ω(s_j | V) ∝ u_D(s_j) · 1[s_j ∉ B̄(W)] · 1[R_G(s_j)|FREE(W) = r_j] · h_t(G, W, I_j) · P̄_acc(G, s_j)

    h_t(G, W, I) = Σ_k κ_t(k) Σ_{m'=0}^{k−1} C(k−1, m') a^{m'} b^{k−1−m'} c_t(I + m', L, k) 1[I + m' ≥ 1]

where:
* a = P_{s∼u_D}(s ∉ B̄, I(s; W) = 1) and b = P_{s∼u_D}(s ∉ B̄, I(s; W) = 0), by independence of the other sources (H3);
* the 2⁻²⁰ anti-duplicate factor is omitted;
* P̄_acc is P_acc averaged over s₋ⱼ.

h_t does not depend on s_j. Hence **q_G ∝ 1[s ∈ F_W^j] is exact iff P̄_acc(G, ·) is constant on F** [A]. This is the
SELECTION_LAW_1K theorem, re-derived independently from the code.

**Hypotheses actually used** (each checked):

| | hypothesis | status |
|---|---|---|
| H1 | Window proposals independent of S | B: `select_side` streams are keyed by (seed, slot, L, attempt) |
| H2 | (k, c_t) independent of S of the same scene | B: `run_physical.py` chooses them before `run_scene` |
| H3 | S_i i.i.d. Uniform(D(G)) given G | B, C (E2 PIT) |
| H4 | E[ω \| G, S, accepted windows] = Λ_L(G, S), independent of the windows' content | A (inverse binomial and separate stream), B |
| H5 | P_acc constant in s_j | C: 0/1000 rejections, so r ≤ 0.0046 at 99%. The SELECTION_LAW §3 bound applies |
| H6 | Same deterministic physics in generator and labeller (closed solid, closed segment, voxel centres, SOLID receivers unlit) | B (engine docstring, `Window.receivers`); C (E8: seven fixtures, 0 MC-feasible points outside the engine support; E4: 333/333 true sources inside the label support; swapped axes 0/333) |
| H7 | Label domain = D(G) \ B̄(W) | B: the engine excludes the closed box and solid interiors. The clearance shell only lowers `lo`. See 06 §3 for the all-dark special case |
| H8 | The same reference measure (Lebesgue on D(G)) | B |

**When is selection ignorable? [A]** The selection factor P(accept | s_j, V) must be constant in s_j on
{s ∉ B̄ : R_G(s)|W = r_j}:
* c_t(M) depends on s_j only through I_j, which is fixed by r_j: **ignorable**;
* 1/Λ_L(G, S) of the unweighted accepted population depends on s_j through its acceptance rate over all windows: **non-ignorable**, and removed by ω;
* scene rejection P_acc(G, S) is non-ignorable in principle and negligible here;
* source exclusion 1[s_j ∉ B̄] is part of F's definition. It is not double-counted, because the label excludes the same closed box.

**Toy confirmation** (E5, exact enumeration of 1024 geometries × 256 source cells × 4 windows, k = 2, 1K thinning c = (0.01, 1)) [C]:
* the ω-weighted per-channel posterior equals Uniform(F) to 2.8·10⁻¹⁶ (max TV);
* the unweighted one has mean TV 0.022 and max TV 0.34.

**Real-label confirmation** (E4/E4b, the 333 certified labels of scenes 0–3) [C]:
* every true source lies in its label's support (hi > 0 and lo > 0);
* with x and z swapped, 0/333 do; with another source of the same scene, 3/333 do;
* the HPD-rank test needs care. Over all 333 labels, KS p = 2·10⁻⁵, but the labels come from only **15 distinct sources**, and ranks of one source across windows are strongly dependent (per-source mean ranks 0.26–0.71);
* drawing one label per source, the KS test rejects at 5% in 4.65% of 2000 draws, i.e. at the nominal rate. Calibration is consistent, at the low power that 15 independent units allow.

## 2. Six "uniform" distributions (brief §3.2)

| distribution | formula | implemented? |
|---|---|---|
| uniform in each scene's legal FREE domain | u_{D(G)} | **yes**, it is the prior of every source |
| uniform over the unit cube | 1 on (0,1)³ | no. The aggregate source density is E_G[1_D/vol D], measurably non-uniform (E2: p = 0.006 vs cube) |
| uniform given exclusion of W | u_{D(G)\B̄(W)} | = P(S_j \| G, W, s_j ∉ B̄), because W ⟂ S (H1) |
| uniform among sources of accepted scenes | u_D (P_acc ≈ 1) | yes per scene. In the ω-population the scene marginal is tilted ∝ Π u_D(s_i) · Λ_L(G, S) |
| uniform over effective source-window examples | ∝ Σ_W I_j δ_{s_j} | approximately: measured tilt of the PIT deciles ≤ 1.7 percentage points (E3) |
| uniform under training weights | ∝ Σ_W ω I_j δ_{s_j} | approximately, same order (E3) |

None of these marginals has to be uniform for the conditional label to be right. Conversely, a flat marginal would not
prove anything about the conditional.

## 3. The Student's population optimum (single-channel, flattened)

The Student sees X = (SOLID∩W, r_j, L, o) only. Under the flattened ω-population (01 (∗∗)), integrating s_j over F gives

    P_flat(G | X) ∝ π(G) · [vol F_W^j(G) / vol D(G)] · H_t(G, W) · Λ_e(G, W, r_j)

    H_t(G, W) = Σ_k κ_t(k) · k · Σ_{m'} C(k−1, m') a^{m'} b^{k−1−m'} c_t(1 + m', L, k)

The factor k counts the k exchangeable channels of a scene. The optimum is p*(s | X) = Σ_G P_flat(G | X) q_G(s).
The natural single-source reference N1 has the same expression with H ≡ 1 and Λ_e ≡ 1. Hence:

* **Balancing tilt [A].** H_t depends on G through a(G, W) and b(G, W): how often a random hidden source produces a partial shadow in W. With c(1) = 0.01 and c(4) = 1 (k = 4), H varies by up to ~100× across geometries compatible with the same X. If H ∈ [H_min, H_max] on the support of P_N1(G | X), then TV(p*_flat, p*_N1) ≤ 1 − H_min/H_max (the bound of SELECTION_LAW_1K §4, applied to G). Here the bound is vacuous, so the effect must be measured.
* **Measured in the toy** (E5: same structure, realised 1K κ and median L64 c-table) [C]:
  * Student-level TV between the B1K optimum and the N1 optimum: mean 0.031 under the training distribution, p90 0.053, max 0.087;
  * excess log-loss of the B1K-optimal Student evaluated on N1: 0.0042 nats;
  * the distribution of X itself shifts by TV 0.24;
  * with the debalancing weight ω/c(M) the Student-level TV drops to 2.5·10⁻⁵. The remainder comes from the exclusion of the other sources, which tilts weakly.
* **Label-availability tilt [A, C].** Λ_e depends on the label path (single-cell vs statistical vs exact), which is a function of (G, W, r_j).
  * The current default loader (`accept_statistical=False`) keeps only `certified` labels.
  * E5b (toy, keep only single-cell labels): mean TV 0.22, max 0.96, and 42% of the observation mass disappears.
  * Dropping the broadest 5% of labels instead: mean TV 0.004, max 0.25.
* **The direction of the bias is interpretable (E).** Balancing up-weights hidden geometries with many shadow-casting structures near W from random directions. The Student is therefore trained to believe that the hidden world around a window is more cluttered than a natural world would be. Single-cell filtering trains it to believe that posteriors are point-like.

**Multi-channel Student [A].** Given all k channels, the joint posterior given G factorises:

    P(s_1..s_k | G, W, r_1..r_k) = Π_j Uniform(F_W^j)      (anti-duplicate factor omitted)

c_t(M, L, k) and κ_t(k) are functions of the observed (k, M) within a slot, so they cancel from P(G | X_all), except
through the slot mixture (different slots have different c-tables and family). The balancing is therefore benign for a
Student that sees every channel, and harmful for the flattened single-channel Student that the labels are designed for.

**Uninformative channels [A].** Their per-channel label Uniform(F) is still exact (SELECTION_LAW note). But they exist
only when M₋ⱼ ≥ 1, so h_t(G, W, 0) forces "some hidden source was informative here". Training a single-channel Student
on the release's all-lit or all-dark channels would learn that artefact.

## 4. Definitive statement (brief §3.5)

1. Uniform(F) labels and ω are **mutually consistent at the label level**. Under H1–H8, q_G is exactly P_ω(S_j | t, G, W, r_j). The proof is in §1, confirmed by toy and real-label checks.
2. They are **not sufficient for the intended Student target**, unless that target is declared to be the release's own balanced multi-source population B1K (which depends on the controller's run history, 01 §4).
3. For a natural single-source target N1:
   * the weight ω/c_t(M, L, k) is exact and computable from stored data, but costs 6.5× in ESS (02 §3);
   * the label filter must not depend on the label path (02 §6);
   * the uninformative channels of this release must not be used as single-channel training data.
4. A training weight cannot correct a mismatch whose factor is not a function of stored variables. Here the factors c(M) and Λ_e are stored, so correction is possible in this release. The cost is variance (ω/c) or labels (all paths must be accepted).
