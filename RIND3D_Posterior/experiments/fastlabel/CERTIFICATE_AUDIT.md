# FastLabel certificate audit (2026-11-12)

**Scope:**
* the statistical tier of FastLabel v2 (`adaptive2.py`, checkpoint 3);
* its replacement v3 (`adaptive3.py` with `gpuverify.py` / `verify.cu`).

**What is audited:** what is certified, under which assumptions, with what probability, and which error terms remain.
Empirical validation (agreement with references, Monte Carlo checks) is kept separate from the certificate.

## 1. Target

The target is the contract `rind3d-1k-contract-v1` (docs/CONTRACT_1K.md and SELECTION_LAW_1K.md, Theorem §2):

    q_F = cell masses on the 128³ grid of Uniform(F),
    F = F_W^j = {s ∈ D(G) \ B̄(W) : V_G(s, x) = r_j(x) for every FREE receiver x of W},
    D(G) = {p ∈ (0,1)³ : p ∉ solid, certified clearance > 2⁻²⁶}.

**Relation to the sampling law.** The theorem shows that q_F is the exact per-channel posterior of the weighted 1K
population, under two stated conventions:
* P_acc ≡ 1;
* the 2⁻²⁰ anti-duplicate factor is omitted.

FastLabel does not change the target, the prior, the selection law or the weights. It is a cheaper way to compute a
label with a guarantee *relative to this q_F*.

## 2. The identity used by the statistical tier

Let B be measurable with F ⊆ B and 0 < |F|. Let U_A denote the uniform law on A, and p = |F| / |B|. Then:

    TV(U_B, U_F) = sup_A |U_B(A) − U_F(A)| = U_F(F) − U_B(F) = 1 − p.

The sup is attained at A = F, because U_F(A) − U_B(A) = |A∩F|/|F| − |A|/|B| ≤ |A∩F| (1/|F| − 1/|B|) ≤ 1 − p.

The cell masses q_B and q_F are pushforwards of U_B and U_F under the same map (the cell index). By data processing:

    TV(q_B, q_F) ≤ 1 − p.

**Conditions for using the identity, and how they hold in the code:**

| # | condition | status |
|---|---|---|
| C1 | F ⊆ B | Deterministic. B = F(X′) is the TPE engine's feasible set for a receiver subset X′ ⊆ FREE(W). Dropping receivers only removes constraints. Rounds r ≥ 2 intersect with an outer filter of B_{r−1}, so B_r ⊇ F inductively. **Assumption A1:** the engine's OUT decisions are sound. This is the same assumption that the reference (exact) labels make; the engine uses the binary64-rounded geometry rows with decision margins. **Assumption A2:** the B-filter boxes (vertex bounding boxes of pieces smaller than ¼ cell, margin 1e-13) contain the pieces, i.e. the float vertices of the engine's polytopes are accurate to ≪ 1e-13. |
| C2 | the label is q_B for the *same* B that is sampled | Yes: in both v2 and v3 the label is `cell_masses(res.cell, res.hi, res.hi)` of the round that certified, and the samples are drawn from that round's pieces. |
| C3 | the samples are i.i.d. U_B | Pieces are chosen ∝ `hi` (exact polytope volume in binary64). The point is drawn uniformly in the convex hull of the piece's vertices: Delaunay tetrahedralisation, tetrahedron ∝ volume, Rocchini–Cignoni uniform barycentric. Audit `audit_sampler.py` on 6 windows (up to 335,806 pieces): the tetrahedralisation volume and `hi` agree to ≤ 3.3e-8 relative per piece (p99 ≤ 3.5e-12), and the mass-weighted discrepancy is ≤ 1.2e-11. So TV(sampling law, U_B) ≤ η_samp, with η_samp ≈ 1e-11 (deterministic, negligible). Within a round the points are independent given the history (fresh draws from the label's stream). |
| C4 | the membership test of a sample in F is exact | The test checks three conditions: the closed window box (exact binary64 comparisons of dyadic numbers); legality (v2: `legal_source`; v3: a GPU strict-separator certificate of clearance > r for every body, which implies `legal_source`, else `legal_source` itself); and V_G(s, x) = r_j(x) at every FREE voxel centre. v2 used the certified CUDA renderer with exact fallback, **plus tile culling by the engine's float-filtered predicates (margins 1e-11) — an assumption like A1**. v3 uses `verify.cu` with no float-margin culling; every decision is rigorous (§4) and Gate A validates it bit for bit. |
| C5 | vol(F) > 0 (single-cell path only) | The true source s* ∈ F. Almost surely no FREE voxel centre lies on a shadow boundary of s*, and its clearance is strict, so s* is interior to F. |

## 3. The statistical decision

### v2 (checkpoint 3), audited as implemented

**Procedure:**
* Discovery rounds of 16 samples never certify.
* A clean discovery round triggers attempt c (c = 1…8), with δ_c = δ·2⁻ᶜ and n_c = ⌈ln δ_c / ln(1−ε)⌉ fresh samples (104, 117, 131, …).
* Certify iff all n_c pass.

**Validity:**
* Given the history, the attempt's samples are fresh, so P(all pass | p_c < 1−ε) ≤ (1−ε)^{n_c} ≤ δ_c.
* The union over attempts gives ≤ Σ δ_c = δ(1 − 2⁻⁸) < δ = 0.01. Valid per label.

**Findings:**
1. `seed = 0` for every label means the same random stream for every label.
   * Per-label guarantees are unaffected, and so are dataset-level union bounds and expectations, which need no independence.
   * But labels are not independent, and reruns cannot be distinguished from "re-rolls".
   * **Fixed in v3:** a keyed stream per label, SeedSequence(base_seed, scene, window, source).
2. Tile culling relied on float-margin predicates (C4); removed in v3.
3. All-pass tests waste attempts when p ≈ 0.99. This caused the one 195 s exact fallback in run4e, where 8 attempts each saw one failure.

### v3

**Procedure.** Round r = 1…R. The following are all fixed in advance as functions of r only:
* n_r = 1024;
* δ_r = δ·2⁻ʳ;
* k*_r = max{k : P(Bin(n_r, ε) ≤ k) ≤ δ_r}.

At round r:
* n_r fresh i.i.d. points of U_{B_r} are tested, in chunks of 128, 256 and 640.
* The round stops as soon as K_r > k*_r, since a certificate is then impossible.
* Certify iff all n_r are tested and K_r ≤ k*_r.
* Otherwise the violated receivers of the failing points are added to X′ and the next round starts.

**Theorem.** P(∃ r: a certificate is issued at r and p_r < 1 − ε) ≤ Σ_r δ_r < δ.

*Proof:*
1. Let H_{r−1} be the history before round r. B_r, and hence p_r, is H_{r−1}-measurable.
2. Given H_{r−1}, the failure count K_r is Bin(n_r, 1 − p_r).
3. On {p_r < 1−ε}, P(K_r ≤ k*_r | H_{r−1}) ≤ P(Bin(n_r, ε) ≤ k*_r) ≤ δ_r. This holds because the binomial CDF decreases in the failure probability.
4. Stopping a round early only shrinks the certificate event.
5. Take the union bound over r. ∎

**v3.2 two-stage rounds** (`design2`, now the default). Each round has two stages:
* stage 1: n₁ = 512 points; certify iff K₁ ≤ a₁;
* stage 2: otherwise continue to n₂ = 1024 points; certify iff K ≤ a₂.

The thresholds are fixed in advance per round. They are computed exactly at the boundary failure rate ε:
* P(Bin(n₁, ε) ≤ a₁) ≤ δ_r / 2;
* a₂ is the largest value with P(K₁ ≤ a₁) + Σ_{k₁ = a₁+1}^{a₂} P(K₁ = k₁) P(Bin(n₂ − n₁, ε) ≤ a₂ − k₁) ≤ δ_r.

The certificate event is an increasing function of the per-point pass indicators. Coupling the indicators as 1[U_i < p] shows that its probability is nondecreasing in p, so for every p < 1 − ε it is at most its value at p = 1 − ε, i.e. at most δ_r. The theorem above therefore holds unchanged.

**Round-1 values at the dataset-simultaneous δ:**

| quantity | value |
|---|---|
| δ_1 | 3.1e-8 |
| a₁ | 3 |
| a₂ | 17 |
| exact α | 2.92e-8 |

* The power at p = 0.99 is the same as the single-stage n = 1024 design (0.983).
* Labels with p ≥ 0.998 stop at stage 1 with probability ≥ 0.98.
* In the run4e sample, 402 of 457 statistical labels stopped at stage 1.

**Remarks:**
* Adaptive receiver selection, the data-dependent number of rounds, and the reuse of failing samples for constraints are all covered, because every certificate uses only fresh samples whose law is fixed by the past.
* Failed or interrupted labels: the stream is keyed, so a rerun reproduces the same decisions. It is not a fresh "re-roll" that would need extra δ.
  * **Rule:** never rerun a label with a new seed after a failed certificate unless that is charged to δ.
* Exact paths (single-cell support, exact small B, exact race, exact fallback) carry the engine's deterministic `tv_bound`. In run4e it is at most 3.3e-7.

## 4. Exactness of the v3 point test (`verify.cu`)

**Shadow rows.** They are built from the exact body-row enclosures with directed binary64 interval arithmetic
(`__dmul_rd/ru`, `__dadd_rd/ru`, `__dsub_rd/ru`, `__ddiv_rd/ru`).
* Each row encloses a positive multiple of the exact rational row of `polyhedra.shadow_planes`: same face signs, same pair rows (c_j r_i − c_i r_j, −n·p), and a positive normalisation.
* A source whose face sign or row cannot be certified is redone on the host with the exact rational construction (`fastplanes`, verified identical to `shadow_planes`).

**FP32 filter.** Rows are normalised so that max|coef| ≤ 1. Voxel centres (2i+1)/2048 and box corners are exact in binary32.
* `--fmad=false`, so every product and sum is rounded separately.
* With u = 2⁻²⁴, |fl32(v) − v_exact| ≤ γ₄ · 4(1+2⁻²³) + 4(2⁻²⁴ + 2⁻³⁰) ≤ 1.25·2⁻²⁰ < E32 = 2⁻¹⁸.
  * The first term is the summation of 4 terms, each |·| ≤ 1 + 2⁻²³.
  * The second term is the coefficient rounding plus the enclosure width; rows wider than 2⁻³⁰ get NaN FP32 coefficients and always use FP64.
* So |fl32(v)| > E32 decides the sign of v. The same bound applies to box minima and maxima.

**Fallbacks:**
* Otherwise the row is evaluated with the renderer's directed FP64 intervals.
* Otherwise the voxel is UNKNOWN and decided by the native GMP predicate (`native.points`), as in the renderer.

**Semantics:**
* closed bodies;
* closed shadows (v ≤ 0 inside);
* SOLID from the stored crop, which is checked once per window against the canonical geometry by the same kernel in solid mode, with exact fallback.

**Gate A** (`validate_verify.py`, results in `gatea/`). Per legal candidate, the kernel's predicted class of every FREE voxel (dump mode, UNKNOWN decided exactly) is compared with the reference certified renderer over all tiles.
* Candidates:
  * uniform in B;
  * near the true source (1e-9 … 1e-3);
  * at engine vertices;
  * uniform in the world;
  * within one ulp of body face planes (uncertifiable signs, which take the host-row path).
* Separate builds widen the FP32 band (E32 = 1e-3, exercising FP64) and force UNKNOWN (|v| < 2e-4, exercising the exact host path).
* Batch decisions and every reported violated receiver are checked too.

## 5. Error budget (TV of the stored label to q_F)

| term | kind | magnitude |
|---|---|---|
| receiver-subset relaxation (B vs F) | certified statistically | ≤ ε = 0.05, except with probability ≤ δ |
| sampling law vs U_B | deterministic | η_samp ≈ 1e-11 (measured, §2 C3) |
| engine volumes / float polytopes | deterministic | relative ~1e-12 (binary64 clipping) |
| point-test predicates and renderer | exact | 0 (§4) |
| legality domain, window exclusion | exact | 0 |
| outer-bound soundness F ⊆ B | assumption A1, A2 | 0 if A1 and A2 hold, as for the reference labels |
| selection law / prior | contract-level | 0 relative to the contract target (conventions of §1) |
| storage | float64 masses | 0 |

Hence, w.p. ≥ 1 − δ: TV(label, q_F) ≤ ε + η_samp + η_vol ≈ 0.05.

The confidence δ is a probability that this bound fails; it is never added to the TV.

**Confidence curve.** One certificate (n, K) also yields, for every ε′, P(TV > ε′ and certified) ≤ P(Bin(n, ε′) ≤ K). For example, with n = 1024 and K ≤ 17 the bound is ≤ 3e-26 at ε′ = 0.10. So the secondary target holds with overwhelming confidence.

## 6. Per-label and dataset-level reliability

**Policies:**

| policy | per-label δ | statement | v3 cost (n, k* at round 1) |
|---|---|---|---|
| v2 | 0.01 | each label: P(invalid certificate) ≤ 1%; E[#invalid] ≤ 0.01 × N_stat (worst case ≈ 500 for ≈ 53k statistical labels) | n = 104, k* = 0 (all-pass) |
| per-label 1e-4 | 1e-4 | E[#invalid] ≤ 5.3 (worst case) | n = 768, k* = 16 |
| **dataset-simultaneous (v3 default)** | 0.01 / 159,802 = 6.3e-8 | **P(any invalid statistical certificate in the whole 1K) ≤ 0.01** (union bound; no independence needed) | n = 1024, k* = 17 |

* A per-label 99% certificate does **not** imply that all labels are simultaneously correct with 99% confidence.
* The simultaneous policy needs about 10× more point tests per round than v2. This was unaffordable at 20–40 ms per test. With the batched GPU verifier, 1024 tests cost about 0.1–0.3 s per round (measured in §C of REPORT.md).
* **Recommendation:** adopt the dataset-simultaneous policy for the training dataset. Record (n, K, k*, δ_r, round, seed) per label, so that any user can recompute per-label confidence curves.

## 7. What earlier empirical numbers mean

**"TV ≤ 0.015 vs reference"** is the cell-mass TV between our label and the TPE v3.1 reference label.
* Each reference has its own deterministic `tv_bound` (≤ 0.05 for certified references).
* By the triangle inequality, TV(ours, q_F) ≤ 0.015 + tv_bound_ref.
* This is empirical evidence for the certificate, not a proof.

**The case 000195/24/0** (reference `tv_bound` 0.42): the 2000-sample Monte Carlo check (p̂ = 0.9965, 99% CI [0.9915, 0.9990]) is empirical validation. It is not itself a certificate, although it is the same kind of statistical statement.
