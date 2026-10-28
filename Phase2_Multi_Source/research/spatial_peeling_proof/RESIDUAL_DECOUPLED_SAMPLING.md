# Proof of Residual Decoupled Sampling for the Peeling Algorithm

## 1. Result, model, and scope

**Main result.** Once a response is decomposed into validated full-window unit-source masks, candidate selection decouples: any combination of exact single-layer matches reconstructs the observation. With approximate matches, the total reconstruction MAE is at most the decomposition error plus the sum of layer errors. A shared candidate pass avoids enumerating joint configurations; independent rejection sampling has an additive expected draw budget instead of the multiplicative budget for a specified labeled joint target. These results are conditional on obtaining the masks and covering their candidate supports. They do not prove an unconditional recovery or runtime guarantee.

Fix scene geometry and one sampled observation window. Let its full set of pixels, **including obstacle pixels with zero response**, have cardinality \(P>0\). An admissible candidate position \(s\) is inside the world, outside the observation window, and outside blockers. Let

\[
V(s)\in\{0,1\}^{P}
\]

be its full-window unit-strength visibility mask, zero on obstacles. The finite candidate grid is \(\mathcal G\), \(|\mathcal G|=G\). Visibility depends on fixed geometry and candidate position, not on the other light sources. Sources do not block each other; there is no decay, noise, clipping, pixel averaging, or brightness quantization. Masks may overlap arbitrarily in image space.

For a validated decomposition, write

\[
R_0=\sum_{t=1}^{m}L_t,\qquad
C_t=\{s\in\mathcal G:V(s)=L_t\}.
\tag{1}
\]

Here \(m\) is the number of selected layers, fixed for the following sampling analysis. Validation means both the response identity and \(C_t\ne\varnothing\), with physically admissible candidates, not merely a nonnegative residual. The selected layers need not be the generating sources. Generating positions off the grid are harmless if some on-grid positions have the required sampled masks; otherwise coverage can fail. Zero/invisible sources cannot be recovered from this observation and are excluded from the peeling target.

Candidate position and source truth are distinct. Multiple positions can have the same mask. Different layers can overlap spatially even when their candidate-position supports are disjoint. If \(L_i\ne L_j\), then \(C_i\cap C_j=\varnothing\), because a fixed position has only one mask.

The target is compatibility on these sampled pixels. This does not establish unsampled continuous-field agreement, unique source positions, a calibrated posterior, or coverage of every compatible decomposition. Shared geometry, unknown source count, priors coupling source positions, and interactions between sources are not silently integrated out.

## 2. The physical residual invariant

Index the initial residual by \(R_1=R_0\), and set \(R_{t+1}=R_t-L_t\).

**Lemma 1 (two different invariants).** Algebraically,

\[
R_t=R_0-\sum_{j<t}L_j
\tag{2}
\]

holds for every sequence of subtractions. If, additionally, the masks removed at each stage are summands of one compatible physical decomposition of the current residual, then \(R_t\) is the sum of the remaining physical source contributions.

**Proof.** The algebraic identity follows by induction on the update equation. For the physical statement, suppose \(R_t=\sum_{j\in I_t}M_j\), where each \(M_j\) is an admissible unit-source mask, and \(L_t=M_{i_t}\) for some \(i_t\in I_t\). Then

\[
R_{t+1}=\sum_{j\in I_t\setminus\{i_t\}}M_j.
\]

This supplies the next physical decomposition and preserves nonnegativity. ∎

Theorem 7 of [the spatial-ambiguity note](PROOF.md#6-what-can-be-proved-about-peeling) supplies sufficient singleton-witness, pure-anchor, coverage, and identification conditions for obtaining this physical invariant. Nothing here removes its geometric qualifications. In particular, crossing-shadow ambiguity can persist as spatial resolution increases.

**Why nonnegativity is insufficient.** In the admissible mask dictionary

\[
\{(1,1,0),(0,1,1),(1,0,0)\},
\]

response \((1,1,1)\) is the sum of the last two masks. Subtracting the first mask is nonnegative but leaves \((0,0,1)\), which has no decomposition in the dictionary. Equation (2) still holds; its physical interpretation fails. Local subtraction can fail in a different way: subtracting only part of a true source mask leaves some of that source in the residual and need not produce a physical whole-source decomposition.

**Correctness of the decomposition remains a separate obligation.** An image can telescope into binary effective layers without any layer being realizable by a source. A finite candidate check can validate sampled physical realizability after extraction. It does not justify inferring layer membership from response transitions alone.

For a physical example, in a blocker-free scene every exterior unit source has the all-ones mask. The valid observation \((1,1,1,1)\) can be split algebraically into \((1,0,0,0)+(0,1,1,1)\), but neither nonconstant layer can be a source mask anywhere in that geometry. This is stronger than the separate case of a realizable mask missed by a finite grid.

## 3. Exact residual decoupling

**Theorem 2 (Cartesian compatibility).** Under (1), every tuple

\[
(s_1,\ldots,s_m)\in C_1\times\cdots\times C_m
\]

is physically admissible and satisfies \(\sum_tV(s_t)=R_0\). If distributions \(q_t\) are supported on \(C_t\), independent samples \(S_t\sim q_t\) reconstruct \(R_0\) with probability one.

**Proof.** Each selected candidate has \(V(s_t)=L_t\). Add the identities and apply (1). Source admissibility is checked individually and there are no extra cross-source constraints in this model. The product law assigns probability one to the Cartesian support. In fact, independence is not necessary for compatibility: any joint law supported on that Cartesian set has the same guarantee. ∎

This is a sufficient, tractable subset of the full inverse solution set. It can be strict. For example, masks \((1,1,0)\) and \((0,0,1)\) produce the same response as masks \((1,0,0)\) and \((0,1,1)\). A sampler restricted to the first layer family need not return the second one. These finite-mask examples illustrate the algebraic statement; they are not claims that arbitrary binary masks are realizable in every fixed blocker geometry.

**Corollary 3 (safe observed-zero screening).** If (1) holds, removing candidates illuminating any observed zero-response pixel does not remove any \(C_t\).

**Proof.** At a pixel where \(R_0=0\), the sum of nonnegative \(L_t\) is zero, so every \(L_t=0\). Every exact match therefore has zero visibility there. ∎

This corollary is specific to exact, nonnegative, noiseless decomposition. It does not justify the same coverage claim when approximate observations or layers introduce erroneous zero pixels. Any additional pruning must be shown separately to preserve the desired supports.

If a mask is repeated, independent sampling can choose the same position more than once. This is allowed under the repository's with-replacement joint proposal convention. A model requiring distinct source positions needs an additional constraint and potentially a different sampler. For distinct layer masks the supports are already disjoint, so exact selections cannot coincide in position.

**Not a posterior-factorization theorem.** A product of arbitrary supported \(q_t\) is a compatible sampling distribution, not automatically the joint distribution induced by a physical prior and a global observation cost. Conditioning on a selected decomposition can exclude other solutions. A normalized elementwise maximum or average of layer probabilities lives on one candidate grid, not on \(\mathcal G^m\), and loses source-combination information.

## 4. Approximate decomposition and sampling

Use the full-window normalized MAE

\[
\|x\|_{\mathrm{MAE}}=\frac1P\sum_{p=1}^{P}|x_p|.
\]

This matches the joint evaluator's unweighted mean absolute additive-intensity error, including obstacle zeros. Fix extracted layers and assume

\[
\left\|R_0-\sum_tL_t\right\|_{\mathrm{MAE}}\le\delta,
\qquad e_t(s)=\|V(s)-L_t\|_{\mathrm{MAE}}.
\tag{3}
\]

**Theorem 4 (error budget).** For arbitrary admissible selections,

\[
D=\left\|R_0-\sum_tV(s_t)\right\|_{\mathrm{MAE}}
\le\delta+\sum_t e_t(s_t).
\tag{4}
\]

**Proof.** Write the error as \((R_0-\sum_tL_t)+\sum_t(L_t-V(s_t))\) and apply the triangle inequality to this norm. ∎

Consequences, conditional on the fixed extracted layers:

1. **Deterministic:** if each sampled support has \(e_t\le\epsilon_t\), then \(D\le\delta+\sum_t\epsilon_t\) for every tuple.
2. **Expected:** if \(\mathbb E e_t(S_t)\le\bar e_t\), then \(\mathbb E D\le\delta+\sum_t\bar e_t\), without independence.
3. **Tail budgets:** if \(\Pr(e_t(S_t)>\epsilon_t)\le\alpha_t\), then
   \[
   \Pr\{D>\delta+\sum_t\epsilon_t\}\le\sum_t\alpha_t.
   \tag{5}
   \]
   This is a union bound and also needs no independence.
4. **Expectation-only tail:** for any \(a>0\), Markov's inequality gives \(\Pr(D>a)\le\min(1,(\delta+\sum_t\bar e_t)/a)\).

These are upper bounds, not identities: layer errors can cancel. Swapping two distinct target masks between two samples gives nonzero individual errors but can give total error zero. Therefore failure to match every selected layer does not necessarily imply reconstruction failure.

Errors in recursively inferred layers can change subsequent connectivity and anchors discontinuously. Equation (4) uses a measured or established decomposition defect \(\delta\); it does not assume a small first-step error guarantees small later layer errors. If valid decomposition occurs with probability at least \(1-\gamma\), an unconditional failure bound adds \(\gamma\) to the conditional sampling tail. Bounds \(\gamma_t\) conditional on all preceding peeling steps being correct can be summed by first-failure events; stage independence is unnecessary. The previous note does not supply a universal resolution-dependent \(\gamma\).

### Finite-temperature distributions

Suppose a probability measure \(\mu\) on the candidate grid is used as base mass, \(\tau>0\), and

\[
q_t(s)=\frac{\mu(s)\exp[-e_t(s)/\tau]}{Z_t},\qquad
Z_t=\sum_{s\in\mathcal G}\mu(s)\exp[-e_t(s)/\tau].
\tag{6}
\]

If \(p_t=\mu(C_t)>0\), then \(Z_t\ge p_t\). Every finite-cost candidate with positive base mass receives positive probability. Hence off-support mass is strictly positive whenever such off-support candidates exist. Exact layer matching requires conditioning/filtering onto \(C_t\), or separately proving all sampled candidates are exact. Off-support mass does not rule out accidental cancellation in the final reconstruction.

For \(\epsilon>0\),

\[
q_t\{e_t>\epsilon\}
\le\min\left(1,\frac{1-p_t}{p_t}\exp[-\epsilon/\tau]\right).
\tag{7}
\]

Indeed the numerator is bounded by \((1-p_t)e^{-\epsilon/\tau}\), and the denominator by \(p_t\). Insert this bound into (5), or compute the actual finite-grid mean/tail directly. If \(p_t=0\), this particular bound is unavailable; even filtering cannot produce an exact grid match.

The implementation may score a boundary-weighted response cost plus an edge cost. Such a score must not be identified with \(e_t\) without conversion. If \(c_t=\alpha c_{w,t}+\beta c_{\mathrm{edge},t}\), \(\alpha>0\), \(\beta\ge0\), \(c_{\mathrm{edge},t}\ge0\), and \(w_p\ge w_{\min}>0\), then

\[
e_t\le\frac{\sum_pw_p}{P w_{\min}\alpha}\,c_t.
\tag{8}
\]

This follows by bounding the weighted error below by \(w_{\min}\sum_p|V-L_t|\). Uniform weights, \(\alpha=1\), \(\beta=0\) recover the raw MAE. If \(\alpha=0\), no MAE guarantee follows from this argument. Formula (7) as written uses the MAE-based Gibbs law (6), rather than silently substituting an arbitrary code score.

## 5. Work saved by a shared candidate pass

Let \(C_{\mathrm{render}}(P)\) denote the cost of rendering one candidate against fixed geometry; blocker count and backend are included in this function. Let \(T_{\mathrm{peel}}\) include layer extraction and any mask-identification work needed to obtain the validated targets. Rendering once per admissible grid position and comparing it with each layer costs

\[
T_{\mathrm{decoupled}}
=T_{\mathrm{peel}}+O(G C_{\mathrm{render}}(P)+mGP).
\tag{9}
\]

Geometry/grid construction and candidate screening must be added if not included in these terms. Normalizing \(m\) distributions costs \(O(mG)\), absorbed by the scoring term. The expression describes unweighted or linear-time weighted response scoring; more expensive edge metrics contribute their own scoring cost. Cached masks use \(O(GP)\) bits if packed; streaming masks can reduce mask storage, while retaining all layer scores uses \(O(mG)\) numbers.

An exhaustive **labeled, with-replacement** search has \(G^m\) tuples. Reusing cached visibility still requires only \(G\) distinct physical renders, so it is incorrect to claim that decoupling reduces \(G^m\) renders to \(G\) when comparing with a cached baseline. It avoids enumerating and evaluating those joint tuples. Direct assembly and MAE evaluation cost \(O(mP)\) per tuple; incremental assembly can reduce factors, but does not make the enumeration count linear in \(m\).

If source order is irrelevant, exhaustive position multisets with repetition number

\[
\binom{G+m-1}{m},
\]

and distinct-position subsets number \(\binom Gm\). These counts describe different domains. For fixed \(m\) and large \(G\), removing order contributes approximately a factor \(m!\); it does not justify comparing unlike priors or claiming exponential growth in every parameter regime.

Exact mask supports can alternatively be indexed by hashing cached masks, avoiding an explicit all-layer cost matrix when only exact equality is needed. Equation (9) records the existing layer-scoring approach, not an optimality lower bound. No training implementation change is proposed.

## 6. Independent rejection sampling versus joint proposals

Fix a declared proposal law \(\mu\), with replacement, and \(p_t=\mu(C_t)>0\). Counts here refer to **proposal draws**, not distinct renders: repeated positions can reuse cached masks. Target masks are held fixed for the comparison.

**Proposition 5 (separate streams).** Independently propose candidates for layer \(t\) until one belongs to \(C_t\). The stopping time \(T_t\) is geometric and

\[
\mathbb E\sum_tT_t=\sum_t1/p_t.
\tag{10}
\]

For distinct layer masks under the same base proposal, the supports are disjoint and \(\sum_tp_t\le1\). Cauchy's inequality therefore gives \((\sum_t1/p_t)(\sum_tp_t)\ge m^2\), so this rejection budget is at least \(m^2\), not universally \(O(m)\). Drawing once from each already-constructed supported distribution takes \(m\) output draws; constructing those distributions and rejecting base proposals are different operations. Adaptive or layer-specific proposal laws need their own masses and are not subject to this common-proposal calculation.

With budgets \(b_t\), independent streams have exact all-layer success probability

\[
\prod_t[1-(1-p_t)^{b_t}],
\]

and failure is at most \(\sum_t(1-p_t)^{b_t}\le\sum_te^{-b_tp_t}\). Thus \(b_t=\lceil\log(m/\eta)/p_t\rceil\) ensures failure at most \(\eta\in(0,1)\). For \(p_t=1\), one draw suffices. If any \(p_t=0\), exact rejection sampling for that layer cannot succeed. These follow from the geometric tail and a union bound. Independence is used for the product success formula, not for linearity of expectation or the union bound.

**Proposition 6 (joint proposals for the same target family).** One labeled trial draws \(m\) independent positions from \(\mu\). Its probability of hitting \(C_1\times\cdots\times C_m\) is

\[
a_{\mathrm{lab}}=\prod_tp_t.
\tag{11}
\]

The expected number of trials is \(1/a_{\mathrm{lab}}\), representing \(m/a_{\mathrm{lab}}\) position draws if every trial draws all positions. Early-abort implementations can spend fewer draws per failed trial; their cost must be counted separately. With independent trials, \(N\) trials succeed with probability \(1-(1-a_{\mathrm{lab}})^N\). For \(0<a<1\), the exact confidence budget is

\[
N\ge\left\lceil\frac{\log\eta}{\log(1-a)}\right\rceil;
\tag{12}
\]

use one trial when \(a=1\). For distinct masks, accepting any permutation gives

\[
a_{\mathrm{unord}}=m!\prod_tp_t,
\tag{13}
\]

because the supports are disjoint and the permutation events are disjoint. For \(k\) distinct mask classes of multiplicities \(r_j\) and masses \(p_j\), the general expression is

\[
a_{\mathrm{unord}}=\frac{m!}{\prod_{j=1}^k r_j!}\prod_{j=1}^kp_j^{r_j}.
\tag{14}
\]

The same geometric trial calculation then applies. A uniform distribution over unordered multisets is different from drawing iid ordered positions and canonicalizing them; do not mix these measures.

**Illustration.** Two distinct supports with \(p_1=p_2=1/4\) give expected separate draws 8, labeled joint trials 16 (32 full-trial position draws), and unordered joint trials 8 (16 position draws). The advantage is not always large. In a family with \(m\) disjoint classes each of mass \(1/m\), separate draws average \(m^2\), labeled joint trials average \(m^m\), and unordered joint trials average \(m^m/m!\). These are target-family calculations, not unconditional guarantees about arbitrary scenes or the cost of extracting the layers.

The event “some source tuple reconstructs \(R_0\)” can be larger than this chosen family, as in the alternative-decomposition example in section 3. Its joint-proposal success probability can therefore exceed (11) or (13), potentially substantially. Unknown-count proposal mixtures add the declared count prior. This note compares proposals conditional on the selected count \(m\), without reading generating source-count truth.

## 7. Shared pools are not independent rejection streams

The current code traverses a common candidate grid, rather than drawing a fresh independent pool for every layer. For \(B\) iid shared draws, let \(H_t\) mean the pool contains a match for layer \(t\). Then

\[
\Pr(\exists t:H_t^c)\le\sum_t(1-p_t)^B.
\tag{15}
\]

Coverage events need not be independent. If two supports are disjoint and \(B=1\), covering both is impossible, despite both individual hit probabilities being positive. For \(k\) distinct mask classes, inclusion-exclusion gives the exact formula

\[
\Pr(\text{all covered})=
\sum_{J\subseteq\{1,\ldots,k\}}(-1)^{|J|}
\left(1-\sum_{j\in J}p_j\right)^B,
\tag{16}
\]

with the empty-set term 1 and \(B\ge1\). To prove it, the event that all classes in \(J\) are missing has probability \((1-\sum_{j\in J}p_j)^B\), since distinct mask classes have disjoint supports. Apply inclusion-exclusion to missing-class events. Repeated target masks require only one matching candidate if reuse of a position is permitted; a distinct-position model needs multiplicity-aware coverage instead.

For a uniform pool of \(B\le G\) candidates sampled **without replacement**, a class with \(g_t\) positions is missed with probability

\[
\frac{\binom{G-g_t}{B}}{\binom GB},
\tag{17}
\]

where impossible binomial choices are zero. The union bound remains valid, and inclusion-exclusion replaces each term of (16) with \(\binom{G-\sum_{j\in J}g_j}{B}/\binom GB\). Enumerating the entire grid deterministically covers every nonempty \(C_t\), provided screening preserves it. A finite grid still cannot cover a mask absent from that grid.

## 8. Comparison with the pinned implementation

The proof branch was rechecked at `d785a7e`; the unit-strength implementation remains pinned to `77c03a031b923953197184ba16cf6dd6e55c5e48`.

| Current behavior | Consequence for this proof |
|---|---|
| [Full-window joint MAE](https://github.com/yiyuqian0912/G2-02/blob/77c03a031b923953197184ba16cf6dd6e55c5e48/Phase2_Multi_Source/src/rind_phase2/physics.py): all sampled pixels, obstacle zeros included | Matches the norm in section 4; response is not clipped or binarized. |
| [One-hop teacher](https://github.com/yiyuqian0912/G2-02/blob/77c03a031b923953197184ba16cf6dd6e55c5e48/Phase2_Multi_Source/src/rind_phase2/layered_teacher.py): bounded sampling/beam, local subtraction, final full-response check | Local subtraction is not automatically a full-source removal. Grid gaps and pruning can destroy coverage. Final compatibility checks help validate retained combinations but do not prove completeness. |
| [BFS residual extraction](https://github.com/yiyuqian0912/G2-02/blob/77c03a031b923953197184ba16cf6dd6e55c5e48/Phase2_Multi_Source/research/layered_residual/layered_residual.py): infers effective layers before candidate scoring | Layer membership is not established merely by equal/increasing brightness. Effective layers above strength 1 cannot be treated as individual unit sources. Algebraic reconstruction does not validate \(C_t\ne\varnothing\). |
| Same residual module: one visibility pass, score against each layer, observed-zero screening | Supports the rendering/scoring accounting in (9). Exact zero screening is safe by Corollary 3 when its hypotheses hold. It is not independent per-layer rejection sampling. |
| Same module: layer softmax over retained candidates | At finite temperature, low-cost distributions can retain nonmatching masks; use actual means/tails or explicit filtering. Weighted/edge costs require the conversion discussed in (8). |
| Same module: normalized elementwise maximum of layer distributions | A one-position summary, not a joint source law. Sampling it \(m\) times can repeatedly select the same layer and omit another, even when each original layer distribution is exact. |

For the last row, two exact point-mass layer distributions on different positions combine into a uniform distribution on those two positions. Two independent draws from that summary choose one of each with probability only \(1/2\). In contrast, drawing once from each original layer distribution reconstructs with probability 1.

The earlier proof and its assumptions remain unchanged. No student, teacher, training, or inference API is modified. A student conditioned only on local observations does not gain access to hidden geometry merely because the geometry-assisted teacher can validate masks.

## 9. Verification and reproducibility

Run from this directory:

```sh
python3 -m unittest -v
python3 verify_residual_sampling.py --output residual_sampling_results.json
```

The new checks use only the Python standard library. Exact finite enumeration verifies overlapping-mask reconstruction, permutation factors including repeated mask classes, shared-pool inclusion-exclusion, without-replacement coverage, and cancellation of layer errors. Seeded sampling checks independent rejection expectations and joint trial probabilities. Additional checks distinguish grid-coverage failure from physically impossible layers in a blocker-free geometry, and cover softmax leakage, tail bounds, the normalized-max counterexample, and the earlier local-subtraction/mixed-anchor failures. Operation counters distinguish renders, layer scores, and joint tuple evaluations.

Synthetic masks are explicit finite visibility dictionaries; these experiments do not assert that every listed mask can be embedded in every physical geometry. The proof applies to any physically admissible dictionary satisfying its assumptions. The scripts verify identities and examples, not actual dataset frequencies, asymptotic optimality, or decomposition correctness of the current BFS pipeline.
