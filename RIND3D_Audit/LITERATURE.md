# RIND3D Scientific Audit - Literature Review

Scope: focused literature review for the RIND3D foundations audit (deterministic, noise-free binary
visibility inverse problem; local window observation; hidden global polyhedral geometry G; 1-4 point
sources uniform in free space; exact feasible-set posterior labels q_G; Student network trained without G).

Method: web search and page fetches (2026-11-13). Each entry carries a `Verified:` tag describing how
the bibliographic details were checked. Anything not checked is marked UNVERIFIED. No numerical claims
are made beyond what the cited source states.

Status: COMPLETE (2026-11-13). Sections 1-11 plus implications.

## 1. Bayesian inverse problems

**[Stuart2010] Stuart, A. M. (2010). Inverse problems: A Bayesian perspective. Acta Numerica 19, 451-559. DOI 10.1017/S0962492910000061.**
Review of the Bayesian formulation of inverse problems in function space: prior on the unknown, a
forward map, a data likelihood, and well-posedness of the posterior (existence, and stability with
respect to data perturbations) under conditions on the forward map and prior. The well-posedness
theory is built around additive observational noise (typically Gaussian), where the likelihood is a
smooth function of the data misfit. RIND3D has no noise: the "likelihood" is an indicator
1[s reproduces r], so the posterior is the prior restricted to a feasible set and is generally NOT
continuous in the data (one flipped receiver can change F discontinuously). The framing (posterior =
prior x likelihood, conditioning on observations) applies; the stability theorems do not apply
without adding a noise model.
Verified: search result (Cambridge Core, ADS, Caltech authors record agree on DOI, volume, pages).

**[KaipioSomersalo2005] Kaipio, J. P., & Somersalo, E. (2005). Statistical and Computational Inverse Problems. Applied Mathematical Sciences, vol. 160. Springer. DOI 10.1007/b138659.**
Textbook treatment of inverse problems as Bayesian inference: the solution is the posterior, not a
point estimate; it covers prior modelling, noise modelling, MCMC, and approximation/model errors
(the "approximation error approach" of treating discretisation/model error as an additional random
term). Applies to RIND3D in the basic sense that the answer to "where is the source given the
window" is a distribution; the noise and approximation-error machinery is relevant only if one
argues that the 128^3 discretisation or label certification error should be modelled as noise.
(Some catalogues list 2004 as the year; Springer/CERN list 2005.)
Verified: search result (Springer link page for DOI 10.1007/b138659; CERN record; Aalto portal). Content: the CERN record confirms noise modelling, Bayesian estimation and MCMC; the "approximation error approach" remark is from general knowledge of the book, not re-checked.

**[Tarantola2005] Tarantola, A. (2005). Inverse Problem Theory and Methods for Model Parameter Estimation. SIAM. DOI 10.1137/1.9780898717921.**
Formulates the general inverse problem as combining prior information on model parameters with
information from observations into a "posterior" state of information (a probability density on
model space); emphasises that the full posterior, including multimodality, is the solution. Its
formulation explicitly allows exact (noise-free) theory relations as a limiting case, which is the
RIND3D regime: the posterior becomes the prior restricted to the set of models consistent with the
data. Applies directly as conceptual grounding for "label = feasible-set posterior Uniform(F)".
Verified: search result (SIAM epubs entry for the DOI; front matter shows (c) 2005 SIAM). Content: the "exact theory as a limiting case" remark is from general knowledge of the book (conjunction of states of information), not re-checked page-by-page.

## 2. Simulation-based inference / amortized neural posterior estimation

**[Cranmer2020] Cranmer, K., Brehmer, J., & Louppe, G. (2020). The frontier of simulation-based inference. PNAS 117(48), 30055-30062. DOI 10.1073/pnas.1912789117. arXiv 1911.01429.**
Review of likelihood-free inference for simulators whose likelihood is intractable but which can be
sampled: ABC, neural posterior/likelihood/ratio estimation, amortisation, and the use of extra
simulator information ("mining gold": latent variables, joint likelihood ratios, scores available
inside the simulator). RIND3D is an extreme SBI case: the simulator is deterministic, and the label
generator has privileged access to the latent G - exactly the kind of "augmented" simulator output
the review discusses. Directly applicable framing; the specific augmented-data estimators reviewed
target likelihood ratios/scores rather than feasible-set posteriors.
Verified: search result (PNAS DOI page, arXiv 1911.01429, ADS record with pages 30055-30062).

**[PapamakariosMurray2016] Papamakarios, G., & Murray, I. (2016). Fast epsilon-free inference of simulation models with Bayesian conditional density estimation. NeurIPS 29. arXiv 1605.06376.**
KEY FACT (Proposition 1, Sec. 2.2, close paraphrase): if N pairs (theta_n, x_n) are drawn
independently with theta_n ~ p~(theta) (a "proposal prior") and x_n ~ p(x|theta_n), then as N -> inf
the likelihood prod_n q_phi(theta_n | x_n) is maximised w.r.t. phi if and only if
q_phi(theta|x) proportional to [p~(theta)/p(theta)] p(theta|x), provided a phi achieving this exists.
With p~ = p (the prior) the maximum-likelihood conditional density estimator IS the true posterior
for every x (amortisation); with p~ != p one must reweight by p(theta)/p~(theta) (their Eq. 3,
"SNPE-A"). Assumptions: i.i.d. pairs, infinite data, a realisable (sufficiently flexible) family,
and the global optimum is reached. Applies directly to RIND3D one-hot NLL on (X, S) pairs: the
population minimiser of -log p_theta(S|X) is P(S|X) provided the training pairs come from the
target joint. Note the proposition concerns a proposal on theta (the source); selection that
depends only on x leaves the conditional unchanged, selection that depends on theta or on latents
(G) given x does not.
Verified: fetched arXiv PDF; Proposition 1 and Eqs. (1)-(3) read directly on page 2-3.

**[Lueckmann2017] Lueckmann, J.-M., Goncalves, P. J., Bassetto, G., Ocal, K., Nonnenmacher, M., & Macke, J. H. (2017). Flexible statistical inference for mechanistic models of neural dynamics. NeurIPS 30. arXiv 1711.01861.**
Introduces the importance-weighted SNPE variant (later called SNPE-B): minimise
-sum_j [p(theta_j)/p~(theta_j)] log q(theta_j | x_j) so that the true posterior is recovered
directly when simulations come from a proposal. This is the exact analogue of importance-weighted
NLL under selection. Directly relevant to RIND3D's importance-weighted training; see [Greenberg2019]
for the documented variance cost of these weights.
Verified: search result (NeurIPS proceedings PDF and arXiv 1711.01861); loss form confirmed via the
description in [Greenberg2019] p.3. Full author list taken from the query; the search snippets did
not display all six names - treat the author list as PARTLY UNVERIFIED (first and last authors
confirmed).

**[Greenberg2019] Greenberg, D. S., Nonnenmacher, M., & Macke, J. H. (2019). Automatic Posterior Transformation for likelihood-free inference. ICML 2019, PMLR 97:2404-2414. arXiv 1905.07488.**
States (Sec. 2.2) that training -sum log q_{F(x_j,phi)}(theta_j) on prior-drawn pairs learns the map
x -> p(theta|x) as N -> inf for sufficiently expressive F, q; and (Sec. 2.3, Eq. 1) that training on
a proposal p~(theta) instead yields the "proposal posterior" p~(theta|x) = p(theta|x) p~(theta)p(x) /
(p(theta) p~(x)). It reports that SNPE-B's importance weights p(theta)/p~(theta) "greatly increase the
variance of parameter updates during learning, which can lead to slow or inaccurate inference"
(Fig. 1). APT (SNPE-C) instead trains the reparameterised proposal posterior so q -> p(theta|x) with
no weights. Applies to RIND3D: importance weights are a valid but high-variance correction; if the
selection rule's dependence can be written as a function of the observed window alone, no
correction to the conditional is needed at all.
Verified: fetched arXiv PDF; Sec. 2.2-3 read directly (pages 2-3); venue/pages via PMLR search result.

**[Radev2020] Radev, S. T., Mertens, U. K., Voss, A., Ardizzone, L., & Kothe, U. (2020/2022). BayesFlow: Learning complex stochastic models with invertible neural networks. IEEE Trans. Neural Netw. Learn. Syst. 33(4), 1452-1466. DOI 10.1109/TNNLS.2020.3042395. arXiv 2003.06281.**
Amortised posterior estimation with a learned summary network plus a conditional invertible network,
trained on simulated (theta, x) pairs and then applied to any number of observed datasets. The
abstract presents it as an amortised framework for any simulable forward model; the abstract itself
does not state a formal recovery theorem (the paper body reportedly argues recovery of the true
posterior at the KL-optimum - not checked here). Relevant to RIND3D as the template for a summary
network over a variable local window feeding a global density head.
Verified: search result (IEEE Xplore 9298920, dblp, arXiv); abstract fetched. Online 2020, issue 2022.

**[Lueckmann2021] Lueckmann, J.-M., Boelts, J., Greenberg, D., Goncalves, P., & Macke, J. (2021). Benchmarking simulation-based inference. AISTATS 2021, PMLR 130:343-351. arXiv 2101.04653.**
Benchmark of SBI algorithms on tasks with reference posteriors. Findings stated in the abstract:
the choice of performance metric is critical, sequential methods are more sample-efficient, and
neural-network-based methods generally perform better, with no single best method. Relevant to
RIND3D in that the project has the rare luxury of exact reference posteriors (q_G marginalised is not
available, but q_G per scene is), so it can evaluate metrics such as C2ST/NLL against references; it
warns that conclusions depend on the metric.
Verified: search result (PMLR v130 page, arXiv abstract, sbi-benchmark site).

## 3. Simulation-based calibration and posterior coverage checks

**[Talts2018] Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A. (2018). Validating Bayesian inference algorithms with simulation-based calibration. arXiv 1804.06788 (v2 2020).**
Uses the self-consistency identity (their Eq. 1): integrating the exact posterior over the Bayesian
joint returns the prior, pi(theta) = int dy dtheta~ pi(theta|y) pi(y|theta~) pi(theta~). SBC draws
theta~ from the prior, y from the simulator, L posterior draws, and checks that the rank of theta~
among the draws is discrete-uniform. The paper states SBC "requires just one assumption: that we have
a generative model for our data", and that it is limited to the computational aspect - it gives "no
guarantee that the posterior will cover the ground truth for any single observation". Applies
directly to RIND3D (it needs no likelihood, only joint samples): (a) to validate the label engine,
the importance-weighted average of q_G over the joint must reproduce the source prior restricted to
the admissible region; (b) to validate the Student, ranks/coverage of the true S under p_theta(.|X).
Note (our deduction from Eq. 1, not a claim of the paper): an estimator that ignores X and outputs
the prior also passes this test, so SBC is necessary-not-sufficient. Selection/acceptance changes the
joint, so the identity must be evaluated with the importance weights or on the unselected joint.
Verified: fetched arXiv PDF; Eq. (1) and Sec. 4 quotes read directly on pages 3-4. No journal
version confirmed (arXiv only).

**[Hermans2022] Hermans, J., Delaunoy, A., Rozet, F., Wehenkel, A., Begy, V., & Louppe, G. (2022). A trust crisis in simulation-based inference? Your posterior approximations can be unfaithful. Transactions on Machine Learning Research. arXiv 2110.06581.**
Large empirical study using expected coverage of highest-posterior-density credible regions: all
benchmarked SBI methods (S)NPE, (S)NRE, SNL and ABC variants "can yield overconfident posterior
approximations"; ensembling surrogates mitigates this; calls for conservative inference algorithms.
Expected-coverage curves (empirical vs nominal credibility) are the recommended diagnostic.
Directly applicable to RIND3D's Student: evaluate expected coverage of p_theta HPD regions for the
true S on held-out scenes (with importance weights if the evaluation set is selected). In RIND3D
one can additionally measure coverage relative to the exact q_G, which is stronger than coverage
alone.
Verified: fetched arXiv abstract (comment "TMLR version", v3 Dec 2022) and search results; exact
TMLR issue/date not confirmed.

**[Lemos2023] Lemos, P., Coogan, A., Hezaveh, Y., & Perreault-Levasseur, L. (2023). Sampling-based accuracy testing of posterior estimators for general inference. ICML 2023. arXiv 2302.03026.**
Proposes TARP (Tests of Accuracy with Random Points), a coverage test using random reference points
that needs only posterior samples; the abstract claims a proof that the test is "necessary and
sufficient to show that a posterior estimator is accurate". Relevant to RIND3D as a remedy for the
weakness of plain rank/coverage tests (which an X-ignoring prior passes). Sufficiency is a
population statement (all reference-point distributions / infinite test data); finite-sample power is
a separate question.
Verified: fetched arXiv abstract (comment states accepted at ICML 2023).

## 4. Proper scoring rules and loss decompositions

**[GneitingRaftery2007] Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. JASA 102(477), 359-378. DOI 10.1198/016214506000001437.**
Defines (positively oriented) S(P, omega); S is proper if the expected score under the true Q is
maximised by reporting P = Q, strictly proper if uniquely so. Sec. 2.2 defines the entropy
G(Q) = sup_P S(P, Q) (Eq. 6) and divergence d(P, Q) = S(Q, Q) - S(P, Q) >= 0 (Eq. 7), strictly
positive for P != Q when S is strictly proper. Example 3: the logarithmic score S(p, i) = log p_i
has entropy = negative Shannon entropy and divergence = KL(q || p) = sum_j q_j log(q_j/p_j); Example 1
the Brier/quadratic score has squared-Euclidean divergence. Hence expected log loss under truth q
= H(q) + KL(q || p): an irreducible entropy term plus a divergence that is zero only at p = q.
Applies directly to RIND3D with q = P(S | X) (the Student's Bayes target): one-hot NLL and
cross-entropy to any unbiased soft target share the same minimiser and the same excess-risk KL;
the entropy floor H(S | X) is large when windows are ambiguous, so raw NLL values must be compared
against that floor. Since conditioning reduces entropy, E[H(q_G)] = H(S | X, G) <= H(S | X): the mean entropy of the privileged labels is only a LOWER bound on the Student's achievable expected NLL (see Sec. 5).
Verified: fetched UW-hosted PDF; Theorem 1, Eqs. (6)-(7), Examples 1 and 3 read directly (pp. 361-363).

**[Brier1950] Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. Monthly Weather Review 78(1), 1-3. DOI 10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2.**
Origin of the quadratic (Brier) score. Strictly proper but bounded; on a 128^3-cell output its
per-example value is dominated by the mass assigned to the true cell and is nearly insensitive to
how mass is spread over large feasible sets, which makes it a weak discriminator for diffuse
posteriors compared with the log score (our remark, not a claim of the paper).
Verified: search result (AMS journal page and ADS bibcode 1950MWRv...78....1B); attribution of the
quadratic score to Brier (1950) also confirmed in [GneitingRaftery2007] Example 1.

**[Good1952] Good, I. J. (1952). Rational decisions. JRSS-B 14(1), 107-114. DOI 10.1111/j.2517-6161.1952.tb00104.x.**
Introduces the logarithmic scoring rule for eliciting honest probabilities. It is the scoring rule
behind maximum-likelihood/NLL training. Applies directly; the log score is unbounded, so a Student
assigning (near-)zero mass to the true source cell incurs very large loss - relevant to RIND3D because
exact feasible sets have hard boundaries and a model trained on sharp q_G targets may put ~0 mass
outside F_G-like regions.
Verified: search result (Wiley/OUP records with DOI, pages); attribution of the log score to Good
(1952) confirmed in [GneitingRaftery2007] Example 3.

**[Bernardo1979] Bernardo, J. M. (1979). Expected information as expected utility. Annals of Statistics 7(3), 686-690. DOI 10.1214/aos/1176344689.**
Defines proper and local utility functions for reported densities (Defs. 2-3; "local" = depends only
on the density at the true value) and states that, if u is smooth enough for a calculus-of-variations
argument, "the logarithmic is the only proper, local utility function", u = A log p(psi) + B(psi).
Applies to RIND3D's choice of objective: if the score should depend only on the probability assigned
to the realised source cell, the log score is forced. (For finite outcome spaces the analogous
characterisation is usually stated for >= 3 outcomes; the paper's own argument is for densities.)
Verified: fetched author-hosted scanned PDF (uv.es); Sec. 3-4 read directly (pp. 687-688).

**[Dawid2007] Dawid, A. P. (2007). The geometry of proper scoring rules. Annals of the Institute of Statistical Mathematics 59(1), 77-93. DOI 10.1007/s10463-006-0099-8.**
Shows that any decision problem induces a proper scoring rule, an entropy, a divergence, a
Riemannian metric and an unbiased estimating equation (abstract). Supports viewing RIND3D objective
choices (NLL, Brier, Hamming-tempered likelihoods) as different decision problems with different
divergences; only the log score yields KL.
Verified: search result (Springer article page; IDEAS/RePEc record; abstract).

**[Brocker2009] Brocker, J. (2009). Reliability, sufficiency, and the decomposition of proper scores. Quarterly Journal of the Royal Meteorological Society 135(643), 1512-1519. DOI 10.1002/qj.456. arXiv 0806.0813.**
Shows that expected strictly proper scores decompose into reliability (calibration) and resolution
terms (plus uncertainty), first for binary and then for finite-valued targets. For RIND3D, it
justifies reporting calibration (reliability) separately from sharpness/resolution, e.g. a Student
that outputs the marginal source prior is perfectly reliable but has zero resolution.
Verified: search result (CentAUR repository record with DOI; arXiv 0806.0813; Wikidata).

**[KullFlach2015] Kull, M., & Flach, P. A. (2015). Novel decompositions of proper scoring rules for classification: score adjustment as precursor to calibration. ECML PKDD 2015, LNCS 9284, 68-85. DOI 10.1007/978-3-319-23528-8_5.**
Decomposes expected proper loss into adjustment loss, post-adjustment calibration loss, grouping
loss and irreducible loss (proofs for Brier and log loss). The "irreducible loss" plus "grouping
loss" split is the formal version of RIND3D's question of how much NLL is due to genuine window
ambiguity (irreducible given X) versus the model grouping together inputs with different posteriors.
Verified: search result (Springer chapter page, Bristol portal, dblp); abstract content via indexed
summaries only.

## 5. Soft / probabilistic labels, distillation, and privileged information

**[Menon2021] Menon, A. K., Rawat, A. S., Reddi, S. J., Kim, S., & Kumar, S. (2021). A statistical perspective on distillation. ICML 2021, PMLR 139:7632-7642. arXiv 2005.10419 (earlier title "Why distillation helps: a statistical perspective").**
KEY FACT (Lemma 1, Sec. 3.1, verbatim): with the Bayes class-probability p*(x) = [P(y|x)]_y, the
"Bayes-distilled risk" R*_hat(f; S) = (1/N) sum_n p*(x_n)^T l(f(x_n)) and the standard one-hot
empirical risk R_hat(f; S) are both unbiased for the population risk, and "For any fixed predictor
f : X -> R^L, V_{S~P^N}[R*_hat(f;S)] <= V_{S~P^N}[R_hat(f;S)], where equality holds iff for all x the
loss values l(f(x)) are constant on the support of p*(x)." Proof is Jensen on Z(x) = l(y, f(x)) with
i.i.d. sampling. Proposition 2 turns the lower empirical variance into a tighter uniform-convergence
bound (requires a bounded loss and finite uniform covering number; footnote: boundedness can be
enforced e.g. by regularisation). Sec. 4 (Prop. 3-4) gives a bias-variance trade-off when the teacher
only approximates p*. Their Fig. 2 shows the gain is largest at small N.
Applicability to RIND3D (our deduction, extra assumptions stated): q_G = P(S|X,G) is NOT the Bayes
teacher for the Student's input X; it is a finer conditional with E[q_G | X] = P(S|X) under the
data-generating joint. By the law of total variance, Var[l(S, f(X))] = Var[q_G^T l(f(X))] +
E[Var(l | X, G)], so for a fixed f the q_G-weighted risk is unbiased and has variance between the
Bayes-distilled and one-hot risks (Rao-Blackwellisation over S given (X, G)). It removes the
within-scene source-sampling noise but NOT the between-scene (G) variance, which with only 1,000
scenes is likely the dominant term. Lemma 1 is for a fixed predictor and i.i.d. examples; RIND3D's
~160 examples per scene are dependent, so the variance reduction applies to the scene-level sums,
not to an effective N of 160k. Unbiasedness also requires that labels and training pairs come from
the same (selected) joint - see Sec. 9/11.
Verified: fetched PMLR PDF; Lemma 1, its proof, Proposition 2 and Sec. 3.2 read directly (pp. 3-4);
venue/pages via PMLR search result.

**[Hinton2015] Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. arXiv 1503.02531 (NIPS 2014 Deep Learning Workshop).**
Introduces knowledge distillation: train a student on a teacher's temperature-softened class
probabilities, optionally mixed with the hard-label loss. The paper argues (paraphrase, not
re-checked verbatim) that high-entropy soft targets carry more information per case and give lower
gradient variance than hard targets, so students can be trained on less data. Applies to RIND3D as
the origin of the "soft target" objective; RIND3D's teacher is exact (q_G) rather than a learned
network, and temperature softening would change the target away from any posterior.
Verified: search result (arXiv abstract page, ADS entry with workshop note). Content paraphrase
UNVERIFIED in detail.

**[VapnikVashist2009] Vapnik, V., & Vashist, A. (2009). A new learning paradigm: Learning using privileged information. Neural Networks 22(5-6), 544-557. DOI 10.1016/j.neunet.2009.06.042.**
Introduces LUPI: extra ("privileged") information x* available for training examples only, not at
test time, used to improve learning of a decision rule on x (originally via SVM+, correcting slack
variables). This is exactly the RIND3D structure: the full geometry G (and hence q_G) is privileged
and unavailable to the Student at test time. The original theory concerns SVM rates (privileged info
as an "oracle" for the slacks), not posterior estimation, so only the paradigm transfers directly.
Verified: search result (ScienceDirect, PubMed, ACM DL records agree on DOI, volume, pages).

**[LopezPaz2016] Lopez-Paz, D., Bottou, L., Scholkopf, B., & Vapnik, V. (2016). Unifying distillation and privileged information. ICLR 2016. arXiv 1511.03643.**
"Generalized distillation": a teacher trained on privileged features x* produces soft labels, and a
student on regular features x is trained to imitate them (mixed with hard labels by an imitation
parameter). This is the closest formal template for RIND3D's "privileged posterior" target: the
teacher is f_t(x*) = q_G(.|X, G), computed exactly instead of learned. The paper's analysis is
about learning-rate improvement under assumptions on teacher quality; it does not address
posterior calibration of the student.
Verified: fetched arXiv abstract page (journal-ref: Proceedings of ICLR 2016).

**[Peterson2019] Peterson, J. C., Battleday, R. M., Griffiths, T. L., & Russakovsky, O. (2019). Human uncertainty makes classification more robust. ICCV 2019. DOI 10.1109/ICCV.2019.00971. arXiv 1908.07086.**
CIFAR-10H: full distributions of human labels for the CIFAR-10 test images; training on these soft
labels (instead of one-hot) improved generalisation to out-of-distribution test sets and adversarial
robustness (abstract). Empirical support for soft targets that represent genuine label ambiguity.
Applies to RIND3D only by analogy: human label distributions are a noisy estimate of P(y|x), whereas
q_G is an exact but privileged-conditioned distribution.
Verified: search result (arXiv, CVF open-access listing; DOI from Princeton/Semantic Scholar records).

## 6. Robust, tempered and generalized likelihoods; ABC kernels as model error

**[Wilkinson2013] Wilkinson, R. D. (2013). Approximate Bayesian computation (ABC) gives exact results under the assumption of model error. Statistical Applications in Genetics and Molecular Biology 12(2), 129-141. DOI 10.1515/sagmb-2013-0010. arXiv 0811.3355.**
KEY FACT (abstract, verbatim): "under the assumption of the existence of a uniform additive model
error term, ABC algorithms give exact results when sufficient summaries are used", and "The
acceptance density gives the distribution of the error term." So an ABC acceptance kernel K(d) is
exactly a likelihood under an explicit model-error/measurement-error model.
Application to RIND3D (our derivation, consistent with but not stated in the paper): for a window
with n FREE receivers, independent bit flips with probability eps on each receiver give
P(r_obs | s) = (1-eps)^n (eps/(1-eps))^{Hamming(r_obs, r_s)} proportional to exp(-beta * Hamming) with
beta = log((1-eps)/eps), since n is fixed given the window. Hence the soft response-matching
likelihood exp(-beta * Hamming) is the exact posterior of an i.i.d. bit-flip noise model (prior on
s unchanged), and beta -> inf recovers Uniform(F) whenever F is non-empty. In the noise-free RIND3D
world this is a deliberately misspecified model: it is a principled smoothing, not the true
posterior, and its use must be justified as robustness (e.g. to discretisation or certification
error), not as correctness.
Verified: fetched arXiv abstract page (journal-ref and DOI listed there).

**[Beaumont2002] Beaumont, M. A., Zhang, W., & Balding, D. J. (2002). Approximate Bayesian computation in population genetics. Genetics 162(4), 2025-2035. DOI 10.1093/genetics/162.4.2025.**
Introduces smooth kernel weighting of simulations by summary-statistic distance plus local-linear
regression adjustment in ABC, replacing hard rejection. It is the standard reference for "weighting
simulations by a kernel of the data distance", i.e. the operational form of exp(-beta d). For
RIND3D, it underlines that kernel-weighted posteriors are approximations to the epsilon -> 0 exact
posterior unless reinterpreted via [Wilkinson2013].
Verified: search result (OUP Genetics page, PMC, PubMed agree on DOI and pages). Method description
from general knowledge of the paper, not re-checked in full text.

**[Bissiri2016] Bissiri, P. G., Holmes, C. C., & Walker, S. G. (2016). A general framework for updating belief distributions. JRSS-B 78(5), 1103-1130. DOI 10.1111/rssb.12158. arXiv 1306.6430.**
General Bayesian updating: a prior can be updated to a posterior pi(theta|x) proportional to
exp(-w * loss(theta, x)) pi(theta) for parameters linked to data through a loss rather than a
likelihood; the standard likelihood posterior is the special case loss = negative log-likelihood,
w = 1, and coherence arguments single out this exponential form. Applies to RIND3D as the formal
licence for "Gibbs posteriors" such as exp(-beta * Hamming) when one does not believe any noise
model; the learning rate w (here beta) is not determined by the theory and must be calibrated.
Verified: search result (PubMed, Wiley, PMC, arXiv agree on citation). Exponential-loss form from
general knowledge of the paper; the PMC full text was not fetched.

**[GrunwaldVanOmmen2017] Grunwald, P., & van Ommen, T. (2017). Inconsistency of Bayesian inference for misspecified linear models, and a proposal for repairing it. Bayesian Analysis 12(4), 1069-1103. DOI 10.1214/17-BA1085. arXiv 1412.3730.**
Shows that standard Bayesian inference can be inconsistent under misspecification in simple linear
regression (model averaging/selection, Bayesian ridge) and proposes SafeBayes: a likelihood raised to
a learning rate eta learned from the data. Relevance to RIND3D: only if the Student or label
generator is used under misspecification (e.g. Hamming-tempered likelihood on noise-free data, or a
model family that cannot represent the posterior). In the well-specified noise-free case the
feasible-set posterior is correct and tempering is not needed for consistency.
Verified: search result (Project Euclid listing with DOI, volume, pages; arXiv).

**[MillerDunson2019] Miller, J. W., & Dunson, D. B. (2019). Robust Bayesian inference via coarsening. JASA 114(527), 1113-1125. DOI 10.1080/01621459.2018.1469995. arXiv 1506.06101.**
Coarsened posterior: condition on the event that model-generated data are close to the observed data
in a distributional discrepancy, instead of exactly equal. With relative entropy as the discrepancy
and an exponential prior on the tolerance, the c-posterior is approximated by a power posterior
(likelihood raised to a fractional power). Applies to RIND3D conceptually: "s reproduces r up to a
tolerance" is a coarsening; with a Hamming discrepancy one obtains exactly the response-matching
family, so the feasible-set label is the zero-tolerance limit. The power-posterior asymptotics assume
i.i.d. data and a relative-entropy discrepancy, which do not hold for one window's spatially
dependent receivers.
Verified: search result (T&F page, PubMed, arXiv, RePEc agree on citation; power-posterior link stated
in search snippets of the abstract).

## 7. Probabilistic inverse graphics, analysis-by-synthesis, and shadow-based inference

**[Mansinghka2013] Mansinghka, V. K., Kulkarni, T. D., Perov, Y. N., & Tenenbaum, J. B. (2013). Approximate Bayesian image interpretation using generative probabilistic graphics programs. NeurIPS 26, 1520-1528. arXiv 1307.0060.**
Writes short probabilistic graphics programs (scene prior + renderer + stochastic likelihood
comparing rendered and observed images) and inverts them with generic MCMC; demonstrated on degraded
text and 3D road models. Establishes the analysis-by-synthesis posterior framing. The likelihood is
an explicitly stochastic image-comparison model (a tolerance/noise model), which is the [Wilkinson2013]
device; with RIND3D's exact binary visibility the analogous posterior is the indicator-likelihood
feasible set, for which generic MCMC mixes poorly (hard constraints, disconnected F).
Verified: search result (arXiv 1307.0060, NeurIPS proceedings listing with pages). Mechanism summary
from abstract-level information.

**[Kulkarni2015] Kulkarni, T. D., Kohli, P., Tenenbaum, J. B., & Mansinghka, V. (2015). Picture: A probabilistic programming language for scene perception. CVPR 2015, 4390-4399. DOI 10.1109/CVPR.2015.7299068.**
Probabilistic programming language for generative analysis-by-synthesis vision (3D face, body pose,
object shape), combining a renderer with learned "discriminative" proposals to speed up inference.
The latter is an early instance of amortised/learned proposals for inverse graphics, conceptually
close to a Student network that proposes source locations from a local window. Applies as framing;
the paper's likelihoods are approximate (stochastic comparison in a feature space), not exact.
Verified: search result (CVF open-access page with pages; IEEE Xplore DOI).

**[KriegmanBelhumeur2001] Kriegman, D. J., & Belhumeur, P. N. (2001). What shadows reveal about object structure. JOSA A 18(8), 1804-1813. (Conference version: ECCV 1998, LNCS, 399-414, DOI 10.1007/BFb0054755.)**
For any finite set of point light sources, shapes producing identical shadows form an equivalence
class related by a four-parameter family of projective transformations; under orthographic
projection this is the generalized bas-relief (GBR) family, the only family leaving all imaged
shadows unchanged (abstract). This is a rigorous non-identifiability result for shadow-based
inference, the classical counterpart of RIND3D's feasible-set ambiguity. Differences: it concerns
recovering shape from images of shadows with unknown light directions, while RIND3D recovers a
point source from binary visibility at sampled 3D receivers with only local geometry known; the
lesson that shadows determine geometry/light only up to structured equivalence classes transfers.
Verified: search result (Optica abstract page; Springer ECCV chapter record; Semantic Scholar).
Exact JOSA A issue (8) taken from the Optica URL (josaa-18-8-1804).

**[Panagopoulos2009] Panagopoulos, A., Samaras, D., & Paragios, N. (2009). Robust shadow and illumination estimation using a mixture model. CVPR 2009, 651-658. DOI 10.1109/CVPR.2009.5206665.**
Estimates illumination (as a von Mises-Fisher mixture) and cast shadows jointly from a single image
with only coarse 3D geometry, using a graphical model. A follow-up, Panagopoulos, Wang, Samaras &
Paragios (2011), "Illumination estimation and cast shadow detection through a higher-order graphical
model", CVPR 2011, 673-680, DOI 10.1109/CVPR.2011.5995585, uses a higher-order MRF. These are
"light from shadows" inverse problems, but with known (coarse) global geometry and distant
illumination - the reverse of RIND3D's information pattern (global geometry hidden, point source).
Verified: search result (ML Anthology records with DOIs and pages; Stony Brook-hosted PDF listed).

**[Velten2012] Velten, A., Willwacher, T., Gupta, O., Veeraraghavan, A., Bawendi, M. G., & Raskar, R. (2012). Recovering three-dimensional shape around a corner using ultrafast time-of-flight imaging. Nature Communications 3, 745. DOI 10.1038/ncomms1747.**
Non-line-of-sight (NLOS) imaging: reconstructs hidden 3D shape from time-resolved light scattered
off a visible diffuse wall. Relevant only as a neighbouring field where hidden geometry is inferred
from indirect light transport; NLOS uses rich continuous time-of-flight measurements and calibrated
geometry, unlike RIND3D's binary, local, single-bounce-free visibility.
Verified: search result (Nature article page, ADS bibcode 2012NatCo...3..745V).

## 8. Multimodal inverse prediction: mixture density networks, invertible networks, flows

**[Bishop1994] Bishop, C. M. (1994). Mixture density networks. Technical Report NCRG/94/004, Neural Computing Research Group, Aston University.**
A neural network outputs the parameters of a mixture model, representing (in principle) arbitrary
conditional densities p(t|x), trained by maximum likelihood. Motivation (abstract): least-squares or
cross-entropy regression yields conditional averages, which describe multi-valued inverse problems
poorly. Applies directly: in RIND3D a regression to the source position would return the conditional
mean of a possibly multi-lobed feasible set, which can lie inside an obstacle or outside every lobe.
RIND3D's 128^3 categorical (softmax-over-cells) head is the nonparametric limit of this idea and,
with NLL, inherits the [PapamakariosMurray2016] consistency result. (Aston's repository also lists
the report as "NCRG No. 4288".)
Verified: search result (Aston Research Explorer and Aston-hosted PDF; Microsoft Research listing).

**[Ardizzone2019] Ardizzone, L., Kruse, J., Wirkert, S., Rahner, D., Pellegrini, E. W., Klessen, R. S., Maier-Hein, L., Rother, C., & Kothe, U. (2019). Analyzing inverse problems with invertible neural networks. ICLR 2019. arXiv 1808.04730.**
Invertible networks learn the (well-defined) forward process, with extra latent outputs capturing
information lost in the forward map; running the network backwards with sampled latents yields
posterior samples for ambiguous inverse problems, revealing multimodality, parameter correlations and
unrecoverable parameters (abstract). Relevant to RIND3D as an alternative Student head (sample-based
posterior over continuous s); however, it assumes a smooth continuous forward model and is trained on
the forward map, whereas RIND3D's forward map (binary visibility over a window) is discontinuous and
many-to-one with set-valued preimages, better matched by direct posterior estimation.
Verified: search result (arXiv record; v3 PDF header "Published as a conference paper at ICLR 2019").

**[Kruse2021] Kruse, J., Ardizzone, L., Rother, C., & Kothe, U. (2021). Benchmarking invertible architectures on inverse problems. arXiv 2101.10763.**
Compares ten invertible architectures and related models on two low-dimensional, intuitive
ambiguous inverse problems with known posteriors; coupling-layer flows and simple autoencoders did
best (abstract). Relevant as methodology: evaluate posterior estimators on tasks with exact reference
posteriors, which RIND3D has per scene. Results are for low-dimensional toy tasks and do not transfer
to a 128^3 output space. (An earlier version appears to be an ICML 2019 INNF workshop paper, judging
from the host filename "innf19kruse.pdf" - UNVERIFIED.)
Verified: search result (arXiv abstract page, dblp CoRR record, HCI Heidelberg page).

## 9. Importance weighting, covariate shift, and sample selection bias

**[Zadrozny2004] Zadrozny, B. (2004). Learning and evaluating classifiers under sample selection bias. ICML 2004. DOI 10.1145/1015330.1015425.**
KEY FACT (Sec. 2-3, read directly): with selection indicator s, four cases: (1) s independent of x
and y (no bias); (2) P(s|x,y) = P(s|x) - bias depends only on x; (3) P(s|x,y) = P(s|y) - a change in
label priors; (4) no independence - one cannot learn the feature-to-label map from the selected
sample unless one observes the selection-controlling variables x_s. In case (2) "by definition
P(y|x, s=1) = P(y|x)", so "local" learners (asymptotically depending only on P(y|x)) are unaffected,
while "global" learners (depending also on P(x)) are affected. Footnote: cases (1), (2), (4)
correspond to MCAR, MAR and NMAR in Little & Rubin. Sec. 4 corrects by estimating P(s=1|x) and
reweighting/rejection sampling. Applies DIRECTLY to RIND3D: if window acceptance depends only on
the Student-visible input X (solid mask, response, window position/size), selection is case (2)
and the target P(S|X) is unchanged - weights are not needed for the correct posterior, only for
the X-marginal (which inputs the finite-capacity model prioritises, and population-level metrics).
If acceptance depends on S, on G outside the window, or on anything not in X, it is case (3)/(4)
and the selected-sample conditional differs from P(S|X); the importance weights are then required
for consistency of one-hot NLL (and must be exactly known, which they are by construction here).
Verified: fetched KTH-hosted PDF; pp. 1-2 (abstract, Secs. 1-3, footnote 1) read directly; DOI via
ACM DL search result.

**[Shimodaira2000] Shimodaira, H. (2000). Improving predictive inference under covariate shift by weighting the log-likelihood function. Journal of Statistical Planning and Inference 90(2), 227-244. DOI 10.1016/S0378-3758(00)00115-4.**
Under covariate shift (q(x) != p(x), same p(y|x)), weighted MLE with weights w(x) = q(x)/p(x)
(target over training covariate density) is asymptotically optimal under model MISSPECIFICATION,
whereas under a correctly specified model the unweighted MLE is asymptotically optimal (search
abstract). Applies directly to RIND3D's case-(2) selection: a sufficiently flexible Student (posterior
family contains P(S|X)) gains nothing asymptotically from the weights and pays their variance cost;
weights help only to the extent the Student is misspecified/capacity-limited.
Verified: search result (ScienceDirect abstract snippets; DOI via Google Scholar/Semantic Scholar).

**[Sugiyama2007] Sugiyama, M., Krauledat, M., & Muller, K.-R. (2007). Covariate shift adaptation by importance weighted cross validation. JMLR 8, 985-1005.**
Standard cross-validation is biased under covariate shift; importance-weighted CV (IWCV) is proven
unbiased for the target-distribution risk (abstract). Applies to RIND3D model selection/early
stopping: validation losses computed on the selected window population should be importance-weighted
if the goal is the unselected (natural) window population; this is required even in case (2),
because the risk (not the Bayes posterior) depends on the X-marginal.
Verified: search result (JMLR v8 page, ACM DL record).

**[ByrdLipton2019] Byrd, J., & Lipton, Z. C. (2019). What is the effect of importance weighting in deep learning? ICML 2019, PMLR 97:872-881. arXiv 1812.03372.**
Empirical finding (abstract): importance weighting "impacts models early in training, its effect
diminishes over successive epochs"; L2 regularisation and batch normalisation (not dropout) restore
some of the effect. Directly relevant: for an over-parameterised Student trained long on separable or
near-memorisable data, weights may have little effect on the learned function. Caveat for RIND3D:
the effect fades in classification when training data become separable; with an irreducible
multimodal posterior (NLL cannot reach zero) the weighted and unweighted population optima differ
under case (3)/(4) selection, so the "diminishing effect" cannot be relied on to remove that bias -
it may instead mean the weights are silently ineffective. (Our analysis; the paper studies
classification and does not address soft/posterior targets.)
Verified: fetched arXiv abstract; venue/pages via PMLR search result.

**[Heckman1979] Heckman, J. J. (1979). Sample selection bias as a specification error. Econometrica 47(1), 153-161. DOI 10.2307/1912352.**
Treats selection bias as an omitted-variable problem and proposes a two-stage correction (model the
selection probability, then include the correction term). The underlying insight - estimate the
probability of selection and use it to correct the model - is the basis of RIND3D's inverse-
acceptance weights. Parametric (linear/normal) assumptions do not apply. (Some records give pages
153-162.)
Verified: search result (Econometric Society page, RePEc, Semantic Scholar DOI).

**[Kish1965] Kish, L. (1965). Survey Sampling. New York: Wiley.**
Source usually cited for the design effect due to unequal weights, deff = 1 + relvar(w), and the
effective sample size n_eff = (sum w)^2 / sum w^2. Applies to RIND3D to quantify how much the
importance weights reduce effective sample size; the formula ignores clustering (here: scenes) and
correlation between weights and outcomes, so it must be computed at the scene level as well as the
example level. Exact page of the formula in the book not verified.
Verified: search result (secondary sources attributing the formula and design effect to Kish 1965);
book itself not accessed - formula attribution PARTLY UNVERIFIED.

## 10. Grouped / clustered data and generalization across independent environments

**[Roberts2017] Roberts, D. R., Bahn, V., Ciuti, S., ..., & Dormann, C. F. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography 40(8), 913-929. DOI 10.1111/ecog.02881.**
Random cross-validation can badly underestimate predictive error for dependent (structured) data;
recommends blocked/grouped cross-validation, noting blocking can overstate error if it forces
extrapolation. Applies directly: RIND3D's ~160k (window, source) examples are nested in 1,000 scenes
(and sources); train/validation/test splits and bootstrap CIs must be at the scene level
(leave-scenes-out), and the number of independent units for uncertainty statements is ~1,000, not
~160k.
Verified: search result (Wiley DOI page, ADS, Hartig lab page).

**[CameronMiller2015] Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. Journal of Human Resources 50(2), 317-372. DOI 10.3368/jhr.50.2.317.**
Guide to inference when errors are independent across clusters but correlated within them,
including the few-clusters problem, multiway clustering and fixed effects (abstract). Applies to
RIND3D metric confidence intervals: cluster = scene (1,000 clusters), so cluster-robust or
scene-level bootstrap intervals are required; naive per-example standard errors over ~160k
correlated examples will be far too narrow.
Verified: search result (JHR publisher page, RePEc/EconPapers, eScholarship copy with DOI).

**[Cobbe2019] Cobbe, K., Klimov, O., Hesse, C., Kim, T., & Schulman, J. (2019). Quantifying generalization in reinforcement learning. ICML 2019, PMLR 97:1282-1289. arXiv 1812.02341.**
Using procedurally generated levels (CoinRun), agents overfit to surprisingly large numbers of
training levels; generalisation to unseen levels improves with the number of distinct training
levels, and a sizeable gap can remain even with many thousands of levels (abstract/blog). This is
direct evidence that the number of independent environments (here: levels, in RIND3D: scenes) can
be the binding data quantity, regardless of how many samples are drawn per environment. Domain is
RL, so the transfer to supervised posterior estimation is by analogy.
Verified: search result (PMLR v97 page, arXiv 1812.02341, OpenAI blog). Exact sweep sizes not checked.

## 11. Missing labels and label-dependent selection

**[Rubin1976] Rubin, D. B. (1976). Inference and missing data. Biometrika 63(3), 581-592. DOI 10.1093/biomet/63.3.581.**
Defines missing at random (MAR) and gives the weakest general conditions under which the
missingness process can be ignored: for likelihood/Bayesian inference, MAR plus distinctness of the
missingness-mechanism parameters from theta; for sampling-distribution inference, MAR plus "observed
at random" (search abstract summaries). Applies directly to RIND3D's dropped uncertified labels:
missingness R = 1[label certifies] is decided by the label engine, which sees (X, S, G). Relative to
the Student's observed variables X, any dependence of R on G beyond X is missing-NOT-at-random
(Zadrozny case 4), so ignoring dropped examples biases the learned P(S|X) in general.
Verified: search result (OUP Biometrika page with DOI; Harvard DASH and BibBase records).

**[LittleRubin2019] Little, R. J. A., & Rubin, D. B. (2019). Statistical Analysis with Missing Data, 3rd ed. Wiley. DOI 10.1002/9781119482260.**
Standard reference for MCAR/MAR/MNAR, likelihood-based inference with a model of the missingness
mechanism, and weighting/imputation corrections. [Zadrozny2004] footnote 1 maps its selection cases
(1), (2), (4) to MCAR, MAR, NMAR (citing the 2002 2nd edition). Applies as the vocabulary for
RIND3D's certification dropout and window thinning.
Verified: search result (Wiley Online Library book page with DOI, ISBNs, 12 April 2019).

Note for RIND3D (our deduction from [Rubin1976], [Zadrozny2004], [Menon2021]; not a published
result): the missingness is MNAR for the Student but MAR given the generator's variables (X, S, G),
which are fully observed for every generated example, certified or not. Two consequences:
(a) The one-hot pair (X, S) is available even when q_G fails to certify, so one-hot NLL never needs
to drop examples; dropping them introduces avoidable selection.
(b) A hybrid target y~ = R * q_G + (1 - R) * e_S satisfies E[y~ | X, G] = q_G whenever R is
conditionally independent of S given (X, G) (certification depends on the geometry/window, not on
which point in F the source is), hence E[y~ | X] = P(S | X): unbiased, with Rao-Blackwell variance
reduction on the certified fraction. Alternatively, inverse-probability weighting by
1/P(R = 1 | X, G) is consistent because R is observed for all generated examples, but it inflates
variance (see [Kish1965], [Greenberg2019]).

## Direct implications for RIND3D

- One-hot NLL is the right population objective: by [PapamakariosMurray2016] Prop. 1 (and
  [Greenberg2019] Sec. 2.2), the maximum-likelihood conditional density estimator trained on pairs
  from the target joint converges to P(S|X) for every X, given realisability, infinite data and a
  global optimum. Determinism and lack of noise do not break this: it needs only joint samples.
- The log score is forced if the metric should depend only on the mass at the true source cell
  ([Bernardo1979]); its expected value is H(S|X) + KL(P(S|X) || p_theta) ([GneitingRaftery2007]
  Eq. 7, Ex. 3). Report NLL against an entropy floor, and note E[H(q_G)] = H(S|X,G) <= H(S|X) is only
  a lower bound on the achievable NLL (needs no extra assumption).
- Cross-entropy to q_G (forward KL to the privileged posterior) has the same population minimiser
  as one-hot NLL provided E[q_G | X] = P(S|X) on the training distribution (true when labels and
  pairs are from the same joint, i.e. no G-dependent selection). This is an extra assumption that
  the selection/certification pipeline must be shown to satisfy.
- Variance reduction from q_G targets is real but limited: [Menon2021] Lemma 1 proves it for the
  Bayes teacher P(y|x) and a fixed predictor with i.i.d. data; for q_G it follows from the law of
  total variance (Rao-Blackwellisation over S given (X,G)), and it removes only within-scene source
  noise, not between-scene variance. With 1,000 scenes the latter likely dominates; this is an
  empirical question, not settled by the literature.
- Soft response-matching likelihoods exp(-beta * Hamming) are exact posteriors of an explicit
  i.i.d. bit-flip noise model ([Wilkinson2013] logic; derivation in Sec. 6), or Gibbs posteriors
  ([Bissiri2016]). In RIND3D's noise-free world they are misspecified by construction; they can be
  justified only as robustness/smoothing, with beta calibrated, never as "the" posterior.
- Tempering/SafeBayes/coarsening ([GrunwaldVanOmmen2017], [MillerDunson2019]) address
  misspecification; the noise-free feasible-set posterior is well specified, so these are not needed
  for correctness of labels, only possibly for the Student's capacity limits.
- Window acceptance that depends only on Student-visible X leaves P(S|X) unchanged ([Zadrozny2004]
  case 2): importance weights are then unnecessary for posterior correctness and asymptotically
  suboptimal for a well-specified model ([Shimodaira2000]); they matter for the X-marginal, i.e.
  for population-level metrics and model selection ([Sugiyama2007] IWCV).
- If acceptance depends on S, on G outside the window, or on any non-X variable, it is Zadrozny
  case 3/4: one-hot NLL on the selected data is biased and the exact inverse-acceptance weights are
  REQUIRED ([PapamakariosMurray2016] Eq. 3, SNPE-B [Lueckmann2017]). The audit should establish
  exactly which variables the thinning rule reads.
- Importance weights are high-variance ([Greenberg2019]) and, in over-parameterised deep nets, their
  effect can fade with training ([ByrdLipton2019]); in RIND3D that would mean case-3/4 bias is NOT
  removed in practice. Report Kish n_eff ([Kish1965]) at both example and scene level.
- Dropping uncertified labels is selection on privileged variables (MNAR for the Student,
  [Rubin1976]); it is correctable because (X, S, G) are known for every dropped example: use one-hot
  fallback for uncertified examples or IPW by the certification probability (Sec. 11 note).
- The label F excludes the window box, so the target is P(S | X, S not in W); this is S-dependent
  selection handled consistently by the label construction, but all metrics and SBC/coverage checks
  must use the same conditioning ([Talts2018] identity holds only for the joint actually sampled).
- Calibration diagnostics: SBC ranks and expected coverage ([Talts2018], [Hermans2022]) apply
  directly (they need only joint samples), but an X-ignoring prior passes them; complement with
  TARP ([Lemos2023]) and with direct comparison against q_G, which RIND3D uniquely has. Expect
  overconfidence as the default failure mode of neural posterior estimators ([Hermans2022]).
- Splits, confidence intervals and data-scaling claims must be at the scene level: ~1,000
  independent units, not ~160k ([Roberts2017], [CameronMiller2015]); evidence from procedurally
  generated environments ([Cobbe2019]) suggests the number of distinct scenes may be the binding
  data quantity (analogy, RL domain).
- Regression to a point estimate is inappropriate for multi-lobed feasible sets ([Bishop1994]);
  the categorical 128^3 head is a valid nonparametric posterior family. Shadow-based inference has
  provable structured non-identifiability ([KriegmanBelhumeur2001]), so irreducible posterior
  spread is expected, not a model failure.
- Statements in this file that are marked "our deduction/derivation" are logical consequences of
  the cited results under the stated assumptions; they are not claims made by the cited papers.

## Citations not verified / omitted

- Bouman et al. (2017) "Turning corners into cameras" (ICCV) - search did not return it; omitted.
- Hinton2015: bibliographic details verified; the paraphrased variance remark is UNVERIFIED in detail.
- Lueckmann2017: middle author names not displayed in search snippets (PARTLY UNVERIFIED).
- Kish1965: formula attribution from secondary sources only (PARTLY UNVERIFIED).
- Kruse2021: workshop venue UNVERIFIED (arXiv only confirmed).
- Content remarks for KaipioSomersalo2005, Tarantola2005, Beaumont2002, Bissiri2016 beyond their
  abstracts are from general knowledge and flagged as such in their Verified tags.
