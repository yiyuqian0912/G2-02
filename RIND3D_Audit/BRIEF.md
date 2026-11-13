# RIND3D — Comprehensive Scientific Foundations Audit and Research Direction Review (user brief, 2026-11-12)

## Mission

I want you to conduct a rigorous, independent scientific audit of the current RIND3D research program.

We have made substantial progress in physical scene generation, inverse geometry, tau-pencil posterior labeling, certified FastLabel, and large-scale dataset production.

However, several foundational questions have emerged concerning the probabilistic meaning of the labels, the validity of the sampling distribution, the student learning objective, uncertainty, and generalization.

**Before we continue expanding the dataset or designing the next Student model, I want these questions thoroughly investigated.**

This is not simply a code review.

It is a combined:

- Mathematical and probabilistic modeling audit.
- Data-generating-process audit.
- Dataset sampling and selection-bias audit.
- Posterior-label correctness audit.
- Machine-learning objective analysis.
- Identifiability and uncertainty analysis.
- Empirical validation study.
- Research literature review.
- Scientific design recommendation.

The ultimate question is:

**What is the most scientifically sound, informative, scalable, and learnable formulation of RIND3D, given our actual physics, datasets, labeling capabilities, and research goals?**

Do not assume that our existing choices are correct.

Do not assume that they are wrong either.

Challenge every important assumption and give evidence-based recommendations.

I want you to exercise independent research judgment, rather than merely implementing the ideas discussed below.

---

# 1. Existing Project and Research Context

The maintained project is associated with Git branch:

`yushu/3d`

Inspect the actual repository, its current Git state, dataset specifications, physical generator, posterior solver, FastLabel implementation, sampling policies, training adapters, and research documentation.

Relevant subsystems include:

- RIND3D physical geometry and scene generation.
- RIND3D Dataset1K production.
- Multi-source response generation and storage.
- Window proposal and selection.
- Source-excluded observations.
- Layer balancing.
- Rate-stream and combined training weights.
- Tau-pencil posterior labeling.
- Certified FastLabel.
- Student/Teacher research designs.
- Existing RIND/RIND3D neural learning experiments.

Important recent materials include:

`RIND3D_Dataset1K/docs/PRODUCTION_LOG_1K_v1.md`

`RIND3D_Posterior/experiments/fastlabel/REPORT.md`

`RIND3D_Posterior/experiments/fastlabel/CERTIFICATE_AUDIT.md`

Locate the actual corresponding source files and tests.

The physical 1K dataset has already been generated:

- 1,000 accepted scenes.
- 64 physical windows per scene.
- 64,000 physical windows.
- Approximately 159,802 informative source-window channels.
- 1–4 physical sources per scene.
- Nine geometry families.
- Dataset-level effective layer counts approximately balanced across 1–4.
- Source-specific response channels stored separately.
- Source-aligned posterior slots.
- Training weights exposed by the canonical loader and flattened training adapter.

The recent FastLabel research projects a full 1K posterior backfill near five hours on the laptop, but that performance estimate still requires broader held-out validation.

**Do not interfere with that ongoing work.**

This scientific audit is a separate research task. Do not launch a full backfill, regenerate the 1K dataset, or modify frozen scientific outputs.

---

# 2. First Establish the Actual Generative Model

Before deriving any probability formulas, reconstruct exactly how the current dataset is generated.

Do not rely on documentation alone.

Inspect the code and determine:

1. How geometry G is sampled.
2. How the physical source count K is selected.
3. How each source coordinate S_j is sampled.
4. Whether sources are uniform in the legal FREE domain.
5. Whether source positions are independent conditional on geometry.
6. Whether source spacing, clearance, collision, or other constraints induce dependence.
7. Whether source locations are ever resampled for window feasibility.
8. How local observation windows W are proposed.
9. Whether window proposals depend on source positions.
10. How source exclusion is enforced.
11. How mixed-response windows are accepted.
12. How effective response layers are counted.
13. How dataset-level layer balancing changes acceptance.
14. How candidate exhaustion and scene rejection operate.
15. How geometry family balancing is achieved.
16. How sample weights are computed.
17. What exactly the Student receives as its input.
18. What information remains privileged and unavailable to the Student.

Use the actual implementation as the source of truth.

Construct an explicit probabilistic generative model, ideally as a factorization or graphical model.

Define the appropriate random variables, including:

- G: global geometry.
- K: number of physical sources.
- S_1,...,S_K: source positions.
- W: selected local observation window.
- O_j: source-specific local response.
- A: selection/acceptance event.
- M: effective layer count.
- X: actual Student-visible input.
- Ω: training/sample weight.
- Any additional latent variables used by the generator.

Clearly distinguish the original proposal distribution, accepted dataset distribution, training distribution, and intended evaluation distribution.

**Do not begin with an assumed uniform posterior and work backward to justify it.**

Derive what the actual code implies.

---

# 3. Is Uniform(F) Really the Correct Posterior?

The current tau-pencil posterior is based on the feasible source set:

F(G,W,O) = {s: s is legal, outside the closed window, and reproduces every observed FREE receiver response}.

The current reference formulation is approximately:

q(s | G,W,O) = Uniform(F).

We need to determine when this is mathematically correct.

## 3.1 Derive from Bayes' rule

Start with the actual source prior:

π(s | G, ...)

and deterministic physical likelihood:

P(O | s,G,W).

Derive the exact posterior under the declared generative process.

Identify the assumptions under which:

q(s | G,W,O) ∝ 1[s ∈ F]

is valid.

Do not assume that merely having a uniform initial source sampler makes Uniform(F) correct after all selection events.

## 3.2 Uniform conditional prior versus uniform global source distribution

Distinguish:

- Uniform within each scene's legal FREE domain.
- Uniform over the entire unit cube.
- Uniform conditional on excluding the observation window.
- Uniform among sources in accepted scenes.
- Uniform over effective source-window training examples.
- Uniform under training weights.

These distributions need not be equal.

Determine which one is intended and which one is actually implemented.

In particular, explain why aggregating scene-conditionally uniform sources across different geometries does not necessarily produce a globally uniform spatial histogram.

## 3.3 Effect of source-dependent selection

Analyze whether selecting W based on source-dependent information modifies the correct conditional source posterior.

Derive the relevant selection likelihood rather than using an informal explanation.

For example, determine the correct conditioning under:

P(S | G,W,O,A)

and whether the factor:

P(A | S,G,W,O)

can be ignored.

Be precise about when selection is ignorable because it is a deterministic function of already conditioned variables, and when it is non-ignorable.

Investigate any dependence of W itself on S, including source exclusion and source-conditioned window proposals.

Do not double-count or incorrectly omit selection factors.

## 3.4 Multi-source complications

Our canonical scenes contain 1–4 physical sources, but posterior labels are single-source and stored per channel.

Investigate whether the posterior of S_j can legitimately be represented as:

q_j(s | G,W,O_j)

without conditioning on other source channels, K, or source-selection constraints.

Consider:

- Source-source spacing.
- Source-count sampling.
- Window selection using the effective layer count.
- Other source responses influencing acceptance.
- Hidden information in the selected window.
- Dependence induced by global quotas.
- Whether other sources are observed or hidden at Student training time.

Independent physical response generation does not automatically imply posterior independence after selection.

Determine the correct conditional posterior for the actual single-source Student learning task.

## 3.5 Training weights

Audit the rate-stream and combined estimators.

Determine:

- Which target distribution each estimator represents.
- Whether the weights are mathematically correct.
- Whether weights are nonnegative and finite.
- Whether there are zero-weight or zero-support failures.
- Whether effective sample size becomes unacceptably small.
- Whether the weights correct marginal training frequencies or conditional posterior bias.
- Whether the weighting method remains valid under adaptive layer balancing.

Do not treat sample reweighting and posterior correction as interchangeable.

Produce a definitive conclusion about whether the current Uniform(F) labels and training weights are mutually consistent.

---

# 4. One-Hot Labels Versus FastLabel Probability Fields

We previously discussed an important theoretical claim:

Under a correct generative model, One-hot source supervision and complete-geometry posterior supervision can have the same Bayes-optimal Student predictor.

Verify this rigorously.

Let:

X = Student-visible observation.

S = actual source position.

G = privileged global geometry.

q_G(s) = P(S=s | X,G).

pθ(s | X) = Student prediction.

Compare two objectives:

L_onehot = E[-log pθ(S | X)]

L_posterior = E[CE(q_G, pθ)].

## Required analysis

Prove under which assumptions these objectives share the same population optimum:

p*(s | X) = P(S=s | X).

Investigate:

- Correct conditional source priors.
- Source-window selection.
- Weighted training distributions.
- Multi-source dependencies.
- Source-excluded domains.
- Finite source-space discretization.
- Approximate posterior labels.
- Missing or uncertified labels.
- Student model misspecification.

Show precisely when the equivalence fails.

For weighted objectives, derive the correct weighted conditional optimum.

Do not assume the unweighted conditional expectation remains correct when weights depend on privileged variables.

## 4.1 What is the value of FastLabel if the Bayes optimum is identical?

This question deserves a serious analysis.

Investigate whether FastLabel provides:

- Rao–Blackwellized supervision.
- Reduced gradient variance.
- More informative finite-data learning signals.
- Better coverage of multimodal source distributions.
- Improved calibration.
- Faster convergence.
- Better generalization with limited independent scenes.
- Reduced dependence on individual sampled source positions.

Also evaluate its costs and potential disadvantages.

Determine whether these benefits hold under the actual sampling and label definitions.

If possible, propose a controlled experiment comparing:

A. One-hot source supervision.

B. FastLabel posterior-field supervision.

C. A hybrid objective.

Use the same Student architecture, scenes, computational budget and evaluation distribution.

Avoid claiming benefits based only on theoretical intuition.

---

# 5. Is Hard Geometric Posterior the Best Target?

An important conceptual issue emerged from our discussion.

The current exact geometry posterior assigns zero mass to every source position that fails to reproduce even one observed FREE receiver.

This is physically justified under a deterministic noise-free observation model.

However, a soft response-matching loss can treat near-matching candidate sources as useful approximations.

Consider two local responses:

Case A: almost entirely illuminated, with only a few shadowed receivers.

Case B: a large structured shadow region.

A completely illuminated candidate response may be close to Case A under normalized Hamming distance but very different from Case B.

The exact geometric posterior excludes that candidate in both cases.

We do not want a superficial answer that says one approach is always superior.

## 5.1 Analyze the scientific trade-off

Compare:

### Hard posterior

q_hard(s | G,W,O) ∝ π(s) 1[R_G(s)|W = O].

### Soft response likelihood

q_soft(s | G,W,O) ∝ π(s) exp(-β d(R_G(s)|W,O)).

### Hybrid training

L = L_hard + λ L_soft.

Investigate the validity, consequences, and learning behavior of each formulation.

## 5.2 Potential advantage of hard constraints

Small shadow regions can contain crucial geometric information.

Even a tiny patch of shadow may eliminate large portions of the candidate source domain.

A soft normalized Hamming loss may incorrectly favor candidate sources that explain most illuminated receivers while ignoring a small but decisive shadow.

Evaluate when hard constraints are scientifically better.

## 5.3 Potential advantage of soft constraints

Hard feasibility can be sensitive to:

- Receiver discretization.
- Geometric boundary perturbations.
- Small source-coordinate changes.
- Measurement noise.
- Finite sensor resolution.
- Near-grazing rays.

Study whether soft objectives improve robustness or merely blur physically important distinctions.

## 5.4 What uncertainty are we modeling?

Distinguish:

A. Uncertainty about global geometry unavailable to the Student.

B. Observation noise and physical measurement errors.

C. Numerical approximation error in the posterior labeler.

D. Finite-data epistemic/model uncertainty.

These are different phenomena.

Do not introduce a soft observation likelihood merely to compensate for missing global geometry without examining whether that is conceptually appropriate.

## 5.5 Literature review

Review relevant work in:

- Bayesian inverse problems.
- Probabilistic inverse graphics.
- Simulation-based inference.
- Proper scoring rules.
- Learning from probabilistic labels.
- Robust likelihoods and observation noise.
- Multimodal inverse prediction.
- Amortized posterior inference.

Use credible primary sources where possible.

Identify which findings directly apply to our deterministic binary visibility problem and which require additional assumptions.

Provide a reasoned recommendation for RIND3D rather than simply listing techniques.

---

# 6. Student Has Incomplete Information: What Should It Learn?

This is one of the central theoretical questions.

Our Student does not observe the full global geometry.

It receives only the intended local geometry and source-specific local response, together with any explicitly authorized input metadata.

Its objective is a global source-position probability field.

The target should be analyzed as:

P(S | X)

rather than simply expecting the Student to reproduce:

P(S | X,G)

for every individual world.

## 6.1 Derive the optimal predictor

Under forward KL or cross-entropy supervision, investigate:

p*(s | X) = E[q_G(s) | X].

Derive this result and its assumptions.

Determine whether the actual dataset sampling and weighting preserve this interpretation.

## 6.2 Irreducible ambiguity

Analyze the decomposition:

E_G|X KL(q_G || p)

= E_G|X KL(q_G || q_X)
  + KL(q_X || p)

where q_X = E[q_G | X].

Explain:

- Which component is irreducible because global geometry is hidden.
- Which component the Student can reduce.
- When the irreducible term corresponds to conditional mutual information.
- Why an optimal Student may still have nonzero training loss.
- Why a nonzero loss does not necessarily indicate model failure.

## 6.3 Overfitting risk

The existing 1K dataset has many source-window labels but only 1,000 independent physical scenes.

Assess the implications of correlated examples sharing:

- Global geometry.
- Physical sources.
- Local windows.
- Geometry families.
- Scene-generation seeds.

Investigate whether Student could memorize scene-specific shortcuts.

Study whether 100K independent scenes with fewer windows per scene would improve generalization more than adding many windows to a smaller number of worlds.

## 6.4 Probability calibration

Recommend evaluation methods that test whether predicted source probabilities correspond to actual source frequencies.

Consider:

- Proper scoring rules.
- Held-out source NLL.
- Forward KL against FastLabel.
- Brier score or appropriately adapted regional calibration.
- Spatial probability calibration.
- Entropy and effective support size.
- Multimodal coverage.
- Confidence versus observed success frequency.
- Calibration by geometry family and response type.

Do not use entropy alone as a measure of correctness.

A highly uncertain prediction may be appropriate or inappropriate depending on the observation.

## 6.5 Counterfactual ambiguity experiments

Design experiments where:

- Local geometry remains identical or sufficiently controlled.
- Local binary observations are identical.
- Hidden distant geometry changes.
- The corresponding complete-world posterior changes.

Use these to test whether Student appropriately represents ambiguity.

Ideally, construct families of compatible worlds and estimate their conditional posterior mixture.

Be careful: arbitrarily constructing compatible worlds does not automatically sample from the true conditional distribution G | X.

Distinguish controlled diagnostic counterfactuals from a statistically representative conditional posterior experiment.

---

# 7. All-Lit, All-Dark, and Solid-Separated Responses

Our existing informative-window selection emphasizes observations containing both illuminated and shadowed FREE receivers.

However, a general inverse-geometry model must be able to handle other physically valid observations.

Investigate three response categories.

## 7.1 Mixed response

Contains lit and dark FREE receivers.

This is the current primary training category.

## 7.2 All-lit response

Every FREE receiver is illuminated.

This does NOT necessarily imply a uniform global posterior.

It only implies that every observed FREE receiver is visible from the candidate source.

In a completely unobstructed world, the posterior may equal the prior over legal, source-excluded positions.

In the presence of obstacles, the observation can still exclude substantial source-space regions.

## 7.3 All-dark response

Every FREE receiver is shadowed.

This also does NOT necessarily imply a uniform global posterior.

Depending on geometry, the valid source region can be highly structured, disconnected, or concentrated on particular sides of occluders.

Within the existing binary visibility model, distance alone does not make a receiver dark.

However, after marginalizing over unobserved random global geometry, the Student may learn distance-dependent visibility probabilities.

Determine what is implied by the actual world distribution rather than assuming a universal exponential distance law.

## 7.4 Lit–SOLID–dark separation

Consider the following schematic local observation:

    1 1 1
    # # #
    0 0 0

Here `#` denotes SOLID.

The illuminated and shadowed FREE regions have no directly adjacent lit/dark FREE boundary.

This response may still be physically meaningful and informative.

Audit whether the generator, receiver selection, FastLabel boundary logic, exact point verifier, and posterior representation correctly handle this case.

In three dimensions, define appropriate connectivity and adjacency diagnostics rather than making conclusions from a 2D slice alone.

## 7.5 Dataset inclusion policy

Evaluate whether all-lit/all-dark observations should:

A. Remain excluded from the primary dataset.

B. Be included as separately labeled supplementary examples.

C. Be incorporated directly into the main sampling distribution.

D. Be included with explicit response-type balancing and importance weights.

Compare scientific validity, learning value, computational cost, distribution shift and certification feasibility.

Do not automatically force equal class counts.

Find a principled policy consistent with the target generative distribution.

Do not alter the existing 1K release contract during this audit.

---

# 8. What Exactly Does the TV Certificate Guarantee?

The current tau-pencil/FastLabel development has produced both exact geometric bounds and statistical certificates.

Audit their actual meaning.

Distinguish:

1. Deterministic TV approximation bounds.
2. Statistical confidence that a TV bound holds.
3. Numerical correctness of the geometry engine.
4. Correctness of the underlying posterior target.
5. Student approximation and generalization error.

These must not be conflated.

## 8.1 Receiver-subset approximation

For full feasible source set F and relaxed set B with F ⊆ B, verify under the same prior:

TV(q_B,q_F) = 1 - π(F)/π(B).

Determine the precise conditions for this identity.

## 8.2 Statistical certificate

Inspect the actual FastLabel implementation.

Verify:

- Correct proposal distribution.
- Independence or dependence assumptions.
- Random seed handling.
- Adaptive stopping.
- Multiple certification attempts.
- Confidence-budget allocation.
- Handling of rare feasible regions.
- The full-response verification oracle.

The latest report claims a dataset-wide certification failure probability at most 1% for the 1K workload.

Verify the code and mathematical argument.

Do not assume that 99% per-label reliability implies 99% simultaneous reliability.

## 8.3 TV target versus posterior correctness

A statistically valid certificate showing:

TV(q_approx,q_reference) ≤ 0.015

is useful only if q_reference is actually the correct target under the stated generative model.

If source priors or selection mechanisms are inconsistent, an extremely accurate approximation to Uniform(F) may still be an inaccurate approximation to the desired conditional posterior.

This distinction is crucial.

Quantify or bound the difference wherever feasible.

## 8.4 Effect on Student learning

Investigate how errors in posterior labels affect the population optimum of forward-KL or cross-entropy training.

Under what conditions does a per-label TV ≤ ε guarantee imply that conditional averaging preserves TV ≤ ε?

How do importance weights affect this statement?

What can and cannot be concluded about the trained Student's actual TV error?

Do not claim that the Student automatically inherits the labeler's TV certificate.

---

# 9. Empirical Audits on Existing Data

Theoretical analysis should be supported by focused empirical tests.

Do not regenerate 1,000 scenes.

Use existing physical data, known reference labels and small reproducible synthetic examples.

## 9.1 Source spatial prior audit

Measure and compare:

- Declared source proposal prior.
- Realized physical source distribution.
- Sources represented by accepted windows.
- Sources represented by informative channels.
- Source distribution after training weights.

Analyze these by:

- Geometry family.
- Number of physical sources.
- Window size.
- Effective layer count.
- Source coordinate region.
- Source exclusion constraints.

Use distributions relative to legal FREE-space volume and the theoretical prior, not just raw uniform grid counts.

Provide statistical uncertainty and effective sample sizes.

A roughly flat marginal spatial histogram is neither necessary nor sufficient for correctness.

## 9.2 Posterior consistency audit

Where the correct generative posterior can be derived or evaluated, compare it to the current tau-pencil Uniform(F) reference.

Investigate situations where selection depends on:

- Source position.
- Window location.
- Informative response.
- Other physical sources.
- Dataset balancing.

Construct minimal synthetic counterexamples if necessary.

## 9.3 One-hot versus posterior supervision

For selected source-space regions A, compare appropriately weighted estimates of:

E[1(S ∈ A)]

and

E[q_G(A)]

under a common well-defined target distribution.

If they differ systematically beyond sampling uncertainty, investigate possible prior/selection/posterior mismatch.

Do not use raw one-hot-versus-label disagreement on individual examples as evidence of a problem.

## 9.4 Response-type tests

Create or reuse compact exact physics fixtures for:

- Mixed observations.
- All-lit observations.
- All-dark observations.
- Solid-separated lit/dark observations.
- Thin obstacles.
- Narrow openings.
- Grazing rays.
- Broad feasible regions.
- Highly constrained feasible regions.
- Disconnected posterior support.

Check whether the physical generator and inverse solver agree.

## 9.5 Training distribution and generalization

Where existing Student experiments are available, audit their:

- Loss definitions.
- Input features.
- Privileged information leakage.
- Scene-level train/test separation.
- Calibration.
- Distribution shift.
- Sampling weights.

Do not launch a large new Student training project during this phase.

If meaningful model experiments are necessary, design inexpensive controlled pilots and clearly separate measured results from theoretical predictions.

---

# 10. Compare Candidate Scientific Designs

I want a serious recommendation among several possible research formulations.

At minimum, compare:

## Design A — Current hard geometric posterior

Uniform legal source prior, deterministic full-response likelihood, source-excluded feasible region, forward-KL Student training.

## Design B — Corrected hard posterior

Same deterministic physics, but with any source-prior or selection-law corrections required by the actual dataset.

## Design C — Explicit noisy-observation posterior

A well-defined nonzero observation-noise model, giving soft probabilities to near-matching source positions.

## Design D — Hard posterior plus auxiliary soft-response objective

Preserve strict posterior supervision while encouraging useful geometric similarity or robustness.

## Design E — One-hot source supervision

Use only the sampled physical source location, with a proper probabilistic training objective.

## Design F — Hybrid or curriculum training

For example, distributional FastLabel supervision with optional auxiliary objectives, or different response types assigned appropriate training policies.

You may propose additional designs if theoretically justified.

Assess each against:

- Scientific clarity.
- Correct probabilistic interpretation.
- Identifiability.
- Use of geometric information.
- Multimodal expressiveness.
- Robustness to local perturbations.
- Calibration.
- Dataset bias.
- Student generalization.
- Computational cost.
- Feasibility with current labels.
- Scalability to 100K scenes.
- Research novelty and publishability.

Recommend the most defensible **primary scientific formulation**, plus a small number of useful ablations.

Do not automatically recommend a hybrid loss simply because it combines more components.

A simpler, better-founded objective is preferable to an unnecessarily complicated system.

---

# 11. Recommendations for Future RIND3D-100K

We are separately investigating a future dataset with:

- 100,000 independent scenes.
- 1024³ physical resolution.
- Approximately 12 physical windows per scene.
- Four windows each of side 64, 128 and 256.
- 1–4 sources per scene.
- Approximately balanced effective layer counts 1–4.
- Canonical source-separated responses and posterior labels.
- Local-only response generation to avoid full global rendering.
- Multi-GPU production on a 16×H20 server.

This larger dataset has not been scientifically frozen yet.

Use the audit findings to recommend whether its sampling contract should change.

In particular:

- Should source positions be uniform in legal FREE space?
- Should scene source counts remain adaptive?
- Should layer balancing remain approximately 25% per category?
- Should sampling weights target natural or deliberately balanced distributions?
- Should all-lit/all-dark examples be included?
- How many independent scenes versus windows per scene are desirable?
- Which posterior quantities must be stored?
- What metadata is required to reproduce training weights?
- How should train/test distribution shifts be evaluated?

Distinguish necessary scientific corrections from optional design improvements.

Do not modify or restart the separate 100K production branch without explicit authorization.

---

# 12. Research Methodology and Quality Bar

I do not want an answer based on intuition alone.

For every important claim, classify its evidence:

**A. Mathematically proved**

Provide assumptions and derivation.

**B. Verified from current code**

Provide exact files, functions and relevant implementation behavior.

**C. Empirically measured**

Provide experiment configuration, sample counts, outputs and uncertainty.

**D. Supported by external literature**

Provide accurate citations and explain how the findings apply.

**E. Hypothesis or recommendation**

Explain what remains unverified and how to test it.

Be especially critical of the following tempting but potentially invalid claims:

- Uniform source proposals imply Uniform(F) labels after arbitrary selection.
- Balanced response layers imply balanced source positions.
- One-hot and FastLabel always have the same Bayes optimum.
- A training weight can correct any posterior-selection mismatch.
- All-lit implies no source information.
- All-dark implies a distance-based source distribution.
- Every physically feasible source is equally likely under any generation scheme.
- A small percentage of shadow pixels means little information.
- Low posterior TV guarantees low Student generalization error.
- A large number of windows means a large number of independent training examples.
- Lower training loss necessarily means improved geometric reasoning.
- Soft response matching is always more robust.
- Hard posterior supervision necessarily forces overfitting.
- Multiple sources can always be treated as independent after window selection.

Construct counterexamples where useful.

Search relevant scientific literature selectively and use it to sharpen the analysis, not to substitute for reasoning about our actual system.

---

# 13. Deliverables

Create a research directory under the maintained project, with appropriately named documentation and lightweight reproducible experiments.

Produce:

1. **Scientific Model Specification** — a precise mathematical description of the actual RIND3D generative model and intended Student learning problem. Define all conditioning variables, priors, likelihoods, selection events and target distributions.
2. **Source-Prior and Selection-Bias Audit** — a code-grounded analysis of the existing 1K generator, source sampling, source exclusion, adaptive layer balancing, rate-stream and combined weights. Identify correctness issues and quantify them where possible.
3. **Posterior Validity Report** — determine whether Uniform(F) is the correct posterior for the intended task, the implemented dataset, and the flattened and weighted Student training distribution. Explain any discrepancies.
4. **Learning-Theory Report** — one-hot versus posterior-field supervision, Bayes-optimal prediction, irreducible uncertainty, conditional averaging, calibration and overfitting. Include formal derivations and necessary assumptions.
5. **Hard-vs-Soft Posterior Study** — exact geometric constraints, soft response matching, observation-noise models and candidate training objectives. Provide a principled primary recommendation.
6. **Extreme-Response Study** — all-lit, all-dark and solid-separated response cases, with minimal exact test fixtures where feasible. Recommend whether and how to include them in future datasets.
7. **TV Certification Interpretation** — exactly what the current certificate guarantees, what it does not guarantee, and which error sources remain outside it.
8. **Experimental Evidence** — lightweight reproducible scripts, numerical results and diagnostic figures where useful. Do not fabricate quantitative results where experiments were not run.
9. **Recommended RIND3D-vNext Scientific Contract** — MUST FIX (demonstrated mathematical or implementation inconsistencies), SHOULD CHANGE (strongly supported design improvements), OPTIONAL (ablations and further research), KEEP (existing choices supported by the audit).
10. **Executive Research Decision** — a concise final recommendation answering:
    * What exactly should the Student predict?
    * Which probability distribution should the dataset labels represent?
    * Which loss should be the primary training objective?
    * What source prior and selection policy should we use?
    * How should all-lit and all-dark responses be treated?
    * What should we do to minimize overfitting and encourage genuine geometric reasoning?
    * Which existing parts of RIND3D are scientifically sound, and which need correction?
    * Can we proceed with the current 1K dataset, or is some correction required before Student training?
    * What changes should be made before committing to the 100K dataset?

---

# 14. Execution and Safety Constraints

This is an audit-and-research task first.

* Do not rewrite the physical generator, posterior solver, or canonical dataset format immediately.
* Do not alter the already generated 1K dataset.
* Do not replace certified labels.
* Do not start full posterior backfill.
* Do not launch the 100K production campaign.
* Do not perform large training runs or expensive GPU experiments without a clear research need.
* Read the existing work first. Prioritize mathematical clarity and inexpensive counterexamples before expensive computation.
* If you find a serious correctness issue, document it with a minimal reproducible case and a proposed fix.
* Implement only isolated diagnostic tests or clearly safe fixes that do not change the frozen scientific contract.
* Commit the research reports, validation scripts and reproducible evidence to the appropriate maintained Git branch, without interfering with parallel development.
* Do not confuse a passing software regression test with a proof that the target posterior is scientifically correct.
* Preserve negative results and unresolved findings.

# Final Instruction

**I do not want you to defend our current design by default.**

I want you to determine, through mathematical reasoning, source-code inspection, controlled experiments and relevant scientific research, which formulation is actually best for RIND3D.

Our central research goal is to train a Student that sees incomplete local physical information and predicts a meaningful global probability field over unknown source positions.

The Student should learn genuine geometric constraints, correctly express uncertainty caused by missing global information, generalize across unseen worlds, and avoid relying on accidental source-position or dataset-selection biases.

We now possess a powerful certified geometric posterior solver, but that capability alone does not settle the probabilistic learning problem.

**Find the best scientifically justified way to use it.**

If the existing Uniform(F) posterior and forward-KL loss are correct, demonstrate why.

If they are incomplete or biased, derive the correct target and provide evidence.

If one-hot, hard posterior, soft posterior, or a hybrid approach is preferable for certain objectives, explain precisely under what conditions and why.

The goal is a coherent, rigorous, learnable scientific system—not merely a pipeline that generates large quantities of labels.

Proceed autonomously with the audit, focused research experiments and documentation. Prioritize a defensible scientific conclusion over implementation volume.
