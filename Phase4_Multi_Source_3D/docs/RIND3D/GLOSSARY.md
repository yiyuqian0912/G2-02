# RIND3D glossary: terminology and information boundaries

**The task in general terms.** RIND3D studies inverse inference of a hidden *driver source* from one local window of
*binary responses* produced by occlusion.
* A world holds a continuous solid and 1–4 driver sources.
* Each driver source gives every FREE receiver a binary response.
* A Student sees the sampled geometry inside one window and the responses of one source there. It must infer where in the
  whole world that source can be.

**What the general vocabulary claims, and what it does not.**
* The vocabulary names the abstract inference problem. All current (contract v3) documentation uses it.
* One response mechanism is implemented: closed-segment visibility (`closed-solid-closed-segment-v1`,
  [SEMANTICS.md](SEMANTICS.md)). A receiver is response-active iff the closed segment from the driver source to the
  receiver misses the closed solid. No other response mechanism is implemented or validated.
* The graphics terms (light, lit, dark, shadow) name the same objects. They remain in stored data, code, file names,
  frozen contracts, release notes and dated research records (§4).
* The Chinese design document uses the same vocabulary: 驱动源, 二值响应, 有响应 / 无响应, 响应交界面
  (`../RIND3D_数据集设计与技术说明.md` §9).

## 1. Terms

| term | definition | legacy names (still used where §4 applies) |
|---|---|---|
| **driver source** (short: *source*) | a point s of the legal domain D(G): inside the open cube (0,1)³, outside the closed solid, with certified clearance > 2⁻²⁶. A world has K ∈ {1..4} of them, drawn i.i.d. uniform on D(G), with unit intensity | light, light source, point light |
| **receiver** | a voxel centre x = (i + ½)/N, N = 1024. It is SOLID iff it lies in the closed solid, otherwise FREE | — |
| **binary response** | R_s(x) ∈ {0, 1} at a FREE receiver x: 1 iff the closed segment [s, x] misses the closed solid. Contact and tangency block | illumination; *visibility* stays the name of the implemented mechanism |
| **response-active** receiver | a FREE receiver with R_s(x) = 1 | lit (`lit` bit 1, `cls` 1) |
| **response-inactive** receiver | a FREE receiver with R_s(x) = 0 | dark, in shadow (`cls` 0) |
| SOLID receiver | has no response. It is never response-active (its `lit` bit is 0) and is not counted as response-inactive | "SOLID receivers unlit" (`cls` 2) |
| **inactive set**, **occlusion set** | In(s) = {y : [s, y] meets the solid}. For one convex body C and a receiver x, Occ(x, C) = {s : [s, x] ∩ C ≠ ∅} | shadow, shadow volume |
| **response interface** | the boundary between the response-active and the response-inactive region of one source in FREE space | shadow boundary, lit/dark boundary |
| **direct (FREE–FREE) response interface** | two 6-adjacent FREE receivers with different responses, called an *interface edge*. This is the eligibility primitive of contract v3 (`RIND3D_Dataset100K/rind3d_100k/interface.py`) | 6-adjacent lit/dark pair, lit/dark contact face |
| **interface plane** | a plane through the driver source and an occluder edge or face that carries part of the response interface (the facet law, [LOCAL_INVERSE_GEOMETRY.md](../LOCAL_INVERSE_GEOMETRY.md) Theorem 1) | shadow plane |
| **source-specific response channel** (short: *channel*) | the binary responses of one driver source on the receivers of one window. Each window stores one channel per physical source; channels are never superposed | per-light channel, shadow channel, lit channel (`lit` crops, `responses[j]`) |
| channel types | *all-active*: every FREE receiver is active. *All-inactive*: every FREE receiver is inactive. *Mixed*: both occur. *SOLID-separated*: mixed, but with no direct interface. *Direct-interface*: has at least one interface edge | all-lit, all-dark; `mixed`, `mixed_mask` |
| **informative channel**; m(W) | contracts v1/v2: a mixed channel. Contract v3: a channel with a direct response interface. m(W) is the number of informative channels of window W | effective layer |

## 2. Six kinds of information

These must not be confused.

| | kind | what it is | where it is stored | who may use it |
|---|---|---|---|---|
| 1 | **continuous global geometry** G | the solid: a finite union of bounded closed convex polyhedra strictly inside (0,1)³, with binary64 parameters read as exact rationals. It is not a voxel object. The same canonical bodies define both the SOLID samples and the responses. Curved research scenes add closed balls ([CURVED_OCCLUDERS.md](../CURVED_OCCLUDERS.md)) | scene container (`geometry`; the labeller's halfspace rows `P` and per-body row counts `K`) | generator, labellers |
| 2 | **locally observed geometry** | the SOLID crop of a window: 1[x ∈ solid] at the L³ receivers of W, L ∈ {64, 128, 256}. It samples G at voxel centres inside W only, and is stored once per window, shared by all its channels. It does not determine G: bodies outside W are absent, and a sub-voxel part of a body inside W can be missed ([LOCAL_INVERSE_GEOMETRY.md](../LOCAL_INVERSE_GEOMETRY.md) Theorem 3′) | `solid` crop | Student (input), labellers |
| 3 | **source-specific observations** | r_j = R_{s_j} on the FREE receivers of W, one channel per driver source. A channel depends on all of G and on s_j: occluders outside W shape the response interface inside W | `lit` crops, `responses[j]` | one channel per Student example; labellers |
| 4 | **full-world teacher information** | all privileged information: G inside and outside W, every source position, K, the other channels of the window, m and the informative mask, the selection logs, training weights, label statuses, and the labels themselves (supervision targets, never inputs). "Teacher" names this privileged information; the current pipeline has no neural Teacher | scene container (`geometry`, `sources_hex`, logs), label journals | generator, labellers, loss weights |
| 5 | **local-only Student inputs** | X = (SOLID crop, r_j, L, o): `rind3d_100k.loader.student_inputs`, keys `solid`, `response`, `L`, `origin`. The output is a probability over the 128³ source cells outside the window box | loader | Student |
| 6 | **deterministic posterior certification** | a proof that a stored label lies within a stated total-variation (TV) distance of the full-world posterior q_G. `certified_stat` labels carry a statistical certificate instead (§3) | label `tv_bound`, status, `meta` | validation; never a Student input |

**What the Student can learn.** Its target is p*(s | X) = E[q_G(s) | X], the average of the full-world posteriors over
all worlds compatible with X (`RIND3D_StudentResearch/A_TASK_DEFINITION.md` §4). It is not expected to reproduce an
individual q_G. The loss floor H(S | X) = H(S | X, G) + I(S; G | X) is positive.

**"Local-only" has two uses.**
* *Local-only Student inputs* (row 5) are an information boundary.
* *Local-only rendering* (the 100K physical stage, `RIND3D_Dataset100K/docs/LOCAL_RENDERING_100K.md`) is a cost
  property. It computes responses only where the selection law and the stored crops need them, always against the
  full continuous global geometry. No global response field is computed or stored.

## 3. The posterior label and its certificates

**The full-world posterior.** For window W, channel j and world G:

    F_W^j(G) = {s ∈ D(G) \ B̄(W) : R_s(x) = r_j(x) for every FREE receiver x of W},   q_G = Uniform(F_W^j(G))

* B̄(W) is the closed window box [o/N, (o + L)/N]³.
* q_G is stored as cell masses on the 128³ grid.
* q_G conditions on the full geometry. Computing it is a teacher-side operation.

**Certificate statuses** (contracts v2 and v3, `labels.statuses`):
* `certified`: a *deterministic* tv_bound ≤ 0.05, from single-cell support or an exact path. The tau-pencil engine
  bounds the volume of F in every cell from below and above, and derives a deterministic `tv_bound`.
* `certified_loose`: a deterministic bound, 0.05 < tv_bound ≤ 0.10.
* `certified_stat`: a *statistical* certificate (FastLabel v3.4): TV ≤ 0.05, with probability of a false certificate
  ≤ family_delta / n_tasks = 0.01 / 4.8·10⁶ per label. It is not deterministic. It still requires a sound outer bound
  B ⊇ F.
* `uncertified`: tv_bound > 0.10. Such labels are stored and never presented as qualified.

**Distinctions that matter.**
* *Reproducible is not deterministic.* FastLabel keys every random stream, so a label is a deterministic function of
  the task and the configuration. Its `certified_stat` guarantee is still probabilistic.
* *Limits of the deterministic path.* The tau-pencil engine's exactness rests on the decision margins of its
  floating-point predicates ([README.md](README.md), "Not proven, by design"). An independent exact-kernel verifier of
  deterministic certificates is in development (`RIND3D_Posterior/src/rind3d_posterior/certify/`). It is not part of
  any contract yet.
* *What a certificate bounds.* The error of the label relative to q_G. It does not bound the Student's error relative
  to p*.
* *Local structure guides, the certificate decides.* Local-first research may use the local response interface to
  choose which receivers' exact constraints to evaluate first. The certificate always comes from the exact
  constraints with the global geometry ([LOCAL_INVERSE_GEOMETRY.md](../LOCAL_INVERSE_GEOMETRY.md) Proposition 8).

## 4. Where legacy names remain

* **Stored data and APIs**, kept for compatibility with v0.1, the 1K and contracts v1/v2:
  - `lit` crops and bits (`lit_bits` in FastLabel tasks), `lit_frac`;
  - `cls` codes: 0 = response-inactive FREE, 1 = response-active FREE, 2 = SOLID;
  - `mixed_mask` (the v1/v2 rule, kept in v3 containers);
  - code identifiers such as `DARK` / `LIT` in the labellers and the `(lit, dark)` endpoints returned by
    `interface_edges`.
* **Contract files.** Contract values are not reworded. Contract v3 records the mapping in its `terminology` key.
* **File and directory names**, e.g. `RIND3D_Generator/experiments/shadow_volumes/` and
  `RIND3D_Generator/docs/scientific/shadow-volume-design.md`.
* **Frozen and dated documents** keep their wording:
  - the v0.1 generator documentation (`RIND3D_Generator/docs/`, sha256-pinned in `provenance/ORIGINAL_SOURCE.json`);
  - the 1K release contract and its documents (`RIND3D_Dataset1K/docs/`);
  - the 2026-11-13 audit (`RIND3D_Audit/`) and Student research phase (`RIND3D_StudentResearch/`), including names
    such as "all-dark S6";
  - experiment reports and validation records;
  - `RIND3D_HANDOFF/RIND3D_IMPLEMENTATION_FACTS.md`.
* **Literature.** Cited titles and their summaries keep the authors' wording.
* **Writing rule.** New prose uses the terms of §1. Where a reader needs the legacy name, give it once in
  parentheses, e.g. "response-active (stored as `lit`)".
