# RIND3D research archive (index)

The research behind RIND3D-100K v3, frozen on 2026-11-19 when FastLabel v3.4 was chosen as the production labeller.
None of it is on the production runtime path of `yushu/3d`. Everything here is recoverable from the listed branches
(all pushed to `origin`) and the work directory `/home/yushu/Projects/RIND/RIND3D_100K_v3_work/` on the development
machine (a local directory, not in git).

## Branches and commits

| branch | tip | content | status |
|---|---|---|---|
| `yushu/3d` | tag `release/rind3d-100k-v3-2026-11-18` | production branch | authoritative |
| `yushu/3d-v3-release` | integration branch of this release | v3 tasks 1, 3, 4 + FastLabel v3.4 production path | merged into `yushu/3d` (tag `release/rind3d-100k-v3-2026-11-18`) |
| `yushu/3d-v3-fastlabel-cert` | 98b1668 | FastLabel v3.4 certification review: certificate rules (contract-pinned streams, strict validation, curved guard), `FASTLABEL_V34_CERTIFICATION.md`, validation scripts and results | integrated (cherry-picked) |
| `yushu/3d-100k-v3` | b3823ad | v3 work: direct-interface eligibility, thin slabs, contract v3, curved research layer, local-first notes, benchmarks | integrated (fast-forward base of the release) |
| `yushu/3d-v3-terminology` | 9029e61 | driver-source / binary-response vocabulary, glossary | integrated (cherry-picked) |
| `yushu/3d-v3-curved-assessment` | b9b7f2b | `docs/CURVED_ASSESSMENT.md`: closed-ball occluders | integrated (cherry-picked); balls stay research-only |
| `yushu/3d-v3-staging` | 672127a (tag `checkpoint/v3-verifier-2026-11-18`: f80400d) | deterministic certification of the tau-pencil engine: exact kernel `certify/` (ar, poly, scene, tvb, check, tree, verify), certificate-emitting engine, `docs/CERTIFIED_LABELS.md`, OUT-region physical audits | research, frozen |
| `yushu/3d-localfirst-research` | d1929c6 | local-first E1–E3: Proposition A′ certified interface regions, root clipping (1.54× on the exact phase) | research, frozen |
| `yushu/3d-voxel-physics` | a69c1b9 | Model B (voxel physics) theory; no certified labeller | research, frozen |
| `yushu/3d-xlf-research` | 1c2f6e8 | XLF self-certifying exact local-first solver + replay checker | research, frozen |
| `yushu/3d-xlf-wip` | cc512b3 | XLF multi-resolution scan, solid splits | research, frozen |
| `yushu/3d-xlf-exp-median-pencil` | c89c078 | XLF median pencil split prototype | research, frozen |
| `yushu/3d-xlf-math` | dcd2520 | XLF mathematical foundations / algorithm correctness / implementation assurance (incomplete), exact tests, fixes D1–D4 | research, frozen |
| `astra/3d-v3-receiver-certificates` | 94b4820 | Astra's independent exact rational inverse-shadow research, mixed-interface constraints, blind posterior benchmarks | research (Astra), pushed as-is |
| `yushu/3d-experiment` | 3c30fe3 | Student lab | research |
| `yushu/3d-100k-h20` | 5560a20 (tag `archive/rind3d-100k-h20-2026-11-14`) | historical v1 100K profile | historical |

Other archive tags: `archive/pre-fastlabel-core-2026-11-13`, `archive/rind3d-research-2026-11-13`,
`archive/yushu-3d-pre-phase4-2026-11-14`.

**Uncommitted work left untouched**:
* `G2-02-astra-inverse`: Astra's XLF proof documents and `experiments/proof_labels/`, owned by another agent. A
  read-only snapshot is in `RIND3D_100K_v3_work/research_archive/astra_uncommitted_snapshot_2026-11-19/`.
* `G2-02-3d`: Phase3 adaptive-teacher work, unrelated to v3.

## Work-directory artifacts (`RIND3D_100K_v3_work/`)

| path | content |
|---|---|
| `cmp_v3/` | 54 v3 canary tasks (`tasks.json`), FastLabel labels and runs (`labels/*_fl.npz`, `fl.jsonl`), local-first variants |
| `xlf/ref/` | 12 certified deterministic tau-pencil labels with per-cell enclosures (`bench_tp.jsonl`: label + verify times) |
| `perf/` | 3 more certified tau-pencil labels and certificates |
| `xlf/batch2_*.jsonl`, `batch3_*.jsonl`, `batch3_partial_summary.txt` | XLF benchmarks (pre- and post-multires) |
| `research_archive/astra_uncommitted_snapshot_2026-11-19/` | Astra's XLF proof notes (snapshot) |
| `session_backup_2026-11-18/scratchpad/` | development scratch of the deterministic-certification and Astra-review work |
| `e12/`, `e3/`, `e2_*`, `lf_audit/` | local-first E1–E3 runs |
| `sampling_impact*`, `curved_stats*` | v3 sampling-impact and curved-occluder statistics |

## Findings worth keeping

### Deterministic certification of the tau-pencil engine (`yushu/3d-v3-staging`)

* Astra's report of a tau-pencil volume-enclosure violation was traced to vertex projection and EPS snapping in
  `tpl/poly.py`: regions are not the exact polytopes their planes define.
* An independent exact-kernel verifier certifies tau-pencil labels deterministically. Examples:
  * 19-task sweep: TV ≤ 7.2e-4;
  * the 12 reference labels in `xlf/ref/`.

  Cost: label + verify 3.5–396 s per task. The verifier is 1–3× the labeller and nearly as complex as the solver.
* Physical audit of every engine-excluded region on 3 tasks: 0 feasible sources in ~21.9K renders
  (`experiments/certify/out_audit*.py`).

### XLF: exact local-first solver (`yushu/3d-xlf-*`)

* **Design.**
  * An exact BSP on plane-based polytopes.
  * Rigorous segment-criterion predicates: guaranteed shadow I_c(Q) = ∩ C_c(v), pencil separation planes, convex
    local constraints of interface receivers with angular extremes per occluder edge.
  * A hierarchical receiver-block scan.
  * The certificate is replayed cheaply: the checker takes 0.2–5 s, against 1.7–21 s for the legacy verifier.
* **Results.**
  * Where it finishes, it agrees with the certified tau-pencil labels: 0 per-cell conflicts, identical cells.
  * With the multi-resolution scan it is 2.5–3× faster than v3 on one-cell L=128/256 tasks.
* **Not production-ready.**
  * Pre-multires (1c2f6e8): 7 of 13 tasks timed out at 900 s.
  * After solid splits (cc512b3): 4 of 4 completed runs timed out.
  * Near occluder edges and vertices the ambiguity does not shrink under x/y/z bisection. Pencil splits of stalled
    nodes exploded; median pencil splits still timed out.
* **Soundness defects found by the math audit** (fixed on `yushu/3d-xlf-math`, 4d44da5):
  * D1: per-cell sums used round-to-nearest addition.
  * D2: the checker's inner-plane test missed the rounding of sc·h_k. Exact counterexample: facet −3x+y−3·2⁻³⁰.
  * D3, D4: the kernel's volume and γₙ bounds.
* **Astra's counterexample (R1, must be kept).** The XLF replay checker at 1c2f6e8 and af10030 accepted an
  incomplete partition tree as a global posterior certificate. On a 64³ thick-box case, a single-cell answer was
  accepted with TV 0 when the true TV is ≥ 119/120.

  The checker only flagged internal nodes with no children. The required invariant: every node is either a leaf with
  exactly one claim, or has exactly the two complementary children of one plane; every node is reached, including
  after timeouts; a run that timed out never yields a certificate.

  Evidence: `research_archive/astra_uncommitted_snapshot_2026-11-19/XLF_CHECKER_FREE_FEASIBILITY.md`; risk R1 in
  `docs/XLF_IMPLEMENTATION_ASSURANCE.md` on `yushu/3d-xlf-math`.
* **Astra's normalization counterexample.** Floats that sum to 1.0 are not exactly normalized: three copies of
  fl(1/3) sum to 1 − 2⁻⁵⁴. It applies to every serialized probability vector, including FastLabel's; see
  `FASTLABEL_V34_CERTIFICATION.md`.
* **Open mathematical questions** (status headers of the three XLF documents):
  * localization bounds for unknown edge correspondence;
  * a receiver sufficiency set;
  * the complexity of the exact TV supremum;
  * fixed-time completion and near-edge complexity;
  * mechanical checking of ancestor inheritance;
  * an end-to-end exact audit;
  * curved occluders.

### Local-first (E1–E3, A′)

* Proposition A′ certified regions are informative.
* Plane selection is about as good as random (the E1 rule was not met).
* Root clipping gives 1.54× on the exact phase.
* The exact pair constraint C(k) \ C(l) is strictly stronger than A′. Counterexample: s=0, l=(10,1,0), k=(10,−1,0),
  K=[4,5]×[−0.1,0.1]².
* Interfaces alone do not determine F; multiple explanations make F non-convex.

### Curved occluders

* Closed balls are physically implemented, with exact intersection, fixtures and CPU/GPU comparisons.
* Generation costs 5.6–7.1× a polyhedral scene.
* There is no certified labeller, so they are excluded from the certified production profile, enforced by
  `rind3d_100k/profile.py`.
* Assessment: `docs/CURVED_ASSESSMENT.md`, `docs/CURVED_OCCLUDERS.md`, and the research contract
  `RIND3D_100K_CURVED_RESEARCH_v0.json`.

## How to resume

* **XLF:** start from `yushu/3d-xlf-math` (dcd2520). It has the fixes D1–D4 and the three documents.
  1. Fix R1 first.
  2. The open efficiency problem is the near-edge/vertex stall.
  3. Validate against `xlf/ref/` with `experiments/xlf/run_one.py KEY TL` (env `XLF_CHECK=1`) and
     `experiments/xlf/summarize.py`.
* **Deterministic tau-pencil certification:** `yushu/3d-v3-staging`, `experiments/certify/bench.py`.
