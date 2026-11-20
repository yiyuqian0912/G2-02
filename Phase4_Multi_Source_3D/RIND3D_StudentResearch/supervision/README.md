# Certified posterior targets for Student research

For a new training experiment, open an authenticated shard with `IssuedExamples`, keep the default unique-cell binary64 target, and pass it to `sparse_cross_entropy` with the model's log probabilities. The reader returns only the local observation and its training target. It requires every planned channel to have an issued certificate. A missing, incomplete, statistical or one-hot label cannot enter this path.

## What a Student can learn

Let X include both local ZYX masks, integer XYZ origin and side. At fixed hidden geometry G the Teacher's target is the 128³ pushforward of uniform continuous volume on legal sources reproducing every FREE receiver in X. A world can produce the same X as another world with a different feasible-source distribution. For example, if two equally weighted hidden worlds produce identical X and support disjoint single cells A and B respectively, the local Student's optimum is (½,½), even with unlimited capacity.

For the **actual training-example law** D and a prediction u(X), conditioning cross-entropy on X gives

    E_D[−Σᵢ q_G(i|X) log uᵢ(X) | X] = H(m) + KL(m ‖ u(X)),
    m(X) = E_D[q_G(·|X) | X].

This identifies the population minimizer almost surely. With the issued approximate labels p, the same argument gives the minimizer E_D[p | X]. Convexity of TV yields TV(E_D[p|X], m(X)) ≤ E_D[ε|X] ≤ 0.05. This bounds the target shift. It does not bound the trained model's approximation error or an unbounded log loss. If all evaluated log losses lie in [a,b], the discrepancy of their expectations under p and q is bounded by (b−a) TV(p,q).

The generating-source posterior coincides with the Teacher's per-world target only when the induced conditional source law is uniform on the feasible set (or gives the same cell masses). Even in one world, two equally large feasible subregions with generating prior masses 0.9 and 0.1 give a different posterior after an identical observation. Channel inclusion, window selection, scene weights and dropping incomplete examples can all change this conditional law. Use the original natural-v3 observations. Declare whether training samples channels, windows, or scenes with equal weight. Split by geometry before fitting. The 807 channels here qualify a reader, not a proposed training/test split. No hidden geometry, source coordinates, global field, solver diagnostics or alternative witnesses enter X.

## Why float32 is an explicit choice

Each signed-off nonzero probability is exactly k·2⁻⁵², with integer k>0 and Σk=2⁵². Flat cell ID is (x·128+y)·128+z. Omitted cells have zero *issued* weight. The physical distribution may still have mass there. The binary64 reader returns one record per positive cell, with precisely the authenticated values.

A plain float32 cast can give the two legal wire masses (2⁻⁵², 1−2⁻⁵²) a total of 1+2⁻⁵². Historically 507/807 canary vectors lost exact unit mass under a simple float32 cast. The pilot loader also renormalizes and the historical training collator has a true-source fallback. Those remain reproducible historical interfaces. The opt-in strict reader is the supported path for the deterministic guarantee.

For a consumer that expects binary32 coefficients, select `float32_terms`. Keep directly representable values. Otherwise expand k=d₀+2²⁴d₁+2⁴⁸d₂ in base 2²⁴. Emit each positive dⱼ·2^(24j−52) at the **same** original cell ID. Each digit is exactly representable, and its power-of-two scaling lies in the normal binary32 range. Per cell the real sum of the stored coefficients equals k·2⁻⁵². Repeated IDs therefore carry additive loss contributions. This preserves support and normalization exactly for all valid wires, including one whose TV bound already equals the approved tolerance. The code verifies the emitted tick sum for every cell. TV(training target, wire)=0 in both modes.

Use gather plus additive reduction for duplicate IDs. Overwrite scatter, top-k pruning, floating coalescing, dtype conversion or re-normalization requires separate error accounting. CPU `sparse_cross_entropy` casts selected log probabilities and weights to double before multiplying and summing. Model log probabilities, gradients and the loss scalar are still computed in finite arithmetic. The theorem concerns the target coefficients. GPU normalization, AMP, softmax and arbitrary training pipelines have no automatic error budget.

## Open an existing shard

The following is runnable from `RIND3D_StudentResearch` with CPU PyTorch, NumPy and zstandard. It exercises the actual full-grid loss with a uniform model for illustration.

```sh
export RIND_CBC_CAMPAIGN=/home/yushu/Projects/RIND/reports/astra-cbc-production-proof/canary
export RIND_V3_SCENES=/home/yushu/Projects/RIND/RIND3D_100K_v3_work/release/canary/scenes
export STUDENT_PY=/home/yushu/Projects/RIND/RIND3D/.venv/bin/python
"$STUDENT_PY" - <<'PY'
import math, os
from pathlib import Path
import torch
from supervision.decoder import IssuedExamples, sparse_cross_entropy
index = Path("../RIND3D_Posterior/experiments/credit_solver/assurance/results/issued-sha256.json")
data = IssuedExamples(os.environ["RIND_CBC_CAMPAIGN"], os.environ["RIND_V3_SCENES"], index,
    manifest_sha256="0f1fa5c3df22f32085cd25f44a73db66dc475d9af8490830c0bea657be833f3d",
    index_sha256="13a0dacfe3141dc551c355ac51b9b1c743f292fac4fd98510d44d193ea1baaef")
X, target = data["000004_w00_s0"]
log_probs = torch.full((1,128,128,128), -math.log(128**3), dtype=torch.float64)
print(len(data), X["L"], X["origin"], len(target.cells), target.tv_bound)
print(sparse_cross_entropy(log_probs, [target]).item())
PY
```

For the historical hierarchical model's `pilot.student.target_ce`, request `float32_terms` and pass `*torch_batch(targets)` to its final three arguments. Match local feature axes explicitly when converting ZYX to the pilot's XYZ features. Its legacy data collator does not use this reader automatically. The real integration test uses two DataLoader workers, two verified tasks, the original hierarchical loss and an independent dense-log-probability comparison. A separate synthetic loss/gradient fixture has no scientific data dependency.

## Issue a bounded new research shard

1. Follow the [frozen-build and campaign instructions](../../RIND3D_Posterior/experiments/credit_solver/assurance/OPERATIONS.md) through construction of the CPU environment and the `py`, `kernel`, `phase`, `PYTHONPATH` shell variables. Use the exact detached commit and checked runtime lock. Set `sup` to this **handoff checkout's absolute supervision directory** before changing directory. On the authenticated host run `"$py" "$sup/release_gate.py" "$kernel" check`. It compares the entire scientific build and two additional imported code digests.
2. Predeclare the unchanged natural-v3 scene inventory and resource limits. For a new shard, run `"$py" "$kernel/assurance/campaign.py" init "$scenes" "$root"`. Review the entire plan, population, file digests and build and distribute its trusted SHA-256 separately as `pin`. Bound the compressed bytes per shard, worker memory × worker count, attempts, watchdog and time tiers. The frozen campaign holds the compressed shard in coordinator memory. Run and resume with the documented tier commands using this root/pin. A hard stop leaves a recoverable incomplete attempt. Never supply a fallback target or quietly discard a slow channel.
3. After the workers finish, run `"$py" "$kernel/assurance/campaign.py" verify "$scenes" "$root" --manifest-sha "$pin"`. Then run `"$py" "$sup/release_gate.py" "$kernel" export "$root" "$pin" "$index"` with a **new file** in an existing durable directory. This checks the build, takes the coordinator's shared lock, checks complete task coverage, verifies artifact status/identity/content including historical incomplete labels, refuses nonfinal or duplicate issuance, and publishes via fsynced temporary file and exclusive hard link. Empty pre-start directories have no scientific identity. A protected local issuer must authenticate/distribute the resulting index digest independently. Self-hashes from an untrusted download do not establish issuer authority.
4. Independently validate with the handoff's `assurance/verify_release.py ROOT PLAN_SHA INDEX INDEX_SHA` under the frozen runtime paths. Give the training reader both pins and the original scene containers. Reading checks the source file and recomputes the actual XYZ class hash of each local crop. Keep scientific files, digests, task identities, build and ledger for future reproduction. The reader verifies the exact previously issued artifact; it does not establish the original geometric proof anew.

The operator still needs a trusted means of distributing pins for each new shard. A successful historical canary alone cannot authenticate new data. New platforms or unsupported scene families require a separately qualified build.

## Qualification results

The unchanged numerical run previously certified 795/807 at the first budget and all 807 after retries. Its recorded total CPU cost was about 2.33× the archived FastLabel comparator. This iteration adds no operations to a numerical solve. It measures the minting/readback path separately.

| Measurement on the retained 807 | Unique binary64 | Additive float32 |
|---|---:|---:|
| Exact matches of actual CPU tensor masses to authenticated wire, by address | 807/807 | 807/807 |
| Target records over all channels | 4,223,573 | 8,417,482 |
| Checked read + crop + CPU batch, summed wall time | 50.37 s | 50.33 s |
| Per-channel wall P50 / P95 / maximum | 12.9 / 175.5 / 278.1 ms | 13.1 / 184.1 / 264.4 ms |

The 14 complete independently integrated rational references matched task, canonical geometry, crop hash, origin, side and resolution. All actual target errors were within their individual certificates. Maximum observed TV was 5.86×10⁻¹⁵ or less, including the archived 214-cell case in the comparison. This empirical maximum cannot replace the per-label bound. Of 807 channels, 295 used at most one binary32 term per cell, 426 needed up to two and 86 needed three. The audit alternated mode order, included both readers and an independent integer comparison, took 103.71 s wall and 103.67 s CPU and peaked at 508 MiB RSS on this host. Shared OS caches and concurrent tests affect these timings. They are not training throughput or physical solve benchmarks. Per-channel records and environment versions are in [results](results/summary.json).

The fresh exporter reproduced the retained 807-entry index byte-for-byte. A separate fresh one-channel restart trial killed its own coordinator during active work, recovered under the original identity, reissued identical wire and bound, refused corrupted bytes, and registered zero new attempts on the next resume. The new exporter accepted that recovered shard. The full numerical/related regression run passed 801 cases. The reader, transition-surface and tensor integration suite passed 29, including both real-data tests. All 32 existing arithmetic implications and three additional integer lemmas passed.

```sh
# In RIND3D_StudentResearch, with the reader environment variables shown above.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$STUDENT_PY" -m pytest -q -rs supervision
"$STUDENT_PY" supervision/audit.py "$RIND_CBC_CAMPAIGN" "$RIND_V3_SCENES" /tmp/new-cbc-training-audit
# Use a previously unused output directory for each audit.
# Z3 and the historical numerical suites use the developer environment.
bootstrap=/home/yushu/Projects/RIND/G2-02-3d-100k/.venv/bin/python
"$bootstrap" supervision/smt_training.py
```

For scientific and operational limits see the [invariant proof and executable correspondence](../../RIND3D_Posterior/docs/ASTRA_SUPERVISION_FOUNDATION.md).
