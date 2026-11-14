# Restricted production operations

A production campaign is mutable, private, append-only work. A public release is an immutable whitelist snapshot. Never distribute the campaign directory: it contains the master seed, true sources, witnesses, physical evidence, worker journals and operational state.

Use the certified producer checkout/environment and a final accepted build certificate. These commands are producer commands, not part of the reader-only wheel. Do not run an environment sync, rebuild native extensions, or change producer dependencies during a campaign: the frozen build check will refuse a different implementation. A new installation needs its own certification, not a copied success flag. From the exact producer repository/environment, select its frozen reader sources and create a new campaign using its matching certificate:

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/reader/src:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -m rind3d.campaign_cli create --root /data/core-campaign --config configs/lite-core-1k-v01.yaml --certificate outputs/lite-core-1k-v01/production-certification.json --target-total 1000
.venv/bin/python -m rind3d.campaign_cli run --root /data/core-campaign --max-new-scenes 100 --max-runtime 3600 --verification-workers 4
.venv/bin/python -m rind3d.campaign_cli status --root /data/core-campaign
```

The configuration and certificate must actually exist and pass all gates before those production commands can run. Diagnostic certificates are capped at twenty scenes and cannot authorize the named 1K release. The campaign generates a private random 256-bit master seed unless explicitly supplied; protect the campaign directory and backups. Geometry-ordinal and source-attempt HMAC domains are distinct. Extending the target does not change previously assigned seeds or identities.

`--target-total` is cumulative. `--max-new-scenes` bounds this execution's additional READY publications. `--max-runtime` is a soft stop-admission limit. The coordinator pauses admission, finishes/checkpoints current work and waits up to the frozen grace interval before terminating isolated children. A single uninterruptible native/driver call is not assumed instantly cancellable. SIGINT and SIGTERM use the same safe pause path; the pause command requests it from another terminal:

```bash
.venv/bin/python -m rind3d.campaign_cli pause --root /data/core-campaign
.venv/bin/python -m rind3d.campaign_cli run --root /data/core-campaign --resume --max-new-scenes 100 --max-runtime 3600 --verification-workers 4
.venv/bin/python -m rind3d.campaign_cli verify --root /data/core-campaign
```

All run invocations resume; `--resume` makes that intent explicit. One coordinator owns the root. Scientific workers hold writer leases and attempt locks, verify ownership epochs, and have Linux parent-death signals. Recovery requires an exclusive writer fence before changing ownership. If a surviving native call still holds a lease, recovery refuses to race it. It does not guess that a stale PID means safe ownership.

On restart, committed scene seals/checksums are checked and valid READY scenes are not regenerated. A fully committed scene awaiting catalog registration is recovered from its immutable seal. Partial rendering and independent-verification journals retain the original attempt identity. Incomplete or watchdog-aborted work is retried under bounded rules; it is not physical invalidity. Certified source infeasibility has a separate recorded proof and resamples the source while retaining geometry. Unproved window failure stops as incomplete. Failed seeds are not silently replaced by easier geometries.

Use `campaign verify --recover-catalog` for explicit orphan registration. `--metadata-only` checks seals/sizes with a narrower guarantee. `--physical` reruns authorized scientific validation and can be expensive. Preserve corrupt/conflicting files for diagnosis rather than deleting evidence to make a campaign pass. Resuming with a changed scientific build or configuration is rejected; incompatible changes need an explicit versioned partition.

After 1K, future extension changes only the cumulative target:

```bash
.venv/bin/python -m rind3d.campaign_cli extend --root /data/core-campaign --target-total 2000
```

This command alone generates no scenes. Do not run future 2K/5K/10K production without separate authorization. Geometry-group splits and scientific IDs are independent of target count and worker completion order.

The accepted configuration uses one bounded CUDA producer and four single-thread CPU verifiers. CUDA is optional; `campaign run --cpu-only` uses the certified CPU route. Full fields are complete before window selection. A and B remain independent; optional additional-source search is zero in Core. The 30-second threshold is a cooperative verification scheduling quantum, not a reason to reject a valid slow scene.

Per-session records retain wall time, CPU usage, attempt states and telemetry. Gracefully finalized sessions have complete process-rusage accounting. Abruptly killed sessions expose checkpoint lower bounds and explicitly unmeasured crash tails. Do not sum only successful scene latencies to estimate production cost. Disk/RAM/VRAM and throughput must be measured for the actual build/distribution; old GPU kernel timings are not READY throughput guarantees.

After reaching a verified prefix, export without changing the source campaign:

```bash
.venv/bin/python -m rind3d.campaign_cli export --root /data/core-campaign --output /data/RIND3D-v0.1 --name RIND3D-v0.1 --count 1000
```

The named 1K export refuses insufficient or diagnostic data. License and citation remain pending until the owner supplies them. Producing a verified local package is separate from authorization or completion of an online upload.
