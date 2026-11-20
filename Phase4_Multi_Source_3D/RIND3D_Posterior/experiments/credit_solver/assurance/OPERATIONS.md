# Running and verifying the frozen CPU labeler

The calibration population is the original 36 authenticated canary scene containers. Their file SHA256s, 807 task keys, physical input hashes and numerical build are in the immutable campaign manifest. The trusted receipt inventory for this experiment is results/issued-sha256.json. Use a trustworthy checkout or signed distribution channel for the expected hashes. An artifact and a replacement hash obtained from the same untrusted sender do not establish provenance.

## Reproduce on the tested machine

Run from the handoff repository root. Select a fresh output directory. The following uses the installed 100K Python solely as a source of verified wheel payloads for an independent CPU environment. The detached checkout is important because the scientific build fingerprint includes the commit. Clean installation on a second operating system or a different CPU/compiler is outside this certification.

```sh
set -eu
repo=$(git rev-parse --show-toplevel)
notes="$repo/Phase4_Multi_Source_3D/RIND3D_Posterior/experiments/credit_solver/assurance"
base=/home/yushu/Projects/RIND/RIND3D_100K_v3_work
scenes="$base/release/canary/scenes"
bootstrap=/home/yushu/Projects/RIND/G2-02-3d-100k/.venv/bin/python
runroot=$(mktemp -d)
git -C "$repo" worktree add --detach "$runroot/src" acc21857771fb64b0c012b0236d0d4b979d2751f
"$bootstrap" "$notes/rebuild_runtime.py" "$runroot/env" > "$runroot/runtime-lock.json"
cmp "$runroot/runtime-lock.json" "$notes/results/runtime-lock.json"
py="$runroot/env/bin/python"
phase="$runroot/src/Phase4_Multi_Source_3D"
kernel="$phase/RIND3D_Posterior/experiments/credit_solver"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR="$runroot/numba"
export PYTHONPATH="$kernel:$phase/RIND3D_Posterior/src:$phase/RIND3D_Generator/src:$phase/RIND3D_Generator/vendor/canonical_geometry/src"
cd "$kernel"
"$py" assurance/campaign.py init "$scenes" "$runroot/canary"
pin=0f1fa5c3df22f32085cd25f44a73db66dc475d9af8490830c0bea657be833f3d
printf '%s  %s\n' "$pin" "$runroot/canary/manifest.json" | sha256sum -c -
"$py" assurance/campaign.py run "$scenes" "$runroot/canary" --manifest-sha "$pin"
"$py" assurance/campaign.py run "$scenes" "$runroot/canary" --manifest-sha "$pin" --seconds 45 --cap 8192 --workers 3 --watchdog 135
"$py" assurance/campaign.py run "$scenes" "$runroot/canary" --manifest-sha "$pin" --seconds 360 --cap 32768 --workers 2 --mem-mib 6500 --watchdog 485
"$py" assurance/campaign.py verify "$scenes" "$runroot/canary" --manifest-sha "$pin"
"$py" study/check_exact.py "$base" "$runroot/exact.jsonl"
```

Compare the pinned plan before performing physical work. For a new approved inventory, freeze its digest as a distinct run. Each session verifies prior receipts and file hashes. An incomplete channel remains in the inventory. Retrying uses a fresh process and restarts its search on the same observation. A successful label is immutable. The same exhausted soft-budget tier is skipped after a durable INCOMPLETE. The cap on tries bounds repeated interruptions. A directory with no start record has no scientific commitment and its attempt number is never reused. The wall deadlines are cooperative search limits; actual elapsed time includes finalization. Ctrl+C/SIGTERM stops dispatch and drains active workers. A separate hard watchdog kills an individual attempt. Receipt recovery can lack CPU/RSS for a worker lost to a coordinator crash; it records interruption and does not impute a physical result or zero lost work. Workers inherit one Linux owner lock and install parent-death SIGKILL plus a parent-PID race check. The fixed per-worker virtual-memory limit includes inherited shared mappings. Per-process RSS cannot be summed to obtain PSS.

Worker files in an attempt directory are staging until the coordinator independently checks the statement and registers the completion receipt. Filesystem requirements include same-device hard links, atomic rename, advisory flock, and directory fsync on a trusted local Linux file system. Provision the campaign root and its parent on durable storage before production; if provisioning creates directories, persist their parent directory entries as well. The recovery experiment tests process loss on a local filesystem; laptop power-cut and multi-host recovery have not been measured. The ledger and expected digests must be protected from concurrent same-user modification. This is a local research issuer, with no distributed transaction or remote signer. The coordinator eagerly caches the **compressed** input scene blobs in one campaign shard. Explicitly bound the number/bytes of scenes per shard before scaling beyond this canary. No scientific label or hidden source is cached across observations. Only canonical rows may be shared as immutable values keyed by the complete geometry bytes.

The verify action validates existing receipts. Independently compare certified keys against the entire planned key set before describing a campaign as finished. summarize.py does this population-level join against the archived comparison tables and records every attempted channel. A digest inventory is specific to one issuance. Re-solving under a wall-clock limit can change an authorized scientific result, and timing diagnostics change raw artifact hashes even when the posterior wire is identical.

## Audit and recovery evidence

The results directory includes the frozen build, installed wheel RECORD hashes, the per-channel issuance index, all attempts (including elapsed time and peak RSS), exact reference comparisons, tests, timings and full-population summary. The rebuilt venv contains checked runtime wheel files, the same base CPython and detached canonical sources. It omits the editable generator import hook. No CUDA is needed. This reproduces the executable environment on the tested system; it does not rebuild CPython/NumPy/LLVM from source or establish cross-platform equivalence. rebuild_runtime.py checks wheel payloads against their own installed RECORDs; independently pin and compare the reported RECORD identities. The historical source-checkout environment was also exercised separately with the optional native GMP comparison extension. Neither environment uses GMP grid credits to issue labels.

For the developer suite install pytest, pybind11, GMP headers and Z3 in the existing project test environment, with the same path exports. From the credit_solver directory:

```sh
bash native/build.sh "$bootstrap"
"$bootstrap" -m pytest -q -rs .
"$bootstrap" -m pytest -q -rs ../global_inverse/test_global.py ../mixed_patch ../xlf_assurance ../../../RIND3D_Dataset100K/tests/test_interface_v3.py ../../../RIND3D_Dataset100K/tests/test_sampling.py ../../../RIND3D_Dataset100K/tests/test_thin_slabs_v3.py
"$bootstrap" smt_kernel.py
"$bootstrap" smt_screen.py
"$bootstrap" smt_projective.py
"$bootstrap" "$notes/smt_account.py"
```

The first suite includes the new tests when run from the handoff checkout with its own kernel on PYTHONPATH. Reproduce the crash experiment by running `"$py" "$notes/recovery_trial.py" "$kernel/assurance/campaign.py" "$scenes" "$runroot/canary" "$runroot/recovery"` on a completed canary. This creates a disposable child manifest, kills only its own active coordinator, checks that its child ceases writing, recovers the same wire, tests repeat-resume and corrupts a disposable copy to exercise rejection.

For the original retained output, independent complete-issuance verification from the handoff checkout uses the following pinned input and issuance hashes. An independent rerun has its own issuance index.

```sh
"$py" "$notes/verify_release.py" /home/yushu/Projects/RIND/reports/astra-cbc-production-proof/canary \
  0f1fa5c3df22f32085cd25f44a73db66dc475d9af8490830c0bea657be833f3d \
  "$notes/results/issued-sha256.json" 13a0dacfe3141dc551c355ac51b9b1c743f292fac4fd98510d44d193ea1baaef
```

For the saved-wire precision audit run `"$py" "$notes/training_conversion.py" /home/yushu/Projects/RIND/reports/astra-cbc-production-proof/canary "$runroot/float32.jsonl"` with the frozen kernel paths above. Each row contains an exact rational normalization and TV allowance.
