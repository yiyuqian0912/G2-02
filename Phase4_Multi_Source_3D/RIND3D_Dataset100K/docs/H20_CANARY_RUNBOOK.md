# RIND3D-100K: H20 production gate and canary runbook

**Scope.** This runbook covers everything that must pass on the 16 × H20 target before the 100K campaign under contract v2 (`contract/RIND3D_100K_CONTRACT_v2.json`). It is run in order and stops at the first failure.

**Status.** Every check below is **PENDING**: none has been executed on an H20. Mark a check PASS only after running it on the target machine, and record the command output in the canary report (§9).

**Laptop reference.** The development results that the gate reproduces were measured on an RTX 3060 Laptop on 2026-11-13:
* `RIND3D_HANDOFF/VALIDATION_RECORD.md`;
* `RIND3D_HANDOFF/100K_FASTLABEL_INTEGRATION.md`;
* the integration record in `RIND3D_HANDOFF/README.md`.

**What the gate protects.**

| guarantee | depends on | checked in |
|---|---|---|
| bit-exact physics | the certified renderer's FP64 directed-rounding intervals and the exact CPU fallbacks | §2, §4 |
| labels sound and reproducible | FastLabel's exact GPU verifier (FP32 filter → FP64 intervals → GMP) and keyed RNG streams | §3, §4, §5 |
| the declared population | κ draw, natural / coverage streams, complete rate-stream histograms | §7 |
| no silent data loss | fsync'd manifests, journals, consolidated-label sha256 | §8 |

## 1. Environment (PENDING)

```bash
bash RIND3D_Dataset100K/handoff/env/setup_env.sh && source r3d_env.sh       # or the Apptainer / Docker image
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
python RIND3D_Dataset100K/scripts/probe_hardware.py --out probe.json --bench --paths <data>,<scratch> --io-mb 2048
#   NVRTC vs driver; --bench: per-GPU FP64 / FP32 throughput and the certified renderer; storage fsync throughput
```

Pass iff:
* all 16 GPUs are visible;
* NVRTC compiles for the device architecture (sm_90);
* the driver supports the NVRTC version;
* FP64 throughput is recorded (H20 FP64 is reported low, and the renderer and the verifier rely on FP64 directed rounding).

## 2. Regression suites on the target (PENDING)

```bash
OMP_NUM_THREADS=1 python -m pytest -q RIND3D_Generator/tests        # geometry, CPU/CUDA certified rendering, RINDPack, counts
python -m pytest -q RIND3D_Posterior/tests                          # tau-pencil digests (CPU and GPU pass), FastLabel core, Gate A smoke
python -m pytest -q RIND3D_Dataset1K/tests                          # 1K loader, FastLabel 1K entry points (needs the 1K release; skips otherwise)
python -m pytest -q RIND3D_Dataset100K/tests                        # sampling, weights, storage, loader, validator, v2 fixture reproduction (GPU)
```

* **Pass criterion:** every test passes. A test may be skipped only for missing 1K data.
* **Directed rounding.** The Generator's CPU/CUDA equivalence tests and the FP64-forced Gate A build (§3) together check the directed-rounding semantics on the new device.

## 3. Gate A: exact GPU verifier against the certified renderer (PENDING, 3 builds)

**Windows:**
* every window of the four v2 fixture scenes (written by the smoke test to `<work>/fixtures_v2`);
* 30 canary windows stratified by side and response type, after §6.

```bash
W=<work>/fixtures_v2
SPECS=$(python - <<'EOF'
import json; r = json.load(open("RIND3D_Dataset100K/handoff/fixtures/fastlabel_v2/INDEX.json"))["labels"]
print(" ".join(f"100k:$W/slot_{x['slot']:06d}.npz:{x['window']}:{x['source']}" for x in r))
EOF
)
python -m rind3d_posterior.fastlabel.gate_a gateA_default.jsonl $SPECS
FASTLABEL_VERIFY_DEFINES="E32=1e-3f"   python -m rind3d_posterior.fastlabel.gate_a gateA_fp64.jsonl $SPECS   # forces the FP64 interval path
FASTLABEL_VERIFY_DEFINES="UNK_BAND=2e-4f" python -m rind3d_posterior.fastlabel.gate_a gateA_gmp.jsonl $SPECS   # forces the native GMP path
```

Pass iff every record has `field_diff = batch_disagree = bad_violation = 0` (exit code 0), in all three builds.

**Laptop references:**
* research Gate A: 27 windows, 2,684 / 904 / 896 legal candidates, 0 differences;
* production port: 188/188 fields on two 100K windows.

## 4. Cross-device bit-exactness (PENDING)

```bash
python RIND3D_Dataset100K/scripts/smoke_test.py --work <work> --gpu 0      # repeat with --gpu 15 (a different device)
```

Pass iff `SMOKE PASS`, with all of the following:

| step | content | pass criterion |
|---|---|---|
| 3 | TPE label fixtures on CPU and GPU | digests equal to the 1K references |
| 4 | v1 physical fixtures | windows, crop bits and Λ̂ equal |
| 4b | v2 physical fixtures | windows with streams, crop bits, rate-stream m histograms, P̂_A, Λ̂_cov and scene ids equal |
| 4c | FastLabel v2 fixture labels | cell masses, status and tv_bound digests equal. Keyed integer RNG streams make any difference a numeric divergence of the device |
| 4d | Gate A smoke | passes |
| 5 | mini node run (contract v2, FastLabel) | validation and label validation pass |
| 6 | journal recovery | passes |

Run it on at least two devices of the node.

## 5. FastLabel against the exact tau-pencil reference (PENDING; after §6)

```bash
python RIND3D_Dataset100K/scripts/compare_engines.py <canary_root> --n 240 --max-L 128 --workers 64
# plus a small L256 sample (minutes per exact label):
python RIND3D_Dataset100K/scripts/compare_engines.py <canary_root> --n 24 --max-L 256 --seed 1 --workers 24 --out ENGINES_L256.json
```

**Strata:** (side, status), so solid-separated channels are included in proportion.

Pass iff every task has TV(q_FastLabel, q_exact) ≤ tv_bound(FastLabel) + tv_bound(exact) (exit code 0), with the maximum and the quantiles reported.

**References:**

| run | tasks | max TV |
|---|---|---|
| 1K | 447 | 0.0164 |
| laptop 100K, 26 tasks of 8 v2 scenes | 26 | 0.0067 |
| laptop 100K, slot 0, all 32 channels | 32 | 0.0147 |

## 6. Canary of ≥ 256 scenes under the provisional contract v2 (PENDING)

```bash
python RIND3D_Dataset100K/scripts/run_node.py --root <canary_root> --slots 0:256 --gpus all \
       --label-cpu auto --rss-gb 3 --progress-s 60          # contract v2 and FastLabel are the defaults
python RIND3D_Dataset100K/scripts/verify_root.py <canary_root> --workers 64 --labels --physics-every 8 --deep-every 32
python RIND3D_Dataset100K/scripts/coverage_report.py <canary_root>
python RIND3D_Dataset100K/scripts/report.py <canary_root>         # throughput and timing (existing report tool)
```

**Pass criteria:**
* `verify_root` ok: every accepted scene valid; physics spot checks 0 mismatches; labels valid (sha256, certificates, true source in support, n_tasks equal to the contract); no duplicate accepted slot.
* Label failures (timeout, memory, engine) below 0.5% of informative channels, each one in `labels/failures.jsonl`, none silently missing (`coverage_report.label_statuses`).
* `kappa`: the realised k histogram is consistent with (0.15, 0.15, 0.15, 0.55) (χ², p > 0.01).
* Rejected attempts (`quota_unfilled`) below 2% of slots; their reasons are reported.

## 7. Complexity coverage and training ESS (PENDING; from §6)

`coverage_report.py` writes `COVERAGE.json`:
* per (L, m): windows, channels and independent scenes;
* the minimum number of independent scenes with m = 4 per family;
* the stream shares;
* T* weights per side: Kish ESS fraction, max / median weight, the coverage share of the weight mass.

**Compare against the E19 predictions** (`RIND3D_StudentResearch/J_COMPLEXITY_COVERAGE.md` §3, κ with ≥ 50% K = 4, 3+1 at L64):
* L64 m = 4 window share near the prediction;
* T* ESS fraction at L64 ≥ 0.8;
* max weight ≤ 1.4× natural;
* coverage weight share at L64 ≈ 15%.

**Decision for the operator / PI:**
* Freeze κ, or change it and bump the contract id.
* Keep, drop or retune the L64 coverage slot.
* Confirm the absolute coverage targets of `J_COMPLEXITY_COVERAGE.md` §2, scaled to 100K.

**The provisional default must not be treated as the final population until this review is signed off.**

## 8. Crash, restart and data integrity (PENDING)

1. During §6, `kill -9` the coordinator once while scenes and labels are in flight, then rerun the same `run_node.py` command.
   * Pass iff the run completes without manual repair.
   * `verify_root --labels` must be ok, with no duplicate accepted slot. A duplicate journal record of one key is allowed: consolidation keeps the first.
2. `touch <canary_root>/STOP` during a run. Pass iff the coordinator drains within its grace period and a rerun resumes.
3. Corrupt a copy of one consolidated label file (flip one byte). Pass iff `verify_root --labels` reports a sha256 mismatch.
4. Append a torn line to a copy of `manifest/physical.jsonl`. Pass iff the coordinator resumes.

**Laptop reference:** steps 1, 3 and 4 passed on scratch roots. After a kill at 2 scenes / 31 labels, the resume completed 8 scenes with 203 labels; `verify_root` was clean; the sha256 mismatch was detected.

## 9. Throughput, memory and the go / no-go (PENDING)

**From §6** (`logs/progress.jsonl`, `report.py`, `nvidia-smi dmon`):
* scenes per hour and labels per second per GPU, against FastLabel workers per GPU (try 4 / 8 / 12; one GPU semaphore per device);
* GPU memory per worker;
* RSS per worker;
* the share of time in solid-separated channels;
* I/O rate.

**Projection.** Project the 100K wall time from steady state: `python scripts/report.py <canary_root> --costs costs.jsonl`, then `python scripts/estimate.py --report <canary_root>/REPORT.json --cost-samples costs.jsonl --workers <label workers> --gpus 16`. For FastLabel the 1K laptop rate was 10.2 labels/s with 15 workers on one GPU. ≈ 2.7–3.0 M labels are expected under κ = (0.15, 0.15, 0.15, 0.55).

**Canary report.** Write `<canary_root>/CANARY_REPORT.md` with:
* every command of §1–§9;
* PASS / FAIL with the output excerpts;
* the git commit;
* the contract digest (`coordinator.contract_digest`).

**Go** iff every section passes and §7 is signed off. Production then runs under the frozen contract on a fresh root.

| § | check | result |
|---|---|---|
| 1 | environment, NVRTC, FP64 probe | PENDING |
| 2 | regression suites | PENDING |
| 3 | Gate A × 3 builds | PENDING |
| 4 | cross-device bit-exactness (2 devices) | PENDING |
| 5 | FastLabel vs tau-pencil | PENDING |
| 6 | canary ≥ 256 scenes, verification | PENDING |
| 7 | coverage, ESS, κ review | PENDING |
| 8 | crash / restart / integrity | PENDING |
| 9 | throughput, memory, projection | PENDING |
