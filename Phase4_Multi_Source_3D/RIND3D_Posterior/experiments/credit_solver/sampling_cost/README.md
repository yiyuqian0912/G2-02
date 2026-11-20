# Observation sampling and wide-posterior experiment

[Scientific report and decision](../../../docs/ASTRA_OBSERVATION_SAMPLING_COST.md). All archived input scenes are read-only. Outputs in `results/` contain observations' aggregate counts, window/geometry identifiers, exact proof receipts and measurements. The raw generated source coordinates, original scene containers and the 6.47 MB wide posterior are excluded from Git. The local wide-label file is retained separately and can be reproduced below.

## Inputs and environments

Use a Linux research environment with the repo's RIND3D geometry/reader native extensions, NumPy, SciPy, Numba, zstandard and pytest. The measured runs used the existing v3 generation environment and one low-priority CPU worker per process. Different comparisons sometimes ran on separate cores concurrently. The CPU native certified backend re-rendered the candidate windows. The historical canary timings and the original GPU selection timings remain separate datasets. Wall-budgeted runs may stop on a different valid frontier on a different machine.

From `Phase4_Multi_Source_3D/RIND3D_Posterior/experiments/credit_solver`, set up a fresh output directory and point `V3_WORK` at the archived research workspace containing `sampling_impact/v3`, `release/canary` and `xlf_cbc/v301_canary_10s`. The public v0.1 archive lacks these v3 inputs.

```sh
export V3_WORK=/home/yushu/Projects/RIND/RIND3D_100K_v3_work
export PYTHONPATH="$PWD:$PWD/../../src:$PWD/../../../RIND3D_Generator/src:$PWD/../../../RIND3D_Dataset1K:$PWD/../../../RIND3D_Dataset100K"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export OUT="$(mktemp -d)"
S=sampling_cost
python "$S/population.py" "$V3_WORK" "$OUT/channels.jsonl"
python "$S/geometric.py" "$V3_WORK" "$OUT/channels.jsonl" "$OUT/two_pairs.jsonl"
python "$S/generate_pool.py" "$V3_WORK" "$OUT/prefix.jsonl"
```

The prefix generator replays decisions until the first two natural accepts. It reads sources solely to run the canonical physical generator. The observation-only diagnostic and solver read the geometry and crop without opening the source array. Masks must match across the paired choices of the ordinary policies. `selection.wants_alternative` implements the natural holdback before a second proposal is generated. Its selector accepts only natural windows. Digital component counts are connected components of **occupied coarse bins**. Two-edge volumes retain alternate blocker explanations. No class-ratio or geometric diagnostic can issue a posterior certificate.

## Reproduce the matched costs

Keep the following archived implementations available in independent worktrees if exact source-version replication is required. Set `OLD` and `ACCOUNTING` to the absolute **credit_solver directory** in worktrees at commits `528b060` and `19153b0`, respectively. The scheduling measurements used the scientific code in `ddf2924`. Pass each kernel directory explicitly to the harness. Thresholds, seeds and panel selection appear in the scripts. Slot 1 is an explicitly known stress sentinel. Slots 18–35 are held out from cost fitting.

```sh
python "$S/bench.py" "$OLD" "$V3_WORK" "$OUT/baseline.jsonl" --cases 000004_w00_s0:10 000000_w03_s0:10 000001_w02_s0:45 000001_w02_s0:120
python "$S/bench.py" "$OLD" "$V3_WORK" "$OUT/baseline_second.jsonl" --cases 000023_w07_s0:45
python "$S/bench.py" "$ACCOUNTING" "$V3_WORK" "$OUT/optimized.jsonl" --cases 000004_w00_s0:10 000000_w03_s0:10 000001_w02_s0:45 000001_w02_s0:120 000023_w07_s0:45
python "$S/bench.py" "$PWD" "$V3_WORK" "$OUT/scheduling_fixed_cap.jsonl" --cap 4096 --cases 000001_w02_s0:120 000023_w07_s0:45
python "$S/bench.py" "$PWD" "$V3_WORK" "$OUT/scheduling.jsonl" --cap 16384 --cases 000001_w02_s0:180 000023_w07_s0:120
python "$S/bench.py" "$PWD" "$V3_WORK" "$OUT/tail_extension.jsonl" --cap 32768 --cases 000001_w02_s0:360 --artifacts "$OUT/labels"
python "$S/verify_tail.py" "$V3_WORK" "$OUT/labels/000001_w02_s0.json"
```

The fresh prefix panel includes every eligible channel of both windows before any policy selection. The tail pilot separately modifies only one natural position at a time in the actual four-window quota. Each original window remains available for evaluation. The focal version preserves the rarest original channel if replaced and all newly eligible channels receive actual CBC runs. The strict alternate-mask trial may abstain from replacement. Raw timings were collected for all flags regardless of holdback. The original research receipt used the scene digest as the holdback key. The final selector uses the slot/window identity known before source generation. The summary reports both versions and enumerates all fair holdback assignments on the small measured panel. Both trials are capped at 16 proposals per flagged slot. Their generated crops are temporary. All failed label attempts still count toward measured time.

```sh
python - "$OUT/channels.jsonl" > "$OUT/cases.txt" <<'PY'
import json,sys
sys.path.insert(0,"sampling_cost")
from generate_pool import groups
g=set(groups())
for row in map(json.loads,open(sys.argv[1])):
    if (row["slot"],row["L"]) in g and row["w"]%4<2:
        print(row["key"]+":10")
PY
xargs -a "$OUT/cases.txt" python "$S/bench.py" "$ACCOUNTING" "$V3_WORK" "$OUT/pilot_labels.jsonl" --cases
python "$S/analyze.py" "$OUT" "$V3_WORK" > "$OUT/summary.json"
python "$S/rare_trial.py" "$V3_WORK" "$OUT/channels.jsonl" "$OUT/rare.jsonl"
python "$S/rare_trial.py" "$V3_WORK" "$OUT/channels.jsonl" "$OUT/focal.jsonl" --focal-from "$OUT/rare.jsonl"
python "$S/rare_summary.py" "$OUT/focal.jsonl" > "$OUT/rare_summary.json"
python study/check_exact.py "$V3_WORK" "$OUT/exact.jsonl"
python -m pytest -q .
python -m pytest -q ../global_inverse/test_global.py ../mixed_patch ../xlf_assurance ../../../RIND3D_Dataset100K/tests/test_{interface_v3,sampling,thin_slabs_v3}.py
```

The optional independent GMP/pybind11 reference can be built with `bash native/build.sh "$(command -v python)"` before running the tests. For the descriptive chart use `python sampling_cost/plot.py "$OUT"` in an environment with Matplotlib and NumPy. Verified output hashes, compiler-reference build fingerprints, per-task raw timings and final counts are in the [results directory](results/). Historical statuses in the population table are not reissued as new certificates.

If reproducing a different cutoff or a selected case that never certifies, the durable label file will be absent. The attempt record remains authoritative about the termination reason. The hash of the exact wire can change if a time-limited search reaches a different valid frontier; verify the resulting decoded vector and certificate against their own binding.
