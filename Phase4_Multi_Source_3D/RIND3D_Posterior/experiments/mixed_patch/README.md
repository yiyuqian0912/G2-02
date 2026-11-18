# Reproducing the mixed-interface pilot

Read the [scientific report](../../docs/ASTRA_MIXED_INTERFACE.md) and [selection protocol](PROTOCOL.md). All programs run on CPU and read the prior canary without changing it. Inputs are the existing `cmp_v3/tasks.json`, 54 archived `labels/*_tp.npz` and the corresponding scene containers. The manifest contains original absolute container paths. On another host, mount the dataset at the same paths, or explicitly rebase those path strings in a copy and record the new manifest hash. Scripts write to the output path provided. Use a new output directory for a rerun.

Commands below start at the repository root. Use Python with the generator's canonical geometry reader (`rind`), NumPy, zstandard, SciPy, Numba and pytest installed. On this workstation the existing `/home/yushu/Projects/RIND/G2-02-3d-100k/.venv/bin/python` provides the dependencies. The PYTHONPATH below selects the algorithms from this checkout.

```bash
cd Phase4_Multi_Source_3D/RIND3D_Posterior/experiments/mixed_patch
export PYTHONPATH="../../src:../../../RIND3D_Generator/src:../../../RIND3D_Dataset1K"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR=/tmp/rind3d_astra_inverse_v1_numba
py=/home/yushu/Projects/RIND/G2-02-3d-100k/.venv/bin/python
cmp=/home/yushu/Projects/RIND/RIND3D_100K_v3_work/cmp_v3
run_dir=$(mktemp -d)
"$py" -m pytest -q -p no:cacheprovider test_certificates.py test_whole.py ../../tests/test_localfirst.py ../../tests/test_regression.py -k 'not gpu'
"$py" pilot.py "$cmp" "$run_dir/pilot.json"
"$py" joint.py "$cmp" "$run_dir/pilot.json" "$run_dir/joint.json"
"$py" audit_cell_volumes.py "$cmp" "$run_dir/capacity.json"
"$py" whole.py "$cmp" "$run_dir/whole.json"
```

`pilot_v1.json` records every selected block, cell and outcome, exact band endpoints, extremal receiver coordinates in units of 1/(2N), geometry digests, timing and cross-check counts. These witnesses are observed receiver positions. `joint_v1.json` records the intersection volumes including unresolved cases, where missing constraints leave an outer bound. `capacity_v1.json` enumerates the historical lower-volume capacity violations with label file SHA256 and exact excess relative to one grid cell. `whole_v1.json` preserves the initial timing; `whole_paired_v1.json` adds the five matched repeats. Wall times vary. Exact fractions and test identities are reproducible.

Validated on x86-64 Linux (Ryzen 7 5800H), CPython 3.12, one thread. Existing CPU reference digest and Local-first tests run as part of the command above. GPU tests are outside this CPU experiment.
