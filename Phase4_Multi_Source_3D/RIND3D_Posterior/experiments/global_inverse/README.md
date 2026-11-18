# Blind global inverse visibility experiment

See the [scientific findings](../../docs/ASTRA_GLOBAL_INVERSE.md), [frozen protocol](PROTOCOL.md), and [per-window comparison](records/comparison.json). This directory contains research code for world-contained convex unions and source-separated binary observations. It reads the v3 canary. It has no production writer.

## Inspect a result

The seven main run files are losslessly compressed JSON Lines. Each record contains the public task key, canonical geometry digest, source-space cut trace, measured work, mass bounds and any proved inner polytopes. Coordinates of cut witnesses are local **receiver** indices. Grid IDs are `(ix*128+iy)*128+iz`. A missing grid ID has zero mass under the emitted estimate. An unresolved case can still put true mass there.

```bash
cd Phase4_Multi_Source_3D/RIND3D_Posterior/experiments/global_inverse
python3 - <<'PY'
from fractions import Fraction
from query import load_records, cell_mass
r = next(r for r in load_records("records/heldout_raster.jsonl.gz") if r["task"]["key"] == "000004_w00_s0")
i = max(r["probabilities"], key=lambda k: Fraction(r["probabilities"][k]))
print("exact occupied cells", len(r["probabilities"]))
print("largest grid probability", i, float(cell_mass(r, int(i))))
PY
(cd records && sha256sum -c SHA256SUMS)
```

The polytope meshes give the exact inner distribution on demand. Files under `records/grids*` are ready-to-use sparse integer weights for all runs meeting the TV target. Divide each weight by the integer denominator. The export's rational error bound is `tv_numerator / tv_denominator`. A result marked `positive` has positive proved volume. A successful TV certificate additionally requires `certified_tv_005`. Exact volume solutions have positive volume and zero unresolved pieces. The dyadic ledger can show a tiny positive rounding allowance even for a solved volume. Explicit rational cell probabilities in such a record are exact. Reported times include checked work until the budget, so a rerun near a deadline can produce different valid bounds.

## Reproduce

Start from the directory above with the repository's canonical geometry reader and native CPU extension, NumPy, SciPy, Numba, zstandard and pytest installed. The first two examples above need only Python's standard library. The canonical decoder used here was checked byte-for-byte against this branch. [Measured environment and native extension hash](records/environment.json)

```bash
export PYTHONPATH="../../src:../../../RIND3D_Generator/src:../../../RIND3D_Dataset1K"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR=/tmp/rind3d_astra_inverse_v1_numba
PY=/home/yushu/Projects/RIND/G2-02-3d-100k/.venv/bin/python
root=/home/yushu/Projects/RIND/RIND3D_100K_v3_work/sampling_impact/v3
comparison=/home/yushu/Projects/RIND/RIND3D_100K_v3_work/cmp_v3
out=$(mktemp -d)
"$PY" verify_inputs.py "$root" records/inputs.json
"$PY" research.py "$root" fresh.json --split holdout --method raster --seconds 2 --output "$out/fresh.jsonl"
"$PY" materialize.py "$out/fresh.jsonl" "$out/grids"
"$PY" check_witnesses.py "$root" "$out/fresh.jsonl" "$out/witnesses.jsonl"
"$PY" check_replay.py records/fresh_raster.jsonl.gz "$out/fresh.jsonl"
"$PY" compare_cpu.py "$root" fresh.json "$out/reference_cpu.jsonl" --all-sizes
"$PY" audit_results.py records "$comparison" "$out/comparison.json"
cmp records/comparison.json "$out/comparison.json"
CUDA_VISIBLE_DEVICES='' "$PY" -m pytest -q -p no:cacheprovider -rs test_global.py ../mixed_patch/test_certificates.py ../mixed_patch/test_whole.py ../../tests -k 'not gpu and not cuda and not gate_a and not release_labels'
```

Replace local data and interpreter paths on other hosts. Writers require new output files. A warm Numba cache was used for the reciprocal comparisons. To distinguish compilation from solving, run one development task first using `tasks.json --split archive --indices 0` and a separate output file.

For the first 27 windows substitute `tasks.json --split holdout`. For all 54 archived windows use `tasks.json --split archive`. The default pairing is `--strategy interface`. The receiver-grouping compiler uses `--method envelope`. Use `--method raster_support` for extra possible-shadow planes. The random-class ablation used `--method envelope --strategy random_classes --indices 0 3 6 9 12 15 18 21 24` on the first holdout. The retrospective cell-mass check uses `--method raster --project`. Only the post-run comparison script accesses archived reference labels.

`check_witnesses.py` chooses a binary64 interior witness derived from each proved inner mesh, checks clearance, and rerenders the complete crop in 64³ blocks with the distinct native parametric-clipping solver. The first diagnostic used a homogeneous-coordinate weighted center that could round onto a face. Equal Euclidean vertex weights passed all 35 checks. The diagnostic emits no source positions. The forward tests establish physical consistency at selected witnesses. The cover and integration proofs carry the completeness claim.

The full disk records include seven main solver arms, the retrospective replay, two native witness-check runs, the matched CPU reference grids, input and replay fingerprints, test output, and a small figure. The figure can be regenerated with `python plot_grids.py records OUTPUT.png` using a Python environment containing Matplotlib and NumPy.
