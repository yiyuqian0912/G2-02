# Reproducing RIND3D 1K: tests, canaries, labels, validation, resume, loading

## Environments

Two environments are used:

* **Producer.** Certified geometry, rendering and fields, with the rind3d natives built with GMP: `RIND3D_Generator/.venv`. Run producer commands from `RIND3D_Generator/` with `PYTHONPATH=../RIND3D_Dataset1K`.
* **Labeller.** Python ≥ 3.11 with numpy ≥ 2, numba ≥ 0.60, zstandard, and `cupy-cuda12x` for the GPU engine. Set `PYTHONPATH=RIND3D_Posterior/src:RIND3D_Dataset1K`.

Recommended variables:

    export PYTHONDONTWRITEBYTECODE=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=/tmp/numba_rind3d

## 1. Core geometry and engine tests

    cd RIND3D_Posterior && PYTHONPATH=src python -m pytest tests

These are the bit-exact regression tests of the CPU reference and the GPU pass. The Phase-1 tau-lemma tests and the engine studies are archived in `RIND3D_Research/posterior_labels/`.

## 2. A small canary at 128³ (about 1 minute physical, a few minutes of labels)

    # producer env
    cat > /tmp/c128.json <<EOF
    {"seed": 20261112, "resolution": 128, "cap": {"64": 4000, "128": 4000, "256": 4000}, "rate_n": 1024,
     "c_table": {"64": [1,1,1,1], "128": [1,1,1,1], "256": [1,1,1,1]}, "k_cycle": [1,2,3,4], "max_attempts": 2}
    EOF
    python ../RIND3D_Dataset1K/scripts/run_physical.py /tmp/c128.json /data/c128 0 12
    # labeller env
    python RIND3D_Dataset1K/scripts/run_labels.py /data/c128 16 0 --kw '{"all": {"nq_cap": 33554432, "ne_cap": 67108864}}'

Production uses `"controller": true` instead of `k_cycle` / `c_table`: see `contract/RIND3D_1K_CONTRACT_v1.json`, `"resolution": 1024`.

## 3. Labeller on one window

    import rind3d_posterior as RP
    rec = RP.label_window(P, K, cls, origin, N, engine="gpu")   # or "cpu"
    cells, q, tv = RP.point_estimate(rec)

## 4. Validate a shard or campaign

    # producer env (structural checks + physics spot checks against the canonical certified predicates)
    python -c "from rind3d_1k.validate import validate_scene, physics_spot_check; ..."
    python RIND3D_Dataset1K/scripts/validate_release.py /data/release [--physics]

## 5. Resume interrupted production

Both runners are idempotent:
* `run_physical.py` skips slots that already have an accepted or final record in `manifest.jsonl`. It rebuilds the controller state deterministically from accepted scenes in slot order. A dead attempt's stage directory is discarded; scenes are only ever published by an atomic rename.
* `run_labels.py` skips existing task files, which are written atomically.

To resume, just rerun the same command.

## 6. Load and flatten

    from rind3d_1k.loader import RIND3D1K, flatten
    ds = RIND3D1K("/data/release", split="train")
    rec = ds[0]       # window 0 of scene 0: solid [L,L,L], responses [k,L,L,L], posteriors[k]
    for ex in flatten(ds, population="Tstar")    # population is required: "Tstar" | "Tstar_mis" | "B1K" (see C_LOADER.md):
        ...           # single-source examples (solid, response_j, label_j)

## 7. Statistics

`scripts/release_stats.py <root>` reports:
* the m histogram (raw and ω-weighted), by side, by k and by family;
* overlap statistics;
* certification coverage;
* rejection rates;
* runtime and storage.
