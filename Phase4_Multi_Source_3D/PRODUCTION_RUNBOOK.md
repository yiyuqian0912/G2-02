# RIND3D-100K v3 production runbook (operator entry point)

All commands run from `Phase4_Multi_Source_3D/` (the RIND3D root of the G2-02 repository, branch `yushu/3d`). The
contract is `RIND3D_Dataset100K/contract/RIND3D_100K_CONTRACT_v3.json`, explained in `RIND3D_100K_V3_CONTRACT.md`. The
labeller is FastLabel v3.4; its guarantee is in `RIND3D_Posterior/docs/FASTLABEL_V34_CERTIFICATION.md`. Target-machine
gates for the 16 × H20 server are in `RIND3D_Dataset100K/docs/H20_CANARY_RUNBOOK.md`.

## 1. Environment (one per checkout)

```bash
# system: g++ >= 11, cmake >= 3.24, libgmp-dev, libmpfr-dev, libzstd-dev, curl, xz-utils, bsdtar, git, NVIDIA driver
bash RIND3D_Dataset100K/handoff/env/setup_env.sh      # builds .venv + CGAL/Boost natives, writes r3d_env.sh (~2 min)
source r3d_env.sh                                      # before every command below
```

* **Pinned:** Python 3.12, `handoff/env/requirements-lock.txt`, CGAL 6.2.1 and Boost 1.92.0-1 (SHA-256 checked).
* **NVRTC** matches the driver.
* **Offline:** `R3D_DEPS=<dir with CGAL-6.2.1.tar.xz, boost-1.92.0-1-x86_64.pkg.tar.zst>` and
  `R3D_WHEELS=<wheelhouse>`.
* **Process rule:** one rind3d CUDA runtime per process. The coordinator runs physics and FastLabel in separate
  worker processes; do not mix rind3d and CuPy in one process.

## 2. Validation before any run

```bash
python -m pytest -q RIND3D_Dataset100K/tests RIND3D_Dataset1K/tests RIND3D_Posterior/tests
(cd RIND3D_Generator && python -m pytest -q tests)
python RIND3D_Dataset100K/scripts/smoke_test.py --work /scratch/r3d_smoke      # all 11 steps must PASS
```

The smoke test checks:
* imports and the NVRTC/cupy kernels;
* the engine regression suites;
* the label fixtures;
* the v1 physical fixtures;
* v2 physical and FastLabel fixtures (regression);
* **v3 physical and FastLabel fixtures** (production);
* Gate A: the exact GPU verifier against the certified renderer;
* a 2-scene node run with validation;
* journal torn-tail recovery.

## 3. Canary, then production

```bash
ROOT=/data/rind3d_100k_v3                                   # a fresh root per campaign
python RIND3D_Dataset100K/scripts/run_node.py --root $ROOT --slots 0:256 --gpus all --label-cpu auto --rss-gb 3 --progress-s 60
python RIND3D_Dataset100K/scripts/verify_root.py $ROOT --labels --workers 16 --physics-every 8 --deep-every 32
python RIND3D_Dataset100K/scripts/coverage_report.py $ROOT
python RIND3D_Dataset100K/scripts/compare_engines.py $ROOT --n 240 --max-L 128 --workers 16   # FastLabel vs exact tau-pencil
```

* **Defaults:** contract v3 and FastLabel. `run_node` refuses any contract outside the certified profile:
  non-polyhedral (curved) families or a non-certified labeller (`rind3d_100k/profile.py`).
* **Slots** are deterministic task ids. Scene = slot, and labels are keyed `slot:window:source`.
* **Production** is the same command with `--slots 0:100000`, spread over nodes by slot range. It needs separate
  explicit authorization and the H20 gates (`docs/H20_CANARY_RUNBOOK.md` §1–9; κ review in §7).

## 4. Monitoring, interruption, resume

* **Progress:** `$ROOT/logs/progress.jsonl` (scenes/h, labels/s, failures, RSS).
* **Failures:** `$ROOT/labels/failures.jsonl` and `$ROOT/manifest/quarantine.jsonl`.
* **Stop:** `touch $ROOT/STOP`. The coordinator drains within `--stop-grace`; rerun the same command to resume.
* **Crash** (kill, power loss): rerun the same command.
  * Scene containers are atomic (tmp, fsync, rename, directory fsync).
  * Label journals are append-only and checksummed, with torn-tail tolerance; consolidation keeps the first record of
    a key.
  * Completed slots are never recomputed.
* **Timeouts:** 1800 / 3600 / 7200 s for L = 64 / 128 / 256. A timed-out, crashed or failed label is quarantined with
  its reason and never written as certified. `verify_root --labels` checks every consolidated label's sha256 and
  certificate, that the true source lies in the support, and that n_tasks matches the contract.
* **Never re-roll:** never rerun a failed statistical certificate with a different seed; the δ budget does not cover
  re-rolls.

## 5. Dataset validation and loading

```bash
python RIND3D_Dataset100K/scripts/verify_root.py $ROOT --labels
python RIND3D_Dataset100K/scripts/student_smoke.py $ROOT          # loader -> features -> one Student step
```

```python
from rind3d_100k.loader import RIND3D100K, flatten
ds = RIND3D100K(ROOT, split="train")
for ex in flatten(ds, population="Tstar"):      # certified + certified_stat with tv_bound <= 0.05
    ex["target"], ex["status"], ex["certificate"], ex["provenance"]
```

## 6. Laptop release canary (2026-11-18, RTX 3060 Laptop 6 GB, Ryzen 7 5800H, 31 GB)

See `RELEASE_CANARY_2026-11-18.md` for the measured numbers, the crash/resume test, the validation outputs and the
100K cost estimate.
