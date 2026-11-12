# Reproducible commands (v0.3 training-ready candidate)

All commands run from `/home/yushu/Projects/RIND/G2-02-gaussian/RIND3D_Training_v03`. There are two interpreters:

- `./run.sh`: the certified producer interpreter (`G2-02-3d/RIND3D_Generator/.venv`, read-only), with `PYTHONPATH` set to this directory plus `RIND3D_Production` (imported read-only) and `PYTHONDONTWRITEBYTECODE=1`. Use it for physics, indexing, export, audits and tests.
- `./run_gpu.sh`: the benchmark-only CUDA venv `.venv-gpu`. It contains numpy, zstandard and torch 2.14.1+cu130, taken from the local uv cache. Use it for Student-side loader and training benchmarks only.

The physics checkout, the v0.2 tree, the accepted canary, the v0.1 campaign and the Phase 3 tree are never written. `outputs/preservation/compare.json` proves this.

```bash
# 0. preservation baseline (SHA256 trees of every read-only input)
./run.sh rind3d_training.preservation snapshot before

# 1. canary: expanded index (P102 prefix == stored v0.2 records), two-view export, leakage audit
./run.sh rind3d_training.index canary --profile P256 --workers 12        # ~11 min, 216 scenes
./run.sh rind3d_training.export canary --name canary-v03 --workers 12    # ~17 min, 55,184 samples
./run.sh rind3d_training.leakage canary-v03

# 2. integration tests (10-step Teacher interop + audits) on the exported files
RIND3D_V03_EXPORT=canary-v03 ./run.sh pytest -q -p no:cacheprovider tests

# 3. study and benchmarks (benchmarks on an otherwise idle machine)
./run.sh rind3d_training.study                                            # 102 vs 256 vs dynamic vs hybrid
./run.sh rind3d_training.bench_fields                                     # v0.2 reader, earlier numbers, dynamic
./run_gpu.sh rind3d_training.bench_student canary-v03                     # student loader, GPU transfer, training
./run_gpu.sh rind3d_training.blocksparse canary-v03                       # L256/L128 lossless representations
./run.sh rind3d_training.interop_audit                                    # v0.2 / Phase 3 / v0.3 Teacher audit

# 4. SB 1K training candidate from v0.1 (no physics regenerated; private index/export keys created in ./private)
./run.sh rind3d_training.index v01 --profile P256 --workers 12
./run.sh rind3d_training.export v01 --name v01-sb-v03 --workers 12
./run.sh rind3d_training.leakage v01-sb-v03
./run_gpu.sh rind3d_training.bench_student v01-sb-v03

# 5. reports, evidence, review archive
./run.sh rind3d_training.scale_cost --export canary-v03 --bench v01-sb-v03
./run.sh rind3d_training.report
./run.sh rind3d_training.preservation snapshot after && ./run.sh rind3d_training.preservation compare
./run.sh rind3d_training.evidence --archive outputs/RIND3D_TRAINING_READY_REVIEW.zip
```

**Private material.** `./private/*.secret` holds the index, export and dynamic keys (mode 0600, directory 0700). The privileged views live under `outputs/privileged` (mode 0700). Neither is ever placed in the review archive.

**Determinism.** Rebuilding from the same keys reproduces the outputs exactly:
- index records, the P102 prefix and the weights;
- sample ids and shard order;
- sampler epoch plans.
