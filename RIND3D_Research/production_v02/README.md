# RIND3D production readiness (contract v0.2 candidate)

This directory takes RIND3D from research prototyping to controlled, reproducible dataset production. It contains:
- a benchmarked backend decision;
- a frozen production contract;
- production-format code: generation, sharding, recovery, validation, loaders, and a Teacher interface;
- a bounded canary;
- a lifecycle cost model.

Nothing here modifies RIND3D v0.1, the producer checkout, the Gaussian studies or the Teacher branch. Nothing has been committed or pushed.

| Deliverable | File |
|---|---|
| Decision report | `RIND3D_PRODUCTION_DECISION.md` |
| Frozen contract (canary) | `RIND3D_PRODUCTION_CONTRACT.json` (copy frozen in `outputs/canary/contract.json`, SHA256 in `outputs/canary/FROZEN.json`) |
| Backend benchmarks | `RIND3D_BACKEND_BENCHMARKS.json` (raw records in `outputs/benchmarks/`) |
| Lifecycle cost model | `RIND3D_LIFECYCLE_COST_MODEL.json` |
| Canary results and gates | `RIND3D_PRODUCTION_CANARY_RESULTS.json` |
| Teacher interface | `DATASET_TEACHER_INTERFACE_PROPOSAL.md` |
| Gaussian bounded investigation | `outputs/gaussian/gaussian-investigation.json`, `configs/gaussian_v3_preregistration.json` |
| Evidence manifest | `EVIDENCE_MANIFEST.json` |

## Environment

All entry points run with the certified producer interpreter of the read-only producer checkout, through `run.sh`:

```bash
cd RIND3D_Production
./run.sh rind3d_production.<module> ...
```

- **Producer checkout:** `/home/yushu/Projects/RIND/G2-02-3d/RIND3D_Generator`, with its built native and CUDA extensions. Its existing scientific certification `outputs/integration/scientific-certification.json` is re-validated against the current scientific identity at campaign creation (`env.certificate()`).
- **Mesh/BVH engines:** Embree 4 and Warp 1.19, used only by the benchmarks. They are appended read-only from the mesh-study environment (`env.mesh_engines()`).

## Reproduce

```bash
# Stage B benchmarks (fresh benchmark-only seed 20261201)
./run.sh rind3d_production.bench_backends 2            # random 64^3 windows + 9 full 1024^3 fields
./run.sh rind3d_production.bench_backends 2 near       # windows centred on geometry
./run.sh rind3d_production.bench_grazing               # silhouette-grazing rays vs certified truth
./run.sh rind3d_production.bench_materialization       # A/B/C/D strategies, Teacher batches
./run.sh rind3d_production.summarize_bench             # -> RIND3D_BACKEND_BENCHMARKS.json

# Stage D canary (contract frozen at create; config configs/canary-v0.2.json)
./run.sh rind3d_production.runner create outputs/canary configs/canary-v0.2.json
./run.sh rind3d_production.runner run outputs/canary --max-runtime 10800   # resumable; SIGTERM pauses safely
outputs/logs/interrupt_test.sh                                            # recovery test: SIGKILL mid-scene, resume
./run.sh rind3d_production.runner status outputs/canary
./run.sh rind3d_production.validate outputs/canary --determinism 3 60     # independent validation (separate process)
./run.sh rind3d_production.usability outputs/canary
RIND3D_TEST_ROOTS=$PWD/outputs/canary ./run.sh pytest -q -p no:cacheprovider tests
./run.sh rind3d_production.canary_analysis              # -> RIND3D_PRODUCTION_CANARY_RESULTS.json
./run.sh rind3d_production.costmodel                    # -> RIND3D_LIFECYCLE_COST_MODEL.json
./run.sh rind3d_production.evidence                     # -> EVIDENCE_MANIFEST.json
```

## Reading data

```python
from rind3d_production.loader import ProductionDataset
ds = ProductionDataset("outputs/canary", split="train")                 # student_local, natural windows
x = ds[0]            # scene_id, window_id, profile, window_origin, window_size, observed_response, observed_solid
ds = ProductionDataset("outputs/canary", subsets=("informative", "extreme", "rare", "nonlocal"), sparse=("uniform", 256))
t = ProductionDataset("outputs/canary", view="teacher", research_access=True)
phys = t.physics(0)  # TeacherPhysics: legal(), visibility(), response(), window(), free_mask(), receiver_bank()
```
