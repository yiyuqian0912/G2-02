# Planned entry points

These scripts are work items, not existing commands:

- `generate_data.py`: create a new single-source dataset using the RIND generator.
- `make_splits.py`: write disjoint scene-ID manifests.
- `build_teacher.py`: create candidate coordinates, masks, costs, and soft targets.
- `train.py`: train the conditional energy student.
- `evaluate.py`: evaluate on a common grid and export figures and metrics.

Implement thin entry points calling `src/rind_phase1` modules. Reject unresolved required configuration values rather than silently selecting scientific parameters. Do not overwrite existing datasets.
