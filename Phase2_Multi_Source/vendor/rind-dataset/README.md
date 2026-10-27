# RIND reader runtime

This local dependency is installed together with Phase II by uv. It reads schema-v3 data, rerenders its continuous geometry, and includes the data browser. The legacy random-strength quadtree layout and the unit-strength random-window layout are both supported. Generation and release packaging are excluded. The project's optional `browser` extra provides Flask and Pillow; `scripts/browser.sh` installs them automatically.

The reader, geometry, renderer, and browser were copied from the `dataset-g1024-l128-binary` package on 2026-09-30. `runtime.py` preserves its region decoding and rerendering functions; `dataset.py` imports them here instead of importing the generator. `SOURCE_SNAPSHOT.json` records upstream hashes and local adaptation notes. Dependency versions remain locked in the project; this layout adaptation adds no dependencies. The downloaded ZIP does not install or replace Python code.

For `candidate_layout="scene-source-sets"`, ten exact sampled-world source sets are stored once per scene and exposed through the unchanged `get_candidates()` API. Random windows have no quadtree; the reader rejects tree reconstruction for this variant. Browser diagnostics identify variation across free-pixel intensities and retain the actual additive response scale.
