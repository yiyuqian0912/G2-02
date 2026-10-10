# RIND reader runtime

This local dependency is installed together with Phase I by uv. It reads canonical schema-v3 data, rerenders its continuous geometry, and includes the original browser page. Generation and release packaging are excluded. The Phase I project's optional `browser` extra provides Flask and Pillow; `scripts/browser.sh` installs them automatically.

The reader, geometry, renderer, and browser were copied from the local `dataset-g1024-l128-binary` package on 2026-09-30. `runtime.py` preserves its region decoding and rerendering functions; `dataset.py` imports them here instead of importing the generator. `SOURCE_SNAPSHOT.json` records upstream hashes. Update this snapshot and the lockfile together when the runtime changes. The downloaded ZIP does not install or replace Python code.
