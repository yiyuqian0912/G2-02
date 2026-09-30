# RIND · Phase I Research Workspace

An English-language interface for the Phase I method handbook, team assignments, and proposed evaluation protocol.

## Open

Open `index.html` in a browser. No build step or dependencies are required. For a stable local origin, run:

```sh
python3 -m http.server 8000
```

Then visit http://localhost:8000. Google Fonts are optional; system fonts are used offline.

## Coordinate work

Use **Team & tasks** to assign owners, statuses, target dates, and notes. Expand a workstream to read its deliverables and acceptance criteria. Search by owner, title, or ID and filter by status or assignment.

Changes are saved in browser localStorage. **Export plan** downloads a JSON snapshot including all eight workstreams and their assignments. **Import** replaces current assignments from a valid snapshot; export first if you want to retain the current version. This is a local prototype, without accounts, a shared backend, or automatic multi-user synchronization. Browser storage for file:// pages can vary; use the local server for a consistent origin.

## Research content

The handbook translates and expands the supplied Phase I description. Architecture, cross-entropy training, metric choices, edge-distance conventions, and acceptance criteria are explicitly proposed implementation details. It flags candidate-density weighting, coordinate metadata, and the information gap between a scene-conditioned physical teacher and an observation-only student. No simulation, neural training, or experiment results are implemented here. The overview visualization is conceptual.

Files: `index.html` (shell), `style.css` (responsive design), `app.js` (handbook, workstreams, and local interactions).

## Confirmed Phase I protocol

- Newly generated single-source scenes; fixed intensity value pending.
- Source-driven quadtree on a 1024 × 1024 world: split source-containing nodes, retain source-free leaves, exclude occupied 16 × 16 leaves.
- Student input: response and window (x, y, side); query candidates in world coordinates.
- Team members self-select workstreams. First review: Thursday, October 1, 2026. This is a pilot milestone, not the deadline for all final acceptance criteria.
- Compute resources pending. Local snapshots are manually shared; self-selection is not synchronized between browsers.
- At inspection, this folder contains only the interface files. The RIND Python package, generator, and data have not yet been located here; no data generation has run.
