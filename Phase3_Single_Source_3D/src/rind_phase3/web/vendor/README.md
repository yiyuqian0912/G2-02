# Display-only dependencies

Three.js **0.170.0 (r170)** and its matching OrbitControls are vendored for offline local viewing. Their upstream MIT license is `THREE-LICENSE.txt`. It applies to these third-party files, not to the RIND3D dataset or unrelated repository software.

Sources, downloaded 2026-11-04:

- https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js
- https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/controls/OrbitControls.js
- https://cdn.jsdelivr.net/npm/three@0.170.0/LICENSE

SHA256:

```text
ce1fa418de16a19495a9f72495580e3015d7745c296d3ce0485897f902ddedfb  three.module.js
80efaadea4f8a636a65fb0bd08bfef62f3d93a0bb94e2e7500f23176c5c07f4e  OrbitControls.js
4c40a1ef62450b857c3b2aaf294936304cd552d965fbcd9d32d4c5bcf4ba4454  THREE-LICENSE.txt
```

Three.js is used only to display geometry and stored field textures. It does not generate scientific visibility, geometry labels, or teacher costs.
