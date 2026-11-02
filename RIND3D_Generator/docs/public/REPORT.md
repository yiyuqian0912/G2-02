# RIND3D-Lite-Core-1K-v0.1 measured report

This report describes the original 1K production campaign. An explicitly named diagnostic subset or future composed release has its own authoritative counts in `manifest.json`. License and citation authors remain pending. This is a locally verified release candidate, not a claim of an online upload or a grant of redistribution rights.

## Actual data

The campaign completed **1,000 READY_LITE scenes with 1,000 distinct canonical geometries and 256,000 informative observation windows**. Each scene stores complete 1024³ binary response and solid fields. There are 85,000 windows of side 64, 85,000 of side 128, and 86,000 of side 256. Windows use arbitrary legal integer origins; no aligned-grid restriction was introduced. Every window excludes the true source, contains both lit and shadowed FREE voxels, and is unique within its scene.

The geometry-group split contains 776 training scenes (198,656 windows), 116 validation scenes (29,696 windows), and 108 test scenes (27,648 windows). IDs and splits remain stable under campaign extension.

| Geometry family | Independent scenes |
| --- | --- |
| barriers | 111 |
| convex_polyhedra | 111 |
| corridors | 111 |
| multiscale_convex | 111 |
| occluder_clusters | 111 |
| prisms | 111 |
| rotated_boxes | 112 |
| thin_slabs | 111 |
| wedges | 111 |

All nine certified convex-union families were retained. Assignment cycles through the nine families, rather than drawing an independent categorical variable for each scene. No geometry was selected according to rendering speed. Sources pass strict FREE-space and clearance acceptance (clearance > 2^-26 in unit-world coordinates). Three source-field attempts had an independently rechecked sufficient proof that their informative-window centers could occupy only one octant, incompatible with the existing spatial-coverage gate. Those sources were resampled while their canonical geometries were retained. Unproved proposal exhaustion is never treated as physical rejection.

Every window retains exactly one fully verified true-source reference in the restricted campaign; public snapshots withhold it. Optional equivalent-source search was zero. One reference never proves uniqueness; finite binary observations can admit continuous sets of equivalent sources. The Lite maximum of eight references and minimum separation 32/1024 remain unchanged for optional future enrichment.

## Realized observation distribution

The frozen mixed policy targets 30% natural, 50% ten-bin coverage and 20% extreme-ratio selection as soft dataset-level objectives. Actual component counts were natural=76,800, uniform=128,000, and extreme=51,200. All selected windows remain subject to exact validity. These selections are distribution-changing; they are not unbiased uniform accepted windows.

`lit_ratio = lit_FREE / (lit_FREE + shadow_FREE)`. Solid voxels never contribute to either FREE class. Ratios exactly 0 and 1 are invalid and absent.

| Lit/FREE interval | Windows | Share |
| --- | --- | --- |
| [0.0, 0.1) | 51,868 | 20.26% |
| [0.1, 0.2) | 17,975 | 7.02% |
| [0.2, 0.3) | 17,975 | 7.02% |
| [0.3, 0.4) | 17,975 | 7.02% |
| [0.4, 0.5) | 17,975 | 7.02% |
| [0.5, 0.6) | 17,974 | 7.02% |
| [0.6, 0.7) | 17,974 | 7.02% |
| [0.7, 0.8) | 17,974 | 7.02% |
| [0.8, 0.9) | 17,974 | 7.02% |
| [0.9, 1.0] | 60,336 | 23.57% |

| Side | Windows | Dark dominated (<0.5) | Light dominated (>=0.5) |
| --- | --- | --- | --- |
| 64 | 85000 | 47320 | 37680 |
| 128 | 85000 | 42138 | 42862 |
| 256 | 86000 | 34310 | 51690 |

Realized median lit ratio is 0.5274. The two extreme bins are intentionally prominent; the policy is not an attempt to make every final bin exactly 10%. Some per-scene/size proposal pools lack bins, especially dark large windows; the immutable production evidence records these shortages and bounded fallback. Available bins are used to reduce the cumulative deficit. No scene is forced to supply physically unavailable observations.

The median per-scene mean pairwise window IoU is 0.00283, P95 0.00882, maximum 0.04270. This is an average over distinct window pairs, not the worst individual overlap. Overlap is allowed. Counts of occupied window-center octants across scenes: `{"2": 5, "3": 3, "4": 15, "5": 14, "6": 47, "7": 90, "8": 826}`. Windows within a scene are correlated and must not be split independently across train/test.

## Nonlocal occlusion diagnostic

A bounded, descriptive audit selected 27 scenes from the first 841 committed scenes, spanning every family. For each side, windows near lit ratios 0.1 and 0.9 were selected, then up to sixteen shadow-FREE receivers were spatially stratified. Exact blocker/counterfactual predicates were evaluated for 1,999 receivers in 162 windows.

| Side | Sampled windows | Exact shadow-FREE samples | External-necessary samples |
| --- | --- | --- | --- |
| 64 | 54 | 643 | 598 |
| 128 | 54 | 666 | 539 |
| 256 | 54 | 690 | 398 |

Of these queried shadow receivers, 1,753 had their first blocker outside the window, 246 inside, and 1,535 passed the stronger external-necessary test. The last number means that removing outside obstruction would change the sampled response under the exact predicate; a long ray or first-outside label alone was not used as causal proof. The mean within-window spatially weighted estimate was 0.745. These are exact point decisions but descriptive sample fractions, not an exhaustive or unbiased population certificate. Private point coordinates and oracle outputs are excluded from the public release.

## Scientific and engineering acceptance

Canonical binary64 continuous geometry, strict closed-segment/tangency semantics, conservative predicates with exact fallback, geometry/source clearance gates and fail-closed UNKNOWN/ABORTED behavior were preserved. Complete fields are certified by the renderer. A parametric-clipping/GMP verifier and an independent rational-shadow-projection verifier rerender both fields for every physical brick intersecting the 256 windows. This complete window coverage is distinct from claiming two CPU replays of every world brick. Checksums are integrity checks, not substitutes for physical verification.

The accepted bounded optimization uses the existing A predicate at subdivision leaf 2 instead of 8. It shares no visibility decisions with B. On ten matched complete end-to-end scenes it improved READY throughput by 1.382×, with identical scientific files. The full scientific regression passed 355 tests, including actual CUDA and difficult contact/thin cases. The original full 4096-window profile and historical READY datasets were preserved.

The 1K campaign used one bounded RTX 3060 Laptop CUDA producer and four single-thread CPU verification workers. Separate bounded process sessions were used, and a real system restart occurred after 832 committed scenes. All 832 earlier sealed scenes were SHA-verified unchanged after recovery. Incomplete field/verification work resumed under its original identity.

Fully finalized sessions measured **332.2 READY scenes/hour**, including preparation, verification, rejections, restarts and catalog work in those sessions. Whole-campaign checkpoint-accounted active time is at least 3.017 hours, and CPU usage at least 8.867 CPU-hours. The abruptly ended session has an unmeasured crash tail: these totals are lower bounds, so a whole-campaign rate computed from them is only an upper bound. Calendar time to the last finalized session was 7.643 hours, including laptop downtime; it is not generation throughput.

| Stage (wall seconds) | P50 | P95 | P99 | Maximum |
| --- | --- | --- | --- | --- |
| geometry_and_source | 0.062 | 0.679 | 1.281 | 2.163 |
| complete_field_and_index | 3.271 | 4.906 | 5.933 | 6.766 |
| window_selection | 6.085 | 9.678 | 17.638 | 60.411 |
| verifier_primary | 15.354 | 29.600 | 43.900 | 54.525 |
| verifier_independent | 3.677 | 5.030 | 6.317 | 7.015 |
| core_gate_and_publication | 1.108 | 1.501 | 2.115 | 3.282 |

Times are recorded stage wall seconds, with overlapping scenes running concurrently. Stage sums are not READY intervals. The median CUDA kernel time was 0.778s, whereas complete field/index time was 3.271s; kernel time alone is not production cost. Peak sampled producer-family PSS was 1.268 GiB, RSS 1.952 GiB, and device temperature 55°C. Peak owned CUDA allocations were 12.13 MiB; device-global memory includes desktop/driver use and is a different quantity.

First accepted fields in separate process sessions had median field/index time 3.277s; subsequent accepted fields had median 3.268s. Median CUDA context/compiler construction metadata, counted once per prepared source-process session, was 0.388s. These are descriptive process-start versus later-field measurements with different geometries, not a matched power-cold/cache-cold benchmark. Sampled mean CPU usage in finalized sessions was 2.95 cores.

The byte-identical public scientific file whitelist totals 1,011,894,718 bytes (0.942 GiB), before small public metadata, SDK and documentation. Full archive bytes and SHA256 are in the supplied archive sidecar and final external acceptance report; they are not embedded recursively in this report. A full RINDPack pair would be 256 MiB uncompressed per scene. No window response is duplicated in storage.

## Reproducibility and limits

Scientific configuration SHA256: `5039276d9d83ce388824cce34e2aa49d427e02d2d262ff7504d3d89096f50fd8`.

Frozen production build identity: `b038ba8842f5ea5e059f42173dbbb840cdef4544e52f02dab3cec82f0d8e19a2`. The manifest carries the complete identity and numerical policy. The private master seed/attempt recipes are retained for authorized reproduction, not distributed as hidden-label recovery instructions.

Public policy `full-field-training-v1` reveals canonical geometry and complete response/solid targets. It withholds serialized sources, candidates, witnesses, seeds and oracle evidence, but it is **not a blind hidden-field benchmark** and cannot prevent source inference from public physics. Default `structure` contains full canonical geometry. Restrict it explicitly when studying local-only model inputs.

Reader-only installation, full/delta integrity, deterministic archive transport, CPU-only reading and multiworker PyTorch loading are documented in the accompanying five documents. Linux x86_64/Python 3.12.14 was tested; the declared Python 3.13 range is not a cross-platform validation claim. No online upload was performed. License and citation remain pending.
