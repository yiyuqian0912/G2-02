# 06 — All-lit, all-dark and solid-separated responses

## 1. Definitions in 3D [A]

For a channel j and a window W, with FREE(W) ≠ ∅:
* **mixed:** ≥ 1 lit and ≥ 1 dark FREE receiver (the generator's I_j = 1);
* **all-lit / all-dark:** every FREE receiver lit, or every FREE receiver dark (I_j = 0);
* **solid-separated:** mixed, and no lit FREE voxel is 6-adjacent to a dark FREE voxel.

**Lemma.** "No 6-adjacent lit/dark pair" ⟺ every 6-connected component of FREE(W) is monochromatic.
* (⇐) Two 6-adjacent FREE voxels lie in the same component.
* (⇒) A FREE path from a lit to a dark voxel moves between 6-neighbours, so some step changes colour.

So the lit and dark regions are separated by SOLID in the 6-connectivity sense. The 26-connectivity variant is reported
too; it differs in ≤ 3% of the solid-separated cases (E9). A 2D slice can show a separation that does not exist in 3D,
and the converse, so these diagnostics must be computed on the 3D crop.

## 2. What the binary observation says in each case [A]

* **All-lit** ⟹ F = {s ∈ D \ B̄ : no solid meets any segment [s, x], x ∈ FREE(W)}. Only in an obstacle-free world is this D \ B̄, the prior. Every obstacle removes its "shadow-casting" region with respect to the whole window.
* **All-dark** ⟹ F = {s : every segment [s, x] meets a solid}. This is a strongly structured region behind occluders, often adjacent to solid surfaces. Distance plays no role in the binary physics: a receiver is dark only because a segment meets a solid.
* **Solid-separated** ⟹ the lit side and the dark side each impose their constraints. The SOLID layer between them is itself part of the observation, through the SOLID crop and closed-solid visibility.

## 3. Exact physics fixtures (E8; tau-pencil CPU engine vs independent brute-force Monte Carlo)

World (0,1)³ with axis-aligned closed boxes, N = 32, window L = 4 (side ⅛), true source chosen to produce each response
type. The MC uses 4·10⁶ uniform candidates, an independent float64 Liang–Barsky renderer, and exact feasibility over all
64 receivers.

| fixture | type | vol F: engine [lo, hi] | vol F: MC ± s.e. | MC-feasible points outside engine support | TV engine vs MC (8³ blocks) | MC split-half TV (noise) | engine tv_bound | components |
|---|---|---|---|---|---|---|---|---|
| box_mixed | mixed | 0.002822 | 0.002856 ± 0.000027 | 0 | 0.009 | 0.023 | 0 | 1 |
| box_alllit | all-lit | 0.8201 | 0.8202 ± 0.0002 | 0 | 0.005 | 0.009 | 0 | 1 |
| box_alldark | all-dark | **[0.04274, 0.04886]** | 0.04275 ± 0.00010 | 0 | 0.006 | 0.015 | **0.125** | 1 |
| slab_separated | solid-separated | 0.4709 | 0.4707 ± 0.0002 | 0 | 0.005 | 0.010 | 0 | 1 |
| thin_plate_mixed (thickness 2⁻¹³) | mixed | 0.01458 | 0.01455 ± 0.00006 | 0 | 0.010 | 0.020 | 0 | 1 |
| holes_mixed (narrow opening) | mixed | 0.000100 | 0.000103 ± 0.000005 | 0 | 0.035 | 0.063 | 0 | 1 |
| holes_alllit (wall with two holes) | all-lit | 0.6085 | 0.6082 ± 0.0002 | 0 | 0.005 | 0.009 | 0 | 1 |

**Readings** [C]:
* **Generator and inverse solver agree** on every response type. Every engine–MC TV is below the MC split-half noise; volumes agree within 1.3 s.e.; no feasible point falls outside the engine support. The true source lies in the support in all 7 fixtures.
* **All-lit is informative.**
  * box_alllit excludes 17.6% of the prior volume: TV(posterior, prior) = 1 − 0.820/0.995 = 0.18.
  * holes_alllit excludes 38%. Its F is connected but very structured: a half-space plus two narrow cones through the holes, holding 1.2% of the mass beyond the wall.
  * In real windows the all-lit posteriors certified by FastLabel occupy 7–95% of the cube (E11, §4).
* **Solid-separated is informative.** F is essentially the half-space on the lit side of the plate (volume 0.471 of the cube; the lit-side slab y > 0.5287 has volume 0.4713), so about half of the prior is excluded. Generator, engine and verifier handle it correctly:
  * `physical.evaluate` counts it as informative;
  * the engine's lit/dark `boundary_mask` is used only to choose refutation sentinels, never for correctness;
  * FastLabel's `point_items` falls back to interior receivers when no lit/dark boundary exists;
  * the exact verifier tests every FREE voxel.
* **All-dark exposes a certificate weakness (B, C).** The label is correct: its lo-normalised q matches MC within noise. But hi − lo = 0.00612 = exactly the volume of the solid box. In the window-side ("wmode") branch of `engine_p2.stage2`, a source region inside a component is not excluded. A source inside a closed solid sees no receiver, so it is consistent with an all-dark pattern, and its piece is emitted as IN. Only `_shell_volume`'s conservative path then sets its lower bound to 0. The interval [lo, hi] therefore contains the solid interior, and `tv_bound` = 0.125, which is "uncertified" under the 1K status rules. For mixed and all-lit observations the interior is excluded automatically: a lit receiver cannot be lit from inside a solid. This is why no current label is affected.

## 4. Real data

**Frequencies in accepted windows** (E1, E9) [C]:
* solid-separated informative channels: L64 6.7% (2578 of 38,631, all scenes); L128 2.5% (250 scenes); L256 0.39% (40 scenes);
* all-lit channels: 25,795; all-dark channels: 13,443. They are stored but unlabelled, and occur only alongside an informative channel (k ≥ 2);
* informative channels with lit fraction ≥ 0.99 (a few decisive dark voxels): 17.5%; with lit fraction ≤ 0.01: 4.7%.

**FastLabel on uninformative and solid-separated channels** (E11, GPU, train scenes, nothing written): see the table appended in §6.

**Darkness vs distance after marginalising geometry** (E10) [C]:
* P(segment between two random FREE points is blocked | length) increases with length in every family, but the level and shape are family-specific.
* At lengths 1.0–1.75 it ranges from 0.15 (prisms) to 0.75 (multiscale_convex).
* A one-parameter exponential exp(−λℓ) for P(unblocked) misfits by up to 0.126 (multiscale_convex) and 0.065 (barriers).
* In the release, the median source-to-window distance is 0.71 for all-dark channels, 0.62 for informative channels and 0.52 for all-lit channels.
* So after marginalising G, an all-dark observation does carry a statistical distance signal, but it is a property of π(G) (Boolean-model-like clutter), not of the physics, and not a universal exponential law. Under a Poisson Boolean model of convex grains the unblocked probability would be exactly exponential in length [A, derivation standard]. The v0.1 families are not Poisson (fixed counts, anchored and clustered layouts, a 0.01 wall gap), which is consistent with the misfit.

## 5. Inclusion policy (brief §7.5)

**Key principle [A, D].** Thinning that depends only on the Student's own input X leaves P(S | X) unchanged: selection
on x only ([Zadrozny2004]; 03 §1). Response type, lit fraction and the solid-separated flag of the **same** channel are
X-measurable. The informativeness of **other** channels is not.

| option | scientific validity | learning value | cost | shift | certification |
|---|---|---|---|---|---|
| A: exclude from the primary dataset (current) | valid for P(S\|X) **on mixed X**; the Student is undefined or extrapolating on all-lit/all-dark X | none for those X | 0 | test-time shift if deployment includes them | — |
| B: separately labelled supplement | valid if drawn with its own declared X-measurable selection law and labelled for that law | high: all-lit and all-dark are informative (§3) | all-lit labels broad: 10⁵–2·10⁶ cells, seconds to tens of seconds each (E11) | none if strata are reported | all-lit certifies; all-dark needs fix S6 first |
| C: fold into the main sampling law with natural frequencies | valid | dominated by all-lit mass at L64/L128 (natural rates) | large | — | as B |
| D: response-type stratified thinning with declared rates | **valid without importance weights for P(S\|X)**, since the rate depends only on X. Weights by a function of X restore any target P(X) for aggregate metrics | controllable | tunable | explicit | as B |
| using the 1K's existing uninformative channels as single-channel examples | **invalid for N1**: they exist only when another hidden channel is informative (03 §3, M4) | — | free | hidden | — |

**Recommendation (E):** Option D in a new release (100K or a 1K supplement). Draw (window, channel) examples with a
declared thinning rate per (side, response type), never conditioned on other channels. Label with FastLabel after S6
(explicit solid-interior exclusion). Report metrics per stratum, and on the natural mixture by re-weighting with a
function of X only. Do not force equal class counts: equal counts are just one choice of X-measurable rates, harmless for
P(S|X) but arbitrary for aggregate metrics. Keep the 1K release contract unchanged: Option A, with M4 for its
uninformative channels.

## 6. E11 results (FastLabel v3.4 on extreme channels; filled in from out/e11_fastlabel_extreme.jsonl)

Configuration:
* FastLabel v3.4 production defaults: ε = 0.05, the 1K dataset-simultaneous δ, keyed seeds;
* 6 random channels per (side, type) from train-split scenes 10–399, and 6 solid-separated informative channels found by E9;
* 150 s wall cap per task. The first 21 tasks ran alone; the remaining 21 ran 3 in parallel on the same GPU, so their wall times carry contention;
* labels computed in memory and never written. "Label volume" is Σ hi, i.e. vol(B) ≥ vol(F) for statistical labels.

| side | type | n | certified (det.) | certified_stat | uncertified / loose | timeout (150 s) or error | wall s median (max) | label volume range | cells median |
|---|---|---|---|---|---|---|---|---|---|
| 64 | alldark | 6 | 0 | 3 | 2 | 1 | 23.0 (112.9) | 0.191–0.477 | 910240 |
| 64 | alllit | 6 | 0 | 6 | 0 | 0 | 13.1 (15.7) | 0.070–0.835 | 1021980 |
| 64 | separated | 3 | 0 | 3 | 0 | 0 | 3.6 (4.3) | 0.025–0.290 | 341766 |
| 128 | alldark | 6 | 0 | 3 | 1 | 2 | 6.1 (104.6) | 0.105–0.418 | 394161 |
| 128 | alllit | 6 | 0 | 6 | 0 | 0 | 13.3 (24.0) | 0.326–0.947 | 1363338 |
| 128 | separated | 3 | 0 | 3 | 0 | 0 | 7.3 (8.9) | 0.088–0.364 | 291000 |
| 256 | alldark | 6 | 0 | 3 | 0 | 3 | 8.7 (9.3) | 0.023–0.169 | 211705 |
| 256 | alllit | 6 | 0 | 6 | 0 | 0 | 17.7 (28.8) | 0.302–0.614 | 941301 |

Uncertified/loose labels: 000234/w5/s2 L64 alldark tv_bound 0.843; 000014/w2/s3 L64 alldark tv_bound 0.129; 000043/w33/s2 L128 alldark tv_bound 0.722

**Readings** [C]:
* **All-lit: 18/18 certified statistically**, median 13–18 s, max 29 s.
  * Posteriors cover 7–95% of the cube with ~10⁶ cells: broad but far from the prior.
  * One all-lit L64 channel restricts the source to 7% of the cube.
* **Solid-separated informative channels: 6/6 certified**, 3.6–8.9 s. The missing lit/dark boundary in `point_items` does not prevent certification.
* **All-dark: 9/18 certified, 3 uncertified, 6 timed out.**
  * This is consistent with the mechanism of §3: B keeps solid interiors that no receiver constraint can remove. The legality check fails on them, so p = vol(F)/vol(B) cannot reach 1 − ε, and the run falls back to the exact engine, whose bound is loose for the same reason.
  * Certifying all-dark labels requires fix S6 (09): explicit exclusion of solid interiors from B in the wmode path.
* The sample is small (6 per cell): rates are indicative, not estimates.

