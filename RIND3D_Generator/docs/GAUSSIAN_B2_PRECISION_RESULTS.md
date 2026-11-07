# B2 interval-local precision results

The existing independent implementation D is the baseline, with original
256-bit A and rational/Arb as controls. This study measures E, which applies
256-bit arithmetic only after the same interval's 64-bit natural enclosure
is inconclusive, and F, which also reuses the field's fixed-source offset
enclosures and constructs the line vector directly. The
[pre-implementation audit and proof](GAUSSIAN_B2_PRECISION.md) describe
which operations are removed and how closed coverage is preserved.

## What D repeats

D already keeps the unresolved dyadic queue and all completed FREE proofs
through its normal escalation. It rebuilds the 256-bit affine line
enclosures for every kernel. It computes the same source offsets again and
copies each freshly built line into its vector. Exact rational inverse
construction belongs to geometry setup. The normal transition has neither
a second rational inversion nor a repeated certified interval.

The full-field counts and stratified timers put 256-bit line preparation
at about 4.20 s across the four 64³ fields and 12.27 s across the two
128³ fields. The latter include roughly 5.33 s of offset work and 0.78 s
of line copying. The corresponding complete D fields took 82.87 and
247.34 s. Exp endpoints consume much more scalar time than the line
preparation. On 3,745 sampled high-stage intervals, the original natural
decision and its 64-bit counterpart agree, including 1,810 FREE proofs.

## Ray gate

All three methods give the same certified labels on 1,398 distinct
frozen rays. Times sum the per-ray median of three rotated trials on
the same AMD Ryzen 7 5800H, Python 3.12.14, GCC 16.2.1 and directed
MPFR settings. The new results use the same canonical scene and source
hashes, region policy, cover depth and scalar proof budgets.

| Population | Queries | A s | D s | E s | E reduction versus D |
|---|---:|---:|---:|---:|---:|
| Cost-stratified | 374 | 2.5764 | 0.5937 | 0.4693 | 20.9% |
| Expensive MISS subset | 96 | 1.7733 | 0.2907 | 0.2281 | 21.6% |
| Cost-unselected residual | 512 | 0.9446 | 0.4402 | 0.3575 | 18.8% |
| Ordinary FREE | 512 | 0.2015 | 0.1233 | 0.1149 | 6.9% |

The expensive subset overlaps the cost-stratified corpus. The ordinary
ray improvement is smaller. F's additional reuse is scoped to a field,
so the scalar timing measures E without charging a source cache for each
one-off query.

The 17 numerical adversaries preserve the expected certified labels.
E retains UNKNOWN on the same four controls as D. They are the exact
transcendental contact, quartic degenerate contact, subnormal tail
upper bound at the threshold, and difficult subnormal receiver endpoint.
The strict near-contact and one-ULP tests resolve on the correct sides.
Arb retains its separate budget-exhaustion status on unresolved controls.

## Complete 32³ and 64³ fields

Each row sums cold seconds across a group. Cold time includes source
clearance, construction, traversal, array materialization and writing.
The matched records also retain CPU, native wall, fresh-process wall and
peak RSS. The frozen and existing holdout A/P controls use their preserved
measurements on the same host and exact input identities. Their 32³ D
measurements also come from the previous experiment. All 64³ D/E/F and
fresh-seed A/P/D/E/F rows are current measurements.

| Grid and group | Supports | A s | D s | E s | F s | D → F | P+D → P+F |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32³ frozen | 4 | 36.44 | 27.51 | 25.17 | 24.63 | 10.4% | 8.1% |
| 32³ existing holdout | 4 | 52.73 | 39.69 | 36.76 | 36.19 | 8.8% | 6.9% |
| 32³ fresh seeds | 2 | 12.05 | 8.73 | 8.18 | 8.01 | 8.2% | 6.4% |
| 64³ frozen | 4 | 148.87 | 80.84 | 71.18 | 68.23 | 15.6% | 12.0% |
| 64³ existing holdout | 4 | 206.69 | 110.15 | 98.21 | 94.37 | 14.3% | 11.2% |
| 64³ fresh seeds | 2 | 53.40 | 29.93 | 26.17 | 25.01 | 16.5% | 12.6% |

The first scene-21 E and F timings overlapped a brief compilation.
An uncontended repeat of its full D/E/F comparison supplies the table
and agrees on both physical fields. The first records remain in the
machine-readable evidence. This also provides a check on timing noise.

All response and solid bits agree with A and D, and all voxels are valid.
D/E/F agree on the independent evaluation-path mask and the region
cover counts. The four previous holdout scenes cover 10–19 kernels and
different clustered, branching, elongated and multiscale layouts. The
new recipe uses a disjoint predeclared seed and includes both accepted
attempts. Scene 72 is clustered with eight kernels and about 2.50%
sampled solid occupancy. Scene 75 is elongated with 13 kernels and
about 0.18%. Scene selection and source clearance precede the timings.

### What work changed

| Complete 64³ frozen workload | D | E | F |
|---|---:|---:|---:|
| Scalar MISS s | 43.89 | 34.95 | 32.98 |
| Scalar HIT s | 11.80 | 11.07 | 9.96 |
| Total scalar interval visits | 709,550 | 709,550 | 709,550 |
| Interval visits with a high-precision stage | 204,632 | 99,938 | 99,938 |
| Kernel line preparations at all precisions | 1,751,863 | 1,114,297 | 1,114,297 |
| 256-bit kernel line rebuilds | 637,566 | 0 | 0 |
| Directed exp endpoints including cover and region | 30,002,016 | 30,002,016 | 30,002,016 |
| 256-bit scalar exp endpoints | approx. 6,549,759 | 2,158,998 | 2,158,998 |
| Peak process RSS MiB | 139.02 | 139.70 | 138.91 |

The D high-precision exp entry is a stratified estimate from the timed
baseline audit. The E/F entries are direct counters. A low and high
stage can belong to the same interval visit. The identical total visit
and exp counts show a shift toward cheaper arithmetic and fewer
preparations. They also exclude a saving from the cover choosing
different receivers. Across all ten 64³ supports E/F avoid 1,705,520
high-precision kernel line preparations. They retain 374,287
low-precision FREE intervals before the first high stage and reuse
3,770,643 Gaussian term enclosures in signed interval tests. The
maximum observed pending queue length is 14. F prepares at most
5,040 accounted source-cache bytes per field. The four frozen source
preparations total 0.178 ms. The largest measured E/F 64³ process
RSS is about 154 MiB, close to D's 153 MiB.

F's additional reduction is about 4–5% of E's 64³ time. This includes
both scoped offset reuse and direct vector construction. The compact
source cache is paid once and charged in cold time. The proof queue
records the same exact dyadic coordinates throughout. There were no
final original-A fallbacks or high-line rebuilds on these complete
fields.

## Bounded 128³ repeat

Scene 14 completed before scene 7. All five current D/E/F runs and the
preserved A/P/D controls agree on all 2,097,152 response and solid bits
per support. The scene-14 D time was remeasured. Scene 7 uses its most
recent same-host D control. The scalar receiver paths, total ray visits
and total exp endpoint counts agree within each scene.

| Case | A s | P s | D s | E s | F s | D → F | P+D → P+F |
|---|---:|---:|---:|---:|---:|---:|---:|
| 14 | 184.03 | 33.36 | 84.98 | 73.53 | 70.62 | 16.9% | 12.1% |
| 7 | 336.89 | 56.94 | 159.45 | 131.79 | 128.28 | 19.6% | 14.4% |

The independent aggregate falls from 244.43 to 198.90 s, down 18.6%.
Adding primary yields 334.73 → 289.20 s, down 13.6%. This is a meaningful
additional benefit over D. It falls below the 20% complete-field
aspiration. Peak F RSS is 199.07 MiB. Case 14's scalar MISS/HIT time
goes from 58.79/8.00 to 45.50/7.09 s, and case 7's goes from
116.34/23.00 to 88.82/19.83 s.

D rebuilds 1,877,038 high-precision kernel lines in these two supports.
E/F rebuild zero. D visits 710,106 nodes at the high stage and E/F
attempt a high stage at 389,875 unresolved intervals. The measured
E/F scalar high-precision exp total is 10.97 million endpoints. The D
audit estimates 30.83 million. Actual total exp endpoints, including
region work, are 110.31 million in both methods. The improvement is a
precision and preparation change on the same proof support.

F grows from 8.98 to 17.83 to 70.62 s across 32³, 64³ and 128³ for case
14. Case 7 grows from 7.68 to 23.85 to 128.28 s. The last doubling of
side costs about 4.0× and 5.4×, respectively. The fixed cover cost and
changing residual ray population shape these finite ratios. The two
128³ improvements exceed the 10% admission requirement for a separate
256-grid sample, with process peaks below 256 MiB. A full 256³ attempt
has a further resource gate based on the sampled grid itself.

## Source-bound tile reuse

A session owns one backend, exact source, options and global sampling
grid. It checks source clearance once with that backend. The cover is
built by the same independent or primary box predicates used in a cold
run. Each backend has its own session. It can retain its continuous
FREE/SOLID boxes and the projection planes constructed for that exact
source. Receiver coordinates and local proofs are evaluated on every
tile. The experiment serializes access to a session's mutable projection
cache.

Case 14 uses three adjacent tile pairs with unaligned origins in the
1024 grid, totaling 196,608 voxels. Both backends compute every tile under
each policy and agree on response and solid bits.

| Backend | Cold rebuild each tile s | Reuse engine and clearance s | Reuse continuous cover s |
|---|---:|---:|---:|
| Primary P | 11.958 | 11.957 | 1.996 |
| Independent F | 42.997 | 42.966 | 7.758 |

The one-time preparation is included. For F, separate cold cover
constructions cost 42.98 s, versus 7.76 s in the reused session.
The independently constructed source projection count falls from 18
to 9. The primary and independent cover payloads are about 15.1 MiB
each. The sequential tile experiment's cumulative process peak is
138.09 MiB. Geometry and clearance reuse alone has little measured
wall impact in this control.

Every voxel in these particular tiles is region-certified. The speedup
therefore measures fixed preparation savings on this support. A separate
test compares three unaligned 8³ crops in the global 64 grid against the
complete 64³ field, for both backends. All 3,072 crop/backend values agree
on both physical masks and validity. The 1024-grid experiment evaluates
only the selected tiles.

## 256-grid resource gate

The follow-up fixes case 14 and divides the global 256 grid into 64
disjoint 64³ tiles. It chooses two tiles in each of eight octants before
seeing timing. Primary and independent sessions evaluate the same 16
probe tiles, including scalar residuals. The planned caps are 360 s
for P, 900 s for F and 512 MiB process RSS. Admission uses the slower
of each octant's two measured tiles, weighted by the octant size, plus
20% and the measured preparation charges. This is a conservative
planning estimate based on the sample.

| Backend | Probe wall s | Probe CPU s | Peak RSS MiB | Slowest tile s | Mean-based projection s | Guarded projection s | Cap s |
|---|---:|---:|---:|---:|---:|---:|---:|
| P | 66.15 | 66.14 | 129.84 | 23.55 | 260.56 | 518.94 | 360 |
| F | 128.53 | 128.51 | 134.84 | 37.53 | 499.87 | 918.26 | 900 |

The guarded projections exceed both caps. The work stops at the exact
sampled coverage of **4,194,304 voxels, 25% of the 256³ grid**. Both
backends independently certify every sampled voxel and agree on
every response and solid bit. Coverage, tile coordinates, validity
arrays and integrity hashes are saved. This is partial field
evidence. The unsampled octant tiles remain unevaluated. The deciding
constraint is time variability. The memory measurements are well below
the declared cap. The mean-based projections alone would hide the
large difference between an entirely region-certified tile and a tile
with substantial scalar work.

The slowest tile for both backends starts at global XYZ (64, 192, 0).
Many probes finish in roughly 4 ms, while this one takes tens of
seconds. F executes 153,793 residual rays over the sample and uses
68.86 million scalar directed exp endpoints. Region certificates fill
4,018,744 voxels. P uses its own cover and scalar ray evaluation.
The sample confirms that the single-cover policy also preserves
independent agreement in tiles that need scalar work. These numbers
do not establish complete 256³ throughput.

## Research default

Use F for complete-field research and E for stand-alone rays. D remains
the frozen performance and scientific control. A and Arb remain
independently available. E reduces expensive MISS time and preserves
ordinary-ray performance. F adds a small, measured preparation gain.
The region-cover reuse policy belongs to a fixed source and backend
session, with its separate preparation cost recorded. The four
adversarial UNKNOWNs remain computational limits. Exact contact
semantics and the production publication gate continue to apply.

The remaining scalar cost is predominantly directed Gaussian exponential
enclosure. Applying the measured D per-precision endpoint costs to the
actual E 64³ endpoint counts estimates roughly 24 s of exp work out of
46 s of E scalar time. That is a cost attribution estimate. The exact
counts show that further offset preparation changes have little room
left. A next bounded test could audit whether a directed low-precision
midpoint filter and reuse of its certified term enclosures can avoid
some of the remaining 256-bit midpoint work. Any such test should
charge the failed filters and keep strict HIT and MISS bounds.

The [machine-readable measurements](../experiments/gaussian_geometry/results/b2-precision-measured-v1.json)
record matched timings, precision work, validity and the 256-grid
admission result. The [acceptance record](../experiments/gaussian_geometry/results/b2-precision-acceptance-v1.json)
records the scientific and preservation checks. The
[reproduction guide](../experiments/gaussian_geometry/b2_precision/README.md)
contains the audit, field and tile commands.

The combined generator and Gaussian regression suite passed **620 tests**
with the existing real CUDA controls enabled. Seven existing checks
skipped because their optional private pilot or campaign artifacts were
absent. SHA256 checks cover the starting source, the Phase 3 worktree,
earlier experiment outputs, all 13,013 RIND3D v0.1 files and both archive
copies.
