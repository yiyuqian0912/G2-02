# Codex Implementation Instructions — G2-02 Phase I Student Model and Training

## 1. Task Scope

Implement the **Phase I student model and training pipeline** for the G2-02 RIND project.

This task covers only the student-side conditional energy model and its training/validation utilities.

Do **not** implement scene generation, obstacle geometry, physical rerendering, teacher candidate search, or teacher physical-cost computation. Those are upstream modules owned by other team members.

The student-side pipeline must consume teacher records produced upstream and learn a probability distribution over candidate source locations.

The required high-level interface is:

\[
(\text{response image}, \text{candidate source coordinates})
\rightarrow
\text{candidate energies}
\rightarrow
\text{student source probability distribution}.
\]

The teacher provides a physics-derived target distribution over the same candidate set.

The model is trained to match that teacher distribution.

## 2. Scientific Objective

The model receives only:

1. a fixed-size 2D response/shadow map;
2. candidate source coordinates.

It must **not** receive:

- obstacle geometry;
- obstacle mask;
- shadow-edge map;
- source labels as a direct regression target;
- first-blocker identity;
- tree depth;
- original quadtree leaf size;
- source distance metadata;
- analytic shadow-line constraints.

The intended experiment is to test whether a neural model can recover the geometry of the inverse RIND problem from simple physical supervision.

A single observed shadow boundary may correspond to an extended family of physically valid source locations. Therefore the model must predict a distribution, not a single coordinate.

The model should be able to represent:

- one localized mode;
- multiple modes;
- extended ridge-like feasible regions.

Direct coordinate regression is explicitly excluded.

## 3. Fixed MVP Assumptions

Assume the first-stage dataset contains:

- 2D continuous scene geometry;
- exactly one point source per sample;
- fixed source intensity;
- hard line-of-sight visibility;
- no distance attenuation;
- no reflections;
- no indirect illumination;
- no noise;
- no material variation;
- fixed-size response windows;
- source outside the observation window;
- train/validation/test splits made by `scene_id`.

The model code must remain agnostic to hidden geometry.

## 4. Required Files

Implement the student work primarily in:

```text
model.py
train.py
```

Additional helper files are allowed if they make the code cleaner, but the public interfaces below must remain simple and stable.

Suggested organization:

```text
model.py
    FourierXY
    ObservationEncoder
    CandidateAttention
    SourceEnergyField

train.py
    build_model
    normalize_teacher_distribution
    teacher_student_loss
    train_step
    train_epoch
    validate
    save_checkpoint
    load_checkpoint
    evaluate_source_grid
```

If a config system already exists in the repository, reuse it rather than creating a competing system.

## 5. Teacher Record Contract

The upstream teacher/search modules should provide records equivalent to:

```python
record = {
    "response": ...,          # FloatTensor [1, H, W]
    "window": ...,            # metadata used only for preprocessing/normalization
    "candidate_xy": ...,      # FloatTensor [K, 2]
    "physical_cost": ...,     # FloatTensor [K]
    "teacher_prob": ...,      # FloatTensor [K], optional if physical_cost is present
    "area_weight": ...,       # FloatTensor [K], optional but strongly preferred
    "valid_mask": ...,        # BoolTensor [K], optional
    "scene_id": ...,
    "view_id": ...,
}
```

The student code must support both cases:

1. `teacher_prob` already supplied;
2. only `physical_cost` supplied, in which case the student-side loader may construct the teacher distribution.

Candidate ordering must be consistent across:

```text
candidate_xy
physical_cost
teacher_prob
area_weight
valid_mask
```

Do not reorder one field independently.

## 6. Candidate Coordinate Normalization

Candidate coordinates must be normalized relative to the observation window, not passed as absolute world coordinates.

For world-space source coordinate:

\[
s=(s_x,s_y)
\]

and fixed-size observation window with origin:

\[
(x_0,y_0)
\]

and side length:

\[
L,
\]

use:

\[
\tilde{s}_x=\frac{s_x-x_0}{L},
\qquad
\tilde{s}_y=\frac{s_y-y_0}{L}.
\]

The network consumes:

\[
\tilde{s}=(\tilde{s}_x,\tilde{s}_y).
\]

This normalization is part of preprocessing.

Do not feed quadtree leaf size or tree depth to the model.

## 7. Model Definition

The model is a **conditional energy field**.

For observation:

\[
O
\]

and candidate source:

\[
s,
\]

the neural network outputs:

\[
E_\theta(O,s)\in\mathbb R.
\]

Interpretation:

- low energy = candidate source is compatible with observation;
- high energy = candidate source is incompatible.

For a finite candidate set:

\[
\{s_1,\dots,s_K\},
\]

convert energy into the student posterior:

\[
p_\theta(s_i\mid O)
=
\frac{
\exp[-E_\theta(O,s_i)]
}{
\sum_j \exp[-E_\theta(O,s_j)]
}.
\]

Implement this numerically using:

```python
log_prob = torch.log_softmax(-energy, dim=-1)
prob = log_prob.exp()
```

## 8. Network Architecture

Use three components:

1. spatial observation encoder;
2. candidate-source encoder;
3. candidate-conditioned attention followed by an energy head.

Do not add explicit edge extraction to the student model.

### 8.1 Observation encoder

Input shape:

```text
[B, 1, H, W]
```

Use a small CNN that preserves a spatial feature map.

Reference architecture:

```python
self.image_encoder = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1),
    nn.GELU(),

    nn.Conv2d(32, 64, 3, stride=2, padding=1),
    nn.GELU(),

    nn.Conv2d(64, hidden, 3, stride=2, padding=1),
    nn.GELU(),
)
```

Do not immediately global-average-pool.

The resulting feature map should remain spatial:

```text
[B, C, h, w]
```

Flatten spatial positions into tokens:

```python
tokens = feat.flatten(2).transpose(1, 2)
```

giving:

```text
[B, N, C]
```

where:

\[
N=h\times w.
\]

## 9. Candidate Coordinate Encoding

Use Fourier features.

Reference implementation:

```python
import math
import torch
import torch.nn as nn


class FourierXY(nn.Module):
    def __init__(self, frequencies=6):
        super().__init__()
        self.frequencies = frequencies

    def forward(self, xy):
        # xy: [B, K, 2]
        out = [xy]

        for k in range(self.frequencies):
            w = (2.0 ** k) * math.pi
            out.append(torch.sin(w * xy))
            out.append(torch.cos(w * xy))

        return torch.cat(out, dim=-1)
```

For six frequencies:

```python
xy_dim = 2 * (1 + 2 * 6)
```

Encode the Fourier coordinate with an MLP:

```python
self.source_query = nn.Sequential(
    nn.Linear(xy_dim, hidden),
    nn.GELU(),
    nn.Linear(hidden, hidden),
)
```

## 10. Candidate-Conditioned Attention

Each source candidate should query the spatial observation features.

Let:

\[
q_i
\]

be the candidate query for candidate \(s_i\).

Let:

\[
k_j,v_j
\]

be key/value projections of spatial token \(j\).

Use:

\[
a_{ij}
=
\operatorname{softmax}_j
\left(
\frac{q_i^\top k_j}{\sqrt{d}}
\right).
\]

Then:

\[
z_i
=
\sum_j a_{ij}v_j.
\]

The energy head consumes:

\[
[q_i,z_i].
\]

Reference implementation:

```python
class SourceEnergyField(nn.Module):
    def __init__(self, hidden=128, fourier_frequencies=6):
        super().__init__()

        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, hidden, 3, stride=2, padding=1),
            nn.GELU(),
        )

        self.xy = FourierXY(frequencies=fourier_frequencies)

        xy_dim = 2 * (1 + 2 * fourier_frequencies)

        self.source_query = nn.Sequential(
            nn.Linear(xy_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        self.keys = nn.Linear(hidden, hidden)
        self.values = nn.Linear(hidden, hidden)

        self.energy_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, observation, candidates):
        """
        observation: [B, 1, H, W]
        candidates:  [B, K, 2]

        returns:
            energy: [B, K]
        """

        feat = self.image_encoder(observation)

        B, C, H, W = feat.shape

        tokens = feat.flatten(2).transpose(1, 2)

        k = self.keys(tokens)
        v = self.values(tokens)

        q = self.source_query(self.xy(candidates))

        logits = torch.matmul(q, k.transpose(1, 2))
        logits = logits / math.sqrt(C)

        attention = torch.softmax(logits, dim=-1)

        context = torch.matmul(attention, v)

        energy_input = torch.cat([q, context], dim=-1)

        energy = self.energy_head(energy_input).squeeze(-1)

        return energy
```

This is the required baseline architecture.

Keep it simple until the baseline is validated.

## 11. Teacher Distribution

The upstream physics teacher evaluates source candidates.

The teacher physical cost is conceptually:

\[
D_{\rm phys}(s)
=
\alpha D_{\rm weighted}(s)
+
\beta D_{\rm edge}(s).
\]

The student module does not compute these terms from geometry.

It only consumes the resulting costs or the already-normalized teacher probability.

If only physical cost is supplied:

\[
q^*(s_i\mid O)
\propto
A_i
\exp[-D_i/\tau],
\]

where:

- \(D_i\) is candidate physical cost;
- \(A_i\) is the physical source-space area represented by candidate/cell \(i\);
- \(\tau\) is teacher temperature.

This area correction is important when adaptive coarse-to-fine search samples some source-space regions more densely than others.

Without area correction, dense candidate sampling may artificially inflate probability mass.

Implement teacher normalization approximately as:

```python
def normalize_teacher_distribution(
    physical_cost,
    temperature,
    area_weight=None,
    valid_mask=None,
    eps=1e-12,
):
    """
    physical_cost: [B, K]
    area_weight:   [B, K] or None
    valid_mask:    [B, K] bool or None
    """

    log_w = -physical_cost / temperature

    if area_weight is not None:
        log_w = log_w + torch.log(area_weight.clamp_min(eps))

    if valid_mask is not None:
        log_w = log_w.masked_fill(~valid_mask, float("-inf"))

    teacher_prob = torch.softmax(log_w, dim=-1)

    return teacher_prob
```

If upstream provides `teacher_prob`, use it directly after validating normalization.

## 12. Training Objective

The main training objective is teacher-to-student cross entropy:

\[
\mathcal L_{\rm posterior}
=
-\sum_i
q_i^*
\log p_{\theta,i}.
\]

This is equivalent to minimizing:

\[
D_{\rm KL}
\left(
q^*
\parallel
p_\theta
\right)
\]

up to a constant teacher entropy term.

The KL direction is deliberate because the model must not drop valid modes or ridge-like feasible regions.

Reference implementation:

```python
def teacher_student_loss(
    energy,
    teacher_prob,
    valid_mask=None,
):
    """
    energy:       [B, K]
    teacher_prob: [B, K]
    valid_mask:   [B, K] bool or None
    """

    logits = -energy

    if valid_mask is not None:
        logits = logits.masked_fill(~valid_mask, float("-inf"))

    log_student = torch.log_softmax(logits, dim=-1)

    loss = -(teacher_prob * log_student).sum(dim=-1)

    return loss.mean()
```

Do not add:

- source coordinate MSE;
- analytic shadow-line loss;
- obstacle reconstruction loss;
- edge-orientation loss.

## 13. Student Probability Output

Training and evaluation should expose:

```python
energy = model(response, candidate_xy)
student_log_prob = torch.log_softmax(-energy, dim=-1)
student_prob = student_log_prob.exp()
```

The required public outputs are:

```text
energy
student_probability
```

for the same candidate ordering supplied by the teacher.

## 14. Mock Teacher Dataset

Do not wait for the upstream physics code.

Create a mock teacher generator so the entire student pipeline can be tested independently.

Reference:

```python
def fake_record(B=4, K=64, H=128, W=128, temperature=0.1):
    response = torch.rand(B, 1, H, W)

    candidates = torch.rand(B, K, 2) * 2.0 - 1.0

    physical_cost = torch.rand(B, K)

    teacher_prob = torch.softmax(
        -physical_cost / temperature,
        dim=-1,
    )

    return {
        "response": response,
        "candidate_xy": candidates,
        "physical_cost": physical_cost,
        "teacher_prob": teacher_prob,
    }
```

Also create a deterministic structured mock target with:

- one Gaussian peak;
- two Gaussian peaks;
- one elongated ridge.

Use these to verify that the student can fit multimodal and ridge-like target distributions.

## 15. Required Unit Tests

Implement tests for at least the following.

### 15.1 Shape tests

Verify:

```text
response [B,1,H,W]
candidate_xy [B,K,2]
energy [B,K]
student_prob [B,K]
```

for multiple values of:

```text
B
K
H
W
```

### 15.2 Probability normalization

Verify:

\[
\sum_i p_\theta(s_i\mid O)=1
\]

for each batch element.

### 15.3 Masking

If `valid_mask` is supplied:

- invalid candidates receive zero probability;
- loss ignores invalid candidates.

### 15.4 Teacher normalization

Verify:

\[
\sum_i q_i^*=1.
\]

Also test area-weighted normalization.

### 15.5 Tiny overfit test

Take 5–20 fixed teacher records.

Train until the student nearly reproduces them.

If this fails, do not launch full training.

### 15.6 Checkpoint round-trip

1. train several steps;
2. save checkpoint;
3. reload into a fresh model;
4. verify matching validation output;
5. continue training.

## 16. Dense Source-Grid Evaluation

Training uses sparse candidate sets.

Evaluation must support dense source-space plots.

Implement:

```python
@torch.no_grad()
def evaluate_source_grid(
    model,
    response,
    xy_grid,
    chunk=4096,
):
    """
    response: [1, 1, H, W]
    xy_grid:  [N, 2]

    returns:
        energy: [N]
    """

    model.eval()

    all_energy = []

    for start in range(0, len(xy_grid), chunk):
        xy = xy_grid[start:start + chunk]

        candidates = xy.unsqueeze(0)

        e = model(response, candidates)

        all_energy.append(e.squeeze(0).cpu())

    return torch.cat(all_energy, dim=0)
```

Use dense-grid evaluation for diagnostics such as:

- one-edge ridge;
- two-edge intersection;
- multimodal posterior;
- teacher/student comparison.

## 17. Validation Metrics

Primary validation metrics should compare distributions, not only source coordinates.

Record at least:

### Cross entropy

\[
-\sum_i q_i^*\log p_i.
\]

### Forward KL

\[
D_{\rm KL}(q^*\|p).
\]

### L1 distribution distance

\[
\sum_i|q_i^*-p_i|.
\]

### Student entropy

\[
H(p)
=
-\sum_i p_i\log p_i.
\]

### Teacher entropy

\[
H(q^*)
=
-\sum_i q_i^*\log q_i^*.
\]

### Entropy mismatch

\[
|H(p)-H(q^*)|.
\]

Do not make coordinate RMSE the main validation metric.

Coordinate error may be reported only when the teacher posterior is genuinely localized.

## 18. Diagnostic Outputs

For a fixed validation subset, periodically save:

```text
response image
candidate positions
physical cost
teacher probability
student energy
student probability
teacher entropy
student entropy
```

When candidate positions cover a grid, render probability and energy heatmaps.

These diagnostics are required to determine whether the model learns ridge/intersection structure.

## 19. Training Loop

Reference training step:

```python
def train_step(
    model,
    batch,
    optimizer,
    teacher_temperature=0.1,
):
    response = batch["response"]
    candidates = batch["candidate_xy"]

    teacher_prob = batch.get("teacher_prob")

    if teacher_prob is None:
        teacher_prob = normalize_teacher_distribution(
            physical_cost=batch["physical_cost"],
            temperature=teacher_temperature,
            area_weight=batch.get("area_weight"),
            valid_mask=batch.get("valid_mask"),
        )

    energy = model(response, candidates)

    loss = teacher_student_loss(
        energy,
        teacher_prob,
        valid_mask=batch.get("valid_mask"),
    )

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    optimizer.step()

    return {
        "loss": float(loss.detach()),
    }
```

Add gradient clipping only if needed.

Do not add architectural complexity before this baseline is stable.

## 20. Checkpointing

Checkpoints must contain:

```python
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict() if scheduler is not None else None,
    "epoch": epoch,
    "global_step": global_step,
    "config": config,
    "best_val_loss": best_val_loss,
}
```

Implement:

```text
save_checkpoint
load_checkpoint
```

and support resuming training.

## 21. Configuration

Make the following configurable:

```text
hidden_dim
fourier_frequencies
batch_size
learning_rate
weight_decay
teacher_temperature
candidate_count
checkpoint_interval
validation_interval
random_seed
```

Also record upstream metadata:

```text
dataset version
scene split version
candidate-search config
teacher-cost config
area-weight convention
```

Avoid hard-coding these values throughout the code.

## 22. Data Partition Rules

Train/validation/test splits must be scene-level.

Never randomly split windows from the same scene into different sets.

The student code should assume upstream records preserve:

```text
scene_id
view_id
```

for traceability.

## 23. Dataset Rules Relevant to Student Development

The upstream dataset pipeline is expected to enforce:

- fixed-size response inputs;
- source-free observation windows;
- tree depth hidden from the model;
- original leaf size hidden from the model;
- source-distance metadata hidden from the model;
- consistent geometric augmentation;
- single-source fixed-intensity subset for MVP;
- scene-level splits.

The student module should not depend on any source-conditioned quadtree metadata.

## 24. Geometric Augmentation

Rotation, reflection, and translation are applied upstream to reduce statistical shortcuts.

The model should not assume any global source orientation preference.

Do not explicitly encode the analytic transformation law in the first model.

An equivariance loss may be added later, but set its coefficient to zero initially unless augmentation proves insufficient.

## 25. Required Scientific Diagnostics

The evaluation workstream will test whether the student independently recovers geometric source-space structure.

Your implementation must therefore support dense posterior evaluation for the following.

### One-edge case

Expected:

```text
extended low-energy / high-probability ridge
```

The analytic shadow-line relation is evaluation-only.

### Two-edge case

Expected:

```text
ridge intersection -> concentrated probability region
```

### Ambiguous case

Expected:

```text
high-entropy distribution
```

rather than an arbitrary point prediction.

### Held-out geometry

Expected:

ridge/intersection behavior on unseen obstacle compositions.

No analytic line-incidence loss should be added to training.

## 26. Explicitly Excluded Features

Do not implement the following unless explicitly requested later:

- obstacle geometry input;
- obstacle mask input;
- edge-map input;
- source-coordinate regression loss;
- analytic shadow-line loss;
- full-field decoder;
- obstacle decoder;
- variable source intensity;
- multiple-source inference;
- distance attenuation;
- reflection;
- material modeling;
- measurement noise;
- diffusion posterior;
- normalizing flow posterior;
- Phase II camera RL policy.

## 27. Integration Contract with Upstream Modules

The student module should require only:

```text
response
candidate_xy
teacher_prob or physical_cost
optional area_weight
optional valid_mask
scene_id
view_id
```

No direct import of upstream geometry objects should be required.

Do not move physics or rerendering logic into `model.py` or `train.py`.

This separation is deliberate.

## 28. Integration Contract with Evaluation

Expose enough information for downstream evaluation to receive:

```text
energy
student_probability
trained checkpoint
training metrics
validation metrics
```

Dense source-grid evaluation must be callable from evaluation scripts.

## 29. Implementation Order

Complete work in this order.

### Step 1

Freeze and document the teacher record schema.

### Step 2

Implement `FourierXY`.

### Step 3

Implement `SourceEnergyField`.

### Step 4

Add shape/unit tests.

### Step 5

Implement energy-to-probability conversion.

### Step 6

Implement teacher normalization.

### Step 7

Implement teacher/student loss.

### Step 8

Implement a mock teacher dataset.

### Step 9

Verify fitting of:

- one peak;
- two peaks;
- one ridge.

### Step 10

Implement validation metrics.

### Step 11

Implement dense-grid inference.

### Step 12

Implement checkpoint save/load/resume.

### Step 13

Overfit a tiny mock dataset.

### Step 14

Integrate the first real upstream teacher records.

### Step 15

Overfit 5–20 real teacher records.

### Step 16

Train on a small real subset.

### Step 17

Run held-out validation.

### Step 18

Only after all prior steps succeed, begin full-scale training and hyperparameter tuning.

## 30. Acceptance Criteria

The student task is considered structurally complete when all of the following are true:

1. `model(response, candidate_xy)` returns `[B,K]` energies.
2. Student probabilities normalize correctly.
3. Teacher probabilities normalize correctly.
4. Area-weighted teacher normalization is supported.
5. Variable candidate count is supported either natively or through padding and masks.
6. The model can overfit a tiny teacher set.
7. The model can fit a synthetic multimodal teacher.
8. The model can fit a synthetic ridge-shaped teacher.
9. Dense-grid visualization works.
10. Checkpoint save/load/resume works.
11. Validation metrics are logged.
12. The model integrates with a real upstream teacher record without importing geometry.
13. The student code exposes energy and probability to downstream evaluation.
14. Scene-level split metadata is retained.
15. No forbidden inputs or analytic shadow-line supervision are used.

## 31. Important Integration Question to Resolve Before Full Training

Before full training begins, confirm with the teacher/search owner exactly how adaptive candidates map to probability mass.

If each adaptive candidate represents a cell of different source-space area, teacher probability should account for represented area:

\[
q_i^*
\propto
A_i
\exp(-D_i/\tau).
\]

Do not assume that equally weighted sampled candidate points correspond to equal physical probability mass.

This issue must be resolved before interpreting teacher/student probability calibration.

## 32. Final Required Output from Codex

When implementation is complete, provide:

```text
1. list of modified/created files
2. concise architecture summary
3. exact teacher-record schema used
4. exact loss implemented
5. tests added
6. test results
7. mock overfit result
8. checkpoint round-trip result
9. dense-grid evaluation status
10. remaining integration assumptions/blockers
```

Do not claim success for tests that were not run.

Do not modify upstream physics/data modules unless required to satisfy an agreed interface.

Do not introduce extra model complexity without documenting the reason.
