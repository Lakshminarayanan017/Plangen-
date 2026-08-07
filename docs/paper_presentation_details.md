# PlanGen: A Neuro-Symbolic Engine for Residential Floor-Plan Generation

*Reference document for paper / presentation preparation. Covers motivation,
problem formulation, system architecture, the learned placement transformer,
data pipeline, training methodology, evaluation, and results/status.*

---

## 1. Abstract

PlanGen generates residential floor plans to an "architect-usable" quality bar
from a natural-language brief (and optionally an image of the plot boundary),
targeting the **Indian residential context** (Vastu orientation preferences,
National Building Code minimums, Indian room programs). Its central design
claim is that **end-to-end neural generation is the wrong tool for floor-plan
layout**, because a floor plan is a *partition of a polygon* — a hard geometric
constraint that free-form regression cannot satisfy. Instead, PlanGen adopts a
**neuro-symbolic** architecture: machine learning is used only for the
*judgment* sub-tasks it excels at (where to place rooms, which candidate is
best), while deterministic algorithms *guarantee* correctness (no overlaps,
full coverage, code compliance). The learned centerpiece is an
**autoregressive transformer placement model** that predicts *discrete* seed
cells on a 32×32 grid with **masked decoding**, making geometrically invalid
layouts literally unsamplable. A deterministic hub-first carver then converts
seeds into exact, overlap-free geometry.

---

## 2. Motivation and the v1 Post-Mortem (the paper's central lesson)

The project's first version was a conventional **GNN + autoregressive
transformer** (512-dim, 12 layers, Mixture-of-Gaussians output heads, trained
46 epochs on RPLAN). It failed, and the failure was **measured**, not guessed —
this post-mortem is the strongest motivating evidence for the redesign:

| Root cause | Measured evidence | Contribution to failure |
|---|---|---|
| **Wrong task formulation** — regressing continuous `(cx, cy, w, h)` boxes | Free boxes cannot represent a *partition*; overlap is a representation flaw, not a learning flaw | ~60% |
| **Corrupted training data** | The bbox training cache had **13.4% room-pair overlap in the ground truth itself**; room-area/plot ratio p10–p90 = 0.79–1.12 (rooms literally didn't tile the plot) | ~30% |
| **No boundary conditioning** | The global token carried only plot width/height — no polygon mask, so irregular plots were impossible | (structural) |
| **Invisible failure** | Only MoG negative-log-likelihood was tracked (fell smoothly to −33); **no layout-quality metric**, so garbage output looked like success | (methodological) |
| Data *quantity* | 80K RPLAN plans is ample for a discrete formulation | ~10% |

**Key insight for the paper:** *post-hoc physics (push-apart, wall-snap) cannot
repair an overlap that the representation permits.* The fix is to change the
representation so overlaps are impossible by construction.

---

## 3. Problem Formulation

**Input:** a plot polygon (rectangular from text like `40x30`, or an arbitrary
polygon from an image) + a room program (a list of rooms with types, target
areas, and zones) + an entrance side + context (multi-floor, Vastu, NBC).

**Output:** a floor plan — a *partition* of the plot interior into
non-overlapping rooms with walls, doors/openings, and circulation, satisfying
hard constraints (building code minimums, reachability, structural rules).

**Reformulated learning task (the v2 fix):** instead of regressing geometry,
the model performs **discrete classification**: for each room (in a canonical
generation order), predict
- a **seed cell** — one of `32×32 = 1024` cells on a boundary-clipped grid, and
- a **size class** — one of 40 discrete area buckets (25 sqft each; top bucket
  absorbs ≥1000 sqft).

Geometry is *not* the model's job. A deterministic carver expands seeds into
exact rectilinear regions. This **decouples placement (learned) from geometry
(algorithmic)** and is the architectural crux of the contribution.

---

## 4. System Architecture — 6-Tier Neuro-Symbolic Pipeline

An **orchestrator** conducts a best-of-K loop; every tier is a plug-in behind
typed **contracts**, so a statistical fallback can be swapped for a trained
model by changing one constructor argument.

```
Brief (+image)
   │
   ▼
Step 1  PARSE      NL/image → structured RoomProgram        [LLM + rules]
Step 2  MATCH      priors from ~5K extracted plans          [statistical/ML]
Step 3  ENRICH     adjacency graph, Vastu, NBC minimums     [rules]
   │
   ▼  EngineRequest (typed contract)
   │
   ▼  ORCHESTRATOR  — per candidate, K times:
        ├─ T2  PROPOSE   seed cells + size classes          [ML placer ⇄ prior fallback]
        ├─ T3  REALIZE   hub-first carver → exact geometry  [algorithm]
        ├─ T4  SETTLE    squeeze-&-settle area calibration  [algorithm]
        ├─      CONNECT  openness gradient + reachability   [algorithm]
        ├─      VALIDATE hard-rule + soft-score verdict     [rules]
        ├─     [REPAIR → re-validate]  (≤2–3 rounds)        [algorithm]
        └─  RANK   critic or soft-score                     [ML critic ⇄ soft-score]
   │
   ▼
Renderer (SVG)
```

**Design principles:**
- **ML decides, algorithms guarantee.** No ML tier ever emits final
  coordinates. Correctness (no overlap, coverage, code) is always algorithmic.
- **Every ML tier has a statistical-prior twin** as a fallback and as an A/B
  baseline. An ML tier ships *only if it beats its fallback* on the harness.
- **K-sampling + bounded repair + per-tier failure telemetry** make the engine
  robust and debuggable.

### 4.1 Core data structure — the ownership grid
Geometry lives on a **cell-ownership grid** (`core/grid_plan.py`, 1.5-inch
cells). Each cell is owned by exactly one room, so **overlap is structurally
impossible** — the single most important guarantee, and the direct answer to
the v1 failure. The grid supports rectilinear (L-shaped) rooms; the current
carver emits rectangles via guillotine splits, but irregular rooms are
addable later *without retraining* because the model only places seeds.

### 4.2 The Rules Catalog (symbolic tier)
Hard and soft rules are modular (`engine/rules/`): `structure` (e.g. STR-003:
parking must have an exterior vehicle gate), `circulation` (reachability),
`doors`, `openings`, `light`, `walls`, `freespace`, and `nbc` (National
Building Code minimum room dimensions). The validator returns a **verdict**
with hard-violation flags and a **soft score** (0–100+) used for ranking.

---

## 5. The Learned Placement Transformer (Tier-2, the ML centerpiece)

Package: `plangen_remastered/tier2_placer/`. ~**7.36M parameters** (Colab-
trainable). Built to be a drop-in `proposer`.

### 5.1 Inputs and encoders
- **Program encoder — GATv2 graph attention** (`model/encoder.py`): the room
  program is a graph (nodes = rooms with type/zone/floor embeddings, edges =
  desired adjacencies). 3 GATv2 layers, node dim 128, 4 heads, ELU + residual +
  LayerNorm. Produces one token per room.
- **Boundary encoder — CNN** (`model/encoder.py`): the plot is a **2-channel
  64×64 raster** (channel 0 = interior footprint mask, channel 1 = entrance
  edge). A 4-block strided CNN → 8×8 = 64 spatial memory tokens. *This is what
  v1 lacked* — real polygon-boundary conditioning, enabling irregular plots.
- **Global context**: `[log width, log height, log aspect, n_rooms/scale]`
  injected as an extra memory token.

### 5.2 Decoder — causal autoregressive transformer
`model/decoder.py`: 6 layers, `d_model=256`, 8 heads, FF dim 1024, pre-norm,
dropout 0.1. Cross-attends to the concatenated memory `[program tokens ‖
boundary tokens ‖ global]`. Per room step, two heads:
- **cell head** → 1024 logits (the seed cell), and
- **size head** → 40 logits (the size class).

Teacher-forced during training; each step is conditioned on the previous
room's placement (a START token seeds step 0).

### 5.3 Masked decoding — the no-repeat-of-v1 guarantee
`masked_decode.py`: before sampling each cell, logits are set to **−∞** for
cells that are (a) outside the boundary mask, (b) within the *claimed radius* of
an already-placed room, or (c) zone-/adjacency-forbidden. **Invalid layouts are
therefore unsamplable** — the property v1 never had. Temperature sampling over
the legal remainder yields **K diverse proposals**. A **confidence** signal
(mean top-1 post-mask probability) < τ (0.35) triggers graceful **fallback to
the statistical `PriorProposer`**.

### 5.4 Inference is torch-free in production
`numpy_infer.py` reimplements the entire forward pass in **pure NumPy + SciPy**
(linear, LayerNorm, GELU via `erf`, im2col conv2d, multi-head attention, GATv2
segment-softmax). A unit test asserts **logit agreement < 1e-4** with the
PyTorch model (measured 7e-7). Result: the **production server never imports
torch** — it loads a `.npz` and runs NumPy. PyTorch is a *training-only*
dependency.

---

## 6. Data Pipeline (M4)

Package: `plangen_remastered/training/`. Source: **CubiCasa5K** (4,989 plans
with per-room polygons; CC BY-NC). RPLAN (80K) is planned as source #2.

**Pipeline:** `schema_audit` (never trust the docs — dump the real record
shape) → `vocab` (CubiCasa room types → engine vocabulary; drop
undefined/outdoor/sauna) → `geometry` (dependency-free shoelace area/centroid +
NumPy polygon rasterizer) → `prep_cubicasa` (per plan: polygons → interior
footprint bbox → 64×64 boundary mask; each room → seed cell + size class;
adjacency edges from shared polygon edges; entrance side) → `gates` (5 hard
quality gates) → `render_samples` (eyeball HTML artifact).

**Key data findings (honesty as method):**
- CubiCasa5K is **Finnish** — it literally contains *sauna* rooms. This is a
  real domain gap vs. Indian homes, acknowledged up front.
- **22.9%** of rooms are labeled "undefined" (dropped); per-room ft-scale
  present in 99.7% (so real dimensions are recoverable).
- Frame everything on the **interior footprint bbox**, not the raw plan bbox
  (which includes the yard) — this lifted mask fill from 0.29 → 0.74.

**The 5 hard gates (a gate failing aborts prep with a named reason):**
1. **Seed-collision rate = 0%** — after a collision-nudge step,
   **0 / 32,574** seeds collide.
2. **Every seed inside its boundary** — 0 out-of-boundary.
3. **Coverage sanity** — Σ size-class areas within [0.3, 1.25]× plot.
4. **Distribution report** — room-count / type / size histograms (domain-gap
   visibility).
5. **Frozen val split** — 10% by plan-id, versioned to a manifest.

**Yield:** 4,989 → **4,430 prepared plans (89%)**; skips are all-undefined
plans. All 5 gates pass.

The seed-collision gate is notable: a raw pass had 5/32,574 collisions; rather
than *relax the gate*, the pipeline adds a **collision-nudge** (move a later-
generation room's seed to the nearest free in-boundary cell), achieving a true
0% — an example of "fix the data, don't lower the bar."

---

## 7. Training Methodology (M5)

`tier2_placer/train.py`. The v1 lesson — *loss measured the wrong thing* —
drives every choice here:

- **Loss:** cross-entropy on the seed cell with **Gaussian spatial label
  smoothing** (σ = 1 cell — a near-miss gets partial credit) + cross-entropy on
  the size class.
- **Augmentation:** ×8 dihedral (4 rotations × flip), remapping mask, seed
  cells, and entrance side consistently — multiplies effective data 8×.
- **Curriculum:** small plans (≤6 rooms) first, then the full program.
- **Checkpoint selection by ENGINE METRICS, never by loss.** After each eval
  epoch, the model's proposals are run through the **real engine**
  (realize→settle→connect→validate) on 50 held-out boundaries; the checkpoint
  with the best **mean soft score** is kept. This is the direct fix for v1's
  invisible failure.
- **Resumable, portable training:** a single `checkpoint.pt` (model +
  optimizer + epoch + best-so-far + log) is written atomically after every
  epoch, enabling training to continue across multiple free-Colab sessions /
  accounts. The production `.npz` is re-exported whenever a new best is found.

### 7.1 The merge gate (non-negotiable ship criterion)
The trained placer is wired into the live engine **only if**, on the golden
harness, it (1) **beats** the statistical `PriorProposer` on mean best-score,
(2) regresses **no single brief by > 5 points**, and (3) keeps **proposal
fidelity ≥ 0.8** (seed displacement + size-class drift after realization —
if the realizer mutes the model, any A/B is meaningless). Until it passes, it
stays behind the confidence fallback / on a branch.

---

## 8. Evaluation

- **Golden harness** (`harness/`): a fixed set of representative briefs
  (plots × programs) scored by the validator. The statistical baseline scores a
  mean soft score of **~75.5**.
- **Hard metrics first** (before any training): overlap = 0, coverage = 100%,
  NBC pass — a plan that violates a hard rule is a failure regardless of score.
- **Human preference (Phase A):** a learning-to-rank study on the ~25 soft-
  score weights. Finding: `soft_score` is **exactly linear** in the weights
  (`score = 100 + w·g`), so tuning is **convex pairwise learning-to-rank** (no
  GPU, no CMA-ES). Objective is *human-pick agreement*, never raw score
  (maximizing score is degenerate). Verdict: **18 labeled briefs cannot
  identify 20 free weights** (top-1 CV caps at 28%); this is a data-sufficiency
  ceiling, so the tuned weights were **not** shipped — the deliverable is the
  tested infrastructure. A candid negative result worth reporting.

---

## 9. Results and Status (as of this writing)

- **Engine (algorithmic core):** complete and tested; produces valid,
  overlap-free, code-respecting plans. Baseline golden-harness mean ≈ 75.5.
- **Tier-2 placer:** fully implemented (7.36M params); NumPy path verified to
  1e-4 vs. torch; 114/114 remastered tests green; drop-in fallback verified.
- **Training:** in progress on Colab GPU (resumable across accounts). The
  merge-gate A/B decides whether it ships.
- **Honest expectation:** because the training corpus is Finnish, the first
  model may *lose* to the Indian statistical fallback — the merge gate exists
  precisely to prevent shipping a regression. A model that loses here is the
  safety system working, not a project failure; the next step is an Indian
  fine-tune.

---

## 10. Engineering / Reproducibility Notes (useful in a systems paper)

- **Typed contracts** (`engine/contracts.py`) isolate every tier; multi-floor
  context is carried from day 1 to avoid contract rot.
- **Determinism & guarantees on the ownership grid** — overlap impossible by
  construction; 1.5-inch cells.
- **Torch-free production** via verified NumPy inference (7e-7 agreement).
- **Portable, atomic checkpointing** for interrupted free-tier GPU training.
- **Gate-driven data prep** — no silent passes; every gate aborts with a named
  reason.
- **~78 Python modules**, full unit-test coverage of model shapes, mask
  legality, overfit-a-batch sanity, NumPy≡torch equality, device-agnosticism,
  augmentation correctness, and the data-prep polygon math.

---

## 11. Limitations and Future Work

- **Domain gap:** Finnish/Chinese training data vs. Indian target → Indian
  fine-tune (M6-era) with owned Indian priors for the program.
- **Room shape:** currently rectilinear rectangles; the seed-cell contract
  makes **irregular (5–6 sided) rooms** addable later *without retraining* (the
  carver owns shape).
- **Learned critic (Tier-5):** two-tower CNN(render)+GNN(graph) ranker trained
  on human pairwise preference — not yet built (Phase C).
- **Zone planner (Tier-1)** and optional **RL fine-tune** of the proposer
  against validator+critic reward.
- **Second data source:** RPLAN (80K) and ResPlan (17K vector plans) prep.
- **Multi-floor** realization (stair/wet-stack/structural continuity) — a
  first-class requirement, contracts already carry floor context.

---

## 12. One-Slide Summary (for the talk)

> **PlanGen reframes floor-plan generation from a regression problem into a
> constrained discrete-placement problem.** A GATv2+transformer places rooms as
> *discrete seed cells* with *masked decoding* (invalid = unsamplable); a
> deterministic carver turns seeds into exact, overlap-free geometry. ML
> decides where; algorithms guarantee correctness. Every learned tier must beat
> a statistical twin on a quality harness before it ships. This directly fixes
> the measured causes of the v1 end-to-end model's failure (continuous boxes +
> overlapping data + no boundary conditioning + loss-only evaluation).**
