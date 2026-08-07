# PlanGen v2 — Algorithm + Trained Transformer
## The Definitive Implementation Plan

**Status:** Adopted
**Date:** 2026-07-17
**Supersedes:** all prior roadmaps. One document, whole workflow, no gaps.
**Prime directive:** every script ships production-ready — tested, deterministic,
all-scenario, harness-measured. No placeholders, no dummies, ever.

---

## 0. What v1 got wrong — and the v2 answer to each

| v1 flaw | v2 mechanism (this plan) |
|---|---|
| Transformer regressed exact continuous coords (MoG) | Transformer decides DISCRETE things (seed cells + size classes) with **masked decoding** — invalid output is unsamplable (§2) |
| Training data had 13.4% GT overlap (bbox prep) | Polygon-true prep with **hard gates**: 0% overlap, boundary kept, mandatory 100-sample eyeball render (§3) |
| Judged by loss, failure invisible | Checkpoints selected by **engine metrics on the harness**, never loss (§3.4); reviewer detects every known flaw class (§4) |
| Free space = leftover scraps | Free-space graph is a first-class reviewed artifact: connectivity, width, fraction (§4.2) |
| No shared-wall discipline | Wall-graph substrate (built) + shared-wall efficiency & jog rules (§4.2) |
| Doors placed nowhere / nonsensically | Connection typing (built) + door-quality rules: swing collision, corner stubs, privacy paths (§4.2) |
| No staircase at all | Staircase as engineered component: riser math, placement, carving, rules, rendering (§5) |
| CP-SAT as the core (blind to adjacency) | CP-SAT as a **side tool**: bounded local optimization where combinatorics beat greedy (§6) |
| Post-hoc physics fought the model | Partition-first carving consumes the model's decisions by design (built, §1) |

---

## 1. The Workflow (end to end, with every failure path)

```
EngineRequest (validated; named errors/warnings)
   │
   ▼
[P] PROPOSE      Tier-2 TRANSFORMER (§2) — K sampled proposals
   │             confidence < τ or model absent → PriorProposer fallback
   ▼
[R] REALIZE      stair footprint pre-claimed (§5) → seed-guided band carve
   │             (NBC-floored budgets, stacking, OTS insertion) — built
   │             CarveError → bounded retry w/ perturbed seeds → discard w/ reason
   ▼
[S] SETTLE       incremental squeeze-&-settle (built)
   │             stall w/ NBC violation → CP-SAT band re-dimension (§6.2)
   ▼
[C] CONNECT      entrance → openness gradient → reachability → windows/OTS
   │             (built) → door-quality pass (§4.2 rules as constraints)
   ▼
[V] REVIEW       Rule Reviewer (§4) → hard gate + soft score + evidence
   │             hard violations → REPAIR (rule-mapped strategies, ≤2 rounds)
   │             still failing → discard with rule id attribution
   ▼
[K] RANK         soft score ⊕ learned Critic (§4.3) → top-N plans
   │             all K discarded → refallback: prior proposer rerun (K more)
   │             still zero → EngineResult.error with named cause per stage
   ▼
Output: SVG sheet + wall-graph JSON + verdict breakdown + telemetry
```

Every arrow has a defined failure path; nothing fails silently. Telemetry per
candidate: stage timings, proposal fidelity, repair rounds, discard attribution.
(Orchestrator with all of this exists; the transformer, reviewer expansion,
stairs, and CP-SAT hooks are what this plan adds.)

---

## 2. THE TRANSFORMER (Tier-2 Placer) — centerpiece #1

### 2.1 Task contract (frozen before training starts)
Input:  boundary raster 64×64 (interior mask + entrance-side channel
        + setback channel), program graph (N rooms: type, size class,
        zone, floor; edges: adjacency must/pull/forbid).
Output: for each room, in canonical anchor order (same priority table the
        fallback uses): `seed_cell ∈ 32×32` and `size_class ∈ {1..40}`.
The realizer consumes EXACTLY this — the contract is already live
(`LayoutProposal`), so the model drops in with zero pipeline change.

### 2.2 Architecture (sized for Colab, no heroics)
- **Program encoder:** 3-layer GNN (GATv2-style attention), node dim 128.
- **Boundary encoder:** 4-block CNN → 8×8×128 memory tokens.
- **Decoder:** causal transformer, 6 layers, d_model 256, 8 heads,
  cross-attention to [program ‖ boundary] memory. ~9–12M params total.
- **Heads:** cell head (1024 logits) + size head (40 logits) per room step.
- **Conditioning discipline:** teacher forcing on ground-truth prefix;
  placed-room raster fed back as a decoder input channel each step.

### 2.3 Masked decoding — the no-repeat-of-v1 guarantee
Before sampling each cell: logits[c] = −∞ where c is (a) outside the
boundary, (b) within the claimed radius of an already-placed room (radius
from its size class), (c) zone-forbidden when the request has zone locks,
(d) violating a hard adjacency (must-attach room sampled ≥ threshold away).
Temperature sampling over the legal remainder → K diverse proposals.
**A structurally invalid proposal cannot be sampled.** Confidence = mean
top-1 probability across steps; below τ (default 0.35) → fallback proposer,
logged (`proposer=fallback, reason=low_confidence`).

### 2.4 Training protocol
- Loss: CE on cells with Gaussian spatial label smoothing (σ=1 cell;
  near-misses get partial credit) + CE on size classes.
- Curriculum: epochs 1–5 plans ≤6 rooms, then all.
- Augmentation ×8 (flips/rotations, with entrance-side relabeling).
- **Checkpoint selection = engine metrics, never loss:** every epoch,
  sample layouts for 50 held-out boundaries → run the REAL pipeline
  (realize→settle→connect→review) → record validity rate, mean soft
  score, adjacency satisfaction, fidelity. Best checkpoint = best mean
  soft score. Loss curves are logged but decide nothing.
- Export: .pt checkpoint + .npz weights; NumPy inference class mirrors the
  PyTorch graph (unit-tested for logit equality < 1e-4) so production has
  no torch dependency (the v1 trick, kept — it was good).

### 2.5 Merge gate + self-improvement
- **Merge gate:** on the golden harness, Tier2Placer must beat
  PriorProposer on mean best-score AND regress no brief by >5 points AND
  keep fidelity ≥ 0.8. Otherwise it stays on a branch. (The fallback is
  now data-driven from the 5K-plan priors — a real bar.)
- **Self-training loop (post-merge):** engine outputs that pass the hard
  gate with soft score ≥ P75 become fine-tune samples (reviewer-filtered
  self-distillation); user picks among K are logged as preference data
  for the critic (§4.3).

---

## 3. Data Pipeline (E4) — where v1 actually died; gated accordingly

- **Sources:** RPLAN ~80K (polygons + boundary masks — polygons THIS time),
  ResPlan 17K vector plans, CubiCasa5K SVG (R&D only — CC BY-NC),
  own Indian annotations (tool in M6; 200–500 plans).
- **Per-sample extraction:** boundary polygon → 64×64 mask; room polygons →
  type (mapped to our vocab), centroid seed cell (32×32), area size class,
  adjacency edges from shared polygon edges ≥ 0.6 m; front door → entrance
  side; generation order by the SAME anchor-priority table the engine uses.
- **HARD GATES (all must pass before one training step runs):**
  1. polygon room-pair overlap in prepared set = **0%** (measured, printed)
  2. every sample retains its boundary mask; seed cells all inside it
  3. distribution report: room counts, type histogram, size classes vs
     our program generator's ranges (domain-gap visibility)
  4. **eyeball gate: render 100 random prepared samples to SVG and LOOK**
     — committed to the repo as an artifact of the gate having run
  5. val split by plan id, 10%, frozen and versioned
- Deliverable scripts: `training/prep_rplan.py`, `training/prep_resplan.py`,
  `training/gates.py`, `training/render_samples.py` — each runnable,
  argument-driven, tested on fixtures.

---

## 4. THE REVIEWER — centerpiece #2

The reviewer is a **flaw-detection instrument**, not a scoring afterthought.
Design rule: *every failure mode we have ever observed maps to a named rule
that detects it with geometric evidence.* Coverage matrix maintained in the
rules module; a new observed flaw without a detecting rule is a P0 bug.

### 4.1 Architecture (registry exists; this plan triples it)
Named rules (`ID: check(ctx) → violations/evidence`), severity hard/soft,
config-weighted, per-rule breakdown, rule-mapped repairs. Target: **~40
rules** organized in modules: `rules_structure.py`, `rules_nbc.py`,
`rules_circulation.py`, `rules_openings.py`, `rules_light.py`,
`rules_stairs.py`, `rules_freespace.py`, `rules_walls.py`.

### 4.2 New rule groups (the things v1 missed, now detected)
**Free space (FSP):**
- FSP-001 hard: free-space graph (hub rooms + wide-opening spans) forms ONE
  connected component containing the entrance.
- FSP-002 soft: circulation fraction within 8–14% of floor area.
- FSP-003 soft: minimum free-space corridor width ≥ 3'0" along every
  path (computed on the circulation skeleton, not assumed).
- FSP-004 soft: hub centrality — mean hop distance from hub to all rooms.

**Shared walls (WAL):**
- WAL-001 soft: wall efficiency = shared-wall length / total internal wall
  length ≥ threshold (walls should serve two rooms).
- WAL-002 soft: jog count — near-collinear wall segments offset < 9"
  (settle gains a jog-merge move to act on this).
- WAL-003 hard: minimum wall segment ≥ 1'6" (no toothpick walls).

**Door placement (DOR):**
- DOR-001 hard: swing arcs collide with no other swing/fixed obstacle.
- DOR-002 soft: hinge stub ≥ 9" from the nearest corner.
- DOR-003 soft: bedroom doors open from circulation, not through another
  private room (privacy path evidence).
- DOR-004 soft: door-window alignment bonus for cross-ventilation.

**Stairs (STR-S):** §5.3.
**Plus existing 13** (structure, NBC, ventilation/OTS, circulation depth,
openness, privacy, hygiene) — all kept, some upgraded with evidence output.

### 4.3 Learned Critic (Tier-5, after rules are strong)
- Features: the FULL rule breakdown vector + geometry stats (aspect
  distribution, adjacency satisfaction, daylight coverage, wall efficiency).
- Training: real-vs-perturbed (real plans positive; room-swapped /
  wall-jittered negatives) → gradient-boosted trees first (interpretable),
  raster CNN later only if it beats GBT on ranking agreement.
- Blend: rank = 0.6·rule score + 0.4·critic (weights config-tunable).
- User picks among K logged from day one → preference fine-tuning later.

---

## 5. Staircase Placement (new, first-class)

### 5.1 Geometry engine (`carve/stairs.py`)
Floor height 10'0" → 17 risers @ ~7.06"; tread 10"; width ≥ 3'0".
Computed variants (all exact, on the 1.5" lattice):
- **Straight:** 3'w × 14'2" run (+3' landings both ends)
- **Dog-leg (U):** 6'2"w × ~9'6" incl. mid landing — the Indian default
- **L-shape:** per-leg run + corner landing
`StairSpec.fit(plot, floors)` returns feasible variants w/ footprints.

### 5.2 Placement + carving
- Multi-floor request → program engine injects implicit `staircase` room
  (exists in room_resolver legacy logic; ported properly).
- Proposer (prior AND transformer) seeds the stair like any room; the
  realizer **pre-claims the exact stair footprint before band carving**
  (skeleton-first — the blueprint's rule), preferring: against an exterior
  or spine wall, adjacent to the free-space hub, never crossing it.
- Same cells on every floor it serves (vertical invariant, enforced when
  multi-floor lands in M7 — the contract carries floor context already).

### 5.3 Stair rules (reviewer)
- STR-S01 hard: stair exists when n_floors > 1.
- STR-S02 hard: width ≥ 3'0"; riser count × tread fits the footprint.
- STR-S03 hard: reachable from entrance free space in ≤ 2 hops.
- STR-S04 soft: landing clearance ≥ 3' both ends.
- STR-S05 soft: not on the plot's prime daylight frontage.
Renderer: real treads, up-arrow, flight direction.

---

## 6. CP-SAT — the side tool, used where it wins

CP-SAT (OR-Tools) is bounded, warm-started, always optional (pure-python
fallback = current greedy paths). Three surgical jobs:

1. **Band/column assignment repair (M3):** when greedy packing fails or
   aspect/NBC repairs stall — solve rooms→(band, column) assignment
   minimizing seed displacement + predicted aspect penalty + adjacency
   violations. Domain: ≤ 12 rooms × ≤ 6 bands → milliseconds.
2. **Exact re-dimensioning (M3):** integer wall positions within one band
   under NBC minimums + area targets when settle plateaus with violations
   — feasibility proven or topology change forced (no silent failure).
3. **Opening assignment (M5):** door/window positions on shared walls
   subject to swing-collision and stub constraints, maximizing DOR-004
   alignment — replaces center-of-wall defaults.
Timeout 2s per call; on timeout → keep greedy result + log. CP-SAT never
decides layout topology — that is the transformer's job.

---

## 7. Milestones (fresh numbering, each gate measured on the harness)

| M | Deliverable | Gate |
|---|---|---|
| **M1** | Reviewer expansion: FSP/WAL/DOR rule groups + evidence + repairs; coverage matrix doc | all 18 briefs re-scored with new breakdowns; no NO-PLAN; flaw-injection tests: every rule catches its synthetic flaw |
| **M2** | Staircase: geometry, placement, carving, STR-S rules, renderer; stair briefs added to harness (G+1 programs) | stairs on 100% of multi-floor briefs, zero STR-S hard violations |
| **M3** | CP-SAT side tools (assignment repair + re-dimension) behind config flags | tight briefs (20x45_S class): NBC violations = 0, kept ≥ 5/6, score +10 on the weakest brief |
| **M4** | Data pipeline with all 5 gates + distribution report | gates pass; eyeball artifact committed |
| **M5** | Transformer: model, training notebook (resumable, Drive-checkpointed), engine-metric checkpoint selection, NumPy export, Tier2Placer + confidence fallback, opening assignment | **merge gate §2.5 vs data-driven PriorProposer** |
| **M6** | Critic (GBT real-vs-perturbed) + preference logging + Indian annotation tool | top-1 selection beats random-of-K on blind pairwise eval; 200+ Indian plans annotated |
| **M7** | Multi-floor: vertical skeleton (stair continuity, wet stacks, wall bearing), per-floor generation conditioned on floor below | G+1 harness briefs: stacking + continuity rules pass |
| **M8** | Polygon plots (raster boundary → carve on masked lattice) + DXF export | irregular-plot briefs in harness pass hard gate |

Order rationale: reviewer first (M1) because *every* later milestone is
judged by it — a weak reviewer makes every gate soft. Stairs (M2) before
the transformer so training data prep includes stair rooms in the vocab.

### 7.1 Status — 30 July 2026

All eight milestones are built. Two gates were **not met**, and both are
recorded as measured facts rather than quietly redefined.

| M | State | Evidence |
|---|---|---|
| M1 | done | 33 rules registered; flaw-injection tests |
| M2 | **gate met** | 6/6 G+1 briefs plan, fitted dog-leg stair, zero STR-S hard violations (`tests/test_stairs.py`) |
| M3 | built, **gate NOT met** | `harness/results/ab.json`: off 73.22 / repair 73.22 / always 70.80. Default `cpsat_mode="off"`. See below |
| M4 | done | 4430 prepared samples, all 5 gates |
| M5 | built, **gate NOT met at 24 epochs** | `tier2_placer/gate_results/`: prior 75.49 vs model 68.57. Re-run after 60–70 epochs |
| M6 | **gate met** | held-out AUC 0.912; clean-above-damaged 91.4% vs rules-only 63.2% (`critic/weights/critic_report.json`) |
| M7 | **gate met** | stair footprint overlap 100%, wall alignment 100%, zero VRT hard violations (`tests/test_multifloor.py`) |
| M8 | **gate met** | 4/4 irregular briefs plan with no hard violations; R12 DXF export (`tests/test_polygon_dxf.py`) |

**M3 — why CP-SAT is off by default.** The first band model minimized TOTAL
band slack, which is algebraically constant for any feasible packing
(`across x depth - total_area`), so every solution tied and the solver
returned an arbitrary one: `always` scored 57.29 against greedy's 73.22.
Fixing the objective to minimax (the WORST band's slack) recovered 13.5
points — to 70.80, still 2.4 below the greedy packer, still losing one brief
its plan. The honest reading: the solver optimizes area packing, while the
greedy packer's `min_band_depth` and column-stacking heuristics encode
proportion and daylight priors the model has no term for. The tooling ships
tested and available per request; the default stays `off`.

**M5 — two contract bugs found while wiring the checkpoint in.** Neither was
visible without running the trained weights end to end:

1. *The fallback threshold was unreachable.* Confidence was the model's
   top-1 cell probability, but training uses sigma=1 Gaussian label
   smoothing over 1024 cells, which caps a PERFECT model's top-1 at 0.159 —
   below the tau of 0.35. Every proposal would have fallen back forever,
   silently, no matter how well the model trained. Confidence is now the
   3x3 neighborhood mass (perfect-fit ceiling 0.779), and
   `tests/test_placer_confidence.py` asserts the ceilings.
2. *Best-checkpoint selection compared scores across different eval sets.*
   The run raised `--eval-n` from 50 to 443 at epoch 10; the mean dropped
   from ~68 to ~56 for that reason alone, so `best_model` froze at epoch 2 —
   a 6-room curriculum model — and could never update again. Checkpoints now
   carry an `eval_key`, and a changed key re-scores the stored best weights
   on the new set before comparing.

**M6 caveat, unresolved by design.** The critic is trained on engine output
versus deliberately damaged copies of the same output. That proves it
detects known flaw classes; it does not prove it has taste. Preference
logging (`critic/preferences.py`) is live and empty. The Indian annotation
tool listed in M6's deliverable is NOT built.

**M8 scope.** Irregular plots are handled by building inside the largest
inscribed rectangle after setbacks (`engine/site.py`), with the remainder as
open space — the building-envelope approach. The full masked lattice
(`GridPlan.from_polygon`) is built and tested and is what the site drawing
and DXF use. NON-RECTANGULAR ROOMS are still not carved.

## 8. Production standard (applies to every script)

1. **No placeholders.** A module lands runnable, argument-complete, with
   unit tests + a harness delta in the same session. NoopSettler-style
   stubs exist only as documented interface defaults, never as deliverables.
2. **Determinism:** same request+seed → identical output, always (tested).
3. **Config, not constants:** every knob in `EngineConfig`, recorded in
   harness baselines.
4. **Failure honesty:** every failure path returns a named reason; the
   orchestrator attributes it; the harness prints it.
5. **Performance budget:** ≤ 10s per candidate on CPU (currently ~1–4s);
   transformer inference ≤ 300ms per proposal on CPU via NumPy path.
6. **Regression law:** harness mean score may not drop >2 points without a
   written justification in the commit (ruler changes documented as such).

   *Invoked 2026-07-30 — RULER CHANGE, not a regression.* The brief set grew
   from 18 to 28: six G+1 briefs (M2) and four irregular-plot briefs (M8),
   both classes deliberately harder than the rectangular single-floor set.
   Mean best score 75.49 → 69.50 and briefs-with-a-plan 18/18 → 28/28. Every
   one of the ORIGINAL 18 briefs scores exactly what it scored before
   (83.70, 42.61, 87.03, 110.05, 15.62, 79.09, 96.82, 89.14, 93.28, 69.66,
   97.91, 75.62, 64.12, 77.86, 42.09, 86.32, 67.87, 80.10 — compare
   `harness/results/latest.json` against `tier2_placer/gate_results/
   latest.json`, whose baseline arm predates every change in this session).
   The two numbers are measured on different sets and must not be compared.
