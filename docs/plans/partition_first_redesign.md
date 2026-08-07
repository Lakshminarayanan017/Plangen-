# PlanGen v2 — Partition-First Redesign
## Post-mortem of the AR Transformer + the Quality-First Architecture

**Status:** Proposed
**Date:** 2026-07-15
**Supersedes:** `step4_gnn_diffusion_layout_engine.md`, the AR engine described in `autoregressive_engine_deep_dive.md`

---

## Part 1 — Why the AR Transformer Failed (Evidence-Based Post-Mortem)

The model trained fine. The *task it was given* was unwinnable. Four root causes, in order of severity:

### 1.1 The training data itself contains overlaps (measured, not guessed)

The cache (`ar_val_b040c629.pkl`, 8,564 val samples, ~80K train) was built by converting
RPLAN's **polygonal** rooms into **axis-aligned bounding boxes**. Measured directly on the cache:

| Metric | Value |
|---|---|
| Ground-truth room-pair overlap rate | **13.4%** of all room pairs |
| Sum of room areas ÷ plot bbox area | mean 0.96, **p10 = 0.79, p90 = 1.12** |

A model trained on this data is *rewarded* for producing overlapping, gappy boxes — that is
what the ground truth looks like. Even a perfect model reproducing its training distribution
exactly would emit overlapping rooms. Post-processing (SAT push-apart, wall snap) then fought
the model instead of finishing its work, destroying whatever spatial relationships it learned.

### 1.2 Wrong representation: free boxes instead of a partition

A real floor plan is a **partition of the buildable polygon**: every interior point belongs to
exactly one room (or circulation), walls are shared, coverage is 100%. `(cx, cy, w, h)` boxes
sampled independently from Mixture-of-Gaussians heads cannot express "these two rooms share a
wall" or "nothing overlaps" — those are *measure-zero* events for continuous distributions.
Sampling noise accumulates autoregressively across 5 tokens × N rooms. No amount of training
data or epochs fixes this; it is a representation problem, not a learning problem.

### 1.3 No plot-boundary conditioning

The GLOBAL token carried plot width/height only. The core product requirement — arbitrary
polygonal / irregular / incomplete plot boundaries — was architecturally impossible for this
model. (Ironically, RPLAN *ships* boundary masks per plan; they were discarded during prep.)

### 1.4 Training signal measured the wrong thing

MoG negative log-likelihood fell smoothly (train −5.1 → −29.4, val −10.2 → −33.0 over 46
epochs). By its own metric the run was a success. But NLL says nothing about overlap,
coverage, adjacency satisfaction, or NBC compliance. There was no layout-quality metric in the
loop, so failure was invisible until visual inspection.

**Verdict:** don't resume this checkpoint. The GNN encoder, trainer infra, CP-SAT solver, and
steps 1–3 survive. The MoG coordinate heads, box decoding, and the bbox dataset do not.

---

## Part 2 — The Core Principle of v2

> **Floor plans are partitions, not arrangements of boxes.**
> Neural networks decide *topology and rough allocation*. Deterministic geometry produces
> *exact coordinates*, as a space partition, correct by construction.
> **Neural coordinates never reach the output.**

This is not a hunch — it is where the field converged:

- **RPLAN** (Wu et al., SIGGRAPH Asia 2019): CNN predicts room locations iteratively on the
  boundary raster; walls are extracted afterward; output is a pixel partition.
- **Graph2Plan** (Hu et al., SIGGRAPH 2020): boundary mask + room graph → raster layout →
  vectorized partition. Open source.
- **WallPlan** (SIGGRAPH 2022): generates the *wall graph* directly — partition guaranteed.
- **GFLAN** (arXiv 2512.16275, Dec 2025): explicitly factorizes generation into
  **Stage A: topological planning** (sequential room-centroid allocation via discrete
  probability maps over the boundary raster) and **Stage B: geometric realization**
  (Transformer-augmented GNN regresses room boundaries against the envelope). Reports fewer
  connectivity/adjacency failures than prior work. This is the same conclusion, published.

Overlap is eliminated not by penalty terms, not by post-hoc physics, but by making it
**unrepresentable** in the output format.

---

## Part 3 — v2 Pipeline

```
User brief + plot (dims / photo / sketch)
        │
        ▼
[S1] Plot Ingestion (keep, extend)
     text dims OR image → plot POLYGON (any shape) + road side + entrance
     → normalized vector polygon + 128×128 raster mask
        │
        ▼
[S2/S3] Program Synthesis (keep — this is the moat)
     room list, target areas (NBC minimums), adjacency graph,
     zone priors, Vastu locks — Indian-specific intelligence
        │
        ▼
[S4a] NEURAL ALLOCATION  (the retrained model)
     Input : boundary mask + entrance + GNN room-graph embeddings (reuse encoder)
     Output: per-room CENTROID probability map over interior grid cells,
             predicted sequentially (anchor rooms first — keep the AR insight,
             but in discrete raster space where "inside the polygon" and
             "not on top of the living room" are trivially enforceable at decode)
     Train : RPLAN polygons + boundary masks (80K), ResPlan (17K vector),
             CubiCasa5K SVG polygons — NOT bounding boxes
        │
        ▼
[S4b] GEOMETRIC REALIZATION  (deterministic, zero ML)
     1. Seeded region growing over grid cells inside the polygon:
        every cell assigned to exactly one room; growth cost blends
        target-area deficit, aspect-ratio, zone prior, adjacency pull
     2. Wall straightening: merge jagged cell boundaries into min-segment
        rectilinear walls; enforce min wall lengths
     3. Circulation check: carve corridor if any room unreachable from entrance
     4. Door + window placement by rule (shared-wall spans, exterior walls,
        NBC ventilation ratios)
     → overlap-free and gap-free BY CONSTRUCTION, CAD-crisp
        │
        ▼
[S4c] REFINE + COMPLY
     CP-SAT / simulated-annealing polish warm-started from S4b (reuse solver);
     NBC validator as a HARD GATE with automatic repair
     (undersized room steals cells from the neighbor with most slack)
        │
        ▼
[S5] Render (keep renderer; walls with thickness, doors, windows, labels, dims)
```

### Why sequential centroid maps instead of the old MoG heads
Predicting "which cell does the kitchen's center land in" is a **classification over ~16K
cells**, masked to the polygon interior and to cells not yet claimed. It is exactly as
autoregressive as the old model (later rooms see earlier rooms), but every decode step is
constraint-checkable, the loss (cross-entropy on real data) directly measures placement
quality, and irregular boundaries are native input. This is the Graph2Plan / GFLAN Stage-A
formulation, proven on the exact datasets we hold.

---

## Part 4 — Evaluation Harness FIRST (Phase 0, non-negotiable)

The AR run failed silently because nothing measured layout quality. Before any new training:

1. **Golden brief set**: 50 fixed briefs — rectangular (40×30, 50×70…), L-shaped, trapezoid,
   irregular-polygon, and sketch-image plots; 1BHK → 4BHK Indian programs; Vastu on/off.
2. **Hard metrics** (a plan failing any of these is a failed plan):
   - room-pair overlap area = **0**
   - buildable-polygon coverage = **100%** (rooms + circulation)
   - NBC minimum dimension pass rate = 100%
   - circulation: every room reachable from entrance
3. **Soft scores** (tracked per run, trended): adjacency satisfaction %, zone/Vastu
   compliance %, aspect-ratio sanity, wall-count (fewer walls = cleaner plan),
   golden-ratio-ish room proportions.
4. **Visual review grid**: one HTML page rendering all 50 briefs per run for eyeball A/B
   against the previous run (reuse `tpl_viewer.html` machinery).

Every future change — heuristic tweak or retrained model — is judged by this harness, never
by training loss.

---

## Part 5 — Data Strategy (including the Indian gap)

| Dataset | Size | Use | License caution |
|---|---|---|---|
| RPLAN (already held, re-extract polygons + boundary masks) | 80K | primary geometry teacher | research-use; request-based |
| [ResPlan](https://arxiv.org/abs/2508.14006) (2025) | 17K | clean vector plans w/ walls, doors, windows, connectivity graphs | check Kaggle terms |
| CubiCasa5K — **SVG polygons**, not bboxes | 5K | stylistic diversity | **CC BY-NC** — non-commercial |
| Own extraction (`rooms_extracted.json` etc., ~5K plans) | 5K | Indian adjacency/zone priors (already powering S2/S3) | owned |

**Key insight on the Indian shortage:** at the partition level, geometry is culture-agnostic —
a wall is a wall in Beijing or Chennai. The *Indian-ness* lives in the program layer PlanGen
already owns: room mixes (pooja, utility, attached baths), Vastu zone locks, NBC dimensions,
adjacency priors from the 5K extraction. So:

- Foreign datasets teach the allocation model *geometry and packing*.
- Steps 2–3 inject the *Indian program* at inference time.
- **Fine-tune later** with a small curated Indian set: build a 1-page annotation tool
  (click room corners on a plan image, pick type), annotate 200–500 plans at ~3 min each.
- Once the harness says quality is high, generated+human-approved plans become synthetic
  fine-tuning data (self-training with a human filter).

**Licensing flag:** RPLAN and CubiCasa5K restrict commercial use. Fine for R&D; before any
commercial launch, the allocation model must be retrained on ResPlan + owned/annotated data,
or licenses cleared.

---

## Part 6 — Phased Roadmap

| Phase | Duration | Deliverable | Milestone gate |
|---|---|---|---|
| **0** | ~1 week | Eval harness + 50 golden briefs + visual review page | harness runs on current engine, baseline recorded |
| **1** | 1–2 weeks | Geometric core: polygon → seeded region-grow partition → walls/doors, seeds from existing zone priors (zero ML) | **zero-overlap, full-coverage plans on all 50 briefs** — should already beat the AR engine |
| **2** | 2–3 weeks | Neural allocation model (boundary mask + graph → sequential centroid maps) trained on RPLAN polygons; swaps in for heuristic seeds | beats Phase 1 on soft scores in harness A/B |
| **3** | ongoing | Indian fine-tune (annotation tool + 200–500 plans), door/window/dimension polish, multi-floor, furniture | user-visible "real architect" quality |

Phase 1 is the insurance policy: even if all ML work stalls, the product ships clean,
valid, boundary-respecting plans. Phase 2 adds learned taste on top of a floor that can no
longer produce garbage.

### Reuse / retire

**Reuse:** steps 1–3 wholesale · GNN encoder (`gnn_encoder.py`) · CP-SAT solver (`solver.py`)
· trainer/checkpoint infra · renderer skeleton · zone/adjacency priors.
**Retire:** MoG coordinate heads · free-box decoding · SAT push-apart + wall-snap post-hoc
physics · `ar_*_b040c629.pkl` bbox dataset · the epoch-46 checkpoint.

---

## References

- RPLAN: Wu et al., *Data-driven Interior Plan Generation for Residential Buildings*, SIGGRAPH Asia 2019
- Graph2Plan: Hu et al., SIGGRAPH 2020 — [paper](https://www.researchgate.net/publication/343625455_Graph2Plan_learning_floorplan_generation_from_layout_graphs)
- WallPlan: Sun et al., SIGGRAPH 2022
- HouseDiffusion: Shabani et al., CVPR 2023
- GFLAN: *Generative Functional Layouts*, arXiv Dec 2025 — [2512.16275](https://arxiv.org/abs/2512.16275)
- ResPlan dataset (17K vector plans), arXiv Aug 2025 — [2508.14006](https://arxiv.org/abs/2508.14006)
- Survey: *Computer-Aided Layout Generation for Building Design: A Review* — [2504.09694](https://arxiv.org/html/2504.09694v1)
- Merrell et al., *Computer-Generated Residential Building Layouts*, SIGGRAPH Asia 2010 (the classic no-ML baseline)
