# The Layout Engine — Multi-Tier Neuro-Symbolic Architecture

**Status:** Adopted design
**Date:** 2026-07-16
**Scope:** THE ENGINE ONLY — the module that takes (plot polygon + room program +
adjacency graph) and returns finished wall-graph plans. Everything else (intake,
program synthesis, rendering, app) is out of scope here.

---

## 1. Honest post-mortem: what exactly was lacking in the v1 transformer

Ranked by how much each contributed to the failure:

**#1 — Task formulation (the biggest problem, ~60% of the failure).**
The model was asked to regress *exact continuous coordinates* (cx, cy, w, h) through
Mixture-of-Gaussian heads. Continuous regression can never express "these rooms share a
wall" or "nothing overlaps" — for continuous distributions those are measure-zero events.
Sampling noise accumulated across 5 tokens × N rooms with no constraint applied at decode
time. **The transformer was not the problem; the I/O contract was.** Every published system
that works (RPLAN, Graph2Plan, GFLAN) has the network decide *discrete, coarse* things and
lets geometry code make them exact.

**#2 — Data content / preparation (~30%).**
The prep converted RPLAN's room *polygons* into axis-aligned *bounding boxes*: measured
13.4% of ground-truth room pairs overlapped, and sum-of-areas ranged 0.79–1.12× the plot.
The model was literally supervised to produce overlaps. The boundary masks RPLAN ships were
discarded, so irregular-plot conditioning was impossible. The source data was fine —
our preparation destroyed it.

**#3 — Data amount (~10%, the smallest issue).**
~80K plans is *enough* for discrete/coarse formulations (Graph2Plan and WallPlan trained on
exactly this dataset). It is on the low side for high-capacity continuous generative models
from scratch — one more reason the formulation had to change. **Verdict: content and
formulation were lagging; amount was adequate and remains adequate for the v2 formulation.**

A structural corollary: the coarser the decision the ML makes, the more effective every
training sample becomes. v2 deliberately lowers the resolution of the ML's job (which coarse
cell, which size class — not which inch) so that 80K–100K samples is comfortably sufficient.

---

## 2. Design principle

> **ML decides — the algorithm guarantees.**
> Neural tiers make every *creative, learned* decision: zones, placement order, positions,
> approximate sizes, and which candidate is best. Algorithmic tiers convert those decisions
> into exact, legal geometry. ML is load-bearing (it designs the layout); the algorithm is
> the drafting hand (it cannot be creative, and the ML cannot be exact — each covers the
> other's blind side).

This is not ML bolted onto an algorithm, nor an algorithm rescuing a fuzzy model: the
interface between them is *designed* — the realizer consumes exactly what the placer
produces (seeds + size classes), rather than fighting free-form output after the fact.

---

## 3. The tiers

```
                        ┌──────────────────────────────────────────┐
   plot polygon ───────►│ TIER 0  ENCODER                    [ML]  │
   room program ───────►│ GNN over program graph → room embeddings │
   adjacency graph ────►│ CNN over boundary raster → plot context  │
                        └───────────────┬──────────────────────────┘
                                        │ embeddings (shared conditioning)
                        ┌───────────────▼──────────────────────────┐
                        │ TIER 1  ZONE PLANNER               [ML]  │
                        │ public/service/private masks + entrance- │
                        │ hub spine sketch on coarse raster        │
                        └───────────────┬──────────────────────────┘
                        ┌───────────────▼──────────────────────────┐
                        │ TIER 2  PLACEMENT SAMPLER          [ML]  │
                        │ AR transformer, MASKED DISCRETE decoding │
                        │ room-by-room: seed cell + size class     │
                        │ sampled K× (temperature) → K proposals   │
                        └───────────────┬──────────────────────────┘
                                        │ K × (seeds, size classes)
                        ┌───────────────▼──────────────────────────┐
                        │ TIER 3  GEOMETRIC REALIZER        [ALGO] │
                        │ hub-first carving on GridPlan lattice →  │
                        │ exact partition, walls, connection types │
                        └───────────────┬──────────────────────────┘
                        ┌───────────────▼──────────────────────────┐
                        │ TIER 4  SETTLE OPTIMIZER          [ALGO] │
                        │ squeeze-&-settle wall sliding → exact    │
                        │ target areas, aligned walls, openings    │
                        └───────────────┬──────────────────────────┘
                        ┌───────────────▼──────────────────────────┐
                        │ TIER 5  CRITIC / RANKER            [ML]  │
                        │ learned plan-quality score (real-vs-     │
                        │ perturbed training) blended w/ rule score│
                        └───────────────┬──────────────────────────┘
                                        ▼
                          best plans (validated, ranked)
```

Three genuine ML models in the main path — the encoder, the placement transformer, and the
critic — doing the core design work. Two algorithmic tiers guaranteeing precision.

### Tier 0 — Encoder [ML]
- **In:** program graph (rooms + adjacency musts/pulls), boundary polygon.
- **Out:** per-room embeddings + global context vector.
- **Model:** GNN (reuse/retrain the existing `gnn_encoder.py` design) + small CNN over the
  boundary mask (64×64). ~2–4M params.

### Tier 1 — Zone Planner [ML, foldable into Tier 2]
- **In:** encoder outputs. **Out:** coarse zone masks (public/service/private) + spine
  sketch from the entrance, on a 32×32 raster clipped to the boundary.
- **Model:** small U-Net-style CNN decoder. May be merged into Tier 2 as an auxiliary
  prediction head if separate training proves redundant.

### Tier 2 — Placement Sampler [ML — the centerpiece]
- **In:** encoder outputs + zone maps. **Out:** for each room, in learned order: a **seed
  cell** on the 32×32 boundary-clipped raster + a **size class** (quantized area/aspect
  bucket).
- **Model:** autoregressive causal transformer with cross-attention to encoder outputs —
  the same family as v1, with the corrected I/O contract:
  - Output is a **classification over cells**, not continuous regression. Cross-entropy
    against real plans; no MoG heads.
  - **Masked decoding:** at every step, logits for illegal cells (outside boundary, already
    claimed, zone-inconsistent, violating a hard adjacency) are set to −∞ *before*
    sampling. Invalid layouts are not discouraged — they are **unsamplable**. This is the
    single mechanism that makes a v1-style failure impossible by construction.
  - Sampling with temperature → K diverse proposals per request (diversity is a feature:
    Tier 5 picks the best).
- **Training data:** RPLAN (~80K) prepared *correctly this time* — true room polygons →
  centroid cells + size classes, boundary masks kept, zones derived from room types;
  ResPlan (17K vector) for fine-tune/validation; ×8 flip/rotation augmentation; Indian
  fine-tune later via annotated set. Val metric = harness layout metrics, never loss alone.

### Tier 3 — Geometric Realizer [ALGO]
- Hub-first carving on the GridPlan lattice: circulation spine from the entrance first,
  rooms attached around it at their seed positions with their size classes as budgets.
  Zero overlap / full coverage by construction. Deterministic given a proposal.

### Tier 4 — Settle Optimizer [ALGO]
- Squeeze-&-settle wall sliding (3" steps, annealed) to hit exact target areas, clean wall
  jogs, enforce NBC minimums; then connection typing (openness gradient: kitchen/dining
  doorless wide openings; doors only on private rooms) and windows/OTS.

### Tier 5 — Critic / Ranker [ML]
- **In:** finished plan features (adjacency satisfaction, zone conformity, circulation
  depths, daylight, proportion stats, wall alignment) and/or the plan raster.
- **Model:** starts as a gradient-boosted / small MLP scorer trained **real-vs-perturbed**
  (real plans = positives; the same plans with swapped rooms / jittered walls = negatives —
  self-supervised, no labeling). Blended with the explicit rule score. Upgradable to a
  raster CNN later.

---

## 4. The Orchestrator

One conductor owns the loop; tiers are plug-ins behind typed contracts.

```
def generate(request) -> List[Plan]:
    ctx        = tier0.encode(request)                       # once
    zones      = tier1.plan_zones(ctx)        or fallback.zones(request)
    proposals  = tier2.sample(ctx, zones, k=K) or fallback.propose(request, k=K)
    finished   = []
    for p in proposals:
        plan = tier3.realize(request, p)                     # never fails structurally
        plan = tier4.settle(plan, request)
        issues = validator.check(plan)                       # rules catalog
        if issues.hard:
            plan = repair(plan, issues)                      # targeted, ≤3 rounds
            if validator.check(plan).hard:  continue         # discard, keep others
        finished.append(plan)
    return rank(finished, critic=tier5, rule_score=validator.soft_score)[:N]
```

Orchestrator responsibilities:
- **Fallbacks:** every ML tier has a statistical-prior twin (zone priors, seed sampling
  from `zone_patterns` data). Out-of-distribution plot? Low placer confidence (entropy
  threshold)? The engine degrades gracefully and still ships a valid plan. The engine can
  therefore run end-to-end from day 1, before any model is trained.
- **Diversity control:** K, temperatures, and parti variation per candidate.
- **Telemetry:** per-tier failure attribution (which tier caused discards) — this is how we
  know where to invest next, instead of guessing.

## 5. Contracts between tiers (the module boundaries)

```
EngineRequest   plot polygon (lattice), entrance, RoomSpecs, adjacency graph, options
EncodedContext  room embeddings [N×d], global vector, boundary raster
ZonePlan        32×32 zone masks + spine polyline
LayoutProposal  ordered [(room_id, seed_cell, size_class)], confidence
Plan            GridPlan (walls, faces, openings) + metadata
Verdict         hard violations [], soft score breakdown {}
```

Each tier is a separate sub-package with its own tests and can be developed, trained, and
swapped independently — `engine/tier2_placer/` doesn't know carving exists.

---

## 6. Alternatives to the AR transformer (considered honestly)

| Option | Verdict |
|---|---|
| **Diffusion (HouseDiffusion-style)** | Real contender, strong published quality. But: constraint enforcement mid-denoising is much harder than logit masking, training is heavier for Colab budgets, inference slower, and output still needs snapping. Keep as a Tier-2b *experiment* behind the same contract — the architecture makes it swappable. |
| **MaskGIT-style parallel masked transformer** | Same discrete formulation, faster sampling, slightly weaker ordering control. Legitimate variant of the same backbone; easy pivot if AR sampling feels slow. |
| **GAN (HouseGAN++)** | Superseded; unstable training; no. |
| **LLM fine-tuning on serialized plans** | Weak geometric precision, heavy; no for the engine (fine for the conversational intake elsewhere). |

**Verdict: AR transformer with masked discrete decoding.** It keeps explicit step-by-step
constraint control (the property v1 lacked), it's trainable on a Colab budget, it reuses
the team's existing transformer/trainer experience, and it is the most explainable of the
options — every placement step has inspectable masked logits.

---

## 7. Build order (engine only) and gates

| Milestone | What | Gate |
|---|---|---|
| E1 | Contracts + orchestrator + fallback tiers (priors) | engine runs end-to-end, all-algorithmic, on the demo briefs |
| E2 | Tier 3 hub-first realizer (connected free space, openness gradient) | reference-style topology on 20×45 brief |
| E3 | Tier 4 settle | areas within ±5% of target on harness |
| E4 | Data prep v2 (polygons + boundaries + seeds/size classes) | prep validator: 0% GT overlap, boundary kept |
| E5 | Tier 2 placer trained; swap behind contract | beats fallback priors on harness A/B |
| E6 | Tier 5 critic; best-of-K ranking | top-1 selection beats random-of-K on human eval |
| E7 | Tier 1 zone planner (or fold into Tier 2) | further harness gain, else fold |

Each ML tier is only merged when it **beats its algorithmic fallback on the harness** —
ML earns its place in the engine by measurement, which is also the strongest possible
story to tell about the project: not "we used a transformer", but "our transformer
measurably beats a strong statistical baseline at layout design, and our decode-time
constraint masking makes invalid output unsamplable."

---

## 8. Risks & amendments (honest review, adopted)

1. **Critical path = Tier 3, not the ML.** Hub-first carving from arbitrary seeds on
   arbitrary polygons is the hardest engineering in the engine. It gets the most tests and
   the most schedule. Mitigation: rect/L plots first; infeasible seeds snap to the nearest
   feasible cell rather than failing.
2. **Domain gap is real.** RPLAN = Chinese apartments (no parking/pooja/OTS, no
   independent-house entry logic). Expected outcome: the freshly trained placer LOSES to
   the Indian statistical fallback on Indian briefs until Indian fine-tuning is done.
   → The annotation tool + 200–500 Indian plans move UP the schedule (start alongside E5,
   not after).
3. **Proposal-fidelity metric (mandatory orchestrator telemetry).** Measure how far the
   finished plan deviates from the Tier-2 proposal (seed displacement, size-class drift).
   If fidelity is low, the realizer is muting the model and every ML-vs-fallback A/B is
   meaningless — fix obedience before judging the model.
4. **Data prep gets an eyeball gate.** Besides the 0%-overlap metric gate: render 100
   random prepped training samples and LOOK at them before any training run. (The v1 bbox
   disaster was visible to the naked eye and nobody looked.)
5. **Contracts carry floors from day 1.** `EngineRequest` includes floor index + vertical
   skeleton context even while multi-floor is deferred — retrofitting contracts later rots
   the orchestrator.
6. **Tier 1 is expected to fold into Tier 2** as an auxiliary head; don't build it
   standalone beyond a week of effort.
7. **Log the user's pick whenever a human chooses among K candidates** — free preference
   data; later fine-tunes the critic from "realism" toward taste.
8. **The kill-risk is process, not architecture:** judging by eyeball instead of the
   harness. Harness + fidelity metric + A/B gates are non-negotiable; every other failure
   above is visible and recoverable if they exist.
