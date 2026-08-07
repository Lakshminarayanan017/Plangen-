# PlanGen v2 — Master Blueprint
## Precision Room Specs, Rule-Governed Generation, and Multi-Floor Coherence

**Status:** Proposed (fresh plan, reusing v1 assets)
**Date:** 2026-07-15
**Companion:** `partition_first_redesign.md` (the geometry/ML core; this doc is the full system around it)

---

## 0. Design Tenets

1. **Partition-first.** Exact geometry always comes from deterministic space-partitioning;
   neural nets only propose topology/allocation. (Established in the companion doc.)
2. **One Rules Catalog, three consumers.** Every architectural rule lives in ONE declarative
   catalog, consumed by (a) the generator as constraints, (b) the repair loop as objectives,
   (c) the validator as gates. A rule that exists only in code is a bug.
3. **The building is designed vertically first, horizontally second.** Staircase, plumbing
   stacks, and structure are decided *before* any floor layout, then every floor is generated
   against that shared skeleton. Floors are never generated independently.
4. **Precision by construction.** All geometry lives on a 6-inch (152 mm) module grid; walls
   have real thickness (9" / 230 mm external, 4.5" / 115 mm internal); every displayed
   dimension is buildable feet-inches. No float soup.
5. **Never ship the first draft.** Generate K candidates, score, rank, repair, gate.

---

## 1. System Overview

```
        brief + plot image/dims
                 │
   ┌─────────────▼─────────────┐
   │ A. INTAKE                 │  plot polygon (any shape), road side, entrance,
   │ (step1 — keep, extend)    │  orientation, setbacks → buildable polygon per floor
   └─────────────┬─────────────┘
   ┌─────────────▼─────────────┐
   │ B. PROGRAM ENGINE          │  RoomSpec list w/ precise areas & characteristics
   │ (steps 2-3 — keep, formal- │  + floor assignment + area budget reconciliation
   │  ize into Rules Catalog)   │  (§3, §4)
   └─────────────┬─────────────┘
   ┌─────────────▼─────────────┐
   │ C. VERTICAL SKELETON       │  staircase, wet stacks, structural lines,
   │ (new)                      │  light wells — fixed across all floors (§5)
   └─────────────┬─────────────┘
   ┌─────────────▼─────────────┐
   │ D. PER-FLOOR LAYOUT        │  neural centroid allocation → seeded region
   │ (companion doc)            │  growing → partition; floor i conditioned on
   │                            │  skeleton + floor i−1 (§6)
   └─────────────┬─────────────┘
   ┌─────────────▼─────────────┐
   │ E. OPENINGS & DETAIL       │  doors w/ swing clearance, windows w/ NBC
   │ (new)                      │  ventilation ratios, stair detail (§7)
   └─────────────┬─────────────┘
   ┌─────────────▼─────────────┐
   │ F. QUALITY ENGINE          │  best-of-K, rule scoring (+ learned critic later),
   │ (new)                      │  targeted repair, hard validation gate (§8)
   └─────────────┬─────────────┘
                 ▼
        multi-floor plan sheets (renderer — keep, extend)
```

---

## 2. The Rules & Specs Catalog (single source of truth)

Formalize everything now scattered across `nbc_schema.py`, `indian_standards.py`,
`enricher_rules.json`, `vastuRules1.json`, and hardcoded logic into one versioned catalog
(`rules/catalog/*.yaml`). The existing `ConstraintSeverity` enum (mandatory / recommended /
guideline) is exactly right — keep it.

**Rule schema:**

```yaml
id: BAL-001
title: Balconies are upper-floor elements
severity: mandatory
applies_to: { room_type: balcony }
predicate: room.floor >= 1
on_violation:
  repair: substitute            # balcony on floor 0 → verandah / sit-out / porch
  substitute_type: verandah
source: CONV (Indian residential convention; NBC silent)
```

```yaml
id: ADJ-007
title: Toilet must not open directly into kitchen or dining
severity: mandatory
applies_to: { room_type: [bathroom, toilet] }
predicate: none(door.connects(room, r) for r in [kitchen, dining_room])
on_violation: { repair: relocate_door, fallback: add_lobby_buffer }
source: NBC 2016 Part 3 (hygiene separation)
```

**Catalog sections** (≈120–180 rules total; most already exist informally):

| Section | Examples | Today's source |
|---|---|---|
| Floor-level rules | balcony ≥ F1; verandah/porch/parking = F0 only; terrace = top; kitchen on F0 unless duplex-with-upper-kitchen requested; elderly bedroom pref F0 | `floor_assignment_rules` in enricher_rules.json (formalize) |
| Dimensional minimums | habitable ≥ 9.5 m², kitchen ≥ 4.5 m², bath ≥ 1.8 m², min widths, ceiling heights | `indian_standards.py`, nbc_schema |
| Adjacency musts / must-nots | kitchen↔dining pull; bed↔attached-bath must; kitchen↮toilet; pooja↮toilet wall-share | `adjacency_rules`, `forbidden_adjacencies` |
| Light & ventilation | habitable rooms need exterior wall; window ≥ 1/6 floor area (habitable), 1/10 (kitchen); no borrowed-light bedrooms | `indian_standards.py` ventilation table |
| Circulation | every room reachable from entrance without passing through a bedroom (privacy rule); corridor ≥ 0.9 m; door swings don't collide | `circulation_rules` |
| Stair & vertical | width ≥ 0.9 m, riser ≤ 190 mm, tread ≥ 250 mm, headroom ≥ 2.1 m, landing ≥ stair width; identical stair footprint on every served floor; arrival lobby on each upper floor | partially in room_resolver (staircase implicit room) |
| Structure & services | upper walls bear on lower walls or beams (alignment tolerance 115 mm); bathrooms/kitchen stack vertically across floors (wet-stack distance ≤ 1.5 m); no toilet above pooja/kitchen | **new** |
| Plot & bylaws | setbacks by plot category, ground coverage %, FSI/FAR cap on total built area, balcony projection ≤ 0.9–1.2 m into setback | `get_setbacks()`, `plot_budget_rules` |
| Vastu (user-toggled) | pooja NE, kitchen SE, master SW, entrance auspicious directions — severity `recommended` unless user says strict | vastuRules1.json, vastu_mapper.py |

Ship with a `catalog lint` command (every rule parseable, predicate executable, has source)
and a `catalog coverage` report (which rules the validator actually exercises on the golden set).

---

## 3. RoomSpec — the precision contract for every room

The Program Engine's output (per room) becomes a rich, validated object. This is where
"room characteristics should be accurate" is won:

```
RoomSpec:
  type, display_name, floor                  # bedroom, "Master Bedroom", 1
  area_target, area_min, area_max            # from conditional priors + NBC clamps (§4)
  aspect_band                                # e.g. [1.0 … 1.6] for bedrooms
  needs_exterior_wall: bool                  # habitable → true
  orientation_pref                           # e.g. master: SW (Vastu), kitchen: SE
  zone_pref                                  # front / middle / back  (from zone_patterns)
  privacy_tier                               # public / semi / private → circulation depth
  adjacency_must / adjacency_pull / adjacency_forbid
  attached_to                                # attached bath → its bedroom (same floor, shared wall)
  wet_room: bool                             # → must sit on a wet stack (§5)
  min_door_width, window_area_min            # from catalog
  vastu_lock: hard | soft | none
```

### Where the numbers come from — conditional size priors, not static ranges

Static min/max ranges produce same-y, poorly proportioned rooms. Instead, fit **conditional
distributions from real data** (your 5K Indian extraction + ResPlan/RPLAN):

> P(area_fraction | room_type, total_floor_area, n_rooms_on_floor, bhk)

Implementation: simple quantile lookup tables (deciles per bucket) — deterministic,
auditable, no model risk. `area_target` = median of the matching bucket; `area_min/max` =
NBC-clamped p10/p90. A 3BHK on 1,400 sq ft gets a 140 sq ft master bedroom, not the same
"12×10" a 2,400 sq ft plot gets.

---

## 4. Area Budget Reconciliation (exactness, per floor)

Because the partition covers 100% of the buildable polygon, room areas must sum *exactly* to
floor area — so precision starts at the budget, not at drawing time:

```
usable(floor_i) = polygon_area(floor_i) − stair − wet_shafts − walls_allowance
circulation_i   = c · usable          (c ≈ 8–14%, from circulation_patterns.json)
room_budget_i   = usable − circulation_i
```

Deterministic waterfall solver: start every room at prior-median `area_target` → scale all
proportionally to hit `room_budget_i` → clamp violators to [area_min, area_max] →
redistribute the residual over unclamped rooms by prior weight → if infeasible (mins alone
exceed budget), fail UP to the Program Engine to drop/shrink/move a room to another floor —
never squeeze below NBC minimums silently.

Output: every room enters layout with an exact, feasible, data-calibrated target area.

---

## 5. Vertical Skeleton — how multi-floor coherence is achieved

**The single biggest quality lever for multi-floor plans.** Decided once, before any floor
layout, then imposed on all floors as pre-claimed geometry:

1. **Floor assignment of the program** (extends existing room_resolver logic): public+wet on
   F0 (living, kitchen, dining, common toilet, parking/porch), private above (bedrooms,
   family lounge, balconies), terrace on top. Respects FSI/coverage budget: if program area >
   coverage × plot, rooms overflow upward automatically.
2. **Staircase placement** — the keystone. Choose location scoring: reachable from entrance
   lobby (depth ≤ 2 rooms), compact dead-corner or central-core position, identical footprint
   on all served floors, headroom/landing feasibility, Vastu (SW/W/S preferred). Its cells
   are pre-claimed on every floor before region growing runs.
3. **Wet stacks** — 1–2 vertical service lines. All wet rooms (`wet_room: true`) on every
   floor are constrained to touch a stack (distance ≤ 1.5 m). This is what real plumbing
   costs demand, and it automatically produces the bathroom-above-bathroom alignment that
   makes plans read as professional.
4. **Structural lines** — the main shared walls of floor 0 become soft alignment attractors
   for floor 1's region growing (reward walls that land on walls, tolerance 115 mm), plus a
   hard check: no upper heavy wall spanning mid-room below without a beam flag.
5. **Light wells / open-to-sky** for deep plots (guarantees interior habitable rooms get
   ventilation compliance on large footprints).

### Floor-by-floor generation order (autoregressive across floors — the right granularity)

```
skeleton → floor 0 partition → floor 1 partition | (skeleton, floor-0 walls) → …
```

Floor i's region grower input: buildable polygon(i) + pre-claimed skeleton cells + wall map
of floor i−1 (alignment reward) + stair arrival cell (its lobby/passage seed). Inter-floor
interaction is therefore *structural input*, not a post-hoc check.

**Inter-floor validation gates:** stair continuity + landings on every floor; wet-stack
compliance; wall-bearing alignment score ≥ threshold; balcony has support below (room/beam)
and projection within bylaw; terrace accessible; F1+ area ≤ F0 area unless cantilever flag.

---

## 6. Per-Floor Layout

Per the companion doc: neural centroid allocation (boundary raster + GNN program-graph
embeddings → sequential per-room centroid probability maps) seeds a deterministic region
grower; cell costs blend area-deficit, aspect penalty, zone prior, adjacency pull,
exterior-wall need, and (new) skeleton attractions from §5. Wall straightening → rectilinear
partition on the 6" grid. Phase 1 runs with heuristic seeds (zone priors) — no ML needed to
get clean plans; the model upgrades taste in Phase 3.

---

## 7. Openings & Detail (where plans start looking like an architect drew them)

- **Doors:** on shared walls per adjacency graph; hinge side away from switches/corners;
  swing arcs collision-checked (90° arc must not hit fixtures/other swings); widths from
  catalog (main 1.0 m, internal 0.9 m, bath 0.75 m); main door orientation per Vastu setting.
- **Windows:** exterior walls only; sized to hit the NBC ventilation ratio for that room
  exactly (round up to standard sizes: 4'×4', 6'×4' …); cross-ventilation bonus scored;
  sill heights from catalog; no window into setback-violating neighbor wall (privacy).
- **Stairs:** actual flight computed (riser count from floor height 10' → 17 risers @ ~178 mm,
  tread 250 mm, landing) — drawn for real, not a symbol box.
- **Wall graph output:** walls as first-class entities with thickness and openings — the
  renderer draws double-line walls, door arcs, window sills, room labels with areas in
  ft-in, and dimension lines. This output format is also what a future DXF export needs.

---

## 8. The Quality Engine

1. **Best-of-K:** run D(+C variations) with K = 8–16 seeds (region growing is fast; this is
   cheap). Multi-floor candidates are generated as complete buildings, not per-floor mixes.
2. **Scoring:** total = hard-gate pass (binary) → weighted soft score: adjacency satisfaction,
   zone/Vastu %, wall-alignment (inter-floor), compactness, daylight (habitable exterior
   exposure), circulation efficiency (entry→kitchen hops etc. vs `circulation_patterns.json`
   norms), proportion sanity. Later (Phase 4): a learned critic (real-vs-perturbed training on
   ResPlan) blended in — but rule score alone ships first.
3. **Targeted repair:** every catalog rule carries a `repair` strategy (substitute room type,
   steal cells from the slack-richest neighbor, relocate door, add lobby buffer, re-seed one
   room). Repair → re-validate loop, max 3 iterations, then discard candidate.
4. **Hard gate:** zero overlap (by construction), 100% coverage, all `mandatory` rules pass,
   all inter-floor gates pass. A plan failing the gate is never shown to a user.
5. **Golden harness** (from companion doc) extended with multi-floor briefs: G+1 3BHK on
   40×30, duplex on irregular pentagon, G+2 on 50×70, etc. Every change is A/B'd here.

---

## 9. Build Phases (fresh plan, maximum reuse)

| Phase | Weeks | Build | Reuses |
|---|---|---|---|
| **0. Foundations** | 1–1.5 | Rules Catalog (migrate all existing rule sources) + validator + golden harness (incl. multi-floor briefs) + visual review grid | nbc_schema.py, indian_standards.py, enricher_rules.json, vastuRules1.json, tpl_viewer.html |
| **1. Geometric core** | 1.5–2 | 6"-grid region grower on arbitrary polygons, heuristic seeds, wall straightening, area budget solver, single floor | grid.py, zone/adjacency priors, solver.py (refine) |
| **2. Vertical system** | 2 | skeleton planner (stair, wet stacks, structural lines), floor-conditioned generation, inter-floor gates, multi-floor renderer sheets | room_resolver floor logic, renderer.py |
| **3. Learned allocation** | 2–3 | centroid-map model on RPLAN polygons + boundary masks (+ ResPlan), swaps in for heuristic seeds, A/B on harness | gnn_encoder.py, trainer infra, Colab workflow |
| **4. Detail & taste** | ongoing | doors/windows/stair detail, best-of-K + critic, Indian fine-tune via annotation tool, DXF export | — |

Gate to advance a phase: harness pass-rate and soft-score beat the previous phase on all 50+
golden briefs. Phase 1+2 alone (zero ML) must already produce clean, multi-floor,
rule-compliant plans — that's the quality floor; ML only raises the ceiling.

---

## 10. What this buys, mapped to the stated goals

| Goal | Mechanism |
|---|---|
| Accurate room sizes/positions | conditional size priors (§3) + exact area budget (§4) + zone/adjacency-driven growth (§6) |
| Accurate room characteristics | RoomSpec contract + catalog rules per type (§2, §3) |
| "Balcony can't be on ground floor" class of rules | declarative catalog w/ floor predicates + auto-substitution repair (§2) |
| Floor-to-floor connection quality | vertical skeleton: stair keystone, wet stacks, wall bearing (§5) |
| Inside-floor interaction quality | adjacency graph + privacy-tier circulation + openings engine (§6, §7) |
| No overlaps / crap layouts ever | partition-by-construction + hard gate (§6, §8) |
| Architect-grade look | real walls/doors/windows/dimensions + best-of-K selection (§7, §8) |
