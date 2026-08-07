# The Carving Algorithm — Wall-Based Plan Synthesis
## "Squeeze and Settle": how PlanGen carves a plan like an architect

**Status:** Proposed (refines Stage D of `plangen_v2_master_blueprint.md`)
**Date:** 2026-07-16
**Reference example:** 20'×45' G-floor row house (3 bed, 2 C.bath, kitchen/dining/drawing,
parking, stair, 2×OTS) — the target quality bar.

---

## 0. The two decisions this doc locks in

**Decision 1 — Representation: WALLS, not blocks.**
A plan is a rectilinear **wall graph**: wall segments (with thickness) forming a planar
subdivision of the plot polygon. Rooms are the *faces* of that graph. Openings (doors, wide
openings, windows) are intervals on wall segments.

Consequences, all of which blocks cannot give:
- **Overlap is unrepresentable.** Faces of a planar subdivision cannot overlap. Zero-overlap
  is structural, not an optimization target.
- **Coupled adjustment.** Moving one wall simultaneously grows one room and shrinks its
  neighbor — area is conserved, walls stay shared. This is literally how an architect
  adjusts a plan on tracing paper, and it is the formal version of "recursively varying
  room size and position within constraints."
- **Shared walls are one entity.** 4.5" partition between two bedrooms is ONE wall, so
  drawings, dimensions, and door placement are exact by definition — no butted block edges.
- **Partial walls are natural.** A "wall between dining and drawing with a 6' opening" is a
  wall segment with an opening interval — first-class, not a hack.

**Decision 2 — Division of labor: ML proposes, the optimizer disposes.**
Machine learning CANNOT deliver final wall coordinates at architect precision — continuous
neural outputs are fuzzy, and precision lives at 3-inch resolution. ML's proper jobs here:
choosing the *parti* (typology/band ordering priors), room→band assignment, connection-type
priors, warm-start seeds, and candidate ranking. Every coordinate the user sees comes from
the deterministic optimizer below. (Phases run with zero ML first; priors from our extracted
statistics stand in.)

---

## 1. Precision substrate

- Grid module: **3" (76 mm)** for wall centerlines; displayed dims rounded to 1".
- Wall thicknesses: external 9" (230 mm), internal 4.5" (115 mm), plumbing walls 6"–9".
- All room dims/areas are **clear inner** face-to-face (like the example's "10' × 9'").
- Area budget includes a wall allowance (gross − Σclear ≈ 8–12%); the budget solver
  (blueprint §4) works on clear areas.

---

## 2. Pipeline overview

```
RoomSpecs + plot polygon + skeleton (stair, wet stacks, entrance)   [from blueprint B, C]
   │
   ▼
P1  PARTI SELECTION        typology + band structure (the architect's first sketch)
P2  BAND ALLOCATION        depth-axis zoning bands, budgeted depths
P3  STRIP CARVING          cut bands into rooms with wall insertions (never place blocks)
P4  SQUEEZE & SETTLE       iterative wall-sliding optimization (the precision engine)
P5  TOPOLOGY MOVES         discrete jumps when sliding stalls (swap/rotate/reassign)
P6  CONNECTION TYPING      door vs wide-opening vs open-plan per adjacency + windows/OTS
P7  FREE-SPACE AUDIT       circulation skeleton check; loops back to P5 if deficient
```

P3–P5 loop until converged; the whole of P1–P7 runs K times (different parti/seeds) for
best-of-K selection (blueprint §8).

---

## 3. P1 — Parti selection (typology)

Inputs: plot polygon, aspect ratio, road side(s), attachment context (shared side walls?),
program size. Output: a **parti** = band axis + band order + spine plan + stair/entry zone.

Catalog of partis (extensible; each is a parametric template, not a fixed plan):
| Parti | Trigger | Structure |
|---|---|---|
| Row-house spine | aspect ≥ 1:1.8, side walls shared | central longitudinal spine wall; bands front→rear: PUBLIC (parking/drawing) → HUB (dining+kitchen) → PRIVATE (beds/baths); OTS shafts mandatory for interior wet/hab rooms |
| Corner-plot L | two road sides | public wing along each road, private at inner corner |
| Square courtyard | aspect ≤ 1:1.3, area ≥ 1800 sqft | rooms ring a central hall/courtyard |
| Wide frontage | frontage > depth | lateral bands, entry mid-front |

The 20×45 example is the Row-house spine parti (1:2.25, shared side walls). Parti selection
is scoring, not hardcoding: each parti computes feasibility (can budget fit? stair reachable?
ventilation solvable?) and a prior score (from data: which parti do real plans of this
shape/BHK use). ML later replaces the prior table with a learned classifier.

## 4. P2 — Band allocation

Divide the depth axis into bands (front→rear), assign rooms to bands by privacy tier +
catalog floor/zone rules, set initial band depths from the area budget solver. Bands are
soft — a room may later straddle a boundary (the example's drawing room sits partly beside
the dining band); bands guide carving, walls decide reality.

## 5. P3 — Strip carving (initial partition, architect-style)

Within each band, insert walls to subdivide — like drawing lines on the plot:

```
carve(region, rooms):
  if len(rooms) == 1: assign(region, rooms[0]); return
  axis   = choose_cut_axis(region, rooms)        # prefer cuts landing on spine/stack lines
  groups = split_rooms(rooms, axis)              # by adjacency affinity + area balance
  x      = cut_position(region, groups)          # proportional to group area budgets,
                                                 # snapped to 3" grid + alignment attractors
  insert_wall(region, axis, x)
  carve(left, groups.left); carve(right, groups.right)
```

Not pure guillotine: **T-junctions and wall retractions are allowed** after carving (a wall
may be trimmed to create an L-shaped room or a hub space — the example's dining hub is the
result of retracting the walls around the circulation zone). Skeleton cells (stair, OTS,
shafts) are pre-claimed before carving and are uncuttable.

Output: a valid wall graph — already zero-overlap, full-coverage — but with mediocre
dimensions. Precision comes next.

## 6. P4 — Squeeze & Settle (the core optimizer)

Formalization of "recursively varying sizes/positions within constraints, again and again":

- **Variables:** one scalar per movable wall segment (vertical wall → its x; horizontal →
  its y). External walls, stair walls, wet-stack walls: frozen or tightly bounded.
- **Hard constraints:** NBC clear minimums (width and area per room type), min wall segment
  length 1'6", opening feasibility (a wall carrying a required door keeps ≥ door+stub room),
  plot boundary, stair/stack footprints, max aspect per room type.
- **Score (soft):**
  `Σ w_a·areaErr + w_r·aspectErr + w_adj·adjacencyGap + w_align·wallJogs
   + w_day·daylightDeficit + w_circ·circulationExcess + w_zone·zoneDrift + w_vastu·vastuDrift`
  - `areaErr`: |clear area − RoomSpec.target| / target (quadratic near target, linear far)
  - `wallJogs`: count of offsets between nearly-collinear walls < 9" (drives the clean
    continuous wall lines visible in the example)
  - `daylightDeficit`: habitable room with no exterior/OTS wall span ≥ 3'
  - `circulationExcess`: circulation area beyond target fraction (free space is *budgeted*,
    not leftover — see P7)
- **Search:** coordinate descent with annealing.
  ```
  repeat until no accepted move for full sweep (typ. 200–600 sweeps, <1s):
    for each movable wall (random order):
      for δ in {±3", ±6", ±12"}:
        tentatively slide wall by δ (auto-clips to hard constraints)
        Δscore = incremental re-evaluation of ONLY affected rooms/edges
        accept if Δscore < 0, or with prob e^(−Δ/T)   # T cools each sweep
  ```
  Sliding a wall re-dimensions its two (or more) adjacent rooms simultaneously — the
  coupled, conservation-of-area adjustment an architect performs. With fixed topology, room
  area is linear in each single wall coordinate, so per-move evaluation is exact and O(1).

## 7. P5 — Topology moves (when sliding can't fix it)

If settle converges with hard violations or a poor score plateau, make a discrete jump and
re-settle (accepted by annealing on total score):
`swap_rooms(A,B)` · `rotate_cut(region)` (vertical↔horizontal split) ·
`reassign_band(room)` · `retract_wall(w)` / `extend_wall(w)` (creates/removes hub spaces,
L-rooms) · `slide_room_along_band` · `flip_stair_side` · `move_OTS`.
Typical budget: ≤ 40 topology moves, each followed by a short settle. This inner loop —
carve → settle → jump → settle — is exactly the "again and again" refinement requested, and
it terminates because annealing temperature and move budget are finite.

## 8. P6 — Connection typing (walls, openings, the "partial wall" effect)

For every adjacency edge in the final graph, the catalog assigns a **connection type**:

| Type | Between | Geometry |
|---|---|---|
| `door` | private/service rooms (beds, baths, utility, parking→house) | 2'6"–3'6" leaf, swing-arc collision-checked, hinge at wall stub ≥ 4.5" |
| `wide_opening` | public↔public: kitchen↔dining, dining↔drawing, hall↔foyer | 4'–7' clear opening; **wall stubs ≥ 9" remain at both ends** (structure + the partial-wall reading of the example); optional arch/beam over |
| `open_plan` | user-requested merges | no wall; zone boundary dashed on drawing |
| `sealed` | forbidden pairs (toilet↮kitchen), neighbor walls | full wall, no opening |

Then **windows** are placed on exterior/OTS walls sized to meet each room's NBC ventilation
ratio exactly (rounded up to standard sizes), cross-ventilation scored. **OTS shafts** are
first-class rooms in the catalog: mandated when a bath/habitable room has no exterior wall
(the narrow-plot case), min ~3'×4', positioned adjacent to the rooms they serve — this rule
is what foreign datasets will never teach and the catalog encodes directly.

## 9. P7 — Free-space audit (free space is designed, not leftover)

Circulation in good small plans flows THROUGH rooms (the example's dining hub), not through
corridors. Audit on the final graph:
1. Build the **free-space skeleton**: union of circulation faces + traversable spans of
   rooms whose connection types permit through-movement (`wide_opening`/`open_plan`).
2. Check: every room reachable from entrance; path clear width ≥ 3'0"; privacy — no path to
   a public room passes through a bedroom/bath; depth norms (entry→kitchen ≤ 2 hops,
   entry→drawing = 1) from `circulation_patterns.json`.
3. Circulation fraction within band (8–14% target): too little → rooms are landlocked →
   topology move; too much → squeeze walls to return area to rooms.
Fail → back to P5 with the specific deficiency as a directed move hint.

---

## 10. Worked trace — the 20'×45' example (what the pipeline should reproduce)

1. **P1:** aspect 1:2.25 + shared side walls → Row-house spine parti; entry at south
   (road); stair reserved bottom-left strip; 2 wet stacks (one per bath cluster); OTS
   required (interior baths inevitable).
2. **P2:** bands front→rear: [parking+drawing] → [kitchen+dining hub] → [bed+bath+OTS] →
   [bed+bed at rear]. Budget solver depths ≈ 12' / 10' / 11' / 10'2" (+ walls).
3. **P3:** spine wall splits 20' into ~10'|10'; carving inserts cross walls per band;
   stair+OTS pre-claimed; dining hub created by retracting the wall between dining and the
   central passage zone.
4. **P4:** walls slide in 3" steps: kitchen settles at 7'6" clear (donating 2'6" to dining
   circulation), beds settle at 10'×9' and 10'×8'2" hitting area targets, baths clamp at
   NBC minimums (4'×5', 4'×6'6"), jog penalty pulls the bath/OTS walls collinear with
   bedroom walls.
5. **P6:** doors with arcs on all beds/baths/parking; `wide_opening` kitchen↔dining and
   dining↔drawing (stubs left = the partial walls); windows on front/rear walls; bath
   windows onto OTS.
6. **P7:** free-space skeleton: parking → drawing → dining hub → {kitchen, stair, beds} —
   all depths within norms, no corridor waste. PASS → candidate scored, joins best-of-K.

---

## 11. Answers to the open questions

- **"Will ML give this precision?"** No — and it shouldn't try. ML picks parti/assignment/
  seeds and ranks candidates (taste); the wall optimizer owns every final coordinate
  (precision). This split is also what the strongest 2025 research (GFLAN's
  topology-vs-realization factorization) converged on.
- **"Are rooms placed like blocks?"** No. Nothing is ever *placed*. Space is *subdivided* by
  wall insertions, then walls slide. Overlap cannot occur at any point in the process.
- **"Recursively adjusting sizes/positions within constraints?"** Yes — P4 (continuous
  sliding) + P5 (discrete jumps) under P2's budgets, with annealing so early iterations
  explore and late iterations refine. Deterministic per seed, ~1–2 s per candidate floor.

## 12. Implementation notes

- New module `modules/step4_generate/carve/`: `wall_graph.py` (planar subdivision, faces,
  openings), `parti.py`, `band_alloc.py`, `strip_carver.py`, `settle.py` (optimizer),
  `topology_moves.py`, `connections.py`, `freespace.py`.
- Reuse: RoomSpec/budget (blueprint §3–4), skeleton (§5), catalog (§2), CP-SAT
  (`solver.py`) usable as an alternative exact settle for small floors; renderer consumes
  the wall graph directly (double-line walls, arcs, dims).
- The wall graph is the SINGLE output format for renderer, validator, and future DXF.
- Unit-test seams: each P-stage is pure (state in, state out) → goldens per stage.
