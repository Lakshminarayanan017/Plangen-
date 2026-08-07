# Reviewer Coverage Matrix

**Law (implementation plan §4):** every observed flaw class maps to a named
rule that detects it with geometric evidence, proven by a flaw-injection
test. A new observed flaw without a detecting rule is a P0 bug.

24 rules · flaw-injection suite: `tests/test_reviewer_rules.py`

| Flaw class (observed where) | Rule | Sev | Evidence | Injection test |
|---|---|---|---|---|
| Overlapping/corrupt geometry (v1 engine) | STR-001 | hard | verify() invariants | test_grid_plan corruption tests |
| Room lost during carving | STR-002 | hard | name lookup | engine tests |
| No entrance door | CIR-001 | hard | exterior-door scan | test_cir002_unreachable (implicit) |
| Unreachable room (v0 carver) | CIR-002 | hard | BFS over openings | test_cir002_unreachable |
| Kitchen buried deep (legacy plans) | CIR-003 | soft | hops vs norms | harness breakdown |
| Doored kitchen (user quality bar) | OPN-001 | hard | opening scan | test_opn001_kitchen_door |
| Sealed house / no flow | OPN-002 | soft | wide-opening count | harness breakdown |
| Bath opening into kitchen | HYG-001 | hard | pair-type scan | repair tests |
| Bedroom-through-bedroom door (harness 2026-07-16) | PRV-001 | soft | door pair types | harness breakdown |
| Bath door into living/dining | PRV-002 | soft | door pair types | harness breakdown |
| Area far from program (v0 carver +22%) | QLT-001 | soft | drift vs scaled targets | full-engine tests |
| Narrow habitable rooms | QLT-002 | soft | clear dims | harness breakdown |
| Sliver rooms (aspect 4.15, harness 2026-07-16) | QLT-003 | soft | worst aspect | harness breakdown |
| Below-code rooms on tight plots | NBC-001 | soft | standards table | harness breakdown |
| Landlocked habitable room (25x50 brief) | VEN-001 | soft | exterior runs + OTS adjacency | harness breakdown |
| Room without window | VEN-002 | soft | window openings incl. onto OTS | full-engine tests |
| Room hanging off a private room; fragmented hub (harness 2026-07-17) | **FSP-001** | config (soft→hard) | wide-opening component analysis | test_fsp001_* (3 tests) |
| Corridor waste (v1 layouts) | FSP-002 | soft | circulation fraction | test_fsp002 |
| Sub-3' passages / squeezed wide openings | FSP-003 | soft | passage + opening widths | test_fsp003 |
| Rooms scattered far from hub | FSP-004 | soft | mean hub distance | test_fsp004 |
| Dead single-sided walls | WAL-001 | soft | shared/internal length ratio | test_wal001 |
| Jogging band walls (visible in v0 SVGs) | WAL-002 | soft | collinear offset pairs | test_wal002 |
| Toothpick wall fragments | WAL-003 | soft | shared runs < 1'6" | test_wal003 |
| Door leaves colliding | DOR-001 | hard | swing-rect intersection | test_dor001 |
| Door hinged into corner | DOR-002 | soft | jamb-to-run-end distance | test_dor002 |
| Bedroom door swinging into hall | DOR-003 | soft | swing side vs pair types | test_dor003 |
| No cross-ventilation | DOR-004 | soft (bonus) | door/window wall axes | test_dor004 |

## Severity notes

- **FSP-001** currently `fsp001_hard=False`: the 2026-07-17 harness proved
  the reviewer detects free-space fragmentation that the current
  proposer/carver cannot yet always avoid (13/18 briefs). The violation
  count (`freespace_violations`, −10 each) is the driving metric for M3
  (CP-SAT band assignment) and M5 (transformer). It flips to hard when the
  harness shows generators satisfying it on ≥17/18 briefs.
- **WAL-003** is soft (weight 3) until settle grows a jog-merge move;
  band-boundary offsets currently make short shared runs common.

## Repairs wired (engine/repair.py)

OPN-001 → door→wide conversion · HYG-001 → seal · CIR-001 → entrance retry
at reduced widths · CIR-002 → reachability repair · FSP-001 → open room
onto free component · DOR-001 → shift colliding door along its run.

## Baseline under the 24-rule ruler (2026-07-17)

18/18 briefs valid, mean best score 75.35 (was 95.42 under 13 rules — the
ruler grew; scores are not comparable across rule-set changes), fidelity
0.846, drift 2.8%. Weakest briefs — 25x50_E (11.6), 20x45_S (41.1),
50x70_S (40.6) — are the generator-gap workload for M3/M5.

---

## Stair rules — STR-S (M2, 2026-07-30)

| Flaw | Rule | Severity | Evidence | Test |
|---|---|---|---|---|
| Multi-floor plan with no staircase | STR-S01 | hard | requested types vs `n_floors` | test_str_s01 |
| Stair face too small/wrong shape for any flight | STR-S02 | hard | face dims vs smallest buildable footprint | test_str_s02 |
| Stair buried behind other rooms | STR-S03 | hard | hops from the entrance (>2) | test_str_s03 |
| Flight arrives with no landing | STR-S04 | soft | fitted variant's landing policy | test_str_s04 |
| Stair eats the entrance frontage | STR-S05 | soft | exterior run on the entrance side | test_str_s05 |

STR-S02 checks the FITTED geometry (`plan.stairs`, attached by the
stair-fitting tier), so the reviewer and the renderer judge the same object.
The carver is sized for the dog-leg — `carve/standards.py` derives the
staircase area, min side AND min long side from `carve.stairs
.carvable_variant()`, never from a literal.

## Vertical rules — VRT (M7, 2026-07-30)

These compare TWO floors and therefore live outside the per-floor registry,
in `engine/vertical.py::check_stacking`. Same severity semantics.

| Flaw | Rule | Severity | Evidence | Test |
|---|---|---|---|---|
| Upstairs stair not over the flight below | VRT-001 | hard | footprint overlap < 90% | test_a_moved_stair_fails_vrt001 |
| Upper floor overhangs the floor below | VRT-002 | hard | built cells outside the lower footprint | test_a_different_footprint_fails_vrt002 |
| Bathroom over a living room (no stack) | VRT-003 | soft | wet-area overlap fraction | test_unstacked_wet_rooms_cost_score |
| Partition with nothing under it | VRT-004 | soft | interior wall cells over walls below | test_wall_alignment_is_measured |
| Stair arrives into a dead corner | VRT-005 | soft | openings off the upper stair | (covered by the building gate) |

VRT-001 is only satisfiable because the stair footprint is RESERVED before
carving (`carve/reserved.py`) and then FROZEN in settle
(`carve/settle.py::settle(frozen=...)`). Seeding the proposer at the right
cell was tried first and measured 0–1% overlap; freezing was added after
reservation alone still drifted to 23–59% under settle.

## Reviewer entrance semantics (M7)

`ReviewContext.entrance_room()` returns the exterior door's room on the
ground floor and THE STAIRCASE on any upper floor. Everything measured from
the entrance — CIR-001, CIR-002, CIR-003, STR-S03, the critic's hop
features — follows that one definition. Before this, the connector cut a
front door into the first floor's exterior wall (a doorway into open air)
and every entrance-relative rule measured from it.

## Rule count

33 registered rules (`engine/rules/`), plus 5 vertical rules
(`engine/vertical.py`) that need two floors to evaluate.
