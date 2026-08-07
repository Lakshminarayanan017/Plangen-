# PlanGen Remastered

Clean-room rebuild of the PlanGen layout engine on the **partition-first / wall-graph**
architecture. Design docs (in the legacy repo):

- `../docs/plans/partition_first_redesign.md` — why the old AR engine failed, the v2 core
- `../docs/plans/plangen_v2_master_blueprint.md` — full system (Rules Catalog, RoomSpec,
  vertical skeleton, quality engine)
- `../docs/plans/carving_algorithm.md` — the carving pipeline this package implements

## Principles

1. Rooms are never *placed*; space is *split* by wall insertions → overlap is impossible.
2. All geometry is integer cells on a **1.5-inch lattice** (4.5" wall = 3 cells,
   9" wall = 6 cells). No floats in the geometry core.
3. ML proposes (parti, seeds, ranking); the deterministic optimizer owns every coordinate.
4. Every mutation is followed by `verify()` — invariants are checked, not assumed.
5. Every stage renders to SVG — quality is judged visually and by harness, never by loss.

## Layout

```
core/     units.py       lattice units, ft-in formatting
          grid_plan.py   GridPlan: ownership grid, splits, faces, walls,
                         adjacency, openings, stairs, polygon plots
          polygon.py     rasterization, erosion, largest inscribed rectangle
          console.py     UTF-8 console (Windows cp1252 kills report glyphs)
carve/    hub_carver.py  seed-guided band-and-column carving
          reserved.py    carve AROUND a fixed footprint (upper-floor stairs)
          settle.py      Squeeze & Settle optimizer (with frozen rooms)
          stairs.py      lattice-exact stair geometry + fitting
          connections.py openness gradient, entrance, reachability, windows
          standards.py   NBC minimums (stair minimums DERIVED, not typed)
engine/   orchestrator.py  propose→realize→settle→stairs→connect→review→rank
          contracts.py     typed tier interfaces + EngineConfig
          program.py       implicit rooms (a 2-floor request implies a stair)
          stairs.py        the stair-fitting tier
          site.py          irregular plot → buildable rectangle + open space
          multifloor.py    bottom-up floors, each conditioned on the one below
          vertical.py      VRT rules (two floors at a time)
          rules/           33 registered reviewer rules
          cpsat/           optional exact band assignment / re-dimensioning
critic/   gbt.py         gradient-boosted trees in NumPy (no ML dependency)
          features.py    rule breakdown + geometry → 47-feature vector
          perturb.py     controlled damage (the negatives)
          preferences.py append-only log of real user picks
tier2_placer/            the trained AR placer + merge_gate.py
render/   svg_render.py  wall-graph → SVG
          dxf_export.py  → AutoCAD R12 DXF, layered, real-world units
harness/  run_harness.py golden briefs → JSON baseline + HTML review grid
          ab_configs.py  A/B any EngineConfig field over the whole brief set
tests/                   unittest suite (267 tests)
```

## Run

```powershell
# from plangen_remastered/  (needs numpy + scipy; ortools optional)
python -m unittest discover -s tests            # 267 tests
python -m harness.run_harness                   # baseline + review.html
python -m harness.ab_configs --vary cpsat_mode=off,always
python -m critic.train                          # corpus → GBT → report
python -m tier2_placer.export --checkpoint ../sources/checkpoint.pt
python -m tier2_placer.merge_gate --weights tier2_placer/weights/placer_last.npz
```

## Status

Milestones M1–M8 of `docs/implementation_plan_v2.md` are built; §7.1 of that
document records which gates were met and which were not, with the numbers.
Two are open by measurement, not by omission:

- **CP-SAT (M3)** is off by default — it scored 70.80 against the greedy
  packer's 73.22 on the full brief set.
- **The trained placer (M5)** does not yet beat the prior proposer at 24
  epochs (68.57 vs 75.49); re-run `merge_gate` after 60–70 epochs.

The learned critic (M6), the vertical skeleton (M7) and irregular plots plus
DXF export (M8) all met their gates and are wired into the app through
`api/engine_bridge.py`.
