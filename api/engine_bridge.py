"""
engine_bridge.py — genuine layout generation via modules/step4_generate.

Adapts the pipeline's EnrichedPlan (step 3 output) into an EngineRequest for
the wall-graph partition engine, runs the real orchestrator (propose → carve
→ settle → connect → review → rank), and converts the winning candidate back
into the LayoutPlan schema + SVG the frontend consumes.

Multi-floor: the engine carves ONE floor at a time. This bridge runs it once
per floor (rooms are split across floors by the enricher), injects a staircase
on every floor of a multi-floor building, and returns all floors. Every floor
is a genuine carved plan — no placeholders.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

from models import EnrichedPlan, LayoutFloor, LayoutPlan, PlacedRoom
from modules.step4_generate.core import units
from modules.step4_generate.critic.critic import LearnedCritic
from modules.step4_generate.engine.contracts import (
    EngineConfig, EngineRequest, RoomSpec,
)
from modules.step4_generate.engine.multifloor import generate_building
from modules.step4_generate.engine.orchestrator import Orchestrator
from modules.step4_generate.render.dxf_export import export_dxf
from modules.step4_generate.render.svg_render import render_svg

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── room-type adaptation ──────────────────────────────────────────────
# enriched snake_case type → (engine rtype, engine zone)
_TYPE_MAP: Dict[str, Tuple[str, str]] = {
    "living_room":    ("living_room", "public"),
    "drawing_room":   ("drawing_room", "public"),
    "foyer":          ("foyer", "public"),
    "dining_room":    ("dining_room", "service"),
    "kitchen":        ("kitchen", "service"),
    "master_bedroom": ("master_bedroom", "private"),
    "bedroom":        ("bedroom", "private"),
    "bathroom":       ("bathroom", "private"),
    "toilet":         ("toilet", "private"),
    "study_room":     ("study_room", "private"),
    "pooja_room":     ("pooja_room", "public"),
    "store_room":     ("store", "service"),
    "storage":        ("store", "service"),
    "utility":        ("utility", "service"),
    "staircase":      ("staircase", "service"),
    "car_parking":    ("parking", "public"),
    "parking":        ("parking", "public"),
    "garage":         ("garage", "public"),
    "passage":        ("hallway", "private"),
    "hallway":        ("hallway", "private"),
    "corridor":       ("hallway", "private"),
}

# Rooms that cannot be carved as interior floor space — omitted honestly
# rather than mis-carved as a sealed indoor box. room_resolver.py's alias
# table (sources/enricher_rules.json) already routes most real-world
# phrasing to a known canonical type before this bridge ever sees it (e.g.
# "prayer hall" -> pooja_room, "garage" -> car_parking); these are the
# canonical types that genuinely have no honest interior-room realization.
_UNSUPPORTED = {
    "balcony", "verandah", "porch", "terrace", "open_terrace",
    "garden", "barsati", "swimming_pool", "pool", "lawn",
}

_DIRECTION_MAP = {
    "n": "N", "north": "N", "s": "S", "south": "S",
    "e": "E", "east": "E", "w": "W", "west": "W",
    "north_east": "N", "north_west": "N",
    "south_east": "S", "south_west": "S",
    "northeast": "N", "northwest": "N",
    "southeast": "S", "southwest": "S",
}

# staircase footprint (dog-leg flight + landing ≈ 8' x 12')
_STAIR_SQFT = 95.0

_FLOOR_LABELS = ["Ground Floor", "First Floor", "Second Floor", "Third Floor"]
_FLOOR_SLUGS = ["ground_floor", "first_floor", "second_floor", "third_floor"]


def _floor_label(i: int) -> str:
    return _FLOOR_LABELS[i] if i < len(_FLOOR_LABELS) else f"Floor {i}"


def _floor_slug(i: int) -> str:
    return _FLOOR_SLUGS[i] if i < len(_FLOOR_SLUGS) else f"floor_{i}"


def _entrance_side(direction: str) -> str:
    return _DIRECTION_MAP.get((direction or "").strip().lower(), "S")


def _seed_from(run_id: str) -> int:
    """Deterministic per-run seed → same run_id reproduces the same plan,
    while REGENERATE (new run_id) explores a different candidate set."""
    return int(hashlib.sha256(run_id.encode()).hexdigest()[:8], 16)


def _build_floor_request(enriched: EnrichedPlan, floor_idx: int, run_id: str,
                         plot_w: int, plot_h: int, multi: bool
                         ) -> Tuple[EngineRequest, List[str]]:
    """Build one floor's EngineRequest from the rooms assigned to that floor.
    Injects a staircase on every floor of a multi-floor building (the stair
    occupies real space on each level it serves)."""
    warnings: List[str] = []
    rooms_on_floor = enriched.get_rooms_on_floor(floor_idx)

    specs: List[RoomSpec] = []
    seen_names: Dict[str, int] = {}
    omitted: List[str] = []
    unrecognized: List[str] = []
    has_stair = False
    for room in rooms_on_floor:
        rtype = room.room_type.lower()
        if rtype in _UNSUPPORTED:
            omitted.append(room.display_name)
            continue
        if rtype == "staircase":
            has_stair = True
        # By this point room_resolver.py's alias table (sources/
        # enricher_rules.json room_name_aliases) has already resolved any
        # recognizable phrasing to a canonical type. A type still absent
        # from _TYPE_MAP is genuinely unrecognized — carve it generically
        # (never silently drop a room the user asked for) but say so, so
        # nothing is silently wrong.
        if rtype not in _TYPE_MAP:
            unrecognized.append(room.display_name)
        engine_type, zone = _TYPE_MAP.get(rtype, (rtype, "private"))
        name = room.display_name
        n = seen_names.get(name, 0)
        seen_names[name] = n + 1
        if n:
            name = f"{name} ({n + 1})"
        specs.append(RoomSpec(
            name=name, rtype=engine_type,
            target_sqft=float(room.target_area_sqft), zone=zone))

    # Every floor of a multi-floor home carries the staircase footprint.
    if multi and not has_stair:
        specs.append(RoomSpec(name="Staircase", rtype="staircase",
                              target_sqft=_STAIR_SQFT, zone="service"))

    if omitted:
        warnings.append(
            f"{_floor_label(floor_idx)}: not carved by the engine yet "
            f"(omitted): {', '.join(omitted)}.")

    if unrecognized:
        warnings.append(
            f"{_floor_label(floor_idx)}: no specific engine rule for "
            f"{', '.join(unrecognized)} — carved as a generic room (no "
            f"specialized ventilation, siting, or connectivity rules "
            f"applied).")

    if not specs:
        raise RuntimeError(f"{_floor_label(floor_idx)} has no placeable rooms")

    # Pre-scale over-tight programs so the request is always buildable.
    plot_sqft = plot_w * plot_h
    total = sum(s.target_sqft for s in specs)
    if total > plot_sqft * 0.90:
        scale = plot_sqft * 0.85 / total
        specs = [RoomSpec(s.name, s.rtype, round(s.target_sqft * scale, 1),
                          s.zone) for s in specs]
        warnings.append(
            f"{_floor_label(floor_idx)}: program ({total:.0f} sqft) exceeded "
            f"the footprint ({plot_sqft} sqft); targets scaled by {scale:.2f}.")

    request = EngineRequest(
        plot_w_ft=plot_w, plot_h_ft=plot_h,
        entrance_side=_entrance_side(enriched.entrance_direction),
        rooms=specs, k=6,
        # distinct-but-reproducible seed per floor
        seed=_seed_from(f"{run_id}#f{floor_idx}"),
        name=f"{run_id}_f{floor_idx}",
        floor_index=floor_idx, n_floors=max(1, enriched.total_floors),
    )
    return request, warnings


def _floor_from_plan(plan, floor_idx: int, plot_w: int, plot_h: int
                     ) -> LayoutFloor:
    """Carved GridPlan (integer 1.5-inch lattice) → one LayoutFloor."""
    cpf = float(units.CELLS_PER_FOOT)
    placed: List[PlacedRoom] = []
    for rid, room in sorted(plan.rooms.items()):
        x0, y0, x1, y1 = plan.face_bbox(rid)
        placed.append(PlacedRoom(
            room_id=room.name.lower().replace(" ", "_").replace(".", ""),
            room_type=room.rtype,
            display_name=room.name,
            floor=floor_idx,
            # SVG grid has row 0 at the top; LayoutPlan origin is SW.
            x_ft=round(x0 / cpf, 2),
            y_ft=round((plan.h - y1) / cpf, 2),
            width_ft=round((x1 - x0) / cpf, 2),
            length_ft=round((y1 - y0) / cpf, 2),
            area_sqft=round(plan.area_sqft(rid), 1),
        ))
    return LayoutFloor(
        floor_number=floor_idx, floor_label=_floor_label(floor_idx),
        net_width_ft=plot_w, net_length_ft=plot_h,
        rooms=placed,
        floor_area_placed_sqft=round(sum(r.area_sqft for r in placed), 1),
        floor_coverage_pct=100.0,   # partition covers the plot by construction
    )


def generate_layout(enriched: EnrichedPlan, run_id: str, run_dir: Path
                    ) -> Tuple[LayoutPlan, List[str], Dict[str, str]]:
    """Run the real engine once per floor. Returns (layout_plan,
    svg_filenames, tier_notes). Every floor is a genuine carved plan; raises
    RuntimeError with the engine's own reasons if a floor yields no valid
    plan — never returns a fake plan."""
    import time

    plot_w = max(12, min(200, int(round(enriched.net_buildable_width_ft))))
    plot_h = max(12, min(200, int(round(enriched.net_buildable_length_ft))))
    n_floors = max(1, enriched.total_floors)
    multi = n_floors > 1

    config = EngineConfig()
    # The trained critic reorders candidates the rules already accepted.
    # Absent weights simply mean "rank by the rules", which is the engine's
    # own default — never an error (critic/critic.py).
    orch = Orchestrator(config=config,
                        critic=LearnedCritic.load_if_available(config=config))
    floors: List[LayoutFloor] = []
    svg_names: List[str] = []
    warnings: List[str] = []
    fidelities: List[float] = []
    drifts: List[float] = []
    scores: List[float] = []
    kept_note: List[str] = []

    t0 = time.perf_counter()
    requests: List[EngineRequest] = []
    for i in range(n_floors):
        request, fwarn = _build_floor_request(
            enriched, i, run_id, plot_w, plot_h, multi)
        request.validate()
        requests.append(request)
        warnings.extend(fwarn)

    if multi:
        # Floors are planned BOTTOM-UP against each other: the staircase is
        # reserved over the flight below, wet rooms are pulled onto the
        # stacks, and a floor that breaks a vertical rule is rejected in
        # favour of the next candidate (engine/multifloor.py).
        building = generate_building(
            requests[0], [r.rooms for r in requests], config=config,
            orchestrator=orch)
        warnings.extend(building.warnings)
        for floor in building.floors:
            if not floor.ok:
                raise RuntimeError(
                    f"{_floor_label(floor.index)}: engine produced no valid "
                    f"plan. " + "; ".join(building.warnings))
            warnings.extend(floor.notes)
        chosen = [(f.index, f.candidate, requests[f.index].k)
                  for f in building.floors]
    else:
        result = orch.generate(requests[0])
        if not result.best:
            reasons = sorted({c.notes[-1] for c in result.discarded if c.notes})
            raise RuntimeError(
                f"{_floor_label(0)}: engine produced no valid plan. "
                + ("Reasons: " + "; ".join(reasons) if reasons else ""))
        warnings.extend(result.warnings)
        chosen = [(0, result.best, requests[0].k)]

    for i, best, k in chosen:
        floors.append(_floor_from_plan(best.plan, i, plot_w, plot_h))
        fidelities.append(best.fidelity or 0.0)
        drifts.append(float((best.verdict.breakdown or {}).get("area_drift", 0.0)))
        scores.append(best.verdict.soft_score)
        kept_note.append(f"{_floor_label(i)} ok/{k}")

        svg = render_svg(best.plan, title=f"{_floor_label(i)} — {plot_w}' x {plot_h}'")
        slug = _floor_slug(i) + ".svg"
        (Path(run_dir) / slug).write_text(svg, encoding="utf-8")
        svg_names.append(slug)

        # CAD deliverable alongside the picture (R12 DXF, no dependency)
        try:
            export_dxf(best.plan, str(Path(run_dir) / (_floor_slug(i) + ".dxf")))
        except Exception as exc:                 # never fail a run over CAD
            warnings.append(f"{_floor_label(i)}: DXF export skipped ({exc})")

    solve_ms = (time.perf_counter() - t0) * 1000.0

    def _avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    layout = LayoutPlan(
        run_id=run_id,
        plot_width_ft=enriched.plot_width_ft,
        plot_length_ft=enriched.plot_length_ft,
        net_buildable_width_ft=plot_w,
        net_buildable_length_ft=plot_h,
        setback_front_ft=enriched.setbacks.front or 0.0,
        setback_rear_ft=enriched.setbacks.rear or 0.0,
        setback_left_ft=enriched.setbacks.left or 0.0,
        setback_right_ft=enriched.setbacks.right or 0.0,
        entrance_direction=_entrance_side(enriched.entrance_direction),
        north_direction=enriched.north_direction,
        vastu_enabled=enriched.vastu_enabled,
        total_floors=len(floors),
        floors=floors,
        total_rooms_placed=sum(len(f.rooms) for f in floors),
        total_area_placed_sqft=round(
            sum(f.floor_area_placed_sqft for f in floors), 1),
        # Real engine metrics, averaged across floors (UI: FIDELITY / AREA MATCH).
        overall_adjacency_score=round(_avg(fidelities), 3),
        overall_zone_score=round(max(0.0, 1.0 - _avg(drifts)), 3),
        layout_quality_score=round(max(0.0, min(1.0, _avg(scores) / 100.0)), 3),
        solver_used="wall_graph_carver",
        solver_status="valid",
        solve_time_ms=round(solve_ms, 1),
        layout_warnings=warnings,
    )

    notes: Dict[str, str] = {
        "floors_generated": str(len(floors)),
        "kept_candidates": "; ".join(kept_note),
        "best_score": f"{_avg(scores):.1f} avg",
    }
    return layout, svg_names, notes
