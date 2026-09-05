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

import dataclasses
import hashlib
import json
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
from modules.step3_enrich.program_synth import capacity_for, plan_program
from modules.step4_generate.engine.vastu import report as vastu_report
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

# Geographic north -> the CARDINAL grid edge that faces it. The engine's
# compass frame is quarter-turn based, so a site whose north is NE has to be
# resolved to a cardinal; doing it here, once, with a warning, beats every
# rule guessing separately.
_NORTH_MAP = {
    "n": "N", "north": "N", "e": "E", "east": "E",
    "s": "S", "south": "S", "w": "W", "west": "W",
    "ne": "N", "north_east": "N", "nw": "N", "north_west": "N",
    "se": "S", "south_east": "S", "sw": "S", "south_west": "S",
}
_DIAGONAL_NORTH = {"ne", "nw", "se", "sw", "north_east", "north_west",
                   "south_east", "south_west"}

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

# Below this a leftover is a gap to absorb, not a court worth drawing.
_MIN_COURTYARD_SQFT = 80.0
# The engine refuses anything under 12'; keep a little headroom above that.
MIN_FOOTPRINT_FT = 14
# Above this it is not a courtyard, it is unbuilt land — and the honest
# answer is a smaller building, which `_footprint_for_program` provides.
_MAX_COURTYARD_SQFT = 420.0
# Share of the footprint the program should occupy. The remainder is walls,
# circulation slack and the court; packing tighter than this is what made
# tight plots score badly in the first place.
_TARGET_PROGRAM_FILL = 0.78

_FLOOR_LABELS = ["Ground Floor", "First Floor", "Second Floor", "Third Floor"]
_FLOOR_SLUGS = ["ground_floor", "first_floor", "second_floor", "third_floor"]


def _capacity_advice(enriched: EnrichedPlan, floor_idx: int,
                     plot_w: int, plot_h: int) -> str:
    """What the user can actually do about a floor that would not plan.

    "engine produced no valid plan" tells a builder nothing. This says how
    much area the program wants, how much the plot has, and what size of
    program that plot does carry — which is the difference between an error
    and advice."""
    rooms = [r for r in enriched.get_rooms_on_floor(floor_idx)
             if r.room_type.lower() not in _UNSUPPORTED]
    if not rooms:
        return ""
    net = float(plot_w * plot_h)
    want = sum(float(r.target_area_sqft) for r in rooms)
    beds = sum(1 for r in rooms if "bedroom" in r.room_type)
    tier, tier_beds = capacity_for(net)
    return (f" The program asks for {want:.0f} sqft of rooms on "
            f"{net:.0f} sqft of buildable area ({want / net:.0%}). "
            f"This plot comfortably carries a {tier}"
            + (f" — {tier_beds} bedroom{'s' if tier_beds != 1 else ''} rather "
               f"than {beds}." if beds > tier_beds else ".")
            + " Reduce the room count, or increase the plot or floor count.")


def _floor_label(i: int) -> str:
    return _FLOOR_LABELS[i] if i < len(_FLOOR_LABELS) else f"Floor {i}"


def _floor_slug(i: int) -> str:
    return _FLOOR_SLUGS[i] if i < len(_FLOOR_SLUGS) else f"floor_{i}"


def _entrance_side(direction: str) -> str:
    return _DIRECTION_MAP.get((direction or "").strip().lower(), "S")


def _north_side(direction: str, warnings: List[str]) -> str:
    """Which grid edge faces geographic north.

    This one value is what makes Vastu expressible at all: without it the
    engine cannot distinguish north-east from top-right, which is why a
    "Vastu compliant" plan used to be byte-identical to a non-Vastu one.
    """
    raw = (direction or "").strip().lower()
    if raw in _DIAGONAL_NORTH:
        warnings.append(
            f"north was given as {direction!r}; the mandala is aligned to a "
            f"cardinal axis, so it has been resolved to "
            f"{_NORTH_MAP[raw]}. Vastu sectors are accurate to within 45 "
            f"degrees for this site.")
    return _NORTH_MAP.get(raw, "N")


def _room_vastu(room) -> Dict[str, object]:
    """The Vastu step 3 already computed for a room, in engine terms.

    `EnrichedRoom.vastu` has carried this since the enricher was written; the
    bridge simply never read it. Prohibited directions are kept even when
    there is no preferred one — "not in the north-east" is a complete
    instruction on its own.
    """
    constraint = getattr(room, "vastu", None)
    if constraint is None:
        return {}
    preferred = [d for d in (constraint.preferred_directions or []) if d]
    prohibited = tuple(d for d in (constraint.prohibited_directions or [])
                       if d and d not in preferred[:1])
    if not preferred and not prohibited:
        return {}
    return {
        "vastu_dir": preferred[0] if preferred else None,
        "vastu_avoid": prohibited,
        "vastu_strength": ("hard" if constraint.constraint_type == "hard"
                           else "soft"),
    }


def _seed_from(run_id: str) -> int:
    """Deterministic per-run seed → same run_id reproduces the same plan,
    while REGENERATE (new run_id) explores a different candidate set."""
    return int(hashlib.sha256(run_id.encode()).hexdigest()[:8], 16)


def _footprint_for_program(enriched: EnrichedPlan, plot_w: int, plot_h: int,
                           multi: bool) -> Tuple[int, int, List[str]]:
    """Size the BUILDING to the program, instead of the program to the plot.

    The old behaviour handed the engine the whole net-buildable rectangle and
    let the settler spread the program across it, so a six-room brief on a
    60x70 plot produced a 266 sqft bathroom. Capping the rooms fixed the
    bathroom and produced a 2,893 sqft "courtyard" instead, which is just the
    same mistake wearing a hat.

    A 2BHK on a large plot is a 2BHK house standing in a garden. So the
    footprint is derived from what the program actually needs, the building
    keeps the plot's proportions, and the land it does not cover is reported
    as open ground rather than absorbed.

    Sized from the LARGEST floor and applied to all of them: the vertical
    rules require every floor to share one lattice (VRT-002), so the
    footprint cannot vary per storey.
    """
    notes: List[str] = []
    n_floors = max(1, enriched.total_floors)
    needed = 0.0
    for i in range(n_floors):
        rooms = [r for r in enriched.get_rooms_on_floor(i)
                 if r.room_type.lower() not in _UNSUPPORTED]
        floor_sqft = sum(float(r.max_area_sqft or r.target_area_sqft)
                         for r in rooms)
        if multi and not any(r.room_type.lower() == "staircase"
                             for r in rooms):
            floor_sqft += _STAIR_SQFT
        needed = max(needed, floor_sqft)
    if needed <= 0:
        return plot_w, plot_h, notes

    net_sqft = float(plot_w * plot_h)
    want_sqft = needed / _TARGET_PROGRAM_FILL
    if want_sqft >= net_sqft * 0.97:
        return plot_w, plot_h, notes          # the plot is already the size

    scale = (want_sqft / net_sqft) ** 0.5
    new_w = max(MIN_FOOTPRINT_FT, int(round(plot_w * scale)))
    new_h = max(MIN_FOOTPRINT_FT, int(round(plot_h * scale)))
    if new_w >= plot_w and new_h >= plot_h:
        return plot_w, plot_h, notes

    open_ground = net_sqft - new_w * new_h
    notes.append(
        f"The program needs about {needed:.0f} sqft of rooms, so the house "
        f"is planned at {new_w}' x {new_h}' ({new_w * new_h} sqft) rather "
        f"than spread across the full {plot_w}' x {plot_h}' buildable area. "
        f"That leaves {open_ground:.0f} sqft of open ground.")
    return new_w, new_h, notes


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
            target_sqft=float(room.target_area_sqft), zone=zone,
            min_sqft=float(room.min_area_sqft or 0) or None,
            max_sqft=float(room.max_area_sqft or 0) or None,
            **_room_vastu(room)))

    # Every floor of a multi-floor home carries the staircase footprint.
    if multi and not has_stair:
        specs.append(RoomSpec(name="Staircase", rtype="staircase",
                              target_sqft=_STAIR_SQFT, zone="service",
                              vastu_dir="SW" if enriched.vastu_enabled else None,
                              vastu_avoid=(("NE", "N")
                                           if enriched.vastu_enabled else ())))

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

    # ── a modest courtyard absorbs what is left after the footprint fit ──
    # `_footprint_for_program` has already shrunk the building to the
    # program, so anything left here is a small remainder, not the several
    # thousand square feet a large plot used to hand over. Bounded, because
    # a 2,893 sqft "courtyard" is not a courtyard — it is an unbuilt plot,
    # and saying so is the footprint fit's job, not this one's.
    room_capacity = sum(s.max_sqft or s.target_sqft for s in specs)
    buildable_sqft = plot_w * plot_h * 0.88          # net of wall thickness
    surplus = buildable_sqft - room_capacity
    if surplus >= _MIN_COURTYARD_SQFT:
        court = min(surplus, _MAX_COURTYARD_SQFT)
        specs.append(RoomSpec(
            name="Courtyard", rtype="ots",
            target_sqft=round(court, 1), zone="public",
            min_sqft=_MIN_COURTYARD_SQFT, max_sqft=_MAX_COURTYARD_SQFT,
            vastu_dir="center" if enriched.vastu_enabled else None))
        warnings.append(
            f"{_floor_label(floor_idx)}: {court:.0f} sqft is planned as an "
            f"open courtyard rather than added to the rooms"
            + (" (at the Brahmasthan, which Vastu asks to keep open)"
               if enriched.vastu_enabled else "") + ".")

    # Pre-scale over-tight programs so the request is always buildable.
    plot_sqft = plot_w * plot_h
    total = sum(s.target_sqft for s in specs)
    if total > plot_sqft * 0.90:
        scale = plot_sqft * 0.85 / total
        specs = [dataclasses.replace(
            s, target_sqft=round(s.target_sqft * scale, 1)) for s in specs]
        warnings.append(
            f"{_floor_label(floor_idx)}: program ({total:.0f} sqft) exceeded "
            f"the footprint ({plot_sqft} sqft); targets scaled by {scale:.2f}.")

    request = EngineRequest(
        plot_w_ft=plot_w, plot_h_ft=plot_h,
        entrance_side=_entrance_side(enriched.entrance_direction),
        north_side=_north_side(enriched.north_direction, warnings),
        vastu=bool(enriched.vastu_enabled),
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
    # Real room fill, not the partition's coverage. The partition covers the
    # plot by construction, so reporting 100% here said nothing and hid the
    # number a user actually wants: how much of the footprint is ROOM rather
    # than wall. Measured range is 82-92%, the remainder being the exterior
    # ring and the internal partitions.
    gross_sqft = units.area_sqft(plan.w * plan.h)
    room_sqft = sum(r.area_sqft for r in placed)
    return LayoutFloor(
        floor_number=floor_idx, floor_label=_floor_label(floor_idx),
        net_width_ft=plot_w, net_length_ft=plot_h,
        rooms=placed,
        floor_area_placed_sqft=round(room_sqft, 1),
        floor_coverage_pct=round(100.0 * room_sqft / gross_sqft, 1)
        if gross_sqft else 0.0,
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

    # size the building to the program before anything else sees the plot
    plot_w, plot_h, footprint_notes = _footprint_for_program(
        enriched, plot_w, plot_h, multi)

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

    warnings.extend(footprint_notes)

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
                    f"plan." + _capacity_advice(enriched, floor.index,
                                                plot_w, plot_h)
                    + " " + "; ".join(building.warnings))
            warnings.extend(floor.notes)
        chosen = [(f.index, f.candidate, requests[f.index].k)
                  for f in building.floors]
    else:
        result = orch.generate(requests[0])
        if not result.best:
            reasons = sorted({c.notes[-1] for c in result.discarded if c.notes})
            raise RuntimeError(
                f"{_floor_label(0)}: engine produced no valid plan."
                + _capacity_advice(enriched, 0, plot_w, plot_h)
                + (" Engine reasons: " + "; ".join(reasons[:2])
                   if reasons else ""))
        warnings.extend(result.warnings)
        chosen = [(0, result.best, requests[0].k)]

    vastu_floors: List[Dict] = []
    for i, best, k in chosen:
        floors.append(_floor_from_plan(best.plan, i, plot_w, plot_h))
        # The Vastu scorecard. `vastu_enabled: true` on its own told a user
        # nothing — and told it for months while the engine ignored Vastu
        # entirely. This is the per-room account behind that flag.
        report = vastu_report.build(best.plan, best.request or requests[i],
                                    best.room_ids, best.verdict)
        if report.active:
            vastu_floors.append({"floor": i,
                                 "floor_label": _floor_label(i),
                                 **report.to_dict()})
            warnings.extend(
                f"{_floor_label(i)}: {n}" for n in report.notes)
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
    if vastu_floors:
        grades = "; ".join(f"{v['floor_label']} {v['grade']} "
                           f"({v['score']:.0%})" for v in vastu_floors)
        notes["vastu"] = grades
        notes["vastu_report"] = json.dumps(vastu_floors)
    return layout, svg_names, notes
