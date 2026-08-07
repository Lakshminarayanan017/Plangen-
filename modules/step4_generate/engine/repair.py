"""
repair.py — targeted repairs for hard-rule violations (engine doc §4).

Each repair maps a violation pattern to a concrete plan mutation. Repairs
run inside the orchestrator's bounded repair loop (config.max_repair_rounds)
and must be conservative: fix the named violation without breaking the
partition. Anything unrepairable stays a discard — never paper over.

Coverage (violation id → strategy):
  OPN-001  kitchen has a door      → convert to wide opening / remove
  HYG-001  wet room opens→kitchen  → seal the opening
  CIR-001  no main entrance        → retry entrance with narrower widths
  CIR-002  unreachable rooms       → re-run reachability repair
  STR-001/STR-002                  → unrepairable by design (carver bugs
                                     must fail loudly, not be patched)
"""

from __future__ import annotations

from typing import Dict, List

from modules.step4_generate.carve.connections import (
    FALLBACK_DOOR_WIDTHS, _try_opening, ensure_reachable, place_entrance,
)
from modules.step4_generate.core.grid_plan import CarveError, GridPlan
from modules.step4_generate.engine.contracts import EngineRequest


def attempt(plan: GridPlan, request: EngineRequest,
            room_ids: Dict[str, int], hard: List[str]) -> List[str]:
    """Try to fix the given hard violations in place.
    Returns notes for every repair performed (empty = nothing repairable).
    Each repair strategy runs at most once per attempt — repairs recompute
    the live violations themselves rather than trusting stale messages."""
    notes: List[str] = []
    ran = set()
    for violation in hard:
        key = violation.split(" ", 1)[0]
        if key in ran:
            continue
        ran.add(key)
        if violation.startswith("OPN-001"):
            notes += _fix_kitchen_door(plan)
        elif violation.startswith("HYG-001"):
            notes += _remove_wet_kitchen_openings(plan)
        elif violation.startswith("CIR-001"):
            notes += _retry_entrance(plan, request, room_ids)
        elif violation.startswith("CIR-002"):
            notes += _reconnect(plan, room_ids)
        elif violation.startswith("FSP-001"):
            notes += _connect_freespace(plan, request, room_ids)
        elif violation.startswith("DOR-001"):
            notes += _separate_swings(plan, request, room_ids)
    return notes


def _fix_kitchen_door(plan: GridPlan) -> List[str]:
    """Replace any door touching a kitchen with a wide opening on the same
    wall (or drop it if the kitchen has another wide connection)."""
    notes = []
    for op in list(plan.openings):
        if op.is_exterior or op.kind != "door":
            continue
        types = {plan.rooms[op.room_a].rtype, plan.rooms[op.room_b].rtype}
        if "kitchen" not in types:
            continue
        plan.openings.remove(op)
        if _try_opening(plan, op.room_a, op.room_b, "wide", op.swing):
            notes.append(
                f"repair OPN-001: door -> wide "
                f"({plan.rooms[op.room_a].name} <-> "
                f"{plan.rooms[op.room_b].name})")
        else:
            notes.append(
                f"repair OPN-001: removed kitchen door "
                f"({plan.rooms[op.room_a].name} <-> "
                f"{plan.rooms[op.room_b].name})")
    return notes


def _remove_wet_kitchen_openings(plan: GridPlan) -> List[str]:
    notes = []
    wet = {"bathroom", "toilet"}
    for op in list(plan.openings):
        if op.is_exterior:
            continue
        types = {plan.rooms[op.room_a].rtype, plan.rooms[op.room_b].rtype}
        if types & wet and "kitchen" in types:
            plan.openings.remove(op)
            notes.append(
                f"repair HYG-001: sealed "
                f"{plan.rooms[op.room_a].name} <-> "
                f"{plan.rooms[op.room_b].name}")
    return notes


def _retry_entrance(plan: GridPlan, request: EngineRequest,
                    room_ids: Dict[str, int]) -> List[str]:
    """Main door failed at full width — retry the priority ladder with
    progressively narrower (still legal) main-door widths."""
    try:
        entry = place_entrance(plan, request, room_ids,
                               widths=FALLBACK_DOOR_WIDTHS)
        return [f"repair CIR-001: main door placed at reduced width "
                f"({plan.rooms[entry].name})"]
    except CarveError:
        return []


def _reconnect(plan: GridPlan, room_ids: Dict[str, int]) -> List[str]:
    entries = [op for op in plan.openings if op.is_exterior
               and op.kind == "door"]
    if not entries:
        return []
    try:
        fixed = ensure_reachable(plan, entries[0].room_a, room_ids)
        return [f"repair CIR-002: {n}" for n in fixed]
    except CarveError:
        return []


def _review_ctx(plan: GridPlan, request: EngineRequest,
                room_ids: Dict[str, int]):
    from modules.step4_generate.engine.contracts import EngineConfig
    from modules.step4_generate.engine.rules import ReviewContext
    return ReviewContext(plan=plan, request=request, room_ids=room_ids,
                         config=EngineConfig())


def _connect_freespace(plan: GridPlan, request: EngineRequest,
                       room_ids: Dict[str, int]) -> List[str]:
    """FSP-001: give every disconnected room a direct opening onto the
    free-space component (recomputed live, not parsed from messages)."""
    from modules.step4_generate.engine.rules.base import PRIVATE, SOCIAL, WET
    notes: List[str] = []
    ctx = _review_ctx(plan, request, room_ids)
    free = ctx.free_component()
    if not free:
        return notes
    graph = ctx.opening_graph()

    for spec in request.rooms:
        rid = room_ids.get(spec.name)
        if rid is None or rid in free or spec.rtype in SOCIAL:
            continue
        neighbors = {n for n, _ in graph.get(rid, ())}
        if neighbors & free:
            continue
        if spec.rtype in WET and any(ctx.rtype(n) in PRIVATE
                                     for n in neighbors):
            continue
        walls = [sw for sw in plan.shared_walls()
                 if rid in (sw.room_a, sw.room_b)
                 and (sw.room_a in free or sw.room_b in free)]
        for sw in sorted(walls, key=lambda s: -s.length):
            swing = "a" if sw.room_a == rid else "b"
            kind = "wide" if spec.rtype == "kitchen" else "door"
            if _try_opening(plan, sw.room_a, sw.room_b, kind, swing):
                notes.append(f"repair FSP-001: opened {spec.name} onto "
                             f"the free space")
                break
    return notes


def _separate_swings(plan: GridPlan, request: EngineRequest,
                     room_ids: Dict[str, int]) -> List[str]:
    """DOR-001: recompute live swing collisions; shift the second door of
    each colliding pair toward the far end of its host wall run."""
    notes: List[str] = []
    ctx = _review_ctx(plan, request, room_ids)

    def collisions():
        doors = [op for op in plan.openings
                 if op.kind == "door" and not op.is_exterior]
        rects = [(ctx.door_swing_rect(op), op) for op in doors]
        out = []
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                (ax0, ay0, ax1, ay1), a = rects[i]
                (bx0, by0, bx1, by1), b = rects[j]
                if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                    out.append((a, b))
        return out

    for _ in range(4):                       # bounded shifting rounds
        pairs = collisions()
        if not pairs:
            break
        moved = False
        for a, b in pairs:
            host = [sw for sw in plan.shared_walls()
                    if sw.axis == b.axis
                    and sw.wall_lo == b.wall_lo and sw.wall_hi == b.wall_hi
                    and {sw.room_a, sw.room_b} == {b.room_a, b.room_b}]
            if not host:
                continue
            run = max(host, key=lambda s: s.length)
            plan.openings.remove(b)
            # try the run end farther from door a's span first
            far_first = run.along_hi - b.width if a.along_lo <= run.along_lo \
                else run.along_lo
            for at in (far_first, run.along_lo, run.along_hi - b.width):
                try:
                    plan.add_opening(b.room_a, b.room_b, "door",
                                     width=b.width, swing=b.swing, at=at)
                    notes.append(
                        f"repair DOR-001: shifted door "
                        f"{plan.rooms[b.room_a].name} <-> "
                        f"{plan.rooms[b.room_b].name}")
                    moved = True
                    break
                except CarveError:
                    continue
            else:
                plan.openings.append(b)      # restore; unrepairable pair
            # ctx caches are stale after a mutation — rebuild
            ctx = _review_ctx(plan, request, room_ids)
            break
        if not moved:
            break
    return notes
