"""
connections.py — automatic connection typing (the openness gradient)
plus fenestration (windows) and the whole-house connectivity guarantee.

Real plans breathe through a gradient: social rooms merge into each other
through doorless wide openings (the partial walls of the reference plan);
private and service rooms close behind door leaves; some pairs stay sealed.
This module applies that gradient as RULES over the final adjacency graph —
no wish lists — then guarantees the house works:

  1. main entrance door through the exterior wall (entrance room priority)
  2. rule-typed openings on qualifying shared walls
  3. reachability repair: any room not reachable from the entrance through
     openings gets a door to its best reachable neighbor (social side
     preferred — privacy), or the candidate dies honestly
  4. windows on exterior walls: habitable rooms sized toward NBC
     ventilation ratios, wet rooms get small ventilators

Kitchen NEVER receives a door — wide openings only (user quality bar).
The public functions are reused by the engine's repair module.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from modules.step4_generate.core import units
from modules.step4_generate.core.grid_plan import (
    MIN_STUB, CarveError, GridPlan, SharedWall,
)
from modules.step4_generate.engine.contracts import EngineConfig, EngineRequest

SOCIAL = {"living_room", "drawing_room", "dining_room", "hallway",
          "foyer", "passage"}
KITCHEN = {"kitchen"}
PRIVATE = {"bedroom", "master_bedroom", "study", "office"}
WET = {"bathroom", "toilet"}
SERVICE = {"store", "storage", "utility", "laundry"}
PARKING = {"parking", "garage"}
STAIR = {"staircase"}
HABITABLE = SOCIAL | KITCHEN | PRIVATE

# Preferred widths tried in order (cells): wide 5' → 4' → 3'; doors 30".
WIDE_WIDTHS = (40, 32, 24)
DOOR_WIDTHS = (20,)
MAIN_DOOR_WIDTH = 28          # 3'6"
FALLBACK_DOOR_WIDTHS = (24, 20)   # entrance repair may try 3' then 30"

# Vehicle gate for parking/garage (doorless — a carport opening, not a
# leafed door): 10' → 9' → 8' → 7', the smallest still fitting one car.
VEHICLE_GATE_WIDTHS = (80, 72, 64, 56)   # 10' / 9' / 8' / 7'

# Parking/garage no longer competes for the PEDESTRIAN main door — real
# homes enter through a foyer/living/drawing/dining room, not the garage.
# Parking gets its own vehicle gate instead (place_vehicle_gate below).
_ENTRANCE_PRIORITY = ("foyer", "hallway", "drawing_room", "living_room",
                      "dining_room")

_SIDES = ("N", "E", "S", "W")


# ═════════════════════════ RULES ════════════════════════════════════════════

def _cross(rt_a: str, rt_b: str, group_x: set, group_y: set) -> bool:
    """True iff one room type is in group_x and the OTHER is in group_y."""
    return (rt_a in group_x and rt_b in group_y) or \
           (rt_b in group_x and rt_a in group_y)


def _rule(rt_a: str, rt_b: str) -> Optional[Tuple[str, str]]:
    """(kind, swing) for a room-type pair, or None to stay sealed.
    swing "a"/"b" = leaf opens into that side (the more private side).

    The openness gradient: social↔social and kitchen↔social merge through
    doorless wide openings; private/wet/service close behind doors; pairs
    like bedroom↔bedroom or kitchen↔bedroom stay sealed walls."""
    if _cross(rt_a, rt_b, KITCHEN, SOCIAL):
        return ("wide", "b")
    if rt_a in SOCIAL and rt_b in SOCIAL:
        return ("wide", "b")
    if _cross(rt_a, rt_b, STAIR, SOCIAL | PARKING):
        return ("wide", "b")
    if _cross(rt_a, rt_b, KITCHEN, SERVICE):
        return ("wide", "b")          # utility/store off the kitchen: open
    if _cross(rt_a, rt_b, PRIVATE, SOCIAL):
        return ("door", "a" if rt_a in PRIVATE else "b")
    if _cross(rt_a, rt_b, WET, PRIVATE | SOCIAL):
        return ("door", "a" if rt_a in WET else "b")
    if _cross(rt_a, rt_b, SERVICE, SOCIAL):
        return ("door", "a" if rt_a in SERVICE else "b")
    if _cross(rt_a, rt_b, PARKING, SOCIAL):
        return ("door", "a" if rt_a in SOCIAL else "b")
    return None


def _try_opening(plan: GridPlan, a: int, b: int, kind: str,
                 swing: str) -> bool:
    widths = WIDE_WIDTHS if kind == "wide" else DOOR_WIDTHS
    for width in widths:
        try:
            plan.add_opening(a, b, kind, width=width, swing=swing)
            return True
        except CarveError:
            continue
    if kind == "wide":
        # a social pair whose wall is too short for even a 3' wide opening
        # still must not be sealed — degrade to the widest door-size gap,
        # but NEVER for the kitchen (kitchen is doorless by rule; a short
        # kitchen wall stays sealed and reachability finds another path).
        rtypes = {plan.rooms[a].rtype, plan.rooms[b].rtype}
        if not (rtypes & KITCHEN):
            for width in DOOR_WIDTHS:
                try:
                    plan.add_opening(a, b, "wide", width=width, swing=swing)
                    return True
                except CarveError:
                    continue
    return False


# ═════════════════════════ ENTRANCE ═════════════════════════════════════════

def place_entrance(plan: GridPlan, request: EngineRequest,
                   room_ids: Dict[str, int],
                   widths: Tuple[int, ...] = (MAIN_DOOR_WIDTH,)) -> int:
    """Cut the main door on the entrance side; returns the entrance room id.
    Raises CarveError if no room offers a long-enough exterior wall."""
    side = request.entrance_side
    candidates: List[Tuple[int, int]] = []          # (priority, room_id)
    for spec in request.rooms:
        rid = room_ids[spec.name]
        runs = plan.exterior_runs(rid, side)
        if any(hi - lo >= min(widths) + 2 * MIN_STUB for lo, hi in runs):
            try:
                prio = _ENTRANCE_PRIORITY.index(spec.rtype)
            except ValueError:
                prio = len(_ENTRANCE_PRIORITY)
            candidates.append((prio, rid))
    if not candidates:
        raise CarveError(
            f"no room offers an exterior wall on entrance side {side!r} "
            f"for the main door")
    entry = min(candidates)[1]
    for width in widths:
        try:
            plan.add_exterior_opening(entry, side, width=width)
            return entry
        except CarveError:
            continue
    raise CarveError("entrance room found but no width fits")


def _arrival_room(plan: GridPlan, room_ids: Dict[str, int]) -> Optional[int]:
    """Where you set foot on an upper floor: the staircase."""
    return next((rid for rid in room_ids.values()
                 if plan.rooms[rid].rtype in STAIR), None)


# ═════════════════════════ VEHICLE GATE ══════════════════════════════════════

def _gate_side_order(entrance_side: str) -> List[str]:
    """Sides to try for a parking/garage vehicle gate, entrance side first
    (the modeled road side), then the two corner-adjacent sides, then the
    far side last. A plot's actual driveway need not fall on whichever
    side the carver happened to seed parking toward — refusing to look
    elsewhere would reject an otherwise-good layout over an arbitrary side
    choice and starve the orchestrator's retry loop for no real reason."""
    i = _SIDES.index(entrance_side)
    return [_SIDES[i], _SIDES[i - 1], _SIDES[(i + 1) % 4], _SIDES[i - 2]]


def place_vehicle_gate(plan: GridPlan, request: EngineRequest,
                       room_ids: Dict[str, int]) -> List[str]:
    """Cut a doorless vehicle-width gate onto the exterior for every
    parking/garage room, independent of the pedestrian main entrance.
    Tries every side (entrance side first) and every width in
    VEHICLE_GATE_WIDTHS before giving up on a room. A parking room that
    finds no gate on any side gets none — STR-003 (engine/rules/structure)
    disqualifies the candidate and the orchestrator's retry/top-up loop
    tries a different seed (or, in relaxed mode, the room is dropped;
    see engine/repair.py)."""
    notes: List[str] = []
    sides = _gate_side_order(request.entrance_side)
    for spec in request.rooms:
        if spec.rtype not in PARKING:
            continue
        rid = room_ids.get(spec.name)
        if rid is None:
            continue
        for side in sides:
            placed = False
            for width in VEHICLE_GATE_WIDTHS:
                try:
                    plan.add_exterior_opening(rid, side, kind="wide",
                                              width=width)
                    ft = width // units.CELLS_PER_FOOT
                    notes.append(f"vehicle gate: {spec.name} "
                                f"({side} side, {ft}')")
                    placed = True
                    break
                except CarveError:
                    continue
            if placed:
                break
    return notes


# ═════════════════════════ REACHABILITY ═════════════════════════════════════

def reachable_rooms(plan: GridPlan, entry: int) -> Set[int]:
    seen = {entry}
    frontier = [entry]
    while frontier:
        cur = frontier.pop()
        for op in plan.openings:
            if op.is_exterior:
                continue
            other = (op.room_b if op.room_a == cur
                     else op.room_a if op.room_b == cur else None)
            if other is not None and other not in seen:
                seen.add(other)
                frontier.append(other)
    return seen


def ensure_reachable(plan: GridPlan, entry: int,
                     room_ids: Dict[str, int]) -> List[str]:
    """Add privacy-preferring doors until every room is reachable from the
    entrance. Raises CarveError if connectivity cannot be established."""
    notes: List[str] = []
    all_rooms = set(room_ids.values())
    for _ in range(len(all_rooms)):
        got = reachable_rooms(plan, entry)
        missing = all_rooms - got
        if not missing:
            return notes

        def _repair_rank(sw: SharedWall) -> tuple:
            outer = sw.room_a if sw.room_a in got else sw.room_b
            social = plan.rooms[outer].rtype in (SOCIAL | PARKING | STAIR)
            return (0 if social else 1, -sw.length)

        progressed = False
        for sw in sorted(plan.shared_walls(), key=_repair_rank):
            a_in, b_in = sw.room_a in got, sw.room_b in got
            if a_in == b_in:
                continue
            inner = sw.room_a if not a_in else sw.room_b
            if inner not in missing:
                continue
            swing = "a" if sw.room_a == inner else "b"
            pair_types = {plan.rooms[sw.room_a].rtype,
                          plan.rooms[sw.room_b].rtype}
            kind = "wide" if pair_types & KITCHEN else "door"
            if _try_opening(plan, sw.room_a, sw.room_b, kind, swing):
                notes.append(
                    f"connectivity door: {plan.rooms[sw.room_a].name} <-> "
                    f"{plan.rooms[sw.room_b].name}")
                progressed = True
                break
        if not progressed:
            names = ", ".join(plan.rooms[r].name for r in missing)
            raise CarveError(f"unreachable rooms, no repair possible: {names}")
    return notes


# ═════════════════════════ WINDOWS ══════════════════════════════════════════

def place_windows(plan: GridPlan, request: EngineRequest,
                  room_ids: Dict[str, int],
                  config: EngineConfig) -> List[str]:
    """Windows on exterior walls: habitable rooms sized toward the NBC
    ventilation ratio (window area ≥ ~1/8 floor area at 4' sill-to-lintel),
    wet rooms get small ventilators. Rooms with no exterior wall get a
    window onto an adjacent OTS shaft when one exists; otherwise the
    validator's daylight rule penalizes the plan."""
    notes: List[str] = []
    placed = 0
    ots_walls = [sw for sw in plan.shared_walls()
                 if plan.rooms[sw.room_a].rtype == "ots"
                 or plan.rooms[sw.room_b].rtype == "ots"]

    for spec in request.rooms:
        rid = room_ids[spec.name]
        if spec.rtype in HABITABLE:
            area = plan.area_sqft(rid)
            want_ft = max(config.window_habitable_min_ft,
                          min(config.window_habitable_max_ft,
                              round(area / 32)))   # area/(8 ratio × 4' tall)
            width = units.cells(want_ft)
        elif spec.rtype in WET:
            width = units.cells(config.window_wet_ft)
        else:
            continue

        candidates: List[Tuple[int, str]] = []      # (run_len, side)
        for side in _SIDES:
            for lo, hi in plan.exterior_runs(rid, side):
                candidates.append((hi - lo, side))
        # longest runs first; a clash with the main door falls through to
        # the next exterior wall instead of costing the room its window
        done = False
        for run_len, side in sorted(candidates, reverse=True):
            w = min(width, run_len - 2 * MIN_STUB)
            if w < units.cells(2):
                continue
            try:
                plan.add_exterior_opening(rid, side, kind="window", width=w)
                placed += 1
                done = True
                break
            except CarveError:
                continue
        if done:
            continue

        # no exterior wall — window onto an adjacent OTS shaft
        my_ots = [sw for sw in ots_walls if rid in (sw.room_a, sw.room_b)]
        for sw in sorted(my_ots, key=lambda s: -s.length):
            w = min(width, units.cells(3), sw.length - 2 * MIN_STUB)
            if w < units.cells(2):
                continue
            try:
                plan.add_opening(sw.room_a, sw.room_b, "window", width=w)
                placed += 1
                break
            except CarveError:
                continue

    notes.append(f"windows placed: {placed}")
    return notes


# ═════════════════════════ COMPOSITION ══════════════════════════════════════

def connect(plan: GridPlan, request: EngineRequest,
            room_ids: Dict[str, int],
            config: Optional[EngineConfig] = None) -> List[str]:
    """Apply the full gradient: entrance → vehicle gate → rule openings →
    reachability → windows. Returns notes; raises CarveError if the house
    cannot be made reachable (the orchestrator discards such candidates)."""
    config = config or EngineConfig()
    notes: List[str] = []

    if request.floor_index > 0:
        # An upper floor has no street door — you arrive on the staircase.
        # Cutting an exterior door up here would put a doorway into open
        # air, and every entrance-relative rule would then measure from it.
        entry = _arrival_room(plan, room_ids)
        if entry is None:
            raise CarveError(
                f"floor {request.floor_index} has no staircase to arrive on")
        notes.append(f"arrival: {plan.rooms[entry].name} "
                     f"(from the flight below)")
    else:
        entry = place_entrance(plan, request, room_ids)
        notes.append(f"main door: {plan.rooms[entry].name} "
                     f"({request.entrance_side} side)")
    notes += place_vehicle_gate(plan, request, room_ids)

    walls = plan.shared_walls()
    best_run: Dict[Tuple[int, int], SharedWall] = {}
    for sw in walls:
        key = tuple(sorted((sw.room_a, sw.room_b)))
        if key not in best_run or sw.length > best_run[key].length:
            best_run[key] = sw

    for (a, b), sw in sorted(best_run.items()):
        decision = _rule(plan.rooms[a].rtype, plan.rooms[b].rtype)
        if decision is None:
            continue
        kind, swing = decision
        if not _try_opening(plan, a, b, kind, swing):
            notes.append(f"sealed (wall too short): "
                         f"{plan.rooms[a].name} | {plan.rooms[b].name}")

    notes += ensure_reachable(plan, entry, room_ids)
    notes += place_windows(plan, request, room_ids, config)

    plan.verify()
    return notes
