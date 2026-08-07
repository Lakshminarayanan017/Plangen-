"""
reserved.py — carve a floor AROUND a footprint that is not negotiable.

The upper floor of a house is not a fresh plot. The staircase from below
arrives in specific cells, and a stair that lands two feet to the left
upstairs does not connect to anything. Seeding the proposer at the right
spot is a hint, and hints lose: the carver decides band boundaries from
areas, so the realized face falls wherever the packing put it (measured:
0-1% overlap with the floor below).

So the footprint is RESERVED before any room is placed. The plot is cut with
a guillotine sequence that isolates exactly that rectangle, and the rest of
the program is carved into the rectangular regions left over, each with the
ordinary band-and-column packer. The reservation is exact by construction:
the reserved face's bbox equals the requested rectangle, and a test asserts
it rather than measuring an overlap and hoping.

Up to four regions surround an interior rectangle:

        +---------------------------+
        |          front            |     (entrance side)
        +--------+--------+---------+
        |  left  | RESVD  |  right  |
        +--------+--------+---------+
        |          rear             |
        +---------------------------+

Regions that would be degenerate (the rectangle touching an edge, or a strip
too thin to hold anything) are simply absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from modules.step4_generate.carve.hub_carver import (
    WALL_ALLOWANCE, _RoomPlan, carve_region, room_plans, scaled_budgets,
)
from modules.step4_generate.core import units
from modules.step4_generate.core.grid_plan import INT_WALL_CELLS, CarveError, GridPlan
from modules.step4_generate.engine.contracts import (
    EngineConfig, EngineRequest, LayoutProposal, RoomSpec,
)

# A residual strip narrower than this cannot hold a room and is merged away
# by never being created (the reservation grows to the plot edge instead).
MIN_STRIP_CELLS = units.cells(4)

Rect = Tuple[int, int, int, int]


@dataclass(frozen=True)
class Region:
    """One rectangular leftover, with the face id it became."""
    name: str                  # front | rear | left | right
    face: int
    rect: Rect

    @property
    def area_cells(self) -> int:
        x0, y0, x1, y1 = self.rect
        return (x1 - x0) * (y1 - y0)


def snap_reservation(plan: GridPlan, rect: Rect) -> Rect:
    """Push a requested rectangle out to the plot edges when the strip it
    would leave behind is too thin to be a room.

    Without this, reserving a footprint 2'0" from the east wall creates a
    2'0" strip that no program can occupy and the carve fails — losing the
    whole floor over a detail the reservation does not care about."""
    ext = plan.ext_wall
    x0, y0, x1, y1 = rect
    lo_x, hi_x = ext, plan.w - ext
    lo_y, hi_y = ext, plan.h - ext

    if x0 - lo_x < MIN_STRIP_CELLS + INT_WALL_CELLS:
        x0 = lo_x
    if hi_x - x1 < MIN_STRIP_CELLS + INT_WALL_CELLS:
        x1 = hi_x
    if y0 - lo_y < MIN_STRIP_CELLS + INT_WALL_CELLS:
        y0 = lo_y
    if hi_y - y1 < MIN_STRIP_CELLS + INT_WALL_CELLS:
        y1 = hi_y
    return (x0, y0, x1, y1)


def isolate(plan: GridPlan, rect: Rect, side: str
            ) -> Tuple[int, List[Region]]:
    """Cut the plot so `rect` is exactly one face. Returns (face, regions).

    Cuts run depth-first (front slab, rear slab) then across, so the
    reserved face keeps the full depth band it sits in — the same shape the
    band carver produced downstairs."""
    ext = plan.ext_wall
    x0, y0, x1, y1 = rect
    if not (ext <= x0 < x1 <= plan.w - ext and ext <= y0 < y1 <= plan.h - ext):
        raise CarveError(f"reservation {rect} lies outside the plot interior")

    regions: List[Region] = []
    face = 1

    def cut(axis: str, pos: int, keep_far: bool, label: str) -> None:
        """Split `face`; `keep_far` says the reservation is on the far side."""
        nonlocal face
        far = plan.split(face, axis, pos, min_side=1, name=f"_{label}")
        near, face = (face, far) if keep_far else (far, face)
        regions.append(Region(label, near, plan.face_bbox(near)))

    # depth cuts first (full width), then across cuts inside the band
    depth_axis = "h" if side in ("S", "N") else "v"
    lo, hi = (y0, y1) if depth_axis == "h" else (x0, x1)
    span_lo, span_hi = (x0, x1) if depth_axis == "h" else (y0, y1)
    plot_lo = ext
    plot_hi = (plan.h - ext) if depth_axis == "h" else (plan.w - ext)
    across_lo = ext
    across_hi = (plan.w - ext) if depth_axis == "h" else (plan.h - ext)
    across_axis = "v" if depth_axis == "h" else "h"

    if lo > plot_lo:
        cut(depth_axis, lo - INT_WALL_CELLS, keep_far=True, label="front")
    if hi < plot_hi:
        cut(depth_axis, hi, keep_far=False, label="rear")
    if span_lo > across_lo:
        cut(across_axis, span_lo - INT_WALL_CELLS, keep_far=True, label="left")
    if span_hi < across_hi:
        cut(across_axis, span_hi, keep_far=False, label="right")

    got = plan.face_bbox(face)
    if got != rect:
        raise CarveError(
            f"reservation landed at {got}, expected {rect}")
    return face, regions


def _assign(rooms: List[_RoomPlan], regions: List[Region], plan: GridPlan,
            side: str) -> Dict[int, List[_RoomPlan]]:
    """Distribute rooms among the residual regions.

    Capacity first (a region cannot hold more area than it has), seed
    position second: within the rooms a region can still take, the ones
    whose proposed position is nearest that region go there. Deterministic
    — no randomness, ties broken by name."""
    if not regions:
        raise CarveError("no space left after the reservation")

    order = sorted(regions, key=lambda r: -r.area_cells)
    capacity = {r.face: r.area_cells * WALL_ALLOWANCE for r in order}
    centers = {r.face: _center_depth_across(r.rect, plan, side)
               for r in order}
    out: Dict[int, List[_RoomPlan]] = {r.face: [] for r in order}

    for room in sorted(rooms, key=lambda r: (-r.area_cells, r.name)):
        best, best_cost = None, None
        for region in order:
            room_min = room.area_cells
            if capacity[region.face] < room_min and best is not None:
                continue
            cd, ca = centers[region.face]
            cost = (room.depth - cd) ** 2 + (room.across - ca) ** 2
            if capacity[region.face] < room_min:
                cost += 100.0            # last resort: overfill the region
            if best_cost is None or cost < best_cost:
                best, best_cost = region, cost
        assert best is not None
        out[best.face].append(room)
        capacity[best.face] -= room.area_cells

    empty = [r for r in order if not out[r.face]]
    for region in empty:
        # an empty region is a hole in the plan; hand it the nearest room
        # from the most crowded neighbour rather than leave dead space
        donor = max(order, key=lambda r: len(out[r.face]))
        if len(out[donor.face]) > 1:
            cd, ca = centers[region.face]
            room = min(out[donor.face],
                       key=lambda r: (r.depth - cd) ** 2 + (r.across - ca) ** 2)
            out[donor.face].remove(room)
            out[region.face].append(room)
    return out


def _center_depth_across(rect: Rect, plan: GridPlan,
                         side: str) -> Tuple[float, float]:
    """Region centre in the carver's (depth-from-entrance, across) space."""
    x0, y0, x1, y1 = rect
    cy = (y0 + y1) / 2 / plan.h
    cx = (x0 + x1) / 2 / plan.w
    if side == "S":
        return 1.0 - cy, cx
    if side == "N":
        return cy, cx
    if side == "E":
        return 1.0 - cx, cy
    return cx, cy


def carve_with_reservation(plan: GridPlan, request: EngineRequest,
                           proposal: LayoutProposal, *,
                           reserved_room: str, rect: Rect,
                           config: Optional[EngineConfig] = None
                           ) -> Dict[str, int]:
    """Carve the floor with `reserved_room` pinned to exactly `rect`.

    Same contract as carve_from_proposal: returns room name -> face id, and
    raises CarveError when the program cannot be packed around the
    reservation (the realizer's retry loop then perturbs and tries again).
    """
    side = request.entrance_side
    specs: Dict[str, RoomSpec] = {s.name: s for s in request.rooms}
    if reserved_room not in specs:
        raise CarveError(f"reserved room {reserved_room!r} is not in the "
                         f"program")

    rect = snap_reservation(plan, rect)
    reserved_face, regions = isolate(plan, rect, side)
    spec = specs[reserved_room]
    plan.rename(reserved_face, spec.name, spec.rtype)
    ids: Dict[str, int] = {spec.name: reserved_face}

    rest = [p for p in proposal.placements if p.room != reserved_room]
    if not rest:
        plan.verify()
        return ids

    usable = sum(r.area_cells for r in regions) * WALL_ALLOWANCE
    budgets = scaled_budgets(rest, specs, usable)
    rooms = room_plans(rest, specs, budgets, side)

    for face, members in _assign(rooms, regions, plan, side).items():
        carve_region(plan, face, members, side, ids, config)

    # regions that ended up with no rooms would leave an unnamed face in the
    # grid; that is a carve failure, not something to paper over
    unnamed = [rid for rid, room in plan.rooms.items()
               if room.name.startswith("_") or room.rtype == "undefined"]
    if unnamed:
        raise CarveError(
            f"{len(unnamed)} region(s) left unassigned around the "
            f"reservation")

    plan.verify()
    return ids
