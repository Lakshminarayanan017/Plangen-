"""
strip_carver.py — v0 recursive carver.

Cuts a face into a room program by recursive wall insertion, sizing each cut
proportionally to the area budgets on either side. Deliberately simple: no
optimization, no adjacency objectives, no parti — its job is to exercise the
GridPlan substrate end-to-end and provide the initial partition that the
Squeeze & Settle optimizer (next milestone) will refine.

Guarantees inherited from the substrate: zero overlap, full coverage, walls
shared — regardless of how dumb the cut choices are.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from modules.step4_generate.core import units
from modules.step4_generate.core.grid_plan import INT_WALL_CELLS, CarveError, GridPlan


@dataclass(frozen=True)
class RoomBrief:
    """v0 room request. The full RoomSpec (blueprint §3) replaces this later."""
    name: str
    rtype: str
    target_sqft: float


DEFAULT_MIN_SIDE = units.cells(4)   # no room narrower than 4'

# Service rooms may legitimately be narrower than habitable rooms.
_TYPE_MIN_SIDE = {
    "bathroom": units.cells(3),
    "toilet":   units.cells(3),
    "store":    units.cells(3),
    "storage":  units.cells(3),
    "utility":  units.cells(3),
    "ots":      units.cells(3),
}


def _group_min_side(briefs: Sequence[RoomBrief]) -> int:
    return min(_TYPE_MIN_SIDE.get(b.rtype, DEFAULT_MIN_SIDE) for b in briefs)


def carve(plan: GridPlan, face_id: int, briefs: Sequence[RoomBrief], *,
          min_side: int = None) -> List[int]:
    """Recursively subdivide `face_id` into one room per brief (order
    preserved: earlier briefs land on the lower-coordinate side of each cut).
    Returns the face ids in brief order."""
    if not briefs:
        raise CarveError("no room briefs given")
    if len(briefs) == 1:
        plan.rename(face_id, briefs[0].name, briefs[0].rtype)
        return [face_id]

    left, right = _balance_split(briefs)
    cut_min = min_side if min_side is not None else min(
        _group_min_side(left), _group_min_side(right))
    axis, pos = _choose_cut(plan, face_id, left, right, cut_min)
    new_id = plan.split(face_id, axis, pos, thick=INT_WALL_CELLS,
                        name="(carving)", min_side=cut_min)
    ids_left = carve(plan, face_id, left, min_side=min_side)
    ids_right = carve(plan, new_id, right, min_side=min_side)
    return ids_left + ids_right


def _balance_split(briefs: Sequence[RoomBrief]
                   ) -> tuple[List[RoomBrief], List[RoomBrief]]:
    """Split the (order-preserved) brief list at the index that best balances
    total area between the two sides."""
    total = sum(b.target_sqft for b in briefs)
    best_k, best_err = 1, float("inf")
    prefix = 0.0
    for k in range(1, len(briefs)):
        prefix += briefs[k - 1].target_sqft
        err = abs(prefix - (total - prefix))
        if err < best_err:
            best_k, best_err = k, err
    return list(briefs[:best_k]), list(briefs[best_k:])


def _choose_cut(plan: GridPlan, face_id: int, left: Sequence[RoomBrief],
                right: Sequence[RoomBrief], min_side: int) -> tuple[str, int]:
    """Cut across the face's longer dimension at a position proportional to
    the two area budgets, snapped to the 3" module and clamped so both sides
    stay buildable. Falls back to the other axis if the first can't fit."""
    w, h = plan.clear_dims(face_id)
    x0, y0, x1, y1 = plan.face_bbox(face_id)
    area_l = sum(b.target_sqft for b in left)
    ratio = area_l / (area_l + sum(b.target_sqft for b in right))

    for axis in (("v", "h") if w >= h else ("h", "v")):
        lo, hi = (x0, x1) if axis == "v" else (y0, y1)
        span = hi - lo - INT_WALL_CELLS
        pos = lo + units.snap_to_module(round(span * ratio))
        pos = max(lo + min_side, min(pos, hi - INT_WALL_CELLS - min_side))
        if lo + min_side <= pos and pos + INT_WALL_CELLS + min_side <= hi:
            return axis, pos
    raise CarveError(
        f"face {face_id} ({units.fmt_ft_in(w)} x {units.fmt_ft_in(h)}) cannot "
        f"fit a cut with min side {units.fmt_ft_in(min_side)}"
    )
