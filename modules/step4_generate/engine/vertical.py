"""
vertical.py — the rules that only exist BETWEEN floors.

A per-floor reviewer cannot see these: every one of them compares two plans.
A staircase that lands in a different place upstairs is not a staircase; a
first-floor bathroom over a living room needs a drainage stack through the
ceiling of the room you sit in; an upstairs partition that misses every wall
below has nothing holding it up.

    VRT-001 hard  the stair occupies the SAME cells on every floor it serves
    VRT-002 hard  no upper room hangs outside the floor below
    VRT-003 soft  wet rooms stack over wet rooms (plumbing)
    VRT-004 soft  interior walls land on walls below (load path)
    VRT-005 soft  the upper floor's entrance-side frontage is not blocked
                  by the stair arriving into a corner

Same shape as the per-floor reviewer: hard violations disqualify, soft ones
are weighted penalties with evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from modules.step4_generate.core import units
from modules.step4_generate.core.grid_plan import OUTSIDE, WALL, GridPlan
from modules.step4_generate.engine.contracts import EngineConfig
from modules.step4_generate.engine.rules.base import KITCHEN, WET

# A stair whose footprints overlap less than this between floors is not the
# same stair. 0.9 rather than 1.0 because settle may have moved a wall by a
# cell or two; anything below is a different room in a different place.
STAIR_OVERLAP_MIN = 0.90

STACKABLE_WET = WET | KITCHEN


@dataclass
class VerticalVerdict:
    hard: List[str] = field(default_factory=list)
    soft_score: float = 100.0
    breakdown: Dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.hard


def _type_mask(plan: GridPlan, room_ids: Dict[str, int],
               rtypes: set) -> np.ndarray:
    mask = np.zeros_like(plan.grid, dtype=bool)
    for rid in room_ids.values():
        if plan.rooms[rid].rtype in rtypes:
            mask |= plan.grid == rid
    return mask


def _stair_mask(plan: GridPlan, room_ids: Dict[str, int]) -> np.ndarray:
    return _type_mask(plan, room_ids, {"staircase"})


def _interior_wall_mask(plan: GridPlan) -> np.ndarray:
    """WALL cells that are not part of the exterior ring."""
    mask = plan.grid == WALL
    ext = plan.ext_wall
    inner = np.zeros_like(mask)
    inner[ext:plan.h - ext, ext:plan.w - ext] = True
    return mask & inner


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of `a` that also lies in `b` (0 when `a` is empty)."""
    total = int(a.sum())
    return float((a & b).sum()) / total if total else 0.0


def check_stacking(lower: GridPlan, lower_ids: Dict[str, int],
                   upper: GridPlan, upper_ids: Dict[str, int],
                   config: Optional[EngineConfig] = None,
                   *, upper_floor: int = 1) -> VerticalVerdict:
    """Review the upper floor against the one it sits on."""
    config = config or EngineConfig()
    verdict = VerticalVerdict()
    penalty = 0.0

    if lower.grid.shape != upper.grid.shape:
        verdict.hard.append(
            f"VRT-002 floor {upper_floor} lattice {upper.grid.shape} does "
            f"not match the floor below {lower.grid.shape}")
        verdict.soft_score = 0.0
        return verdict

    # ── VRT-001: stair continuity ────────────────────────────────────────
    low_stair = _stair_mask(lower, lower_ids)
    up_stair = _stair_mask(upper, upper_ids)
    if low_stair.any() and up_stair.any():
        overlap = _overlap(up_stair, low_stair)
        verdict.breakdown["stair_overlap"] = round(overlap, 3)
        if overlap < STAIR_OVERLAP_MIN:
            verdict.hard.append(
                f"VRT-001 floor {upper_floor} staircase overlaps the one "
                f"below by only {overlap:.0%} (needs "
                f"{STAIR_OVERLAP_MIN:.0%}) — the flights do not meet")
    elif low_stair.any() and not up_stair.any():
        verdict.hard.append(
            f"VRT-001 floor {upper_floor} has no staircase above the "
            f"flight arriving from below")

    # ── VRT-002: nothing hangs off the building ──────────────────────────
    below_built = lower.grid != OUTSIDE
    above_built = upper.grid != OUTSIDE
    hanging = int((above_built & ~below_built).sum())
    if hanging:
        verdict.breakdown["cantilever_sqft"] = round(
            units.area_sqft(hanging), 1)
        verdict.hard.append(
            f"VRT-002 floor {upper_floor} overhangs the floor below by "
            f"{units.area_sqft(hanging):.0f} sqft")

    # ── VRT-003: wet stacks ──────────────────────────────────────────────
    low_wet = _type_mask(lower, lower_ids, STACKABLE_WET)
    up_wet = _type_mask(upper, upper_ids, STACKABLE_WET)
    if up_wet.any():
        stacked = _overlap(up_wet, low_wet)
        verdict.breakdown["wet_stack_fraction"] = round(stacked, 3)
        penalty += config.w_wet_stack * (1.0 - stacked) * 100

    # ── VRT-004: load path ───────────────────────────────────────────────
    low_walls = _interior_wall_mask(lower)
    up_walls = _interior_wall_mask(upper)
    if up_walls.any():
        # a wall is "supported" if a wall below lies within one partition
        # thickness of it — real construction tolerates a small offset
        pad = units.INT_WALL_CELLS
        supported = low_walls.copy()
        for shift in range(1, pad + 1):
            supported |= np.roll(low_walls, shift, axis=0)
            supported |= np.roll(low_walls, -shift, axis=0)
            supported |= np.roll(low_walls, shift, axis=1)
            supported |= np.roll(low_walls, -shift, axis=1)
        aligned = _overlap(up_walls, supported)
        verdict.breakdown["wall_alignment"] = round(aligned, 3)
        penalty += config.w_wall_alignment * max(0.0, 0.75 - aligned) * 100

    # ── VRT-005: the stair does not arrive into a dead corner ────────────
    if up_stair.any():
        neighbors = _stair_neighbors(upper, upper_ids)
        verdict.breakdown["stair_upper_neighbors"] = len(neighbors)
        if len(neighbors) < 2:
            penalty += config.w_stair_landing

    verdict.soft_score = round(max(0.0, 100.0 - penalty), 2)
    return verdict


def _stair_neighbors(plan: GridPlan, room_ids: Dict[str, int]) -> List[int]:
    stair_ids = {rid for rid in room_ids.values()
                 if plan.rooms[rid].rtype == "staircase"}
    out = set()
    for op in plan.openings:
        if op.is_exterior:
            continue
        if op.room_a in stair_ids:
            out.add(op.room_b)
        elif op.room_b in stair_ids:
            out.add(op.room_a)
    return sorted(out)


def stair_footprint(plan: GridPlan, room_ids: Dict[str, int]
                    ) -> Optional[Tuple[int, int, int, int]]:
    """(x0, y0, x1, y1) of the stair face, for conditioning the floor above.
    None when the plan has no staircase."""
    for rid in room_ids.values():
        if plan.rooms[rid].rtype == "staircase":
            return plan.face_bbox(rid)
    return None


def wet_footprints(plan: GridPlan, room_ids: Dict[str, int]
                   ) -> List[Tuple[int, int, int, int]]:
    """Bounding boxes of every wet/kitchen face — the stack positions the
    floor above should try to land on."""
    return [plan.face_bbox(rid) for rid in room_ids.values()
            if plan.rooms[rid].rtype in STACKABLE_WET]
