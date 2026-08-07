"""
settle.py — Tier 4 adapter: the Squeeze & Settle optimizer (carve/settle.py)
wired into the engine. Targets are the request's room areas rescaled to the
plan's ACTUAL total room area, so the optimizer chases a feasible optimum
(the plot decides the pie; targets decide the shares).
"""

from __future__ import annotations

from typing import Dict

from modules.step4_generate.carve.settle import settle
from modules.step4_generate.carve.standards import (
    clamp_to_minimums, nbc_min_area_cells, type_min_side,
)
from modules.step4_generate.core.grid_plan import GridPlan
from modules.step4_generate.engine.contracts import EngineConfig, EngineRequest


def scaled_targets_cells(plan: GridPlan, request: EngineRequest,
                         room_ids: Dict[str, int]) -> Dict[int, int]:
    """Request areas rescaled to the plan's actual room total, then floored
    at NBC minimums (the same clamp the carver used — settle must not pull
    a legal room back below code)."""
    total_cells = sum(plan.face_area_cells(rid) for rid in room_ids.values())
    total_target = sum(s.target_sqft for s in request.rooms)
    raw = {
        room_ids[spec.name]: total_cells * spec.target_sqft / total_target
        for spec in request.rooms
    }
    floors = {
        room_ids[spec.name]: float(nbc_min_area_cells(spec.rtype) or 0)
        for spec in request.rooms
    }
    return {rid: max(1, v)
            for rid, v in clamp_to_minimums(raw, floors).items()}


class SqueezeSettler:
    def __init__(self, config: EngineConfig = None, frozen_rooms=None):
        """`frozen_rooms` names rooms (by name) whose walls must not move —
        the upper-floor staircase reserved over the flight below. Their
        area drift is still reported; they are simply not optimizable."""
        self.config = config or EngineConfig()
        self.frozen_rooms = set(frozen_rooms or ())

    def settle(self, plan: GridPlan, request: EngineRequest,
               room_ids: Dict[str, int]) -> float:
        """Runs settle in place; returns final mean relative area error."""
        targets = scaled_targets_cells(plan, request, room_ids)
        # minimum sides for EVERY room in the plan — engine-inserted rooms
        # (OTS shafts) must not be crushed by line slides either
        min_sides = {
            rid: type_min_side(room.rtype)
            for rid, room in plan.rooms.items()
        }
        frozen = {room_ids[name] for name in self.frozen_rooms
                  if name in room_ids}
        return settle(
            plan, targets, min_sides,
            max_sweeps=self.config.settle_sweeps,
            aspect_limit=self.config.settle_aspect_limit,
            aspect_weight=self.config.settle_aspect_weight,
            frozen=frozen,
        )


class NoopSettler:
    def settle(self, plan: GridPlan, request: EngineRequest,
               room_ids: Dict[str, int]) -> float:
        return -1.0
