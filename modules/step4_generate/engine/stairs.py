"""
stairs.py — the stair-fitting tier (Tier 3.5).

Runs after settle (which is what finally decides a face's dimensions) and
before connect (which needs to know the stair is a real, enterable room).
Pure adapter: the geometry lives in carve/stairs.py, the judgement lives in
engine/rules/stairs.py, and this only wires the two to the pipeline and
reports what happened.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from modules.step4_generate.carve import stairs as stair_geom
from modules.step4_generate.core import units
from modules.step4_generate.core.grid_plan import GridPlan
from modules.step4_generate.engine.contracts import EngineConfig, EngineRequest


class StairFitter:
    """Fits a buildable flight into every staircase face of the plan."""

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()

    def fit(self, plan: GridPlan, request: EngineRequest,
            room_ids: Dict[str, int]) -> List[str]:
        flights = stair_geom.fit_plan(
            plan, room_ids, floor_height_ft=self.config.floor_height_ft)
        notes: List[str] = []
        for f in flights:
            w, l = f.variant.min_footprint
            notes.append(
                f"stair fitted: {f.variant.kind} — {f.variant.risers} risers "
                f"@ {f.variant.riser_in:.2f}\", needs "
                f"{units.fmt_ft_in(w)} x {units.fmt_ft_in(l)}"
                f"{'' if f.both_landings else ' (single landing)'}")
        for rid in stair_geom.unfitted_faces(plan):
            x0, y0, x1, y1 = plan.face_bbox(rid)
            notes.append(
                f"stair DOES NOT FIT {plan.rooms[rid].name}: face is "
                f"{units.fmt_ft_in(x1 - x0)} x {units.fmt_ft_in(y1 - y0)}")
        return notes
