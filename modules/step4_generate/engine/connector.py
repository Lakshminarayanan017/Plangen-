"""
connector.py — engine stage between settle and validation: applies the
openness gradient + windows (carve/connections.py) and honors any extra
user wishes the rules didn't already create.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from modules.step4_generate.carve.connections import connect
from modules.step4_generate.core.grid_plan import CarveError, GridPlan
from modules.step4_generate.engine.contracts import EngineConfig, EngineRequest


class RuleConnector:
    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()

    def connect(self, plan: GridPlan, request: EngineRequest,
                room_ids: Dict[str, int]) -> List[str]:
        notes = connect(plan, request, room_ids, self.config)

        # user wishes on top of the rules (already-open pairs are no-ops)
        opened = {frozenset((op.room_a, op.room_b))
                  for op in plan.openings if not op.is_exterior}
        for wish in request.wishes:
            a = room_ids.get(wish.room_a)
            b = room_ids.get(wish.room_b)
            if a is None or b is None or frozenset((a, b)) in opened:
                continue
            try:
                plan.add_opening(a, b, wish.kind)
                notes.append(f"wish honored: {wish.room_a} <-> {wish.room_b}")
            except CarveError as e:
                notes.append(f"wish skipped: {wish.room_a} <-> "
                             f"{wish.room_b}: {e}")
        return notes
