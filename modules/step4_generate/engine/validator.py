"""
validator.py — the reviewer runner.

Rules live in engine/rules/* (registered at import); this module runs
them: hard rules first (any violation disqualifies — a failing plan is
never shown), then soft rules (config-weighted penalties/bonuses with a
per-rule evidence breakdown, so every score is explainable).
"""

from __future__ import annotations

from typing import Dict

from modules.step4_generate.core.grid_plan import GridPlan
from modules.step4_generate.engine.contracts import EngineConfig, EngineRequest, Verdict
from modules.step4_generate.engine.rules import RULES, ReviewContext


class BasicValidator:
    def __init__(self, config: EngineConfig = None):
        self.config = config or EngineConfig()

    def check(self, plan: GridPlan, request: EngineRequest,
              room_ids: Dict[str, int]) -> Verdict:
        ctx = ReviewContext(plan=plan, request=request, room_ids=room_ids,
                            config=self.config)

        for rule_id, severity, fn in RULES:
            if severity == "hard":
                fn(ctx)
                if ctx.hard and rule_id.startswith("STR-0"):
                    # structural failures (STR-001..003) make later geometry
                    # queries unreliable — stop immediately. NOT the stair
                    # group (STR-S*): an unbuildable staircase disqualifies
                    # the plan but leaves its geometry perfectly readable, so
                    # the remaining rules still produce useful evidence.
                    return Verdict(hard=ctx.hard, soft_score=0.0,
                                   breakdown=ctx.breakdown)
        if ctx.hard:
            return Verdict(hard=ctx.hard, soft_score=0.0,
                           breakdown=ctx.breakdown)

        for rule_id, severity, fn in RULES:
            if severity == "soft":
                fn(ctx)
        return Verdict(hard=[], breakdown=ctx.breakdown,
                       soft_score=round(100.0 - ctx.penalty + ctx.bonus, 2))
