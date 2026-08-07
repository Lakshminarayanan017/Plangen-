"""WAL rules — shared-wall discipline.

In a real plan every internal wall serves two rooms, long walls run
straight, and no toothpick wall fragments exist. These rules measure the
wall graph directly.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from modules.step4_generate.core import units
from modules.step4_generate.core.grid_plan import WALL
from modules.step4_generate.engine.rules.base import ReviewContext, cells_ft, rule

_JOG_MAX = units.cells(0, 9)        # offsets under 9" read as jogs
_TOOTHPICK = units.cells(1, 6)      # wall runs under 1'6"


@rule("WAL-001: internal walls are shared (wall efficiency)", "soft")
def wall_efficiency(ctx: ReviewContext) -> None:
    """shared wall length ÷ total internal wall length. Low values mean
    walls that serve one room only — dead structure."""
    plan = ctx.plan
    ext = plan.ext_wall
    total_wall_cells = int((plan.grid == WALL).sum())
    ring_cells = 2 * (plan.w + plan.h) * ext - 4 * ext * ext
    internal_cells = total_wall_cells - ring_cells
    if internal_cells < 100:                     # single-room plan: moot
        ctx.breakdown["wall_efficiency"] = 1.0
        return
    internal_len = internal_cells / 3.0          # 4.5" band thickness
    shared_len = sum(sw.length for sw in ctx.shared_walls())
    efficiency = min(1.0, shared_len / internal_len)
    ctx.breakdown["wall_efficiency"] = round(efficiency, 3)
    ctx.penalty += ctx.config.w_wall_efficiency * max(0.0, 0.70 - efficiency) * 10


@rule("WAL-002: collinear walls do not jog", "soft")
def wall_jogs(ctx: ReviewContext) -> None:
    """Two parallel wall runs offset by less than 9" whose extents nearly
    touch — the hallmark of an unresolved band boundary."""
    runs_by_axis: Dict[str, List[Tuple[int, int, int]]] = {"v": [], "h": []}
    for sw in ctx.shared_walls():
        runs_by_axis[sw.axis].append((sw.wall_lo, sw.along_lo, sw.along_hi))

    jogs = 0
    for axis, runs in runs_by_axis.items():
        runs.sort()
        for i, (pos_a, lo_a, hi_a) in enumerate(runs):
            for pos_b, lo_b, hi_b in runs[i + 1:]:
                offset = pos_b - pos_a
                if offset == 0:
                    continue
                if offset > _JOG_MAX:
                    break                        # runs sorted by position
                touching = lo_b <= hi_a + 3 and lo_a <= hi_b + 3
                if touching:
                    jogs += 1
    ctx.breakdown["wall_jogs"] = jogs
    ctx.penalty += ctx.config.w_wall_jog * jogs


@rule("WAL-003: no toothpick wall runs (< 1'6\")", "soft")
def toothpick_walls(ctx: ReviewContext) -> None:
    """Shared runs shorter than 1'6\" are barely-overlapping rooms — they
    read as drafting errors and can't host any opening. Soft (weighted
    hard-ish) until settle grows a jog-merge move to eliminate them."""
    count = 0
    worst_ft = 99.0
    for sw in ctx.shared_walls():
        if sw.length < _TOOTHPICK:
            count += 1
            worst_ft = min(worst_ft, cells_ft(sw.length))
    ctx.breakdown["toothpick_walls"] = count
    if worst_ft < 99.0:
        ctx.breakdown["min_shared_run_ft"] = round(worst_ft, 2)
    ctx.penalty += ctx.config.w_toothpick * count
