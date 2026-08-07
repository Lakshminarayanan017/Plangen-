"""VEN rules — daylight and ventilation, OTS-aware."""

from __future__ import annotations

from typing import Set

from modules.step4_generate.core import units
from modules.step4_generate.engine.rules.base import HABITABLE, ReviewContext, rule


def _ots_adjacent_ids(ctx: ReviewContext) -> Set[int]:
    """Rooms sharing a ≥2' wall with an open-to-sky shaft (light source)."""
    ots = {rid for rid, room in ctx.plan.rooms.items()
           if room.rtype == "ots"}
    if not ots:
        return set()
    adjacent = set()
    for sw in ctx.shared_walls():
        if sw.length < units.cells(2):
            continue
        if sw.room_a in ots:
            adjacent.add(sw.room_b)
        elif sw.room_b in ots:
            adjacent.add(sw.room_a)
    return adjacent


@rule("VEN-001: habitable rooms have light exposure (exterior or OTS)",
      "soft")
def daylight(ctx: ReviewContext) -> None:
    lit_by_ots = _ots_adjacent_ids(ctx)
    dark = 0
    for spec in ctx.request.rooms:
        if spec.rtype not in HABITABLE:
            continue
        rid = ctx.room_ids[spec.name]
        if rid in lit_by_ots:
            continue
        runs = []
        for side in ("N", "E", "S", "W"):
            runs += ctx.plan.exterior_runs(rid, side)
        if not any(hi - lo >= units.cells(3) for lo, hi in runs):
            dark += 1
    ctx.breakdown["daylightless"] = dark
    ctx.penalty += ctx.config.w_daylight * dark


@rule("VEN-002: habitable rooms have a window (exterior or onto OTS)",
      "soft")
def windows(ctx: ReviewContext) -> None:
    windowed = set()
    for op in ctx.plan.openings:
        if op.kind != "window":
            continue
        windowed.add(op.room_a)
        if not op.is_exterior:
            windowed.add(op.room_b)
    missing = 0
    for spec in ctx.request.rooms:
        if spec.rtype in HABITABLE \
                and ctx.room_ids[spec.name] not in windowed:
            missing += 1
    ctx.breakdown["windowless"] = missing
    ctx.penalty += ctx.config.w_window * missing
