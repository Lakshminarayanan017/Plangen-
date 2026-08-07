"""STR-S rules — the staircase as a reviewed, buildable object.

A staircase is the one room whose dimensions are dictated by physics rather
than by taste: the floor height fixes the riser count, the riser count fixes
the run, and a face shorter than that run cannot be climbed. These rules
check the FITTED geometry (carve/stairs.py attached it to plan.stairs during
the stair-fitting tier), so they judge the same object the renderer draws.

STR-S01/02/03 are hard: a multi-floor plan with no stair, an unbuildable
stair, or a stair you cannot walk to is not a plan.
"""

from __future__ import annotations

from modules.step4_generate.carve import stairs as stair_geom
from modules.step4_generate.core import units
from modules.step4_generate.engine.rules.base import STAIR, ReviewContext, cells_ft, rule

# STR-S03: hops from the entrance room, through openings, to the stair.
MAX_ENTRANCE_HOPS = 2


def _stair_ids(ctx: ReviewContext) -> list:
    return [rid for rid in ctx.requested_ids() if ctx.rtype(rid) in STAIR]


@rule("STR-S01: multi-floor plans have a staircase", "hard")
def stair_exists(ctx: ReviewContext) -> None:
    if ctx.request.n_floors <= 1:
        return
    if not _stair_ids(ctx):
        ctx.hard.append(
            f"STR-S01 no staircase in a {ctx.request.n_floors}-floor plan")


@rule("STR-S02: the staircase face fits a buildable flight", "hard")
def stair_fits(ctx: ReviewContext) -> None:
    """The face must hold a real variant for THIS floor height. Evidence is
    the face's own dimensions against the smallest footprint that exists."""
    ids = _stair_ids(ctx)
    if not ids:
        return
    fitted = {f.face_id: f for f in getattr(ctx.plan, "stairs", [])}
    smallest = min(stair_geom.variants(ctx.config.floor_height_ft),
                   key=lambda v: v.min_area_cells)
    need_w, need_l = smallest.min_footprint

    for rid in ids:
        flight = fitted.get(rid)
        if flight is not None:
            ctx.breakdown["stair_kind"] = flight.variant.kind
            ctx.breakdown["stair_risers"] = flight.variant.risers
            continue
        x0, y0, x1, y1 = ctx.plan.face_bbox(rid)
        ctx.hard.append(
            f"STR-S02 {ctx.rname(rid)} is "
            f"{units.fmt_ft_in(x1 - x0)} x {units.fmt_ft_in(y1 - y0)} — no "
            f"flight fits (smallest is {units.fmt_ft_in(need_w)} x "
            f"{units.fmt_ft_in(need_l)} for a "
            f"{ctx.config.floor_height_ft:g}' floor height)")


@rule("STR-S03: the staircase is reachable from the entrance", "hard")
def stair_reachable(ctx: ReviewContext) -> None:
    """Not merely connected — CLOSE. A stair you reach by walking through a
    bedroom is a planning failure even though the graph says 'reachable'."""
    ids = _stair_ids(ctx)
    if not ids:
        return
    hops = ctx.hops_from_entrance()
    for rid in ids:
        d = hops.get(rid)
        if d is None:
            ctx.hard.append(
                f"STR-S03 {ctx.rname(rid)} is not reachable from the entrance")
        elif d > MAX_ENTRANCE_HOPS:
            ctx.hard.append(
                f"STR-S03 {ctx.rname(rid)} is {d} rooms deep from the "
                f"entrance (max {MAX_ENTRANCE_HOPS})")
        else:
            ctx.breakdown["stair_entrance_hops"] = d


@rule("STR-S04: landings are clear at both ends", "soft")
def stair_landings(ctx: ReviewContext) -> None:
    """A flight that arrives straight into a doorway has no landing. The
    fitter records whether the face had room for a 3'0" landing at BOTH
    ends; anything less is legal but cramped."""
    flights = [f for f in getattr(ctx.plan, "stairs", [])
               if f.face_id in set(_stair_ids(ctx))]
    if not flights:
        return
    cramped = sum(1 for f in flights if not f.both_landings)
    ctx.breakdown["stair_single_landing"] = cramped
    ctx.penalty += ctx.config.w_stair_landing * cramped


@rule("STR-S05: the staircase does not take the prime frontage", "soft")
def stair_frontage(ctx: ReviewContext) -> None:
    """Daylight on the entrance frontage belongs to habitable rooms. A stair
    parked across it steals the plot's best light for a space nobody sits
    in. Penalized in proportion to how much of that frontage it eats."""
    ids = _stair_ids(ctx)
    if not ids:
        return
    side = ctx.request.entrance_side
    frontage = ctx.plan.w if side in ("N", "S") else ctx.plan.h
    stolen = sum(hi - lo
                 for rid in ids
                 for lo, hi in ctx.plan.exterior_runs(rid, side))
    fraction = stolen / frontage if frontage else 0.0
    if stolen:
        ctx.breakdown["stair_frontage_ft"] = round(cells_ft(stolen), 2)
    ctx.penalty += ctx.config.w_stair_frontage * fraction * 100
