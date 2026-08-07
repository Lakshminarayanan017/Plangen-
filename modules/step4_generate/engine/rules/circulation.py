"""CIR rules — entrance, reachability, circulation depth."""

from __future__ import annotations

from modules.step4_generate.engine.rules.base import ReviewContext, rule

# rtype → max hops from the entrance room
_DEPTH_NORMS = {"living_room": 2, "drawing_room": 2, "dining_room": 2,
                "kitchen": 3}


@rule("CIR-001: main entrance exists", "hard")
def main_entrance(ctx: ReviewContext) -> None:
    """Ground floor: a front door. Upper floors: the staircase you arrive
    on (see ReviewContext.entrance_room)."""
    if ctx.entrance_room() is None:
        ctx.hard.append(
            "CIR-001 no staircase to arrive on" if ctx.request.floor_index
            else "CIR-001 no main entrance door")


@rule("CIR-002: every room reachable from entrance", "hard")
def reachability(ctx: ReviewContext) -> None:
    if ctx.entrance_room() is None:
        return                                  # CIR-001 already fired
    hops = ctx.hops_from_entrance()
    unreachable = ctx.requested_ids() - set(hops)
    if unreachable:
        names = ", ".join(sorted(ctx.rname(r) for r in unreachable))
        ctx.hard.append(f"CIR-002 unreachable rooms: {names}")


@rule("CIR-003: circulation depth norms", "soft")
def circulation_depth(ctx: ReviewContext) -> None:
    hops = ctx.hops_from_entrance()
    excess = 0
    for spec in ctx.request.rooms:
        norm = _DEPTH_NORMS.get(spec.rtype)
        rid = ctx.room_ids[spec.name]
        if norm is not None and rid in hops:
            excess += max(0, hops[rid] - norm)
    ctx.breakdown["circ_excess_hops"] = excess
    ctx.penalty += ctx.config.w_circ_depth * excess
