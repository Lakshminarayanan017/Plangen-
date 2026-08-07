"""STR rules — structural invariants. Unrepairable by design: a carver
bug must fail loudly, never be patched over."""

from __future__ import annotations

from modules.step4_generate.engine.rules.base import PARKING, ReviewContext, rule


@rule("STR-001: structural invariants", "hard")
def structural(ctx: ReviewContext) -> None:
    try:
        ctx.plan.verify()
    except AssertionError as e:
        ctx.hard.append(f"STR-001 structural: {e}")


@rule("STR-002: all requested rooms exist", "hard")
def rooms_exist(ctx: ReviewContext) -> None:
    for spec in ctx.request.rooms:
        if spec.name not in ctx.room_ids:
            ctx.hard.append(f"STR-002 missing room: {spec.name}")


@rule("STR-003: parking/garage has a vehicle gate onto the exterior", "hard")
def parking_vehicle_gate(ctx: ReviewContext) -> None:
    """A carport with no vehicle access is not a carport — it's a walled
    interior room. Checks the OUTCOME (carve/connections.py's
    place_vehicle_gate cuts a doorless exterior "wide" opening onto the
    room), not the geometry, so this stays correct even if the gate
    placement strategy changes later."""
    for spec in ctx.request.rooms:
        if spec.rtype not in PARKING:
            continue
        rid = ctx.room_ids.get(spec.name)
        if rid is None:
            continue
        has_gate = any(op.is_exterior and op.kind == "wide"
                       and op.room_a == rid for op in ctx.plan.openings)
        if not has_gate:
            ctx.hard.append(
                f"STR-003 {spec.name} has no vehicle gate onto the exterior")
