"""OPN/HYG/PRV rules — the openness gradient and its privacy/hygiene
boundaries."""

from __future__ import annotations

from modules.step4_generate.engine.rules.base import (
    KITCHEN, PRIVATE, WET, ReviewContext, rule,
)

_SOCIAL_CORE = {"living_room", "drawing_room", "dining_room"}


@rule("OPN-001: kitchen is doorless (wide openings only)", "hard")
def kitchen_doorless(ctx: ReviewContext) -> None:
    for spec in ctx.request.rooms:
        if spec.rtype not in KITCHEN:
            continue
        kid = ctx.room_ids.get(spec.name)
        for op in ctx.plan.openings:
            if op.kind == "door" and not op.is_exterior \
                    and kid in (op.room_a, op.room_b):
                ctx.hard.append(f"OPN-001 {spec.name} has a door leaf")


@rule("HYG-001: no opening joins a wet room to the kitchen", "hard")
def wet_kitchen_separation(ctx: ReviewContext) -> None:
    for op in ctx.plan.openings:
        if op.is_exterior:
            continue
        types = {ctx.rtype(op.room_a), ctx.rtype(op.room_b)}
        if types & WET and types & KITCHEN:
            ctx.hard.append(
                f"HYG-001 wet room opens into kitchen "
                f"({ctx.rname(op.room_a)} <-> {ctx.rname(op.room_b)})")


@rule("OPN-002: openness bonus (wide connections)", "soft")
def openness(ctx: ReviewContext) -> None:
    wides = sum(1 for op in ctx.plan.openings if op.kind == "wide")
    ctx.breakdown["wide_openings"] = wides
    ctx.bonus += ctx.config.w_wide_bonus * min(wides, 6)


@rule("PRV-001: no doors joining two private rooms", "soft")
def privacy_doors(ctx: ReviewContext) -> None:
    count = sum(
        1 for op in ctx.plan.openings
        if not op.is_exterior and op.kind == "door"
        and ctx.rtype(op.room_a) in PRIVATE
        and ctx.rtype(op.room_b) in PRIVATE
    )
    ctx.breakdown["privacy_doors"] = count
    ctx.penalty += ctx.config.w_privacy_door * count


@rule("PRV-002: wet rooms do not open into living/dining", "soft")
def wet_into_social(ctx: ReviewContext) -> None:
    count = 0
    for op in ctx.plan.openings:
        if op.is_exterior:
            continue
        types = (ctx.rtype(op.room_a), ctx.rtype(op.room_b))
        if (types[0] in WET and types[1] in _SOCIAL_CORE) or \
                (types[1] in WET and types[0] in _SOCIAL_CORE):
            count += 1
    ctx.breakdown["wet_social_doors"] = count
    ctx.penalty += ctx.config.w_wet_social * count
