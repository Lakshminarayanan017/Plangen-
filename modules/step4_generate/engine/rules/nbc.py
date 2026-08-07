"""NBC/QLT rules — dimensional legality and proportion quality."""

from __future__ import annotations

from modules.step4_generate.carve.standards import NBC_MINIMUMS
from modules.step4_generate.core import units
from modules.step4_generate.engine.rules.base import HABITABLE, ReviewContext, cells_ft, rule

MIN_HABITABLE_SIDE_FT = 6


@rule("QLT-001: area drift vs plot-scaled targets", "soft")
def area_drift(ctx: ReviewContext) -> None:
    total_cells = sum(ctx.plan.face_area_cells(r)
                      for r in ctx.room_ids.values())
    total_target = sum(s.target_sqft for s in ctx.request.rooms)
    drifts = []
    for spec in ctx.request.rooms:
        share = total_cells * spec.target_sqft / total_target
        got = ctx.plan.face_area_cells(ctx.room_ids[spec.name])
        drifts.append(abs(got - share) / share)
    drift = sum(drifts) / len(drifts)
    ctx.breakdown["area_drift"] = round(drift, 3)
    ctx.penalty += ctx.config.w_area_drift * drift


@rule("QLT-002: habitable rooms not narrow", "soft")
def narrow_rooms(ctx: ReviewContext) -> None:
    narrow = 0
    min_side = units.cells(MIN_HABITABLE_SIDE_FT)
    for spec in ctx.request.rooms:
        rid = ctx.room_ids[spec.name]
        if spec.rtype in HABITABLE and ctx.plan.face_is_rect(rid):
            w, h = ctx.plan.clear_dims(rid)
            if min(w, h) < min_side:
                narrow += 1
    ctx.breakdown["narrow_habitable"] = narrow
    ctx.penalty += ctx.config.w_narrow * narrow


@rule("QLT-003: habitable aspect sanity", "soft")
def aspect(ctx: ReviewContext) -> None:
    worst = 1.0
    for spec in ctx.request.rooms:
        rid = ctx.room_ids[spec.name]
        if spec.rtype in HABITABLE and ctx.plan.face_is_rect(rid):
            w, h = ctx.plan.clear_dims(rid)
            worst = max(worst, max(w, h) / min(w, h))
    ctx.breakdown["worst_aspect"] = round(worst, 2)
    ctx.penalty += ctx.config.w_aspect * max(0.0, worst - 2.2)


@rule("NBC-001: minimum room dimensions", "soft")
def nbc_minimums(ctx: ReviewContext) -> None:
    violations = 0
    for spec in ctx.request.rooms:
        mins = NBC_MINIMUMS.get(spec.rtype)
        rid = ctx.room_ids[spec.name]
        if mins is None or not ctx.plan.face_is_rect(rid):
            continue
        min_sqft, min_side_ft = mins
        area = ctx.plan.area_sqft(rid)
        w, h = ctx.plan.clear_dims(rid)
        if area < min_sqft * 0.92 or cells_ft(min(w, h)) < min_side_ft - 0.25:
            violations += 1
    ctx.breakdown["nbc_violations"] = violations
    ctx.penalty += ctx.config.w_nbc * violations
