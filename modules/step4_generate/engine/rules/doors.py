"""DOR rules — door placement quality: swings, stubs, direction,
cross-ventilation. All computed on real opening geometry."""

from __future__ import annotations

from typing import List, Tuple

from modules.step4_generate.core import units
from modules.step4_generate.engine.rules.base import (
    HABITABLE, PRIVATE, SOCIAL, ReviewContext, rule,
)

_MIN_HINGE_STUB = units.cells(0, 9)


def _rects_intersect(a: Tuple[int, int, int, int],
                     b: Tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


@rule("DOR-001: door swings do not collide", "hard")
def swing_collisions(ctx: ReviewContext) -> None:
    doors = [op for op in ctx.plan.openings
             if op.kind == "door" and not op.is_exterior]
    rects: List[Tuple[Tuple[int, int, int, int], object]] = [
        (ctx.door_swing_rect(op), op) for op in doors
    ]
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if _rects_intersect(rects[i][0], rects[j][0]):
                a, b = rects[i][1], rects[j][1]
                ctx.hard.append(
                    f"DOR-001 swing collision: "
                    f"({ctx.rname(a.room_a)}<->{ctx.rname(a.room_b)}) vs "
                    f"({ctx.rname(b.room_a)}<->{ctx.rname(b.room_b)})")


@rule("DOR-002: hinges keep a 9\" corner stub", "soft")
def hinge_stubs(ctx: ReviewContext) -> None:
    """A door hinged hard into a corner cannot mount a frame. Measures
    the distance from each door's jambs to the ends of its host run."""
    violations = 0
    for op in ctx.plan.openings:
        if op.kind != "door" or op.is_exterior:
            continue
        host = [sw for sw in ctx.shared_walls()
                if sw.axis == op.axis
                and sw.wall_lo == op.wall_lo and sw.wall_hi == op.wall_hi
                and sw.along_lo <= op.along_lo
                and op.along_hi <= sw.along_hi]
        if not host:
            continue
        run = host[0]
        if (op.along_lo - run.along_lo < _MIN_HINGE_STUB
                or run.along_hi - op.along_hi < _MIN_HINGE_STUB):
            violations += 1
    ctx.breakdown["hinge_stub_violations"] = violations
    ctx.penalty += ctx.config.w_hinge_stub * violations


@rule("DOR-003: doors swing into the private side", "soft")
def swing_direction(ctx: ReviewContext) -> None:
    """A bedroom door must swing into the bedroom, not into the hall —
    real-life privacy + circulation convention."""
    wrong = 0
    for op in ctx.plan.openings:
        if op.kind != "door" or op.is_exterior:
            continue
        rt_a, rt_b = ctx.rtype(op.room_a), ctx.rtype(op.room_b)
        if rt_a in PRIVATE and rt_b in SOCIAL and op.swing != "a":
            wrong += 1
        elif rt_b in PRIVATE and rt_a in SOCIAL and op.swing != "b":
            wrong += 1
    ctx.breakdown["wrong_swing_doors"] = wrong
    ctx.penalty += ctx.config.w_swing_direction * wrong


@rule("DOR-004: cross-ventilation bonus (door opposite window)", "soft")
def cross_ventilation(ctx: ReviewContext) -> None:
    """A habitable room whose door and window sit on different wall axes
    (or opposite walls) ventilates through — bonus per such room."""
    door_axes = {}
    window_axes = {}
    for op in ctx.plan.openings:
        rooms = [op.room_a] if op.is_exterior else [op.room_a, op.room_b]
        for rid in rooms:
            if op.kind == "door":
                door_axes.setdefault(rid, set()).add(
                    (op.axis, op.wall_lo))
            elif op.kind == "window":
                window_axes.setdefault(rid, set()).add(
                    (op.axis, op.wall_lo))
    crossed = 0
    for spec in ctx.request.rooms:
        if spec.rtype not in HABITABLE:
            continue
        rid = ctx.room_ids[spec.name]
        pairs = [(d, w) for d in door_axes.get(rid, ())
                 for w in window_axes.get(rid, ())]
        if any(d[0] != w[0] or abs(d[1] - w[1]) > units.cells(6)
               for d, w in pairs):
            crossed += 1
    ctx.breakdown["cross_vent_rooms"] = crossed
    ctx.bonus += ctx.config.w_cross_vent * crossed
