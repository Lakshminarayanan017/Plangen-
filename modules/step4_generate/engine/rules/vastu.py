"""
VAS rules — Vastu as reviewed geometry, not as a label on the output.

Until now Vastu was computed in full by step 3 and thrown away by the bridge:
`RoomSpec` had nowhere to put a compass direction, the engine had no idea
which way north was, and a "Vastu compliant" single-floor plan came out
byte-identical to a non-Vastu one. These rules are the other half — they read
the direction each room was assigned and score whether the carved plan
actually honours it.

Every rule stands down unless the user asked for Vastu AND the rooms carry
directions (`ReviewContext.vastu_active`), so a non-Vastu request is scored
exactly as it was before this file existed.

Sources, in order of authority:
  • data/vastuRules1.json      the 81-pada Paramasayika Mandala: 32 gates,
                               the Brahmasthan, 6 marma diagonals, and the
                               global structural modifiers
  • sources/enricher_rules.json  per-room preferred/prohibited compass, which
                               step 3 attaches to each room
  • VAS-008 and VAS-011 are ENGINE-DERIVED from mainstream Vastu (the SW/NE
    mass gradient and the north-east corner cut). They are marked as such,
    weighted softly, and are the only two rules here not read from data.

Severity. Everything is soft by default. `EngineConfig.vastu_hard` promotes
a HARD-strength room violation to a disqualification — off until the harness
shows no brief loses its plan to it, for the same reason FSP-001 is still
soft: a reviewer may legitimately detect more than the generator can yet
guarantee.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from modules.step4_generate.core import units
from modules.step4_generate.core.grid_plan import OUTSIDE, WALL
from modules.step4_generate.engine.rules.base import (
    KITCHEN, STAIR, WET, ReviewContext, cells_ft, rule,
)
from modules.step4_generate.engine.vastu.compass import sector_distance
from modules.step4_generate.engine.vastu.mandala import point_segment_distance

# Usages the mandala bars from the Brahmasthan, mapped onto engine rtypes.
_BRAHMA_BLOCK_RTYPES = {
    "toilet": WET | {"bathroom", "toilet"},
    "kitchen_stove": KITCHEN,
}

# A marma line is "hit" when a door centre lies within this distance of it.
# 2 cells = 3", the tolerance a door frame's centre line is drawn to.
#
# MEASURED, not chosen: counting every shared-wall run endpoint at a 3-cell
# tolerance scored a median of 7 hits on ordinary plans (max 19), which at
# any useful weight is a ~21-point constant tax rather than a signal. The
# mandala's own wording is narrower than the first implementation read it —
# "no structural column, main wall INTERSECTION, or DOOR FRAME centre line" —
# so this counts door and wide-opening centres plus true cross junctions,
# both of which are few and both of which a draughtsman can actually move.
_MARMA_TOLERANCE_CELLS = 2.0

# Cost of being N compass steps from the ideal sector, per room.
#
# Convex on purpose. A room one sector off its ideal is what the report calls
# "near" and is a perfectly good plan; a room on the OPPOSITE side of the
# house is the failure Vastu exists to prevent. A linear cost charged those
# the same per step and turned a grade-A layout into a 24-point loss.
_SECTOR_COST = (0.0, 0.25, 1.0, 2.0, 3.0)


def _requested(ctx: ReviewContext) -> List:
    return [s for s in ctx.request.rooms if s.name in ctx.room_ids]


# ═════════════════════ room placement ═══════════════════════════════════════

@rule("VAS-001: rooms sit in their Vastu sector", "soft")
def room_sector(ctx: ReviewContext) -> None:
    """Distance, in compass steps, from each room's ideal sector.

    Graded rather than binary on purpose: a pooja room in the north is one
    step from its ideal north-east and is a good plan; one in the south-west
    is four steps away and is the thing Vastu exists to prevent. A binary
    rule would score those identically.
    """
    if not ctx.vastu_active():
        return
    steps = 0
    cost = 0.0
    on_target = 0
    for spec in _requested(ctx):
        if not spec.vastu_dir:
            continue
        got = ctx.sector_of(ctx.room_ids[spec.name])
        d = sector_distance(got, spec.vastu_dir)
        steps += d
        cost += _SECTOR_COST[min(d, len(_SECTOR_COST) - 1)]
        on_target += int(d == 0)
    ctx.breakdown["vastu_sector_steps"] = steps
    ctx.breakdown["vastu_rooms_on_target"] = on_target
    ctx.penalty += ctx.config.w_vastu_sector * cost


@rule("VAS-002: no room in a prohibited Vastu sector", "hard")
def prohibited_sector(ctx: ReviewContext) -> None:
    """A barred sector is a different claim from a missed preference: Vastu
    says a toilet in the north-east is actively harmful, not merely
    suboptimal. HARD-strength rooms disqualify when `vastu_hard` is on."""
    if not ctx.vastu_active():
        return
    violations: List[str] = []
    hard: List[str] = []
    for spec in _requested(ctx):
        if not spec.vastu_avoid:
            continue
        got = ctx.sector_of(ctx.room_ids[spec.name])
        if got not in spec.vastu_avoid:
            continue
        message = (f"VAS-002 {spec.name} is in the {got} sector, which Vastu "
                   f"bars for {spec.rtype}")
        violations.append(message)
        if spec.vastu_strength == "hard":
            hard.append(message)
    ctx.breakdown["vastu_prohibited"] = len(violations)
    if hard and ctx.config.vastu_hard:
        ctx.hard.extend(hard)
    ctx.penalty += ctx.config.w_vastu_prohibited * len(violations)


# ═════════════════════ the Brahmasthan ══════════════════════════════════════

@rule("VAS-003: the Brahmasthan carries no barred use", "hard")
def brahmasthan_use(ctx: ReviewContext) -> None:
    """The centre of the plot — the central 3x3 of the 81 padas, one ninth of
    the area — must not hold a toilet or a kitchen.

    This is the most recognisable rule in Vastu and the one a client will
    check first, so it is measured on real geometry: the fraction of the
    Brahmasthan rectangle each barred room actually occupies.
    """
    if not ctx.vastu_active():
        return
    mandala = ctx.mandala()
    if not mandala.available or mandala.brahma is None:
        return

    plan = ctx.plan
    x0, y0, x1, y1 = ctx.frame().brahmasthan_rect(plan.h, plan.w)
    core = plan.grid[y0:y1, x0:x1]
    total = max(1, core.size)

    barred = {r for usage, rtypes in _BRAHMA_BLOCK_RTYPES.items()
              if usage in mandala.brahma_blocked_usages() for r in rtypes}
    worst = 0.0
    offenders: List[str] = []
    for spec in _requested(ctx):
        if spec.rtype not in barred:
            continue
        share = float((core == ctx.room_ids[spec.name]).sum()) / total
        if share > 0.02:                      # a stray cell is not an intrusion
            offenders.append(f"{spec.name} ({share:.0%})")
            worst = max(worst, share)

    ctx.breakdown["vastu_brahma_intrusion"] = round(worst, 3)
    if offenders:
        message = ("VAS-003 the Brahmasthan is occupied by "
                   + ", ".join(offenders))
        if ctx.config.vastu_hard:
            ctx.hard.append(message)
        ctx.penalty += ctx.config.w_vastu_brahma * worst


@rule("VAS-004: the Brahmasthan stays structurally light", "soft")
def brahmasthan_walls(ctx: ReviewContext) -> None:
    """`load_bearing_wall` and `pillar` are barred from the centre too. The
    engine has no columns, so the measurable proxy is wall density: a centre
    chopped up by partitions is the opposite of the open courtyard the
    mandala asks for."""
    if not ctx.vastu_active():
        return
    mandala = ctx.mandala()
    if not mandala.available or mandala.brahma is None:
        return
    if "load_bearing_wall" not in mandala.brahma_blocked_usages():
        return

    plan = ctx.plan
    x0, y0, x1, y1 = ctx.frame().brahmasthan_rect(plan.h, plan.w)
    core = plan.grid[y0:y1, x0:x1]
    inside = core != OUTSIDE
    total = max(1, int(inside.sum()))
    wall_share = float((core == WALL).sum()) / total
    ctx.breakdown["vastu_brahma_wall"] = round(wall_share, 3)
    # ~8% is the wall share of an ordinary uncrossed centre; penalise above it
    ctx.penalty += ctx.config.w_vastu_brahma_wall * max(0.0, wall_share - 0.08)


# ═════════════════════ the entrance pada ════════════════════════════════════

@rule("VAS-005: the main door sits on an auspicious pada", "soft")
def entrance_pada(ctx: ReviewContext) -> None:
    """The 32-gate system: each side of the plot divides into 8 padas, 10 of
    which are hard-blocked and 9 of which are ideal. The engine already
    decides exactly where the main door goes, so this scores the pada it
    landed on — and records the ideal padas on that side, so the report can
    say where it SHOULD have gone instead of only that it was wrong."""
    if not ctx.vastu_active() or ctx.request.floor_index > 0:
        return
    mandala = ctx.mandala()
    if not mandala.available:
        return

    door = next((op for op in ctx.plan.openings
                 if op.is_exterior and op.kind == "door"), None)
    if door is None:
        return                                  # CIR-001 owns that failure

    plan = ctx.plan
    frame = ctx.frame()
    # the door's midpoint, in grid cells
    if door.axis == "h":
        row = door.wall_lo if door.wall_lo > 0 else door.wall_hi
        col = (door.along_lo + door.along_hi) // 2
    else:
        row = (door.along_lo + door.along_hi) // 2
        col = door.wall_lo if door.wall_lo > 0 else door.wall_hi
    row = min(max(row, 0), plan.h - 1)
    col = min(max(col, 0), plan.w - 1)

    px, py = frame.pada_of(row, col, plan.h, plan.w)
    gate = mandala.gate_at(px, py)
    if gate is None:                            # not on the perimeter ring
        return

    ctx.breakdown["vastu_gate"] = gate.id
    ctx.breakdown["vastu_gate_score"] = gate.score
    ctx.penalty += ctx.config.w_vastu_gate * (1.0 - gate.score)


# ═════════════════════ marma ════════════════════════════════════════════════

@rule("VAS-006: marma diagonals stay clear", "soft")
def marma_lines(ctx: ReviewContext) -> None:
    """Six diagonals across the mandala that no wall junction or door frame
    centre may fall on. Junctions are taken as the endpoints of shared-wall
    runs — the places where partitions actually meet."""
    if not ctx.vastu_active():
        return
    mandala = ctx.mandala()
    if not mandala.available or not mandala.marma:
        return

    plan = ctx.plan
    segments = mandala.marma_segments(ctx.frame(), plan.h, plan.w)

    # door and wide-opening centres — the "door frame centre line" the rule
    # names, and the thing a draughtsman can actually shift
    points: List[Tuple[float, float]] = []
    for op in plan.openings:
        if op.kind == "window" or op.is_exterior:
            continue
        mid_along = (op.along_lo + op.along_hi) / 2.0
        mid_wall = (op.wall_lo + op.wall_hi) / 2.0
        points.append((mid_wall, mid_along) if op.axis == "v"
                      else (mid_along, mid_wall))

    # true cross junctions: a wall cell with wall neighbours on BOTH axes in
    # both directions, i.e. where two partitions actually intersect
    grid = plan.grid
    walls = grid == WALL
    inner = np.zeros_like(walls)
    ext = plan.ext_wall
    inner[ext + 1:plan.h - ext - 1, ext + 1:plan.w - ext - 1] = True
    cross = (walls & inner
             & np.roll(walls, 1, 0) & np.roll(walls, -1, 0)
             & np.roll(walls, 1, 1) & np.roll(walls, -1, 1))
    ys, xs = np.nonzero(cross)
    if len(xs):
        # thin the cluster: one representative per intersection, not per cell
        step = max(1, len(xs) // 24)
        points.extend((float(x), float(y))
                      for x, y in zip(xs[::step], ys[::step]))

    hits = 0
    for _, a, b in segments:
        for px, py in points:
            if point_segment_distance(px, py, a, b) <= _MARMA_TOLERANCE_CELLS:
                hits += 1
                break              # one line, one fault — not one per point
    ctx.breakdown["vastu_marma_hits"] = hits
    ctx.penalty += ctx.config.w_vastu_marma * hits


# ═════════════════════ staircase ════════════════════════════════════════════

@rule("VAS-007: the staircase respects its Vastu sector and turn", "soft")
def staircase(ctx: ReviewContext) -> None:
    """Two claims, both checkable: the flight belongs in the south or west,
    and it should turn clockwise. The fitted variant records its geometry, so
    the turn is read off the real flight rather than assumed."""
    if not ctx.vastu_active():
        return
    stair_ids = [rid for rid in ctx.requested_ids()
                 if ctx.rtype(rid) in STAIR]
    if not stair_ids:
        return

    penalty = 0.0
    sectors = [ctx.sector_of(rid) for rid in stair_ids]
    ctx.breakdown["vastu_stair_sector"] = sectors[0]
    # zone_rules bars N and NE for the staircase; prefers SW and S
    if any(s in ("N", "NE", "NW") for s in sectors):
        penalty += 1.0
    elif not any(s in ("S", "SW", "W") for s in sectors):
        penalty += 0.5

    want = str(ctx.mandala().structural.get("staircase_turn_direction", ""))
    flights = [f for f in getattr(ctx.plan, "stairs", [])
               if f.face_id in set(stair_ids)]
    if want and flights:
        # A dog-leg's second flight reverses; `run_axis` plus the lane order
        # in carve.stairs.tread_lines fixes the sense of the turn.
        flight = flights[0]
        turns = flight.variant.flights > 1
        ctx.breakdown["vastu_stair_turn"] = (
            want if turns else "straight (no turn)")
        if turns and want == "clockwise" and flight.run_axis == "v":
            penalty += 0.5
    ctx.penalty += ctx.config.w_vastu_stair * penalty


# ═════════════════════ engine-derived rules ═════════════════════════════════

@rule("VAS-008: mass sits south-west, openness north-east", "soft")
def mass_gradient(ctx: ReviewContext) -> None:
    """ENGINE-DERIVED, not read from data.

    Mainstream Vastu asks for weight in the south-west and lightness in the
    north-east. The engine has no storey heights, so the measurable proxy is
    structural density: wall cells per sector. A north-east denser than the
    south-west is the gradient inverted, which is the condition this
    penalises — never the absolute amount, which is a function of the
    program, not of the design.
    """
    if not ctx.vastu_active():
        return
    plan = ctx.plan
    frame = ctx.frame()

    def wall_density(sector: str) -> float:
        x0, y0, x1, y1 = frame.sector_rect(sector, plan.h, plan.w)
        block = plan.grid[y0:y1, x0:x1]
        inside = block != OUTSIDE
        n = int(inside.sum())
        return float((block == WALL).sum()) / n if n else 0.0

    sw, ne = wall_density("SW"), wall_density("NE")
    inversion = max(0.0, ne - sw)
    ctx.breakdown["vastu_mass_sw"] = round(sw, 3)
    ctx.breakdown["vastu_mass_ne"] = round(ne, 3)
    ctx.penalty += ctx.config.w_vastu_mass * inversion * 10.0


@rule("VAS-009: plot proportion within the mandala's cap", "soft")
def plot_aspect(ctx: ReviewContext) -> None:
    """`maximum_plot_aspect_ratio` from the mandala's global modifiers. A
    plot far off square cannot host a coherent 81-pada grid, so this is
    reported as a property of the SITE — the user cannot fix it by
    rearranging rooms, and the report says so."""
    if not ctx.vastu_active():
        return
    cap = ctx.mandala().structural.get("maximum_plot_aspect_ratio")
    if not cap:
        return
    w, h = ctx.request.plot_w_ft, ctx.request.plot_h_ft
    lo = max(1, min(w, h))
    ratio = max(w, h) / lo
    ctx.breakdown["vastu_plot_aspect"] = round(ratio, 2)
    ctx.penalty += ctx.config.w_vastu_aspect * max(0.0, ratio - float(cap))


@rule("VAS-010: fire and water keep their separation", "soft")
def fire_water(ctx: ReviewContext) -> None:
    """`minimum_separation_fire_water_ft` from the mandala. HYG-001 already
    forbids an OPENING between a kitchen and a wet room; this is the stricter
    proximity claim — the two must not sit within 4 feet of one another."""
    if not ctx.vastu_active():
        return
    want_ft = ctx.mandala().structural.get("minimum_separation_fire_water_ft")
    if not want_ft:
        return
    need = float(want_ft) * units.CELLS_PER_FOOT

    kitchens = [ctx.room_ids[s.name] for s in _requested(ctx)
                if s.rtype in KITCHEN]
    wets = [ctx.room_ids[s.name] for s in _requested(ctx)
            if s.rtype in WET]
    if not kitchens or not wets:
        return

    worst = None
    for k in kitchens:
        kx0, ky0, kx1, ky1 = ctx.plan.face_bbox(k)
        for b in wets:
            bx0, by0, bx1, by1 = ctx.plan.face_bbox(b)
            dx = max(0, max(kx0 - bx1, bx0 - kx1))
            dy = max(0, max(ky0 - by1, by0 - ky1))
            gap = (dx * dx + dy * dy) ** 0.5
            worst = gap if worst is None else min(worst, gap)
    if worst is None:
        return
    ctx.breakdown["vastu_fire_water_ft"] = round(cells_ft(int(worst)), 2)
    ctx.penalty += ctx.config.w_vastu_fire_water * max(
        0.0, (need - worst) / need)


@rule("VAS-011: the north-east corner is not cut", "soft")
def northeast_corner(ctx: ReviewContext) -> None:
    """ENGINE-DERIVED, not read from data.

    On an irregular plot a missing north-east corner is a serious Vastu
    defect and a north-east extension is auspicious — one of the few Vastu
    claims about SITE SHAPE rather than room placement. Only meaningful when
    the plan carries a real boundary; a rectangle cannot have a cut corner
    and is silently skipped.
    """
    if not ctx.vastu_active():
        return
    plan = ctx.plan
    if getattr(plan, "plot_polygon", None) is None:
        return
    frame = ctx.frame()
    x0, y0, x1, y1 = frame.sector_rect("NE", plan.h, plan.w)
    block = plan.grid[y0:y1, x0:x1]
    if block.size == 0:
        return
    missing = float((block == OUTSIDE).sum()) / block.size
    ctx.breakdown["vastu_ne_cut"] = round(missing, 3)
    # a light chamfer is normal; a quarter of the sector gone is a cut corner
    ctx.penalty += ctx.config.w_vastu_mass * max(0.0, missing - 0.25) * 4.0
