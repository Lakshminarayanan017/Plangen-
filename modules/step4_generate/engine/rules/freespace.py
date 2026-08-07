"""FSP rules — free space as a first-class reviewed artifact.

The quality bar (the user's reference plan): the house breathes through
one connected free space — the dining hub flowing doorlessly into living,
kitchen, passage — and every room opens onto that free space rather than
hanging off another private room. These rules measure exactly that.
"""

from __future__ import annotations

from modules.step4_generate.core import units
from modules.step4_generate.engine.rules.base import (
    CIRCULATION, KITCHEN, PRIVATE, SERVICE, SOCIAL, STAIR, WET,
    ReviewContext, cells_ft, rule,
)


@rule("FSP-001: every room opens onto the free-space component", "hard")
def free_component_coverage(ctx: ReviewContext) -> None:
    """The free component = rooms mutually joined by doorless wide
    openings, anchored at the largest social room. Requirements:
      • all social rooms belong to it (no fragmented free space)
      • every private/service/parking/kitchen room has an opening
        directly onto it
      • wet rooms may instead open from a private room (attached bath)

    Severity is config-routed: `fsp001_hard=False` (default) records the
    violations as a heavy soft penalty — the reviewer currently detects
    more than the generator can guarantee (the harness proves it), and
    the number drives the generator milestones (M3 band assignment / M5
    transformer). It hardens once the generator closes the gap.
    """
    violations = []
    free = ctx.free_component()
    graph = ctx.opening_graph()

    if not free:
        violations.append("FSP-001 no social free space exists")
    else:
        for spec in ctx.request.rooms:
            rid = ctx.room_ids[spec.name]
            rt = spec.rtype
            if rt in SOCIAL:
                if rid not in free:
                    violations.append(
                        f"FSP-001 social room '{spec.name}' cut off from "
                        f"the free space (fragmented circulation)")
                continue
            neighbors = {n for n, _ in graph.get(rid, ())}
            if rt in WET:
                ok = bool(neighbors & free) or any(
                    ctx.rtype(n) in PRIVATE for n in neighbors)
            elif rt in (PRIVATE | SERVICE | KITCHEN | STAIR):
                ok = bool(neighbors & free)
            else:           # ots, parking etc. — no interior-hub requirement
                # parking's correctness is exterior vehicle access
                # (STR-003), not interior free-space connectivity — a
                # carport that never opens onto the living/dining hub is
                # completely normal.
                continue
            if not ok:
                violations.append(
                    f"FSP-001 room '{spec.name}' does not open onto the "
                    f"free space")

    ctx.breakdown["freespace_violations"] = len(violations)
    if ctx.config.fsp001_hard:
        ctx.hard.extend(violations)
    else:
        ctx.penalty += ctx.config.w_freespace * len(violations)


@rule("FSP-002: dedicated circulation is lean (<= 14% of floor)", "soft")
def circulation_fraction(ctx: ReviewContext) -> None:
    """Hub-through design (dining as the corridor) is the ideal — zero
    dedicated corridor is fine. Penalize only corridor WASTE above 14%."""
    total = sum(ctx.plan.face_area_cells(r) for r in ctx.requested_ids())
    circ = sum(ctx.plan.face_area_cells(ctx.room_ids[s.name])
               for s in ctx.request.rooms if s.rtype in CIRCULATION)
    fraction = circ / total if total else 0.0
    ctx.breakdown["circulation_fraction"] = round(fraction, 3)
    ctx.penalty += ctx.config.w_circ_waste * max(0.0, fraction - 0.14) * 100


@rule("FSP-003: free-space passages are walkably wide (>= 3'0\")", "soft")
def passage_widths(ctx: ReviewContext) -> None:
    """Two measurements, both on real geometry: dedicated circulation
    rooms must be ≥ 3' clear across, and every wide opening (a doorless
    span people walk through) must be ≥ 3' wide."""
    min_w = units.cells(3)
    narrow = 0
    worst_ft = 99.0
    for spec in ctx.request.rooms:
        if spec.rtype not in CIRCULATION:
            continue
        rid = ctx.room_ids[spec.name]
        if ctx.plan.face_is_rect(rid):
            w, h = ctx.plan.clear_dims(rid)
            worst_ft = min(worst_ft, cells_ft(min(w, h)))
            if min(w, h) < min_w:
                narrow += 1
    for op in ctx.plan.openings:
        if op.kind == "wide" and not op.is_exterior and op.width < min_w:
            narrow += 1
            worst_ft = min(worst_ft, cells_ft(op.width))
    ctx.breakdown["narrow_passages"] = narrow
    if worst_ft < 99.0:
        ctx.breakdown["min_passage_ft"] = round(worst_ft, 2)
    ctx.penalty += ctx.config.w_passage_width * narrow


@rule("FSP-004: hub centrality (rooms cluster around the free space)",
      "soft")
def hub_centrality(ctx: ReviewContext) -> None:
    """Mean opening-graph distance from the free component to every room.
    1.0 = every room directly off the hub (the reference-plan ideal)."""
    free = ctx.free_component()
    if not free:
        return
    graph = ctx.opening_graph()
    dist = {rid: 0 for rid in free}
    frontier = list(free)
    while frontier:
        cur = frontier.pop(0)
        for nxt, _ in graph.get(cur, ()):
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                frontier.append(nxt)
    outside = [dist.get(rid, 3) for rid in ctx.requested_ids()
               if rid not in free]
    if not outside:
        ctx.breakdown["hub_mean_dist"] = 0.0
        return
    mean_dist = sum(outside) / len(outside)
    ctx.breakdown["hub_mean_dist"] = round(mean_dist, 2)
    ctx.penalty += ctx.config.w_hub_centrality * max(0.0, mean_dist - 1.35)
