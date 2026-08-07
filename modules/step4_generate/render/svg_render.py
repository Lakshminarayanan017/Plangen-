"""
svg_render.py — GridPlan → SVG drawing.

Visual feedback is a first-class part of the pipeline: every stage of the
carver is judged by looking at its output, so the renderer exists from day
one. Draws: room fills, merged wall rectangles, opening gaps, door leaves
with swing arcs, room labels (name / clear dims / area), plot dimensions.
"""

from __future__ import annotations

from typing import List
from xml.sax.saxutils import escape

import numpy as np

from modules.step4_generate.core import units
from modules.step4_generate.core.grid_plan import WALL, GridPlan, Opening

# Soft archi-print palette keyed by room type.
PALETTE = {
    "living_room":    "#FDEBD0",
    "drawing_room":   "#FDEBD0",
    "dining_room":    "#FCF3CF",
    "kitchen":        "#FADBD8",
    "bedroom":        "#D6EAF8",
    "master_bedroom": "#D4E6F1",
    "bathroom":       "#D1F2EB",
    "toilet":         "#D1F2EB",
    "passage":        "#F4F6F6",
    "hallway":        "#F4F6F6",
    "store":          "#E8DAEF",
    "storage":        "#E8DAEF",
    "utility":        "#E8DAEF",
    "parking":        "#E5E7E9",
    "staircase":      "#EAECEE",
    "ots":            "#FEF9E7",
    "undefined":      "#F8F9F9",
}
WALL_COLOR = "#1C1C1C"
GAP_COLOR = "#FFFFFF"
WINDOW_COLOR = "#CFE4F5"      # glazing
TEXT_MAIN = "#2C3E50"
TEXT_META = "#6B7280"


def _merge_wall_rects(grid: np.ndarray) -> List[tuple]:
    """Merge WALL cells into as few rectangles as possible (row runs, then
    vertical merge of identical runs) — keeps the SVG small and crisp."""
    h, w = grid.shape
    open_rects = {}   # (x0, x1) -> [y_start, y_end)
    done: List[tuple] = []
    for y in range(h + 1):
        row_runs = set()
        if y < h:
            line = grid[y, :]
            change = np.flatnonzero(np.diff(line)) + 1
            bounds = [0, *change.tolist(), w]
            for i in range(len(bounds) - 1):
                if line[bounds[i]] == WALL:
                    row_runs.add((bounds[i], bounds[i + 1]))
        for key in list(open_rects):
            if key not in row_runs:
                y0, y1 = open_rects.pop(key)
                done.append((key[0], y0, key[1], y1))
        for key in row_runs:
            if key in open_rects:
                open_rects[key][1] = y + 1
            else:
                open_rects[key] = [y, y + 1]
    return done


def _swing_side_positive(op: Opening, plan: GridPlan) -> bool:
    """True if the door leaf swings toward the +axis side of the wall
    (right of a "v" wall, below an "h" wall). Decided by checking which
    side of the wall band the swing-target room actually occupies."""
    target = op.room_a if op.swing == "a" else op.room_b
    mid = (op.along_lo + op.along_hi) // 2
    if op.axis == "v":
        if op.wall_hi >= plan.w:
            return False                      # wall at east edge → swing west
        return int(plan.grid[mid, op.wall_hi]) == target
    if op.wall_hi >= plan.h:
        return False                          # wall at south edge → swing north
    return int(plan.grid[op.wall_hi, mid]) == target


def _door_leaf(op: Opening, plan: GridPlan, s: float, m: float) -> str:
    """Door leaf + quarter-circle swing arc into the room named by
    `op.swing` (works for interior walls and exterior doors on any side)."""
    r = op.width * s
    positive = _swing_side_positive(op, plan)
    if op.axis == "v":
        wall_face = op.wall_hi if positive else op.wall_lo
        hx, hy = m + wall_face * s, m + op.along_lo * s
        leaf_end = (hx + (r if positive else -r), hy)
        arc_end = (hx, hy + r)
    else:
        wall_face = op.wall_hi if positive else op.wall_lo
        hx, hy = m + op.along_lo * s, m + wall_face * s
        leaf_end = (hx, hy + (r if positive else -r))
        arc_end = (hx + r, hy)
    cross = ((leaf_end[0] - hx) * (arc_end[1] - hy)
             - (leaf_end[1] - hy) * (arc_end[0] - hx))
    sweep = 1 if cross > 0 else 0
    return (
        f'<path d="M {hx:.1f} {hy:.1f} L {leaf_end[0]:.1f} {leaf_end[1]:.1f} '
        f'A {r:.1f} {r:.1f} 0 0 {sweep} {arc_end[0]:.1f} {arc_end[1]:.1f}" '
        f'fill="none" stroke="{TEXT_META}" stroke-width="1"/>'
    )


def _stair_treads_fitted(flight, s: float, m: float) -> List[str]:
    """Treads of an actually-FITTED staircase (carve.stairs.StairFlight):
    every line is a real tread edge at the real 10.5" going, and the UP
    arrow follows the real flight direction. Drawing the true geometry is
    the point — a plan whose stair is decorative lies to whoever builds it.
    """
    from modules.step4_generate.carve.stairs import tread_lines
    out: List[str] = []
    stroke = f'stroke="{TEXT_META}" stroke-width="1"'
    for lx0, ly0, lx1, ly1 in tread_lines(flight):
        out.append(f'<line x1="{m + lx0 * s:.1f}" y1="{m + ly0 * s:.1f}" '
                   f'x2="{m + lx1 * s:.1f}" y2="{m + ly1 * s:.1f}" {stroke}/>')

    x0, y0, x1, y1 = flight.rect
    px0, py0, px1, py1 = m + x0 * s, m + y0 * s, m + x1 * s, m + y1 * s
    inset = 4.0
    lanes = len(flight.variant.treads_per_flight) \
        if flight.variant.kind == "dogleg" else 1
    for lane in range(lanes):
        if flight.run_axis == "h":            # flight climbs along y
            lane_w = (px1 - px0) / lanes
            cx = px0 + lane_w * (lane + 0.5)
            ay0, ay1 = ((py1 - inset, py0 + inset) if lane % 2 == 0
                        else (py0 + inset, py1 - inset))
            ax0 = ax1 = cx
        else:                                  # flight climbs along x
            lane_h = (py1 - py0) / lanes
            cy = py0 + lane_h * (lane + 0.5)
            ax0, ax1 = ((px1 - inset, px0 + inset) if lane % 2 == 0
                        else (px0 + inset, px1 - inset))
            ay0 = ay1 = cy
        out.append(f'<line x1="{ax0:.1f}" y1="{ay0:.1f}" x2="{ax1:.1f}" '
                   f'y2="{ay1:.1f}" stroke="{TEXT_MAIN}" stroke-width="1.4" '
                   f'marker-end="url(#stairup)" opacity="0.55"/>')
    return out


def _stair_treads(x0: int, y0: int, x1: int, y1: int,
                  s: float, m: float) -> List[str]:
    """Fallback stair hatch for a face labelled 'staircase' that no flight
    was fitted to (single-floor plans that name a stair, or a face STR-S02
    already failed). Evenly spaced lines — indicative, not dimensioned."""
    px0, py0 = m + x0 * s, m + y0 * s
    px1, py1 = m + x1 * s, m + y1 * s
    w, h = px1 - px0, py1 - py0
    out: List[str] = []
    inset = 3.0
    run_vertical = h >= w                      # flight goes top↔bottom
    length = (h if run_vertical else w) - 2 * inset
    n = max(3, int(length / 9))                # ~9px per tread
    stroke = f'stroke="{TEXT_META}" stroke-width="1" stroke-dasharray="3 2"'
    for i in range(1, n):
        t = i / n
        if run_vertical:
            yy = py0 + inset + t * (h - 2 * inset)
            out.append(f'<line x1="{px0 + inset:.1f}" y1="{yy:.1f}" '
                       f'x2="{px1 - inset:.1f}" y2="{yy:.1f}" {stroke}/>')
        else:
            xx = px0 + inset + t * (w - 2 * inset)
            out.append(f'<line x1="{xx:.1f}" y1="{py0 + inset:.1f}" '
                       f'x2="{xx:.1f}" y2="{py1 - inset:.1f}" {stroke}/>')
    return out


def render_svg(plan: GridPlan, title: str = "") -> str:
    s = 2.0                       # px per 1.5" cell → 16 px per foot
    m = 42.0                      # margin for dimension text
    W, H = plan.w * s + 2 * m, plan.h * s + 2 * m
    el: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" '
        f'height="{H + (26 if title else 0):.0f}" '
        f'viewBox="0 0 {W:.0f} {H + (26 if title else 0):.0f}">',
        f'<defs><marker id="stairup" markerWidth="6" markerHeight="6" '
        f'refX="4" refY="3" orient="auto"><path d="M0,0 L5,3 L0,6" '
        f'fill="none" stroke="{TEXT_MAIN}" stroke-width="1.2"/></marker></defs>',
        f'<rect width="100%" height="100%" fill="#FFFFFF"/>',
    ]

    # Room fills + labels
    for rid, room in sorted(plan.rooms.items()):
        fill = PALETTE.get(room.rtype, PALETTE["undefined"])
        x0, y0, x1, y1 = plan.face_bbox(rid)
        if plan.face_is_rect(rid):
            el.append(
                f'<rect x="{m + x0 * s:.1f}" y="{m + y0 * s:.1f}" '
                f'width="{(x1 - x0) * s:.1f}" height="{(y1 - y0) * s:.1f}" '
                f'fill="{fill}"/>'
            )
        else:  # generic face: row runs
            mask = plan.face_mask(rid)
            for y in range(y0, y1):
                xs = np.nonzero(mask[y])[0]
                if len(xs):
                    el.append(
                        f'<rect x="{m + xs.min() * s:.1f}" y="{m + y * s:.1f}" '
                        f'width="{(xs.max() - xs.min() + 1) * s:.1f}" '
                        f'height="{s:.1f}" fill="{fill}"/>'
                    )
        # staircase: draw the FITTED flight when one exists, else a hatch
        if room.rtype == "staircase":
            flight = next((f for f in getattr(plan, "stairs", [])
                           if f.face_id == rid), None)
            el.extend(_stair_treads_fitted(flight, s, m) if flight
                      else _stair_treads(x0, y0, x1, y1, s, m))

        # labels
        cx, cy = m + (x0 + x1) / 2 * s, m + (y0 + y1) / 2 * s
        bw, bh = (x1 - x0) * s, (y1 - y0) * s
        name = escape(room.name)
        if bw > 60 and bh > 44:
            dims = escape(plan.dims_str(rid)) if plan.face_is_rect(rid) else ""
            area = f"{plan.area_sqft(rid):.0f} sqft"
            el.append(
                f'<text x="{cx:.0f}" y="{cy - 8:.0f}" text-anchor="middle" '
                f'font-family="Segoe UI, sans-serif" font-size="12" '
                f'font-weight="600" fill="{TEXT_MAIN}">{name}</text>'
            )
            el.append(
                f'<text x="{cx:.0f}" y="{cy + 6:.0f}" text-anchor="middle" '
                f'font-family="Segoe UI, sans-serif" font-size="10" '
                f'fill="{TEXT_META}">{dims}</text>'
            )
            el.append(
                f'<text x="{cx:.0f}" y="{cy + 19:.0f}" text-anchor="middle" '
                f'font-family="Segoe UI, sans-serif" font-size="9" '
                f'fill="{TEXT_META}">{area}</text>'
            )
        else:
            el.append(
                f'<text x="{cx:.0f}" y="{cy + 3:.0f}" text-anchor="middle" '
                f'font-family="Segoe UI, sans-serif" font-size="9" '
                f'font-weight="600" fill="{TEXT_MAIN}">{name}</text>'
            )

    # Walls
    for x0, y0, x1, y1 in _merge_wall_rects(plan.grid):
        el.append(
            f'<rect x="{m + x0 * s:.1f}" y="{m + y0 * s:.1f}" '
            f'width="{(x1 - x0) * s:.1f}" height="{(y1 - y0) * s:.1f}" '
            f'fill="{WALL_COLOR}"/>'
        )

    # Openings: paint the gap, then door leaves / window glazing
    for op in plan.openings:
        if op.axis == "v":
            gx, gy = op.wall_lo, op.along_lo
            gw, gh = op.wall_hi - op.wall_lo, op.width
        else:
            gx, gy = op.along_lo, op.wall_lo
            gw, gh = op.width, op.wall_hi - op.wall_lo
        fill = WINDOW_COLOR if op.kind == "window" else GAP_COLOR
        el.append(
            f'<rect x="{m + gx * s:.1f}" y="{m + gy * s:.1f}" '
            f'width="{gw * s:.1f}" height="{gh * s:.1f}" fill="{fill}"/>'
        )
        if op.kind == "window":
            # center mullion line along the glazing (plan-symbol style)
            if op.axis == "h":
                cy_ = m + (gy + gh / 2) * s
                el.append(f'<line x1="{m + gx * s:.1f}" y1="{cy_:.1f}" '
                          f'x2="{m + (gx + gw) * s:.1f}" y2="{cy_:.1f}" '
                          f'stroke="{WALL_COLOR}" stroke-width="0.8"/>')
            else:
                cx_ = m + (gx + gw / 2) * s
                el.append(f'<line x1="{cx_:.1f}" y1="{m + gy * s:.1f}" '
                          f'x2="{cx_:.1f}" y2="{m + (gy + gh) * s:.1f}" '
                          f'stroke="{WALL_COLOR}" stroke-width="0.8"/>')
        elif op.kind == "door":
            el.append(_door_leaf(op, plan, s, m))

    # Plot dimensions
    el.append(
        f'<text x="{m + plan.w * s / 2:.0f}" y="{m - 14:.0f}" '
        f'text-anchor="middle" font-family="Segoe UI, sans-serif" '
        f'font-size="13" font-weight="600" fill="{TEXT_MAIN}">'
        f'{escape(units.fmt_ft_in(plan.w))}</text>'
    )
    el.append(
        f'<text x="{m - 14:.0f}" y="{m + plan.h * s / 2:.0f}" '
        f'text-anchor="middle" font-family="Segoe UI, sans-serif" '
        f'font-size="13" font-weight="600" fill="{TEXT_MAIN}" '
        f'transform="rotate(-90 {m - 14:.0f} {m + plan.h * s / 2:.0f})">'
        f'{escape(units.fmt_ft_in(plan.h))}</text>'
    )
    if title:
        el.append(
            f'<text x="{W / 2:.0f}" y="{H + 12:.0f}" text-anchor="middle" '
            f'font-family="Segoe UI, sans-serif" font-size="12" '
            f'fill="{TEXT_META}">{escape(title)}</text>'
        )
    el.append("</svg>")
    return "\n".join(el)


def save_svg(plan: GridPlan, path: str, title: str = "") -> str:
    svg = render_svg(plan, title)
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)
    return path
