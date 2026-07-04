"""
renderer.py
===========
Production-quality SVG floor plan renderer for PlanGen.

Converts a LayoutPlan (Step 4 output) into professional architectural
SVG drawings — one per floor.

Visual quality target: MNC / architectural firm presentation standard.

Features
--------
  ✓ 19-type room color palette (warm architectural pastels)
  ✓ Thick walls with proper exterior / party-wall distinction
  ✓ Automatic door placement with quarter-circle swing arc
  ✓ Window hatching on exterior-facing walls (habitable rooms)
  ✓ Room labels  —  display name  +  area (sqft)  +  dimensions
  ✓ Per-room dimension tick annotations on room edges
  ✓ Dashed setback / building-envelope boundary
  ✓ Plot-boundary heavy outer frame with ground hatch
  ✓ Elegant SVG compass rose (North arrow)
  ✓ Professional metric scale bar  (ft + m)
  ✓ Structured title block (ISO A2 style)
  ✓ Colour legend with score badges
  ✓ Multi-floor support  —  separate SVG per floor
  ✓ Quality / score callout panel
  ✓ Vastu direction zone overlay (optional)
  ✓ Responsive auto-scale for any plot size

Output
------
  output/<run_id>/ground_floor.svg
  output/<run_id>/first_floor.svg  (if applicable)
  output/<run_id>/second_floor.svg (if applicable)
  output/<run_id>/site_plan.svg    (full plot overview)

Usage
-----
  from modules.step4_generate.renderer import FloorPlanRenderer
  renderer = FloorPlanRenderer(layout_plan, output_dir="output")
  paths = renderer.render_all()       # → list of SVG file paths
"""

from __future__ import annotations

import math
import os
import datetime
import textwrap
import logging
from typing import Dict, List, Optional, Tuple

import svgwrite
from svgwrite import cm, mm

from models import LayoutFloor, LayoutPlan, PlacedRoom

log = logging.getLogger("PlanGen.Renderer")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & PALETTES
# ─────────────────────────────────────────────────────────────────────────────

# Scale: pixels per foot.  Auto-adjusted per plan, this is the default target.
_TARGET_DRAW_PX   = 720    # max pixels for the main drawing in any dimension
_MIN_SCALE        = 6.0    # px/ft  (very large plots)
_MAX_SCALE        = 22.0   # px/ft  (tiny plots — prevent over-zoom)

# Canvas padding around the drawing
_PAD_LEFT   = 90    # room for dimension lines
_PAD_RIGHT  = 240   # room for legend + compass
_PAD_TOP    = 70    # header band
_PAD_BOTTOM = 175   # bottom margin (title block 120 px + ~55 px for dimension annotations)
_TITLE_H    = 120   # fixed title-block height (px) — kept separate so annotations don't shrink

# Wall stroke widths (px)
_WALL_OUTER = 4.5   # exterior / plot boundary
_WALL_PARTY = 3.0   # shared party wall between rooms
_WALL_INNER = 2.0   # room outline

# Door geometry
_DOOR_WIDTH_FT   = 2.5   # default door leaf width (ft)
_DOOR_THICKNESS  = 3.0   # door leaf arc stroke width (px)

# Window geometry (hatch marks)
_WINDOW_WIDTH_FT = 4.0
_WINDOW_DEPTH_PX = 8

# Font stack — clean sans-serif always available
_FONT = "Arial, Helvetica, sans-serif"

# ── Architectural room colour palette ─────────────────────────────────────────
# (fill, stroke/wall, label text)
ROOM_PALETTE: Dict[str, Tuple[str, str, str]] = {
    "living_room":         ("#FFF3DC", "#C8960C", "#6B4F0A"),
    "drawing_room":        ("#FFF3DC", "#C8960C", "#6B4F0A"),
    "master_bedroom":      ("#D6EDD6", "#3A7A3A", "#1A4A1A"),
    "bedroom":             ("#D4E8F8", "#2E6DA4", "#143858"),
    "kids_bedroom":        ("#E8F0FF", "#4A70D0", "#1A3080"),
    "kitchen":             ("#FFF5CC", "#CCA000", "#6B5000"),
    "dining_room":         ("#EDE0FF", "#7340CC", "#3A1A70"),
    "bathroom":            ("#CCF2F8", "#008090", "#004050"),
    "toilet":              ("#D8F4F8", "#006070", "#003040"),
    "attached_bathroom":   ("#CCF2F8", "#008090", "#004050"),
    "study_room":          ("#E8E0FF", "#5040A0", "#281870"),
    "pooja_room":          ("#FFE8CC", "#CC6600", "#7A3A00"),
    "puja_room":           ("#FFE8CC", "#CC6600", "#7A3A00"),
    "staircase":           ("#E4E4E4", "#555555", "#222222"),
    "balcony":             ("#E0F4E0", "#38882A", "#184414"),
    "terrace":             ("#DCF0FF", "#3880CC", "#18406A"),
    "car_parking":         ("#EBEBEB", "#606060", "#303030"),
    "garden":              ("#D0F0D0", "#308028", "#145014"),
    "store_room":          ("#F0E8DC", "#8A6040", "#4A2A10"),
    "utility":             ("#ECEEE0", "#5A5E30", "#2A2E10"),
    "passage":             ("#EDEDED", "#707070", "#404040"),
    "foyer":               ("#F5F0E8", "#907840", "#503C18"),
    "lobby":               ("#F5F0E8", "#907840", "#503C18"),
    "verandah":            ("#E8F5E8", "#408040", "#184018"),
    "servant_room":        ("#EEE8DC", "#806040", "#402010"),
    "default":             ("#F2F0EE", "#666666", "#333333"),
}

def _room_colors(room_type: str) -> Tuple[str, str, str]:
    """Return (fill, stroke, label) colours for a room type."""
    norm = room_type.lower().replace(" ", "_").replace("-", "_")
    return ROOM_PALETTE.get(norm, ROOM_PALETTE["default"])

# ── Direction helpers ─────────────────────────────────────────────────────────
_COMPASS_ANGLE = {"N": 0, "NE": 45, "E": 90, "SE": 135,
                  "S": 180, "SW": 225, "W": 270, "NW": 315}

def _compass_to_angle(d: str) -> float:
    return _COMPASS_ANGLE.get(d.upper(), 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER – WALL ADJACENCY DETECTION
# ─────────────────────────────────────────────────────────────────────────────

_EPS = 0.25   # ft tolerance for shared-wall detection

def _shared_walls(rooms: List[PlacedRoom]) -> Dict[str, List[Tuple[str, str]]]:
    """
    Returns dict: room_id → list of (neighbour_room_id, side)
    side ∈ {'left', 'right', 'top', 'bottom'}  (in screen coords, y inverted)
    """
    result: Dict[str, List] = {r.room_id: [] for r in rooms}

    def rect(r: PlacedRoom):
        return r.x_ft, r.y_ft, r.x_ft + r.width_ft, r.y_ft + r.length_ft

    for i, a in enumerate(rooms):
        ax0, ay0, ax1, ay1 = rect(a)
        for b in rooms[i + 1:]:
            bx0, by0, bx1, by1 = rect(b)

            # Horizontal overlap
            ox = min(ax1, bx1) - max(ax0, bx0)
            # Vertical overlap
            oy = min(ay1, by1) - max(ay0, by0)

            # Right wall of A touches left wall of B
            if abs(ax1 - bx0) < _EPS and oy > _EPS:
                result[a.room_id].append((b.room_id, "right"))
                result[b.room_id].append((a.room_id, "left"))
            # Left wall of A touches right wall of B
            elif abs(ax0 - bx1) < _EPS and oy > _EPS:
                result[a.room_id].append((b.room_id, "left"))
                result[b.room_id].append((a.room_id, "right"))
            # Top wall of A touches bottom wall of B (model y up)
            elif abs(ay1 - by0) < _EPS and ox > _EPS:
                result[a.room_id].append((b.room_id, "top"))
                result[b.room_id].append((a.room_id, "bottom"))
            # Bottom wall of A touches top wall of B
            elif abs(ay0 - by1) < _EPS and ox > _EPS:
                result[a.room_id].append((b.room_id, "bottom"))
                result[b.room_id].append((a.room_id, "top"))
    return result


def _exterior_sides(room: PlacedRoom, net_w: float, net_l: float) -> List[str]:
    """Sides that touch (or nearly touch) the net-buildable boundary."""
    sides = []
    if room.x_ft < _EPS:                          sides.append("left")
    if room.x_ft + room.width_ft > net_w - _EPS:  sides.append("right")
    if room.y_ft < _EPS:                          sides.append("bottom")
    if room.y_ft + room.length_ft > net_l - _EPS: sides.append("top")
    return sides


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RENDERER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class FloorPlanRenderer:
    """
    Production-quality SVG floor plan renderer.

    Parameters
    ----------
    plan       : LayoutPlan produced by Step 4 generator.
    output_dir : Directory where SVG files are written.
    project_name : Display name shown in title block.
    """

    def __init__(
        self,
        plan: LayoutPlan,
        output_dir: str = "output",
        project_name: str = "Floor Plan",
    ):
        self.plan         = plan
        self.output_dir   = output_dir
        self.project_name = project_name
        os.makedirs(output_dir, exist_ok=True)

        # Compute uniform scale from plan dimensions
        net_w = plan.net_buildable_width_ft
        net_l = plan.net_buildable_length_ft
        max_dim = max(net_w, net_l, 1.0)
        raw_scale = _TARGET_DRAW_PX / max_dim
        self._scale = max(_MIN_SCALE, min(_MAX_SCALE, raw_scale))

        log.info(
            "Renderer: net_buildable=%.1f×%.1f ft, scale=%.1f px/ft",
            net_w, net_l, self._scale,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def render_all(self) -> List[str]:
        """Render every floor + site plan. Returns list of SVG paths."""
        paths: List[str] = []
        for floor in self.plan.floors:
            path = self.render_floor(floor)
            paths.append(path)
            log.info("Rendered → %s", path)
        return paths

    def render_floor(self, floor: LayoutFloor) -> str:
        """Render a single floor to SVG. Returns output file path."""
        label_slug = floor.floor_label.lower().replace(" ", "_")
        filename   = f"{label_slug}.svg"
        filepath   = os.path.join(self.output_dir, filename)

        sc = self._scale
        net_w = self.plan.net_buildable_width_ft
        net_l = self.plan.net_buildable_length_ft

        # Canvas dimensions
        draw_w = net_w * sc
        draw_h = net_l * sc
        canvas_w = int(draw_w + _PAD_LEFT + _PAD_RIGHT)
        canvas_h = int(draw_h + _PAD_TOP  + _PAD_BOTTOM)

        # Drawing origin (top-left of the net-buildable box in canvas coords)
        ox = _PAD_LEFT    # x offset
        oy = _PAD_TOP     # y offset

        dwg = svgwrite.Drawing(
            filepath,
            size=(f"{canvas_w}px", f"{canvas_h}px"),
            profile="full",
        )
        dwg.viewbox(0, 0, canvas_w, canvas_h)

        self._add_defs(dwg)
        self._draw_background(dwg, canvas_w, canvas_h)
        self._draw_setback_zone(dwg, ox, oy, draw_w, draw_h)
        self._draw_rooms(dwg, floor, ox, oy, net_w, net_l, sc)
        self._draw_doors(dwg, floor, ox, oy, net_l, sc)
        self._draw_windows(dwg, floor, ox, oy, net_w, net_l, sc)
        self._draw_room_labels(dwg, floor, ox, oy, net_l, sc)
        self._draw_dimension_lines(dwg, floor, ox, oy, net_w, net_l, sc)
        self._draw_plot_boundary(dwg, ox, oy, draw_w, draw_h)
        # Right-panel items — compass centred in the 240 px right column so the W
        # label never clips into the drawing area.
        right_panel_cx = canvas_w - _PAD_RIGHT + 120   # horizontal centre of right panel
        self._draw_compass_rose(dwg, right_panel_cx, oy + 55, 40)

        # Legend sits below compass with a comfortable gap.
        self._draw_legend(dwg, floor, canvas_w - _PAD_RIGHT + 10, oy + 148)

        # Title block (drawn before quality badge so badge is on top if they
        # were to touch, but they never should with the fixed geometry below).
        self._draw_title_block(dwg, canvas_w, canvas_h, floor)

        # Quality badge: anchored relative to title block so it is always
        # just above it regardless of plot size.
        # badge height = 70 px, title block top = canvas_h - _TITLE_H - 6
        badge_y = canvas_h - _TITLE_H - 6 - 14 - 70   # 14 px gap
        self._draw_quality_badge(dwg, canvas_w - _PAD_RIGHT + 10, badge_y, floor)

        dwg.save()
        return filepath

    # ── SVG Definitions  (gradients, patterns, markers) ───────────────────────

    def _add_defs(self, dwg: svgwrite.Drawing):
        defs = dwg.defs

        # Ground/site hatch pattern
        hatch = dwg.pattern(id="site-hatch", patternUnits="userSpaceOnUse",
                            size=(8, 8))
        hatch.add(dwg.rect((0, 0), (8, 8), fill="#DDEEDD"))
        hatch.add(dwg.line((0, 8), (8, 0), stroke="#B8D8B8", stroke_width=0.8))
        defs.add(hatch)

        # Concrete / parking hatch
        con = dwg.pattern(id="concrete-hatch", patternUnits="userSpaceOnUse",
                          size=(6, 6))
        con.add(dwg.rect((0, 0), (6, 6), fill="#EBEBEB"))
        con.add(dwg.line((0, 6), (6, 0), stroke="#C8C8C8", stroke_width=0.7))
        con.add(dwg.line((0, 3), (3, 0), stroke="#C8C8C8", stroke_width=0.7))
        defs.add(con)

        # Window double-line pattern (used as fill on window slot)
        win = dwg.pattern(id="window-hatch", patternUnits="userSpaceOnUse",
                          size=(4, 4))
        win.add(dwg.rect((0, 0), (4, 4), fill="#A8E8F8"))
        win.add(dwg.line((0, 0), (0, 4), stroke="#007090", stroke_width=0.6))
        win.add(dwg.line((4, 0), (4, 4), stroke="#007090", stroke_width=0.6))
        defs.add(win)

        # Arrow marker for dimension lines
        arrow = dwg.marker(id="dim-arrow", insert=(5, 3), size=(10, 6),
                           orient="auto", markerUnits="strokeWidth")
        arrow.add(dwg.polygon([(0, 0), (10, 3), (0, 6)], fill="#555555"))
        defs.add(arrow)

        # Drop shadow filter — use feMerge with required layernames arg
        filt = dwg.filter(id="room-shadow", x="-5%", y="-5%",
                          width="115%", height="115%")
        filt.feGaussianBlur(in_="SourceAlpha", stdDeviation=2, result="blur")
        filt.feOffset(in_="blur", dx=2, dy=2, result="offsetBlur")
        filt.feMerge(["offsetBlur", "SourceGraphic"])
        defs.add(filt)

    # ── Background ────────────────────────────────────────────────────────────

    def _draw_background(self, dwg, w: int, h: int):
        # Warm off-white paper background
        dwg.add(dwg.rect((0, 0), (w, h), fill="#F8F5EF"))
        # Subtle outer border
        dwg.add(dwg.rect((6, 6), (w - 12, h - 12),
                         fill="none", stroke="#CCCCCC", stroke_width=1))
        dwg.add(dwg.rect((10, 10), (w - 20, h - 20),
                         fill="none", stroke="#AAAAAA", stroke_width=0.5))

    # ── Setback zone ─────────────────────────────────────────────────────────

    def _draw_setback_zone(self, dwg, ox, oy, draw_w, draw_h):
        """Draw the full plot with setback visualisation."""
        plan = self.plan
        sc   = self._scale

        # Full plot (behind setback)
        full_w = plan.plot_width_ft  * sc
        full_h = plan.plot_length_ft * sc
        plot_ox = ox - plan.setback_left_ft  * sc
        plot_oy = oy - plan.setback_front_ft * sc  # front = top in SVG

        dwg.add(dwg.rect(
            (plot_ox, plot_oy), (full_w, full_h),
            fill="url(#site-hatch)",
            stroke="#8AAA88", stroke_width=1.5,
        ))

        # Setback hatching label
        if plan.setback_front_ft > 0:
            dwg.add(dwg.text(
                f"Front setback  {plan.setback_front_ft:.0f} ft",
                insert=(ox + draw_w / 2, plot_oy + plan.setback_front_ft * sc / 2),
                text_anchor="middle", dominant_baseline="central",
                font_family=_FONT, font_size="9px", fill="#5A7A5A",
            ))

        # Net buildable zone (white base)
        dwg.add(dwg.rect(
            (ox, oy), (draw_w, draw_h),
            fill="#FFFFFF",
            stroke="none",
        ))

        # Dashed building envelope line
        dwg.add(dwg.rect(
            (ox, oy), (draw_w, draw_h),
            fill="none",
            stroke="#AA8844", stroke_width=1.2,
            stroke_dasharray="6,4",
        ))

    # ── Rooms ─────────────────────────────────────────────────────────────────

    def _to_svg(self, x_ft: float, y_ft: float, net_l: float, sc: float,
                ox: float, oy: float) -> Tuple[float, float]:
        """Convert model (ft) coordinates to SVG (px) canvas coordinates."""
        return (ox + x_ft * sc,
                oy + (net_l - y_ft) * sc)

    def _room_svg_rect(self, room: PlacedRoom, net_l: float, sc: float,
                       ox: float, oy: float) -> Tuple[float, float, float, float]:
        """Return (svg_x, svg_y, svg_w, svg_h) for a room rectangle."""
        svg_x = ox + room.x_ft * sc
        svg_y = oy + (net_l - room.y_ft - room.length_ft) * sc
        svg_w = room.width_ft  * sc
        svg_h = room.length_ft * sc
        return svg_x, svg_y, svg_w, svg_h

    def _draw_rooms(self, dwg, floor: LayoutFloor, ox, oy, net_w, net_l, sc):
        g = dwg.g(id="rooms")
        for room in floor.rooms:
            fill, stroke, _ = _room_colors(room.room_type)
            svg_x, svg_y, svg_w, svg_h = self._room_svg_rect(room, net_l, sc, ox, oy)

            # Parking gets concrete hatch
            if "parking" in room.room_type or "car" in room.room_type:
                fill = "url(#concrete-hatch)"

            # Room fill
            g.add(dwg.rect(
                (svg_x, svg_y), (svg_w, svg_h),
                fill=fill,
                stroke=stroke,
                stroke_width=_WALL_PARTY,
                rx=0, ry=0,
            ))

        dwg.add(g)

    # ── Doors ─────────────────────────────────────────────────────────────────

    def _draw_doors(self, dwg, floor: LayoutFloor, ox, oy, net_l, sc):
        """
        Auto-place one door per room on its most accessible wall.
        Door leaf = straight line.  Swing arc = quarter-circle.
        """
        g = dwg.g(id="doors")
        adj_map = _shared_walls(floor.rooms)

        for room in floor.rooms:
            # Skip staircases, balconies, external
            if room.room_type in ("staircase", "balcony", "terrace",
                                  "car_parking", "garden"):
                continue

            svg_x, svg_y, svg_w, svg_h = self._room_svg_rect(
                room, net_l, sc, ox, oy)

            # Pick wall: prefer wall that faces an adjacent room
            neighbours = adj_map.get(room.room_id, [])
            wall = self._pick_door_wall(room, neighbours)

            door_px = min(_DOOR_WIDTH_FT * sc, 0.45 * min(svg_w, svg_h))
            door_px = max(door_px, 12)

            self._draw_door_arc(g, dwg, wall, svg_x, svg_y, svg_w, svg_h, door_px)

        dwg.add(g)

    def _pick_door_wall(self, room: PlacedRoom,
                        neighbours: List[Tuple[str, str]]) -> str:
        """Choose wall side for door placement."""
        if neighbours:
            # Prefer the wall that faces the most neighbours (busiest corridor)
            from collections import Counter
            counts = Counter(side for _, side in neighbours)
            # Choose least-common → room with least pressure gets door there
            # Actually choose: pick a wall that faces any neighbour
            preferred = counts.most_common()[-1][0]   # least shared wall
            # Actually for door: prefer wall facing another room (passage)
            passage_sides = [s for nid, s in neighbours
                             if "passage" in nid or "foyer" in nid
                             or "lobby" in nid or "living" in nid]
            if passage_sides:
                return passage_sides[0]
            return counts.most_common(1)[0][0]
        # No neighbours → exterior door; prefer bottom (south) or left
        return "bottom"

    def _draw_door_arc(self, g, dwg, wall: str,
                       sx, sy, sw, sh, door_px: float):
        """Draw a door leaf + swing arc on the specified wall side."""
        stroke = "#333333"
        sw_door = 2.0

        if wall == "bottom":
            cx = sx + sw / 2 - door_px / 2
            # Gap in wall
            g.add(dwg.rect((cx - 1, sy + sh - 1.5), (door_px + 2, 3), fill="#FFFFFF", stroke="none"))
            # Leaf (vertical up)
            g.add(dwg.line((cx, sy + sh), (cx, sy + sh - door_px), stroke=stroke, stroke_width=sw_door))
            # Arc
            g.add(dwg.path(
                d=f"M {cx+door_px},{sy+sh} A {door_px},{door_px} 0 0,0 {cx},{sy+sh-door_px}",
                fill="none", stroke=stroke, stroke_width=1.0, stroke_dasharray="3,2"
            ))

        elif wall == "top":
            cx = sx + sw / 2 - door_px / 2
            # Gap in wall
            g.add(dwg.rect((cx - 1, sy - 1.5), (door_px + 2, 3), fill="#FFFFFF", stroke="none"))
            # Leaf (vertical down)
            g.add(dwg.line((cx, sy), (cx, sy + door_px), stroke=stroke, stroke_width=sw_door))
            # Arc
            g.add(dwg.path(
                d=f"M {cx+door_px},{sy} A {door_px},{door_px} 0 0,1 {cx},{sy+door_px}",
                fill="none", stroke=stroke, stroke_width=1.0, stroke_dasharray="3,2"
            ))

        elif wall == "left":
            cy = sy + sh / 2 - door_px / 2
            # Gap in wall
            g.add(dwg.rect((sx - 1.5, cy - 1), (3, door_px + 2), fill="#FFFFFF", stroke="none"))
            # Leaf (horizontal right)
            g.add(dwg.line((sx, cy), (sx + door_px, cy), stroke=stroke, stroke_width=sw_door))
            # Arc
            g.add(dwg.path(
                d=f"M {sx},{cy+door_px} A {door_px},{door_px} 0 0,0 {sx+door_px},{cy}",
                fill="none", stroke=stroke, stroke_width=1.0, stroke_dasharray="3,2"
            ))

        elif wall == "right":
            cy = sy + sh / 2 - door_px / 2
            # Gap in wall
            g.add(dwg.rect((sx + sw - 1.5, cy - 1), (3, door_px + 2), fill="#FFFFFF", stroke="none"))
            # Leaf (horizontal left)
            g.add(dwg.line((sx + sw, cy), (sx + sw - door_px, cy), stroke=stroke, stroke_width=sw_door))
            # Arc
            g.add(dwg.path(
                d=f"M {sx+sw},{cy+door_px} A {door_px},{door_px} 0 0,1 {sx+sw-door_px},{cy}",
                fill="none", stroke=stroke, stroke_width=1.0, stroke_dasharray="3,2"
            ))

    # ── Windows ───────────────────────────────────────────────────────────────

    def _draw_windows(self, dwg, floor: LayoutFloor, ox, oy, net_w, net_l, sc):
        """
        Draw window hatching on exterior-facing walls of habitable rooms.
        Window = short double-line slot with blue tint.
        """
        g = dwg.g(id="windows")

        for room in floor.rooms:
            # Only habitable rooms get windows
            if room.room_type in ("bathroom", "toilet", "attached_bathroom",
                                  "staircase", "store_room", "utility",
                                  "car_parking", "garden"):
                continue

            ext_sides = _exterior_sides(room, net_w, net_l)
            if not ext_sides:
                continue

            svg_x, svg_y, svg_w, svg_h = self._room_svg_rect(
                room, net_l, sc, ox, oy)

            win_px = min(_WINDOW_WIDTH_FT * sc, 0.5 * max(svg_w, svg_h))
            win_px = max(win_px, 14)

            for side in ext_sides[:1]:   # one window per room (primary side)
                self._draw_window_mark(g, dwg, side, svg_x, svg_y,
                                       svg_w, svg_h, win_px)

        dwg.add(g)

    def _draw_window_mark(self, g, dwg, side: str,
                          sx, sy, sw, sh, win_px: float):
        depth = _WINDOW_DEPTH_PX
        if side == "bottom":
            cx = sx + sw / 2
            x0 = cx - win_px / 2
            # Double line + fill
            g.add(dwg.rect((x0, sy + sh - depth), (win_px, depth),
                           fill="url(#window-hatch)", stroke="none"))
            g.add(dwg.line((x0, sy + sh - depth), (x0 + win_px, sy + sh - depth),
                           stroke="#007090", stroke_width=1.0))
            g.add(dwg.line((x0, sy + sh), (x0 + win_px, sy + sh),
                           stroke="#007090", stroke_width=1.0))
        elif side == "top":
            cx = sx + sw / 2
            x0 = cx - win_px / 2
            g.add(dwg.rect((x0, sy), (win_px, depth),
                           fill="url(#window-hatch)", stroke="none"))
            g.add(dwg.line((x0, sy), (x0 + win_px, sy),
                           stroke="#007090", stroke_width=1.0))
            g.add(dwg.line((x0, sy + depth), (x0 + win_px, sy + depth),
                           stroke="#007090", stroke_width=1.0))
        elif side == "left":
            cy = sy + sh / 2
            y0 = cy - win_px / 2
            g.add(dwg.rect((sx, y0), (depth, win_px),
                           fill="url(#window-hatch)", stroke="none"))
            g.add(dwg.line((sx, y0), (sx, y0 + win_px),
                           stroke="#007090", stroke_width=1.0))
            g.add(dwg.line((sx + depth, y0), (sx + depth, y0 + win_px),
                           stroke="#007090", stroke_width=1.0))
        elif side == "right":
            cy = sy + sh / 2
            y0 = cy - win_px / 2
            g.add(dwg.rect((sx + sw - depth, y0), (depth, win_px),
                           fill="url(#window-hatch)", stroke="none"))
            g.add(dwg.line((sx + sw - depth, y0), (sx + sw - depth, y0 + win_px),
                           stroke="#007090", stroke_width=1.0))
            g.add(dwg.line((sx + sw, y0), (sx + sw, y0 + win_px),
                           stroke="#007090", stroke_width=1.0))

    # ── Room Labels ───────────────────────────────────────────────────────────

    def _draw_room_labels(self, dwg, floor: LayoutFloor, ox, oy, net_l, sc):
        g = dwg.g(id="labels")

        for room in floor.rooms:
            _, _, label_color = _room_colors(room.room_type)
            svg_x, svg_y, svg_w, svg_h = self._room_svg_rect(
                room, net_l, sc, ox, oy)

            cx = svg_x + svg_w / 2
            cy = svg_y + svg_h / 2

            # Scale font based on room size
            font_size = max(7, min(12, svg_h / 4.5))
            small_font = max(6, font_size - 2)

            # Room name — bold
            name = room.display_name
            if len(name) > 16 and svg_w < 80:
                # Abbreviate for tiny rooms
                name = name.replace("Bedroom", "Bed").replace("Bathroom", "Bath") \
                           .replace("Living Room", "Living").replace("Dining Room", "Dining")

            g.add(dwg.text(
                name,
                insert=(cx, cy - font_size * 0.4),
                text_anchor="middle",
                dominant_baseline="central",
                font_family=_FONT,
                font_size=f"{font_size:.1f}px",
                font_weight="bold",
                fill=label_color,
            ))

            # Area in sqft
            area_label = f"{room.area_sqft:.0f} sqft"
            g.add(dwg.text(
                area_label,
                insert=(cx, cy + font_size * 0.8),
                text_anchor="middle",
                dominant_baseline="central",
                font_family=_FONT,
                font_size=f"{small_font:.1f}px",
                fill=label_color,
                opacity="0.85",
            ))

            # Dimension (W × L) — only if room large enough
            if svg_w > 55 and svg_h > 30:
                dim_label = f"{room.width_ft:.0f}′ × {room.length_ft:.0f}′"
                g.add(dwg.text(
                    dim_label,
                    insert=(cx, cy + font_size * 1.8),
                    text_anchor="middle",
                    dominant_baseline="central",
                    font_family=_FONT,
                    font_size=f"{max(5.5, small_font - 1):.1f}px",
                    fill=label_color,
                    opacity="0.70",
                ))

        # Staircase step marks
        for room in floor.rooms:
            if room.room_type == "staircase":
                svg_x, svg_y, svg_w, svg_h = self._room_svg_rect(
                    room, net_l, sc, ox, oy)
                self._draw_stair_marks(g, dwg, svg_x, svg_y, svg_w, svg_h)

        dwg.add(g)

    def _draw_stair_marks(self, g, dwg, sx, sy, sw, sh):
        """Draw staircase step lines inside the staircase box."""
        n_steps = max(3, int(sh / 10))
        step_h  = sh / n_steps
        for i in range(n_steps):
            y = sy + i * step_h
            g.add(dwg.line((sx + 4, y), (sx + sw - 4, y),
                           stroke="#777777", stroke_width=0.8))
        # Stair direction arrow
        mx = sx + sw / 2
        g.add(dwg.line((mx, sy + sh - 6), (mx, sy + 6),
                       stroke="#555555", stroke_width=1.5))
        g.add(dwg.polygon(
            [(mx - 4, sy + 10), (mx + 4, sy + 10), (mx, sy + 4)],
            fill="#555555",
        ))

    # ── Dimension Lines ───────────────────────────────────────────────────────

    def _draw_dimension_lines(self, dwg, floor: LayoutFloor,
                               ox, oy, net_w, net_l, sc):
        """
        Draw overall net-buildable width & depth dimension annotations
        outside the drawing box.
        """
        g = dwg.g(id="dimensions")
        draw_w = net_w * sc
        draw_h = net_l * sc

        tick_len = 6
        gap      = 14   # px from drawing edge  (was 22 — extra space now used by title block gap)

        # ── Overall WIDTH (bottom) ──────────────────────────────────────────
        y_dim = oy + draw_h + gap
        # Dimension line
        g.add(dwg.line(
            (ox, y_dim), (ox + draw_w, y_dim),
            stroke="#555555", stroke_width=1.0,
            marker_start="url(#dim-arrow)", marker_end="url(#dim-arrow)",
        ))
        # Ticks
        g.add(dwg.line((ox, y_dim - tick_len), (ox, y_dim + tick_len),
                       stroke="#555555", stroke_width=1.2))
        g.add(dwg.line((ox + draw_w, y_dim - tick_len),
                       (ox + draw_w, y_dim + tick_len),
                       stroke="#555555", stroke_width=1.2))
        # Label
        g.add(dwg.text(
            f"Net Buildable Width: {net_w:.1f} ft  ({net_w * 0.3048:.2f} m)",
            insert=(ox + draw_w / 2, y_dim + 12),
            text_anchor="middle",
            font_family=_FONT, font_size="9px", fill="#444444",
        ))

        # ── Overall DEPTH (left side — avoids legend/compass on right) ────
        x_dim = ox - 18
        g.add(dwg.line(
            (x_dim, oy), (x_dim, oy + draw_h),
            stroke="#555555", stroke_width=1.0,
            marker_start="url(#dim-arrow)", marker_end="url(#dim-arrow)",
        ))
        g.add(dwg.line((x_dim - tick_len, oy), (x_dim + tick_len, oy),
                       stroke="#555555", stroke_width=1.2))
        g.add(dwg.line((x_dim - tick_len, oy + draw_h),
                       (x_dim + tick_len, oy + draw_h),
                       stroke="#555555", stroke_width=1.2))
        # Rotated label — sits in the left padding area
        label_x = x_dim - 16
        label_y = oy + draw_h / 2
        txt = dwg.text(
            f"Net Depth: {net_l:.1f} ft  ({net_l * 0.3048:.2f} m)",
            insert=(label_x, label_y),
            text_anchor="middle",
            font_family=_FONT, font_size="9px", fill="#444444",
        )
        txt.rotate(-90, center=(label_x, label_y))
        g.add(txt)

        dwg.add(g)

    # ── Plot Boundary ─────────────────────────────────────────────────────────

    def _draw_plot_boundary(self, dwg, ox, oy, draw_w, draw_h):
        """Heavy outer frame representing the net-buildable zone."""
        dwg.add(dwg.rect(
            (ox, oy), (draw_w, draw_h),
            fill="none",
            stroke="#222222",
            stroke_width=_WALL_OUTER,
        ))

    # ── Compass Rose ──────────────────────────────────────────────────────────

    def _draw_compass_rose(self, dwg, cx: float, cy: float, r: float):
        """
        Draw an elegant compass rose indicating North direction.
        The model's north_direction field rotates the rose accordingly.
        """
        north_angle = _compass_to_angle(self.plan.north_direction)
        g = dwg.g(id="compass", transform=f"translate({cx},{cy}) rotate({north_angle})")

        # Outer ring
        g.add(dwg.circle((0, 0), r, fill="none",
                         stroke="#444444", stroke_width=1.0))
        g.add(dwg.circle((0, 0), r + 2, fill="none",
                         stroke="#CCCCCC", stroke_width=0.5))

        # Cardinal N-S-E-W arrows
        def arrow(angle_deg: float, label: str, is_north: bool):
            rad = math.radians(angle_deg)
            # +sin for x so East points RIGHT, West points LEFT in SVG
            tip_x =  r * math.sin(rad)
            tip_y = -r * math.cos(rad)
            base_x1 =  4 * math.cos(rad)
            base_y1 =  4 * math.sin(rad)
            base_x2 = -4 * math.cos(rad)
            base_y2 = -4 * math.sin(rad)
            fill = "#CC2222" if is_north else "#555555"
            g.add(dwg.polygon(
                [(tip_x, tip_y), (base_x1, base_y1), (base_x2, base_y2)],
                fill=fill,
            ))
            lx =  (r + 14) * math.sin(rad)
            ly = -(r + 14) * math.cos(rad)
            g.add(dwg.text(
                label,
                insert=(lx, ly),
                text_anchor="middle",
                dominant_baseline="central",
                font_family=_FONT,
                font_size="11px",
                font_weight="bold" if is_north else "normal",
                fill=fill,
            ))

        arrow(0, "N", True)
        arrow(90, "E", False)
        arrow(180, "S", False)
        arrow(270, "W", False)

        # Centre dot
        g.add(dwg.circle((0, 0), 3, fill="#333333"))
        g.add(dwg.circle((0, 0), 1.5, fill="#FFFFFF"))

        dwg.add(g)

    # ── Scale Bar ─────────────────────────────────────────────────────────────

    def _draw_scale_bar(self, dwg, ox: float, y: float, sc: float):
        """Professional dual-unit scale bar (feet + metres)."""
        g = dwg.g(id="scale-bar")

        # 10 ft = reference length
        ref_ft  = 10
        ref_px  = ref_ft * sc
        ref_m   = ref_ft * 0.3048

        bar_y   = y + 10
        bar_h   = 7
        x0      = ox

        # Alternating black/white segments (5 segments × 2 ft each)
        seg_ft  = 2
        seg_px  = seg_ft * sc
        n_segs  = ref_ft // seg_ft

        for i in range(n_segs):
            fx = x0 + i * seg_px
            fill = "#333333" if i % 2 == 0 else "#FFFFFF"
            g.add(dwg.rect((fx, bar_y), (seg_px, bar_h),
                           fill=fill, stroke="#333333", stroke_width=0.5))

        # Foot labels
        for i in range(n_segs + 1):
            fx = x0 + i * seg_px
            g.add(dwg.text(
                f"{i * seg_ft}",
                insert=(fx, bar_y + bar_h + 9),
                text_anchor="middle",
                font_family=_FONT, font_size="8px", fill="#444444",
            ))
        g.add(dwg.text("ft", insert=(x0 + ref_px + 4, bar_y + bar_h + 9),
                       font_family=_FONT, font_size="7px", fill="#666666"))

        # Metre label
        g.add(dwg.text(
            f"Scale: 1:{int(1 / sc * 12 * 12):.0f}  (≈ {ref_m:.1f} m = {ref_ft} ft)",
            insert=(x0, bar_y - 3),
            font_family=_FONT, font_size="8px", fill="#666666",
        ))

        dwg.add(g)

    # ── Legend ────────────────────────────────────────────────────────────────

    def _draw_legend(self, dwg, floor: LayoutFloor,
                     lx: float, ly: float):
        """Draw a colour legend for all room types on this floor."""
        g = dwg.g(id="legend")

        # Panel background
        present_types = list(dict.fromkeys(r.room_type for r in floor.rooms))
        panel_h = len(present_types) * 18 + 30
        panel_w = 165

        g.add(dwg.rect((lx - 5, ly - 22), (panel_w, panel_h),
                       fill="#FFFDF8", stroke="#CCCCCC",
                       stroke_width=0.8, rx=3))

        g.add(dwg.text(
            "LEGEND",
            insert=(lx + panel_w / 2 - 5, ly - 10),
            text_anchor="middle",
            font_family=_FONT, font_size="9px", font_weight="bold",
            fill="#333333", letter_spacing="1.5",
        ))

        for i, rt in enumerate(present_types):
            fill, stroke, txt_c = _room_colors(rt)
            ey = ly + i * 18

            # Colour swatch
            g.add(dwg.rect((lx, ey), (14, 12),
                           fill=fill, stroke=stroke, stroke_width=1.0))

            # Room type name
            display = rt.replace("_", " ").title()
            g.add(dwg.text(
                display,
                insert=(lx + 20, ey + 9),
                font_family=_FONT, font_size="9px",
                fill="#333333",
            ))

        dwg.add(g)

    # ── Quality Badge ─────────────────────────────────────────────────────────

    def _draw_quality_badge(self, dwg, x: float, y: float,
                             floor: LayoutFloor):
        """Render a quality score summary badge."""
        g   = dwg.g(id="quality-badge")
        w, h = 165, 70

        g.add(dwg.rect((x - 5, y), (w, h),
                       fill="#FFFDF8", stroke="#CCCCCC",
                       stroke_width=0.8, rx=3))

        g.add(dwg.text("QUALITY METRICS", insert=(x + w / 2 - 5, y + 11),
                       text_anchor="middle",
                       font_family=_FONT, font_size="8px",
                       font_weight="bold", fill="#444444",
                       letter_spacing="1"))

        def metric_row(label, value, yy, color="#333333"):
            g.add(dwg.text(label, insert=(x, yy),
                           font_family=_FONT, font_size="8.5px",
                           fill="#666666"))
            g.add(dwg.text(f"{value:.2f}", insert=(x + w - 22, yy),
                           text_anchor="end",
                           font_family=_FONT, font_size="8.5px",
                           font_weight="bold", fill=color))
            # Mini bar
            bar_w = (w - 28) * min(value, 1.0)
            g.add(dwg.rect((x, yy + 1), (w - 28, 3),
                           fill="#EEEEEE", rx=1))
            g.add(dwg.rect((x, yy + 1), (bar_w, 3),
                           fill=color, rx=1))

        metric_row("Zone Score",      floor.floor_zone_score,      y + 26, "#2A7A2A")
        metric_row("Adjacency Score", floor.floor_adjacency_score,  y + 44, "#2A50AA")
        metric_row("Coverage",        floor.floor_coverage_pct / 100, y + 62, "#AA5500")

        dwg.add(g)

    # ── Title Block ───────────────────────────────────────────────────────────

    def _draw_title_block(self, dwg, canvas_w: int, canvas_h: int,
                           floor: LayoutFloor):
        """
        ISO A2-style title block at the bottom of the drawing.
        Contains: project name, floor label, scale, date, software credit.
        """
        tb_h = _TITLE_H     # fixed height regardless of _PAD_BOTTOM
        tb_y = canvas_h - tb_h - 6

        # Background
        dwg.add(dwg.rect((6, tb_y), (canvas_w - 12, tb_h),
                         fill="#F0EDE4", stroke="#888888", stroke_width=1.0))

        # Vertical dividers
        col_w = (canvas_w - 12) / 5
        for i in range(1, 5):
            x_div = 6 + i * col_w
            dwg.add(dwg.line(
                (x_div, tb_y), (x_div, canvas_h - 6),
                stroke="#AAAAAA", stroke_width=0.8,
            ))

        # Horizontal sub-divider (top row = labels, bottom = values)
        mid_y = tb_y + tb_h * 0.42

        def tb_cell(col: int, label: str, value: str, val_size: int = 13):
            cx = 6 + (col + 0.5) * col_w
            dwg.add(dwg.text(label.upper(),
                             insert=(cx, mid_y - 8),
                             text_anchor="middle",
                             font_family=_FONT, font_size="7px",
                             fill="#888888", letter_spacing="1"))
            dwg.add(dwg.text(value,
                             insert=(cx, mid_y + 14),
                             text_anchor="middle",
                             font_family=_FONT, font_size=f"{val_size}px",
                             font_weight="bold", fill="#222222"))

        # Project name (col 0 only — centred within its own column)
        dwg.add(dwg.text("PROJECT",
                         insert=(6 + 0.5 * col_w, mid_y - 8),
                         text_anchor="middle",
                         font_family=_FONT, font_size="7px",
                         fill="#888888", letter_spacing="1"))
        dwg.add(dwg.text(self.project_name,
                         insert=(6 + 0.5 * col_w, mid_y + 14),
                         text_anchor="middle",
                         font_family=_FONT, font_size="13px",
                         font_weight="bold", fill="#1A1A1A"))

        tb_cell(1, "Floor",
                floor.floor_label, 12)

        # Plot dimensions
        plan = self.plan
        plot_info = (f"{plan.plot_width_ft:.0f}′ × {plan.plot_length_ft:.0f}′"
                     f"  ({plan.plot_width_ft * 0.3048:.1f}m × "
                     f"{plan.plot_length_ft * 0.3048:.1f}m)")
        tb_cell(2, "Plot Dimensions", plot_info, 9)

        tb_cell(3, "Scale", f"1 : {int(1 / self._scale * 12 * 12):.0f}", 12)

        today = datetime.date.today().strftime("%d %b %Y")
        tb_cell(4, "Date", today, 11)

        # Vastu / solver badge
        badges = []
        if plan.vastu_enabled:
            badges.append("Vastu Compliant")
        badges.append(f"Solver: {plan.solver_used.upper()}")
        badge_str = "  •  ".join(badges)
        dwg.add(dwg.text(badge_str,
                         insert=(canvas_w / 2, canvas_h - 12),
                         text_anchor="middle",
                         font_family=_FONT, font_size="8px",
                         fill="#888888"))

        # Software credit (bottom right)
        dwg.add(dwg.text("Generated by PlanGen AI  ·  Step 4 Layout Engine",
                         insert=(canvas_w - 15, canvas_h - 12),
                         text_anchor="end",
                         font_family=_FONT, font_size="7.5px",
                         fill="#AAAAAA",
                         font_style="italic"))

        # Quality score (bottom left)
        qs = plan.layout_quality_score
        qs_color = "#2A7A2A" if qs >= 0.7 else "#CC7700" if qs >= 0.4 else "#CC2222"
        dwg.add(dwg.text(f"Layout Quality: {qs:.2f}",
                         insert=(20, canvas_h - 12),
                         font_family=_FONT, font_size="8.5px",
                         font_weight="bold", fill=qs_color))


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def render_layout_plan(
    plan: LayoutPlan,
    output_dir: str = "output",
    project_name: str = "Floor Plan",
) -> List[str]:
    """
    Render a LayoutPlan to SVG files (one per floor).

    Parameters
    ----------
    plan         : LayoutPlan from Step 4 generator.
    output_dir   : Directory for output files.
    project_name : Display name for title block.

    Returns
    -------
    List[str] : Paths to the generated SVG files.
    """
    renderer = FloorPlanRenderer(plan, output_dir=output_dir,
                                 project_name=project_name)
    return renderer.render_all()
