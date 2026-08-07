"""
dxf_export.py — GridPlan -> DXF, so the output leaves the browser.

An SVG is a picture. A drawing an engineer can dimension, edit and stamp has
to arrive in CAD, on layers, at real-world scale. This writes AutoCAD R12
ASCII DXF by hand — no dependency, and R12 is the format every tool on earth
still reads (AutoCAD, BricsCAD, LibreCAD, QCAD, Revit's importer, FreeCAD).

Layers follow the usual architectural split, so the receiving office can
freeze what it does not want:

    PLOT      surveyed boundary (irregular plots only)
    WALLS     wall outlines
    DOORS     door leaf + swing arc
    WINDOWS   glazing
    STAIRS    treads and the up arrow
    TEXT      room names, dimensions, areas
    DIMS      overall plot dimensions

Coordinates are real: 1 drawing unit = 1 foot by default, or 1 mm with
`units="mm"` (what an Indian municipal submission wants). DXF's Y axis
points up and the lattice's points down, so every Y is mirrored once, here,
at the boundary — never in the geometry core.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from modules.step4_generate.core import units
from modules.step4_generate.core.grid_plan import WALL, GridPlan, Opening
from modules.step4_generate.render.svg_render import _merge_wall_rects

# unit name -> drawing units per lattice cell, and the DXF $INSUNITS code
_UNITS = {
    "ft": (1.0 / units.CELLS_PER_FOOT, 2),      # 2 = feet
    "in": (units.CELL_INCHES, 1),               # 1 = inches
    "mm": (units.CELL_INCHES * 25.4, 4),        # 4 = millimetres
    "m": (units.CELL_INCHES * 0.0254, 6),       # 6 = metres
}

LAYERS = {
    "PLOT": 3,        # green
    "WALLS": 7,       # white/black
    "DOORS": 5,       # blue
    "WINDOWS": 4,     # cyan
    "STAIRS": 8,      # grey
    "TEXT": 2,        # yellow
    "DIMS": 1,        # red
}


class _Writer:
    """Accumulates DXF group-code pairs. Every DXF line is a (code, value)
    pair on two physical lines — keeping that in one place is what stops a
    hand-written exporter from producing a file no reader accepts."""

    def __init__(self) -> None:
        self.out: List[str] = []

    def pair(self, code: int, value) -> None:
        self.out.append(str(code))
        self.out.append(f"{value:.6f}" if isinstance(value, float)
                        else str(value))

    def entity(self, name: str, layer: str, *pairs: Tuple[int, object]
               ) -> None:
        self.pair(0, name)
        self.pair(8, layer)
        for code, value in pairs:
            self.pair(code, value)

    def text(self) -> str:
        return "\n".join(self.out) + "\n"


class DxfExporter:
    """Turns one plan (optionally with its site) into a DXF drawing."""

    def __init__(self, plan: GridPlan, *, unit: str = "ft",
                 site=None, origin_ft: Tuple[float, float] = (0.0, 0.0),
                 text_height_ft: float = 0.75):
        if unit not in _UNITS:
            raise ValueError(f"unit {unit!r} not in {sorted(_UNITS)}")
        self.plan = plan
        self.unit = unit
        self.scale, self.insunits = _UNITS[unit]
        self.site = site
        self.origin_ft = origin_ft
        self.text_height = text_height_ft * units.CELLS_PER_FOOT * self.scale

    # ── coordinate transform (the ONE place Y is mirrored) ───────────────
    def _xy(self, x_cells: float, y_cells: float) -> Tuple[float, float]:
        ox, oy = self.origin_ft
        x = (x_cells / units.CELLS_PER_FOOT + ox) * units.CELLS_PER_FOOT
        y = (y_cells / units.CELLS_PER_FOOT + oy) * units.CELLS_PER_FOOT
        return (x * self.scale, (self._top() - y) * self.scale)

    def _top(self) -> float:
        """Cell-space Y of the drawing's top edge, in plot coordinates."""
        if self.site is not None:
            _, _, _, y1 = _site_bbox_cells(self.site)
            return float(y1)
        oy = self.origin_ft[1] * units.CELLS_PER_FOOT
        return oy + self.plan.h

    def _ft(self, x_ft: float, y_ft: float) -> Tuple[float, float]:
        return self._xy(x_ft * units.CELLS_PER_FOOT
                        - self.origin_ft[0] * units.CELLS_PER_FOOT,
                        y_ft * units.CELLS_PER_FOOT
                        - self.origin_ft[1] * units.CELLS_PER_FOOT)

    # ── entities ─────────────────────────────────────────────────────────
    def _line(self, w: _Writer, layer: str, a: Tuple[float, float],
              b: Tuple[float, float]) -> None:
        w.entity("LINE", layer, (10, a[0]), (20, a[1]), (30, 0.0),
                 (11, b[0]), (21, b[1]), (31, 0.0))

    def _rect(self, w: _Writer, layer: str, x0, y0, x1, y1) -> None:
        p = [self._xy(x0, y0), self._xy(x1, y0),
             self._xy(x1, y1), self._xy(x0, y1)]
        for i in range(4):
            self._line(w, layer, p[i], p[(i + 1) % 4])

    def _text(self, w: _Writer, layer: str, x, y, value: str,
              height: Optional[float] = None, center: bool = True) -> None:
        px, py = self._xy(x, y)
        h = height if height is not None else self.text_height
        w.entity("TEXT", layer, (10, px), (20, py), (30, 0.0), (40, h),
                 (1, _dxf_safe(value)),
                 (72, 1 if center else 0), (73, 2),
                 (11, px), (21, py), (31, 0.0))

    def _arc(self, w: _Writer, layer: str, cx, cy, radius_cells,
             start_deg: float, end_deg: float) -> None:
        px, py = self._xy(cx, cy)
        w.entity("ARC", layer, (10, px), (20, py), (30, 0.0),
                 (40, radius_cells * self.scale),
                 (50, start_deg), (51, end_deg))

    # ── document ─────────────────────────────────────────────────────────
    def build(self) -> str:
        w = _Writer()
        self._header(w)
        self._tables(w)

        w.pair(0, "SECTION")
        w.pair(2, "ENTITIES")
        self._plot_boundary(w)
        self._walls(w)
        self._openings(w)
        self._stairs(w)
        self._labels(w)
        w.pair(0, "ENDSEC")
        w.pair(0, "EOF")
        return w.text()

    def _header(self, w: _Writer) -> None:
        w.pair(0, "SECTION")
        w.pair(2, "HEADER")
        w.pair(9, "$ACADVER")
        w.pair(1, "AC1009")            # R12
        w.pair(9, "$INSUNITS")
        w.pair(70, self.insunits)
        w.pair(0, "ENDSEC")

    def _tables(self, w: _Writer) -> None:
        w.pair(0, "SECTION")
        w.pair(2, "TABLES")
        w.pair(0, "TABLE")
        w.pair(2, "LAYER")
        w.pair(70, len(LAYERS))
        for name, color in LAYERS.items():
            w.pair(0, "LAYER")
            w.pair(2, name)
            w.pair(70, 0)
            w.pair(62, color)
            w.pair(6, "CONTINUOUS")
        w.pair(0, "ENDTAB")
        w.pair(0, "ENDSEC")

    def _plot_boundary(self, w: _Writer) -> None:
        ring = getattr(self.site, "boundary", None) \
            or getattr(self.plan, "plot_polygon", None)
        if not ring:
            return
        pts = [self._ft(x, y) for x, y in ring]
        for i in range(len(pts)):
            self._line(w, "PLOT", pts[i], pts[(i + 1) % len(pts)])

    def _walls(self, w: _Writer) -> None:
        for x0, y0, x1, y1 in _merge_wall_rects(self.plan.grid):
            self._rect(w, "WALLS", x0, y0, x1, y1)

    def _openings(self, w: _Writer) -> None:
        for op in self.plan.openings:
            layer = "WINDOWS" if op.kind == "window" else "DOORS"
            if op.axis == "v":
                x0, y0 = op.wall_lo, op.along_lo
                x1, y1 = op.wall_hi, op.along_hi
            else:
                x0, y0 = op.along_lo, op.wall_lo
                x1, y1 = op.along_hi, op.wall_hi
            # the gap itself: two lines across the wall band, so the wall
            # rectangle reads as interrupted rather than overdrawn
            self._rect(w, layer, x0, y0, x1, y1)
            if op.kind == "door":
                self._door_swing(w, op)

    def _door_swing(self, w: _Writer, op: Opening) -> None:
        radius = op.width
        if op.axis == "v":
            hinge_x = op.wall_hi if _swings_positive(op, self.plan) \
                else op.wall_lo
            hinge_y = op.along_lo
            start, end = (270.0, 0.0) if _swings_positive(op, self.plan) \
                else (180.0, 270.0)
        else:
            hinge_y = op.wall_hi if _swings_positive(op, self.plan) \
                else op.wall_lo
            hinge_x = op.along_lo
            start, end = (180.0, 270.0) if _swings_positive(op, self.plan) \
                else (90.0, 180.0)
        self._arc(w, "DOORS", hinge_x, hinge_y, radius, start, end)

    def _stairs(self, w: _Writer) -> None:
        from modules.step4_generate.carve.stairs import tread_lines
        for flight in getattr(self.plan, "stairs", []) or []:
            for lx0, ly0, lx1, ly1 in tread_lines(flight):
                self._line(w, "STAIRS", self._xy(lx0, ly0),
                           self._xy(lx1, ly1))
            x0, y0, x1, y1 = flight.rect
            self._text(w, "STAIRS", (x0 + x1) / 2, (y0 + y1) / 2,
                       f"UP {flight.variant.risers}R", self.text_height * 0.8)

    def _labels(self, w: _Writer) -> None:
        for rid, room in sorted(self.plan.rooms.items()):
            try:
                x0, y0, x1, y1 = self.plan.face_bbox(rid)
            except KeyError:
                continue
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            self._text(w, "TEXT", cx, cy - self.text_height, room.name)
            if self.plan.face_is_rect(rid):
                self._text(w, "TEXT", cx, cy + self.text_height,
                           self.plan.dims_str(rid), self.text_height * 0.8)
            self._text(w, "TEXT", cx, cy + self.text_height * 2.6,
                       f"{self.plan.area_sqft(rid):.0f} SQFT",
                       self.text_height * 0.7)

        self._text(w, "DIMS", self.plan.w / 2, -self.text_height * 2,
                   units.fmt_ft_in(self.plan.w))
        self._text(w, "DIMS", -self.text_height * 3, self.plan.h / 2,
                   units.fmt_ft_in(self.plan.h))


def _swings_positive(op: Opening, plan: GridPlan) -> bool:
    from modules.step4_generate.render.svg_render import _swing_side_positive
    return _swing_side_positive(op, plan)


def _site_bbox_cells(site) -> Tuple[int, int, int, int]:
    from modules.step4_generate.core import polygon as poly
    w, h, _, _ = poly.bbox_cells(site.boundary)
    return (0, 0, w, h)


def _dxf_safe(value: str) -> str:
    """DXF group values are one line; strip anything that would break the
    pairing. Inches are written as `in` because a literal quote inside a
    TEXT value confuses several older readers."""
    return (str(value).replace("\n", " ").replace("\r", " ")
            .replace('"', "in").strip()) or "-"


def export_dxf(plan: GridPlan, path: str, *, unit: str = "ft",
               site=None, origin_ft: Tuple[float, float] = (0.0, 0.0)
               ) -> str:
    """Write `plan` to `path` as R12 DXF. Returns the path."""
    text = DxfExporter(plan, unit=unit, site=site,
                       origin_ft=origin_ft).build()
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="ascii", errors="replace",
              newline="\r\n") as f:
        f.write(text)
    return path


def dxf_string(plan: GridPlan, **kw) -> str:
    """The DXF text, for callers that stream it (the web app's download)."""
    return DxfExporter(plan, **kw).build()
