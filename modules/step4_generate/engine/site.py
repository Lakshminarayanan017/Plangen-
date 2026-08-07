"""
site.py — irregular plots, without teaching every rule about polygons.

A surveyed plot is a polygon; a building is not. Architects resolve this the
same way every time: apply the setbacks, find the rectangle you can build
in, and let the leftover be open space. This module does exactly that and
hands the engine a rectangular request for the CORE, so the carver, the
settler, the connector and all 33 rules keep working unchanged — an
irregular plot costs the reviewer nothing because it never sees one.

What the site DOES own is everything the plan alone cannot say: the true
boundary, how much of the plot became open space, and the offset of the core
inside the plot (which the renderer and the DXF export need to draw the
building in its real position on the land).

The masked-lattice model — the whole polygon on the grid, open space as
open-to-sky faces — lives in GridPlan.from_polygon and is used for site
drawings. This module is the engine path.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from modules.step4_generate.core import polygon as poly
from modules.step4_generate.core import units
from modules.step4_generate.engine.contracts import EngineRequest, EngineRequestError

Point = Tuple[float, float]


@dataclass(frozen=True)
class Site:
    """A plot, the building envelope inside it, and what is left over."""
    boundary: List[Point]            # feet, plot-local, CCW
    core_rect: Tuple[int, int, int, int]   # cells, in the plot's bbox frame
    origin_ft: Tuple[float, float]   # bbox min corner in the caller's frame
    setback_ft: float

    @property
    def plot_area_sqft(self) -> float:
        return abs(poly.signed_area_sqft(self.boundary))

    @property
    def core_w_ft(self) -> float:
        x0, _, x1, _ = self.core_rect
        return (x1 - x0) / units.CELLS_PER_FOOT

    @property
    def core_h_ft(self) -> float:
        _, y0, _, y1 = self.core_rect
        return (y1 - y0) / units.CELLS_PER_FOOT

    @property
    def core_area_sqft(self) -> float:
        return poly.rect_area_sqft(self.core_rect)

    @property
    def open_space_sqft(self) -> float:
        return max(0.0, self.plot_area_sqft - self.core_area_sqft)

    @property
    def coverage(self) -> float:
        """Ground coverage — the number every municipal bye-law caps."""
        return self.core_area_sqft / self.plot_area_sqft \
            if self.plot_area_sqft else 0.0

    def core_offset_ft(self) -> Tuple[float, float]:
        """Where the building's (0,0) sits in the caller's coordinates."""
        x0, y0, _, _ = self.core_rect
        return (self.origin_ft[0] + x0 / units.CELLS_PER_FOOT,
                self.origin_ft[1] + y0 / units.CELLS_PER_FOOT)

    def describe(self) -> str:
        return (f"plot {self.plot_area_sqft:.0f} sqft -> buildable core "
                f"{self.core_w_ft:.1f}' x {self.core_h_ft:.1f}' "
                f"({self.core_area_sqft:.0f} sqft, "
                f"{self.coverage:.0%} coverage); "
                f"{self.open_space_sqft:.0f} sqft open space"
                + (f" after a {self.setback_ft:g}' setback"
                   if self.setback_ft else ""))


def build_site(boundary: Sequence[Point], *, setback_ft: float = 0.0,
               ext_wall_cells: int = units.EXT_WALL_CELLS) -> Site:
    """Rasterize the plot, apply the setback, find the buildable core."""
    ring = poly.normalize(boundary)
    w, h, ox, oy = poly.bbox_cells(ring)
    inside = poly.rasterize(ring, w, h, origin=(ox, oy))
    setback_cells = int(round(setback_ft * units.CELLS_PER_FOOT))
    core = poly.largest_inscribed_rectangle(
        poly.erode(inside, ext_wall_cells + setback_cells))

    min_core = 2 * ext_wall_cells + 16
    if core is None or (core[2] - core[0]) < min_core \
            or (core[3] - core[1]) < min_core:
        raise EngineRequestError(
            f"plot ({abs(poly.signed_area_sqft(ring)):.0f} sqft) leaves no "
            f"buildable rectangle at least {units.fmt_ft_in(min_core)} on a "
            f"side after a {setback_ft:g}' setback")
    # the core is the OUTER face of the building: the exterior wall ring is
    # carved inside it by GridPlan, so it must not also be eroded here
    x0, y0, x1, y1 = core
    core = (x0 - ext_wall_cells, y0 - ext_wall_cells,
            x1 + ext_wall_cells, y1 + ext_wall_cells)
    return Site(boundary=ring, core_rect=core, origin_ft=(ox, oy),
                setback_ft=setback_ft)


def core_request(request: EngineRequest, site: Site) -> EngineRequest:
    """The rectangular request the engine actually runs.

    Plot dimensions become the CORE's, rounded down to whole feet so the
    building never spills past the envelope by a rounding error."""
    return dataclasses.replace(
        request,
        plot_w_ft=max(units.cells(1), int(site.core_w_ft)),
        plot_h_ft=max(units.cells(1), int(site.core_h_ft)),
        plot_polygon=None, setback_ft=0.0)


def site_for(request: EngineRequest) -> Optional[Site]:
    """The site a request implies, or None when the plot is a rectangle."""
    if not request.plot_polygon:
        return None
    return build_site(request.plot_polygon, setback_ft=request.setback_ft)
