"""
grid.py
=======
Coordinate system and zone mapping for the Step 4 layout generator.

Coordinate system (consistent across all Step 4 modules):
  • Origin (0, 0) = SW corner of the net buildable area.
  • x increases EAST  (toward the front/entrance for east-facing plots)
  • y increases NORTH
  • All values in feet.

Zone regions are derived from the entrance direction so that "front"
always maps to the road-facing side regardless of compass orientation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# ── Constants ─────────────────────────────────────────────────────────────────

# Zone thresholds (fraction of net buildable dimension)
FRONT_ZONE_THRESHOLD = 0.35   # front 35% (road side)
BACK_ZONE_THRESHOLD  = 0.65   # back 35% (rear side)
# middle = 35% – 65%

# Room placement priority for greedy algorithm
# Lower number = placed first (anchors the layout)
ROOM_PLACEMENT_PRIORITY: Dict[str, int] = {
    "staircase":      0,   # anchor — must align on all floors
    "car_parking":    1,   # always at entrance
    "foyer":          2,
    "living_room":    3,
    "drawing_room":   3,
    "dining_room":    4,
    "kitchen":        5,
    "pooja_room":     6,
    "master_bedroom": 7,
    "bedroom":        8,
    "bedroom_kids":   8,
    "bedroom_guest":  8,
    "bathroom":       9,
    "toilet":         9,
    "study_room":    10,
    "passage":       11,   # fills gaps at the end
    "utility_room":  12,
    "store_room":    13,
    "balcony":       14,
    "servant_room":  15,
    "garden":        16,
    "terrace":       16,
    "verandah":      16,
}

# External room types that live OUTSIDE the structural envelope
EXTERNAL_ROOM_TYPES = frozenset({
    "car_parking", "garden", "terrace", "verandah", "balcony", "barsati",
})


@dataclass
class ZoneRegion:
    """
    Axis-aligned rectangle within the net buildable area
    that represents a preferred placement zone.
    """
    x_min: float   # ft from SW origin
    x_max: float
    y_min: float
    y_max: float

    def contains_center(self, x: float, y: float, w: float, l: float) -> bool:
        """True if the room's CENTER is inside this zone."""
        cx = x + w / 2.0
        cy = y + l / 2.0
        return (self.x_min <= cx <= self.x_max and
                self.y_min <= cy <= self.y_max)

    def overlap_fraction(self, x: float, y: float, w: float, l: float) -> float:
        """Fraction of room area that overlaps with this zone (0–1)."""
        ix_min = max(self.x_min, x)
        ix_max = min(self.x_max, x + w)
        iy_min = max(self.y_min, y)
        iy_max = min(self.y_max, y + l)
        if ix_max <= ix_min or iy_max <= iy_min:
            return 0.0
        overlap = (ix_max - ix_min) * (iy_max - iy_min)
        total   = w * l
        return overlap / total if total > 0 else 0.0


class PlotGrid:
    """
    Translates abstract zone names and compass directions into
    concrete coordinate rectangles for a specific plot.

    All coordinates are in feet relative to the SW corner of the
    NET BUILDABLE area (i.e. after setbacks are subtracted).

    Supports entrance directions: N, S, E, W (NE/SE/SW/NW mapped to nearest).
    """

    def __init__(
        self,
        net_width_ft:  float,
        net_length_ft: float,
        entrance_dir:  str,
        north_dir:     str = "N",
    ) -> None:
        self.W   = net_width_ft
        self.L   = net_length_ft
        self.ent = self._normalise_dir(entrance_dir)
        self.nth = self._normalise_dir(north_dir)

        # Pre-compute the axis-aligned zone rectangles for this plot
        self._zones   = self._build_zones()
        self._compass = self._build_compass_zones()

    # ── Public API ─────────────────────────────────────────────────────

    def zone_region(self, zone_name: str) -> ZoneRegion:
        """
        Return the ZoneRegion for a named zone.
        zone_name: 'front' | 'middle' | 'back' | 'left' | 'right'
        Falls back to full plot if unknown.
        """
        return self._zones.get(zone_name.lower(), self._full_plot())

    def compass_region(self, compass: str) -> ZoneRegion:
        """
        Return the ZoneRegion for a compass direction.
        compass: N | NE | E | SE | S | SW | W | NW | center
        Falls back to full plot if unknown.
        """
        return self._compass.get(compass.upper(), self._full_plot())

    def preferred_region(self, zone: str, compass: str) -> ZoneRegion:
        """
        Intersect zone and compass regions to get the tightest target area.
        If the intersection is too small (< 5% of plot), return the zone region.
        """
        z = self.zone_region(zone)
        c = self.compass_region(compass)

        # Intersection
        ix_min = max(z.x_min, c.x_min)
        ix_max = min(z.x_max, c.x_max)
        iy_min = max(z.y_min, c.y_min)
        iy_max = min(z.y_max, c.y_max)

        min_area = self.W * self.L * 0.05   # 5% of total as minimum viable region
        if (ix_max - ix_min) * (iy_max - iy_min) >= min_area:
            return ZoneRegion(ix_min, ix_max, iy_min, iy_max)
        return z   # fallback to just zone

    def score_placement(
        self,
        x: float, y: float, w: float, l: float,
        zone: str, compass: str,
    ) -> float:
        """
        Score how well a placed room satisfies its zone + compass preference.
        Returns 0.0 (no match) to 1.0 (perfect match).
        """
        z_region = self.zone_region(zone)
        c_region = self.compass_region(compass)
        z_score  = z_region.overlap_fraction(x, y, w, l)
        c_score  = c_region.overlap_fraction(x, y, w, l)
        # Weighted: zone 40%, compass 60% (Vastu is compass-based)
        return round(0.4 * z_score + 0.6 * c_score, 3)

    def good_start_position(
        self,
        zone:    str,
        compass: str,
        width:   float,
        length:  float,
    ) -> Tuple[float, float]:
        """
        Return a good (x, y) starting position for placing a room of the
        given dimensions in its preferred zone/compass region.
        Clamps to valid bounds so room stays within net buildable area.
        """
        region = self.preferred_region(zone, compass)
        # Aim for the centroid of the region, adjusted so room fits
        cx = (region.x_min + region.x_max) / 2.0
        cy = (region.y_min + region.y_max) / 2.0
        x = max(0.0, min(cx - width  / 2.0, self.W - width))
        y = max(0.0, min(cy - length / 2.0, self.L - length))
        return (round(x, 2), round(y, 2))

    # ── Private: build zone + compass lookup tables ─────────────────────

    def _build_zones(self) -> Dict[str, ZoneRegion]:
        """
        Build the front/middle/back/left/right zone rectangles.
        "Front" is always the entrance side regardless of compass.
        """
        W, L = self.W, self.L
        ent = self.ent

        # Entrance direction → which axis and which end is "front"
        if ent == "E":
            # East entrance: front is high-x, back is low-x
            # x increases east → front at right side
            return {
                "front":  ZoneRegion(W * (1-FRONT_ZONE_THRESHOLD), W,   0,         L),
                "middle": ZoneRegion(W * FRONT_ZONE_THRESHOLD,      W * (1-FRONT_ZONE_THRESHOLD), 0, L),
                "back":   ZoneRegion(0,                             W * FRONT_ZONE_THRESHOLD,     0, L),
                "left":   ZoneRegion(0, W, L * (1-FRONT_ZONE_THRESHOLD), L),
                "right":  ZoneRegion(0, W, 0,                            L * FRONT_ZONE_THRESHOLD),
            }
        elif ent == "W":
            # West entrance: front is low-x
            return {
                "front":  ZoneRegion(0,                             W * FRONT_ZONE_THRESHOLD,     0, L),
                "middle": ZoneRegion(W * FRONT_ZONE_THRESHOLD,      W * (1-FRONT_ZONE_THRESHOLD), 0, L),
                "back":   ZoneRegion(W * (1-FRONT_ZONE_THRESHOLD), W,                            0, L),
                "left":   ZoneRegion(0, W, 0,                            L * FRONT_ZONE_THRESHOLD),
                "right":  ZoneRegion(0, W, L * (1-FRONT_ZONE_THRESHOLD), L),
            }
        elif ent == "S":
            # South entrance: front is low-y
            return {
                "front":  ZoneRegion(0, W, 0,                             L * FRONT_ZONE_THRESHOLD),
                "middle": ZoneRegion(0, W, L * FRONT_ZONE_THRESHOLD,      L * (1-FRONT_ZONE_THRESHOLD)),
                "back":   ZoneRegion(0, W, L * (1-FRONT_ZONE_THRESHOLD), L),
                "left":   ZoneRegion(0, W * FRONT_ZONE_THRESHOLD,     0, L),
                "right":  ZoneRegion(W * (1-FRONT_ZONE_THRESHOLD), W, 0, L),
            }
        else:
            # North entrance (default): front is high-y
            return {
                "front":  ZoneRegion(0, W, L * (1-FRONT_ZONE_THRESHOLD), L),
                "middle": ZoneRegion(0, W, L * FRONT_ZONE_THRESHOLD,      L * (1-FRONT_ZONE_THRESHOLD)),
                "back":   ZoneRegion(0, W, 0,                             L * FRONT_ZONE_THRESHOLD),
                "left":   ZoneRegion(0, W * FRONT_ZONE_THRESHOLD,     0, L),
                "right":  ZoneRegion(W * (1-FRONT_ZONE_THRESHOLD), W, 0, L),
            }

    def _build_compass_zones(self) -> Dict[str, ZoneRegion]:
        """
        Build compass-direction rectangles based on north_direction.
        North is always at y=L for north-facing, etc.
        """
        W, L = self.W, self.L
        nth = self.nth

        # Thirds along each axis
        t1x, t2x = W / 3.0, 2.0 * W / 3.0
        t1y, t2y = L / 3.0, 2.0 * L / 3.0

        if nth == "N":
            # Standard: N = top, S = bottom, E = right, W = left
            n  = ZoneRegion(0,   W,   t2y, L)
            s  = ZoneRegion(0,   W,   0,   t1y)
            e  = ZoneRegion(t2x, W,   0,   L)
            w  = ZoneRegion(0,   t1x, 0,   L)
            ne = ZoneRegion(t2x, W,   t2y, L)
            se = ZoneRegion(t2x, W,   0,   t1y)
            sw = ZoneRegion(0,   t1x, 0,   t1y)
            nw = ZoneRegion(0,   t1x, t2y, L)
        elif nth == "S":
            # Flipped: N = bottom, S = top
            n  = ZoneRegion(0,   W,   0,   t1y)
            s  = ZoneRegion(0,   W,   t2y, L)
            e  = ZoneRegion(0,   t1x, 0,   L)
            w  = ZoneRegion(t2x, W,   0,   L)
            ne = ZoneRegion(0,   t1x, 0,   t1y)
            se = ZoneRegion(0,   t1x, t2y, L)
            sw = ZoneRegion(t2x, W,   t2y, L)
            nw = ZoneRegion(t2x, W,   0,   t1y)
        elif nth == "E":
            # N = right, S = left
            n  = ZoneRegion(t2x, W,   0,   L)
            s  = ZoneRegion(0,   t1x, 0,   L)
            e  = ZoneRegion(0,   W,   0,   t1y)
            w  = ZoneRegion(0,   W,   t2y, L)
            ne = ZoneRegion(t2x, W,   0,   t1y)
            se = ZoneRegion(0,   t1x, 0,   t1y)
            sw = ZoneRegion(0,   t1x, t2y, L)
            nw = ZoneRegion(t2x, W,   t2y, L)
        else:
            # W: N = left, S = right
            n  = ZoneRegion(0,   t1x, 0,   L)
            s  = ZoneRegion(t2x, W,   0,   L)
            e  = ZoneRegion(0,   W,   t2y, L)
            w  = ZoneRegion(0,   W,   0,   t1y)
            ne = ZoneRegion(0,   t1x, t2y, L)
            se = ZoneRegion(t2x, W,   t2y, L)
            sw = ZoneRegion(t2x, W,   0,   t1y)
            nw = ZoneRegion(0,   t1x, 0,   t1y)

        center = ZoneRegion(t1x, t2x, t1y, t2y)

        return {
            "N": n, "NE": ne, "E": e, "SE": se,
            "S": s, "SW": sw, "W": w, "NW": nw,
            "center": center, "CENTER": center,
        }

    def _full_plot(self) -> ZoneRegion:
        return ZoneRegion(0.0, self.W, 0.0, self.L)

    @staticmethod
    def _normalise_dir(d: str) -> str:
        """Map 8-compass + NE/SW etc. → principal N/S/E/W for zone building."""
        d = d.upper().strip() if d else "N"
        _MAP = {
            "NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W",
            "N": "N", "S": "S", "E": "E", "W": "W",
            "NE": "E", "SE": "E", "SW": "W", "NW": "W",
            "NORTH_EAST": "E", "NORTH_WEST": "W",
            "SOUTH_EAST": "E", "SOUTH_WEST": "W",
        }
        return _MAP.get(d, "N")
