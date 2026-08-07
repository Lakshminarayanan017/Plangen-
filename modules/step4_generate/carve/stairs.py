"""
stairs.py — exact staircase geometry on the 1.5" lattice.

A staircase is not a room-shaped blob with a label. It is a run of risers and
treads whose dimensions are fixed by the floor-to-floor height and by code,
and a face that cannot hold that run is not a staircase — it is a closet the
renderer would happily draw stair lines inside. So the geometry is computed
FIRST (from the floor height), the program reserves the area it needs, and
after the carve a variant is FITTED to the realized face. If nothing fits,
STR-S02 fails the candidate honestly rather than shipping an unbuildable
plan (implementation_plan_v2.md §5).

Every dimension lands on the lattice: the tread is 10.5" (7 cells), NOT the
nominal 10" (6.67 cells), because a dimension that cannot be built is a spec
error — units.cells() would refuse it.

Variants (NBC 2016 residential; floor height 10'0" unless overridden):
  • dogleg   6'6" x 10'0"  — two flights + half landing, the Indian default
  • straight 3'0" x 17'0"  — one flight + landing; needs a long thin face
  • lshape   10'0" x 9'1.5" — quarter landing at the corner

The stair is placed like any other room by the carver; this module only
decides what geometry a face can actually hold, and hands the renderer the
tread lines to draw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from modules.step4_generate.core import units
from modules.step4_generate.core.grid_plan import GridPlan

# ── Code-derived constants (NBC 2016, residential) ───────────────────────────
DEFAULT_FLOOR_HEIGHT_FT = 10.0

TARGET_RISER_IN = 7.0          # comfort target; riser count is chosen for it
MIN_RISER_IN = 5.5
MAX_RISER_IN = 7.5             # NBC residential max (190 mm)

TREAD_CELLS = units.cells(0, 10.5)      # 7 cells — 10.5", above the 250mm min
MIN_WIDTH_CELLS = units.cells(3)        # 24 cells — 3'0" clear flight width
LANDING_CELLS = units.cells(3)          # 24 cells — landing depth >= width
STAIRWELL_GAP_CELLS = units.cells(0, 6)  # 4 cells — the well between flights

# Preference order when several variants fit the same face: the dog-leg is
# the compact Indian default, the straight run needs a corridor-shaped face,
# the L-shape wastes its corner and is the last resort.
_PREFERENCE = ("dogleg", "straight", "lshape")


class StairError(ValueError):
    """No lattice-legal staircase exists for the requested geometry."""


# ── the variant ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StairVariant:
    """One buildable staircase geometry for a given floor height."""
    kind: str                  # "straight" | "dogleg" | "lshape"
    risers: int
    riser_in: float
    tread_cells: int
    flights: int
    width_cells: int           # footprint dimension ACROSS the flight(s)
    run_cells: int             # footprint dimension ALONG the flight(s),
    #                            landings excluded
    landing_cells: int

    # ── footprints (width, length) in cells, landing policy in the name ──
    @property
    def min_footprint(self) -> Tuple[int, int]:
        """Smallest face that can hold this stair: run + the landing the
        geometry itself requires (half landing for a dog-leg, arrival
        landing for a straight flight)."""
        return (self.width_cells, self.run_cells + self.landing_cells)

    @property
    def comfort_footprint(self) -> Tuple[int, int]:
        """Face with a 3'0" clear landing at BOTH ends — what STR-S04 wants."""
        return (self.width_cells, self.run_cells + 2 * self.landing_cells)

    @property
    def min_area_cells(self) -> int:
        w, l = self.min_footprint
        return w * l

    @property
    def min_area_sqft(self) -> float:
        return units.area_sqft(self.min_area_cells)

    @property
    def treads_per_flight(self) -> List[int]:
        """Treads in each flight. A flight of r risers shows r-1 treads: the
        top riser rises onto the landing (or the floor slab above), which is
        the surface you step out onto, not another tread."""
        base = self.risers // self.flights
        extra = self.risers % self.flights
        counts = [base + (1 if i < extra else 0) for i in range(self.flights)]
        return [max(1, c - 1) for c in counts]

    def describe(self) -> str:
        w, l = self.min_footprint
        return (f"{self.kind} — {self.risers} risers @ "
                f"{self.riser_in:.2f}\", tread "
                f"{units.fmt_ft_in(self.tread_cells)}, footprint "
                f"{units.fmt_ft_in(w)} x {units.fmt_ft_in(l)}")


def riser_count(floor_height_ft: float = DEFAULT_FLOOR_HEIGHT_FT) -> Tuple[int, float]:
    """(risers, riser height in inches) for a floor-to-floor height.

    Picks the count nearest the comfort target, then walks outward until the
    riser lands inside the code band. Raises rather than returning an
    illegal stair."""
    total_in = floor_height_ft * 12.0
    if total_in <= 0:
        raise StairError(f"floor height {floor_height_ft}' is not positive")
    n = max(2, round(total_in / TARGET_RISER_IN))
    order = sorted(range(max(2, n - 6), n + 7), key=lambda c: abs(c - n))
    for count in order:
        riser = total_in / count
        if MIN_RISER_IN <= riser <= MAX_RISER_IN:
            return count, riser
    raise StairError(
        f"no riser count in [{MIN_RISER_IN}\", {MAX_RISER_IN}\"] fits a "
        f"{floor_height_ft}' floor height")


def variants(floor_height_ft: float = DEFAULT_FLOOR_HEIGHT_FT,
             width_cells: int = MIN_WIDTH_CELLS) -> List[StairVariant]:
    """Every buildable variant for this floor height, in preference order."""
    if width_cells < MIN_WIDTH_CELLS:
        raise StairError(
            f"flight width {units.fmt_ft_in(width_cells)} is below the "
            f"{units.fmt_ft_in(MIN_WIDTH_CELLS)} minimum")
    n, riser = riser_count(floor_height_ft)
    tread = TREAD_CELLS
    landing = max(LANDING_CELLS, width_cells)

    # straight: one flight, n-1 treads
    straight = StairVariant("straight", n, riser, tread, 1,
                            width_cells, (n - 1) * tread, landing)

    # dog-leg: two flights around a half landing; the footprint length is set
    # by the LONGER flight (they share the landing)
    up = math.ceil(n / 2)
    down = n - up
    dog_run = max(max(1, up - 1), max(1, down - 1)) * tread
    dogleg = StairVariant("dogleg", n, riser, tread, 2,
                          2 * width_cells + STAIRWELL_GAP_CELLS,
                          dog_run, landing)

    # L-shape: quarter landing at the corner; both bbox sides carry a flight
    l_first = max(1, up - 1) * tread
    l_second = max(1, down - 1) * tread
    lshape = StairVariant("lshape", n, riser, tread, 2,
                          l_second + landing, l_first, landing)

    out = {"straight": straight, "dogleg": dogleg, "lshape": lshape}
    return [out[k] for k in _PREFERENCE]


def smallest_footprint_sqft(
        floor_height_ft: float = DEFAULT_FLOOR_HEIGHT_FT) -> float:
    """Area of the most compact buildable staircase, over all variants."""
    return min(v.min_area_sqft for v in variants(floor_height_ft))


def carvable_variant(floor_height_ft: float = DEFAULT_FLOOR_HEIGHT_FT,
                     width_cells: int = MIN_WIDTH_CELLS) -> StairVariant:
    """The variant the band carver must be sized for.

    Not the smallest: the smallest is the straight run, whose 3'0" x 17'0"
    footprint is a corridor no band-and-column carve will ever hand back on
    a normal plot. Reserving area for THAT one produces a face of roughly
    the right size and entirely the wrong shape, and STR-S02 then rejects
    every candidate — which is exactly what happened before this existed.

    So the carver is sized for the most compact variant whose footprint is
    a plausible band-and-column face (aspect <= 2), i.e. the dog-leg. The
    fitter still accepts any variant that fits: a long thin face that CAN
    hold a straight run is welcome, it is just never planned for.
    """
    plausible = [v for v in variants(floor_height_ft, width_cells)
                 if max(v.min_footprint) <= 2 * min(v.min_footprint)]
    pool = plausible or list(variants(floor_height_ft, width_cells))
    return min(pool, key=lambda v: v.min_area_cells)


# ── fitting a variant into a realized face ───────────────────────────────────

@dataclass(frozen=True)
class StairFlight:
    """A variant actually placed inside a face of the plan."""
    variant: StairVariant
    face_id: int
    rect: Tuple[int, int, int, int]   # (x0, y0, x1, y1) of the FACE
    run_axis: str                     # "h" = flight climbs along y, "v" = x
    both_landings: bool               # STR-S04: 3' clear at both ends

    @property
    def width_ft(self) -> float:
        return self.variant.width_cells / units.CELLS_PER_FOOT


def _fits(footprint: Tuple[int, int], face_w: int, face_h: int
          ) -> Optional[str]:
    """Return the run axis if `footprint` fits the face in some orientation.

    "h" means the flight climbs along the face's vertical extent (its length
    lies on y); "v" means along x. Both orientations are tried because the
    carver decides the face's proportions, not this module."""
    fw, fl = footprint
    if fw <= face_w and fl <= face_h:
        return "h"
    if fw <= face_h and fl <= face_w:
        return "v"
    return None


def fit_face(plan: GridPlan, face_id: int, *,
             floor_height_ft: float = DEFAULT_FLOOR_HEIGHT_FT,
             width_cells: int = MIN_WIDTH_CELLS) -> Optional[StairFlight]:
    """Best staircase that fits the realized face, or None if none does.

    Comfort footprints (landings at both ends) are tried for every variant
    before any minimum footprint, so a face big enough for a proper landing
    always gets one instead of a technically-legal cramped stair."""
    if not plan.face_is_rect(face_id):
        return None
    x0, y0, x1, y1 = plan.face_bbox(face_id)
    face_w, face_h = x1 - x0, y1 - y0
    options = variants(floor_height_ft, width_cells)

    for comfort in (True, False):
        for variant in options:
            fp = variant.comfort_footprint if comfort else variant.min_footprint
            axis = _fits(fp, face_w, face_h)
            if axis is not None:
                return StairFlight(variant=variant, face_id=face_id,
                                   rect=(x0, y0, x1, y1), run_axis=axis,
                                   both_landings=comfort)
    return None


def fit_plan(plan: GridPlan, room_ids: Optional[Dict[str, int]] = None, *,
             floor_height_ft: float = DEFAULT_FLOOR_HEIGHT_FT
             ) -> List[StairFlight]:
    """Fit every staircase face in the plan and attach the result.

    Stored on `plan.stairs` — metadata anchored to a face, exactly like
    openings, so the renderer and the reviewer read the SAME fitted geometry
    instead of each re-deriving it."""
    faces = [rid for rid, room in plan.rooms.items()
             if room.rtype == "staircase"]
    if room_ids:
        wanted = set(room_ids.values())
        faces = [f for f in faces if f in wanted] or faces
    flights = [f for f in (fit_face(plan, rid, floor_height_ft=floor_height_ft)
                           for rid in sorted(faces)) if f is not None]
    plan.stairs = flights
    return flights


def unfitted_faces(plan: GridPlan) -> List[int]:
    """Staircase faces with no fitted flight — STR-S02's evidence."""
    fitted = {f.face_id for f in getattr(plan, "stairs", [])}
    return [rid for rid, room in plan.rooms.items()
            if room.rtype == "staircase" and rid not in fitted]


# ── tread geometry for the renderer ──────────────────────────────────────────

def tread_lines(flight: StairFlight) -> List[Tuple[int, int, int, int]]:
    """Tread edges as (x0, y0, x1, y1) segments in plan cells, for the exact
    fitted variant: one line per tread, per flight, positioned from the real
    run rather than by dividing the box into n equal strips."""
    x0, y0, x1, y1 = flight.rect
    v = flight.variant
    tread = v.tread_cells
    lines: List[Tuple[int, int, int, int]] = []

    # lay the flights side by side across the width, each climbing the length
    if flight.run_axis == "h":       # length along y, width along x
        span_lo, span_hi, run_lo, run_hi = x0, x1, y0, y1
    else:                            # length along x, width along y
        span_lo, span_hi, run_lo, run_hi = y0, y1, x0, x1

    counts = v.treads_per_flight
    lanes = len(counts) if v.kind == "dogleg" else 1
    lane_w = max(1, (span_hi - span_lo) // lanes)

    for lane in range(lanes):
        lo = span_lo + lane * lane_w
        hi = lo + lane_w if lane < lanes - 1 else span_hi
        n_treads = counts[lane] if lane < len(counts) else counts[-1]
        # flight climbs away from the landing; alternate lanes reverse (the
        # dog-leg's second flight comes back down the plan toward the entry)
        for i in range(n_treads + 1):
            offset = i * tread
            pos = run_lo + offset if lane % 2 == 0 else run_hi - offset
            if not (run_lo <= pos <= run_hi):
                break
            if flight.run_axis == "h":
                lines.append((lo, pos, hi, pos))
            else:
                lines.append((pos, lo, pos, hi))
    return lines
