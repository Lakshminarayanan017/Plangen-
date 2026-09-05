"""
compass.py — the frame that makes Vastu expressible in the engine at all.

The engine thinks in GRID space: row 0 is the top of the lattice, and
`entrance_side` names a grid edge. Those labels are plot-relative — "N" means
"the top edge", not "north". Vastu is the opposite: every one of its rules is
about ABSOLUTE compass. A pooja room belongs in the north-east of the site,
not the north-east of whichever way the drawing happens to be oriented.

Bridging the two needs exactly one fact the engine never had:
`north_side` — which grid edge faces geographic north. `EnrichedPlan` has
carried it since step 3 (`north_direction`) and the bridge dropped it. With
it, this module maps freely between:

    grid (row, col)  <->  compass sector  (NW N NE / W center E / SW S SE)
    grid (row, col)  <->  mandala pada    (x east 0..8, y north 0..8)

The pada frame matches `data/vastuRules1.json`: the Paramasayika Mandala's
9x9, with x increasing EAST and y increasing NORTH, so the gate coordinates
in that file are usable as written.

Everything here is integer arithmetic on the lattice. No trigonometry, no
floats in the mapping — a room is in a sector or it is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# Quarter-turns (counter-clockwise) that bring `north_side` to the top.
_TURNS = {"N": 0, "E": 1, "S": 2, "W": 3}

SIDES = ("N", "E", "S", "W")

# Sector names in a north-up frame, indexed [row][col] over a 3x3.
_SECTOR_GRID: Tuple[Tuple[str, ...], ...] = (
    ("NW", "N", "NE"),
    ("W", "center", "E"),
    ("SW", "S", "SE"),
)

SECTORS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW", "center")

# The eight compass points in clockwise order, for adjacency scoring: a room
# one sector off its ideal is nearly right; three off is wrong.
_RING = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")

MANDALA = 9          # Paramasayika: 9 x 9 = 81 padas
BRAHMA_LO, BRAHMA_HI = 3, 5      # inclusive pada range of the Brahmasthan


def normalize_direction(value: Optional[str]) -> Optional[str]:
    """'north_east' / 'ne' / ' NE ' -> 'NE'. None for anything unusable."""
    if not value:
        return None
    text = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W",
        "NORTH_EAST": "NE", "NORTHEAST": "NE",
        "NORTH_WEST": "NW", "NORTHWEST": "NW",
        "SOUTH_EAST": "SE", "SOUTHEAST": "SE",
        "SOUTH_WEST": "SW", "SOUTHWEST": "SW",
        "CENTRE": "center", "CENTER": "center", "MIDDLE": "center",
        "BRAHMASTHAN": "center",
    }
    text = aliases.get(text, text)
    return text if text in SECTORS else None


def sector_distance(a: str, b: str) -> int:
    """Steps around the compass ring between two sectors (0..4).

    `center` is treated as 2 away from every ring sector — neither a match
    nor a gross violation — because a room drifting into the middle of the
    plot is a different failure from one landing on the opposite side.
    """
    if a == b:
        return 0
    if a == "center" or b == "center":
        return 2
    ia, ib = _RING.index(a), _RING.index(b)
    d = abs(ia - ib)
    return min(d, len(_RING) - d)


@dataclass(frozen=True)
class CompassFrame:
    """A plot's orientation. `north_side` is the grid edge facing north."""
    north_side: str = "N"

    def __post_init__(self) -> None:
        if self.north_side not in _TURNS:
            raise ValueError(
                f"north_side {self.north_side!r} must be one of {SIDES}")

    @property
    def turns(self) -> int:
        return _TURNS[self.north_side]

    # ── grid <-> north-up ───────────────────────────────────────────────
    def _to_north_up(self, row: int, col: int, h: int, w: int
                     ) -> Tuple[int, int, int, int]:
        """Rotate a grid coordinate CCW until north is at the top."""
        for _ in range(self.turns):
            row, col, h, w = (w - 1 - col), row, w, h
        return row, col, h, w

    def _from_north_up(self, row: int, col: int, h: int, w: int
                       ) -> Tuple[int, int, int, int]:
        """The inverse: rotate CW back into grid space."""
        for _ in range(self.turns):
            row, col, h, w = col, (h - 1 - row), w, h
        return row, col, h, w

    # ── sectors ─────────────────────────────────────────────────────────
    def sector_of(self, row: int, col: int, h: int, w: int) -> str:
        """The compass sector a single grid cell falls in."""
        r, c, hh, ww = self._to_north_up(row, col, h, w)
        sr = min(2, max(0, r * 3 // max(hh, 1)))
        sc = min(2, max(0, c * 3 // max(ww, 1)))
        return _SECTOR_GRID[sr][sc]

    def sector_of_rect(self, rect: Tuple[int, int, int, int],
                       h: int, w: int) -> str:
        """The sector a ROOM occupies, taken at its centroid.

        Centroid rather than majority-overlap on purpose: a room is where its
        middle is, and a large room straddling two sectors should read as the
        one it is centred on rather than the one it happens to have more
        cells in after a settle nudged a wall.
        """
        x0, y0, x1, y1 = rect
        return self.sector_of((y0 + y1) // 2, (x0 + x1) // 2, h, w)

    def sector_rect(self, sector: str, h: int, w: int
                    ) -> Tuple[int, int, int, int]:
        """(x0, y0, x1, y1) in GRID space covering a compass sector."""
        if sector not in SECTORS:
            raise ValueError(f"unknown sector {sector!r}")
        for sr in range(3):
            for sc in range(3):
                if _SECTOR_GRID[sr][sc] != sector:
                    continue
                # corners of this sector in north-up space, then rotated back
                nh, nw = (h, w) if self.turns % 2 == 0 else (w, h)
                r0, r1 = sr * nh // 3, (sr + 1) * nh // 3
                c0, c1 = sc * nw // 3, (sc + 1) * nw // 3
                pts = [self._from_north_up(r, c, nh, nw)[:2]
                       for r in (r0, max(r0, r1 - 1))
                       for c in (c0, max(c0, c1 - 1))]
                rows = [p[0] for p in pts]
                cols = [p[1] for p in pts]
                return (min(cols), min(rows), max(cols) + 1, max(rows) + 1)
        raise ValueError(sector)                       # unreachable

    def sector_center(self, sector: str, h: int, w: int) -> Tuple[int, int]:
        """(row, col) at the middle of a compass sector — the proposer's
        target when it biases a room toward its Vastu direction."""
        x0, y0, x1, y1 = self.sector_rect(sector, h, w)
        return ((y0 + y1) // 2, (x0 + x1) // 2)

    # ── mandala padas ───────────────────────────────────────────────────
    def pada_of(self, row: int, col: int, h: int, w: int) -> Tuple[int, int]:
        """Grid cell -> (x, y) pada, x east 0..8, y north 0..8.

        This is the frame `data/vastuRules1.json` is written in, so gate and
        energy-field coordinates from that file need no translation.
        """
        r, c, hh, ww = self._to_north_up(row, col, h, w)
        px = min(MANDALA - 1, max(0, c * MANDALA // max(ww, 1)))
        py = MANDALA - 1 - min(MANDALA - 1, max(0, r * MANDALA // max(hh, 1)))
        return px, py

    def pada_rect(self, px: int, py: int, h: int, w: int
                  ) -> Tuple[int, int, int, int]:
        """(x0, y0, x1, y1) in GRID space covering one pada."""
        nh, nw = (h, w) if self.turns % 2 == 0 else (w, h)
        r_up = MANDALA - 1 - py
        r0, r1 = r_up * nh // MANDALA, (r_up + 1) * nh // MANDALA
        c0, c1 = px * nw // MANDALA, (px + 1) * nw // MANDALA
        pts = [self._from_north_up(r, c, nh, nw)[:2]
               for r in (r0, max(r0, r1 - 1))
               for c in (c0, max(c0, c1 - 1))]
        rows = [p[0] for p in pts]
        cols = [p[1] for p in pts]
        return (min(cols), min(rows), max(cols) + 1, max(rows) + 1)

    def brahmasthan_rect(self, h: int, w: int) -> Tuple[int, int, int, int]:
        """The Brahmasthan in grid space: padas x,y in [3,5] — the central
        3x3 of the 9x9, one ninth of the plot."""
        corners = [self.pada_rect(px, py, h, w)
                   for px in (BRAHMA_LO, BRAHMA_HI)
                   for py in (BRAHMA_LO, BRAHMA_HI)]
        return (min(c[0] for c in corners), min(c[1] for c in corners),
                max(c[2] for c in corners), max(c[3] for c in corners))

    # ── sides ───────────────────────────────────────────────────────────
    def compass_of_grid_side(self, grid_side: str) -> str:
        """Which compass direction a grid edge actually faces."""
        i = SIDES.index(grid_side)
        return SIDES[(i - self.turns) % 4]

    def grid_side_of_compass(self, compass_side: str) -> str:
        """Which grid edge faces a given compass direction."""
        i = SIDES.index(compass_side)
        return SIDES[(i + self.turns) % 4]

    def describe(self) -> str:
        pairs = ", ".join(f"grid {s}={self.compass_of_grid_side(s)}"
                          for s in SIDES)
        return f"north on the {self.north_side} edge ({pairs})"
