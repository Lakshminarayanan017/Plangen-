"""
grid_plan.py — the wall-graph substrate.

A GridPlan is an ownership grid on the 1.5" lattice: every cell belongs to
exactly one of {OUTSIDE, WALL, a room}. Rooms are never *placed* — space is
*split* by inserting wall bands. Because ownership is exclusive by array
construction, overlap and gaps are unrepresentable, at every step, forever.

Derived views (shared walls, adjacency, clear dimensions) are computed from
the grid — the grid is the single source of truth. Openings (doors / wide
openings) are metadata anchored to shared-wall runs; the wall band itself
stays in the grid (structurally a wall with a gap, like the partial walls
between kitchen/dining/drawing in real Indian plans).

v0 scope: rectilinear plans, rectangular faces (guillotine splits produce
T-junctions across siblings naturally). Irregular plot polygons rasterize
onto the same lattice later without changing this model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from modules.step4_generate.core import units
from modules.step4_generate.core.units import EXT_WALL_CELLS, INT_WALL_CELLS

# ── Cell ownership constants ─────────────────────────────────────────────────
OUTSIDE = -1
WALL = 0
_FIRST_ROOM_ID = 1

# A separation thicker than this (12") is not a shared wall between rooms.
MAX_SHARED_WALL_CELLS = 8

# Opening defaults (cells): 30" door, 60" wide opening, 9" structural stubs.
DOOR_WIDTH = 20
WIDE_WIDTH = 40
MIN_STUB = 6

Axis = Literal["v", "h"]


@dataclass
class Room:
    id: int
    name: str
    rtype: str = "undefined"


@dataclass(frozen=True)
class SharedWall:
    """A straight wall run with room_a on one side and room_b on the other.

    For axis "v" (vertical wall): wall band spans x ∈ [wall_lo, wall_hi),
    the run extends along y ∈ [along_lo, along_hi). For "h", swap roles.
    room_a is always the lower-coordinate side (left of a "v" wall, above
    an "h" wall); ids are NOT sorted.
    """
    room_a: int
    room_b: int
    axis: Axis
    wall_lo: int
    wall_hi: int
    along_lo: int
    along_hi: int

    @property
    def length(self) -> int:
        return self.along_hi - self.along_lo

    @property
    def thickness(self) -> int:
        return self.wall_hi - self.wall_lo


@dataclass(frozen=True)
class Opening:
    """A gap in a wall: door (with leaf) or wide opening (no leaf).

    room_b == OUTSIDE marks an exterior opening (main door). `swing` names
    the side the door leaf opens into ("a" or "b"); exterior doors always
    swing inward ("a").
    """
    room_a: int
    room_b: int
    kind: str            # "door" | "wide"
    axis: Axis
    wall_lo: int
    wall_hi: int
    along_lo: int
    along_hi: int
    swing: str = "b"

    @property
    def width(self) -> int:
        return self.along_hi - self.along_lo

    @property
    def is_exterior(self) -> bool:
        return self.room_b == OUTSIDE


class CarveError(ValueError):
    """A requested mutation would violate the plan's structural rules."""


class GridPlan:
    """Ownership grid + room registry + openings. See module docstring."""

    def __init__(self, width_cells: int, height_cells: int,
                 ext_wall: int = EXT_WALL_CELLS):
        min_dim = 2 * ext_wall + 16  # ring + 2' of interior
        if width_cells < min_dim or height_cells < min_dim:
            raise CarveError(
                f"plot {width_cells}x{height_cells} cells too small for "
                f"{ext_wall}-cell exterior walls"
            )
        self.w = width_cells
        self.h = height_cells
        self.ext_wall = ext_wall
        # grid[y, x]; y grows downward (south), x grows rightward (east)
        self.grid = np.full((height_cells, width_cells), WALL, dtype=np.int16)
        self.grid[ext_wall:height_cells - ext_wall,
                  ext_wall:width_cells - ext_wall] = _FIRST_ROOM_ID
        self.rooms: Dict[int, Room] = {
            _FIRST_ROOM_ID: Room(_FIRST_ROOM_ID, "Plot", "undefined")
        }
        self._next_id = _FIRST_ROOM_ID + 1
        self.openings: List[Opening] = []
        # Fitted staircases (carve.stairs.StairFlight). Metadata anchored to a
        # face, exactly like openings: the reviewer and the renderer read the
        # SAME fitted geometry instead of each re-deriving it. Empty until
        # the stair fitter runs (single-floor plans keep it empty forever).
        self.stairs: List = []
        # Set only by from_polygon: the surveyed boundary (feet, plot-local)
        # and the buildable rectangle inside it. None on a rectangular plot.
        self.plot_polygon: Optional[List[Tuple[float, float]]] = None
        self.buildable_core: Optional[Tuple[int, int, int, int]] = None

    # ── Constructors ─────────────────────────────────────────────────────────
    @classmethod
    def from_feet(cls, width_ft: int, height_ft: int, **kw) -> "GridPlan":
        return cls(units.cells(width_ft), units.cells(height_ft), **kw)

    @classmethod
    def from_polygon(cls, points_ft, ext_wall: int = EXT_WALL_CELLS,
                     setback_ft: float = 0.0) -> "GridPlan":
        """An irregular plot on the same lattice.

        Cells outside the surveyed boundary become OUTSIDE — the ownership
        model needs no change, because "not part of this building" was
        always representable. Inside, the plot is split into:

          • the BUILDABLE CORE (face 1): the largest rectangle that fits
            inside the setback-adjusted boundary. Rectangular, so the
            ordinary band-and-column carver works on it unmodified.
          • OPEN SPACE (one face per connected leftover, rtype "ots"): the
            setback strips and splayed corners. Typed as open-to-sky
            because that is exactly what they are, which also means every
            rule and every renderer that already understands a light shaft
            understands a side yard for free.

        Raises CarveError when the core is too small to build in.
        """
        from modules.step4_generate.core import polygon as poly

        ring = poly.normalize(points_ft)
        w, h, ox, oy = poly.bbox_cells(ring)
        inside = poly.rasterize(ring, w, h, origin=(ox, oy))

        setback_cells = int(round(setback_ft * units.CELLS_PER_FOOT))
        buildable = poly.erode(inside, ext_wall + setback_cells)
        core = poly.largest_inscribed_rectangle(buildable)
        min_core = 2 * ext_wall + 16
        if core is None or (core[2] - core[0]) < min_core \
                or (core[3] - core[1]) < min_core:
            raise CarveError(
                f"polygon plot ({poly.signed_area_sqft(ring):.0f} sqft) has "
                f"no buildable rectangle at least "
                f"{units.fmt_ft_in(min_core)} on a side after a "
                f"{ext_wall / units.CELLS_PER_FOOT:.2f}' wall and a "
                f"{setback_ft:g}' setback")

        plan = cls.__new__(cls)
        plan.w, plan.h, plan.ext_wall = w, h, ext_wall
        plan.grid = np.full((h, w), OUTSIDE, dtype=np.int16)
        plan.grid[inside] = WALL
        plan.openings = []
        plan.stairs = []
        plan.rooms = {}
        plan._next_id = _FIRST_ROOM_ID

        x0, y0, x1, y1 = core
        plan.grid[y0:y1, x0:x1] = _FIRST_ROOM_ID
        plan.rooms[_FIRST_ROOM_ID] = Room(_FIRST_ROOM_ID, "Plot", "undefined")
        plan._next_id = _FIRST_ROOM_ID + 1

        leftover = buildable.copy()
        leftover[y0:y1, x0:x1] = False
        for i, piece in enumerate(poly.connected_components(leftover), 1):
            if int(piece.sum()) < units.cells(2) ** 2:
                plan.grid[piece] = WALL       # a sliver is just thick wall
                continue
            rid = plan._next_id
            plan._next_id += 1
            plan.grid[piece] = rid
            plan.rooms[rid] = Room(rid, f"Open Space {i}", "ots")

        plan.plot_polygon = ring              # kept for render / DXF export
        plan.buildable_core = core
        return plan

    # ── Face queries ─────────────────────────────────────────────────────────
    def face_mask(self, face_id: int) -> np.ndarray:
        return self.grid == face_id

    def face_area_cells(self, face_id: int) -> int:
        return int((self.grid == face_id).sum())

    def face_bbox(self, face_id: int) -> Tuple[int, int, int, int]:
        """(x0, y0, x1, y1), half-open. Raises if the face has no cells."""
        ys, xs = np.nonzero(self.grid == face_id)
        if len(xs) == 0:
            raise KeyError(f"face {face_id} has no cells")
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    def face_is_rect(self, face_id: int) -> bool:
        x0, y0, x1, y1 = self.face_bbox(face_id)
        return (x1 - x0) * (y1 - y0) == self.face_area_cells(face_id)

    def clear_dims(self, face_id: int) -> Tuple[int, int]:
        """Clear inner (width, height) in cells. Rectangular faces only."""
        if not self.face_is_rect(face_id):
            raise CarveError(f"face {face_id} is not rectangular")
        x0, y0, x1, y1 = self.face_bbox(face_id)
        return x1 - x0, y1 - y0

    def dims_str(self, face_id: int) -> str:
        w, h = self.clear_dims(face_id)
        return f"{units.fmt_ft_in(w)} x {units.fmt_ft_in(h)}"

    def area_sqft(self, face_id: int) -> float:
        return units.area_sqft(self.face_area_cells(face_id))

    def find_room(self, name: str) -> Room:
        for room in self.rooms.values():
            if room.name == name:
                return room
        raise KeyError(name)

    # ── Mutations ────────────────────────────────────────────────────────────
    def rename(self, face_id: int, name: str, rtype: str = None) -> None:
        room = self.rooms[face_id]
        room.name = name
        if rtype is not None:
            room.rtype = rtype

    def split(self, face_id: int, axis: Axis, pos: int, *,
              thick: int = INT_WALL_CELLS, name: str = "Room",
              rtype: str = "undefined", min_side: int = 8) -> int:
        """Insert a wall band across a rectangular face; the lower-coordinate
        side keeps `face_id`, the far side becomes a new room (returned id).

        `pos` is the absolute cell coordinate where the wall band starts;
        the band occupies [pos, pos+thick) on `axis`. Both remaining sides
        must keep at least `min_side` cells.
        """
        if not self.face_is_rect(face_id):
            raise CarveError(f"face {face_id} is not rectangular; cannot split")
        x0, y0, x1, y1 = self.face_bbox(face_id)
        lo, hi = (x0, x1) if axis == "v" else (y0, y1)
        if not (lo + min_side <= pos and pos + thick + min_side <= hi):
            raise CarveError(
                f"split at {pos} (thick {thick}) leaves a side under "
                f"{min_side} cells in face {face_id} [{lo},{hi})"
            )
        new_id = self._next_id
        self._next_id += 1
        if axis == "v":
            self.grid[y0:y1, pos:pos + thick] = WALL
            self.grid[y0:y1, pos + thick:x1] = new_id
        else:
            self.grid[pos:pos + thick, x0:x1] = WALL
            self.grid[pos + thick:y1, x0:x1] = new_id
        self.rooms[new_id] = Room(new_id, name, rtype)
        return new_id

    def add_opening(self, room_a: int, room_b: int, kind: str = "door", *,
                    width: Optional[int] = None, stub: int = MIN_STUB,
                    swing: str = "b", at: Optional[int] = None) -> Opening:
        """Cut an opening centered on the longest shared wall run between two
        rooms. Structural stubs of at least `stub` cells remain at both ends
        (this is the 'partial wall' of real plans). Kind "window" is used
        for glazing into open-to-sky shafts (OTS)."""
        if kind not in ("door", "wide", "window"):
            raise CarveError(f"unknown opening kind {kind!r}")
        if width is None:
            width = (DOOR_WIDTH if kind == "door"
                     else WIDE_WIDTH if kind == "wide" else 16)
        runs = [sw for sw in self.shared_walls()
                if {sw.room_a, sw.room_b} == {room_a, room_b}]
        if not runs:
            raise CarveError(
                f"rooms {self.rooms[room_a].name!r} and "
                f"{self.rooms[room_b].name!r} share no wall"
            )
        run = max(runs, key=lambda r: r.length)
        if run.length < width + 2 * stub:
            raise CarveError(
                f"shared wall between {self.rooms[room_a].name!r} and "
                f"{self.rooms[room_b].name!r} is {units.fmt_ft_in(run.length)} "
                f"— too short for a {units.fmt_ft_in(width)} opening + stubs"
            )
        if at is None:
            along_lo = run.along_lo + (run.length - width) // 2
        else:
            # explicit position (repairs shift doors away from collisions),
            # clamped so the stubs survive
            along_lo = max(run.along_lo + stub,
                           min(at, run.along_hi - width - stub))
        opening = Opening(run.room_a, run.room_b, kind, run.axis,
                          run.wall_lo, run.wall_hi, along_lo, along_lo + width,
                          swing=swing)
        self._check_opening_clash(opening)
        self.openings.append(opening)
        return opening

    def exterior_runs(self, room_id: int, side: str
                      ) -> List[Tuple[int, int]]:
        """Spans [lo, hi) along which `room_id` touches the exterior wall
        band on plot side "N"/"E"/"S"/"W"."""
        ext = self.ext_wall
        if side == "S":
            line = self.grid[self.h - ext - 1, :]
        elif side == "N":
            line = self.grid[ext, :]
        elif side == "E":
            line = self.grid[:, self.w - ext - 1]
        elif side == "W":
            line = self.grid[:, ext]
        else:
            raise ValueError(f"bad side {side!r}")
        return [(s, e) for val, s, e in self._runs(line) if val == room_id]

    def add_exterior_opening(self, room_id: int, side: str, *,
                             kind: str = "door", width: int = 28,
                             stub: int = MIN_STUB) -> Opening:
        """Cut a main-door / window / verandah opening through the exterior
        wall on the given plot side, centered on the room's longest run."""
        if kind not in ("door", "window", "wide"):
            raise CarveError(f"unknown exterior opening kind {kind!r}")
        runs = self.exterior_runs(room_id, side)
        runs = [r for r in runs if r[1] - r[0] >= width + 2 * stub]
        if not runs:
            raise CarveError(
                f"{self.rooms[room_id].name!r} has no {side}-side exterior "
                f"wall long enough for a {units.fmt_ft_in(width)} opening"
            )
        lo, hi = max(runs, key=lambda r: r[1] - r[0])
        along_lo = lo + (hi - lo - width) // 2
        ext = self.ext_wall
        if side in ("S", "N"):
            wall_lo = self.h - ext if side == "S" else 0
            axis: Axis = "h"
        else:
            wall_lo = self.w - ext if side == "E" else 0
            axis = "v"
        opening = Opening(room_id, OUTSIDE, kind, axis,
                          wall_lo, wall_lo + ext,
                          along_lo, along_lo + width, swing="a")
        self._check_opening_clash(opening)
        self.openings.append(opening)
        return opening

    def _check_opening_clash(self, opening: Opening) -> None:
        for existing in self.openings:
            same_band = (existing.axis == opening.axis
                         and existing.wall_lo == opening.wall_lo
                         and existing.wall_hi == opening.wall_hi)
            if same_band and not (existing.along_hi <= opening.along_lo
                                  or opening.along_hi <= existing.along_lo):
                raise CarveError("opening overlaps an existing opening")

    # ── Wall sliding (the Squeeze & Settle primitives) ───────────────────────
    def slide_pair_wall(self, sw: SharedWall, delta: int,
                        min_a: int, min_b: int) -> bool:
        """Slide a wall shared by exactly two rectangular rooms, where the
        run spans the COMPLETE side of both (keeps every face rectangular).
        delta > 0 moves the wall toward room_b (a grows). Returns False —
        with no state change — whenever the move is not cleanly possible.

        Must be called before any openings exist (settle-then-connect)."""
        if delta == 0 or self.openings:
            return False
        a, b = sw.room_a, sw.room_b
        if not (self.face_is_rect(a) and self.face_is_rect(b)):
            return False
        ax0, ay0, ax1, ay1 = self.face_bbox(a)
        bx0, by0, bx1, by1 = self.face_bbox(b)
        if sw.axis == "v":
            if not (ay0 == by0 == sw.along_lo and ay1 == by1 == sw.along_hi):
                return False
            if ax1 != sw.wall_lo or bx0 != sw.wall_hi:
                return False
            if delta > 0 and (bx1 - bx0) - delta < min_b:
                return False
            if delta < 0 and (ax1 - ax0) + delta < min_a:
                return False
            rows = slice(sw.along_lo, sw.along_hi)
            new_lo = sw.wall_lo + delta
            if delta > 0:
                self.grid[rows, sw.wall_lo:new_lo] = a
            else:
                self.grid[rows, new_lo + sw.thickness:sw.wall_hi] = b
            self.grid[rows, new_lo:new_lo + sw.thickness] = WALL
        else:
            if not (ax0 == bx0 == sw.along_lo and ax1 == bx1 == sw.along_hi):
                return False
            if ay1 != sw.wall_lo or by0 != sw.wall_hi:
                return False
            if delta > 0 and (by1 - by0) - delta < min_b:
                return False
            if delta < 0 and (ay1 - ay0) + delta < min_a:
                return False
            cols = slice(sw.along_lo, sw.along_hi)
            new_lo = sw.wall_lo + delta
            if delta > 0:
                self.grid[sw.wall_lo:new_lo, cols] = a
            else:
                self.grid[new_lo + sw.thickness:sw.wall_hi, cols] = b
            self.grid[new_lo:new_lo + sw.thickness, cols] = WALL
        return True

    def interior_lines(self) -> List[Tuple[Axis, int]]:
        """Full-width interior wall lines (band walls): (axis, start_pos)."""
        out: List[Tuple[Axis, int]] = []
        t = INT_WALL_CELLS
        ext = self.ext_wall
        for axis, g in (("h", self.grid), ("v", self.grid.T)):
            H, W = g.shape
            interior = slice(ext, W - ext)
            full = [(g[y, interior] == WALL).all()
                    for y in range(H)]
            y = ext
            while y <= H - ext - t:
                if all(full[y:y + t]) and not full[y - 1] and not full[y + t]:
                    out.append((axis, y))
                    y += t
                else:
                    y += 1
        return out

    def slide_line(self, axis: Axis, pos: int, delta: int,
                   min_sides: Dict[int, int]) -> bool:
        """Slide an entire full-width band wall line [pos, pos+thick) by
        delta cells (+ = toward higher coordinate). Rooms on the growing
        side extend uniformly; rooms on the shrinking side must keep their
        `min_sides` depth. Returns False — untouched — if not possible.

        Must be called before any openings exist."""
        t = INT_WALL_CELLS
        if delta == 0 or abs(delta) > MAX_SHARED_WALL_CELLS or self.openings:
            return False
        g = self.grid if axis == "h" else self.grid.T
        H, W = g.shape
        ext = self.ext_wall
        interior = slice(ext, W - ext)
        if pos < ext or pos + t > H - ext:
            return False
        if not (g[pos:pos + t, interior] == WALL).all():
            return False
        new_lo = pos + delta
        if new_lo < ext + 1 or new_lo + t > H - ext - 1:
            return False

        if delta > 0:
            shrink_region = g[pos + t:pos + t + delta, interior]
            givers = g[pos - 1, interior]
        else:
            shrink_region = g[pos + delta:pos, interior]
            givers = g[pos + t, interior]
        if (givers == WALL).all():
            return False   # nothing sensible to extend

        # every room losing depth must survive with its minimum intact
        for rid in [int(v) for v in np.unique(shrink_region) if v > 0]:
            mask = g == rid
            ys = np.nonzero(mask.any(axis=1))[0]
            depth = int(ys.max()) - int(ys.min()) + 1
            if depth - abs(delta) < min_sides.get(rid, 8):
                return False

        if delta > 0:
            g[pos:new_lo, interior] = givers
        else:
            g[new_lo + t:pos + t, interior] = givers
        g[new_lo:new_lo + t, interior] = WALL
        return True

    # ── Derived views ────────────────────────────────────────────────────────
    @staticmethod
    def _runs(line: np.ndarray) -> List[Tuple[int, int, int]]:
        """Run-length encode a 1-D line → [(value, start, end), ...]."""
        change = np.flatnonzero(np.diff(line)) + 1
        bounds = [0, *change.tolist(), len(line)]
        return [(int(line[bounds[i]]), bounds[i], bounds[i + 1])
                for i in range(len(bounds) - 1)]

    def shared_walls(self) -> List[SharedWall]:
        """All straight wall runs with a room on each side."""
        # raw[(room_a, room_b, axis, wall_lo, wall_hi)] -> sorted along-coords
        raw: Dict[Tuple, List[int]] = {}

        def scan(line: np.ndarray, along: int, axis: Axis) -> None:
            runs = self._runs(line)
            for i in range(1, len(runs) - 1):
                val, s, e = runs[i]
                if val != WALL or (e - s) > MAX_SHARED_WALL_CELLS:
                    continue
                a, b = runs[i - 1][0], runs[i + 1][0]
                if a > 0 and b > 0:
                    raw.setdefault((a, b, axis, s, e), []).append(along)

        for y in range(self.h):
            scan(self.grid[y, :], y, "v")
        for x in range(self.w):
            scan(self.grid[:, x], x, "h")

        walls: List[SharedWall] = []
        for (a, b, axis, wl, wh), alongs in raw.items():
            start = prev = alongs[0]
            for c in alongs[1:] + [None]:
                if c is not None and c == prev + 1:
                    prev = c
                    continue
                walls.append(SharedWall(a, b, axis, wl, wh, start, prev + 1))
                if c is not None:
                    start = prev = c
        return walls

    def adjacency(self) -> Dict[Tuple[int, int], int]:
        """(room_a, room_b) sorted pair → total shared wall length (cells)."""
        out: Dict[Tuple[int, int], int] = {}
        for sw in self.shared_walls():
            key = tuple(sorted((sw.room_a, sw.room_b)))
            out[key] = out.get(key, 0) + sw.length
        return out

    # ── Invariants ───────────────────────────────────────────────────────────
    def _is_connected(self, mask: np.ndarray) -> bool:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return False
        seen = np.zeros_like(mask, dtype=bool)
        stack = [(int(ys[0]), int(xs[0]))]
        seen[ys[0], xs[0]] = True
        found = 0
        h, w = mask.shape
        while stack:
            y, x = stack.pop()
            found += 1
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] \
                        and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        return found == int(mask.sum())

    def verify(self) -> bool:
        """Assert all structural invariants; True if the plan is sound."""
        values = set(int(v) for v in np.unique(self.grid))
        allowed = {OUTSIDE, WALL} | set(self.rooms.keys())
        stray = values - allowed
        assert not stray, f"grid contains unregistered ids: {stray}"

        for rid, room in self.rooms.items():
            mask = self.grid == rid
            assert mask.any(), f"room {room.name!r} ({rid}) has no cells"
            assert self._is_connected(mask), \
                f"room {room.name!r} ({rid}) is disconnected"

        walls = self.shared_walls()
        for op in self.openings:
            if op.is_exterior:
                g = self.grid if op.axis == "h" else self.grid.T
                band = g[op.wall_lo:op.wall_hi, op.along_lo:op.along_hi]
                assert (band == WALL).all(), \
                    f"exterior opening {op} band is not solid wall"
                inner = op.wall_hi if op.wall_lo == 0 else op.wall_lo - 1
                inner_line = g[inner, op.along_lo:op.along_hi]
                assert (inner_line == op.room_a).all(), \
                    f"exterior opening {op} no longer borders its room"
                continue
            host = [sw for sw in walls
                    if sw.axis == op.axis
                    and sw.wall_lo == op.wall_lo and sw.wall_hi == op.wall_hi
                    and {sw.room_a, sw.room_b} == {op.room_a, op.room_b}
                    and sw.along_lo <= op.along_lo
                    and op.along_hi <= sw.along_hi]
            assert host, (
                f"opening {op} no longer sits on a shared wall run"
            )
        return True

    # ── Reporting ────────────────────────────────────────────────────────────
    def summary(self) -> str:
        lines = [f"Plot {units.fmt_ft_in(self.w)} x {units.fmt_ft_in(self.h)}"
                 f"  ({units.fmt_area(self.w * self.h)} gross)"]
        for rid in sorted(self.rooms):
            room = self.rooms[rid]
            dims = self.dims_str(rid) if self.face_is_rect(rid) else "L-shape"
            lines.append(
                f"  [{rid:>2}] {room.name:<16} {dims:<20} "
                f"{units.fmt_area(self.face_area_cells(rid))}"
            )
        return "\n".join(lines)
