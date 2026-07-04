"""
greedy_placer.py
================
Greedy 2-D room placement fallback for Step 4 — Layout Generator.

Used when:
  1. OR-Tools CP-SAT times out or returns infeasible.
  2. OR-Tools is not installed.
  3. Number of rooms is very small (≤ 3) — greedy is faster than CP-SAT.

Algorithm:
  1. Sort rooms by ROOM_PLACEMENT_PRIORITY (staircase first, passages last).
  2. For each room:
     a. If it has an adjacency partner already placed, try positions around
        that partner first (edge-snapping for shared-wall placement).
     b. Otherwise, start at the preferred zone centroid.
     c. Scan in a raster pattern across the floor, stepping by STEP_FT (0.5 ft).
     d. First valid (no overlap, within bounds) position wins.
  3. If no position found, try a relaxed scan across the full floor.
  4. Still no position → skip room (logged as warning).

Adjacency heuristic:
  After all rooms are placed, score adjacency by checking if high-weight
  room pairs share a wall edge (within WALL_TOL_FT tolerance).

Output:
  List of RoomPlacement objects (same type as CP-SAT solver output).
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from models import EnrichedRoom
from modules.step4_generate.grid import ROOM_PLACEMENT_PRIORITY
from modules.step4_generate.solver import RoomPlacement

log = logging.getLogger("PlanGen.Greedy")

# Scan step size in feet
STEP_FT = 0.5

# Tolerance for "sharing a wall" in adjacency scoring
WALL_TOL_FT = 0.6

# Maximum reasonable room dimension (ft) — prevents unbounded scans
MAX_DIM_FT = 40.0


class GreedyPlacer:
    """
    Greedy room placement engine.

    Places rooms one by one in priority order, scanning for the first
    non-overlapping position closest to the preferred zone.
    """

    def place(
        self,
        rooms:      List[EnrichedRoom],
        net_w:      float,
        net_l:      float,
        grid,                                   # PlotGrid instance
        adj_graph:  Dict[str, Dict[str, float]],
        staircase_anchor: Optional[Tuple[float, float, float, float]] = None,
    ) -> List[RoomPlacement]:
        """
        Place all rooms greedily.

        Args:
            rooms:            EnrichedRoom list for ONE floor.
            net_w / net_l:    Net buildable area in ft.
            grid:             PlotGrid for zone hints.
            adj_graph:        room_id → {room_id: weight} adjacency.
            staircase_anchor: (x, y, w, l) ft — lock staircase here.

        Returns:
            List of RoomPlacement (some rooms may be absent if they couldn't fit).
        """
        if not rooms:
            return []

        # ── Sort by placement priority ─────────────────────────────────
        def _priority(r: EnrichedRoom) -> int:
            return ROOM_PLACEMENT_PRIORITY.get(r.room_type, 50)

        sorted_rooms = sorted(rooms, key=_priority)

        placed: Dict[str, RoomPlacement] = {}  # room_id → RoomPlacement
        skipped: List[str] = []

        for room in sorted_rooms:
            rid = room.room_id

            # ── Determine dimensions (use target, clamp to plot) ─────────
            w = min(MAX_DIM_FT, max(room.min_width_ft,  room.target_width_ft))
            l = min(MAX_DIM_FT, max(room.min_length_ft, room.target_length_ft))

            # Ensure room can physically fit
            if w > net_w or l > net_l:
                # Try swapping orientation (portrait ↔ landscape)
                if l <= net_w and w <= net_l:
                    w, l = l, w
                else:
                    # Force minimum dimensions
                    w = min(room.min_width_ft,  net_w)
                    l = min(room.min_length_ft, net_l)

            # ── Staircase anchor (floor alignment) ────────────────────────
            if room.room_type == "staircase" and staircase_anchor:
                ax, ay, aw, al = staircase_anchor
                if self._can_place(ax, ay, aw, al, net_w, net_l, placed):
                    placed[rid] = RoomPlacement(rid, ax, ay, aw, al)
                    continue
                # Anchor blocked — fall through to normal placement

            # ── Candidate starting positions ──────────────────────────────
            candidates = self._build_candidate_starts(
                room, w, l, net_w, net_l, grid, placed, adj_graph
            )

            # ── Try each candidate, then scan ─────────────────────────────
            position_found = False
            for sx, sy in candidates:
                pos = self._scan_from(sx, sy, w, l, net_w, net_l, placed)
                if pos:
                    placed[rid] = RoomPlacement(rid, pos[0], pos[1], w, l)
                    position_found = True
                    break

            if not position_found:
                # Full fallback scan: try every valid grid point
                pos = self._full_scan(w, l, net_w, net_l, placed)
                if pos:
                    placed[rid] = RoomPlacement(rid, pos[0], pos[1], w, l)
                else:
                    log.warning("Could not place room '%s' (%s) — skipping.", rid, room.room_type)
                    skipped.append(rid)

        if skipped:
            log.warning("Greedy: %d room(s) could not be placed: %s", len(skipped), skipped)

        return list(placed.values())

    # ── Private helpers ───────────────────────────────────────────────────

    def _build_candidate_starts(
        self,
        room:      EnrichedRoom,
        w:         float,
        l:         float,
        net_w:     float,
        net_l:     float,
        grid,                                    # PlotGrid instance
        placed:    Dict[str, RoomPlacement],
        adj_graph: Dict[str, Dict[str, float]],
    ) -> List[Tuple[float, float]]:
        """
        Return a prioritised list of (x, y) positions to try for this room.

        Priority:
          1. Edge-snap positions next to already-placed high-adjacency partners.
          2. Preferred zone centroid.
          3. Corner candidates within preferred zone.
        """
        candidates: List[Tuple[float, float]] = []

        # 1. Adjacency partners: try to place adjacent to highest-weight neighbour
        room_adj = adj_graph.get(room.room_id, {})
        sorted_partners = sorted(room_adj.items(), key=lambda x: -x[1])

        for partner_id, weight in sorted_partners[:3]:  # top 3 adjacency partners
            if partner_id not in placed:
                continue
            p = placed[partner_id]
            # Try all 4 edges of the partner room
            edge_tries: List[Tuple[float, float]] = [
                (p.x_ft + p.width_ft,             p.y_ft),           # right of partner
                (p.x_ft - w,                       p.y_ft),           # left of partner
                (p.x_ft,                           p.y_ft + p.length_ft),  # above partner
                (p.x_ft,                           p.y_ft - l),       # below partner
                (p.x_ft + p.width_ft,             p.y_ft + p.length_ft - l),  # top-right
                (p.x_ft - w,                       p.y_ft + p.length_ft - l),  # top-left
            ]
            for ex, ey in edge_tries:
                # Clamp to bounds
                ex = max(0.0, min(ex, net_w - w))
                ey = max(0.0, min(ey, net_l - l))
                candidates.append((round(ex, 1), round(ey, 1)))

        # 2. Preferred zone centroid (from grid)
        sx, sy = (0.0, 0.0)
        try:
            from modules.step4_generate.grid import PlotGrid  # noqa: F401 — type ref
            region = room._grid_region if hasattr(room, "_grid_region") else None
        except Exception:
            region = None

        # Use grid hint via duck-typing
        try:
            sx, sy = _hint_start(room, w, l, net_w, net_l)
        except Exception:
            sx, sy = 0.0, 0.0

        candidates.append((sx, sy))

        # 3. Corner candidates for preferred zone
        candidates.extend([
            (0.0, 0.0),
            (0.0, net_l - l),
            (net_w - w, 0.0),
            (net_w - w, net_l - l),
        ])

        return candidates

    def _scan_from(
        self,
        start_x: float,
        start_y: float,
        w:       float,
        l:       float,
        net_w:   float,
        net_l:   float,
        placed:  Dict[str, RoomPlacement],
    ) -> Optional[Tuple[float, float]]:
        """
        Raster-scan from (start_x, start_y) outward, step=STEP_FT.
        Returns first valid (x, y) position or None.

        Scan order: spiral out from start position using row-major order
        but biased toward the starting quadrant.
        """
        # Snap start to STEP_FT grid
        sx = round(round(start_x / STEP_FT) * STEP_FT, 1)
        sy = round(round(start_y / STEP_FT) * STEP_FT, 1)

        # Number of scan steps in each direction
        steps_x = int(math.ceil(net_w / STEP_FT)) + 1
        steps_y = int(math.ceil(net_l / STEP_FT)) + 1

        # Build scan order: rows sorted by distance from sy, columns from sx
        y_offsets = sorted(
            [i * STEP_FT for i in range(-steps_y, steps_y + 1)],
            key=lambda dy: abs(sy + dy - start_y)
        )
        x_offsets = sorted(
            [i * STEP_FT for i in range(-steps_x, steps_x + 1)],
            key=lambda dx: abs(sx + dx - start_x)
        )

        # Limit scan radius to keep it fast
        MAX_CANDIDATES = 500

        count = 0
        for dy in y_offsets:
            y = sy + dy
            if y < 0.0 or y + l > net_l + 0.01:
                continue
            y = max(0.0, min(round(y, 1), net_l - l))
            for dx in x_offsets:
                x = sx + dx
                if x < 0.0 or x + w > net_w + 0.01:
                    continue
                x = max(0.0, min(round(x, 1), net_w - w))
                if self._can_place(x, y, w, l, net_w, net_l, placed):
                    return (x, y)
                count += 1
                if count > MAX_CANDIDATES:
                    return None
        return None

    def _full_scan(
        self,
        w:      float,
        l:      float,
        net_w:  float,
        net_l:  float,
        placed: Dict[str, RoomPlacement],
    ) -> Optional[Tuple[float, float]]:
        """
        Last-resort: scan every STEP_FT position across the entire floor.
        """
        y = 0.0
        while y + l <= net_l + 0.01:
            x = 0.0
            while x + w <= net_w + 0.01:
                cx = round(min(x, net_w - w), 1)
                cy = round(min(y, net_l - l), 1)
                if self._can_place(cx, cy, w, l, net_w, net_l, placed):
                    return (cx, cy)
                x = round(x + STEP_FT, 1)
            y = round(y + STEP_FT, 1)
        return None

    @staticmethod
    def _can_place(
        x:      float,
        y:      float,
        w:      float,
        l:      float,
        net_w:  float,
        net_l:  float,
        placed: Dict[str, RoomPlacement],
    ) -> bool:
        """
        True if a rectangle (x, y, w, l) fits within bounds and
        does not overlap any already-placed room.
        """
        if x < -0.01 or y < -0.01:
            return False
        if x + w > net_w + 0.01 or y + l > net_l + 0.01:
            return False

        for p in placed.values():
            # AABB overlap test (with small tolerance)
            tol = 0.05
            if (x + w - tol > p.x_ft and
                p.x_ft + p.width_ft - tol > x and
                y + l - tol > p.y_ft and
                p.y_ft + p.length_ft - tol > y):
                return False
        return True


def _hint_start(
    room:   EnrichedRoom,
    w:      float,
    l:      float,
    net_w:  float,
    net_l:  float,
) -> Tuple[float, float]:
    """
    Simple heuristic starting position for a room based on its
    preferred_zone and preferred_direction strings (no PlotGrid needed).

    Returns (x, y) — bottom-left corner of the starting position.
    """
    zone    = (room.preferred_zone or "middle").lower()
    compass = (room.preferred_direction or "N").upper()

    # X bias based on compass E/W component
    if any(c in compass for c in ("E",)):
        x_frac = 0.8  # toward east edge
    elif any(c in compass for c in ("W",)):
        x_frac = 0.2  # toward west edge
    else:
        x_frac = 0.5  # center

    # Y bias based on compass N/S component
    if "N" in compass:
        y_frac = 0.8  # toward north (top)
    elif "S" in compass:
        y_frac = 0.2  # toward south (bottom)
    else:
        y_frac = 0.5  # center

    # Override with zone
    _ZONE_X = {"front": 0.8, "back": 0.2, "middle": 0.5, "left": 0.5, "right": 0.5}
    _ZONE_Y = {"front": 0.5, "back": 0.5, "middle": 0.5, "left": 0.8, "right": 0.2}
    x_frac  = _ZONE_X.get(zone, x_frac)
    y_frac  = _ZONE_Y.get(zone, y_frac)

    x = max(0.0, min(x_frac * net_w - w / 2.0, net_w - w))
    y = max(0.0, min(y_frac * net_l - l / 2.0, net_l - l))
    return (round(x, 1), round(y, 1))
