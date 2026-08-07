"""
settle.py — the Squeeze & Settle optimizer (milestone E3, incremental v2).

Greedy coordinate descent over wall positions. Two move classes, both
provided by the GridPlan substrate and both rectangularity-preserving:

  • band-line slides — a full-width wall line moves, re-dimensioning every
    room in the two bands it separates (how an architect deepens a zone)
  • pair-wall slides — a wall spanning the complete side of two rooms
    moves, growing one room at its neighbor's expense

Performance design: the loop maintains an incremental mirror of every
room's area and bbox (_SettleState). Candidate moves are SCORED from the
mirror in O(affected rooms) — no grid scans — and the grid is only touched
for moves that already look like improvements. The mirror is asserted
against the grid on exit, so speed never buys silent divergence.

Deterministic: no randomness, fixed sweep order, terminates when a full
sweep makes no improvement. Runs BEFORE openings exist (asserted).
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional, Set, Tuple

import numpy as np

from modules.step4_generate.core.grid_plan import INT_WALL_CELLS, GridPlan, SharedWall

_DELTAS = (8, -8, 4, -4, 2, -2)     # 12", 6", 3" in both directions
MAX_SWEEPS = 60

# aspect ratios above this cost score — settle trades area accuracy for
# proportion sanity instead of producing sliver rooms
ASPECT_LIMIT = 2.8
ASPECT_WEIGHT = 0.30


class _SettleState:
    """Incremental mirror of room areas/bboxes for O(1) move scoring."""

    def __init__(self, plan: GridPlan, targets: Dict[int, int],
                 aspect_limit: float, aspect_weight: float):
        self.targets = targets
        self.aspect_limit = aspect_limit
        self.aspect_weight = aspect_weight
        self.areas: Dict[int, int] = {}
        self.bboxes: Dict[int, Tuple[int, int, int, int]] = {}
        # track EVERY room — line slides move whole bands, and rooms
        # without targets (engine-inserted OTS shafts) still change size
        for rid in plan.rooms:
            self.areas[rid] = plan.face_area_cells(rid)
            self.bboxes[rid] = plan.face_bbox(rid)

    # ── scoring ──────────────────────────────────────────────────────────
    def room_error(self, rid: int) -> float:
        t = self.targets[rid]
        err = abs(self.areas[rid] - t) / t
        x0, y0, x1, y1 = self.bboxes[rid]
        w, h = x1 - x0, y1 - y0
        if min(w, h) > 0:
            aspect = max(w, h) / min(w, h)
            err += self.aspect_weight * max(0.0, aspect - self.aspect_limit)
        return err

    def error_of(self, rids) -> float:
        return sum(self.room_error(r) for r in set(rids) if r in self.targets)

    def mean_area_error(self) -> float:
        if not self.targets:
            return 0.0
        return sum(abs(self.areas[r] - t) / t
                   for r, t in self.targets.items()) / len(self.targets)

    # ── mutations (mirror the corresponding GridPlan slide exactly) ──────
    def apply_pair(self, sw: SharedWall, delta: int) -> None:
        a, b = sw.room_a, sw.room_b
        length = sw.along_hi - sw.along_lo
        self.areas[a] += delta * length
        self.areas[b] -= delta * length
        ax0, ay0, ax1, ay1 = self.bboxes[a]
        bx0, by0, bx1, by1 = self.bboxes[b]
        if sw.axis == "v":
            self.bboxes[a] = (ax0, ay0, ax1 + delta, ay1)
            self.bboxes[b] = (bx0 + delta, by0, bx1, by1)
        else:
            self.bboxes[a] = (ax0, ay0, ax1, ay1 + delta)
            self.bboxes[b] = (bx0, by0 + delta, bx1, by1)

    def apply_line(self, axis: str, delta: int,
                   gain: Dict[int, int], lose: Dict[int, int]) -> None:
        """gain/lose: room id → its width along the line (cells)."""
        for rid, width in gain.items():
            self.areas[rid] += delta_abs(delta) * width
            self.bboxes[rid] = _stretch(self.bboxes[rid], axis, delta,
                                        grow=True)
        for rid, width in lose.items():
            self.areas[rid] -= delta_abs(delta) * width
            self.bboxes[rid] = _stretch(self.bboxes[rid], axis, delta,
                                        grow=False)

    def verify_against(self, plan: GridPlan) -> None:
        for rid in self.targets:
            actual = plan.face_area_cells(rid)
            assert actual == self.areas[rid], (
                f"settle state diverged for room {rid}: "
                f"mirror={self.areas[rid]} grid={actual}"
            )


def delta_abs(delta: int) -> int:
    return abs(delta)


def _stretch(bbox: Tuple[int, int, int, int], axis: str, delta: int,
             grow: bool) -> Tuple[int, int, int, int]:
    """Adjust a rect bbox for a line slide along `axis` ('h' moves y
    boundaries, 'v' moves x boundaries)."""
    x0, y0, x1, y1 = bbox
    if axis == "h":
        if delta > 0:   # line moves down: upper rooms grow, lower shrink
            return (x0, y0, x1, y1 + delta) if grow else \
                   (x0, y0 + delta, x1, y1)
        return (x0, y0 + delta, x1, y1) if grow else \
               (x0, y0, x1, y1 + delta)
    if delta > 0:
        return (x0, y0, x1 + delta, y1) if grow else \
               (x0 + delta, y0, x1, y1)
    return (x0 + delta, y0, x1, y1) if grow else \
           (x0, y0, x1 + delta, y1)


def _line_membership(plan: GridPlan, axis: str, pos: int, delta: int
                     ) -> Optional[Tuple[Dict[int, int], Dict[int, int]]]:
    """(gainers, losers) room→width for a prospective line slide, read from
    the two rows flanking the line. None if the line looks non-slidable."""
    g = plan.grid if axis == "h" else plan.grid.T
    H, W = g.shape
    ext = plan.ext_wall
    t = INT_WALL_CELLS
    if pos - 1 < 0 or pos + t >= H:
        return None
    upper = g[pos - 1, ext:W - ext]
    lower = g[pos + t, ext:W - ext]
    if delta > 0:
        gain_row, lose_row = upper, lower
    else:
        gain_row, lose_row = lower, upper

    def widths(row: np.ndarray) -> Dict[int, int]:
        vals, counts = np.unique(row, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, counts) if v > 0}

    gain, lose = widths(gain_row), widths(lose_row)
    if not gain or not lose:
        return None
    return gain, lose


def settle(plan: GridPlan, targets: Dict[int, int],
           min_sides: Dict[int, int], max_sweeps: int = MAX_SWEEPS,
           aspect_limit: float = ASPECT_LIMIT,
           aspect_weight: float = ASPECT_WEIGHT,
           frozen: Optional[Set[int]] = None) -> float:
    """Optimize wall positions toward per-room area targets (cells²).
    Returns the final mean relative area error.

    `frozen` names rooms whose walls must not move at all. A room can be
    frozen for reasons no area target can express — an upper-floor
    staircase has to stay over the flight below, and settling it 6" toward
    its "ideal" area disconnects the house (measured: footprint overlap
    fell to 23-59% before this existed). Frozen rooms keep their targets in
    the error metric, so the report stays honest about their drift; they
    simply stop being movable."""
    assert not plan.openings, "settle must run before openings are added"

    frozen = frozen or set()
    state = _SettleState(plan, targets, aspect_limit, aspect_weight)

    for _ in range(max_sweeps):
        improved = False

        # ── band-line slides ─────────────────────────────────────────────
        for axis, pos in plan.interior_lines():
            moved = True
            while moved:                     # let a good line keep sliding
                moved = False
                for delta in _DELTAS:
                    membership = _line_membership(plan, axis, pos, delta)
                    if membership is None:
                        continue
                    gain, lose = membership
                    affected = set(gain) | set(lose)
                    if affected & frozen:
                        continue          # a line slide resizes whole bands
                    before = state.error_of(affected)
                    state.apply_line(axis, delta, gain, lose)
                    after = state.error_of(affected)
                    state.apply_line(axis, -delta, lose, gain)  # undo mirror
                    if after >= before - 1e-9:
                        continue
                    if plan.slide_line(axis, pos, delta, min_sides):
                        state.apply_line(axis, delta, gain, lose)
                        pos += delta
                        improved = moved = True
                        break

        # ── pair-wall slides ─────────────────────────────────────────────
        for sw in plan.shared_walls():
            if sw.room_a not in targets or sw.room_b not in targets:
                continue
            if sw.room_a in frozen or sw.room_b in frozen:
                continue
            min_a = min_sides.get(sw.room_a, 8)
            min_b = min_sides.get(sw.room_b, 8)
            for delta in _DELTAS:
                affected = (sw.room_a, sw.room_b)
                before = state.error_of(affected)
                state.apply_pair(sw, delta)
                after = state.error_of(affected)
                state.apply_pair(sw, -delta)                 # undo mirror
                if after >= before - 1e-9:
                    continue
                if plan.slide_pair_wall(sw, delta, min_a, min_b):
                    state.apply_pair(sw, delta)
                    sw = dataclasses.replace(
                        sw, wall_lo=sw.wall_lo + delta,
                        wall_hi=sw.wall_hi + delta)
                    improved = True
                    break

        if not improved:
            break

    state.verify_against(plan)
    plan.verify()
    return state.mean_area_error()
