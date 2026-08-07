"""
base.py — the reviewer's foundation: rule registry + review context.

A rule is a named, severity-tagged function over a ReviewContext. Rules
register themselves via the @rule decorator at import time; the validator
runs hard rules first (any violation disqualifies), then soft rules
(weighted penalties/bonuses with per-rule breakdown evidence).

ReviewContext carries CACHED derived artifacts (shared walls, the
free-space component, door swing regions) so thirty rules don't recompute
the same geometry thirty times. Every cached artifact is computed from the
plan on first use — rules stay pure functions of the context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from modules.step4_generate.core import units
from modules.step4_generate.core.grid_plan import GridPlan, Opening, SharedWall
from modules.step4_generate.engine.contracts import EngineConfig, EngineRequest

# ── Room-type groups (shared vocabulary of the reviewer) ─────────────────────
SOCIAL = {"living_room", "drawing_room", "dining_room", "hallway",
          "foyer", "passage"}
CIRCULATION = {"hallway", "foyer", "passage"}
KITCHEN = {"kitchen"}
PRIVATE = {"bedroom", "master_bedroom", "study", "office"}
WET = {"bathroom", "toilet"}
SERVICE = {"store", "storage", "utility", "laundry"}
PARKING = {"parking", "garage"}
STAIR = {"staircase"}
HABITABLE = SOCIAL | KITCHEN | PRIVATE


@dataclass
class ReviewContext:
    plan: GridPlan
    request: EngineRequest
    room_ids: Dict[str, int]
    config: EngineConfig
    hard: List[str] = field(default_factory=list)
    breakdown: Dict[str, float] = field(default_factory=dict)
    penalty: float = 0.0
    bonus: float = 0.0

    # ── vocabulary helpers ───────────────────────────────────────────────
    def rtype(self, rid: int) -> str:
        return self.plan.rooms[rid].rtype

    def rname(self, rid: int) -> str:
        return self.plan.rooms[rid].name

    def requested_ids(self) -> Set[int]:
        return set(self.room_ids.values())

    # ── cached derived artifacts (computed on first use, never in init) ──
    _shared_walls: Optional[List[SharedWall]] = field(
        default=None, init=False, repr=False)
    _entrance_room: Optional[int] = field(
        default=-2, init=False, repr=False)
    _free_component: Optional[Set[int]] = field(
        default=None, init=False, repr=False)
    _opening_graph: Optional[Dict[int, List[Tuple[int, Opening]]]] = field(
        default=None, init=False, repr=False)

    def shared_walls(self) -> List[SharedWall]:
        if self._shared_walls is None:
            self._shared_walls = self.plan.shared_walls()
        return self._shared_walls

    def entrance_room(self) -> Optional[int]:
        """The room you arrive in.

        Ground floor: whichever room holds the main exterior door. UPPER
        floors: the staircase — you do not walk in off the street on the
        first floor, and a plan that puts a front door up there is wrong.
        Everything measured from the entrance (reachability, circulation
        depth, STR-S03) follows this definition, which is why it lives in
        one place."""
        if self._entrance_room == -2:
            if self.request.floor_index > 0:
                self._entrance_room = next(
                    (rid for rid in self.requested_ids()
                     if self.rtype(rid) in STAIR), None)
            else:
                doors = [op for op in self.plan.openings
                         if op.is_exterior and op.kind == "door"]
                self._entrance_room = doors[0].room_a if doors else None
        return self._entrance_room

    def opening_graph(self) -> Dict[int, List[Tuple[int, Opening]]]:
        """room id → [(neighbor id, opening), ...] over interior openings."""
        if self._opening_graph is None:
            g: Dict[int, List[Tuple[int, Opening]]] = {}
            for op in self.plan.openings:
                if op.is_exterior:
                    continue
                g.setdefault(op.room_a, []).append((op.room_b, op))
                g.setdefault(op.room_b, []).append((op.room_a, op))
            self._opening_graph = g
        return self._opening_graph

    def free_component(self) -> Set[int]:
        """The main free-space component: rooms mutually connected through
        DOORLESS wide openings, anchored at the largest social room. This
        is the 'house breathes through its free space' artifact — the set
        every room must open onto (FSP-001)."""
        if self._free_component is None:
            wide: Dict[int, Set[int]] = {}
            for op in self.plan.openings:
                if op.is_exterior or op.kind != "wide":
                    continue
                wide.setdefault(op.room_a, set()).add(op.room_b)
                wide.setdefault(op.room_b, set()).add(op.room_a)

            socials = [rid for rid in self.requested_ids()
                       if self.rtype(rid) in SOCIAL]
            if not socials:
                self._free_component = set()
                return self._free_component
            anchor = max(socials, key=self.plan.face_area_cells)
            seen = {anchor}
            frontier = [anchor]
            while frontier:
                cur = frontier.pop()
                for nxt in wide.get(cur, ()):
                    if nxt not in seen:
                        seen.add(nxt)
                        frontier.append(nxt)
            self._free_component = seen
        return self._free_component

    def hops_from_entrance(self) -> Dict[int, int]:
        """BFS hop counts over interior openings from the entrance room."""
        entry = self.entrance_room()
        if entry is None:
            return {}
        hops = {entry: 0}
        frontier = [entry]
        while frontier:
            cur = frontier.pop(0)
            for nxt, _ in self.opening_graph().get(cur, ()):
                if nxt not in hops:
                    hops[nxt] = hops[cur] + 1
                    frontier.append(nxt)
        return hops

    def door_swing_rect(self, op: Opening) -> Tuple[int, int, int, int]:
        """Swing region (x0, y0, x1, y1) of a door: the square the leaf
        sweeps, on the side the door opens into."""
        assert op.kind == "door"
        w = op.width
        mid = (op.along_lo + op.along_hi) // 2
        target = op.room_a if op.swing == "a" else op.room_b
        if op.axis == "v":
            positive = (op.wall_hi < self.plan.w and
                        int(self.plan.grid[mid, op.wall_hi]) == target)
            x0 = op.wall_hi if positive else op.wall_lo - w
            return (x0, op.along_lo, x0 + w, op.along_hi)
        positive = (op.wall_hi < self.plan.h and
                    int(self.plan.grid[op.wall_hi, mid]) == target)
        y0 = op.wall_hi if positive else op.wall_lo - w
        return (op.along_lo, y0, op.along_hi, y0 + w)


# ── Registry ─────────────────────────────────────────────────────────────────

RULES: List[Tuple[str, str, Callable[[ReviewContext], None]]] = []


def rule(rule_id: str, severity: str) -> Callable:
    """Register a reviewer rule. severity: 'hard' | 'soft'."""
    assert severity in ("hard", "soft"), severity

    def wrap(fn: Callable[[ReviewContext], None]) -> Callable:
        RULES.append((rule_id, severity, fn))
        return fn
    return wrap


def cells_ft(n_cells: int) -> float:
    return n_cells / units.CELLS_PER_FOOT
