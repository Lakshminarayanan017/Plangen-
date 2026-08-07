"""
perturb.py — controlled damage, for teaching the critic what bad looks like.

Every perturbation takes a plan the engine was happy with and breaks ONE
thing an architect would notice, leaving the geometry structurally valid so
the critic cannot cheat by detecting corruption instead of judging quality.
Each returns a deep copy plus a label naming the damage — a negative sample
whose provenance is recorded, so a critic that ends up keying on one
perturbation type is visible in the per-type report rather than a mystery.

The user's caveat, taken seriously: a critic trained on real-vs-generated
learns "which generator made this", not "which plan is better". Here BOTH
classes come from the same engine and differ only by the injected flaw, so
the only separable signal is the flaw itself.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from modules.step4_generate.core.grid_plan import GridPlan, Opening
from modules.step4_generate.engine.contracts import EngineRequest
from modules.step4_generate.engine.rules.base import CIRCULATION, PRIVATE, SOCIAL, WET


@dataclass
class Perturbation:
    """A damaged copy of a plan, with the damage named."""
    plan: GridPlan
    kind: str
    detail: str


def _clone(plan: GridPlan) -> GridPlan:
    return copy.deepcopy(plan)


# ── the damage catalog ───────────────────────────────────────────────────────

def swap_room_types(plan: GridPlan, request: EngineRequest,
                    room_ids: Dict[str, int],
                    rng: random.Random) -> Optional[Perturbation]:
    """Put the kitchen where a bedroom belongs. Areas and walls are
    untouched, so ONLY the zoning is wrong — the purest quality signal."""
    pairs = [(a, b) for a in room_ids.values() for b in room_ids.values()
             if a < b
             and plan.rooms[a].rtype != plan.rooms[b].rtype
             and abs(plan.face_area_cells(a) - plan.face_area_cells(b))
             < 0.6 * max(plan.face_area_cells(a), plan.face_area_cells(b))]
    if not pairs:
        return None
    a, b = rng.choice(pairs)
    out = _clone(plan)
    ra, rb = out.rooms[a], out.rooms[b]
    ra.rtype, rb.rtype = rb.rtype, ra.rtype
    ra.name, rb.name = rb.name, ra.name
    return Perturbation(out, "swap_room_types",
                        f"{plan.rooms[a].name} <-> {plan.rooms[b].name}")


def seal_opening(plan: GridPlan, request: EngineRequest,
                 room_ids: Dict[str, int],
                 rng: random.Random) -> Optional[Perturbation]:
    """Wall up a doorway. Rooms become deeper to reach, or unreachable."""
    interior = [i for i, op in enumerate(plan.openings)
                if not op.is_exterior and op.kind != "window"]
    if not interior:
        return None
    i = rng.choice(interior)
    out = _clone(plan)
    removed = out.openings.pop(i)
    return Perturbation(out, "seal_opening",
                        f"{plan.rooms[removed.room_a].name} | "
                        f"{plan.rooms[removed.room_b].name}")


def close_the_gradient(plan: GridPlan, request: EngineRequest,
                       room_ids: Dict[str, int],
                       rng: random.Random) -> Optional[Perturbation]:
    """Turn doorless social spans into doors. Geometrically fine, and it
    destroys the one thing the reference plan is admired for: free space
    that flows."""
    wides = [i for i, op in enumerate(plan.openings)
             if op.kind == "wide" and not op.is_exterior]
    if not wides:
        return None
    out = _clone(plan)
    for i in wides:
        op = out.openings[i]
        mid = (op.along_lo + op.along_hi) // 2
        out.openings[i] = Opening(
            op.room_a, op.room_b, "door", op.axis, op.wall_lo, op.wall_hi,
            mid - 10, mid + 10, swing=op.swing)
    return Perturbation(out, "close_the_gradient",
                        f"{len(wides)} wide openings -> doors")


def bad_privacy_door(plan: GridPlan, request: EngineRequest,
                     room_ids: Dict[str, int],
                     rng: random.Random) -> Optional[Perturbation]:
    """Cut a door between two bedrooms, or from a bath into the living
    room — the two moves that make a plan feel wrong immediately."""
    ids = list(room_ids.values())
    rtype = {r: plan.rooms[r].rtype for r in ids}
    candidates: List[Tuple[int, int]] = []
    existing = {tuple(sorted((op.room_a, op.room_b))) for op in plan.openings}
    for sw in plan.shared_walls():
        a, b = sw.room_a, sw.room_b
        if a not in rtype or b not in rtype:
            continue
        if tuple(sorted((a, b))) in existing or sw.length < 32:
            continue
        both_private = rtype[a] in PRIVATE and rtype[b] in PRIVATE
        wet_social = ((rtype[a] in WET and rtype[b] in SOCIAL)
                      or (rtype[b] in WET and rtype[a] in SOCIAL))
        if both_private or wet_social:
            candidates.append((a, b))
    if not candidates:
        return None
    a, b = rng.choice(candidates)
    out = _clone(plan)
    try:
        out.add_opening(a, b, "door")
    except Exception:
        return None
    return Perturbation(out, "bad_privacy_door",
                        f"{plan.rooms[a].name} -> {plan.rooms[b].name}")


def move_the_entrance(plan: GridPlan, request: EngineRequest,
                      room_ids: Dict[str, int],
                      rng: random.Random) -> Optional[Perturbation]:
    """Enter the house through a bedroom instead of the living room."""
    door = next((i for i, op in enumerate(plan.openings)
                 if op.is_exterior and op.kind == "door"), None)
    if door is None:
        return None
    current = plan.openings[door].room_a
    others = [r for r in room_ids.values()
              if r != current and plan.rooms[r].rtype
              not in (SOCIAL | CIRCULATION)]
    rng.shuffle(others)
    for rid in others:
        out = _clone(plan)
        out.openings.pop(door)
        for side in (request.entrance_side, "N", "E", "S", "W"):
            try:
                out.add_exterior_opening(rid, side, width=28)
                return Perturbation(out, "move_the_entrance",
                                    f"entrance -> {plan.rooms[rid].name}")
            except Exception:
                continue
    return None


PERTURBATIONS: Dict[str, Callable] = {
    "swap_room_types": swap_room_types,
    "seal_opening": seal_opening,
    "close_the_gradient": close_the_gradient,
    "bad_privacy_door": bad_privacy_door,
    "move_the_entrance": move_the_entrance,
}


def perturb_all(plan: GridPlan, request: EngineRequest,
                room_ids: Dict[str, int], *, seed: int = 0,
                kinds: Optional[List[str]] = None) -> List[Perturbation]:
    """Every applicable perturbation of a plan, deterministically."""
    out: List[Perturbation] = []
    for name in (kinds or list(PERTURBATIONS)):
        fn = PERTURBATIONS[name]
        result = fn(plan, request, room_ids,
                    random.Random(seed * 7919 + hash(name) % 10_000))
        if result is not None:
            out.append(result)
    return out
