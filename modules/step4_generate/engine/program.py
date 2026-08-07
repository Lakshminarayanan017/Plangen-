"""
program.py — implicit program rooms the user never asks for but the building
always needs.

A user asks for "3 bedrooms on two floors". Nobody asks for a staircase, and
nobody accepts a two-storey house without one. The engine therefore injects
the rooms the request IMPLIES before proposing, so every downstream tier
(proposer, carver, settler, reviewer) sees one complete program and none of
them needs a special case for "the invisible room".

Injection is idempotent and explicit: an already-requested staircase is left
alone (with its area raised to the buildable minimum if the user under-sized
it), and every injected room is reported in the returned notes so the app can
tell the user what was added and why.
"""

from __future__ import annotations

import dataclasses
from typing import List, Tuple

from modules.step4_generate.carve import stairs as stair_geom
from modules.step4_generate.engine.contracts import EngineRequest, RoomSpec

STAIR_TYPES = {"staircase", "stair", "stairs"}

# The stair face is carved as a room, so it carries the circulation around
# the flight too (the entry landing, the walk-past). 1.15x the bare flight
# footprint is what real plans allocate — enough to matter, not so much that
# a tight plot loses a bedroom to it.
STAIR_AREA_ALLOWANCE = 1.15


def needs_staircase(request: EngineRequest) -> bool:
    return request.n_floors > 1


def stair_area_sqft(floor_height_ft: float = stair_geom.DEFAULT_FLOOR_HEIGHT_FT
                    ) -> float:
    """Area the program reserves for a staircase, from the real geometry."""
    return round(stair_geom.smallest_footprint_sqft(floor_height_ft)
                 * STAIR_AREA_ALLOWANCE, 1)


def inject_implicit_rooms(
        request: EngineRequest, *,
        floor_height_ft: float = stair_geom.DEFAULT_FLOOR_HEIGHT_FT
        ) -> Tuple[EngineRequest, List[str]]:
    """Return (request with implicit rooms, notes). The input is untouched."""
    notes: List[str] = []
    rooms = list(request.rooms)

    if needs_staircase(request):
        wanted = stair_area_sqft(floor_height_ft)
        existing = [i for i, s in enumerate(rooms) if s.rtype in STAIR_TYPES]
        if not existing:
            rooms.append(RoomSpec(
                name="Staircase", rtype="staircase", target_sqft=wanted,
                zone="service", floor=request.floor_index))
            notes.append(
                f"staircase injected ({wanted:.0f} sqft): the request spans "
                f"{request.n_floors} floors")
        else:
            i = existing[0]
            spec = rooms[i]
            # normalize the type so the carver/reviewer see one spelling
            fixed = dataclasses.replace(
                spec, rtype="staircase",
                target_sqft=max(spec.target_sqft, wanted))
            if fixed != spec:
                rooms[i] = fixed
                if fixed.target_sqft > spec.target_sqft:
                    notes.append(
                        f"{spec.name}: area raised {spec.target_sqft:.0f} -> "
                        f"{fixed.target_sqft:.0f} sqft (smallest buildable "
                        f"flight for a {floor_height_ft:g}' floor height)")

    if rooms == list(request.rooms):
        return request, notes
    return dataclasses.replace(request, rooms=rooms), notes
