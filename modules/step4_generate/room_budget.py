"""
room_budget.py
==============
Per-room generation order and area budgets, consumed by Step 3 (the enricher)
before the layout engine ever runs.

These two functions are the only surviving pieces of the retired v1
autoregressive engine (see `unwanted/legacy_step4/autoregressive_engine.py`,
and `docs/plans/partition_first_redesign.md` for why it was retired). They are
pure ranking/budgeting maths with no dependency on the old model weights, so
they live on here rather than inside the engine package.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from models import EnrichedRoom

# ── Generation order priorities (lower = generated first) ────────────────────
# Anchor rooms get lower numbers → generated first → give more space
_GENERATION_PRIORITY: Dict[str, int] = {
    "living_room":    0,
    "drawing_room":   0,
    "master_bedroom": 1,
    "foyer":          2,
    "dining_room":    3,
    "kitchen":        3,
    "bedroom":        4,
    "study_room":     5,
    "study":          5,
    "pooja_room":     6,
    "balcony":        7,
    "bathroom":       8,
    "toilet":         8,
    "utility_room":   9,
    "store_room":     9,
    "staircase":      10,
    "car_parking":    11,
    "servant_room":   12,
    "hallway":        10,
    "passage":        10,
    "gym_room":       6,
    "home_theater":   5,
    "garden":         13,
    "terrace":        13,
    "verandah":       6,
}


def assign_generation_order(rooms: List[EnrichedRoom]) -> List[EnrichedRoom]:
    """
    Assign generation_order to each room based on its type priority.
    Rooms of the same priority are ordered by target_area_sqft (largest first).

    Called by the enricher after size assignment.
    """
    def _key(r: EnrichedRoom) -> Tuple[int, float]:
        prio = _GENERATION_PRIORITY.get(r.room_type, 7)
        return (prio, -r.target_area_sqft)   # largest within same priority first

    sorted_rooms = sorted(rooms, key=_key)
    for i, r in enumerate(sorted_rooms):
        r.generation_order = i
    return rooms


def assign_area_fractions(
    rooms: List[EnrichedRoom],
    net_buildable_area: float,
    floor_number: int,
) -> List[EnrichedRoom]:
    """
    Compute area_fraction for each room on a given floor.

    area_fraction = softmax(log_target_area) among rooms on that floor,
    clamped to NBC minimums.

    This gives a fractional budget allocation that:
    - Sums to 1.0 across all rooms on the floor
    - Is proportional to target_area_sqft from the statistical lookup
    - Never drops below NBC minimum (even very small rooms get ≥ 5% of their NBC min)
    """
    floor_rooms = [r for r in rooms if r.preferred_floor == floor_number]
    _EXTERNAL = {"car_parking", "garden", "terrace", "verandah", "balcony", "barsati"}
    internal   = [r for r in floor_rooms if r.room_type not in _EXTERNAL]

    if not internal:
        return rooms

    # Softmax over log(target_area)
    log_areas = np.array([math.log(max(r.target_area_sqft, 1.0)) for r in internal],
                         dtype=np.float64)
    log_areas -= log_areas.max()          # numerical stability
    weights   = np.exp(log_areas)
    weights  /= weights.sum()             # now sums to 1.0

    # Enforce NBC minimum fractions
    # Each room must get at least (nbc_min / net_area) fraction
    min_fracs = np.array([r.min_area_sqft / max(net_buildable_area, 1.0)
                          for r in internal], dtype=np.float64)
    min_fracs = np.minimum(min_fracs, 0.4)  # cap at 40% for any single room

    # Clip and re-normalise
    weights = np.maximum(weights, min_fracs)
    weights /= weights.sum()

    # Write back
    id_to_idx = {r.room_id: i for i, r in enumerate(internal)}
    for r in rooms:
        if r.room_id in id_to_idx:
            r.area_fraction = float(weights[id_to_idx[r.room_id]])
        elif r.room_type in _EXTERNAL:
            r.area_fraction = 0.0   # external rooms don't consume buildable area

    return rooms
