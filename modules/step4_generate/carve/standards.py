"""
standards.py — dimensional standards shared by carver, settle, validator.

One table, three consumers (blueprint §2: a rule that exists in two places
is a bug). NBC 2016-derived clear minimums per room type, the per-type
minimum side used by carving/settling, and the budget clamp that keeps
area targets legal on tight plots.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, TypeVar

from modules.step4_generate.core import units

# rtype → (min clear sqft, min clear side ft) — NBC 2016 residential.
#
# The staircase entry is registered at the bottom of this module rather than
# written here: it is not a comfort figure but the exact footprint of a
# buildable flight, computed from the floor height in carve/stairs.py.
NBC_MINIMUMS = {
    "living_room": (100, 7.5), "drawing_room": (100, 7.5),
    "master_bedroom": (100, 7.5), "bedroom": (81, 7.5),
    "dining_room": (80, 7.0), "kitchen": (48, 6.0),
    "bathroom": (19, 3.0), "toilet": (12, 3.0),
    "hallway": (25, 3.0), "parking": (110, 8.0),
}

# Carving minimum side per type (service rooms may be narrower).
_TYPE_MIN_SIDE_FT = {
    "bathroom": 3, "toilet": 3, "store": 3, "storage": 3,
    "utility": 3, "ots": 3, "hallway": 3,
}
DEFAULT_MIN_SIDE_FT = 4

# Minimum LONG side: room types whose function needs a run, not just an
# area. A staircase is the only one so far — its flight length is fixed by
# the floor height, so a face with the right area and the wrong proportion
# is useless. Area minimums alone cannot express this (65 sqft is equally
# 6'6"x10'0" and 8'x8'1.5"; only the first is a staircase).
_TYPE_MIN_LONG_FT: Dict[str, float] = {}


def type_min_side(rtype: str) -> int:
    """Minimum clear side in cells for carving/settling."""
    return units.cells(_TYPE_MIN_SIDE_FT.get(rtype, DEFAULT_MIN_SIDE_FT))


def type_min_long(rtype: str) -> int:
    """Minimum clear LONG side in cells (0 when the type has no run
    requirement). The carver keeps the room's band deep enough for it."""
    ft = _TYPE_MIN_LONG_FT.get(rtype)
    return units.cells(ft) if ft else 0


def nbc_min_area_cells(rtype: str) -> Optional[int]:
    mins = NBC_MINIMUMS.get(rtype)
    if mins is None:
        return None
    return int(mins[0] / units.SQFT_PER_CELL2)


K = TypeVar("K")


def clamp_to_minimums(targets: Dict[K, float],
                      minimums: Dict[K, float]) -> Dict[K, int]:
    """Raise every target to its minimum, paying the deficit from rooms
    with slack (proportionally to their slack) so the total is preserved.
    If minimums alone exceed the total, targets are returned min-clamped
    without rebalancing — the validator will flag the over-tight program."""
    total = sum(targets.values())
    clamped = {k: max(v, minimums.get(k, 0.0)) for k, v in targets.items()}
    if sum(minimums.get(k, 0.0) for k in targets) >= total:
        return {k: int(v) for k, v in clamped.items()}

    deficit = sum(clamped.values()) - total
    while deficit > 1:
        slack = {k: clamped[k] - minimums.get(k, 0.0) for k in clamped}
        donors = {k: s for k, s in slack.items() if s > 1}
        pool = sum(donors.values())
        if pool <= 0:
            break
        for k, s in donors.items():
            take = min(s, deficit * s / pool)
            clamped[k] -= take
        deficit = sum(clamped.values()) - total
    return {k: int(v) for k, v in clamped.items()}


def clamp_to_range(targets: Dict[K, float], minimums: Dict[K, float],
                   maximums: Dict[K, float]) -> Tuple[Dict[K, int], float]:
    """Clamp every target into [min, max] while preserving the total.

    The total is not negotiable: the carve partitions the whole plot, so the
    targets must add up to the area that actually exists. What IS negotiable
    is who gets the surplus — and proportional-to-everyone, the old
    behaviour, is the wrong answer. A living room can absorb 60% more; a
    bathroom cannot absorb 500%.

    So: clamp, then move the difference to rooms that still have headroom,
    largest headroom first. Returns (targets, unplaceable) where
    `unplaceable` is the area that could not be given to any room without
    breaching its ceiling — the caller's signal to open a courtyard rather
    than keep inflating.
    """
    total = sum(targets.values())
    out = {k: min(max(v, minimums.get(k, 0.0)),
                  maximums.get(k, float("inf"))) for k, v in targets.items()}

    # too big -> claw back from whoever is furthest above their minimum
    deficit = sum(out.values()) - total
    guard = 0
    while deficit > 1.0 and guard < 64:
        guard += 1
        slack = {k: out[k] - minimums.get(k, 0.0) for k in out}
        donors = {k: s for k, s in slack.items() if s > 1.0}
        pool = sum(donors.values())
        if pool <= 0:
            break
        for k, s in donors.items():
            out[k] -= min(s, deficit * s / pool)
        deficit = sum(out.values()) - total

    # too small -> give the surplus to whoever has room for it
    surplus = total - sum(out.values())
    guard = 0
    while surplus > 1.0 and guard < 64:
        guard += 1
        head = {k: maximums.get(k, float("inf")) - out[k] for k in out}
        takers = {k: h for k, h in head.items() if h > 1.0}
        pool = sum(v for v in takers.values() if v != float("inf"))
        if not takers:
            break
        if pool == 0 or any(h == float("inf") for h in takers.values()):
            # at least one room is unbounded; it absorbs the remainder
            free = [k for k, h in takers.items() if h == float("inf")]                 or list(takers)
            for k in free:
                out[k] += surplus / len(free)
            surplus = 0.0
            break
        for k, h in takers.items():
            out[k] += min(h, surplus * h / pool)
        surplus = total - sum(out.values())

    return {k: int(v) for k, v in out.items()}, max(0.0, surplus)


def _register_stair_minimums() -> None:
    """Derive every staircase floor from the geometry engine, never from a
    literal — change the floor height or the tread and these follow.

    The carver is sized for `carvable_variant` (the dog-leg), so the area,
    the min side and the min long side all come from ONE footprint and can
    never drift out of agreement with each other. Registered here, at import,
    rather than written into the tables above, because the value is computed:
    a hand-typed copy is a copy that goes stale."""
    from modules.step4_generate.carve.stairs import DEFAULT_FLOOR_HEIGHT_FT, carvable_variant
    v = carvable_variant(DEFAULT_FLOOR_HEIGHT_FT)
    width, length = v.min_footprint
    per_ft = units.CELLS_PER_FOOT
    NBC_MINIMUMS["staircase"] = (round(v.min_area_sqft, 1), width / per_ft)
    _TYPE_MIN_SIDE_FT["staircase"] = width / per_ft
    _TYPE_MIN_LONG_FT["staircase"] = length / per_ft


_register_stair_minimums()
