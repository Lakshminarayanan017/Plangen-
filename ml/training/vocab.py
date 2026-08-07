"""
vocab.py — corpus room type -> engine rtype, the single source of truth.

The engine's rtype vocabulary is the set the reviewer/carver understand
(engine/rules/base.py). Corpora use their own labels; this maps them.
Grounded in the measured CubiCasa5K histogram (training/audit/): the type
counts below are from all 4989 plans (51,719 rooms).

Design choices, each deliberate:
  • `undefined` (22.9% of rooms) → DROPPED. It carries no learnable
    placement semantics; keeping it would teach the model to place noise.
    Plans left with too few real rooms after dropping are skipped in prep.
  • `outdoor` → DROPPED as a room (it is the plot exterior / yard, not an
    interior face the carver produces). Its extent still informs the
    boundary mask upstream.
  • `sauna`, `special` → DROPPED (Finnish-specific / unlabelled; no engine
    rule, tiny counts).
  • `entry` → `foyer`; `balcony` → dropped-as-interior (exterior, like the
    engine bridge's _UNSUPPORTED); `office` → `study_room`... actually the
    engine's private set uses `study`/`office` directly (rules/base.py
    PRIVATE = {..., study, office}) so `office` maps to `office`.
  • `dining` → `dining_room`; `living_room`/`kitchen`/`bedroom`/`bathroom`/
    `garage`/`storage`/`utility`/`corridor`/`stairs` map to their engine
    equivalents.

Zones follow the engine bridge's convention (api/engine_bridge.py _TYPE_MAP)
so prepared samples carry the same zoning the live pipeline uses.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

# corpus label -> (engine rtype, zone).  None value == drop this room.
CUBICASA_TO_ENGINE: Dict[str, Optional[Tuple[str, str]]] = {
    "bedroom":      ("bedroom", "private"),
    "bathroom":     ("bathroom", "private"),
    "kitchen":      ("kitchen", "service"),
    "living_room":  ("living_room", "public"),
    "dining":       ("dining_room", "service"),
    "entry":        ("foyer", "public"),
    "storage":      ("store", "service"),
    "utility":      ("utility", "service"),
    "garage":       ("garage", "public"),
    "office":       ("office", "private"),
    "corridor":     ("hallway", "private"),
    "stairs":       ("staircase", "service"),
    # dropped (see module docstring)
    "undefined":    None,
    "outdoor":      None,
    "balcony":      None,
    "sauna":        None,
    "special":      None,
}

# room types we consider "real interior rooms" for the min-rooms gate
DROP = {k for k, v in CUBICASA_TO_ENGINE.items() if v is None}


def map_room(corpus_type: str) -> Optional[Tuple[str, str]]:
    """Return (engine_rtype, zone) or None to drop the room. Unknown corpus
    labels are dropped (conservative — never invent an engine rtype)."""
    return CUBICASA_TO_ENGINE.get((corpus_type or "").strip().lower())


# minimum real (non-dropped) rooms for a plan to be a usable training sample
MIN_REAL_ROOMS = 3


# ── canonical generation order (anchors first) ───────────────────────────────
# The AR model emits rooms in this order; training targets are sorted by it.
# Defined here for ENGINE rtypes (self-contained — the engine does not
# depend on the legacy modules/ tree), mirroring the priority principle of
# modules/step4_generate/room_budget.py::_GENERATION_PRIORITY:
# large public anchors first (they claim prime space), service/wet last.
GENERATION_PRIORITY: Dict[str, int] = {
    "living_room": 0, "drawing_room": 0,
    "master_bedroom": 1,
    "foyer": 2,
    "dining_room": 3, "kitchen": 3,
    "bedroom": 4,
    "study": 5, "office": 5,
    "hallway": 6, "passage": 6,
    "store": 8, "utility": 8,
    "bathroom": 8, "toilet": 8,
    "staircase": 9,
    "parking": 10, "garage": 10,
}
_DEFAULT_PRIORITY = 7


def generation_sort_key(rtype: str, area_sqft: float):
    """Sort key for canonical generation order: by type priority, then
    larger area first within a priority tier."""
    return (GENERATION_PRIORITY.get(rtype, _DEFAULT_PRIORITY), -area_sqft)
