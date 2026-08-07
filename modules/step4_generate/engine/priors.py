"""
priors.py — data-driven placement priors from the legacy extraction.

Loads `zone_patterns.json` (statistics over 5,047 real floor plans: per
room type, depth percentiles front→back and lateral percentiles) and
serves median positions to the proposer. This replaces hand-tuned zone
constants with measured reality — the statistical baseline the trained
placer must beat is built from the same knowledge source it will be.

Degrades gracefully: if the file is missing or a room type is unknown,
callers fall back to their built-in constants. The engine never depends
on the legacy repo's code — only on this exported data file.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional

log = logging.getLogger("PlanGen.Engine")

# modules/step4_generate/engine/priors.py → project root is three levels up
_DEFAULT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data",
    "zone_patterns.json")

# our rtype vocabulary → the extraction's vocabulary
_RTYPE_MAP = {
    "drawing_room": "living_room",
    "toilet": "bathroom",
    "passage": "hallway",
    "foyer": "hallway",
    "store": "storage",
    "parking": "garage",
}


class ZonePriors:
    """Median depth/lateral position per room type, from real plans.

    Depth semantics returned: 0.0 = far (rear) wall, 1.0 = at the entrance
    — matching the proposer's convention. (The extraction stores 0=front,
    so values are flipped on load.)
    """

    def __init__(self, path: Optional[str] = None):
        self._depth: Dict[str, float] = {}
        self._across: Dict[str, float] = {}
        self.loaded = False
        path = path or _DEFAULT_PATH
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for rtype, stats in data.get("zone_distributions", {}).items():
                depth_p = stats.get("depth_percentiles", {})
                lat_p = stats.get("lateral_percentiles", {})
                if "p50" in depth_p:
                    # extraction: 0 = front(entrance) … 1 = back → flip
                    self._depth[rtype] = 1.0 - float(depth_p["p50"])
                if "p50" in lat_p:
                    # extraction: -1 … +1 centered → 0 … 1
                    self._across[rtype] = (float(lat_p["p50"]) + 1.0) / 2.0
            self.loaded = bool(self._depth)
            if self.loaded:
                log.info("zone priors loaded: %d room types from %s",
                         len(self._depth), os.path.normpath(path))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
            log.warning("zone priors unavailable (%s); using constants", e)

    def _key(self, rtype: str) -> str:
        return _RTYPE_MAP.get(rtype, rtype)

    def depth_for(self, rtype: str) -> Optional[float]:
        return self._depth.get(self._key(rtype))

    def across_for(self, rtype: str) -> Optional[float]:
        return self._across.get(self._key(rtype))
