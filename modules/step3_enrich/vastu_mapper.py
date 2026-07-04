"""
vastu_mapper.py
===============
Maps Vastu Shastra compass zones (NE, SW, SE …) to plot-relative zones
(front, middle, back, left, right) based on the entrance direction.

This is the critical bridge between:
  • The absolute Vastu compass system (NE is always NE)
  • The generator's relative plot coordinate system (front = entrance side)

Design principle: pure functions / stateless helpers so the class is
safe to instantiate once and reuse across many requests.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from models import VastuConstraint


# ── Entrance direction → plot-zone → Vastu compass zones ─────────────────────
#
# For each entrance direction (the side that faces the road), which Vastu
# compass zones fall into each plot-relative zone?
#
# Visualisation (east-facing plot):
#       Road  → East side = FRONT
#       ┌─────N(left)─────┐
#       │                 │
#    W(back)   PLOT    E(front)  ← Road
#       │                 │
#       └─────S(right)────┘
#
ENTRANCE_TO_PLOT_ZONES: Dict[str, Dict[str, List[str]]] = {
    "E": {   # East-facing: entrance/road on East side
        "front":  ["E", "NE", "SE"],
        "back":   ["W", "NW", "SW"],
        "left":   ["N", "NW", "NE"],   # North = left when facing East
        "right":  ["S", "SW", "SE"],   # South = right when facing East
        "middle": ["center"],
    },
    "N": {   # North-facing: entrance/road on North side
        "front":  ["N", "NE", "NW"],
        "back":   ["S", "SE", "SW"],
        "left":   ["W", "NW", "SW"],   # West = left when facing North
        "right":  ["E", "NE", "SE"],   # East = right when facing North
        "middle": ["center"],
    },
    "S": {   # South-facing: entrance/road on South side
        "front":  ["S", "SE", "SW"],
        "back":   ["N", "NE", "NW"],
        "left":   ["E", "SE", "NE"],   # East = left when facing South
        "right":  ["W", "SW", "NW"],   # West = right when facing South
        "middle": ["center"],
    },
    "W": {   # West-facing: entrance/road on West side
        "front":  ["W", "NW", "SW"],
        "back":   ["E", "NE", "SE"],
        "left":   ["S", "SW", "SE"],   # South = left when facing West
        "right":  ["N", "NW", "NE"],   # North = right when facing West
        "middle": ["center"],
    },
}
# Diagonal entrances → inherit from cardinal parent
ENTRANCE_TO_PLOT_ZONES["NE"] = ENTRANCE_TO_PLOT_ZONES["N"]
ENTRANCE_TO_PLOT_ZONES["NW"] = ENTRANCE_TO_PLOT_ZONES["N"]
ENTRANCE_TO_PLOT_ZONES["SE"] = ENTRANCE_TO_PLOT_ZONES["E"]
ENTRANCE_TO_PLOT_ZONES["SW"] = ENTRANCE_TO_PLOT_ZONES["S"]

# Priority order when a compass zone spans multiple plot zones
# (front > back are most meaningful for the generator)
ZONE_PRIORITY: List[str] = ["front", "back", "left", "right", "middle"]

# Reverse mapping: entrance + plot_zone → primary compass direction
ZONE_TO_PRIMARY_DIR: Dict[str, Dict[str, str]] = {
    "E": {"front": "E",  "back": "W",  "left": "N",  "right": "S",  "middle": "E"},
    "N": {"front": "N",  "back": "S",  "left": "W",  "right": "E",  "middle": "N"},
    "S": {"front": "S",  "back": "N",  "left": "E",  "right": "W",  "middle": "S"},
    "W": {"front": "W",  "back": "E",  "left": "S",  "right": "N",  "middle": "W"},
}
ZONE_TO_PRIMARY_DIR["NE"] = ZONE_TO_PRIMARY_DIR["N"]
ZONE_TO_PRIMARY_DIR["NW"] = ZONE_TO_PRIMARY_DIR["N"]
ZONE_TO_PRIMARY_DIR["SE"] = ZONE_TO_PRIMARY_DIR["E"]
ZONE_TO_PRIMARY_DIR["SW"] = ZONE_TO_PRIMARY_DIR["S"]

# ── Room type → Vastu rule key mapping ───────────────────────────────────────
# Maps normalised internal room types to vastu_rules.json room_type keys
ROOM_TO_VASTU_KEY: Dict[str, str] = {
    "master_bedroom":  "master_bedroom",
    "bedroom":         "bedroom_kids",       # generic bedroom fallback
    "bedroom_kids":    "bedroom_kids",
    "bedroom_guest":   "bedroom_guest",
    "kitchen":         "kitchen",
    "dining_room":     "dining_room",
    "living_room":     "living_room",
    "drawing_room":    "living_room",
    "pooja_room":      "pooja_room",
    "study_room":      "study_room",
    "bathroom":        "bathroom",
    "toilet":          "toilet",
    "staircase":       "staircase",
    "store_room":      "store_room",
    "utility_room":    "utility_room",
    "car_parking":     "car_parking",
    "passage":         "passage",
    "foyer":           "foyer",
    "balcony":         "balcony",
    "servant_room":    "servant_room",
    "garden":          "garden",
    "verandah":        "verandah",
}


class VastuMapper:
    """
    Translates Vastu compass zone rules to plot-relative placement hints
    and extracts per-room VastuConstraint objects from the rules JSON.

    Designed to be instantiated once per request with the vastu_rules dict
    (already loaded inside KnowledgeBundle.vastu_rules_applied).
    """

    def __init__(self, vastu_rules: Dict[str, Any]) -> None:
        self._full_rules = vastu_rules
        # Build O(1) lookup: vastu room_type key → rule dict
        self._room_rules: Dict[str, Dict[str, Any]] = {
            rule["room_type"]: rule
            for rule in vastu_rules.get("room_zone_rules", [])
            if "room_type" in rule
        }

    # ── Public API ─────────────────────────────────────────────────────────

    def get_vastu_constraint(self, room_type: str) -> Optional[VastuConstraint]:
        """
        Return the VastuConstraint for a normalised room type, or None
        if no Vastu rule exists for this room type.
        """
        rule = self._lookup_rule(room_type)
        if rule is None:
            return None

        priority       = rule.get("priority", "medium")
        constraint_type = "hard" if priority == "high" else "soft"

        return VastuConstraint(
            preferred_directions        = rule.get("preferred_zones", []),
            acceptable_directions       = rule.get("acceptable_zones", []),
            prohibited_directions       = rule.get("prohibited_zones", []),
            preferred_door_directions   = rule.get("preferred_door_directions", []),
            prohibited_door_directions  = rule.get("prohibited_door_directions", []),
            floor_preference            = rule.get("floor_preference", "any_floor"),
            should_not_be_above_or_below= rule.get("should_not_be_above_or_below", []),
            should_not_be_adjacent_to   = rule.get("should_not_be_adjacent_to", []),
            constraint_type             = constraint_type,
            wall_colors_preferred       = rule.get("wall_colors_preferred", []),
            special_rules               = rule.get("special_rules", []),
            notes                       = rule.get("notes", ""),
        )

    def vastu_direction_to_plot_zone(
        self, vastu_dir: str, entrance_dir: str
    ) -> str:
        """
        Convert a Vastu compass direction (e.g. 'SW') to a plot-relative zone
        (front / back / left / right / middle) based on the entrance direction.

        Args:
            vastu_dir:    Vastu compass direction: N, NE, E, SE, S, SW, W, NW, center
            entrance_dir: Plot entrance direction: N, E, S, W (or diagonal)

        Returns:
            One of: 'front', 'back', 'left', 'right', 'middle'
        """
        ent = entrance_dir.upper().strip()
        vd  = vastu_dir.upper().strip()
        zone_map = ENTRANCE_TO_PLOT_ZONES.get(ent, ENTRANCE_TO_PLOT_ZONES["N"])
        for zone in ZONE_PRIORITY:
            if vd in zone_map.get(zone, []):
                return zone
        return "middle"  # fallback for unknown zone strings

    def plot_zone_to_compass(self, plot_zone: str, entrance_dir: str) -> str:
        """
        Convert a plot-relative zone (front/back/…) back to a primary
        compass direction for the given entrance orientation.
        """
        ent = entrance_dir.upper().strip()
        dir_map = ZONE_TO_PRIMARY_DIR.get(ent, ZONE_TO_PRIMARY_DIR["N"])
        return dir_map.get(plot_zone, ent)

    def get_primary_direction_for_room(
        self,
        room_type:      str,
        entrance_dir:   str,
        zone_probs:     Optional[Dict[str, float]] = None,
    ) -> Tuple[str, str]:
        """
        Determine (preferred_direction, preferred_zone) for a room.

        Resolution priority:
          1. Vastu preferred_directions[0]  — if Vastu rule exists
          2. Highest-probability zone from zone_probs, converted to compass
          3. Default: entrance direction + 'front'

        Args:
            room_type:   Normalised room type
            entrance_dir: Plot entrance direction
            zone_probs:  {front: p, middle: p, back: p} from KnowledgeBundle

        Returns:
            (preferred_direction, preferred_zone) e.g. ("SW", "back")
        """
        # 1. Vastu preferred direction
        vastu = self.get_vastu_constraint(room_type)
        if vastu and vastu.preferred_directions:
            pref_dir  = vastu.preferred_directions[0]       # e.g., "SW"
            pref_zone = self.vastu_direction_to_plot_zone(pref_dir, entrance_dir)
            return pref_dir, pref_zone

        # 2. Zone probabilities fallback
        if zone_probs:
            # Pick the highest-probability zone (prefer front/back over middle)
            ordered = sorted(zone_probs.items(), key=lambda kv: kv[1], reverse=True)
            best_zone = ordered[0][0] if ordered else "middle"
            pref_dir  = self.plot_zone_to_compass(best_zone, entrance_dir)
            return pref_dir, best_zone

        # 3. Default
        return entrance_dir.upper(), "front"

    def get_entrance_gate_score(
        self, entrance_dir: str, position_fraction: float = 0.5
    ) -> float:
        """
        Score a main entrance position using the Vastu pada gate system
        (outer_perimeter_gates_32 from vastu_pada_rules.json).

        Args:
            entrance_dir:      N | E | S | W
            position_fraction: 0.0 = one end of side, 1.0 = other end

        Returns:
            0.0 (hard block) → 1.0 (ideal)
        """
        gate_ids_by_dir = {
            "N": ["N1","N2","N3","N4","N5","N6","N7","N8"],
            "E": ["E1","E2","E3","E4","E5","E6","E7","E8"],
            "S": ["S1","S2","S3","S4","S5","S6","S7","S8"],
            "W": ["W1","W2","W3","W4","W5","W6","W7","W8"],
        }
        gates     = self._full_rules.get("outer_perimeter_gates_32", [])
        gate_list = gate_ids_by_dir.get(entrance_dir.upper(), gate_ids_by_dir["N"])
        target_id = gate_list[min(int(position_fraction * 8), 7)]
        for gate in gates:
            if gate.get("id") == target_id:
                return float(gate.get("score", 0.5))
        return 0.5

    # ── Private helpers ────────────────────────────────────────────────────

    def _lookup_rule(self, room_type: str) -> Optional[Dict[str, Any]]:
        """Find the best Vastu rule for a normalised room type."""
        # 1. Direct key match
        vastu_key = ROOM_TO_VASTU_KEY.get(room_type, room_type)
        rule = self._room_rules.get(vastu_key)
        if rule:
            return rule

        # 2. Partial match on room_type string
        for key, r in self._room_rules.items():
            if key in room_type or room_type in key:
                return r

        return None
