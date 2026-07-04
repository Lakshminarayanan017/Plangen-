"""
rule_loader.py
==============
Singleton loader and typed accessor for sources/enricher_rules.json.

Usage:
    from sources.rule_loader import rules

    # Get size defaults for a room type
    size = rules.get_size("kitchen")          # dict with min/target/max dims
    zone = rules.get_zone("master_bedroom")   # dict with zone + vastu info
    adj  = rules.get_adjacency_weight("kitchen", "dining_room")  # float
    implict = rules.get_implicit_trigger("staircase")            # dict
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple


# ── Path resolution ────────────────────────────────────────────────────────────
# sources/ contains Python modules and config.
# All dataset JSONs live in  <project_root>/extracted data/
_HERE       = os.path.dirname(os.path.abspath(__file__))
_PROJECT    = os.path.dirname(_HERE)          # project root
_DATA_DIR   = os.path.join(_PROJECT, "extracted data")
_RULES_PATH = os.path.join(_DATA_DIR, "enricher_rules.json")


# ── Singleton loader ───────────────────────────────────────────────────────────
class EnricherRules:
    """
    Thin wrapper around enricher_rules.json providing typed, safe accessors.

    Loaded once at import time; all accessors are O(1) after the first call
    thanks to built-in Python dict look-ups and @lru_cache on computed views.
    """

    def __init__(self, path: str = _RULES_PATH) -> None:
        with open(path, encoding="utf-8") as fh:
            self._raw: Dict[str, Any] = json.load(fh)

        # Flatten adjacency rules into a single dict for O(1) look-ups:
        # key = frozenset({"room_a", "room_b"}), value = weight float
        self._adj_index: Dict[frozenset, float] = {}
        adj_section = self._raw.get("adjacency_rules", {})
        for priority_bucket in ("critical", "high", "medium", "low"):
            for rule in adj_section.get(priority_bucket, []):
                key = frozenset({rule["room_a"], rule["room_b"]})
                self._adj_index[key] = rule.get("weight", 0.0)

        # Flatten forbidden adjacencies for quick set membership check
        self._forbidden_pairs: set[frozenset] = set()
        self._forbidden_hard:  set[frozenset] = set()
        for rule in self._raw.get("forbidden_adjacencies", {}).get("rules", []):
            pair = frozenset({rule["room_a"], rule["room_b"]})
            self._forbidden_pairs.add(pair)
            if rule.get("severity") == "HARD":
                self._forbidden_hard.add(pair)

        # Flatten name aliases for quick normalisation
        self._aliases: Dict[str, str] = {
            k.lower().strip(): v
            for k, v in self._raw.get("room_name_aliases", {}).items()
            if not k.startswith("_")
        }

        # Attachment trigger phrases
        self._attach_phrases: List[str] = self._raw.get(
            "attached_bathroom_trigger_phrases", []
        )

    # ── Adjacency ──────────────────────────────────────────────────────────────

    def get_adjacency_weight(self, room_a: str, room_b: str) -> float:
        """Return the data-derived adjacency weight between two room types (0.0 if unknown)."""
        return self._adj_index.get(frozenset({room_a, room_b}), 0.0)

    def get_all_adjacency_pairs(self) -> List[Tuple[str, str, float]]:
        """Return list of (room_a, room_b, weight) for all known adjacency pairs."""
        result = []
        for k, w in self._adj_index.items():
            a, b = tuple(k)
            result.append((a, b, w))
        return result

    def is_forbidden_adjacency(self, room_a: str, room_b: str) -> bool:
        """True if this room pair should NEVER share a wall (any severity)."""
        return frozenset({room_a, room_b}) in self._forbidden_pairs

    def is_hard_forbidden_adjacency(self, room_a: str, room_b: str) -> bool:
        """True if this room pair is a HARD Vastu/hygiene violation."""
        return frozenset({room_a, room_b}) in self._forbidden_hard

    def get_forbidden_adjacency_penalty(self, room_a: str, room_b: str) -> float:
        """Return the penalty multiplier for a forbidden pair (negative float, or 0.0)."""
        for rule in self._raw.get("forbidden_adjacencies", {}).get("rules", []):
            if frozenset({rule["room_a"], rule["room_b"]}) == frozenset({room_a, room_b}):
                return rule.get("penalty_multiplier", 0.0)
        return 0.0

    # ── Zone rules ─────────────────────────────────────────────────────────────

    def get_zone(self, room_type: str) -> Optional[Dict[str, Any]]:
        """Return the full zone rule dict for a room type (None if not found)."""
        return self._raw.get("zone_rules", {}).get(room_type)

    def get_primary_zone(self, room_type: str) -> str:
        """Return 'front' | 'middle' | 'back' for the room type (default: 'middle')."""
        z = self.get_zone(room_type)
        return z.get("primary_zone", "middle") if z else "middle"

    def get_vastu_preferred_compass(self, room_type: str) -> List[str]:
        """Return Vastu-preferred compass directions (e.g. ['SW']) for a room type."""
        z = self.get_zone(room_type)
        return z.get("vastu_preferred_compass", []) if z else []

    def get_vastu_prohibited_compass(self, room_type: str) -> List[str]:
        """Return Vastu-prohibited compass directions for a room type."""
        z = self.get_zone(room_type)
        return z.get("vastu_prohibited_compass", []) if z else []

    def get_floor_preference(self, room_type: str) -> Optional[str]:
        """Return floor preference string ('ground_floor_only', 'top_floor_preferred', etc.)"""
        z = self.get_zone(room_type)
        return z.get("floor_preference") if z else None

    # ── Size rules ─────────────────────────────────────────────────────────────

    def get_size(self, room_type: str) -> Optional[Dict[str, Any]]:
        """Return the full size rule dict for a room type (None if not found)."""
        return self._raw.get("size_rules", {}).get("rooms", {}).get(room_type)

    def get_target_area(self, room_type: str) -> float:
        """Return target area in sqft (falls back to NBC minimum × 1.25 for habitable rooms)."""
        s = self.get_size(room_type)
        if s:
            return s.get("target_area_sqft", 0.0)
        nbc_min = self._raw.get("size_rules", {}).get("nbc_habitable_min_area_sqft", 102.3)
        return round(nbc_min * 1.25, 1)

    def get_min_area(self, room_type: str) -> float:
        """Return minimum area in sqft."""
        s = self.get_size(room_type)
        if s:
            return s.get("min_area_sqft", 0.0)
        return self._raw.get("size_rules", {}).get("nbc_habitable_min_area_sqft", 102.3)

    def get_target_width(self, room_type: str) -> float:
        """Return target width in feet."""
        s = self.get_size(room_type)
        return s.get("target_width_ft", 0.0) if s else 0.0

    def get_min_width(self, room_type: str) -> float:
        """Return minimum width in feet."""
        s = self.get_size(room_type)
        return s.get("min_width_ft", 0.0) if s else 0.0

    def is_habitable(self, room_type: str) -> bool:
        """True if this room type is NBC-habitable (requires natural light + ventilation)."""
        s = self.get_size(room_type)
        return s.get("habitable", False) if s else False

    def needs_natural_light(self, room_type: str) -> bool:
        """True if this room type requires natural light (NBC or convention)."""
        s = self.get_size(room_type)
        return s.get("natural_light", False) if s else False

    # ── Implicit room rules ────────────────────────────────────────────────────

    def get_implicit_trigger(self, implicit_type: str) -> Optional[Dict[str, Any]]:
        """Return the implicit room rule dict (e.g. 'staircase', 'passage', 'utility_room')."""
        return self._raw.get("implicit_room_rules", {}).get(implicit_type)

    def get_passage_threshold(self, bhk_label: str) -> float:
        """
        Return the compartmentalization threshold above which a passage should be added.
        bhk_label: 'studio' | '1BHK' | '2BHK' | '3BHK' | '4BHK'
        """
        impl = self.get_implicit_trigger("passage")
        if not impl:
            return 0.55
        thresholds = impl.get("thresholds_by_bhk", {})
        entry = thresholds.get(bhk_label, thresholds.get("default", {}))
        if isinstance(entry, dict):
            return entry.get("threshold", impl.get("fallback_threshold", 0.55))
        return impl.get("fallback_threshold", 0.55)

    # ── Floor assignment ───────────────────────────────────────────────────────

    def get_preferred_floors(self, room_type: str) -> Optional[str]:
        """
        Return: 'ground' | 'first' | 'any' | 'top' based on floor_assignment_rules.
        Returns None if room has no floor preference.
        """
        far = self._raw.get("floor_assignment_rules", {})
        if room_type in far.get("ground_floor_preferred", []):
            return "ground"
        if room_type in far.get("first_floor_preferred", []):
            return "first"
        if room_type in far.get("any_floor_acceptable", []):
            return "any"
        # Check vastu overrides
        for override in far.get("vastu_floor_overrides", []):
            if override.get("room_type") == room_type:
                r = override.get("rule", "")
                if "top_floor" in r:
                    return "top"
                if "ground_floor_only" in r:
                    return "ground"
        return None

    def is_vertical_stacking_prohibited(
        self, room_above: str, room_below: str
    ) -> bool:
        """True if placing room_above directly above room_below is Vastu-prohibited."""
        for rule in self._raw.get("floor_assignment_rules", {}).get(
            "vertical_stacking_prohibitions", []
        ):
            if rule["room_above"] == room_above and rule["room_below"] == room_below:
                return True
        return False

    # ── Circulation ────────────────────────────────────────────────────────────

    def get_max_depth(self, room_type: str) -> int:
        """Return maximum acceptable depth (hops from entry) for this room type."""
        d = self._raw.get("circulation_rules", {}).get(
            "max_depth_from_entry", {}
        ).get(room_type, {})
        return d.get("max", 4) if isinstance(d, dict) else 4

    def get_ideal_depth(self, room_type: str) -> int:
        """Return ideal depth (hops from entry) for this room type."""
        d = self._raw.get("circulation_rules", {}).get(
            "max_depth_from_entry", {}
        ).get(room_type, {})
        return d.get("ideal", 2) if isinstance(d, dict) else 2

    # ── Name normalisation ─────────────────────────────────────────────────────

    def normalize_room_type(self, display_name: str) -> str:
        """
        Convert any user-facing room name to its canonical snake_case type.
        Falls back to sanitized version of the input if not in alias map.
        """
        key = display_name.lower().strip()
        if key in self._aliases:
            return self._aliases[key]
        # Partial match
        for alias, canonical in self._aliases.items():
            if alias in key or key in alias:
                return canonical
        return key.replace(" ", "_")

    def get_all_aliases(self) -> Dict[str, str]:
        """Return the full alias → canonical type mapping."""
        return dict(self._aliases)

    # ── Attached bathroom detection ────────────────────────────────────────────

    def has_attached_bathroom_phrase(self, text: str) -> bool:
        """True if text contains any phrase indicating an attached bathroom request."""
        if not text:
            return False
        lower = text.lower()
        return any(phrase in lower for phrase in self._attach_phrases)

    # ── Placement priority ─────────────────────────────────────────────────────

    def get_placement_priority(self, room_type: str) -> int:
        """Return placement priority integer (lower = placed first)."""
        po = self._raw.get("placement_priority_order", {})
        return po.get(room_type, po.get("default", 50))

    # ── Zone balance targets ───────────────────────────────────────────────────

    def get_zone_balance_target(self, bhk_label: str) -> Dict[str, float]:
        """
        Return ideal {'front', 'middle', 'back'} fractions for a given BHK class.
        """
        targets = self._raw.get("zone_balance_targets_by_bhk", {})
        return targets.get(bhk_label, targets.get("default", {
            "front": 0.17, "middle": 0.45, "back": 0.38
        }))

    # ── FAR / budget ───────────────────────────────────────────────────────────

    def get_far(self, city: str = "default") -> float:
        """Return Floor Area Ratio for a city."""
        far_map = self._raw.get("plot_budget_rules", {}).get("far_by_city", {})
        entry = far_map.get(city.lower(), far_map.get("default", {}))
        return entry.get("residential", 1.5) if isinstance(entry, dict) else 1.5

    def get_default_setbacks(self) -> Dict[str, float]:
        """Return default setback dict {'front', 'rear', 'left', 'right'} in feet."""
        sb = self._raw.get("plot_budget_rules", {}).get("default_setbacks_ft", {})
        return {k: v for k, v in sb.items() if k in ("front", "rear", "left", "right")}

    def get_max_coverage_pct(self) -> float:
        return self._raw.get("plot_budget_rules", {}).get("max_ground_coverage_pct", 0.85)

    # ── Raw access ─────────────────────────────────────────────────────────────

    def raw(self, *keys: str) -> Any:
        """
        Navigate the raw JSON by dot-path keys.
        e.g. rules.raw('vastu_global_rules', 'structural_rules')
        """
        node = self._raw
        for k in keys:
            if not isinstance(node, dict):
                return None
            node = node.get(k)
        return node


# ── Module-level singleton ────────────────────────────────────────────────────
rules = EnricherRules()
