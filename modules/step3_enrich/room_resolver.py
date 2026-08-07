"""
room_resolver.py
================
Normalises the raw RoomRequirement list from BuildingRequirements into
a flat, expanded list of individual ResolvedRoom instances with stable IDs.

Responsibilities:
  1. Expand quantities       → Bedroom(qty=3) → bedroom_1, bedroom_2, bedroom_3
  2. Normalise type names    → "Master Bedroom" → "master_bedroom"
  3. Auto-promote first bedroom → "Master Bedroom" when multiple bedrooms exist
  4. Detect attachment hints   → "attached bathroom" in specific_requirements
  5. Add implicit rooms        → staircase, passage, utility, common bathroom
  6. Resolve bathroom links    → decide which bathroom attaches to which bedroom

Output: List[ResolvedRoom] — one object per physical room instance, fully
annotated, ready for the Enricher to convert to EnrichedRoom.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from models import BuildingRequirements, RoomRequirement
from modules.step2_match.indian_standards import is_habitable
from sources.rule_loader import rules as _rules


# ── Room type normalisation map — loaded from enricher_rules.json ─────────────
# Fallback hard-coded map kept for any alias missing from rule book
_FALLBACK_DISPLAY_TO_NORM: Dict[str, str] = {
    "master bedroom":   "master_bedroom",
    "bedroom":          "bedroom",
    "living room":      "living_room",
    "dining room":      "dining_room",
    "kitchen":          "kitchen",
    "bathroom":         "bathroom",
    "toilet":           "toilet",
    "staircase":        "staircase",
    "passage":          "passage",
    "utility room":     "utility_room",
    "store room":       "store_room",
    "study room":       "study_room",
    "car parking":      "car_parking",
    "pooja room":       "pooja_room",
}

# Build normalisation maps: rule book first, fallback second
DISPLAY_TO_NORM: Dict[str, str] = {
    **_FALLBACK_DISPLAY_TO_NORM,
    **_rules.get_all_aliases(),           # rule book wins on conflicts
}

# Canonical internal type → human-readable display label
NORM_TO_DISPLAY: Dict[str, str] = {
    "master_bedroom":  "Master Bedroom",
    "bedroom":         "Bedroom",
    "bedroom_kids":    "Children's Bedroom",
    "bedroom_guest":   "Guest Bedroom",
    "kitchen":         "Kitchen",
    "living_room":     "Living Room",
    "drawing_room":    "Drawing Room",
    "dining_room":     "Dining Room",
    "pooja_room":      "Pooja Room",
    "bathroom":        "Bathroom",
    "toilet":          "Common Toilet",
    "balcony":         "Balcony",
    "utility_room":    "Utility Room",
    "store_room":      "Store Room",
    "study_room":      "Study Room",
    "servant_room":    "Servant Room",
    "car_parking":     "Car Parking",
    "staircase":       "Staircase",
    "foyer":           "Foyer",
    "passage":         "Passage",
    "verandah":        "Verandah",
    "terrace":         "Terrace",
    "garden":          "Garden",
    "gym_room":        "Gym Room",
    "home_theater":    "Home Theater",
    "barsati":         "Barsati",
}

# Phrases that indicate an attached bathroom — loaded from rule book
_ATTACHED_BATH_PHRASES: Set[str] = set(
    _rules.raw("attached_bathroom_trigger_phrases") or [
        "attached bathroom", "attached bath", "ensuite", "en-suite",
        "attached toilet", "attached washroom", "with bathroom",
        "with bath", "attached wc", "with attached", "en suite",
    ]
)


@dataclass
class ResolvedRoom:
    """
    Mutable intermediate representation of a single physical room instance.

    Created by RoomResolver; consumed by the Enricher to build EnrichedRoom.
    Every instance represents exactly ONE room (quantities already expanded).
    """
    room_id:            str
    room_type:          str            # normalised snake_case
    display_name:       str
    quantity_index:     int            # 1-based index within same type
    implicit_room:      bool = False   # True = added by enricher, not user
    preferred_floor:    Optional[int] = None
    user_specific_requirements: Optional[str] = None
    wants_attached_bathroom: bool = False
    attached_bath_room_id:   Optional[str] = None  # set during bath-link pass
    original_req: Optional[RoomRequirement] = field(default=None, repr=False)


class RoomResolver:
    """
    Normalises BuildingRequirements.rooms into a fully-expanded,
    annotated list of ResolvedRoom instances.

    Stateless — safe to instantiate once and reuse across many requests.
    """

    # ── Public API ──────────────────────────────────────────────────────────

    def resolve(
        self,
        reqs:                BuildingRequirements,
        n_matched_bhk:       int   = 3,
        compartmentalization: float = 0.6,
        add_implicit:        bool  = True,
    ) -> List[ResolvedRoom]:
        """
        Expand and normalise all rooms from BuildingRequirements.

        Args:
            reqs:                 Parsed BuildingRequirements from Step 1.
            n_matched_bhk:        BHK count inferred from matched plans.
                                  Used to decide whether to add utility room.
            compartmentalization: Score from matched plans' circulation stats.
                                  High value → more compartmentalised → add passage.
            add_implicit:         If True, insert implicit rooms per Indian norms.

        Returns:
            Flat list of ResolvedRoom — one per physical room instance.
        """
        type_counters: Dict[str, int] = {}
        floors = reqs.number_of_floors or 1

        # ── Phase 1: Expand user-specified rooms ────────────────────────
        rooms: List[ResolvedRoom] = self._expand_user_rooms(reqs, type_counters)

        # ── Phase 2: Insert implicit rooms ──────────────────────────────
        if add_implicit:
            rooms = self._add_implicit_rooms(
                rooms, reqs, floors, n_matched_bhk,
                compartmentalization, type_counters,
                bhk_label=getattr(reqs, "bhk_label", None),
            )

        # ── Phase 3: Resolve bathroom → bedroom attachments ─────────────
        rooms = self._resolve_bathroom_links(rooms)

        return rooms

    # ── Private: Phase 1 ───────────────────────────────────────────────────

    def _expand_user_rooms(
        self,
        reqs:          BuildingRequirements,
        type_counters: Dict[str, int],
    ) -> List[ResolvedRoom]:
        """Expand each RoomRequirement by its quantity into individual rooms."""
        rooms: List[ResolvedRoom] = []

        # First pass: collect total counts so we know when to auto-promote
        type_totals: Dict[str, int] = {}
        for req in reqs.rooms:
            norm = self._normalize_type(req.room_type)
            type_totals[norm] = type_totals.get(norm, 0) + req.quantity

        for req in reqs.rooms:
            norm_type  = self._normalize_type(req.room_type)
            qty        = max(1, req.quantity)
            wants_bath = self._check_attached_bath(req)

            for i in range(qty):
                type_counters[norm_type] = type_counters.get(norm_type, 0) + 1
                idx          = type_counters[norm_type]
                total_of_type = type_totals.get(norm_type, qty)

                # ── Auto-promote: first generic bedroom in a multi-bedroom plan
                # becomes "Master Bedroom" with a distinct type so Vastu rules,
                # size statistics, and adjacency weights can be looked up correctly.
                if norm_type == "bedroom" and idx == 1 and total_of_type > 1:
                    actual_type  = "master_bedroom"
                    room_id      = "master_bedroom_1"
                    name         = "Master Bedroom"
                    # Ensure master_bedroom has its own counter slot
                    type_counters["master_bedroom"] = 1
                else:
                    actual_type = norm_type
                    room_id     = f"{norm_type}_{idx}"
                    name        = self._make_display_name(
                        norm_type, idx, total_of_type=total_of_type,
                    )

                rooms.append(ResolvedRoom(
                    room_id            = room_id,
                    room_type          = actual_type,
                    display_name       = name,
                    quantity_index     = idx,
                    implicit_room      = False,
                    preferred_floor    = req.preferred_floor,
                    user_specific_requirements = req.specific_requirements,
                    wants_attached_bathroom    = wants_bath,
                    original_req       = req,
                ))

        return rooms

    # ── Private: Phase 2 ───────────────────────────────────────────────────

    def _add_implicit_rooms(
        self,
        rooms:               List[ResolvedRoom],
        reqs:                BuildingRequirements,
        floors:              int,
        n_matched_bhk:       int,
        compartmentalization: float,
        type_counters:       Dict[str, int],
        bhk_label:           Optional[str] = None,
    ) -> List[ResolvedRoom]:
        """
        Insert rooms per Indian residential construction norms.

        Rule thresholds are now loaded from sources/enricher_rules.json so that
        they are data-backed and version-controlled rather than magic numbers.
        """
        existing = {r.room_type for r in rooms}
        new: List[ResolvedRoom] = []

        n_bedrooms  = sum(1 for r in rooms if "bedroom" in r.room_type)
        n_bathrooms = sum(1 for r in rooms if r.room_type in ("bathroom", "toilet"))
        total_rooms = len(rooms)

        # Determine BHK label for threshold look-ups
        if not bhk_label:
            if n_matched_bhk >= 4:
                bhk_label = "4BHK"
            elif n_matched_bhk == 3:
                bhk_label = "3BHK"
            elif n_matched_bhk == 2:
                bhk_label = "2BHK"
            elif n_matched_bhk == 1:
                bhk_label = "1BHK"
            else:
                bhk_label = "studio"

        def _add(room_type: str, display: Optional[str] = None,
                  pref_floor: Optional[int] = None) -> None:
            type_counters[room_type] = type_counters.get(room_type, 0) + 1
            idx = type_counters[room_type]
            new.append(ResolvedRoom(
                room_id        = f"{room_type}_{idx}",
                room_type      = room_type,
                display_name   = display or NORM_TO_DISPLAY.get(room_type,
                                    room_type.replace("_", " ").title()),
                quantity_index = idx,
                implicit_room  = True,
                preferred_floor = pref_floor,
            ))

        # ── Rule 0: Core Habitable Rooms (Minimum Viable House) ────────────────
        # Source: Common Sense — a residential plan must have basic living spaces.
        if "living_room" not in existing and "drawing_room" not in existing:
            _add("living_room", pref_floor=0)
            existing.add("living_room")

        if "kitchen" not in existing:
            _add("kitchen", pref_floor=0)
            existing.add("kitchen")

        if n_bedrooms == 0:
            _add("bedroom", pref_floor=0)
            existing.add("bedroom")
            n_bedrooms = 1  # Update local counter so Rule 2a (bathrooms) triggers

        # ── Rule 1: Staircase ─────────────────────────────────────────────────
        # Source: CONV — every multi-floor building requires a staircase.
        if floors > 1 and "staircase" not in existing:
            _add("staircase", pref_floor=0)

        # ── Rule 2a: Fallback common bathrooms ───────────────────────────────
        # Source: CONV — add bathrooms only when user specified NONE at all.
        # Formula from rule book: max(1, n-1) for 3BHK+, n for 1-2BHK.
        if n_bedrooms > 0 and n_bathrooms == 0:
            n_add = max(1, n_bedrooms - 1 if n_bedrooms > 2 else n_bedrooms)
            for _ in range(n_add):
                _add("bathroom")

        # ── Rule 2b: Implicit attached bathrooms ─────────────────────────────
        # Source: CONV + VASTU — each bedroom that requested an attached bath
        # gets a private bathroom on the same floor.
        for bed in rooms:
            if "bedroom" in bed.room_type and bed.wants_attached_bathroom:
                _add("bathroom", pref_floor=bed.preferred_floor)

        # ── Rule 2c: Master bedroom always gets en-suite ──────────────────────
        # Source: CONV + VASTU (HIGH confidence from rule book).
        # Handled in Phase 3 bathroom-link pass — no implicit room created here,
        # but the master_bedroom_ensuite rule ensures Phase 3 always links one.

        # ── Rule 3: Common toilet for 3BHK+ ──────────────────────────────────
        # Source: CONV — ground floor guest toilet for 3BHK+ homes.
        if (
            n_bedrooms >= 3
            and n_bathrooms == 0
            and "toilet" not in existing
        ):
            _add("toilet", display="Common Toilet", pref_floor=0)

        # ── Rule 4: Passage per qualifying floor ──────────────────────────────
        # Source: CUBICASA — threshold derived from p75 compartmentalization per BHK.
        # Threshold is now looked up from rule book (data-driven) not hard-coded 0.55.
        passage_threshold = _rules.get_passage_threshold(bhk_label)
        min_rooms_for_passage = (
            _rules.raw("circulation_rules", "passage_rules", "add_passage_if_floor_has") or 3
        )

        if compartmentalization > passage_threshold:
            all_so_far = rooms + new
            from collections import defaultdict as _dd
            floor_rooms: dict = _dd(list)
            for _r in all_so_far:
                _floor = _r.preferred_floor if _r.preferred_floor is not None else 0
                floor_rooms[_floor].append(_r)

            for _floor_num in sorted(floor_rooms.keys()):
                _on_floor = floor_rooms[_floor_num]
                _has_passage = any(r.room_type == "passage" for r in _on_floor)
                if not _has_passage and len(_on_floor) >= min_rooms_for_passage:
                    _add("passage", pref_floor=_floor_num)

        # ── Rule 5: Utility room for 3BHK+ with kitchen ──────────────────────
        # Source: CONV + CUBICASA (224 plans show kitchen↔utility adjacency).
        # Threshold (n_matched_bhk >= 3) is consistent with rule book trigger.
        has_kitchen = "kitchen" in existing
        has_utility = "utility_room" in existing
        if (
            has_kitchen
            and not has_utility
            and n_matched_bhk >= 3
            and (total_rooms + len(new)) >= 5
        ):
            _add("utility_room", pref_floor=0)

        return rooms + new

    # ── Private: Phase 3 ───────────────────────────────────────────────────

    def _resolve_bathroom_links(
        self, rooms: List[ResolvedRoom]
    ) -> List[ResolvedRoom]:
        """
        Link bathrooms to bedrooms using a floor-aware matching strategy.

        Matching priority for each bathroom slot:
          1. Master bedroom always gets an attached bathroom (Indian convention).
          2. Bedrooms that explicitly requested an attached bathroom get next.
          3. If #bathrooms >= #bedrooms, remaining bedrooms also get attached.
          4. Unmatched bathrooms remain as common bathrooms (no attachment).

        Floor-preference rule:
          When selecting a bathroom for a bedroom, prefer:
            (a) A bathroom on the SAME floor that is IMPLICIT (was created
                specifically as an attached bath for this floor).
            (b) A bathroom on the SAME floor (any origin).
            (c) Any remaining bathroom (cross-floor fallback).
          After linking, the bathroom's preferred_floor is co-located with
          its bedroom so the enricher floor-assignment step locks them together.
        """
        bedrooms  = [r for r in rooms if "bedroom" in r.room_type]
        bathrooms = [r for r in rooms if r.room_type in ("bathroom", "toilet")]

        if not bathrooms or not bedrooms:
            return rooms

        available: List[ResolvedRoom] = list(bathrooms)

        def _find_best_bath(bed: ResolvedRoom) -> Optional[ResolvedRoom]:
            """
            Return the best available bathroom for this bedroom.
            Preference order: same-floor implicit → same-floor any → any floor.
            """
            # (a) Same floor + implicit (was created specifically for this floor)
            for bath in available:
                if (bath.preferred_floor == bed.preferred_floor
                        and bath.implicit_room):
                    return bath
            # (b) Same floor, any origin
            for bath in available:
                if bath.preferred_floor == bed.preferred_floor:
                    return bath
            # (c) Cross-floor fallback
            return available[0] if available else None

        def _attach(bed: ResolvedRoom) -> None:
            bath = _find_best_bath(bed)
            if bath is None:
                return
            available.remove(bath)
            bed.attached_bath_room_id  = bath.room_id
            bed.wants_attached_bathroom = True
            # Co-locate: move the bathroom to the bedroom's floor so the
            # enricher's user_floor_map will lock both on the same floor.
            bath.preferred_floor = bed.preferred_floor

        # Pass 1: master bedroom (Indian default — master always has en-suite)
        for bed in bedrooms:
            if bed.room_type == "master_bedroom" and not bed.attached_bath_room_id:
                _attach(bed)

        # Pass 2: bedrooms that explicitly requested an attached bathroom
        for bed in bedrooms:
            if bed.wants_attached_bathroom and not bed.attached_bath_room_id:
                _attach(bed)

        # Pass 3: if surplus bathrooms, attach to all remaining bedrooms
        if len(bathrooms) >= len(bedrooms):
            for bed in bedrooms:
                if not bed.attached_bath_room_id:
                    _attach(bed)

        return rooms

    # ── Private: helpers ────────────────────────────────────────────────────

    def _normalize_type(self, room_type: str) -> str:
        """User display name → canonical internal snake_case type."""
        key = room_type.lower().strip()
        if key in DISPLAY_TO_NORM:
            return DISPLAY_TO_NORM[key]
        # Partial match
        for k, v in DISPLAY_TO_NORM.items():
            if k in key or key in k:
                return v
        # Already snake_case or unknown — sanitize and return as-is
        return key.replace(" ", "_")

    def _make_display_name(
        self, norm_type: str, idx: int, total_of_type: int
    ) -> str:
        """
        Create a human-readable display name.
        First bedroom in a multi-bedroom plan is auto-promoted to Master Bedroom.
        """
        base = NORM_TO_DISPLAY.get(norm_type, norm_type.replace("_", " ").title())

        # Auto-promote the first generic bedroom to Master Bedroom
        if norm_type == "bedroom" and idx == 1 and total_of_type > 1:
            return "Master Bedroom"

        # Single instance — no number suffix needed
        if total_of_type == 1:
            return base

        return f"{base} {idx}"

    def _check_attached_bath(self, req: RoomRequirement) -> bool:
        """Detect attached-bathroom request in specific_requirements text."""
        if not req.specific_requirements:
            return False
        txt = req.specific_requirements.lower()
        return any(phrase in txt for phrase in _ATTACHED_BATH_PHRASES)
