"""
enricher.py
===========
Step 3 — Enricher: the orchestrator that converts a KnowledgeBundle into a
fully-specified EnrichedPlan ready for the layout generator (Step 4).

Pipeline executed on each call to Enricher.enrich():
  1. Resolve plot geometry & setbacks
  2. Determine entrance direction & north orientation
  3. Expand + normalise rooms via RoomResolver
  4. Assign NBC-clamped sizes from bundle statistics
  5. Assign floors (user > Vastu > Gemini > statistics > heuristics)
  6. Assign zones + compass directions (Vastu > zone_probs > entrance)
  7. Apply full Vastu constraints per room
  8. Link bathroom → bedroom attachments
  9. Build per-room-ID adjacency graph from type-level weights
 10. Build FloorPlan objects
 11. Compute FAR / coverage / area budget
 12. Return EnrichedPlan

The Gemini call (step 5) is optional: it fires only for multi-floor
buildings where the user did not specify room floors and the bundle's
statistical defaults are insufficient. On any Gemini failure the system
falls back gracefully to statistical + heuristic floor assignment.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from models import (
    BuildingRequirements,
    EnrichedPlan,
    EnrichedRoom,
    FloorPlan,
    KnowledgeBundle,
    Setbacks,
    VastuConstraint,
)
from modules.step2_match.indian_standards import (
    get_door_width,
    get_room_minimums,
    is_habitable,
)
from modules.step3_enrich.room_resolver import (
    NORM_TO_DISPLAY,
    ResolvedRoom,
    RoomResolver,
)
from modules.step3_enrich.vastu_mapper import VastuMapper
from modules.step4_generate.autoregressive_engine import (
    assign_generation_order,
    assign_area_fractions,
)
from sources.rule_loader import rules as _rules

# ── Logging ──────────────────────────────────────────────────────────────────
log = logging.getLogger("PlanGen.Enricher")

# ── Environment ──────────────────────────────────────────────────────────────
_ENV_PATH = Path(__file__).parents[2] / "sources" / ".env"
load_dotenv(_ENV_PATH)

# ── Ceiling heights: rule book first, hard-coded fallback ────────────────────
def _get_ceiling(room_type: str) -> float:
    """Ceiling height in ft from rule book (size_rules), else hard-coded fallback."""
    s = _rules.get_size(room_type)
    if s and s.get("ceiling_height_ft", 0) > 0:
        return s["ceiling_height_ft"]
    _FALLBACK: Dict[str, float] = {
        "master_bedroom": 9.0, "bedroom": 9.0, "bedroom_kids": 9.0,
        "bedroom_guest": 9.0, "kitchen": 9.0, "living_room": 10.0,
        "drawing_room": 9.0, "dining_room": 9.0, "study_room": 9.0,
        "pooja_room": 9.0, "bathroom": 9.0, "toilet": 9.0,
        "utility_room": 9.0, "store_room": 9.0, "staircase": 10.0,
        "passage": 9.0, "foyer": 10.0, "car_parking": 9.0,
        "servant_room": 9.0, "balcony": 9.0,
    }
    return _FALLBACK.get(room_type, 9.0)


# ── Indian default sizes: rule book first, hard-coded fallback ───────────────
# Returns (target_width_ft, target_area_sqft) for rooms not in CubiCasa5K.
def _get_india_default(room_type: str) -> Optional[Tuple[float, float]]:
    """Look up target (width, area) from rule book, then hard-coded fallback."""
    s = _rules.get_size(room_type)
    if s and s.get("target_width_ft", 0) > 0 and s.get("target_area_sqft", 0) > 0:
        return (s["target_width_ft"], s["target_area_sqft"])
    _FALLBACK: Dict[str, Tuple[float, float]] = {
        "pooja_room":   (5.5,  35.0),
        "car_parking":  (10.0, 120.0),
        "staircase":    (4.5,  40.0),
        "utility_room": (6.0,  40.0),
        "passage":      (3.5,  28.0),
        "foyer":        (7.0,  45.0),
        "servant_room": (8.0,  80.0),
        "toilet":       (4.0,  25.0),
        "verandah":     (8.0,  64.0),
        "garden":      (10.0, 200.0),
        "terrace":     (12.0, 200.0),
        "balcony":      (6.0,  40.0),
        "store_room":   (6.0,  40.0),
        "gym_room":    (11.0, 120.0),
        "home_theater":(13.0, 150.0),
        "barsati":      (9.0,  90.0),
    }
    if room_type in _FALLBACK:
        return _FALLBACK[room_type]
    return None


# ── Floor heuristics: rule book first, hard-coded fallback ───────────────────
def _get_floor_heuristic(room_type: str, floors: int) -> int:
    """
    Return default floor number for a room with no user/Vastu/Gemini assignment.
    Priority: rule book floor_assignment_rules → hard-coded fallback → 0.
    """
    pref = _rules.get_preferred_floors(room_type)
    if pref == "ground":
        return 0
    if pref == "first":
        return min(1, floors - 1)
    if pref == "top":
        return floors - 1
    # Hard-coded fallback for any room type not in rule book
    _FALLBACK: Dict[str, int] = {
        "living_room": 0, "drawing_room": 0, "dining_room": 0,
        "kitchen": 0, "pooja_room": 0, "car_parking": 0,
        "utility_room": 0, "store_room": 0, "toilet": 0,
        "foyer": 0, "passage": 0, "staircase": 0,
        "verandah": 0, "servant_room": 0, "garden": 0,
        "master_bedroom": 1, "bedroom": 1, "bedroom_kids": 1,
        "bedroom_guest": 1, "bathroom": 1, "study_room": 1,
        "balcony": 1, "gym_room": 1, "home_theater": 1,
        "terrace": 2, "barsati": 2,
    }
    return min(_FALLBACK.get(room_type, 0), floors - 1)


class Enricher:
    """
    Converts a BuildingRequirements + KnowledgeBundle into an EnrichedPlan.

    Thread-safe: all state is local to each enrich() call except for the
    lazily-initialised GeminiKeyRotator (shared across calls, which is safe
    since the rotator itself is thread-safe).
    """

    def __init__(self, use_gemini: bool = True) -> None:
        """
        Args:
            use_gemini: If True and the plan has > 1 floor, call Gemini to
                        reason about optimal floor distribution.
                        Set False in unit tests or offline environments.
        """
        self._use_gemini   = use_gemini
        self._key_rotator  = None          # lazy init on first Gemini call
        self._resolver     = RoomResolver()

    # ── Public API ─────────────────────────────────────────────────────────

    def enrich(
        self,
        reqs:   BuildingRequirements,
        bundle: KnowledgeBundle,
    ) -> EnrichedPlan:
        """
        Execute the full enrichment pipeline.

        Args:
            reqs:   Validated BuildingRequirements from Step 1.
            bundle: Knowledge bundle from Step 2 (SemanticMatcher).

        Returns:
            Fully-specified EnrichedPlan.
        """
        log.info("Enricher starting — vastu=%s, floors=%s",
                 reqs.vastu_compliant, reqs.number_of_floors)

        warnings:  List[str]            = []
        decisions: List[Dict[str, str]] = []

        # ── 1. Plot geometry ────────────────────────────────────────────
        plot   = self._resolve_plot(reqs, bundle, warnings)
        sb     = self._resolve_setbacks(reqs, bundle, plot["area"])
        net_w  = max(1.0, plot["width"]  - sb.left  - sb.right)
        net_l  = max(1.0, plot["length"] - sb.front - sb.rear)

        # ── 2. Orientation ──────────────────────────────────────────────
        ent_dir, north_dir = self._resolve_orientation(reqs)

        # ── 3. Setup Vastu mapper ───────────────────────────────────────
        vastu_enabled = bool(reqs.vastu_compliant and bundle.vastu_rules_applied)
        vastu_mapper: Optional[VastuMapper] = None
        if vastu_enabled:
            vastu_mapper = VastuMapper(bundle.vastu_rules_applied)

        # ── 4. Resolve + expand rooms ───────────────────────────────────
        floors      = reqs.number_of_floors or 1
        n_bhk       = self._infer_bhk(bundle)
        compart     = self._estimate_compartmentalization(reqs, bundle)

        resolved    = self._resolver.resolve(
            reqs,
            n_matched_bhk       = n_bhk,
            compartmentalization = compart,
            add_implicit        = True,
        )
        log.info("Rooms after resolver: %d (%d user, %d implicit)",
                 len(resolved),
                 sum(1 for r in resolved if not r.implicit_room),
                 sum(1 for r in resolved if r.implicit_room))

        # ── 5. Assign sizes ─────────────────────────────────────────────
        enrichment_source = "full_statistical"
        enriched = self._assign_sizes(resolved, bundle, warnings)
        if any(r.target_area_sqft == get_room_minimums(r.room_type)["min_area_sqft"]
               for r in enriched):
            enrichment_source = "nbc_fallback"

        # ── 5b. Capture user-explicit floor locks BEFORE any heuristics ──
        # The resolver stores preferred_floor=None for rooms with no user-specified
        # floor. _assign_sizes converts None→0 so we can no longer distinguish
        # "user said GF" from "default fallback to 0" using EnrichedRoom alone.
        # We snapshot the original resolved values here so _assign_floors can
        # enforce them with absolute priority.
        user_floor_map: Dict[str, Optional[int]] = {
            r.room_id: r.preferred_floor for r in resolved
        }

        # ── 6. Assign floors ────────────────────────────────────────────
        enriched = self._assign_floors(
            enriched, reqs, bundle, floors, ent_dir,
            vastu_enabled, vastu_mapper, decisions, warnings,
            user_floor_map=user_floor_map,
        )

        # ── 6b. Audit Vastu floor conflicts (user-locked vs Vastu rule) ──
        if vastu_enabled and vastu_mapper:
            self._audit_vastu_floor_conflicts(
                enriched, user_floor_map or {}, floors, vastu_mapper, warnings
            )

        # ── 6c. Scale rooms to fit within net-buildable area per floor ───
        enriched = self._scale_rooms_to_fit(enriched, net_w, net_l, floors, warnings)

        # ── 7. Assign zones + compass directions ────────────────────────
        enriched = self._assign_zones(
            enriched, bundle, ent_dir, vastu_enabled, vastu_mapper
        )

        # ── 8. Apply full Vastu constraints ─────────────────────────────
        if vastu_enabled and vastu_mapper:
            enriched = self._apply_vastu(enriched, vastu_mapper)

        # ── 9. Link bathroom attachments ────────────────────────────────
        enriched = self._link_bathrooms(enriched, resolved)

        # ── 10. Build adjacency graph ───────────────────────────────────
        adj_graph = self._build_adjacency_graph(enriched, bundle)

        # ── 10b. Assign area fractions + generation order (Option 4 AR) ─
        # area_fraction: softmax-normalised share of floor area per room.
        #   Replaces the static lookup-table sizing in the AR engine.
        #   Computed per-floor so fractions sum to ~1.0 within each floor.
        # generation_order: anchor rooms (living room, master bed) are
        #   generated first by the AR transformer; service rooms last.
        net_buildable_area = round(net_w * net_l, 2)
        for floor_num in range(floors):
            enriched = assign_area_fractions(enriched, net_buildable_area, floor_num)
        enriched = assign_generation_order(enriched)
        enrichment_source = enrichment_source or "area_budget_ar"

        # ── 11. Build FloorPlan objects ─────────────────────────────────
        floor_plans = self._build_floor_plans(enriched, floors)

        # ── 12. FAR / coverage / budget ─────────────────────────────────
        max_coverage   = round(plot["area"] * 0.60, 2)
        max_far        = round(plot["area"] * 1.50, 2)
        total_area     = round(sum(r.target_area_sqft for r in enriched), 2)
        area_budget_ok = total_area <= max_far
        if not area_budget_ok:
            warnings.append(
                f"Total target area ({total_area} sqft) exceeds FAR limit "
                f"({max_far} sqft). Generator will scale rooms down."
            )

        # ── 13. Build Vastu direction summary ───────────────────────────
        vastu_summary: Dict[str, str] = {}
        if vastu_enabled:
            for r in enriched:
                if r.vastu and r.vastu.preferred_directions:
                    vastu_summary[r.room_type] = r.vastu.preferred_directions[0]

        # ── 14. Implicit rooms list ──────────────────────────────────────
        implicit_names = [r.display_name for r in enriched if r.implicit_room]

        log.info("Enrichment complete: %d rooms, area=%s sqft, budget_ok=%s",
                 len(enriched), total_area, area_budget_ok)

        return EnrichedPlan(
            original_requirements    = reqs,
            match_quality_score      = bundle.match_quality_score,
            plot_width_ft            = plot["width"],
            plot_length_ft           = plot["length"],
            plot_area_sqft           = plot["area"],
            setbacks                 = sb,
            net_buildable_width_ft   = round(net_w, 2),
            net_buildable_length_ft  = round(net_l, 2),
            net_buildable_area_sqft  = round(net_w * net_l, 2),
            entrance_direction       = ent_dir,
            north_direction          = north_dir,
            total_floors             = floors,
            floors                   = floor_plans,
            rooms                    = enriched,
            implicit_rooms_added     = implicit_names,
            adjacency_graph          = adj_graph,
            vastu_enabled            = vastu_enabled,
            vastu_direction_assignments = vastu_summary,
            max_ground_coverage_sqft = max_coverage,
            max_far_total_sqft       = max_far,
            total_target_area_sqft   = total_area,
            area_budget_ok           = area_budget_ok,
            enrichment_source        = enrichment_source,
            enrichment_warnings      = warnings,
            gemini_decisions         = decisions,
        )

    # ── Step 1: Plot geometry ──────────────────────────────────────────────

    def _resolve_plot(
        self,
        reqs:     BuildingRequirements,
        bundle:   KnowledgeBundle,
        warnings: List[str],
    ) -> Dict[str, float]:
        """Extract and validate plot dimensions."""
        dims = reqs.plot_dimensions

        if dims and dims.width and dims.length:
            w = float(dims.width)
            l = float(dims.length)
        elif dims and dims.total_area_sqft:
            # Area-only: assume a typical 2:3 ratio
            area = float(dims.total_area_sqft)
            l = round(math.sqrt(area * 1.5), 1)
            w = round(area / l, 1)
            warnings.append(
                f"Plot dimensions inferred from area ({area} sqft) as "
                f"{w}×{l} ft (assumed 2:3 aspect ratio)."
            )
        else:
            # Absolute fallback — warn strongly
            w, l = 30.0, 40.0
            warnings.append(
                "No plot dimensions found. Defaulting to 30×40 ft. "
                "Results may not be accurate."
            )

        area = round(w * l, 2)
        return {"width": round(w, 2), "length": round(l, 2), "area": area}

    # ── Step 2: Setbacks ──────────────────────────────────────────────────

    def _resolve_setbacks(
        self,
        reqs:      BuildingRequirements,
        bundle:    KnowledgeBundle,
        plot_area: float,
    ) -> Setbacks:
        """
        Priority: user-specified > NBC bundle recommendations > NBC defaults.
        """
        user_sb = reqs.setbacks
        nbc_sb  = bundle.setbacks_recommended

        def _pick(field: str, fallback: float) -> float:
            user_val = getattr(user_sb, field, None) if user_sb else None
            nbc_val  = getattr(nbc_sb,  field, None) if nbc_sb  else None
            return float(user_val if user_val is not None
                         else (nbc_val if nbc_val is not None else fallback))

        # NBC defaults for common Indian plot sizes (ft)
        if plot_area <= 600:
            df = {"front": 3.0, "rear": 1.5, "left": 1.5, "right": 1.5}
        elif plot_area <= 1200:
            df = {"front": 4.0, "rear": 2.0, "left": 2.0, "right": 2.0}
        elif plot_area <= 2400:
            df = {"front": 5.0, "rear": 3.0, "left": 2.5, "right": 2.5}
        else:
            df = {"front": 6.0, "rear": 3.0, "left": 3.0, "right": 3.0}

        return Setbacks(
            front = _pick("front", df["front"]),
            rear  = _pick("rear",  df["rear"]),
            left  = _pick("left",  df["left"]),
            right = _pick("right", df["right"]),
            unit  = "ft",
        )

    # ── Step 3: Orientation ───────────────────────────────────────────────

    def _resolve_orientation(
        self, reqs: BuildingRequirements
    ) -> Tuple[str, str]:
        """
        Determine entrance_direction and north_direction from plot_context.

        Returns (entrance_direction, north_direction) as uppercase compass strings:
          N | NE | E | SE | S | SW | W | NW

        north_direction semantics:
          "Which physical side of the plot faces geographic North?"
          • If the user explicitly provided north_direction → use it directly.
          • Otherwise default to "N", meaning "we assume the northern side of
            this plot faces geographic North" — the standard assumption when the
            user has not told us the site orientation.
          • We deliberately do NOT derive north as the opposite of the entrance
            direction; that approach would silently impose a specific site
            orientation that may be factually wrong and would corrupt Vastu
            compass calculations downstream.
        """
        ctx = reqs.plot_context

        def _extract(direction_val) -> Optional[str]:
            if direction_val is None:
                return None
            raw = str(direction_val.value if hasattr(direction_val, "value")
                      else direction_val).lower()
            MAP = {
                "north": "N", "south": "S", "east": "E", "west": "W",
                "north_east": "NE", "north_west": "NW",
                "south_east": "SE", "south_west": "SW",
            }
            return MAP.get(raw, raw.upper()[:2])

        ent_dir   = _extract(ctx.entrance_side if ctx else None)
        north_dir = _extract(ctx.north_direction if ctx else None)

        # Infer entrance from road_facing_sides if not explicitly set
        if ent_dir is None and ctx and ctx.road_facing_sides:
            ent_dir = _extract(ctx.road_facing_sides[0])

        # north_direction: use "N" as the universal default when not given.
        # Geographic North is independent of which side the road/entrance faces.
        if north_dir is None:
            north_dir = "N"

        return (ent_dir or "N"), north_dir

    # ── Step 4: Sizes ────────────────────────────────────────────────────

    def _assign_sizes(
        self,
        resolved: List[ResolvedRoom],
        bundle:   KnowledgeBundle,
        warnings: List[str],
    ) -> List[EnrichedRoom]:
        """
        Build EnrichedRoom objects with NBC-clamped sizes from bundle stats.
        Falls back to Indian-specific defaults for rooms not in CubiCasa5K.
        """
        enriched: List[EnrichedRoom] = []

        for res in resolved:
            rt   = res.room_type
            nbc  = get_room_minimums(rt)
            dist = bundle.room_size_distributions.get(rt)

            if dist:
                # Use median from matched plans, enforcing NBC minimums
                target_area  = max(dist.median_area_sqft, nbc["min_area_sqft"])
                target_width = max(dist.median_width_ft,  nbc["min_width_ft"])
                max_area     = max(dist.p75_area_sqft * 1.3, nbc["min_area_sqft"])
                # Warn when the distribution has no real data (sample_count=0)
                # These sizes are fabricated/estimated, not from real plan observations.
                sample_count = getattr(dist, "sample_count", None)
                if sample_count is not None and sample_count == 0:
                    warnings.append(
                        f"⚠ Room '{rt}': size statistics have sample_count=0 — "
                        f"dimensions are estimated (not from real plan data). "
                        f"Target area {round(target_area, 1)} sqft may be inaccurate."
                    )
            else:
                # Rule book default (covers Indian-specific rooms + NBC fallback)
                india_default = _get_india_default(rt)
                if india_default:
                    def_w, def_a = india_default
                    target_width = max(def_w, nbc["min_width_ft"])
                    target_area  = max(def_a, nbc["min_area_sqft"])
                    max_area     = target_area * 1.5
                else:
                    # Pure NBC fallback — no rule book entry exists
                    target_width = nbc["min_width_ft"] * 1.2
                    target_area  = nbc["min_area_sqft"] * 1.25
                    max_area     = nbc["min_area_sqft"] * 2.0
                    warnings.append(
                        f"No size data for '{rt}' — using NBC minimums × 1.25."
                    )

            target_length = max(
                round(target_area / max(target_width, 0.1), 2),
                nbc["min_width_ft"]   # length must also meet min_width (square rooms ok)
            )

            ceiling = _get_ceiling(rt)
            door_w  = get_door_width(rt)
            hab     = is_habitable(rt)

            # Respect user-specified preferred_floor from Step 1 parser
            pref_floor = res.preferred_floor if res.preferred_floor is not None else 0

            enriched.append(EnrichedRoom(
                room_id          = res.room_id,
                room_type        = rt,
                display_name     = res.display_name,
                quantity_index   = res.quantity_index,
                implicit_room    = res.implicit_room,
                target_width_ft  = round(target_width,  2),
                target_length_ft = round(target_length, 2),
                target_area_sqft = round(target_area,   2),
                min_width_ft     = round(nbc["min_width_ft"],  2),
                min_length_ft    = round(nbc["min_width_ft"],  2),  # NBC uses width for both
                min_area_sqft    = round(nbc["min_area_sqft"], 2),
                max_area_sqft    = round(max_area, 2),
                ceiling_height_ft = ceiling,
                preferred_floor  = pref_floor,
                preferred_zone   = "middle",   # will be set in _assign_zones
                preferred_direction = "N",      # will be set in _assign_zones
                is_habitable     = hab,
                needs_exterior_wall = hab,
                door_width_ft    = door_w,
                user_specific_requirements = res.user_specific_requirements,
            ))

        return enriched

    # ── Step 5: Floor assignment ──────────────────────────────────────────

    def _assign_floors(
        self,
        enriched:       List[EnrichedRoom],
        reqs:           BuildingRequirements,
        bundle:         KnowledgeBundle,
        floors:         int,
        entrance_dir:   str,
        vastu_enabled:  bool,
        vastu_mapper:   Optional[VastuMapper],
        decisions:      List[Dict[str, str]],
        warnings:       List[str],
        user_floor_map: Optional[Dict[str, Optional[int]]] = None,
    ) -> List[EnrichedRoom]:
        """
        Assign preferred_floor to every room.

        Priority chain (strictly ordered — first match wins, no override):
          1. ABSOLUTE — user explicitly specified this floor in their prompt.
             Captured via user_floor_map from the original ResolvedRoom values.
             This CANNOT be overridden by Vastu, Gemini, statistics, or heuristics.
          2. Vastu hard constraint — "ground_floor_only" or "top_floor_preferred".
             Only applies to rooms where the user did NOT set an explicit floor.
          3. Gemini AI reasoning (multi-floor plans, optional, API-dependant).
             Vastu "ground_floor_only" still overrides Gemini suggestions.
          4. Bundle statistical floor suggestions from matched plans.
          5. _FLOOR_HEURISTICS — Indian residential defaults.
          6. Ground floor — absolute fallback (0).
        """
        if floors == 1:
            for room in enriched:
                room.preferred_floor = 0
            return enriched

        # Build the user-locked set from the resolver snapshot.
        # Only room_ids where the user explicitly provided a floor (non-None) are locked.
        user_locked: Dict[str, int] = {}
        if user_floor_map:
            user_locked = {
                room_id: floor
                for room_id, floor in user_floor_map.items()
                if floor is not None
            }

        # Gemini call — only for rooms WITHOUT a user-locked floor so we don't
        # waste tokens asking about already-decided rooms.
        gemini_floors: Dict[str, int] = {}
        if self._use_gemini:
            gemini_floors = self._get_gemini_floor_distribution(
                enriched, reqs, floors, entrance_dir, vastu_enabled,
                decisions, warnings,
            )

        for room in enriched:

            # ── Priority 1 (ABSOLUTE): user explicitly specified this floor ──
            # Captured from resolver before any defaulting happened.
            if room.room_id in user_locked:
                room.preferred_floor = min(user_locked[room.room_id], floors - 1)
                log.debug("Floor locked (user): %s → %d", room.room_id,
                          room.preferred_floor)
                continue   # nothing can override this

            # ── Priority 2: Vastu hard floor constraints ──
            if vastu_enabled and vastu_mapper:
                vc = vastu_mapper.get_vastu_constraint(room.room_type)
                if vc:
                    if vc.floor_preference == "ground_floor_only":
                        room.preferred_floor = 0
                        continue
                    elif vc.floor_preference == "top_floor_preferred":
                        room.preferred_floor = floors - 1
                        continue
                    # "any_floor" → fall through

            # ── Priority 3: Gemini AI reasoning ──
            if room.room_id in gemini_floors:
                g_floor = min(gemini_floors[room.room_id], floors - 1)
                # Vastu ground_floor_only still beats Gemini
                if vastu_enabled and vastu_mapper:
                    vc = vastu_mapper.get_vastu_constraint(room.room_type)
                    if vc and vc.floor_preference == "ground_floor_only":
                        room.preferred_floor = 0
                    else:
                        room.preferred_floor = g_floor
                else:
                    room.preferred_floor = g_floor
                continue

            # ── Priority 4: Bundle statistical floor suggestions ──
            stat_floor = bundle.floor_distribution_suggestions.get(room.display_name)
            if stat_floor is not None:
                room.preferred_floor = min(stat_floor, floors - 1)
                continue

            # ── Priority 5: Rule book floor assignment (data + convention) ──
            room.preferred_floor = _get_floor_heuristic(room.room_type, floors)

        return enriched

    def _scale_rooms_to_fit(
        self,
        enriched: List[EnrichedRoom],
        net_w:    float,
        net_l:    float,
        floors:   int,
        warnings: List[str],
    ) -> List[EnrichedRoom]:
        """
        Scale room target sizes DOWN when the total area on a floor exceeds the
        net buildable area, while never dropping below NBC minimum dimensions.

        Rationale:
          • The matched-plan statistics come from a Western/international dataset
            where typical room sizes are larger than those in Indian urban plots.
          • A 30×40 ft plot has only 768 sqft net buildable per floor (after NBC
            setbacks), so room targets derived from 4BHK western plans will
            routinely exceed this.
          • We use 85% of net_buildable_area as the target coverage ceiling,
            leaving 15% for load-bearing walls (~6" thick), structural columns,
            construction tolerances, and door/window reveals.
          • Scaling is proportional — every room on the crowded floor shrinks by
            the same factor — then NBC minimums are re-applied as a hard floor.
          • Car parking (external marked area) is excluded from building coverage
            calculations because it sits in the front setback, not inside the
            structural envelope.

        Args:
            enriched:  Enriched rooms with floor assignments already set.
            net_w:     Net buildable width in ft (plot width minus setbacks).
            net_l:     Net buildable length in ft (plot length minus setbacks).
            floors:    Total number of floors.
            warnings:  Mutable list — scaling events are appended here.

        Returns:
            Same list with updated target_area_sqft / target_width_ft /
            target_length_ft on rooms that needed scaling.
        """
        # Rooms physically external to the building envelope are excluded from
        # the per-floor coverage check.
        _EXTERNAL_TYPES = {"car_parking", "garden", "terrace", "verandah",
                           "balcony", "barsati"}

        net_area_per_floor = net_w * net_l

        # Target: leave 15% for walls, columns, and structural elements.
        # In practice Indian construction uses ~12-15% of floor plate for structure.
        _MAX_COVERAGE_RATIO = 0.85
        max_usable = net_area_per_floor * _MAX_COVERAGE_RATIO

        for floor_num in range(floors):
            # Only internal rooms count towards structural coverage
            internal = [
                r for r in enriched
                if r.preferred_floor == floor_num
                and r.room_type not in _EXTERNAL_TYPES
            ]
            if not internal:
                continue

            total = sum(r.target_area_sqft for r in internal)
            if total <= max_usable:
                continue  # floor fits — no scaling needed

            scale = max_usable / total

            scaled_total = 0.0
            for room in internal:
                new_area  = max(round(room.target_area_sqft * scale, 2),
                                room.min_area_sqft)
                new_width = max(round(room.target_width_ft * math.sqrt(scale), 2),
                                room.min_width_ft)
                new_len   = max(round(new_area / max(new_width, 0.1), 2),
                                room.min_length_ft)

                room.target_area_sqft  = new_area
                room.target_width_ft   = new_width
                room.target_length_ft  = new_len
                scaled_total          += new_area

            log.info(
                "Floor %d scaled: %d rooms, %.1f → %.1f sqft "
                "(net=%.1f sqft, coverage target=%.1f sqft).",
                floor_num, len(internal), total, scaled_total,
                net_area_per_floor, max_usable,
            )
            warnings.append(
                f"Floor {floor_num}: room areas scaled from {total:.1f} to "
                f"{scaled_total:.1f} sqft to fit net buildable area "
                f"({net_area_per_floor:.1f} sqft, target ≤{max_usable:.1f} sqft). "
                f"NBC minimums preserved."
            )

        return enriched

    def _get_gemini_floor_distribution(
        self,
        rooms:         List[EnrichedRoom],
        reqs:          BuildingRequirements,
        floors:        int,
        entrance_dir:  str,
        vastu_enabled: bool,
        decisions:     List[Dict[str, str]],
        warnings:      List[str],
    ) -> Dict[str, int]:
        """
        Call Gemini to reason about optimal floor distribution.
        Returns dict of room_id → floor_number, empty on any failure.
        """
        try:
            from google import genai
            from google.genai import types
            from sources.key_rotator import GeminiKeyRotator
            from docs.prompts.loader import load_prompt

            if self._key_rotator is None:
                self._key_rotator = GeminiKeyRotator(cooldown_seconds=60.0)

            system_prompt = load_prompt("step3_enricher_system.md")

            # Build room list text
            dims = reqs.plot_dimensions
            w = dims.width if (dims and dims.width) else 30
            l = dims.length if (dims and dims.length) else 40

            room_lines = []
            for r in rooms:
                note = ""
                if r.user_specific_requirements:
                    note = f" [user: {r.user_specific_requirements}]"
                room_lines.append(f"  - {r.room_id} ({r.display_name}){note}")

            user_prompt = (
                f"Plot: {w}×{l} ft | Floors: {floors} (0=Ground, 1=First"
                + (", 2=Second" if floors > 2 else "") + ")\n"
                f"Entrance: {entrance_dir}-facing | Vastu: {'enabled' if vastu_enabled else 'disabled'}\n"
                f"BHK type: inferred from {len([r for r in rooms if 'bedroom' in r.room_type])} bedrooms\n\n"
                f"Rooms to distribute across {floors} floors:\n"
                + "\n".join(room_lines)
                + "\n\nReturn ONLY a JSON object: {\"room_id\": floor_number, ...}"
            )

            for attempt in range(3):
                client, slot_idx = self._key_rotator.get_client()
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=user_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=0.05,
                            response_mime_type="application/json",
                            max_output_tokens=600,
                        ),
                    )
                    raw = response.text.strip()
                    # Strip any accidental markdown fences
                    if raw.startswith("```"):
                        raw = raw.split("```")[1]
                        if raw.startswith("json"):
                            raw = raw[4:]
                    result: Dict[str, int] = json.loads(raw)
                    # Validate: all values are valid floor numbers
                    result = {
                        k: max(0, min(int(v), floors - 1))
                        for k, v in result.items()
                        if isinstance(v, (int, float))
                    }
                    decisions.append({
                        "decision":    "floor_distribution",
                        "reasoning":   f"Gemini assigned {len(result)} rooms across "
                                       f"{floors} floors (entrance={entrance_dir}, "
                                       f"vastu={'on' if vastu_enabled else 'off'})",
                        "confidence":  "high",
                    })
                    log.info("Gemini floor distribution: %d room assignments", len(result))
                    return result

                except Exception as e:
                    err = str(e)
                    if "429" in err or ("resource" in err.lower() and "exhausted" in err.lower()):
                        self._key_rotator.report_rate_limited(slot_idx)
                        continue
                    log.warning("Gemini floor distribution failed (attempt %d): %s", attempt+1, e)
                    break

        except ImportError as e:
            log.debug("Gemini not available: %s", e)
        except Exception as e:
            log.warning("Gemini floor distribution error: %s", e)

        warnings.append(
            "Gemini floor distribution unavailable — using statistical/heuristic fallback."
        )
        return {}

    # ── Step 6: Zone + direction assignment ───────────────────────────────

    def _assign_zones(
        self,
        enriched:      List[EnrichedRoom],
        bundle:        KnowledgeBundle,
        entrance_dir:  str,
        vastu_enabled: bool,
        vastu_mapper:  Optional[VastuMapper],
    ) -> List[EnrichedRoom]:
        """Assign preferred_zone and preferred_direction to every room."""
        for room in enriched:
            rt         = room.room_type
            zone_probs = bundle.zone_probabilities.get(rt)

            if vastu_enabled and vastu_mapper:
                pref_dir, pref_zone = vastu_mapper.get_primary_direction_for_room(
                    rt, entrance_dir, zone_probs
                )
            elif zone_probs:
                # Non-Vastu: use highest-probability zone from matched plans
                best_zone = max(zone_probs, key=lambda z: zone_probs[z])
                pref_zone = best_zone
                # Convert zone to compass direction using entrance
                _ZONE_DIR = {
                    "E": {"front": "E",  "back": "W",  "left": "N",  "right": "S",  "middle": "E"},
                    "N": {"front": "N",  "back": "S",  "left": "W",  "right": "E",  "middle": "N"},
                    "S": {"front": "S",  "back": "N",  "left": "E",  "right": "W",  "middle": "S"},
                    "W": {"front": "W",  "back": "E",  "left": "S",  "right": "N",  "middle": "W"},
                }
                ent = entrance_dir.upper()
                if ent not in _ZONE_DIR:
                    ent = "N"
                pref_dir = _ZONE_DIR[ent].get(best_zone, ent)
            else:
                # Default: public rooms → front zone (entrance direction)
                #          private rooms → back zone
                _PUBLIC = {"living_room", "drawing_room", "dining_room", "foyer",
                           "passage", "car_parking", "kitchen", "pooja_room"}
                if rt in _PUBLIC:
                    pref_zone = "front"
                    pref_dir  = entrance_dir.upper()
                else:
                    pref_zone = "back"
                    _OPP = {"N":"S","S":"N","E":"W","W":"E","NE":"SW",
                            "SW":"NE","NW":"SE","SE":"NW"}
                    pref_dir = _OPP.get(entrance_dir.upper(), entrance_dir.upper())

            room.preferred_zone      = pref_zone
            room.preferred_direction = pref_dir

        return enriched

    # ── Step 7: Vastu constraints ─────────────────────────────────────────

    def _apply_vastu(
        self,
        enriched:     List[EnrichedRoom],
        vastu_mapper: VastuMapper,
    ) -> List[EnrichedRoom]:
        """Attach full VastuConstraint to each room, and merge adjacency prohibitions."""
        for room in enriched:
            constraint = vastu_mapper.get_vastu_constraint(room.room_type)
            if constraint:
                room.vastu = constraint
                # Merge should_not_be_adjacent_to from Vastu into room's list
                existing = set(room.should_not_be_adjacent_to)
                for rt in constraint.should_not_be_adjacent_to:
                    existing.add(rt)
                room.should_not_be_adjacent_to = list(existing)
        return enriched

    # ── Step 8: Bathroom link ────────────────────────────────────────────

    def _link_bathrooms(
        self,
        enriched:  List[EnrichedRoom],
        resolved:  List[ResolvedRoom],
    ) -> List[EnrichedRoom]:
        """
        Transfer bathroom attachment decisions from ResolvedRoom to EnrichedRoom.
        Also sets the bathroom's preferred_floor to match the bedroom's floor.
        """
        # Build room_id → EnrichedRoom for fast lookup
        id_map: Dict[str, EnrichedRoom] = {r.room_id: r for r in enriched}

        # Transfer attachment info from resolver
        for res in resolved:
            if res.attached_bath_room_id and res.room_id in id_map:
                bedroom = id_map[res.room_id]
                bedroom.attached_bathroom_id = res.attached_bath_room_id
                # Co-locate bathroom on the same floor as its bedroom
                if res.attached_bath_room_id in id_map:
                    bath = id_map[res.attached_bath_room_id]
                    bath.preferred_floor = bedroom.preferred_floor
                    bath.preferred_zone  = bedroom.preferred_zone

        return enriched

    # ── Step 9: Adjacency graph ───────────────────────────────────────────

    def _build_adjacency_graph(
        self,
        enriched: List[EnrichedRoom],
        bundle:   KnowledgeBundle,
    ) -> Dict[str, Dict[str, float]]:
        """
        Build room_id → room_id → weight adjacency graph.

        Weight sources (highest wins, merged additively):
          1. Attachment link (bedroom↔attached bathroom): fixed weight 10.0
          2. Bundle adjacency weights from matched real plans (Step 2 stats)
          3. Rule book adjacency weights (from 4,983 CubiCasa plans)
          4. Forbidden adjacency penalty (negative weight from rule book)

        Also populates EnrichedRoom.adjacency_preferences (type-level).
        """
        graph: Dict[str, Dict[str, float]] = {r.room_id: {} for r in enriched}

        # Build O(1) lookup: room_type → list of room_ids of that type
        type_to_ids: Dict[str, List[str]] = {}
        for r in enriched:
            type_to_ids.setdefault(r.room_type, []).append(r.room_id)

        # Build merged type-level weight map:
        # Start with bundle weights, then layer rule book weights on top.
        # Rule book wins when bundle has no data for a pair.
        def _merge_type_weights(rt: str) -> Dict[str, float]:
            merged: Dict[str, float] = {}
            # Layer 1: bundle
            for adj_type, w in bundle.adjacency_weights.get(rt, {}).items():
                merged[adj_type] = w
            # Layer 2: rule book (fills gaps and adds forbidden penalties)
            for a, b, w in _rules.get_all_adjacency_pairs():
                if a == rt and b not in merged and w > 0.3:
                    merged[b] = w
                elif b == rt and a not in merged and w > 0.3:
                    merged[a] = w
            # Layer 3: forbidden adjacency penalties (rule book)
            for rule in _rules.raw("forbidden_adjacencies", "rules") or []:
                ra, rb = rule.get("room_a"), rule.get("room_b")
                penalty = rule.get("penalty_multiplier", 0.0)
                if ra == rt and penalty < 0:
                    merged[rb] = penalty     # negative weight = repel
                elif rb == rt and penalty < 0:
                    merged[ra] = penalty
            return merged

        for room in enriched:
            rt_prefs: Dict[str, float] = {}
            type_weights = _merge_type_weights(room.room_type)

            for adj_type, weight in type_weights.items():
                if abs(weight) < 0.3:
                    continue
                rt_prefs[adj_type] = weight

                # Expand to instance-level graph
                for adj_id in type_to_ids.get(adj_type, []):
                    if adj_id != room.room_id:
                        instance_weight = weight * (0.9 if len(
                            type_to_ids.get(adj_type, [])) > 1 else 1.0)
                        # Take highest magnitude (keep penalties negative)
                        existing = graph[room.room_id].get(adj_id, 0.0)
                        if abs(instance_weight) > abs(existing):
                            graph[room.room_id][adj_id] = round(instance_weight, 3)

            # Attachment is the strongest adjacency (weight=10)
            if room.attached_bathroom_id:
                graph[room.room_id][room.attached_bathroom_id] = 10.0
                graph.setdefault(room.attached_bathroom_id, {})[room.room_id] = 10.0

            room.adjacency_preferences = rt_prefs

        return graph

    # ── Step 6b: Vastu floor conflict audit ───────────────────────────────

    def _audit_vastu_floor_conflicts(
        self,
        enriched:       List[EnrichedRoom],
        user_floor_map: Dict[str, Optional[int]],
        floors:         int,
        vastu_mapper:   VastuMapper,
        warnings:       List[str],
    ) -> None:
        """
        Post-assignment audit: detect rooms where the user explicitly set a floor
        that conflicts with the Vastu floor_preference for that room type.

        This does NOT change the floor assignment — user intent always wins.
        It only surfaces actionable warnings so the user can make an informed
        decision about the Vastu trade-off.

        Cases flagged:
          • room has Vastu floor_preference="top_floor_preferred" but is on
            Ground Floor (user placed it there explicitly).
          • room has Vastu floor_preference="ground_floor_only" but is on an
            upper floor (user placed it there explicitly).
        """
        # Only flag rooms where the user explicitly locked a floor
        user_locked_ids = {
            room_id for room_id, floor in user_floor_map.items()
            if floor is not None
        }

        for room in enriched:
            if room.room_id not in user_locked_ids:
                continue  # not user-locked, no conflict to report

            vc = vastu_mapper.get_vastu_constraint(room.room_type)
            if not vc:
                continue

            pref = vc.floor_preference
            actual = room.preferred_floor

            if pref == "top_floor_preferred" and actual == 0 and floors > 1:
                warnings.append(
                    f"⚠ Vastu conflict: '{room.display_name}' is on Ground Floor "
                    f"(user-specified), but Vastu recommends the top floor "
                    f"(SW corner of Floor {floors - 1} is ideal for {room.room_type}). "
                    f"Consider moving it upstairs for better Vastu compliance."
                )
                log.debug("Vastu floor conflict: %s — user=GF, vastu=top_floor_preferred",
                          room.room_id)

            elif pref == "ground_floor_only" and actual > 0:
                warnings.append(
                    f"⚠ Vastu conflict: '{room.display_name}' is on Floor {actual} "
                    f"(user-specified), but Vastu requires ground floor only for "
                    f"{room.room_type}. This violates a hard Vastu rule."
                )
                log.debug("Vastu floor conflict: %s — user=Floor%d, vastu=ground_floor_only",
                          room.room_id, actual)

    # ── Step 10: Floor plans ──────────────────────────────────────────────

    def _build_floor_plans(
        self,
        enriched: List[EnrichedRoom],
        floors:   int,
    ) -> List[FloorPlan]:
        """
        Build one FloorPlan per floor with room IDs and gross area.

        gross_area_sqft = sum of INTERNAL rooms only (within building envelope).
        External rooms (car_parking, garden, terrace, verandah, balcony, barsati)
        are listed separately so gross_area correctly reflects the structural
        floor plate — not the total of all objects on the site.
        """
        _LABELS = {0: "Ground Floor", 1: "First Floor", 2: "Second Floor", 3: "Third Floor"}
        _EXTERNAL_TYPES = {"car_parking", "garden", "terrace", "verandah",
                           "balcony", "barsati"}
        floor_plans = []

        for f in range(floors):
            rooms_on_floor    = [r for r in enriched if r.preferred_floor == f]
            internal_rooms    = [r for r in rooms_on_floor
                                 if r.room_type not in _EXTERNAL_TYPES]
            external_rooms    = [r for r in rooms_on_floor
                                 if r.room_type in _EXTERNAL_TYPES]
            # gross_area = internal building footprint only
            gross = round(sum(r.target_area_sqft for r in internal_rooms), 2)
            floor_plans.append(FloorPlan(
                floor_number    = f,
                floor_label     = _LABELS.get(f, f"Floor {f}"),
                room_ids        = [r.room_id for r in rooms_on_floor],
                gross_area_sqft = gross,
            ))

        return floor_plans

    # ── Helpers ──────────────────────────────────────────────────────────

    def _infer_bhk(self, bundle: KnowledgeBundle) -> int:
        """Infer BHK count from the most common BHK among matched plans."""
        if not bundle.matched_plans:
            return 3  # sensible default
        from collections import Counter
        _BHK_NUM = {"1bhk": 1, "1BHK": 1, "2bhk": 2, "2BHK": 2,
                    "3bhk": 3, "3BHK": 3, "4bhk": 4, "4BHK": 4, "4bhk+": 4}
        counts = Counter(
            _BHK_NUM.get(p.bhk, 3) for p in bundle.matched_plans
        )
        return counts.most_common(1)[0][0]

    def _estimate_compartmentalization(
        self,
        reqs:   BuildingRequirements,
        bundle: KnowledgeBundle,
    ) -> float:
        """
        Estimate compartmentalization score.
        Uses circulation_benchmarks from bundle, falls back to room count proxy.
        """
        # Try bundle's circulation metadata
        circ = bundle.circulation_benchmarks
        if circ:
            # compartmentalization_index is a key in circulation_meta.json
            val = circ.get("compartmentalization_index",
                  circ.get("compartmentalization", None))
            if val is not None:
                return float(val)

        # Proxy: more rooms → more compartmentalised
        total = sum(r.quantity for r in reqs.rooms)
        if total <= 3:
            return 0.35
        elif total <= 5:
            return 0.50
        elif total <= 8:
            return 0.65
        else:
            return 0.75
