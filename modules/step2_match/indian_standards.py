"""
indian_standards.py
===================
Hard-coded Indian National Building Code (NBC 2016) standards.

No JSON file needed — these rules are fixed and validated against
the NBC PDF. Used by the semantic matcher to overlay hard constraints
on top of data-derived statistics.

All dimensions in feet (ft). 1 ft = 304.8 mm.
Area in sqft.

Sources:
  National Building Code of India 2016, Part 4 — Fire & Life Safety
  NBC 2016, Part 3 — Development Control Rules
  SP 7: 2005 (reaffirmed 2018) — Room size minimums
"""

from typing import Any, Dict, Optional, Tuple


# =====================================================================
# ROOM SIZE MINIMUMS  (NBC 2016 Table 1, Part 3)
# All: (min_width_ft, min_length_ft, min_area_sqft)
# =====================================================================

ROOM_MINIMUMS: Dict[str, Tuple[float, float, float]] = {
    # Habitable rooms
    "master_bedroom":    (10.0, 10.0, 120.0),
    "bedroom":           (9.0,  9.0,  96.0),
    "bedroom_kids":      (8.0,  8.0,  64.0),
    "bedroom_guest":     (8.0,  8.0,  64.0),
    "living_room":       (10.0, 10.0, 120.0),
    "drawing_room":      (10.0, 10.0, 120.0),
    "dining_room":       (8.0,  8.0,  80.0),
    "kitchen":           (6.0,  8.0,  60.0),
    "study_room":        (8.0,  8.0,  64.0),

    # Non-habitable rooms
    "bathroom":          (4.0,  4.5,  20.0),
    "toilet":            (3.5,  4.0,  16.0),
    "combined_bathroom_toilet": (4.5, 5.0, 25.0),
    "pooja_room":        (4.0,  4.0,  16.0),
    "store_room":        (5.0,  5.0,  25.0),
    "utility_room":      (5.0,  5.0,  25.0),
    "servant_room":      (6.0,  6.0,  36.0),

    # Circulation & services
    "staircase":         (4.0,  5.0,  25.0),   # min width 1.2m per NBC
    "passage":           (3.5,  5.0,  17.5),   # min width 1.0m
    "foyer":             (5.0,  5.0,  25.0),
    "lobby":             (5.0,  6.0,  30.0),
    "balcony":           (4.0,  4.0,  16.0),
    "terrace":           (5.0,  5.0,  25.0),

    # Parking
    "car_parking":       (8.5,  16.0, 136.0),  # 2.6m × 5.0m per NBC
    "two_wheeler_parking": (4.0, 7.0, 28.0),
}

# Ceiling height minimums (ft)
CEILING_HEIGHTS: Dict[str, float] = {
    "habitable":     9.0,    # min 2.75m ≈ 9 ft for habitable rooms
    "kitchen":       8.5,    # min 2.6m
    "bathroom":      7.5,    # min 2.3m
    "staircase":     7.0,    # min between treads
    "garage":        7.0,
    "basement":      8.0,
}

# Ventilation minimums (window area as fraction of floor area)
VENTILATION_RATIO: Dict[str, float] = {
    "habitable": 1/6,   # NBC: 1/6th of floor area
    "kitchen":   1/6,
    "bathroom":  1/10,  # NBC: 1/10th for bathrooms
}


# =====================================================================
# SETBACK REQUIREMENTS  (NBC 2016 Table — typical residential)
# Keyed by plot_area_sqft ranges: (min_plot_sqft, max_plot_sqft)
# Values: {front_ft, rear_ft, side_ft}
# =====================================================================

SETBACK_TABLE: list = [
    # (min_area_sqft, max_area_sqft, front, rear, left_right)
    (0,      500,   3.0,  2.0,  2.0),
    (500,    900,   4.0,  2.5,  2.5),
    (900,    1350,  5.0,  3.0,  3.0),
    (1350,   2250,  5.0,  3.0,  3.0),
    (2250,   4500,  6.0,  4.0,  4.0),
    (4500,   9000,  8.0,  5.0,  5.0),
    (9000,   float("inf"), 10.0, 6.0, 6.0),
]


def get_setbacks(plot_area_sqft: float) -> Dict[str, float]:
    """
    Return NBC-recommended setbacks in ft for a given plot area.
    Returns front, rear, left, right setbacks.
    """
    for min_a, max_a, front, rear, side in SETBACK_TABLE:
        if min_a <= plot_area_sqft < max_a:
            return {"front": front, "rear": rear, "left": side, "right": side}
    # Default for very large plots
    return {"front": 10.0, "rear": 6.0, "left": 6.0, "right": 6.0}


# =====================================================================
# FLOOR AREA RATIO (FAR) & GROUND COVERAGE
# =====================================================================

# FAR by zone type (typical Indian municipality)
FAR_BY_ZONE: Dict[str, float] = {
    "residential_low":    1.5,
    "residential_medium": 2.0,
    "residential_high":   2.5,
    "commercial":         3.0,
    "default":            1.5,
}

# Maximum ground coverage (fraction of plot)
MAX_GROUND_COVERAGE = 0.60   # 60% of plot area can be built on

# Maximum number of floors for residential
MAX_FLOORS_RESIDENTIAL = 3   # G+2


# =====================================================================
# STAIRCASE STANDARDS  (NBC 2016)
# =====================================================================

STAIRCASE_STANDARDS: Dict[str, Any] = {
    "min_width_ft":         4.0,    # 1.2m minimum
    "min_tread_depth_ft":   0.83,   # 250mm = 0.83ft
    "max_riser_height_ft":  0.59,   # 180mm = 0.59ft
    "min_headroom_ft":      7.0,    # 2.1m
    "max_rise_per_flight":  16,     # max steps before landing
    "landing_min_ft":       4.0,    # minimum landing size
}


# =====================================================================
# DOOR & WINDOW STANDARDS  (NBC 2016)
# =====================================================================

DOOR_WIDTHS: Dict[str, float] = {
    "main_entrance":   3.5,     # ft  (1050mm)
    "bedroom":         2.83,    # ft  (860mm)
    "bathroom":        2.33,    # ft  (710mm)
    "kitchen":         2.5,     # ft  (760mm)
    "utility":         2.0,     # ft  (610mm)
    "passage":         2.5,     # ft
    "default":         2.5,     # ft
}

WINDOW_SIZES: Dict[str, Tuple[float, float]] = {
    # (width_ft, height_ft)
    "bedroom":      (4.0, 4.0),
    "living_room":  (5.0, 4.5),
    "kitchen":      (3.0, 3.5),
    "bathroom":     (2.0, 2.5),
    "study_room":   (3.5, 4.0),
    "default":      (3.5, 4.0),
}

WINDOW_SILL_HEIGHT_FT: float = 2.83   # 860mm from floor


# =====================================================================
# FIRE SAFETY REQUIREMENTS  (NBC Part 4)
# =====================================================================

FIRE_SAFETY: Dict[str, Any] = {
    "max_travel_distance_to_exit_ft": 98.4,    # 30m for residential
    "min_corridor_width_for_exit_ft": 3.28,    # 1.0m
    "smoke_detector_spacing_ft":      32.8,    # 10m spacing
    "fire_extinguisher_spacing_ft":   82.0,    # 25m spacing
    "sprinkler_required_above_floors": 4,      # G+3 and above
}


# =====================================================================
# CONVENIENCE HELPERS
# =====================================================================

def get_room_minimums(room_type: str) -> Dict[str, float]:
    """
    Get NBC minimum dimensions for a room type.
    Returns: {min_width_ft, min_length_ft, min_area_sqft}
    Falls back to bedroom minimums if room not found.
    """
    norm = room_type.lower().replace(" ", "_")

    # Direct match
    if norm in ROOM_MINIMUMS:
        w, l, a = ROOM_MINIMUMS[norm]
        return {"min_width_ft": w, "min_length_ft": l, "min_area_sqft": a}

    # Fuzzy match
    for key in ROOM_MINIMUMS:
        if key in norm or norm in key:
            w, l, a = ROOM_MINIMUMS[key]
            return {"min_width_ft": w, "min_length_ft": l, "min_area_sqft": a}

    # Ultimate fallback
    return {"min_width_ft": 8.0, "min_length_ft": 8.0, "min_area_sqft": 64.0}


def is_habitable(room_type: str) -> bool:
    """
    Returns True if a room type is classified as habitable under NBC.
    Habitable rooms require natural ventilation + min ceiling height 9ft.
    """
    HABITABLE = {
        "bedroom", "master_bedroom", "bedroom_kids", "bedroom_guest",
        "living_room", "drawing_room", "dining_room", "kitchen",
        "study_room", "servant_room",
    }
    norm = room_type.lower().replace(" ", "_")
    return any(h in norm for h in HABITABLE)


def get_door_width(room_type: str) -> float:
    """Get NBC standard door width in ft for a given room type."""
    norm = room_type.lower().replace(" ", "_")
    for key, width in DOOR_WIDTHS.items():
        if key in norm:
            return width
    return DOOR_WIDTHS["default"]


def nbc_summary() -> Dict[str, Any]:
    """Return a summary of key NBC standards for inclusion in KnowledgeBundle."""
    return {
        "room_minimums":        {k: {"min_width_ft": v[0], "min_length_ft": v[1],
                                     "min_area_sqft": v[2]}
                                 for k, v in ROOM_MINIMUMS.items()},
        "ceiling_heights":      CEILING_HEIGHTS,
        "ventilation_ratios":   VENTILATION_RATIO,
        "max_ground_coverage":  MAX_GROUND_COVERAGE,
        "max_floors":           MAX_FLOORS_RESIDENTIAL,
        "staircase":            STAIRCASE_STANDARDS,
        "door_widths":          DOOR_WIDTHS,
        "fire_safety":          FIRE_SAFETY,
        "source":               "NBC 2016 (National Building Code of India)",
    }
