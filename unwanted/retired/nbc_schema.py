"""
National Building Code (NBC) Extraction — Pydantic Output Schema
================================================================
Production-grade schema defining the COMPLETE structured output
for all Indian residential building standards extracted from the
National Building Code of India (NBC 2016 / BIS IS 1893, SP 7).

These models serve as both:
  1. The Gemini structured output schema (response_schema=...)
  2. The runtime validation layer for extracted data

Every field has a description that doubles as extraction guidance
for the LLM — it tells Gemini exactly WHAT to look for in the PDF.

Design Principles:
  - All dimensional values normalized to METRIC (meters, sqm) as primary
  - Imperial equivalents stored separately for display (feet, sqft)
  - Every rule has a `source_clause` field for traceability back to NBC
  - Confidence scoring: "extracted" (from PDF), "inferred" (from context),
    "default" (hardcoded fallback if NBC is silent on the topic)
  - Exhaustive coverage of all categories relevant to residential floor
    plan generation (Steps 3-8 of the PlanGen pipeline)
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


# =====================================================================
# ENUMS
# =====================================================================

class DataConfidence(str, Enum):
    """How the data was obtained."""
    EXTRACTED = "extracted"    # Directly found in NBC PDF text/table
    INFERRED = "inferred"     # Logically derived from NBC context
    DEFAULT = "default"       # Fallback — NBC silent, using industry standard


class ConstraintSeverity(str, Enum):
    """How strictly a rule must be enforced."""
    MANDATORY = "mandatory"   # Hard rule — violation = illegal building
    RECOMMENDED = "recommended"  # Should follow — best practice
    GUIDELINE = "guideline"   # Suggestion — may vary by local authority


class PlotCategory(str, Enum):
    """Plot size categories commonly used in Indian building codes."""
    SMALL = "small"           # < 100 sqm (< 1076 sqft)
    MEDIUM = "medium"         # 100-300 sqm (1076-3229 sqft)
    LARGE = "large"           # 300-500 sqm (3229-5382 sqft)
    VERY_LARGE = "very_large" # > 500 sqm (> 5382 sqft)


class OccupancyType(str, Enum):
    """Building occupancy classification relevant to residential."""
    RESIDENTIAL_LOW = "residential_low_rise"    # Up to G+2
    RESIDENTIAL_MID = "residential_mid_rise"    # G+3 to G+6
    RESIDENTIAL_HIGH = "residential_high_rise"  # G+7 and above
    MIXED_USE = "mixed_use"                     # Residential + commercial


# =====================================================================
# ATOMIC RULE MODELS
# =====================================================================

class DimensionalValue(BaseModel):
    """
    A single dimensional measurement with both metric and imperial values.
    This is the atomic unit for all size-related standards.
    """
    value_m: Optional[float] = Field(
        None,
        description=(
            "Value in meters (for lengths/widths/heights) or square meters "
            "(for areas). This is the PRIMARY unit used in NBC."
        )
    )
    value_ft: Optional[float] = Field(
        None,
        description="Equivalent value in feet or square feet for display."
    )
    unit: str = Field(
        "m",
        description=(
            "Unit type: 'm' for meters, 'mm' for millimeters, 'sqm' for "
            "square meters, 'sqft' for square feet."
        )
    )

    def to_feet(self) -> Optional[float]:
        """Convert meters to feet."""
        if self.value_m is not None:
            if self.unit in ("m",):
                return round(self.value_m * 3.28084, 2)
            elif self.unit in ("mm",):
                return round(self.value_m * 0.00328084, 4)
            elif self.unit in ("sqm",):
                return round(self.value_m * 10.7639, 2)
        return self.value_ft


class RoomMinimumSpec(BaseModel):
    """
    Minimum specification for a specific room type as per NBC.

    These are the ABSOLUTE MINIMUMS — the floor plan generator must
    ensure every room meets or exceeds these values.
    """
    room_type: str = Field(
        ...,
        description=(
            "Room type exactly as referenced in NBC. Map to standard types: "
            "'bedroom', 'master_bedroom', 'kitchen', 'living_room', "
            "'dining_room', 'bathroom', 'toilet', 'combined_bathroom', "
            "'drawing_room', 'study_room', 'store_room', 'pooja_room', "
            "'utility_room', 'servant_room', 'garage', 'balcony', "
            "'corridor', 'passage', 'foyer', 'staircase_room'."
        )
    )
    min_area: Optional[DimensionalValue] = Field(
        None,
        description=(
            "Minimum floor area required by NBC for this room type. "
            "For bedrooms NBC typically specifies 9.5 sqm (102 sqft), "
            "for kitchens around 5.0-5.5 sqm (54-59 sqft). "
            "Extract the EXACT value from the code."
        )
    )
    min_width: Optional[DimensionalValue] = Field(
        None,
        description=(
            "Minimum width (shorter dimension) of the room. "
            "NBC often specifies this separately from area — e.g., "
            "a bedroom must be at least 2.4m (8ft) wide even if area is met. "
            "For kitchens, typically 1.8-2.1m (6-7ft)."
        )
    )
    min_length: Optional[DimensionalValue] = Field(
        None,
        description=(
            "Minimum length (longer dimension) of the room, if specified "
            "separately from width in NBC. Often not specified — room just "
            "needs to meet area + width minimums."
        )
    )
    min_ceiling_height: Optional[DimensionalValue] = Field(
        None,
        description=(
            "Minimum ceiling height (floor to ceiling) for this room type. "
            "NBC typically requires 2.75m (9ft) for habitable rooms, "
            "2.4m (8ft) for kitchens/bathrooms. Extract exact values."
        )
    )
    max_aspect_ratio: Optional[float] = Field(
        None,
        description=(
            "Maximum length-to-width ratio allowed. NBC sometimes specifies "
            "that rooms should not be too elongated — e.g., max 2:1 for "
            "habitable rooms. Extract if mentioned."
        )
    )
    ventilation_required: bool = Field(
        True,
        description=(
            "Whether this room type requires natural ventilation (external "
            "wall with window). Generally True for all habitable rooms "
            "(bedroom, living, kitchen), False for store rooms, corridors."
        )
    )
    natural_light_required: bool = Field(
        True,
        description=(
            "Whether this room type requires natural lighting. Same as "
            "ventilation for most rooms, but toilets may need ventilation "
            "without natural light (mechanical ventilation allowed)."
        )
    )
    notes: Optional[str] = Field(
        None,
        description=(
            "Any additional notes or exceptions for this room type. "
            "E.g., 'at least one bedroom must be >= 12 sqm', "
            "'kitchen in EWS may be reduced to 3.3 sqm'."
        )
    )
    source_clause: Optional[str] = Field(
        None,
        description=(
            "NBC clause/section reference. E.g., 'Part 3, Section 2, "
            "Table 2', 'Clause 8.3.2.1'. Critical for traceability."
        )
    )
    confidence: DataConfidence = Field(
        DataConfidence.EXTRACTED,
        description="How this data was obtained."
    )


class VentilationStandard(BaseModel):
    """
    Ventilation and natural lighting requirements from NBC.

    These are CRITICAL for the floor plan generator — every habitable
    room MUST have access to an exterior wall for ventilation. The
    validator (Step 6) checks these rules.
    """
    room_type: str = Field(
        ...,
        description="Room type this standard applies to."
    )
    min_window_area_ratio: Optional[float] = Field(
        None,
        description=(
            "Minimum window/opening area as a FRACTION of floor area. "
            "NBC typically requires 1/6th (0.167) for habitable rooms, "
            "1/10th (0.1) for kitchens. Extract exact ratios."
        )
    )
    min_ventilation_area_ratio: Optional[float] = Field(
        None,
        description=(
            "Minimum ventilation opening area as a fraction of floor area. "
            "May differ from window area — ventilation is the OPENABLE "
            "portion, while window includes fixed glazing."
        )
    )
    cross_ventilation_required: bool = Field(
        False,
        description=(
            "Whether cross-ventilation (openings on opposite or adjacent "
            "walls) is required. NBC may require this for kitchens and "
            "rooms above a certain size."
        )
    )
    mechanical_ventilation_allowed: bool = Field(
        False,
        description=(
            "Whether mechanical (exhaust fan) ventilation can substitute "
            "for natural ventilation. Typically True for bathrooms/toilets, "
            "False for habitable rooms."
        )
    )
    min_window_sill_height: Optional[DimensionalValue] = Field(
        None,
        description=(
            "Minimum height of window sill from floor level. NBC may specify "
            "this for safety — typically 0.75-0.9m (2.5-3ft)."
        )
    )
    min_window_head_height: Optional[DimensionalValue] = Field(
        None,
        description="Minimum height of top of window from floor level."
    )
    notes: Optional[str] = Field(None, description="Additional ventilation notes.")
    source_clause: Optional[str] = Field(None, description="NBC clause reference.")
    confidence: DataConfidence = Field(DataConfidence.EXTRACTED)


class SetbackRule(BaseModel):
    """
    Building setback requirements from plot boundaries.

    Setbacks vary by plot size, building height, road width, and
    local municipal bylaws. NBC provides base recommendations that
    local authorities may modify.
    """
    plot_category: Optional[PlotCategory] = Field(
        None,
        description="Plot size category this rule applies to."
    )
    plot_area_range_sqm: Optional[List[float]] = Field(
        None,
        description=(
            "Plot area range [min, max] in sqm that this rule applies to. "
            "E.g., [0, 100] for plots up to 100 sqm."
        )
    )
    front_setback: Optional[DimensionalValue] = Field(
        None,
        description=(
            "Minimum setback from front boundary (road side). "
            "Typically 3.0-4.5m for residential."
        )
    )
    rear_setback: Optional[DimensionalValue] = Field(
        None,
        description="Minimum setback from rear boundary. Typically 1.5-3.0m."
    )
    side_setback_left: Optional[DimensionalValue] = Field(
        None,
        description=(
            "Minimum setback from left side boundary. For narrow plots "
            "this may be 0 (party wall). Typically 1.0-1.5m."
        )
    )
    side_setback_right: Optional[DimensionalValue] = Field(
        None,
        description="Minimum setback from right side boundary."
    )
    corner_plot_adjustments: Optional[str] = Field(
        None,
        description=(
            "Special rules for corner plots (two road-facing sides). "
            "Often requires setbacks on both road-facing sides."
        )
    )
    height_dependency: Optional[str] = Field(
        None,
        description=(
            "How setbacks change with building height. E.g., "
            "'Add 0.5m for each additional floor above G+1'."
        )
    )
    notes: Optional[str] = Field(None, description="Additional setback notes.")
    source_clause: Optional[str] = Field(None, description="NBC clause reference.")
    confidence: DataConfidence = Field(DataConfidence.EXTRACTED)


class FARRule(BaseModel):
    """
    Floor Area Ratio and ground coverage regulations.

    FAR = Total built-up area across all floors / Plot area
    Coverage = Ground floor built-up area / Plot area

    These are CRITICAL constraints for the generator — they determine
    how much of the plot can be built upon and how many floors are
    economically viable.
    """
    plot_category: Optional[PlotCategory] = Field(
        None, description="Plot category this rule applies to."
    )
    plot_area_range_sqm: Optional[List[float]] = Field(
        None, description="Plot area range [min, max] in sqm."
    )
    road_width_range_m: Optional[List[float]] = Field(
        None,
        description=(
            "Abutting road width range [min, max] in meters. FAR often "
            "increases with wider road access."
        )
    )
    max_far: Optional[float] = Field(
        None,
        description=(
            "Maximum Floor Area Ratio allowed. E.g., 1.5 means total "
            "built-up area can be 1.5× the plot area. For residential "
            "low-rise, typically 1.0-2.0."
        )
    )
    max_ground_coverage_pct: Optional[float] = Field(
        None,
        description=(
            "Maximum ground coverage as a percentage. E.g., 60.0 means "
            "the ground floor footprint can cover up to 60% of the plot. "
            "Typically 50-75% for residential."
        )
    )
    max_height_m: Optional[DimensionalValue] = Field(
        None,
        description=(
            "Maximum building height allowed in meters for this FAR tier."
        )
    )
    max_floors: Optional[int] = Field(
        None,
        description="Maximum number of floors allowed for this configuration."
    )
    open_space_pct: Optional[float] = Field(
        None,
        description=(
            "Minimum open space as percentage of plot area. "
            "Open space = Plot area - Ground coverage area."
        )
    )
    notes: Optional[str] = Field(None, description="Additional FAR/coverage notes.")
    source_clause: Optional[str] = Field(None, description="NBC clause reference.")
    confidence: DataConfidence = Field(DataConfidence.EXTRACTED)


class StaircaseSpec(BaseModel):
    """
    Staircase dimensional specifications from NBC.

    Critical for multi-floor plan generation (G+1, G+2).
    Staircase must align across all floors — this defines its footprint.
    """
    staircase_type: str = Field(
        "internal_residential",
        description=(
            "Type: 'internal_residential', 'common_staircase', "
            "'external_fire_escape', 'spiral'."
        )
    )
    min_width: Optional[DimensionalValue] = Field(
        None,
        description=(
            "Minimum clear width of staircase flight. For internal "
            "residential: typically 0.9-1.0m (3-3.3ft). For common "
            "stairs in apartments: 1.2-1.5m."
        )
    )
    min_tread_depth: Optional[DimensionalValue] = Field(
        None,
        description=(
            "Minimum tread depth (going). NBC typically specifies "
            "250mm (10in) for residential, 300mm for public."
        )
    )
    max_riser_height: Optional[DimensionalValue] = Field(
        None,
        description=(
            "Maximum riser height. NBC typically specifies 190mm (7.5in) "
            "for residential, 150mm for public buildings."
        )
    )
    min_headroom: Optional[DimensionalValue] = Field(
        None,
        description=(
            "Minimum vertical clearance above any tread. "
            "Typically 2.1-2.2m (7ft)."
        )
    )
    min_landing_length: Optional[DimensionalValue] = Field(
        None,
        description=(
            "Minimum length of landing at top, bottom, and intermediate "
            "turns. Should equal staircase width."
        )
    )
    max_flight_without_landing: Optional[int] = Field(
        None,
        description=(
            "Maximum number of risers in a single flight before a landing "
            "is required. NBC typically says 12-15 risers max."
        )
    )
    handrail_height: Optional[DimensionalValue] = Field(
        None,
        description="Required handrail height. Typically 0.9-1.0m."
    )
    min_footprint_sqm: Optional[float] = Field(
        None,
        description=(
            "Calculated minimum footprint area required for the staircase "
            "enclosure. This helps the generator allocate space."
        )
    )
    notes: Optional[str] = Field(None, description="Additional staircase notes.")
    source_clause: Optional[str] = Field(None, description="NBC clause reference.")
    confidence: DataConfidence = Field(DataConfidence.EXTRACTED)


class DoorSpec(BaseModel):
    """
    Door specifications from NBC — sizes, clearances, swing rules.

    The detail engine (Step 5) uses these to place doors correctly.
    """
    door_type: str = Field(
        ...,
        description=(
            "Door type: 'main_entrance', 'internal_room', 'bathroom', "
            "'kitchen', 'balcony', 'fire_exit', 'garage'."
        )
    )
    min_width: Optional[DimensionalValue] = Field(
        None,
        description=(
            "Minimum clear opening width. Main entrance: 1.0m (3.3ft), "
            "Internal: 0.75-0.9m, Bathroom: 0.6-0.75m."
        )
    )
    min_height: Optional[DimensionalValue] = Field(
        None,
        description=(
            "Minimum door height. NBC typically requires 2.0-2.1m (6.6-7ft) "
            "for all doors."
        )
    )
    swing_direction: Optional[str] = Field(
        None,
        description=(
            "Required or recommended swing direction: 'inward', 'outward', "
            "'either', 'outward_for_bathroom'. NBC requires outward swing "
            "for bathrooms for safety."
        )
    )
    min_clearance_from_corner: Optional[DimensionalValue] = Field(
        None,
        description=(
            "Minimum distance from the door frame to the nearest corner "
            "or perpendicular wall. Typically 100-200mm."
        )
    )
    threshold_requirement: Optional[str] = Field(
        None,
        description=(
            "Threshold/step requirements. Bathrooms typically have a "
            "step-down of 25-50mm."
        )
    )
    notes: Optional[str] = Field(None, description="Additional door notes.")
    source_clause: Optional[str] = Field(None, description="NBC clause reference.")
    confidence: DataConfidence = Field(DataConfidence.EXTRACTED)


class WindowSpec(BaseModel):
    """
    Window specifications from NBC.

    Used by both the detail engine (placing windows) and the
    validator (checking ventilation compliance).
    """
    window_type: str = Field(
        ...,
        description=(
            "Window type: 'habitable_room', 'kitchen', 'bathroom', "
            "'staircase', 'passage'."
        )
    )
    min_area_ratio: Optional[float] = Field(
        None,
        description=(
            "Minimum window area as fraction of floor area. "
            "Typically 1/6 (0.167) for habitable rooms."
        )
    )
    min_openable_area_ratio: Optional[float] = Field(
        None,
        description=(
            "Minimum openable (ventilation) area as fraction of floor area. "
            "May be less than total window area if fixed glazing is included."
        )
    )
    min_sill_height: Optional[DimensionalValue] = Field(
        None,
        description="Minimum sill height from finished floor level."
    )
    max_sill_height: Optional[DimensionalValue] = Field(
        None,
        description=(
            "Maximum sill height — windows must not be too high to provide "
            "natural light at occupant level."
        )
    )
    min_width: Optional[DimensionalValue] = Field(
        None, description="Minimum window width if specified."
    )
    min_height: Optional[DimensionalValue] = Field(
        None, description="Minimum window height if specified."
    )
    notes: Optional[str] = Field(None, description="Additional window notes.")
    source_clause: Optional[str] = Field(None, description="NBC clause reference.")
    confidence: DataConfidence = Field(DataConfidence.EXTRACTED)


class CorridorPassageSpec(BaseModel):
    """
    Corridor and passage requirements from NBC.

    Critical for circulation path design in the layout generator.
    """
    passage_type: str = Field(
        ...,
        description=(
            "Type: 'internal_corridor', 'main_passage', 'access_lobby', "
            "'external_corridor', 'verandah'."
        )
    )
    min_width: Optional[DimensionalValue] = Field(
        None,
        description=(
            "Minimum clear width. Internal corridors: typically 0.9-1.0m "
            "(3-3.3ft). Main passages: 1.2m (4ft)."
        )
    )
    max_dead_end_length: Optional[DimensionalValue] = Field(
        None,
        description="Maximum dead-end corridor length. Typically 7.5-15m."
    )
    ceiling_height: Optional[DimensionalValue] = Field(
        None, description="Minimum ceiling height for corridors/passages."
    )
    notes: Optional[str] = Field(None, description="Additional corridor notes.")
    source_clause: Optional[str] = Field(None, description="NBC clause reference.")
    confidence: DataConfidence = Field(DataConfidence.EXTRACTED)


class WallSpec(BaseModel):
    """
    Wall construction specifications from NBC.

    Used by the detail engine to set correct wall thicknesses.
    """
    wall_type: str = Field(
        ...,
        description=(
            "Wall type: 'external_load_bearing', 'internal_load_bearing', "
            "'external_non_load_bearing', 'internal_partition', "
            "'boundary_wall', 'parapet'."
        )
    )
    min_thickness: Optional[DimensionalValue] = Field(
        None,
        description=(
            "Minimum wall thickness. External load-bearing: 230mm (9in), "
            "Internal partition: 115mm (4.5in), Non-load-bearing: 100mm."
        )
    )
    max_height_without_pier: Optional[DimensionalValue] = Field(
        None,
        description=(
            "Maximum free-standing wall height without intermediate piers "
            "or buttresses."
        )
    )
    fire_rating_hours: Optional[float] = Field(
        None,
        description="Required fire resistance rating in hours."
    )
    notes: Optional[str] = Field(None, description="Additional wall notes.")
    source_clause: Optional[str] = Field(None, description="NBC clause reference.")
    confidence: DataConfidence = Field(DataConfidence.EXTRACTED)


class ParkingSpec(BaseModel):
    """
    Parking requirements from NBC.

    Used when the user requests car parking in their plan.
    """
    parking_type: str = Field(
        ...,
        description=(
            "Parking type: 'open_parking', 'covered_parking', "
            "'stilt_parking', 'garage', 'basement_parking'."
        )
    )
    min_stall_width: Optional[DimensionalValue] = Field(
        None,
        description="Minimum parking stall width. Typically 2.5m (8.2ft)."
    )
    min_stall_length: Optional[DimensionalValue] = Field(
        None,
        description="Minimum parking stall length. Typically 5.0-5.5m (16-18ft)."
    )
    min_drive_aisle_width: Optional[DimensionalValue] = Field(
        None,
        description="Minimum drive aisle width. Typically 3.6-6.0m."
    )
    min_headroom: Optional[DimensionalValue] = Field(
        None,
        description="Minimum headroom in covered/stilt parking. Typically 2.4m."
    )
    ramp_gradient_max_pct: Optional[float] = Field(
        None,
        description="Maximum ramp gradient as percentage for basement parking."
    )
    stalls_required_per_unit: Optional[str] = Field(
        None,
        description=(
            "Parking requirement per residential unit. E.g., "
            "'1 ECS per unit for < 100 sqm, 1.5 ECS for 100-200 sqm'."
        )
    )
    notes: Optional[str] = Field(None, description="Additional parking notes.")
    source_clause: Optional[str] = Field(None, description="NBC clause reference.")
    confidence: DataConfidence = Field(DataConfidence.EXTRACTED)


class FireSafetyRule(BaseModel):
    """
    Fire safety requirements from NBC Part 4.

    Used by the validator to check egress paths and separations.
    """
    rule_id: str = Field(
        ..., description="Unique identifier for this fire safety rule."
    )
    rule_category: str = Field(
        ...,
        description=(
            "Category: 'means_of_egress', 'fire_separation', "
            "'smoke_ventilation', 'fire_extinguisher', 'exit_signage', "
            "'travel_distance'."
        )
    )
    description: str = Field(
        ...,
        description="Full description of the fire safety requirement."
    )
    applies_to: Optional[str] = Field(
        None,
        description=(
            "Building type/size this applies to. E.g., "
            "'All residential buildings > 15m height'."
        )
    )
    threshold_value: Optional[str] = Field(
        None,
        description=(
            "Quantitative threshold. E.g., 'Max travel distance: 22.5m', "
            "'Min 2 exits if floor area > 300 sqm'."
        )
    )
    severity: ConstraintSeverity = Field(
        ConstraintSeverity.MANDATORY,
        description="How strictly this must be enforced."
    )
    source_clause: Optional[str] = Field(None, description="NBC clause reference.")
    confidence: DataConfidence = Field(DataConfidence.EXTRACTED)


class PlumbingSanitaryRule(BaseModel):
    """
    Plumbing and sanitary requirements from NBC.

    Used for bathroom and kitchen placement validation.
    """
    rule_id: str = Field(
        ..., description="Unique identifier."
    )
    category: str = Field(
        ...,
        description=(
            "Category: 'toilet_placement', 'kitchen_drainage', "
            "'water_supply', 'septic_tank', 'bathroom_flooring', "
            "'wet_area_waterproofing'."
        )
    )
    description: str = Field(
        ..., description="Full description of the requirement."
    )
    dimensional_requirement: Optional[str] = Field(
        None,
        description="Any dimensional specification (e.g., 'min 1.2m x 0.9m for WC')."
    )
    placement_rule: Optional[str] = Field(
        None,
        description=(
            "Placement constraints. E.g., 'Toilet must not open directly "
            "into kitchen', 'Bathroom must have 150mm floor slope to drain'."
        )
    )
    source_clause: Optional[str] = Field(None, description="NBC clause reference.")
    confidence: DataConfidence = Field(DataConfidence.EXTRACTED)


class EWSAffordableHousingSpec(BaseModel):
    """
    Relaxations for EWS (Economically Weaker Section) and LIG
    (Low Income Group) housing.

    NBC allows reduced minimum sizes for affordable housing.
    Important context for the enricher.
    """
    housing_category: str = Field(
        ...,
        description="Category: 'EWS', 'LIG', 'MIG', 'HIG'."
    )
    max_carpet_area: Optional[DimensionalValue] = Field(
        None,
        description="Maximum carpet area for this category."
    )
    relaxed_room_minimums: Optional[Dict[str, float]] = Field(
        None,
        description=(
            "Relaxed minimum room areas in sqm for this category. "
            "E.g., {'bedroom': 7.5, 'kitchen': 3.3, 'bathroom': 1.8}."
        )
    )
    notes: Optional[str] = Field(None, description="Additional notes.")
    source_clause: Optional[str] = Field(None, description="NBC clause reference.")
    confidence: DataConfidence = Field(DataConfidence.EXTRACTED)


class GeneralBuildingRule(BaseModel):
    """
    Catch-all for any important building rule that doesn't fit
    neatly into the above categories.
    """
    rule_id: str = Field(..., description="Unique identifier.")
    category: str = Field(
        ...,
        description=(
            "Category: 'site_planning', 'drainage', 'electrical', "
            "'structural', 'accessibility', 'green_building', "
            "'rain_water_harvesting', 'solar', 'other'."
        )
    )
    title: str = Field(..., description="Short title for the rule.")
    description: str = Field(
        ..., description="Full description of the requirement."
    )
    quantitative_value: Optional[str] = Field(
        None, description="Any measurable value associated with this rule."
    )
    severity: ConstraintSeverity = Field(ConstraintSeverity.RECOMMENDED)
    applies_to: Optional[str] = Field(
        None, description="What building type/configuration this applies to."
    )
    source_clause: Optional[str] = Field(None, description="NBC clause reference.")
    confidence: DataConfidence = Field(DataConfidence.EXTRACTED)


# =====================================================================
# TOP-LEVEL EXTRACTION SCHEMAS (per extraction pass)
# =====================================================================

class RoomStandardsExtraction(BaseModel):
    """Pass 1: Room specifications, ventilation, and lighting standards."""
    room_minimums: List[RoomMinimumSpec] = Field(
        default_factory=list,
        description=(
            "ALL room types with their minimum size requirements. "
            "Extract EVERY room type mentioned in NBC — bedrooms, "
            "living rooms, kitchens, bathrooms, toilets, store rooms, "
            "corridors, etc. Include separate entries for different "
            "housing categories (EWS, LIG, MIG, HIG) if different "
            "minimums are specified."
        )
    )
    ventilation_standards: List[VentilationStandard] = Field(
        default_factory=list,
        description=(
            "Ventilation and natural lighting requirements for each "
            "room type. Extract window area ratios, cross-ventilation "
            "rules, mechanical ventilation allowances."
        )
    )
    ceiling_height_general: Optional[DimensionalValue] = Field(
        None,
        description="General minimum ceiling height for habitable rooms."
    )
    ceiling_height_non_habitable: Optional[DimensionalValue] = Field(
        None,
        description="Minimum ceiling height for non-habitable spaces."
    )


class PlotRegulationsExtraction(BaseModel):
    """Pass 2: Setbacks, FAR, coverage, open space, parking."""
    setback_rules: List[SetbackRule] = Field(
        default_factory=list,
        description=(
            "ALL setback rules organized by plot size/category. Include "
            "variations for different plot sizes, building heights, "
            "and road widths."
        )
    )
    far_rules: List[FARRule] = Field(
        default_factory=list,
        description=(
            "Floor Area Ratio and coverage rules by plot category, "
            "road width, and building type."
        )
    )
    parking_specs: List[ParkingSpec] = Field(
        default_factory=list,
        description="Parking dimensional requirements by type."
    )
    open_space_rules: Optional[str] = Field(
        None,
        description=(
            "General open space requirements. E.g., 'Plots > 500 sqm "
            "must leave 25% as open space'."
        )
    )


class StructuralConstructionExtraction(BaseModel):
    """Pass 3: Walls, staircases, doors, windows, corridors."""
    staircase_specs: List[StaircaseSpec] = Field(
        default_factory=list,
        description="Staircase specifications by type (internal, common, fire escape)."
    )
    door_specs: List[DoorSpec] = Field(
        default_factory=list,
        description="Door specifications by type (entrance, internal, bathroom, etc.)."
    )
    window_specs: List[WindowSpec] = Field(
        default_factory=list,
        description="Window specifications by room type."
    )
    wall_specs: List[WallSpec] = Field(
        default_factory=list,
        description="Wall thickness and construction specifications."
    )
    corridor_specs: List[CorridorPassageSpec] = Field(
        default_factory=list,
        description="Corridor and passage width requirements."
    )


class SafetyServicesExtraction(BaseModel):
    """Pass 4: Fire safety, plumbing, accessibility, general rules."""
    fire_safety_rules: List[FireSafetyRule] = Field(
        default_factory=list,
        description="Fire safety and means of egress requirements."
    )
    plumbing_rules: List[PlumbingSanitaryRule] = Field(
        default_factory=list,
        description="Plumbing and sanitary requirements for bathrooms/kitchens."
    )
    ews_housing_specs: List[EWSAffordableHousingSpec] = Field(
        default_factory=list,
        description="Relaxations for affordable housing categories."
    )
    general_rules: List[GeneralBuildingRule] = Field(
        default_factory=list,
        description=(
            "Any other important rules: site drainage, rainwater "
            "harvesting, accessibility, etc."
        )
    )


# =====================================================================
# MASTER SCHEMA — FINAL MERGED OUTPUT
# =====================================================================

class IndianBuildingStandards(BaseModel):
    """
    MASTER SCHEMA: Complete structured Indian building standards
    extracted from the National Building Code.

    This is the final output saved to `indian_standards.json` and
    loaded by the SemanticMatcher (Step 2) and ConstraintValidator
    (Step 6) at runtime.

    Structure mirrors the 4-pass extraction pipeline:
      Pass 1: Room specifications + ventilation
      Pass 2: Plot regulations + parking
      Pass 3: Structural elements (stairs, doors, windows, walls)
      Pass 4: Safety + services + general rules
    """

    # ── Metadata ────────────────────────────────────────────────
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "source": "National Building Code of India",
            "version": "NBC 2016 (BIS SP 7:2016)",
            "extraction_engine": "Gemini 2.5 Flash — Multimodal PDF Analysis",
            "total_extraction_passes": 4,
            "data_confidence_note": (
                "Values marked 'extracted' were directly found in the PDF. "
                "'inferred' values were logically derived from context. "
                "'default' values are industry-standard fallbacks where "
                "NBC was silent."
            ),
        },
        description="Extraction metadata and provenance."
    )

    # ── Pass 1: Room Standards ──────────────────────────────────
    room_minimums: List[RoomMinimumSpec] = Field(
        default_factory=list,
        description="Minimum room specifications by type."
    )
    ventilation_standards: List[VentilationStandard] = Field(
        default_factory=list,
        description="Ventilation and lighting requirements."
    )
    ceiling_height_general: Optional[DimensionalValue] = Field(
        None, description="General habitable room ceiling height."
    )
    ceiling_height_non_habitable: Optional[DimensionalValue] = Field(
        None, description="Non-habitable room ceiling height."
    )

    # ── Pass 2: Plot Regulations ────────────────────────────────
    setback_rules: List[SetbackRule] = Field(
        default_factory=list,
        description="Setback rules by plot category."
    )
    far_rules: List[FARRule] = Field(
        default_factory=list,
        description="FAR and coverage rules."
    )
    parking_specs: List[ParkingSpec] = Field(
        default_factory=list,
        description="Parking dimensional requirements."
    )
    open_space_rules: Optional[str] = Field(
        None, description="Open space requirements."
    )

    # ── Pass 3: Structural & Construction ───────────────────────
    staircase_specs: List[StaircaseSpec] = Field(
        default_factory=list,
        description="Staircase specifications."
    )
    door_specs: List[DoorSpec] = Field(
        default_factory=list,
        description="Door specifications."
    )
    window_specs: List[WindowSpec] = Field(
        default_factory=list,
        description="Window specifications."
    )
    wall_specs: List[WallSpec] = Field(
        default_factory=list,
        description="Wall specifications."
    )
    corridor_specs: List[CorridorPassageSpec] = Field(
        default_factory=list,
        description="Corridor and passage specifications."
    )

    # ── Pass 4: Safety & Services ───────────────────────────────
    fire_safety_rules: List[FireSafetyRule] = Field(
        default_factory=list,
        description="Fire safety requirements."
    )
    plumbing_rules: List[PlumbingSanitaryRule] = Field(
        default_factory=list,
        description="Plumbing/sanitary requirements."
    )
    ews_housing_specs: List[EWSAffordableHousingSpec] = Field(
        default_factory=list,
        description="Affordable housing relaxations."
    )
    general_rules: List[GeneralBuildingRule] = Field(
        default_factory=list,
        description="General building rules."
    )

    def summary(self) -> Dict[str, int]:
        """Quick count of extracted rules per category."""
        return {
            "room_minimums": len(self.room_minimums),
            "ventilation_standards": len(self.ventilation_standards),
            "setback_rules": len(self.setback_rules),
            "far_rules": len(self.far_rules),
            "parking_specs": len(self.parking_specs),
            "staircase_specs": len(self.staircase_specs),
            "door_specs": len(self.door_specs),
            "window_specs": len(self.window_specs),
            "wall_specs": len(self.wall_specs),
            "corridor_specs": len(self.corridor_specs),
            "fire_safety_rules": len(self.fire_safety_rules),
            "plumbing_rules": len(self.plumbing_rules),
            "ews_housing_specs": len(self.ews_housing_specs),
            "general_rules": len(self.general_rules),
            "TOTAL_RULES": sum([
                len(self.room_minimums), len(self.ventilation_standards),
                len(self.setback_rules), len(self.far_rules),
                len(self.parking_specs), len(self.staircase_specs),
                len(self.door_specs), len(self.window_specs),
                len(self.wall_specs), len(self.corridor_specs),
                len(self.fire_safety_rules), len(self.plumbing_rules),
                len(self.ews_housing_specs), len(self.general_rules),
            ]),
        }
