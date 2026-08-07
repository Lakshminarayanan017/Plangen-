"""
models.py
=========
Central Pydantic data models for the entire PlanGen pipeline.

  Step 1  → BuildingRequirements   (parser output)
  Step 2  → KnowledgeBundle        (matcher output — rich, production-grade)
  Step 3  → EnrichedPlan           (enricher output — fully specified)
  Step 4  → RoomLayout             (generator output — exact coordinates)

v2 additions (Option 4 — Autoregressive Layout Transformer):
  EnrichedRoom.area_fraction      — softmax-normalised floor-area share per room
  EnrichedRoom.generation_order   — position in the AR generation sequence
  LayoutPlan.solver_used          — now also accepts "autoregressive"
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


# =====================================================================
# SHARED ENUMS
# =====================================================================

class PlotShape(str, Enum):
    RECTANGULAR  = "rectangular"
    L_SHAPED     = "L-shaped"
    IRREGULAR    = "irregular"
    SQUARE       = "square"
    TRAPEZOIDAL  = "trapezoidal"

class ParkingType(str, Enum):
    STILT       = "stilt"
    GARAGE      = "garage"
    MARKED_AREA = "marked_area"
    NONE        = "none"

class Direction(str, Enum):
    NORTH      = "north"
    SOUTH      = "south"
    EAST       = "east"
    WEST       = "west"
    NORTH_EAST = "north_east"
    NORTH_WEST = "north_west"
    SOUTH_EAST = "south_east"
    SOUTH_WEST = "south_west"


# =====================================================================
# STEP 1 — BUILDING REQUIREMENTS  (parser output)
# =====================================================================

class PlotDimensions(BaseModel):
    """Physical plot dimensions with shape and orientation metadata."""
    length:          Optional[float] = Field(None, description="Length in the dominant axis (depth from road), in chosen unit.")
    width:           Optional[float] = Field(None, description="Width along the road-facing side, in chosen unit.")
    unit:            str             = Field("ft",  description="'ft' or 'm'. Internally always converted to ft.")
    total_area_sqft: Optional[float] = Field(None, description="Total plot area in sqft.")

    @model_validator(mode="after")
    def compute_area(self):
        if self.length and self.width and not self.total_area_sqft:
            factor = 1.0 if self.unit == "ft" else 10.7639
            self.total_area_sqft = round(self.length * self.width * factor, 2)
        return self


class PlotContext(BaseModel):
    """Spatial context about the plot — orientation, road access, shape."""
    shape:              PlotShape           = Field(PlotShape.RECTANGULAR)
    road_facing_sides:  List[Direction]     = Field(default_factory=list)
    north_direction:    Optional[Direction] = Field(None, description="Which side of the plot faces geographic North.")
    entrance_side:      Optional[Direction] = Field(None)
    image_source_notes: Optional[str]       = Field(None)


class Setbacks(BaseModel):
    """Building setback distances from plot boundaries (in ft)."""
    front: Optional[float] = None
    rear:  Optional[float] = None
    left:  Optional[float] = None
    right: Optional[float] = None
    unit:  str             = "ft"


class RoomRequirement(BaseModel):
    """Individual room specification extracted from user input."""
    room_type:             str            = Field(..., description=(
        "Standardised room type. E.g. 'Bedroom', 'Master Bedroom', 'Kitchen', "
        "'Living Room', 'Dining Room', 'Pooja Room', 'Bathroom', 'Balcony', "
        "'Study Room', 'Store Room', 'Staircase', 'Car Parking'."
    ))
    quantity:              int            = Field(1)
    specific_requirements: Optional[str] = None
    preferred_floor:       Optional[int] = Field(None, description="0=ground, 1=first, 2=second.")


class BuildingRequirements(BaseModel):
    """
    Complete parsed output from Step 1.
    Fields the user did NOT mention remain None/empty.
    """
    plot_dimensions:     Optional[PlotDimensions] = None
    plot_context:        Optional[PlotContext]     = None
    setbacks:            Optional[Setbacks]        = None
    number_of_floors:    Optional[int]             = Field(None, ge=1, le=3)
    rooms:               List[RoomRequirement]     = Field(default_factory=list)
    vastu_compliant:     Optional[bool]            = None
    parking_type:        Optional[ParkingType]     = None
    include_furniture:   Optional[bool]            = None
    architectural_style: Optional[str]             = None
    building_type:       Optional[str]             = None
    additional_notes:    Optional[str]             = None


# =====================================================================
# STEP 2 — KNOWLEDGE BUNDLE  (matcher output)
# =====================================================================

class RoomStats(BaseModel):
    """Legacy minimal stats — kept for backward compatibility."""
    min_width:   float = Field(..., description="Minimum width in ft")
    min_length:  float = Field(..., description="Minimum length in ft")
    target_area: float = Field(..., description="Target area in sqft")


class AdjacencyRule(BaseModel):
    """Directional room adjacency rule."""
    room_a:   str
    room_b:   str
    relation: str   = Field(..., description="'adjacent', 'near', 'away'")
    weight:   float = Field(..., ge=0.0, le=10.0)


class RoomSizeDistribution(BaseModel):
    """
    Statistical distribution of room sizes extracted from the top-K
    matched plans. All dimensions in feet (converted from m²).
    """
    room_type:        str
    min_width_ft:     float
    p25_width_ft:     float
    median_width_ft:  float
    p75_width_ft:     float
    max_width_ft:     float
    min_area_sqft:    float
    p25_area_sqft:    float
    median_area_sqft: float
    p75_area_sqft:    float
    max_area_sqft:    float
    sample_count:     int
    source:           str = "cubicasa5k_matched_plans"


class MatchedPlanRef(BaseModel):
    """Summary of one reference plan retrieved from the FAISS index."""
    plan_key:         str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    bhk:              str
    room_count:       int
    aspect_ratio:     float
    zone_balance:     Dict[str, float] = Field(default_factory=dict)


class KnowledgeBundle(BaseModel):
    """
    Production-grade output of Step 2 — the Smart Semantic Matcher.

    Combines:
      * Statistical DNA from the top-15 most similar real floor plans
      * Indian NBC building standards
      * Vastu Shastra rules (if requested)
      * Learned adjacency weights from CubiCasa5K
      * Zone placement probabilities per room type
    """
    original_requirements: BuildingRequirements

    # ── Legacy fields (backward compat) ─────────────────────────────
    room_stats:                     Dict[str, RoomStats]  = Field(default_factory=dict)
    adjacency_rules:                List[AdjacencyRule]   = Field(default_factory=list)
    floor_distribution_suggestions: Dict[str, int]        = Field(default_factory=dict)

    # ── Retrieval results ────────────────────────────────────────────
    matched_plans:       List[MatchedPlanRef] = Field(default_factory=list,
        description="Top-15 most similar real plans from CubiCasa5K.")
    match_quality_score: float               = Field(0.0, ge=0.0, le=1.0,
        description="Average similarity of the top-K matches (0=no good match, 1=perfect).")

    # ── Room size distributions (from matched plans) ─────────────────
    room_size_distributions: Dict[str, RoomSizeDistribution] = Field(default_factory=dict,
        description="Per-room-type size statistics derived from matched plans.")

    # ── Adjacency & zone data ────────────────────────────────────────
    adjacency_weights:      Dict[str, Dict[str, float]] = Field(default_factory=dict,
        description="room_a -> {room_b: weight} from learned_patterns.json.")
    zone_probabilities:     Dict[str, Dict[str, float]] = Field(default_factory=dict,
        description="room_type -> {front: p, middle: p, back: p} from zone_patterns.json.")
    circulation_benchmarks: Dict[str, float]            = Field(default_factory=dict,
        description="Journey efficiency metrics: avg_hops, entry_to_kitchen_hops, etc.")

    # ── Standards & rules ────────────────────────────────────────────
    vastu_rules_applied:  Dict[str, Any]   = Field(default_factory=dict,
        description="Full vastu_rules.json when vastu_compliant=True, else empty.")
    nbc_standards:        Dict[str, Any]   = Field(default_factory=dict,
        description="Relevant NBC room standards from nbc_room_standards.json.")
    nbc_plot_regulations: Dict[str, Any]   = Field(default_factory=dict,
        description="NBC plot regulations (setbacks, FAR, coverage).")
    setbacks_recommended: Optional[Setbacks] = Field(None,
        description="NBC-recommended setbacks for this plot size.")

    # ── Metadata ─────────────────────────────────────────────────────
    index_version:      str   = "1.0"
    total_plans_in_index: int = 0
    retrieval_time_ms:  float = 0.0

    def get_room_target_area(self, room_type: str) -> float:
        """Get median area in sqft for a room type. Falls back to NBC minimum."""
        norm = room_type.lower().replace(" ", "_")
        if norm in self.room_size_distributions:
            return self.room_size_distributions[norm].median_area_sqft
        NBC_MIN = {
            "bedroom": 96.0, "master_bedroom": 120.0, "kitchen": 60.0,
            "living_room": 120.0, "dining_room": 80.0, "bathroom": 25.0,
            "toilet": 16.0, "study_room": 64.0, "pooja_room": 25.0,
            "store_room": 25.0, "balcony": 20.0, "staircase": 30.0,
        }
        for key, val in NBC_MIN.items():
            if key in norm:
                return val
        return 64.0

    def get_adjacency_weight(self, room_a: str, room_b: str) -> float:
        """Get adjacency weight between two room types."""
        a = room_a.lower().replace(" ", "_")
        b = room_b.lower().replace(" ", "_")
        return (
            self.adjacency_weights.get(a, {}).get(b, 0.0) or
            self.adjacency_weights.get(b, {}).get(a, 0.0)
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "matched_plans":       len(self.matched_plans),
            "match_quality_score": round(self.match_quality_score, 3),
            "room_distributions":  len(self.room_size_distributions),
            "adjacency_pairs":     sum(len(v) for v in self.adjacency_weights.values()),
            "zone_types":          len(self.zone_probabilities),
            "vastu_enabled":       bool(self.vastu_rules_applied),
            "nbc_room_types":      len(self.nbc_standards),
            "retrieval_time_ms":   round(self.retrieval_time_ms, 1),
        }


# =====================================================================
# STEP 3 — ENRICHED PLAN  (enricher output)
# =====================================================================

class VastuConstraint(BaseModel):
    """
    Vastu rules extracted for a specific room type.
    Directions use compass notation: N, NE, E, SE, S, SW, W, NW, center.
    """
    preferred_directions:         List[str] = Field(default_factory=list,
        description="Vastu preferred compass zones, e.g. ['SW'] for master bedroom")
    acceptable_directions:        List[str] = Field(default_factory=list)
    prohibited_directions:        List[str] = Field(default_factory=list)
    preferred_door_directions:    List[str] = Field(default_factory=list)
    prohibited_door_directions:   List[str] = Field(default_factory=list)
    floor_preference:             str       = Field("any_floor",
        description="'ground_floor_only' | 'top_floor_preferred' | 'any_floor'")
    should_not_be_above_or_below: List[str] = Field(default_factory=list)
    should_not_be_adjacent_to:    List[str] = Field(default_factory=list)
    constraint_type:              str       = Field("soft",
        description="'hard' = high-priority Vastu rule | 'soft' = medium/low priority")
    wall_colors_preferred:        List[str] = Field(default_factory=list)
    special_rules:                List[str] = Field(default_factory=list)
    notes:                        str       = ""


class EnrichedRoom(BaseModel):
    """
    A fully-specified room ready for the layout generator (Step 4).

    v2 additions for the Autoregressive Layout Transformer (Option 4):
      area_fraction    — proportional share of floor area (softmax-normalised).
                         The AR engine uses this instead of fixed lookup-table sizes,
                         allowing room sizes to be jointly determined and adaptive.
      generation_order — position in the autoregressive sequence.
                         Anchor rooms (living room, master bedroom) are generated
                         first (order 0, 1, …); service rooms (bathrooms, storage)
                         are generated last, conditioned on already-placed rooms.
    """
    # ── Identity ────────────────────────────────────────────────────
    room_id:        str  = Field(...,
        description="Stable unique ID: 'master_bedroom_1', 'bedroom_2', 'kitchen_1'")
    room_type:      str  = Field(...,
        description="Normalised snake_case type: 'master_bedroom', 'kitchen', etc.")
    display_name:   str  = Field(...,
        description="Human-readable label: 'Master Bedroom', 'Bedroom 2', 'Kitchen'")
    quantity_index: int  = Field(1, ge=1,
        description="Index among rooms of same type: 1st bedroom, 2nd bedroom, etc.")
    implicit_room:  bool = Field(False,
        description="True if added by enricher (not explicitly requested by user)")

    # ── Size specification (NBC-clamped, statistics-informed) ────────
    target_width_ft:   float = Field(..., gt=0,
        description="Target width from matched-plan statistics (NBC floor applied)")
    target_length_ft:  float = Field(..., gt=0,
        description="Target length derived from area / width")
    target_area_sqft:  float = Field(..., gt=0,
        description="Target area — median from matched plans, clamped to NBC min")
    min_width_ft:      float = Field(..., gt=0, description="NBC hard minimum width")
    min_length_ft:     float = Field(..., gt=0, description="NBC hard minimum length")
    min_area_sqft:     float = Field(..., gt=0, description="NBC hard minimum area")
    max_area_sqft:     float = Field(..., gt=0,
        description="Reasonable upper bound from statistics (p75 or p75 x 1.5)")
    ceiling_height_ft: float = Field(..., gt=0,
        description="NBC minimum ceiling height for this room classification")

    # ── Autoregressive sizing (Option 4) ────────────────────────────
    area_fraction: float = Field(0.0, ge=0.0, le=1.0,
        description="Fraction of total net-buildable area on this floor that is "
                    "allocated to this room. Softmax-normalised across all rooms on "
                    "the same floor so fractions always sum to ~1.0. "
                    "The AR engine samples (w, h) consistent with this fraction "
                    "rather than reading from the static lookup table.")
    generation_order: int = Field(0, ge=0,
        description="0-indexed position of this room in the autoregressive generation "
                    "sequence. Public anchor rooms (living room, foyer, master bedroom) "
                    "get order 0..2; secondary rooms (bedrooms, kitchen, dining) follow; "
                    "service rooms (bathrooms, utility, store, staircase) are last. "
                    "The AR decoder generates each room conditioned on all rooms with "
                    "lower generation_order that have already been placed.")

    # ── Placement specification ──────────────────────────────────────
    preferred_floor:     int = Field(0, ge=0, le=3,
        description="Floor assignment: 0=ground, 1=first, 2=second")
    preferred_zone:      str = Field("middle",
        description="Plot-relative zone: 'front' | 'middle' | 'back'")
    preferred_direction: str = Field("N",
        description="Absolute compass direction (from Vastu if enabled, else inferred): "
                    "N | NE | E | SE | S | SW | W | NW")

    # ── Vastu ────────────────────────────────────────────────────────
    vastu: Optional[VastuConstraint] = Field(None,
        description="Full Vastu constraint set. None if Vastu is not enabled.")

    # ── Relationships ────────────────────────────────────────────────
    attached_bathroom_id:    Optional[str]    = Field(None,
        description="room_id of the bathroom attached to this bedroom (if any)")
    adjacency_preferences:   Dict[str, float] = Field(default_factory=dict,
        description="room_type -> adjacency weight (0–10). Higher = must be placed near.")
    should_not_be_adjacent_to: List[str]      = Field(default_factory=list,
        description="Room types this room must NOT share a wall with (NBC + Vastu)")

    # ── Ventilation & structure ──────────────────────────────────────
    is_habitable:        bool  = Field(True,
        description="NBC: habitable rooms need natural light + ventilation")
    needs_exterior_wall: bool  = Field(True,
        description="Must touch an exterior wall (for windows/ventilation)")
    door_width_ft:       float = Field(2.5, gt=0,
        description="NBC standard door width for this room type (ft)")

    # ── User-specified overrides ─────────────────────────────────────
    user_specific_requirements: Optional[str] = Field(None,
        description="Verbatim from user Step 1 input, e.g. 'south-facing', 'attached bathroom'")


class FloorPlan(BaseModel):
    """Rooms assigned to a single floor, with gross area summary."""
    floor_number:    int       = Field(..., ge=0, le=3)
    floor_label:     str       = Field(..., description="'Ground Floor', 'First Floor', etc.")
    room_ids:        List[str] = Field(default_factory=list)
    gross_area_sqft: float     = Field(0.0, ge=0.0,
        description="Sum of target areas for rooms on this floor")


class EnrichedPlan(BaseModel):
    """
    Complete, gap-free specification ready for the layout generator (Step 4).

    Every room has: exact dimensions, floor assignment, zone preference,
    compass direction, Vastu rules, adjacency weights, area_fraction,
    and generation_order for the autoregressive engine.

    The generator only needs to PLACE rooms spatially — no reasoning required.
    """
    # ── Source data ──────────────────────────────────────────────────
    original_requirements: BuildingRequirements
    match_quality_score:   float = Field(0.0, ge=0.0, le=1.0,
        description="Average cosine similarity of top-K matched plans (Step 2)")

    # ── Plot geometry ────────────────────────────────────────────────
    plot_width_ft:  float = Field(..., gt=0,
        description="Road-facing frontage (shorter dimension in most Indian plots)")
    plot_length_ft: float = Field(..., gt=0,
        description="Depth running away from road (longer dimension)")
    plot_area_sqft: float = Field(..., gt=0)

    # ── Setbacks (NBC-enforced) ──────────────────────────────────────
    setbacks:                Setbacks
    net_buildable_width_ft:  float = Field(..., gt=0,
        description="plot_width_ft minus left_setback minus right_setback")
    net_buildable_length_ft: float = Field(..., gt=0,
        description="plot_length_ft minus front_setback minus rear_setback")
    net_buildable_area_sqft: float = Field(..., gt=0)

    # ── Orientation ──────────────────────────────────────────────────
    entrance_direction: str = Field("N",
        description="Which compass side the main entrance/gate faces: N | E | S | W")
    north_direction:    str = Field("N",
        description="Which physical side of the plot points to geographic North")

    # ── Floors ───────────────────────────────────────────────────────
    total_floors: int             = Field(1, ge=1, le=3)
    floors:       List[FloorPlan] = Field(default_factory=list,
        description="One FloorPlan per floor (Ground, First, Second)")

    # ── Rooms ────────────────────────────────────────────────────────
    rooms:                List[EnrichedRoom] = Field(default_factory=list)
    implicit_rooms_added: List[str]          = Field(default_factory=list,
        description="Display names of rooms the enricher added (not user-requested)")

    # ── Adjacency graph (room_id -> room_id -> weight) ────────────────
    adjacency_graph: Dict[str, Dict[str, float]] = Field(default_factory=dict,
        description="Per-instance adjacency weights for the generator. "
                    "Expanded from type-level weights to specific room IDs.")

    # ── Vastu ────────────────────────────────────────────────────────
    vastu_enabled:               bool           = False
    vastu_direction_assignments: Dict[str, str] = Field(default_factory=dict,
        description="room_type -> assigned Vastu compass direction, e.g. 'master_bedroom' -> 'SW'")

    # ── FAR & coverage (NBC regulation) ──────────────────────────────
    max_ground_coverage_sqft: float = Field(..., gt=0,
        description="60% of plot_area_sqft — maximum footprint per NBC")
    max_far_total_sqft:       float = Field(..., gt=0,
        description="FAR 1.5 x plot_area_sqft — maximum total built-up area")
    total_target_area_sqft:   float = Field(0.0, ge=0.0,
        description="Sum of all room target areas across all floors")
    area_budget_ok:           bool  = Field(True,
        description="True if total_target_area_sqft <= max_far_total_sqft")

    # ── Quality & traceability ───────────────────────────────────────
    enrichment_source:   str            = Field("full_statistical",
        description="'full_statistical' | 'nbc_fallback' | 'area_budget_ar'")
    enrichment_warnings: List[str]      = Field(default_factory=list)
    gemini_decisions:    List[Dict[str, str]] = Field(default_factory=list,
        description="Traceability log of Gemini reasoning calls: "
                    "[{decision, reasoning, confidence}]")

    # ── Convenience helpers ──────────────────────────────────────────

    def _get_room_index(self) -> Dict[str, "EnrichedRoom"]:
        """O(1) room lookup index, built lazily and cached on instance."""
        # Pydantic v2 BaseModel doesn't support @cached_property;
        # use __dict__ as a manual cache that bypasses Pydantic's __setattr__.
        cache_key = "_room_index_cache"
        cached = self.__dict__.get(cache_key)
        if cached is None:
            cached = {r.room_id: r for r in self.rooms}
            self.__dict__[cache_key] = cached
        return cached

    def get_room_by_id(self, room_id: str) -> Optional["EnrichedRoom"]:
        """O(1) lookup of a room by its stable ID (cached hash-map)."""
        return self._get_room_index().get(room_id)

    def get_rooms_on_floor(self, floor_number: int) -> List[EnrichedRoom]:
        """All rooms assigned to a specific floor."""
        return [r for r in self.rooms if r.preferred_floor == floor_number]

    def get_rooms_by_type(self, room_type: str) -> List[EnrichedRoom]:
        """All rooms of a given normalised type."""
        return [r for r in self.rooms if r.room_type == room_type]

    def get_rooms_in_generation_order(self, floor_number: int) -> List["EnrichedRoom"]:
        """Rooms on a floor sorted by generation_order (ascending) for AR inference."""
        return sorted(
            self.get_rooms_on_floor(floor_number),
            key=lambda r: r.generation_order,
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "total_rooms":            len(self.rooms),
            "implicit_rooms_added":   len(self.implicit_rooms_added),
            "total_floors":           self.total_floors,
            "plot_ft":                f"{self.plot_width_ft}x{self.plot_length_ft}",
            "net_buildable_ft":       f"{self.net_buildable_width_ft:.1f}x{self.net_buildable_length_ft:.1f}",
            "total_target_area_sqft": round(self.total_target_area_sqft, 1),
            "area_budget_ok":         self.area_budget_ok,
            "vastu_enabled":          self.vastu_enabled,
            "match_quality":          round(self.match_quality_score, 3),
            "enrichment_source":      self.enrichment_source,
            "warnings":               len(self.enrichment_warnings),
            "gemini_decisions":       len(self.gemini_decisions),
        }


# =====================================================================
# STEP 4 — LAYOUT PLAN  (generator output — exact 2-D room coordinates)
# =====================================================================

class PlacedRoom(BaseModel):
    """
    A single room that has been placed on a floor with exact coordinates.

    Coordinate system:
      Origin (0, 0) = bottom-left (SW) corner of the net buildable area.
      x increases eastward, y increases northward.
      All values in feet.
    """
    room_id:       str   = Field(..., description="Matches EnrichedRoom.room_id.")
    room_type:     str
    display_name:  str
    floor:         int   = Field(..., description="0=Ground, 1=First, etc.")
    implicit_room: bool  = False

    # Position — bottom-left corner of room rectangle (ft from net-buildable origin)
    x_ft:    float = Field(..., description="Left edge (east offset from SW origin).")
    y_ft:    float = Field(..., description="Bottom edge (north offset from SW origin).")

    # Dimensions (ft)
    width_ft:  float = Field(..., description="Room width along x-axis (east-west).")
    length_ft: float = Field(..., description="Room length along y-axis (north-south).")
    area_sqft: float = Field(..., description="Actual placed area = width x length.")

    # Preferred compass quadrant — copied from enricher for scoring
    preferred_direction: str   = "N"
    preferred_zone:      str   = "middle"

    # Quality metrics set by generator after placement
    zone_score:      float = 0.0   # 0.0-1.0, how well zone preference was met
    adjacency_score: float = 0.0   # 0.0-1.0, how well adjacency preferences were met


class LayoutFloor(BaseModel):
    """One floor's completed room layout."""
    floor_number: int
    floor_label:  str   = ""

    # Net buildable extents for this floor (ft)
    net_width_ft:  float
    net_length_ft: float

    # All placed rooms on this floor
    rooms: List[PlacedRoom] = Field(default_factory=list)

    # Floor-level metrics
    floor_area_placed_sqft: float = 0.0
    floor_coverage_pct:     float = 0.0
    floor_adjacency_score:  float = 0.0
    floor_zone_score:       float = 0.0

    @property
    def room_count(self) -> int:
        return len(self.rooms)

    def _get_room_index(self) -> Dict[str, "PlacedRoom"]:
        """O(1) room lookup index, built lazily and cached on instance."""
        cache_key = "_room_index_cache"
        cached = self.__dict__.get(cache_key)
        if cached is None:
            cached = {r.room_id: r for r in self.rooms}
            self.__dict__[cache_key] = cached
        return cached

    def get_room(self, room_id: str) -> Optional[PlacedRoom]:
        """O(1) lookup by room_id (cached hash-map)."""
        return self._get_room_index().get(room_id)


class LayoutPlan(BaseModel):
    """
    Complete multi-floor layout plan — output of Step 4 (Generator).

    solver_used values:
      "cp_sat"         — Google OR-Tools CP-SAT (constraint programming)
      "greedy"         — Greedy placer (fast fallback)
      "autoregressive" — GNN + Autoregressive Transformer (Option 4)
      "diffusion"      — Legacy DDPM diffusion decoder
    """
    run_id:            str = ""
    source_step3_json: str = ""

    # Plot geometry (copied from EnrichedPlan for self-contained reference)
    plot_width_ft:           float
    plot_length_ft:          float
    net_buildable_width_ft:  float
    net_buildable_length_ft: float
    setback_front_ft:        float = 0.0
    setback_rear_ft:         float = 0.0
    setback_left_ft:         float = 0.0
    setback_right_ft:        float = 0.0

    # Orientation
    entrance_direction: str  = "N"
    north_direction:    str  = "N"
    vastu_enabled:      bool = False

    # Floors
    total_floors: int               = 1
    floors:       List[LayoutFloor] = Field(default_factory=list)

    # Summary metrics
    total_rooms_placed:      int   = 0
    total_area_placed_sqft:  float = 0.0
    overall_adjacency_score: float = 0.0
    overall_zone_score:      float = 0.0
    layout_quality_score:    float = 0.0

    # Solver metadata
    solver_used:   str   = "greedy"   # "cp_sat" | "autoregressive" | "greedy" | "diffusion"
    solve_time_ms: float = 0.0
    solver_status: str   = "unknown"  # "optimal" | "feasible" | "timeout" | "greedy" | "ar_sampled"

    layout_warnings: List[str] = Field(default_factory=list)

    def _get_floor_index(self) -> Dict[int, LayoutFloor]:
        """O(1) floor lookup index, built lazily and cached on instance."""
        cache_key = "_floor_index_cache"
        cached = self.__dict__.get(cache_key)
        if cached is None:
            cached = {f.floor_number: f for f in self.floors}
            self.__dict__[cache_key] = cached
        return cached

    def get_floor(self, floor_number: int) -> Optional[LayoutFloor]:
        """O(1) lookup by floor number (cached hash-map)."""
        return self._get_floor_index().get(floor_number)

    def get_room(self, room_id: str) -> Optional[PlacedRoom]:
        """O(F) lookup across all floors using per-floor O(1) index."""
        for floor in self.floors:
            r = floor.get_room(room_id)
            if r:
                return r
        return None

    def summary(self) -> Dict[str, Any]:
        return {
            "total_rooms_placed":   self.total_rooms_placed,
            "total_area_sqft":      round(self.total_area_placed_sqft, 1),
            "floors":               self.total_floors,
            "solver":               self.solver_used,
            "solver_status":        self.solver_status,
            "solve_time_ms":        round(self.solve_time_ms, 1),
            "adjacency_score":      round(self.overall_adjacency_score, 3),
            "zone_score":           round(self.overall_zone_score, 3),
            "layout_quality_score": round(self.layout_quality_score, 3),
            "warnings":             len(self.layout_warnings),
        }
