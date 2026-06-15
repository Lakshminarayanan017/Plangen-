from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, model_validator


class PlotShape(str, Enum):
    """Supported plot shapes."""
    RECTANGULAR = "rectangular"
    L_SHAPED = "L-shaped"
    IRREGULAR = "irregular"
    SQUARE = "square"
    TRAPEZOIDAL = "trapezoidal"

class ParkingType(str, Enum):
    """Car parking types from PRD."""
    STILT = "stilt"
    GARAGE = "garage"
    MARKED_AREA = "marked_area"
    NONE = "none"

class Direction(str, Enum):
    """Cardinal and inter-cardinal directions."""
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    NORTH_EAST = "north_east"
    NORTH_WEST = "north_west"
    SOUTH_EAST = "south_east"
    SOUTH_WEST = "south_west"


class PlotDimensions(BaseModel):
    """Physical plot dimensions with shape and orientation metadata."""
    length: Optional[float] = Field(None, description="Length of the plot (the longer side, typically depth).")
    width: Optional[float] = Field(None, description="Width of the plot (the shorter side, typically road-facing).")
    unit: str = Field("ft", description="Unit of measurement (e.g., 'ft', 'm'). Default is 'ft'.")
    total_area_sqft: Optional[float] = Field(None, description="Total plot area in sqft, if explicitly stated.")

    @model_validator(mode="after")
    def compute_area(self):
        """Auto-compute area from length × width if both are present and area wasn't given."""
        if self.length and self.width and not self.total_area_sqft:
            if self.unit == "ft":
                self.total_area_sqft = round(self.length * self.width, 2)
            elif self.unit == "m":
                # Convert to sqft for internal consistency (1 m² ≈ 10.764 sqft)
                self.total_area_sqft = round(self.length * self.width * 10.764, 2)
        return self


class PlotContext(BaseModel):
    """Spatial context about the plot — orientation, road access, shape."""
    shape: PlotShape = Field(PlotShape.RECTANGULAR, description="Shape of the plot.")
    road_facing_sides: List[Direction] = Field(
        default_factory=list,
        description="Which side(s) of the plot face a road (e.g., ['north', 'east'] for a corner plot)."
    )
    north_direction: Optional[Direction] = Field(
        None,
        description="Which side of the plot faces geographic north. Critical for Vastu."
    )
    entrance_side: Optional[Direction] = Field(
        None,
        description="Preferred entrance direction (often same as primary road-facing side)."
    )
    image_source_notes: Optional[str] = Field(
        None,
        description=(
            "Provenance notes when data was extracted from an image input. "
            "E.g., 'dimensions from hand-drawn sketch', 'shape detected from site photo', "
            "'full plan extracted from architectural drawing'. Helps downstream modules "
            "understand data confidence level."
        )
    )


class Setbacks(BaseModel):
    """Building setback distances from plot boundaries."""
    front: Optional[float] = Field(None, description="Front setback in ft.")
    rear: Optional[float] = Field(None, description="Rear setback in ft.")
    left: Optional[float] = Field(None, description="Left side setback in ft.")
    right: Optional[float] = Field(None, description="Right side setback in ft.")
    unit: str = Field("ft", description="Setback unit. Default is 'ft'.")


class RoomRequirement(BaseModel):
    """Individual room specification extracted from user input."""
    room_type: str = Field(
        ...,
        description=(
            "Type of room. Common Indian types include: 'Bedroom', 'Master Bedroom', "
            "'Kitchen', 'Living Room', 'Drawing Room', 'Dining Room', 'Pooja Room', "
            "'Bathroom', 'Balcony', 'Utility Room', 'Store Room', 'Study Room', "
            "'Servant Room', 'Car Parking', 'Staircase', 'Foyer', 'Passage'."
        )
    )
    quantity: int = Field(1, description="Number of these rooms required.")
    specific_requirements: Optional[str] = Field(
        None,
        description="Specific user constraints (e.g., 'attached bathroom', 'ground floor only', 'south-facing')."
    )
    preferred_floor: Optional[int] = Field(
        None,
        description="Which floor this room should be on (0 = ground, 1 = first, 2 = second). None if unspecified."
    )


class BuildingRequirements(BaseModel):
    """
    Complete parsed output from Step 1 — everything the user explicitly stated.
    Fields that the user did NOT mention should remain None/empty.
    """
    # --- Plot ---
    plot_dimensions: Optional[PlotDimensions] = Field(
        None, description="The physical dimensions of the plot."
    )
    plot_context: Optional[PlotContext] = Field(
        None, description="Spatial context: shape, road-facing sides, north direction, entrance."
    )
    setbacks: Optional[Setbacks] = Field(
        None, description="User-specified setback distances. None means system will use defaults."
    )

    # --- Building ---
    number_of_floors: Optional[int] = Field(
        None,
        description=(
            "Total number of floors requested (1 = ground only, 2 = G+1, 3 = G+2). "
            "None if the user did not specify. System supports up to G+2 (max 3)."
        )
    )
    rooms: List[RoomRequirement] = Field(
        default_factory=list,
        description="List of all requested rooms and their details."
    )

    # --- Preferences ---
    vastu_compliant: Optional[bool] = Field(
        None, description="Whether Vastu Shastra principles should be applied. None = not mentioned."
    )
    parking_type: Optional[ParkingType] = Field(
        None, description="Type of car parking requested."
    )
    include_furniture: Optional[bool] = Field(
        None, description="Whether to include furniture in the generated plan."
    )
    architectural_style: Optional[str] = Field(
        None, description="Architectural style preference (e.g., 'modern', 'traditional', 'contemporary')."
    )
    building_type: Optional[str] = Field(
        None,
        description=(
            "Type of building structure. Common values: 'villa', 'bungalow', "
            "'independent_house', 'duplex', 'row_house', 'apartment', 'farmhouse', 'kothi'. "
            "None if the user did not specify a building type."
        )
    )
    additional_notes: Optional[str] = Field(
        None, description="Any other specific constraints, styles, or requirements mentioned."
    )

# =====================================================================
# STEP 2 MODELS (KNOWLEDGE BUNDLE)
# =====================================================================
from typing import Dict, Any

class RoomStats(BaseModel):
    """Statistical data for a specific room type from learned patterns."""
    min_width: float = Field(..., description="Minimum acceptable width in ft")
    min_length: float = Field(..., description="Minimum acceptable length in ft")
    target_area: float = Field(..., description="Target ideal area in sqft")

class AdjacencyRule(BaseModel):
    """Rules for room relationships."""
    room_a: str
    room_b: str
    relation: str = Field(..., description="e.g., 'adjacent', 'near', 'away'")
    weight: float = Field(..., description="Importance weight from 0.0 to 1.0")

class KnowledgeBundle(BaseModel):
    """
    The enriched data bundle resulting from Step 2.
    Combines the user's explicit requirements with learned data patterns,
    Indian building standards, and Vastu rules.
    """
    original_requirements: BuildingRequirements
    room_stats: Dict[str, RoomStats] = Field(default_factory=dict)
    adjacency_rules: List[AdjacencyRule] = Field(default_factory=list)
    floor_distribution_suggestions: Dict[str, int] = Field(default_factory=dict)
    vastu_rules_applied: Dict[str, Any] = Field(default_factory=dict)
    standards_applied: Dict[str, Any] = Field(default_factory=dict)
