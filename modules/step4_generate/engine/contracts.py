"""
contracts.py — typed interfaces between engine tiers.

Every tier speaks ONLY these types. A tier can be swapped (statistical
fallback ↔ trained model) without any other tier noticing. Contracts carry
floor context from day 1 even though multi-floor realization is deferred —
retrofitting contracts later rots the orchestrator (engine doc §8.5).

Requests validate themselves (`EngineRequest.problems()` / `.validate()`)
and round-trip through dicts so the harness can store briefs as JSON.
Engine behavior knobs live in EngineConfig, not scattered constants.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

# Tier-2 proposals live on this coarse raster (row, col), clipped to the
# plot boundary. Chosen to match the planned placement model's output head.
SEED_GRID = 32

# Size classes quantize target area into 25-sqft buckets. The count is
# capped so the placement model's size head has a fixed vocabulary
# (SIZE_CLASS_MAX logits); the top bucket absorbs everything >= its area
# (SIZE_CLASS_MAX * 25 sqft). This only sets the carve's INITIAL area hint —
# final areas are calibrated by the settler to the true target_sqft
# (engine/settle.py::scaled_targets_cells), so the cap never limits a large
# room's realized size, only the discrete label the model predicts.
SQFT_PER_SIZE_CLASS = 25.0
SIZE_CLASS_MAX = 40                 # 40 * 25 = 1000 sqft top bucket


def size_class_for(target_sqft: float) -> int:
    """Area (sqft) -> discrete size class in [1, SIZE_CLASS_MAX]. Single
    source of truth shared by RoomSpec, the data-prep pipeline, and the
    model's size head."""
    return max(1, min(SIZE_CLASS_MAX, round(target_sqft / SQFT_PER_SIZE_CLASS)))

VALID_ZONES = ("public", "service", "private")
VALID_SIDES = ("N", "E", "S", "W")
VALID_OPENING_KINDS = ("door", "wide")

MIN_PLOT_FT = 12
MAX_PLOT_FT = 200


class EngineRequestError(ValueError):
    """The request cannot be built as specified."""


@dataclass(frozen=True)
class RoomSpec:
    """One requested room. (Grows toward blueprint §3's full RoomSpec.)"""
    name: str
    rtype: str
    target_sqft: float
    zone: str = "private"          # public | service | private
    floor: int = 0

    @property
    def size_class(self) -> int:
        return size_class_for(self.target_sqft)


@dataclass(frozen=True)
class OpeningWish:
    room_a: str
    room_b: str
    kind: str                      # "door" | "wide"


@dataclass
class EngineConfig:
    """Engine behavior knobs — one place, harness-recordable."""
    settle_sweeps: int = 60
    settle_aspect_limit: float = 2.8
    settle_aspect_weight: float = 0.30
    max_repair_rounds: int = 2
    dedup_candidates: bool = True
    realize_attempts: int = 4      # carve retries w/ perturbed seeds
    top_up_candidates: bool = True # keep generating until k survive (≤2k)
    # STR-003 (parking/garage must have a vehicle gate): strict (default)
    # discards every candidate that can't gate parking and relies on
    # retry/top-up to find a seed that can. If the plot is so constrained
    # that NO seed ever can, set parking_strict=False to retry once with
    # parking dropped from the program instead of returning nothing — users
    # prefer a generated plan with a clear warning over a failed request.
    parking_strict: bool = True
    # window placement (connector)
    window_habitable_min_ft: int = 3
    window_habitable_max_ft: int = 8
    window_wet_ft: int = 2
    # soft-score weights (validator)
    w_area_drift: float = 90.0
    w_narrow: float = 9.0
    w_aspect: float = 6.0
    w_privacy_door: float = 7.0
    w_wide_bonus: float = 1.5
    w_nbc: float = 12.0            # per NBC dimensional violation
    w_daylight: float = 6.0        # per habitable room w/o exterior wall
    w_window: float = 4.0          # per habitable room w/o window
    w_circ_depth: float = 2.5      # per hop beyond circulation-depth norms
    w_wet_social: float = 5.0      # per bath door into living/dining
    # reviewer M1 groups: free space, walls, doors
    fsp001_hard: bool = False      # hardens when the generator can satisfy it
    w_freespace: float = 10.0      # per FSP-001 violation while soft
    w_circ_waste: float = 1.2      # per % of circulation above 14%
    w_passage_width: float = 5.0   # per sub-3' passage / wide opening
    w_hub_centrality: float = 6.0  # per mean hop above 1.35 from hub
    w_wall_efficiency: float = 2.5 # per 0.1 below 0.70 shared ratio
    w_wall_jog: float = 1.5        # per jogging wall pair
    w_toothpick: float = 3.0       # per shared run under 1'6"
    w_hinge_stub: float = 2.0      # per door hinged into a corner
    w_swing_direction: float = 2.0 # per door swinging the wrong way
    w_cross_vent: float = 1.0      # bonus per cross-ventilated room
    # staircase (M2). floor_height_ft drives the riser count and therefore
    # the run length and the whole footprint — carve/stairs.py derives every
    # stair dimension from it, so this is the one knob that matters.
    floor_height_ft: float = 10.0
    w_stair_landing: float = 4.0   # per stair without landings at both ends
    w_stair_frontage: float = 0.4  # per % of entrance frontage a stair eats
    # CP-SAT side tools (M3). "off" = pure-python greedy everywhere;
    # "repair" = solve only when the greedy packing declares the program
    # infeasible; "always" = solve every carve. ortools is optional — a
    # missing solver silently means "off".
    #
    # DEFAULT IS "off", and that is a MEASURED decision, not caution:
    # harness/ab_configs.py over all 24 briefs (results/ab.json) gives
    # off 73.22 / repair 73.22 (identical plans, +27% wall time) /
    # always 70.80 with one brief losing its plan. The solver optimizes
    # area packing; the greedy packer's heuristics encode proportion and
    # daylight priors it has no way to express. Turn it on per-request for
    # a plot the greedy path cannot pack at all — that is what it is for.
    cpsat_mode: str = "off"        # off | repair | always
    cpsat_timeout_ms: int = 2000
    # learned critic (M6). Ranking blend weight; 0 disables the critic
    # entirely even when trained weights exist. The rules always decide
    # legality — this only reorders candidates they already accepted.
    critic_weight: float = 0.4
    # vertical / multi-floor (M7). Both are per-% penalties on the floor
    # ABOVE, scored by engine/vertical.py against the floor below.
    w_wet_stack: float = 0.25       # per % of upper wet area off the stack
    w_wall_alignment: float = 0.30  # per % of unsupported interior wall
    # how hard the floor above is pulled onto the floor below's wet cells;
    # 0 = ignore the floor below entirely (each floor planned alone)
    vertical_bias: float = 0.7

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "EngineConfig":
        return cls(**data)


@dataclass
class EngineRequest:
    """Everything the engine needs for ONE floor of one building."""
    plot_w_ft: int
    plot_h_ft: int
    entrance_side: str             # "N" | "E" | "S" | "W"
    rooms: List[RoomSpec]
    wishes: List[OpeningWish] = field(default_factory=list)
    floor_index: int = 0           # vertical context, used from multi-floor on
    n_floors: int = 1
    k: int = 4                     # candidates to generate
    seed: int = 0                  # rng seed → deterministic engine runs
    name: str = ""                 # harness brief id
    # Irregular plots: the surveyed boundary in feet, plot-local, any
    # winding. When present, plot_w_ft/plot_h_ft describe the polygon's
    # bounding box and the engine builds inside the largest rectangle that
    # fits after the setback (engine/site.py). None = a rectangular plot,
    # which is every request the engine has ever had.
    plot_polygon: Optional[List[Tuple[float, float]]] = None
    setback_ft: float = 0.0

    # ── Validation ───────────────────────────────────────────────────────
    def problems(self) -> Tuple[List[str], List[str]]:
        """Returns (errors, warnings). Errors make the request unbuildable."""
        errors: List[str] = []
        warnings: List[str] = []

        for dim, label in ((self.plot_w_ft, "width"), (self.plot_h_ft, "height")):
            if not (MIN_PLOT_FT <= dim <= MAX_PLOT_FT):
                errors.append(f"plot {label} {dim}' outside "
                              f"[{MIN_PLOT_FT}, {MAX_PLOT_FT}]")
        if self.entrance_side not in VALID_SIDES:
            errors.append(f"entrance_side {self.entrance_side!r} not in "
                          f"{VALID_SIDES}")
        if not self.rooms:
            errors.append("no rooms requested")
        if self.k < 1:
            errors.append(f"k={self.k} must be >= 1")

        seen = set()
        for spec in self.rooms:
            if spec.name in seen:
                errors.append(f"duplicate room name {spec.name!r}")
            seen.add(spec.name)
            if spec.zone not in VALID_ZONES:
                errors.append(f"{spec.name}: zone {spec.zone!r} not in "
                              f"{VALID_ZONES}")
            if spec.target_sqft <= 0:
                errors.append(f"{spec.name}: target_sqft must be positive")
            if spec.floor not in range(self.n_floors):
                errors.append(f"{spec.name}: floor {spec.floor} outside "
                              f"0..{self.n_floors - 1}")
        for wish in self.wishes:
            if wish.kind not in VALID_OPENING_KINDS:
                errors.append(f"wish {wish.room_a}<->{wish.room_b}: kind "
                              f"{wish.kind!r} invalid")
            for name in (wish.room_a, wish.room_b):
                if name not in seen:
                    errors.append(f"wish references unknown room {name!r}")

        if self.plot_polygon is not None:
            if len(self.plot_polygon) < 3:
                errors.append(f"plot_polygon needs at least 3 corners, got "
                              f"{len(self.plot_polygon)}")
            if self.setback_ft < 0:
                errors.append(f"setback {self.setback_ft}' is negative")

        plot_sqft = self.plot_w_ft * self.plot_h_ft
        total_target = sum(s.target_sqft for s in self.rooms)
        if total_target > plot_sqft * 1.4:
            errors.append(f"program ({total_target:.0f} sqft) grossly exceeds "
                          f"plot ({plot_sqft} sqft)")
        elif total_target > plot_sqft * 0.95:
            warnings.append(f"program ({total_target:.0f} sqft) is tight for "
                            f"plot ({plot_sqft} sqft); rooms will compress")
        return errors, warnings

    def validate(self) -> List[str]:
        """Raises EngineRequestError on errors; returns warnings."""
        errors, warnings = self.problems()
        if errors:
            raise EngineRequestError("; ".join(errors))
        return warnings

    # ── Serialization (harness briefs are JSON) ──────────────────────────
    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "EngineRequest":
        data = dict(data)
        data["rooms"] = [RoomSpec(**r) for r in data.get("rooms", [])]
        data["wishes"] = [OpeningWish(**w) for w in data.get("wishes", [])]
        return cls(**data)


@dataclass(frozen=True)
class Placement:
    """One room's proposed position: a seed cell + a size class.
    Deliberately coarse — exact geometry is the realizer's job."""
    room: str
    seed_cell: Tuple[int, int]     # (row, col) on SEED_GRID × SEED_GRID
    size_class: int


@dataclass
class LayoutProposal:
    """A full layout intent, in generation order (anchors first)."""
    placements: List[Placement]
    source: str                    # "fallback-prior" | "tier2-model"
    confidence: float = 1.0


@dataclass
class Verdict:
    hard: List[str]                # violations that disqualify a plan
    soft_score: float              # higher is better
    breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class Candidate:
    plan: object                   # core.grid_plan.GridPlan
    proposal: LayoutProposal
    # The request this candidate was actually built from (implicit rooms
    # included) and the room name -> face id map the reviewer used. Carried
    # so a candidate is self-describing: anything scoring or re-scoring it
    # later (the critic, the app, an export) reads the same mapping the
    # engine did instead of re-deriving one that may not match.
    request: Optional["EngineRequest"] = None
    room_ids: Dict[str, int] = field(default_factory=dict)
    verdict: Optional[Verdict] = None
    fidelity: Optional[float] = None   # 1.0 = plan matches proposal exactly
    discarded: bool = False
    repair_rounds: int = 0
    notes: List[str] = field(default_factory=list)
    timings_ms: Dict[str, float] = field(default_factory=dict)


@dataclass
class EngineResult:
    ranked: List[Candidate]        # kept candidates, best first
    discarded: List[Candidate]
    tier_notes: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def best(self) -> Optional[Candidate]:
        return self.ranked[0] if self.ranked else None
