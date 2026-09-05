"""
config.py — Placer v3's task contract and model dimensions.

v3 keeps v2's FROZEN contract with the engine — the 32x32 seed grid and the
40 size classes come from `engine/contracts.py` and must never drift — and
adds the four things the v2 post-mortem showed were missing:

  R1  a per-step SPATIAL STATE the decoder can look at (v2 conditioned only
      on the previous room's cell/size embeddings, so it packed blind)
  R3  aspect / orientation / band heads, so the carver receives proportion
      and structure instead of deriving everything from one seed cell
  R4  Vastu direction + scale-regime conditioning
  R5  capacity that is grown by measurement, not chosen upfront

Everything here is vocabulary and dimensions only — no torch import — so the
NumPy inference path and the torch training path read the SAME contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List

from modules.step4_generate.engine.contracts import SEED_GRID, SIZE_CLASS_MAX

# ── frozen task contract (must match the engine and the data prep) ──────────
CELL_COUNT = SEED_GRID * SEED_GRID          # 1024 discrete seed cells
SIZE_COUNT = SIZE_CLASS_MAX                 # 40 size classes (1..40 -> 0..39)
BOUNDARY_GRID = 64                          # static footprint raster
MAX_ROOMS = 24

# ── R1: the per-step state raster ───────────────────────────────────────────
# Deliberately on SEED_GRID, not BOUNDARY_GRID: the state exists to tell the
# decoder which SEED CELLS are still available, so it should live in the same
# space the cell head predicts over. One 32x32 plane per channel per step.
STATE_GRID = SEED_GRID
STATE_CHANNELS = 5
#   0  buildable footprint            (static, repeated per step)
#   1  entrance-edge band             (static, repeated per step)
#   2  cells claimed by ANY placed room
#   3  cells claimed by PUBLIC-zone rooms
#   4  cells claimed by PRIVATE-zone rooms
# Channels 3/4 are what make zoning learnable from state rather than from
# sequence memory: "the public half is used up, put the bedroom elsewhere".
STATE_CH_FOOTPRINT = 0
STATE_CH_ENTRANCE = 1
STATE_CH_CLAIMED = 2
STATE_CH_PUBLIC = 3
STATE_CH_PRIVATE = 4

# ── room-type vocabulary (superset of every engine rtype) ───────────────────
ROOM_TYPES: List[str] = [
    "<pad>", "<other>",
    # public / social
    "living_room", "drawing_room", "dining_room", "foyer", "hallway",
    "passage", "pooja_room",
    # service
    "kitchen", "store", "storage", "utility", "laundry",
    # private
    "master_bedroom", "bedroom", "study", "office",
    # wet
    "bathroom", "toilet",
    # parking / stair / misc
    "parking", "garage", "staircase", "ots",
]
ROOM_TYPE_TO_ID = {t: i for i, t in enumerate(ROOM_TYPES)}
PAD_ID = 0
OTHER_ID = 1

ZONES: List[str] = ["<pad>", "public", "service", "private"]
ZONE_TO_ID = {z: i for i, z in enumerate(ZONES)}

# ── R4: Vastu conditioning ──────────────────────────────────────────────────
# Index 0 is "no Vastu opinion about this room", which is the honest state for
# most rooms and for every non-Vastu request — NOT a missing value.
VASTU_DIRS: List[str] = ["none", "N", "NE", "E", "SE", "S", "SW", "W", "NW"]
VASTU_DIR_TO_ID = {d: i for i, d in enumerate(VASTU_DIRS)}
# soft (the rule loader's default) vs hard (promoted rooms: pooja, kitchen)
VASTU_STRENGTHS: List[str] = ["none", "soft", "hard"]
VASTU_STRENGTH_TO_ID = {s: i for i, s in enumerate(VASTU_STRENGTHS)}

# ── R4: scale regime ────────────────────────────────────────────────────────
# Measured against the engine's own behaviour: below ~600 sqft buildable the
# greedy packer starts failing on door swings and minimum sides; above ~1800
# the program under-fills and rooms inflate into slabs. Those two numbers are
# where the STRATEGY should change, so the model is told which side it is on.
REGIMES: List[str] = ["compact", "normal", "spacious"]
REGIME_TO_ID = {r: i for i, r in enumerate(REGIMES)}
REGIME_COMPACT_MAX_SQFT = 600.0
REGIME_SPACIOUS_MIN_SQFT = 1800.0


def regime_for(plot_sqft: float) -> str:
    if plot_sqft <= REGIME_COMPACT_MAX_SQFT:
        return "compact"
    if plot_sqft >= REGIME_SPACIOUS_MIN_SQFT:
        return "spacious"
    return "normal"


# ── R3: aspect + orientation vocabularies ───────────────────────────────────
# Upper bounds on long/short ratio. The last bucket is open-ended and is the
# "this came out a slab" class the reviewer's QLT-003 already penalises.
ASPECT_EDGES = (1.15, 1.40, 1.70, 2.10, 2.60)
ASPECT_COUNT = len(ASPECT_EDGES) + 1        # 6

ORIENTATIONS: List[str] = ["square", "horizontal", "vertical"]
ORIENTATION_COUNT = len(ORIENTATIONS)

# ── R3: band structure ──────────────────────────────────────────────────────
# The carver packs rooms into depth bands measured from the entrance. Which
# band a room lands in is a GLOBAL decision the seed cell alone does not make,
# so predicting it is real information, not a restatement of the cell.
MAX_BANDS = 6

# ── global plot-context vector ──────────────────────────────────────────────
#  0 log(w) - 3          4 regime one-hot: compact
#  1 log(h) - 3          5 regime one-hot: normal
#  2 log(w/h)            6 regime one-hot: spacious
#  3 n_rooms / 12        7 log(area) - 6.5
#                        8 program density (sum target / plot area)
GLOBAL_DIM = 9

# Label index meaning "no target here" — masked out of the loss rather than
# guessed. Used for aspect/orientation/band during CubiCasa pretraining,
# where the prep has no exact carved geometry to supervise against.
IGNORE_INDEX = -100


def room_type_id(rtype: str) -> int:
    return ROOM_TYPE_TO_ID.get(rtype, OTHER_ID)


def zone_id(zone: str) -> int:
    return ZONE_TO_ID.get(zone, 0)


def vastu_dir_id(direction: str | None) -> int:
    if not direction:
        return 0
    return VASTU_DIR_TO_ID.get(str(direction).strip().upper(), 0)


def vastu_strength_id(strength: str | None) -> int:
    if not strength:
        return 0
    return VASTU_STRENGTH_TO_ID.get(str(strength).strip().lower(), 0)


def aspect_class(long_side: float, short_side: float) -> int:
    """Long/short ratio -> aspect bucket in [0, ASPECT_COUNT)."""
    lo = max(float(min(long_side, short_side)), 1e-6)
    ratio = max(float(max(long_side, short_side)), 1e-6) / lo
    for i, edge in enumerate(ASPECT_EDGES):
        if ratio <= edge:
            return i
    return ASPECT_COUNT - 1


def orientation_class(width: float, height: float) -> int:
    """0 square / 1 wider-than-tall / 2 taller-than-wide."""
    w, h = float(width), float(height)
    lo = max(min(w, h), 1e-6)
    if max(w, h) / lo <= ASPECT_EDGES[0]:
        return 0
    return 1 if w >= h else 2


@dataclass
class PlacerV3Config:
    # ── vocab / task (frozen) ───────────────────────────────────────────
    n_room_types: int = len(ROOM_TYPES)
    n_zones: int = len(ZONES)
    n_vastu_dirs: int = len(VASTU_DIRS)
    n_vastu_strengths: int = len(VASTU_STRENGTHS)
    cell_count: int = CELL_COUNT
    size_count: int = SIZE_COUNT
    aspect_count: int = ASPECT_COUNT
    orientation_count: int = ORIENTATION_COUNT
    band_count: int = MAX_BANDS
    boundary_grid: int = BOUNDARY_GRID
    state_grid: int = STATE_GRID
    state_channels: int = STATE_CHANNELS
    max_rooms: int = MAX_ROOMS
    global_dim: int = GLOBAL_DIM

    # ── program (GNN) encoder ───────────────────────────────────────────
    node_dim: int = 128
    gnn_layers: int = 3
    gnn_heads: int = 4

    # ── boundary (CNN) encoder -> 8x8 static memory tokens ──────────────
    cnn_dim: int = 128

    # ── R1: state encoder -> state_tokens per decode step ───────────────
    state_dim: int = 96
    state_tokens: int = 16          # 4x4 pooled grid

    # ── decoder ─────────────────────────────────────────────────────────
    d_model: int = 256
    dec_layers: int = 6
    dec_heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.1

    # ── R3 auxiliary loss weights (cell/size stay at 1.0) ───────────────
    w_cell: float = 1.0
    w_size: float = 0.5
    w_aspect: float = 0.25
    w_orientation: float = 0.15
    w_band: float = 0.25

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PlacerV3Config":
        fields = set(cls().__dict__)
        return cls(**{k: v for k, v in d.items() if k in fields})


# Capacity presets (R5). Grow ONLY when the distillation train/val gap says
# the model is underfitting — at 4,430 real CubiCasa samples the small preset
# is already generous, and stage (b) is what makes headroom worth buying.
PRESET_SMALL = PlacerV3Config()                       # ~8M params
PRESET_LARGE = PlacerV3Config(
    node_dim=192, gnn_layers=4, d_model=384, dec_layers=8,
    dec_heads=12, ff_dim=1536, cnn_dim=192, state_dim=128)   # ~28M params
