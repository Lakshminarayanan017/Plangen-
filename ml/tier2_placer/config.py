"""
config.py — the Tier-2 placer's frozen task contract and model dims.

The task contract (grid sizes, vocab) MUST match engine/contracts.py and the
data pipeline (training/prep_cubicasa.py). The model dims follow
implementation_plan_v2.md §2.2 (sized for a Colab budget, ~9-12M params).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from modules.step4_generate.engine.contracts import SEED_GRID, SIZE_CLASS_MAX

# ── task contract (must not drift from the engine / prep) ────────────────────
CELL_COUNT = SEED_GRID * SEED_GRID          # 1024 discrete seed cells
SIZE_COUNT = SIZE_CLASS_MAX                 # 40 size classes (1..40 -> 0..39)
BOUNDARY_GRID = 64
GLOBAL_DIM = 4          # [log w, log h, log aspect, n_rooms/scale]

# The model's room-type vocabulary: every engine rtype the placer might see at
# inference (superset of the training corpus types), plus <pad> at index 0 and
# <other> for anything unmapped. Fixed order — the id is the token.
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

MAX_ROOMS = 24          # sequence length cap (studio..large villa fit under this)


def room_type_id(rtype: str) -> int:
    return ROOM_TYPE_TO_ID.get(rtype, OTHER_ID)


def zone_id(zone: str) -> int:
    return ZONE_TO_ID.get(zone, 0)


@dataclass
class PlacerConfig:
    # vocab / task
    n_room_types: int = len(ROOM_TYPES)
    n_zones: int = len(ZONES)
    cell_count: int = CELL_COUNT
    size_count: int = SIZE_COUNT
    boundary_grid: int = BOUNDARY_GRID
    max_rooms: int = MAX_ROOMS
    # program (GNN) encoder
    node_dim: int = 128
    gnn_layers: int = 3
    gnn_heads: int = 4
    # boundary (CNN) encoder -> 8x8 memory tokens
    cnn_dim: int = 128
    # decoder
    d_model: int = 256
    dec_layers: int = 6
    dec_heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.1

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PlacerConfig":
        fields = {f for f in cls().__dict__}
        return cls(**{k: v for k, v in d.items() if k in fields})
