"""
data_prep.py
============
Training data preparation for the Option 4 Autoregressive Layout Transformer.

Replaces the old diffusion data_prep.py which built fixed-size arrays of
(room_type, cx, cy, w, h) tensors. The new pipeline builds SEQUENCE tensors:

  GLOBAL | R0_type, R0_cx, R0_cy, R0_w, R0_h | R1_type, ... | R{N-1}_h

with generation_order determined by the same priority table used at inference.

Data sources
------------
1. CubiCasa5K cached .pkl files (from plan_indexer.py)
   Path: modules/step4_generate/weights/cache/rplan_samples_*.pkl
   Format: list of FloorPlanSample namedtuples

2. On-the-fly augmentation: random horizontal/vertical flips + small scale jitter

Output files
------------
  modules/step4_generate/weights/cache/
    ar_train_{hash}.pkl   — list of ARSample (training set)
    ar_val_{hash}.pkl     — list of ARSample (validation set)

ARSample fields
---------------
  global_ctx   : float32 (6,)       — plot context vector
  type_ids     : int32   (N,)        — ground-truth room type IDs
  boxes_norm   : float32 (N, 4)      — normalised [cx, cy, w, h]
  node_features: float32 (N, 24)     — GNN node features
  edge_index   : int32   (2, E)      — GNN edge connectivity
  edge_features: float32 (E, 7)      — GNN edge features
  n_rooms      : int                 — N (for variable-length batching)
  plan_id      : str                 — for traceability

Usage
-----
  from modules.step4_generate.training.data_prep import build_ar_datasets
  train_samples, val_samples = build_ar_datasets(cache_dir, max_samples=50000)
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import pickle
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("PlanGen.DataPrep")

# ── Room vocabulary (must match autoregressive_transformer.py) ───────────────
ROOM_VOCAB: Dict[str, int] = {
    "living_room":    0,
    "master_bedroom": 1,
    "bedroom":        2,
    "kitchen":        3,
    "dining_room":    4,
    "bathroom":       5,
    "balcony":        6,
    "study":          7,
    "storage":        8,
    "outdoor":        9,
    "hallway":       10,
    "garage":        11,
    "laundry":       12,
    "office":        13,
    "utility":       14,
    "undefined":     15,
}
NUM_ROOM_TYPES = len(ROOM_VOCAB)

# ── Generation order priority (lower = generated first) ──────────────────────
_GEN_PRIORITY: Dict[str, int] = {
    "living_room": 0, "drawing_room": 0,
    "master_bedroom": 1,
    "hallway": 2, "foyer": 2,
    "dining_room": 3, "kitchen": 3,
    "bedroom": 4,
    "study": 5, "office": 5,
    "undefined": 6,
    "balcony": 7,
    "bathroom": 8,
    "utility": 9, "storage": 9,
    "laundry": 10,
    "garage": 11,
    "outdoor": 12,
}


@dataclass
class RoomRecord:
    """A single room extracted from a CubiCasa5K plan (normalised coords)."""
    room_type:   str
    x1:          float   # normalised [0, 1] by plan width
    y1:          float
    x2:          float
    y2:          float
    floor_level: float = 0.0   # 0.0 = ground, 1.0 = upper

    @property
    def cx(self) -> float: return (self.x1 + self.x2) * 0.5
    @property
    def cy(self) -> float: return (self.y1 + self.y2) * 0.5
    @property
    def w(self)  -> float: return self.x2 - self.x1
    @property
    def h(self)  -> float: return self.y2 - self.y1
    @property
    def area(self) -> float: return self.w * self.h


@dataclass
class FloorPlanSample:
    """A single floor plan with rooms, graph features, and metadata."""
    plan_id:  str
    source:   str   # "cubicasa5k" | "inference"
    rooms:    List[RoomRecord] = field(default_factory=list)

    # ── GNN tensors (populated by build_sample_tensors) ───────────────────
    node_features: Optional[np.ndarray] = None   # (N, 24)
    edge_index:    Optional[np.ndarray] = None   # (2, E)
    edge_features: Optional[np.ndarray] = None   # (E, 7)


@dataclass
class ARSample:
    """
    A single training sample for the Autoregressive Layout Transformer.

    All room-level arrays are sorted by generation_order (anchor rooms first).
    """
    plan_id:      str
    global_ctx:   np.ndarray    # float32 (6,)
    type_ids:     np.ndarray    # int32   (N,)
    boxes_norm:   np.ndarray    # float32 (N, 4) — normalised [cx, cy, w, h]
    node_features: np.ndarray   # float32 (N, 24)
    edge_index:   np.ndarray    # int32   (2, E)
    edge_features: np.ndarray   # float32 (E, 7)
    n_rooms:      int


# ── Adjacency weight table (from learned_patterns.json) ─────────────────────

class AdjacencyWeightTable:
    """
    Loads room-to-room adjacency weights from the rule-book learned patterns.
    Returns 0.0 for unknown pairs.
    """
    _cache: Optional[Dict[str, Dict[str, float]]] = None

    @classmethod
    def load(cls) -> "AdjacencyWeightTable":
        obj = cls()
        if cls._cache is None:
            try:
                from sources.rule_loader import rules as _rules
                weights: Dict[str, Dict[str, float]] = {}
                for a, b, w in _rules.get_all_adjacency_pairs():
                    weights.setdefault(a, {})[b] = w
                    weights.setdefault(b, {})[a] = w
                cls._cache = weights
            except Exception:
                cls._cache = {}
        return obj

    def get(self, a: str, b: str) -> float:
        if self._cache is None:
            self.load()
        return (self._cache or {}).get(a, {}).get(b, 0.0)


# ── Node feature construction ─────────────────────────────────────────────────

def _node_features(rooms: List[RoomRecord]) -> np.ndarray:
    """
    Build (N, 24) node feature matrix for the GNN.

    Features per room (24-dim):
      0     : type_id (normalised to [0,1] by NUM_ROOM_TYPES)
      1-4   : [cx, cy, w, h] normalised to [0,1]
      5     : area (w*h)
      6     : aspect ratio (w/h clamped to [0.25, 4])
      7     : floor_level (0.0 or 1.0)
      8-9   : sin/cos of type embedding (helps GNN distinguish types)
      10-15 : one-hot zone (front/mid/back × left/right, 3×2 coarse grid)
      16-23 : sinusoidal position encoding of (cx, cy)
    """
    N = len(rooms)
    feats = np.zeros((N, 24), dtype=np.float32)

    for i, r in enumerate(rooms):
        tid = ROOM_VOCAB.get(r.room_type, ROOM_VOCAB["undefined"])
        cx, cy, w, h = r.cx, r.cy, r.w, r.h
        area   = w * h
        aspect = np.clip(w / max(h, 1e-6), 0.25, 4.0)

        feats[i, 0] = tid / NUM_ROOM_TYPES
        feats[i, 1] = cx
        feats[i, 2] = cy
        feats[i, 3] = w
        feats[i, 4] = h
        feats[i, 5] = area
        feats[i, 6] = aspect / 4.0          # normalised
        feats[i, 7] = r.floor_level

        # Type sinusoid
        feats[i, 8]  = math.sin(tid * math.pi / NUM_ROOM_TYPES)
        feats[i, 9]  = math.cos(tid * math.pi / NUM_ROOM_TYPES)

        # Coarse zone (3 rows × 2 cols = 6 cells)
        col = min(int(cx * 2), 1)   # 0=left, 1=right
        row = min(int(cy * 3), 2)   # 0=bottom, 1=mid, 2=top
        feats[i, 10 + row * 2 + col] = 1.0

        # Sinusoidal position (4 sin + 4 cos for cx and cy = 8)
        for k in range(4):
            freq = 2 ** k * math.pi
            feats[i, 16 + k]     = math.sin(freq * cx)
            feats[i, 16 + 4 + k] = math.sin(freq * cy)

    return feats


def _edge_index_and_features(
    rooms:    List[RoomRecord],
    adj_tbl:  AdjacencyWeightTable,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build fully-connected undirected edge set (all pairs i→j and j→i) with
    a 7-dim edge feature vector.

    Edge features:
      0: delta_cx  (cx_j - cx_i)
      1: delta_cy  (cy_j - cy_i)
      2: dist      (Euclidean distance between centres, normalised by sqrt(2))
      3: overlap_x (horizontal overlap / min_width, -ve if gap)
      4: overlap_y (vertical overlap / min_height, -ve if gap)
      5: adj_weight (learned adjacency weight from rule book, normalised)
      6: same_floor (1.0 if both rooms on same floor, else 0.0)
    """
    N = len(rooms)
    if N <= 1:
        return np.zeros((2, 0), dtype=np.int32), np.zeros((0, 7), dtype=np.float32)

    src, dst = [], []
    for i in range(N):
        for j in range(N):
            if i != j:
                src.append(i)
                dst.append(j)

    E = len(src)
    edge_idx  = np.array([src, dst], dtype=np.int32)
    edge_feat = np.zeros((E, 7), dtype=np.float32)

    max_adj_w = 10.0  # normalisation constant

    for k in range(E):
        i, j = src[k], dst[k]
        ri, rj = rooms[i], rooms[j]

        dcx = rj.cx - ri.cx
        dcy = rj.cy - ri.cy
        dist = math.hypot(dcx, dcy) / math.sqrt(2.0)

        # Overlap (positive = touching/overlapping, negative = gap)
        ox = min(ri.x2, rj.x2) - max(ri.x1, rj.x1)
        oy = min(ri.y2, rj.y2) - max(ri.y1, rj.y1)
        min_w = max(min(ri.w, rj.w), 1e-6)
        min_h = max(min(ri.h, rj.h), 1e-6)

        adj_w = adj_tbl.get(ri.room_type, rj.room_type)

        edge_feat[k, 0] = np.clip(dcx, -1.0, 1.0)
        edge_feat[k, 1] = np.clip(dcy, -1.0, 1.0)
        edge_feat[k, 2] = np.clip(dist, 0.0, 1.0)
        edge_feat[k, 3] = np.clip(ox / min_w, -2.0, 2.0)
        edge_feat[k, 4] = np.clip(oy / min_h, -2.0, 2.0)
        edge_feat[k, 5] = np.clip(adj_w / max_adj_w, -1.0, 1.0)
        edge_feat[k, 6] = 1.0 if abs(ri.floor_level - rj.floor_level) < 0.1 else 0.0

    return edge_idx, edge_feat


def build_sample_tensors(
    sample:  FloorPlanSample,
    adj_tbl: AdjacencyWeightTable,
) -> FloorPlanSample:
    """Attach node + edge tensors to a FloorPlanSample (mutates and returns it)."""
    sample.node_features = _node_features(sample.rooms)
    sample.edge_index, sample.edge_features = _edge_index_and_features(
        sample.rooms, adj_tbl
    )
    return sample


# ── Generation order assignment ───────────────────────────────────────────────

def _sort_rooms_by_generation_order(rooms: List[RoomRecord]) -> List[RoomRecord]:
    """Sort rooms by generation priority (anchor rooms first)."""
    def _key(r: RoomRecord):
        prio = _GEN_PRIORITY.get(r.room_type, 6)
        return (prio, -r.area)
    return sorted(rooms, key=_key)


# ── Global context vector ─────────────────────────────────────────────────────

def _global_context(rooms: List[RoomRecord], vastu_on: bool = False) -> np.ndarray:
    """
    Build the 6-dim global context vector for the GLOBAL token.
    The plan dimensions are assumed to be normalised to [0,1] already.
    We infer the bounding box from the rooms.
    """
    if not rooms:
        return np.zeros(6, dtype=np.float32)

    max_x = max(r.x2 for r in rooms)
    max_y = max(r.y2 for r in rooms)
    n     = len(rooms)
    # For training we don't have entrance direction — default N (0, 1)
    return np.array([
        max_x,                              # net_w / 100 proxy (already [0,1])
        max_y,                              # net_l / 100 proxy
        min(n, 25) / 25.0,                  # room count fraction
        1.0 if vastu_on else 0.0,
        1.0,                                # cos(0) = North
        0.0,                                # sin(0) = North
    ], dtype=np.float32)


# ── Data augmentation ─────────────────────────────────────────────────────────

def _augment_rooms(
    rooms: List[RoomRecord],
    rng:   random.Random,
) -> List[RoomRecord]:
    """
    Random augmentation for training:
      - 50% chance: horizontal flip (x → 1-x)
      - 50% chance: vertical flip   (y → 1-y)
      - Small scale jitter: [0.90, 1.10] on each room's dimensions
    """
    flip_h = rng.random() < 0.5
    flip_v = rng.random() < 0.5

    out: List[RoomRecord] = []
    for r in rooms:
        x1, y1, x2, y2 = r.x1, r.y1, r.x2, r.y2

        if flip_h:
            x1, x2 = 1.0 - x2, 1.0 - x1
        if flip_v:
            y1, y2 = 1.0 - y2, 1.0 - y1

        # Scale jitter (preserves centre, changes size slightly)
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        w, h   = x2 - x1, y2 - y1
        sw = rng.uniform(0.90, 1.10)
        sh = rng.uniform(0.90, 1.10)
        w2, h2 = w * sw, h * sh
        x1n = max(0.0, cx - w2 * 0.5)
        x2n = min(1.0, cx + w2 * 0.5)
        y1n = max(0.0, cy - h2 * 0.5)
        y2n = min(1.0, cy + h2 * 0.5)

        out.append(RoomRecord(
            room_type  = r.room_type,
            x1=x1n, y1=y1n, x2=x2n, y2=y2n,
            floor_level = r.floor_level,
        ))
    return out


# ── FloorPlanSample → ARSample ────────────────────────────────────────────────

def _sample_to_ar(
    sample:  FloorPlanSample,
    adj_tbl: AdjacencyWeightTable,
    augment: bool = False,
    rng:     Optional[random.Random] = None,
) -> Optional[ARSample]:
    """
    Convert a FloorPlanSample to an ARSample for training.

    Returns None if the sample has fewer than 2 rooms or invalid geometry.
    """
    rooms = sample.rooms
    if len(rooms) < 2 or len(rooms) > 25:
        return None

    # Validate geometries
    for r in rooms:
        if r.w < 0.01 or r.h < 0.01:
            return None

    if augment and rng is not None:
        rooms = _augment_rooms(rooms, rng)

    # Sort by generation order
    rooms = _sort_rooms_by_generation_order(rooms)

    # Build node/edge features on (possibly augmented) rooms
    node_feats           = _node_features(rooms)
    edge_idx, edge_feats = _edge_index_and_features(rooms, adj_tbl)

    # Build type_ids and boxes_norm
    N = len(rooms)
    type_ids   = np.array([ROOM_VOCAB.get(r.room_type, 15) for r in rooms],
                          dtype=np.int32)
    boxes_norm = np.zeros((N, 4), dtype=np.float32)
    for i, r in enumerate(rooms):
        boxes_norm[i] = [r.cx, r.cy, r.w, r.h]

    global_ctx = _global_context(rooms)

    return ARSample(
        plan_id       = sample.plan_id,
        global_ctx    = global_ctx,
        type_ids      = type_ids,
        boxes_norm    = boxes_norm,
        node_features = node_feats,
        edge_index    = edge_idx,
        edge_features = edge_feats,
        n_rooms       = N,
    )


# ── Cache loading helpers ─────────────────────────────────────────────────────

# ── CubiCasa5K / RPLAN → canonical room-type normalisation ───────────────────
# Handles both snake_case (our convention) and PascalCase (CubiCasa5K native).
_ROOM_NORM_TABLE: Dict[str, str] = {
    # CubiCasa5K PascalCase names
    "LivingRoom":     "living_room",
    "MasterRoom":     "master_bedroom",
    "Bedroom":        "bedroom",
    "SecondRoom":     "bedroom",
    "ThirdRoom":      "bedroom",
    "FourthRoom":     "bedroom",
    "Kitchen":        "kitchen",
    "DiningRoom":     "dining_room",
    "Bathroom":       "bathroom",
    "Toilet":         "bathroom",
    "Balcony":        "balcony",
    "StorageRoom":    "storage",
    "LaundryRoom":    "laundry",
    "Garage":         "garage",
    "Study":          "study",
    "Office":         "office",
    "Hallway":        "hallway",
    "Utility":        "utility",
    "Outdoor":        "outdoor",
    # RPLAN integer label → canonical string (RPLAN uses 0-based integers)
    # If room_type comes back as an integer, it maps to these:
    "0":  "living_room",
    "1":  "master_bedroom",
    "2":  "bedroom",
    "3":  "bathroom",
    "4":  "balcony",
    "5":  "storage",
    "6":  "dining_room",
    "7":  "kitchen",
    "8":  "hallway",
    "9":  "outdoor",
    "10": "study",
    # snake_case names that already match — kept for completeness
    "living_room":    "living_room",
    "master_bedroom": "master_bedroom",
    "bedroom":        "bedroom",
    "kitchen":        "kitchen",
    "dining_room":    "dining_room",
    "bathroom":       "bathroom",
    "balcony":        "balcony",
    "storage":        "storage",
    "study":          "study",
    "hallway":        "hallway",
    "outdoor":        "outdoor",
    "garage":         "garage",
    "laundry":        "laundry",
    "office":         "office",
    "utility":        "utility",
}


def _normalise_room_type(raw: Any) -> str:
    """
    Convert any room type representation (string or int) to a canonical
    lowercase string that matches ROOM_VOCAB.

    Handles:
      - CubiCasa5K PascalCase strings ("LivingRoom" → "living_room")
      - RPLAN integer labels (0 → "living_room", 7 → "kitchen", ...)
      - Our own snake_case strings (passed through unchanged)
      - Unknown types → "undefined"
    """
    if raw is None:
        return "undefined"
    # Integer label (RPLAN)
    if isinstance(raw, (int, np.integer)):
        return _ROOM_NORM_TABLE.get(str(int(raw)), "undefined")
    s = str(raw).strip()
    if s in _ROOM_NORM_TABLE:
        return _ROOM_NORM_TABLE[s]
    # Try lowercase snake_case conversion of PascalCase
    # e.g. "MasterBedroom" → "master_bedroom"
    import re
    snake = re.sub(r"([A-Z])", r"_\1", s).lstrip("_").lower()
    snake = snake.replace("__", "_")
    if snake in _ROOM_NORM_TABLE:
        return _ROOM_NORM_TABLE[snake]
    if snake in ROOM_VOCAB:
        return snake
    return "undefined"


def _probe_pkl_schema(first_room: Any) -> Dict[str, str]:
    """
    Auto-detect the field-name schema of a single room object from a pkl file.

    Returns a schema dict:
      {
        "type_field"  : str  — attribute/key name for room type
        "coord_mode"  : str  — one of "x1y1x2y2", "bbox_list", "xywh", "cxcywh"
        "floor_field" : str  — attribute/key name for floor level, or ""
      }

    Detection is done by probing the most common field names in priority order.
    The function is called ONCE per cache file; the returned schema is reused
    for all rooms in that file.

    Supported formats:
      ┌──────────────────────┬──────────────────────────────────────────────┐
      │ Format               │ Field names detected                         │
      ├──────────────────────┼──────────────────────────────────────────────┤
      │ Our convention       │ room_type, x1, y1, x2, y2                   │
      │ CubiCasa5K           │ category / type, bbox=[x1,y1,x2,y2]         │
      │ RPLAN indexer        │ room_label / label, x1, y1, x2, y2          │
      │ plan_indexer.py v1   │ room_type, bbox=[x1,y1,x2,y2] or x,y,w,h   │
      │ plan_indexer.py v2   │ room_type, cx, cy, w, h                     │
      └──────────────────────┴──────────────────────────────────────────────┘
    """

    def _has(obj, key: str) -> bool:
        if isinstance(obj, dict):
            return key in obj and obj[key] is not None
        return hasattr(obj, key) and getattr(obj, key) is not None

    def _get(obj, key: str, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    # ── Detect room-type field ─────────────────────────────────────────────
    type_field = "undefined_field"
    for candidate in ("room_type", "category", "room_label", "label", "type", "name"):
        if _has(first_room, candidate):
            type_field = candidate
            break

    # ── Detect coordinate mode ─────────────────────────────────────────────
    coord_mode = "x1y1x2y2"   # default
    if _has(first_room, "x1") and _has(first_room, "y1"):
        coord_mode = "x1y1x2y2"
    elif _has(first_room, "bbox"):
        bbox_val = _get(first_room, "bbox")
        if bbox_val is not None and len(bbox_val) >= 4:
            coord_mode = "bbox_list"
    elif _has(first_room, "cx") and _has(first_room, "cy"):
        coord_mode = "cxcywh"
    elif _has(first_room, "x") and _has(first_room, "w"):
        coord_mode = "xywh"

    # ── Detect floor field ─────────────────────────────────────────────────
    floor_field = ""
    for candidate in ("floor_level", "floor", "level", "story"):
        if _has(first_room, candidate):
            floor_field = candidate
            break

    return {
        "type_field":  type_field,
        "coord_mode":  coord_mode,
        "floor_field": floor_field,
    }


def _extract_room_record(rm: Any, schema: Dict[str, str]) -> Optional[RoomRecord]:
    """
    Extract a RoomRecord from a raw room object using a pre-probed schema.

    All coordinate formats are normalised to (x1, y1, x2, y2) in [0, 1].
    Room types are normalised via _normalise_room_type().

    Returns None if the room bbox is degenerate (zero-area or out-of-range).
    """
    def _get(obj, key: str, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    # ── Room type ──────────────────────────────────────────────────────────
    raw_type   = _get(rm, schema["type_field"], "undefined")
    room_type  = _normalise_room_type(raw_type)

    # ── Coordinates → (x1, y1, x2, y2) ───────────────────────────────────
    mode = schema["coord_mode"]
    try:
        if mode == "x1y1x2y2":
            x1 = float(_get(rm, "x1", 0.0))
            y1 = float(_get(rm, "y1", 0.0))
            x2 = float(_get(rm, "x2", x1 + 0.1))
            y2 = float(_get(rm, "y2", y1 + 0.1))

        elif mode == "bbox_list":
            bbox = _get(rm, "bbox")
            # Accept [x1, y1, x2, y2] or [y1, x1, y2, x2] (row-major)
            # We assume [x1, y1, x2, y2] (more common)
            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

        elif mode == "cxcywh":
            cx = float(_get(rm, "cx", 0.5))
            cy = float(_get(rm, "cy", 0.5))
            w  = float(_get(rm, "w",  0.1))
            h  = float(_get(rm, "h",  0.1))
            x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2

        elif mode == "xywh":
            x  = float(_get(rm, "x", 0.0))
            y  = float(_get(rm, "y", 0.0))
            w  = float(_get(rm, "w", 0.1))
            h  = float(_get(rm, "h", 0.1))
            x1, y1, x2, y2 = x, y, x + w, y + h

        else:
            return None

    except (TypeError, ValueError, IndexError):
        return None

    # ── Validate bbox ──────────────────────────────────────────────────────
    # Ensure ordering (some datasets store max before min)
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    # Must be inside [0, 1] (normalised) or [0, plan_dim] (pixel) — clamp
    x1, y1, x2, y2 = (
        max(0.0, min(x1, 0.9999)),
        max(0.0, min(y1, 0.9999)),
        max(0.0001, min(x2, 1.0)),
        max(0.0001, min(y2, 1.0)),
    )
    if (x2 - x1) < 1e-4 or (y2 - y1) < 1e-4:
        return None   # degenerate bbox — skip

    # ── Floor level ────────────────────────────────────────────────────────
    floor_level = 0.0
    if schema["floor_field"]:
        raw_fl = _get(rm, schema["floor_field"], 0.0)
        try:
            floor_level = float(raw_fl)
        except (TypeError, ValueError):
            floor_level = 0.0

    return RoomRecord(
        room_type   = room_type,
        x1          = x1,
        y1          = y1,
        x2          = x2,
        y2          = y2,
        floor_level = floor_level,
    )


def _load_raw_cache(pkl_path: str) -> List[FloorPlanSample]:
    """
    Load raw FloorPlanSample objects from a .pkl cache file.

    Handles all known formats:
      - Native FloorPlanSample objects (fastest path)
      - Dicts or namedtuples from plan_indexer.py v1/v2
      - CubiCasa5K raw format (category, bbox fields)
      - RPLAN format (room_label / integer type IDs)

    A schema probe is run once on the FIRST room object in the file.
    All subsequent rooms use the same schema — O(1) per room.
    The schema probe logs its findings at DEBUG level so misdetections
    are immediately visible in training logs.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, (list, tuple)) or len(data) == 0:
        log.warning("_load_raw_cache: empty or non-list pkl at %s", pkl_path)
        return []

    samples:       List[FloorPlanSample] = []
    schema_probed: bool = False
    schema:        Dict[str, str] = {}

    for item_idx, item in enumerate(data):
        try:
            # ── Fast path: already a FloorPlanSample ──────────────────────
            if isinstance(item, FloorPlanSample):
                # Still normalise room types in case they were stored raw
                norm_rooms = []
                for r in item.rooms:
                    norm_rt = _normalise_room_type(r.room_type)
                    if norm_rt != r.room_type:
                        norm_rooms.append(RoomRecord(
                            room_type=norm_rt, x1=r.x1, y1=r.y1,
                            x2=r.x2, y2=r.y2, floor_level=r.floor_level,
                        ))
                    else:
                        norm_rooms.append(r)
                samples.append(FloorPlanSample(
                    plan_id=item.plan_id, source=item.source, rooms=norm_rooms,
                ))
                continue

            # ── Resolve rooms list from container ─────────────────────────
            if isinstance(item, dict):
                rooms_raw  = item.get("rooms", [])
                plan_id    = str(item.get("plan_id", f"plan_{item_idx}"))
                source     = str(item.get("source", "unknown"))
            elif hasattr(item, "rooms"):
                rooms_raw  = item.rooms
                plan_id    = str(getattr(item, "plan_id", f"plan_{item_idx}"))
                source     = str(getattr(item, "source", "unknown"))
            else:
                log.debug("Skipping unknown item type %s at index %d",
                          type(item).__name__, item_idx)
                continue

            if not rooms_raw:
                continue

            # ── Probe schema from the FIRST non-RoomRecord room object ────
            if not schema_probed:
                first_rm = None
                for candidate in rooms_raw:
                    if not isinstance(candidate, RoomRecord):
                        first_rm = candidate
                        break
                if first_rm is None:
                    # All rooms are already RoomRecord — no probing needed
                    schema_probed = True
                    schema = {}
                else:
                    schema = _probe_pkl_schema(first_rm)
                    schema_probed = True
                    log.debug(
                        "_load_raw_cache %s: detected schema %s",
                        os.path.basename(pkl_path), schema,
                    )

            # ── Convert each room ─────────────────────────────────────────
            rooms: List[RoomRecord] = []
            for rm in rooms_raw:
                if isinstance(rm, RoomRecord):
                    # Normalise type even for pre-built RoomRecords
                    norm_rt = _normalise_room_type(rm.room_type)
                    rooms.append(
                        RoomRecord(
                            room_type=norm_rt, x1=rm.x1, y1=rm.y1,
                            x2=rm.x2, y2=rm.y2, floor_level=rm.floor_level,
                        ) if norm_rt != rm.room_type else rm
                    )
                else:
                    rec = _extract_room_record(rm, schema)
                    if rec is not None:
                        rooms.append(rec)

            if rooms:
                samples.append(FloorPlanSample(
                    plan_id = plan_id,
                    source  = source,
                    rooms   = rooms,
                ))

        except Exception as e:
            log.debug("Skipping malformed sample at index %d: %s", item_idx, e)

    log.debug("_load_raw_cache %s: loaded %d / %d plans",
              os.path.basename(pkl_path), len(samples), len(data))
    return samples


# ── Public API ────────────────────────────────────────────────────────────────

def build_ar_datasets(
    cache_dir:        str,
    max_train:        int   = 80_000,
    max_val:          int   = 10_000,
    augment_factor:   int   = 3,       # generate augment_factor copies per train sample
    seed:             int   = 42,
    force_rebuild:    bool  = False,
) -> Tuple[List[ARSample], List[ARSample]]:
    """
    Build AR training and validation datasets from CubiCasa5K caches.

    Args:
        cache_dir      : Path to weights/cache directory with rplan_*.pkl files
        max_train      : Maximum training samples (including augmented)
        max_val        : Maximum validation samples (no augmentation)
        augment_factor : Number of augmented copies per real training plan
        seed           : Random seed for augmentation
        force_rebuild  : Force rebuild even if cached ARSamples exist

    Returns:
        (train_samples, val_samples)
    """
    # ── Check for pre-built AR cache ──────────────────────────────────────
    params_str = f"{max_train}_{max_val}_{augment_factor}_{seed}"
    hash_key   = hashlib.md5(params_str.encode()).hexdigest()[:8]
    train_path = os.path.join(cache_dir, f"ar_train_{hash_key}.pkl")
    val_path   = os.path.join(cache_dir, f"ar_val_{hash_key}.pkl")

    if not force_rebuild and os.path.exists(train_path) and os.path.exists(val_path):
        log.info("Loading pre-built AR datasets from cache (%s)", cache_dir)
        train_sz = os.path.getsize(train_path) / (1024 ** 2)
        val_sz   = os.path.getsize(val_path)   / (1024 ** 2)
        print(f"[DataPrep] Loading AR train cache ({train_sz:.0f} MB) ... please wait", flush=True)
        with open(train_path, "rb") as f:
            train_samples = pickle.load(f)
        print(f"[DataPrep] Loading AR val cache ({val_sz:.0f} MB) ...", flush=True)
        with open(val_path, "rb") as f:
            val_samples = pickle.load(f)
        print(f"[DataPrep] ✅ Cache loaded: {len(train_samples):,} train + {len(val_samples):,} val samples", flush=True)
        log.info("Loaded %d train + %d val AR samples", len(train_samples), len(val_samples))
        return train_samples, val_samples

    # ── Load raw .pkl files ───────────────────────────────────────────────
    log.info("Building AR datasets from raw cache files in %s ...", cache_dir)
    all_raw: List[FloorPlanSample] = []
    for fname in sorted(os.listdir(cache_dir)):
        if fname.startswith("rplan_samples") and fname.endswith(".pkl"):
            path = os.path.join(cache_dir, fname)
            log.info("  Loading %s ...", fname)
            all_raw.extend(_load_raw_cache(path))

    if not all_raw:
        log.warning("No raw plan cache files found in %s — returning empty datasets", cache_dir)
        return [], []

    log.info("Loaded %d raw plans", len(all_raw))

    # ── Split into train/val ──────────────────────────────────────────────
    rng = random.Random(seed)
    rng.shuffle(all_raw)
    n_val  = min(max_val, len(all_raw) // 8)
    n_train = len(all_raw) - n_val
    raw_train = all_raw[:n_train]
    raw_val   = all_raw[n_train:n_train + n_val]

    adj_tbl = AdjacencyWeightTable.load()

    # ── Build validation set (no augmentation) ────────────────────────────
    log.info("Building validation set (%d raw → AR samples) ...", len(raw_val))
    val_samples: List[ARSample] = []
    for s in raw_val:
        ar = _sample_to_ar(s, adj_tbl, augment=False)
        if ar is not None:
            val_samples.append(ar)
        if len(val_samples) >= max_val:
            break
    log.info("  → %d validation samples", len(val_samples))

    # ── Build training set (with augmentation) ────────────────────────────
    log.info("Building training set (%d raw × %d augment) ...",
             len(raw_train), augment_factor)
    train_samples: List[ARSample] = []
    train_rng = random.Random(seed + 1)

    for s in raw_train:
        # Original (no augment)
        ar = _sample_to_ar(s, adj_tbl, augment=False)
        if ar is not None:
            train_samples.append(ar)

        # Augmented copies
        for _ in range(augment_factor - 1):
            ar_aug = _sample_to_ar(s, adj_tbl, augment=True, rng=train_rng)
            if ar_aug is not None:
                train_samples.append(ar_aug)

        if len(train_samples) >= max_train:
            break

    # Shuffle training set
    train_rng.shuffle(train_samples)
    train_samples = train_samples[:max_train]

    log.info("  → %d training samples", len(train_samples))

    # ── Cache the built datasets ──────────────────────────────────────────
    os.makedirs(cache_dir, exist_ok=True)
    with open(train_path, "wb") as f:
        pickle.dump(train_samples, f, protocol=4)
    with open(val_path, "wb") as f:
        pickle.dump(val_samples, f, protocol=4)

    log.info("Saved AR datasets to %s", cache_dir)
    return train_samples, val_samples
