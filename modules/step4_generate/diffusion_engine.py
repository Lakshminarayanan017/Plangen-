"""
diffusion_engine.py
===================
DiffusionLayoutEngine — wires the trained GNN + Diffusion model into the
Step 4 layout pipeline.

This module is the bridge between:
  • Trained weights  (gnn_encoder.npz  +  diffusion_model.npz)
  • Runtime input    (EnrichedRoom list  +  adjacency graph)
  • Runtime output   (room placements in feet, ready for PlacedRoom)

Architecture
------------
  EnrichedRoom list
        │
        ▼  (feature engineering — same schema as training data)
  node_features  (N, 24)   +   edge_index  (2, E)   +   edge_features  (E, 7)
        │
        ▼  GNNEncoderNumpy.forward_numpy()
  node_embeddings  (N, 256)   +   global_embedding  (1, 256)
        │
        ▼  DiffusionDecoderNumpy.sample_numpy()  — DDIM 50 steps
  predicted_boxes  (N, 4)   — [x1, y1, x2, y2] normalised [0, 1]
        │
        ▼  scale to feet  +  snap to grid  +  overlap resolution
  PlacedRoom list  (x_ft, y_ft, width_ft, length_ft)

Fallback Behaviour
------------------
  If weights files are missing (training not done yet):
      is_ready() → False
      place_floor() → raises RuntimeError   (generator falls back to CP-SAT)

  If inference produces invalid boxes (all zeros, NaN, etc.):
      place_floor() raises RuntimeError   (generator falls back to CP-SAT)

Node Feature Schema (24-dim) — matches training data_prep.py exactly:
  [0:16]  one-hot room type (16 unified classes)
  [16]    normalised width   (target_width_ft  / net_w)
  [17]    normalised height  (target_length_ft / net_l)
  [18]    area_ratio         (target_area / total_floor_area)
  [19]    aspect_ratio       (w/h, clipped [0.1,10], then /10)
  [20]    vastu_zone_cos     (cos of preferred compass direction)
  [21]    vastu_zone_sin     (sin of preferred compass direction)
  [22]    is_attached_bath   (1.0 if attached bathroom/bedroom pair)
  [23]    floor_level        (preferred_floor / 2.0, max 2)

Edge Feature Schema (7-dim):
  [0]    learned_adj_weight  (from adjacency_graph, default 1.0)
  [1]    is_door_shared      (0.5 default — unknown at inference time)
  [2]    is_wall_shared      (1.0 — all graph edges share a wall)
  [3]    cos(direction)      (from room A zone to room B zone)
  [4]    sin(direction)
  [5]    dist_percentile     (0.5 default)
  [6]    room_count_diff     (0.0 — same floor)
"""

from __future__ import annotations

import math
import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from models import EnrichedPlan, EnrichedRoom, PlacedRoom

log = logging.getLogger("PlanGen.DiffusionEngine")

# ── Weight paths ──────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
_WEIGHTS_DIR = os.path.join(_ROOT, "weights")
GNN_WEIGHTS_PATH      = os.path.join(_WEIGHTS_DIR, "gnn_encoder.npz")
DIFFUSION_WEIGHTS_PATH = os.path.join(_WEIGHTS_DIR, "diffusion_model.npz")

# ── Room type vocabulary (must match training data_prep.py ROOM_VOCAB) ─────────
_ROOM_VOCAB: Dict[str, int] = {
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
_NUM_TYPES = 16  # one-hot vector length

# Map from EnrichedRoom.room_type → ROOM_VOCAB index
_TYPE_MAP: Dict[str, int] = {
    "living_room":       0,
    "drawing_room":      0,
    "master_bedroom":    1,
    "bedroom":           2,
    "kids_bedroom":      2,
    "kitchen":           3,
    "dining_room":       4,
    "bathroom":          5,
    "toilet":            5,
    "attached_bathroom": 5,
    "balcony":           6,
    "terrace":           6,
    "verandah":          6,
    "study_room":        7,
    "office":            7,
    "store_room":        8,
    "servant_room":      8,
    "garden":            9,
    "outdoor":           9,
    "staircase":        10,
    "passage":          10,
    "foyer":            10,
    "lobby":            10,
    "car_parking":      11,
    "garage":           11,
    "laundry":          12,
    "utility":          14,
    "pooja_room":       15,
    "puja_room":        15,
    "undefined":        15,
}

# Compass direction → angle in degrees (0=East, CCW)
# We use standard math convention for cos/sin
_DIR_ANGLE_DEG: Dict[str, float] = {
    "N":  90.0,
    "NE": 45.0,
    "E":   0.0,
    "SE": 315.0,
    "S":  270.0,
    "SW": 225.0,
    "W":  180.0,
    "NW": 135.0,
}

# Zone → approximate fractional position in [0,1]² (for direction encoding)
_ZONE_POS: Dict[str, Tuple[float, float]] = {
    "front":  (0.5, 0.15),   # near road / entrance
    "middle": (0.5, 0.50),
    "back":   (0.5, 0.85),
}

# ── Minimum room dimension (ft) — post-placement validation ──────────────────
_MIN_ROOM_DIM_FT = 4.0
_SNAP_GRID_FT    = 0.5    # snap coordinates to 0.5 ft grid


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def _room_type_onehot(room_type: str) -> np.ndarray:
    """16-dim one-hot vector for room type."""
    idx = _TYPE_MAP.get(room_type.lower(), 15)   # default → undefined
    v   = np.zeros(_NUM_TYPES, dtype=np.float32)
    v[idx] = 1.0
    return v


def _compass_sincos(direction: str) -> Tuple[float, float]:
    """Return (cos, sin) for a compass direction label."""
    angle_deg = _DIR_ANGLE_DEG.get(direction.upper().replace("_", ""), 90.0)
    rad = math.radians(angle_deg)
    return math.cos(rad), math.sin(rad)


def _build_node_features(
    rooms:           List[EnrichedRoom],
    net_w:           float,
    net_l:           float,
    total_area:      float,
    attached_pairs:  Dict[str, str],   # room_id → attached_bathroom_id
    floor_level:     int,
    max_floors:      int,
) -> np.ndarray:
    """
    Build (N, 24) node feature matrix from a list of EnrichedRooms.
    All values in [0, 1] or reasonable normalised range.
    """
    N = len(rooms)
    feat = np.zeros((N, 24), dtype=np.float32)

    for i, room in enumerate(rooms):
        # [0:16] — room type one-hot
        feat[i, 0:16] = _room_type_onehot(room.room_type)

        # [16] — normalised width
        feat[i, 16] = float(np.clip(room.target_width_ft / max(net_w, 1.0), 0.0, 1.0))

        # [17] — normalised height (length in y-direction)
        feat[i, 17] = float(np.clip(room.target_length_ft / max(net_l, 1.0), 0.0, 1.0))

        # [18] — area ratio
        feat[i, 18] = float(np.clip(room.target_area_sqft / max(total_area, 1.0), 0.0, 1.0))

        # [19] — aspect ratio (clipped [0.1, 10], then /10)
        ar = room.target_width_ft / max(room.target_length_ft, 0.1)
        ar = float(np.clip(ar, 0.1, 10.0)) / 10.0
        feat[i, 19] = ar

        # [20,21] — vastu zone cos/sin
        cos_v, sin_v = _compass_sincos(room.preferred_direction)
        feat[i, 20] = float(cos_v)
        feat[i, 21] = float(sin_v)

        # [22] — is_attached_bath (bedroom has attached bathroom or vice versa)
        is_attached = (
            room.room_id in attached_pairs or
            room.room_id in attached_pairs.values()
        )
        feat[i, 22] = 1.0 if is_attached else 0.0

        # [23] — floor_level (0.0=ground, 0.5=first, 1.0=second)
        feat[i, 23] = float(floor_level) / max(float(max_floors - 1), 1.0)

    return feat


def _build_edge_index_and_features(
    rooms:            List[EnrichedRoom],
    adjacency_graph:  Dict[str, Dict[str, float]],   # room_id → room_id → weight
    net_w:            float,
    net_l:            float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build edge_index (2, E) and edge_features (E, 7) from adjacency graph.

    All pairs with non-zero adjacency weight become edges (bidirectional).
    Fully connected graph (all room pairs) is also used as a fallback.
    """
    room_ids  = [r.room_id for r in rooms]
    id_to_idx = {rid: i for i, rid in enumerate(room_ids)}
    N         = len(rooms)

    # Zone position lookup
    zone_pos: Dict[str, Tuple[float, float]] = {}
    for r in rooms:
        zone_pos[r.room_id] = _ZONE_POS.get(r.preferred_zone, (0.5, 0.5))

    # Collect edges
    edges: List[Tuple[int, int, float]] = []

    # From adjacency_graph
    for rid_a, neighbours in adjacency_graph.items():
        if rid_a not in id_to_idx:
            continue
        i = id_to_idx[rid_a]
        for rid_b, weight in neighbours.items():
            if rid_b not in id_to_idx:
                continue
            j = id_to_idx[rid_b]
            if i != j and weight > 0:
                edges.append((i, j, float(weight)))

    # If sparse, add all-pairs (fully connected with low weight)
    if len(edges) < N * (N - 1) // 2:
        existing = {(a, b) for a, b, _ in edges}
        for i in range(N):
            for j in range(i + 1, N):
                if (i, j) not in existing and (j, i) not in existing:
                    edges.append((i, j, 0.3))   # weak default link
                    edges.append((j, i, 0.3))

    if not edges:
        # Degenerate case — empty plan
        return (
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0, 7), dtype=np.float32),
        )

    E = len(edges)
    edge_index = np.zeros((2, E), dtype=np.int64)
    edge_feat  = np.zeros((E, 7), dtype=np.float32)

    floor_diag = math.sqrt(net_w**2 + net_l**2) or 1.0

    for k, (i, j, w) in enumerate(edges):
        edge_index[0, k] = i
        edge_index[1, k] = j

        # [0] learned adj weight (clamped to [0,1])
        edge_feat[k, 0] = float(np.clip(w, 0.0, 1.0))

        # [1] is_door_shared — unknown at inference, default 0.5
        edge_feat[k, 1] = 0.5

        # [2] is_wall_shared — 1.0 for all explicit adjacency edges
        edge_feat[k, 2] = 1.0 if w > 0.1 else 0.5

        # [3,4] cos/sin of direction from zone A to zone B
        za = zone_pos.get(room_ids[i], (0.5, 0.5))
        zb = zone_pos.get(room_ids[j], (0.5, 0.5))
        dx = (zb[0] - za[0]) * net_w
        dy = (zb[1] - za[1]) * net_l
        dist = math.sqrt(dx**2 + dy**2) or 1.0
        edge_feat[k, 3] = float(dx / dist)
        edge_feat[k, 4] = float(dy / dist)

        # [5] dist_percentile (estimated from zone centroids)
        edge_feat[k, 5] = float(np.clip(dist / floor_diag, 0.0, 1.0))

        # [6] room_count_diff — always 0 (same floor)
        edge_feat[k, 6] = 0.0

    return edge_index, edge_feat


# ─────────────────────────────────────────────────────────────────────────────
# BOX POST-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def _snap(val: float, grid: float = _SNAP_GRID_FT) -> float:
    """Snap value to nearest grid increment."""
    return round(val / grid) * grid


def _resolve_boxes(
    boxes_norm:  np.ndarray,   # (N, 4) [x1,y1,x2,y2] in [0,1]
    rooms:       List[EnrichedRoom],
    net_w:       float,
    net_l:       float,
) -> List[Tuple[str, float, float, float, float]]:
    """
    Convert normalised [0,1] boxes to feet, enforce minimum sizes,
    snap to grid, and push apart overlapping rooms.

    Returns: list of (room_id, x_ft, y_ft, width_ft, length_ft)
    """
    results: List[Tuple[str, float, float, float, float]] = []

    for i, room in enumerate(rooms):
        x1, y1, x2, y2 = boxes_norm[i]

        # Scale to feet
        x1 = float(x1) * net_w
        y1 = float(y1) * net_l
        x2 = float(x2) * net_w
        y2 = float(y2) * net_l

        # Ensure x1 < x2, y1 < y2
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1

        # Preserve predicted centre; decide which dimension to use
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        tw = max(room.target_width_ft,  room.min_width_ft,  _MIN_ROOM_DIM_FT)
        tl = max(room.target_length_ft, room.min_length_ft, _MIN_ROOM_DIM_FT)

        pred_w = x2 - x1
        pred_l = y2 - y1

        # FIX: use a sensible max-width ceiling (max_area / min_length)
        max_w = room.max_area_sqft / max(room.min_length_ft, 1.0)
        max_l = room.max_area_sqft / max(room.min_width_ft,  1.0)
        use_w = pred_w if room.min_width_ft  <= pred_w <= max_w else tw
        use_l = pred_l if room.min_length_ft <= pred_l <= max_l else tl

        x1 = cx - use_w / 2.0
        x2 = cx + use_w / 2.0
        y1 = cy - use_l / 2.0
        y2 = cy + use_l / 2.0

        # Clamp to buildable area
        x1 = max(0.0, min(x1, net_w - use_w))
        y1 = max(0.0, min(y1, net_l - use_l))

        # Snap to grid
        x1 = _snap(x1)
        y1 = _snap(y1)
        w  = _snap(use_w)
        l  = _snap(use_l)

        # Final clamp after snap
        w  = max(w, room.min_width_ft)
        l  = max(l, room.min_length_ft)
        x1 = max(0.0, min(x1, net_w - w))
        y1 = max(0.0, min(y1, net_l - l))

        results.append((room.room_id, x1, y1, w, l))

    # ── Push-apart: iteratively separate overlapping rooms ────────────────────
    results = _push_apart_overlaps(results, net_w, net_l)

    return results


def _push_apart_overlaps(
    placements: List[Tuple[str, float, float, float, float]],
    net_w:      float,
    net_l:      float,
    iterations: int = 15,
    min_overlap: float = 0.5,   # ft — ignore tiny numerical overlaps
) -> List[Tuple[str, float, float, float, float]]:
    """
    Iteratively push rooms apart along the axis with the smaller overlap.
    Guarantees all rooms stay within [0, net_w] × [0, net_l].
    """
    result = list(placements)

    for _ in range(iterations):
        moved = False

        for i in range(len(result)):
            rid_i, xi, yi, wi, li = result[i]

            for j in range(i + 1, len(result)):
                rid_j, xj, yj, wj, lj = result[j]

                ox = min(xi + wi, xj + wj) - max(xi, xj)
                oy = min(yi + li, yj + lj) - max(yi, yj)

                if ox < min_overlap or oy < min_overlap:
                    continue   # no significant overlap

                # Push along the axis with smaller penetration depth
                if ox <= oy:
                    push = ox / 2.0 + _SNAP_GRID_FT
                    # Move the left room left, right room right
                    if xi < xj:
                        xi = max(0.0, xi - push)
                        xj = min(net_w - wj, xj + push)
                    else:
                        xj = max(0.0, xj - push)
                        xi = min(net_w - wi, xi + push)
                else:
                    push = oy / 2.0 + _SNAP_GRID_FT
                    if yi < yj:
                        yi = max(0.0, yi - push)
                        yj = min(net_l - lj, yj + push)
                    else:
                        yj = max(0.0, yj - push)
                        yi = min(net_l - li, yi + push)

                result[i] = (rid_i, xi, yi, wi, li)
                result[j] = (rid_j, xj, yj, wj, lj)
                moved = True

        if not moved:
            break   # converged — no more overlaps to resolve

    return result


def _validate_boxes(
    placements: List[Tuple[str, float, float, float, float]],
    net_w: float,
    net_l: float,
) -> bool:
    """
    Basic sanity checks on the diffusion output.
    Returns True if output looks usable, False if garbage.
    """
    if not placements:
        return False

    all_same = len(set((x, y) for _, x, y, _, _ in placements)) <= 1
    if all_same:
        log.warning("DiffusionEngine: all boxes at same position — output degenerate")
        return False

    out_of_bounds = sum(
        1 for _, x, y, w, l in placements
        if x < -1 or y < -1 or x + w > net_w + 1 or y + l > net_l + 1
    )
    if out_of_bounds > len(placements) // 2:
        log.warning("DiffusionEngine: >50%% of boxes out of bounds")
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENGINE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class DiffusionLayoutEngine:
    """
    Inference engine: uses trained GNN + Diffusion weights to predict
    room placements for a given floor.

    Usage
    -----
    engine = DiffusionLayoutEngine()
    if engine.is_ready():
        placements = engine.place_floor(rooms, adjacency_graph, net_w, net_l, ...)
    else:
        # fall back to CP-SAT
    """

    def __init__(
        self,
        gnn_path:      Optional[str] = None,
        diffusion_path: Optional[str] = None,
        ddim_steps:    int = 50,
    ):
        self._gnn_path  = gnn_path      or GNN_WEIGHTS_PATH
        self._diff_path = diffusion_path or DIFFUSION_WEIGHTS_PATH
        self._ddim_steps = ddim_steps
        self._gnn_enc  = None
        self._diff_dec = None
        self._loaded   = False
        self._load_weights()

    def _load_weights(self):
        """Lazily load both weight files. Silently skips if missing."""
        if not os.path.exists(self._gnn_path):
            log.info("GNN weights not found at %s — diffusion engine inactive",
                     self._gnn_path)
            return
        if not os.path.exists(self._diff_path):
            log.info("Diffusion weights not found at %s — diffusion engine inactive",
                     self._diff_path)
            return

        try:
            from modules.step4_generate.gnn_encoder import GNNEncoderNumpy
            from modules.step4_generate.diffusion_decoder import DiffusionDecoderNumpy

            self._gnn_enc  = GNNEncoderNumpy(self._gnn_path)
            self._diff_dec = DiffusionDecoderNumpy(self._diff_path)
            self._loaded   = True

            log.info(
                "✓ DiffusionLayoutEngine loaded: GNN=%s, Diffusion=%s",
                os.path.basename(self._gnn_path),
                os.path.basename(self._diff_path),
            )
        except Exception as exc:
            log.warning("Failed to load diffusion weights: %s", exc, exc_info=True)
            self._loaded = False

    def is_ready(self) -> bool:
        """True if both weight files are loaded and engine is functional."""
        return self._loaded and self._gnn_enc is not None and self._diff_dec is not None

    def place_floor(
        self,
        rooms:           List[EnrichedRoom],
        adjacency_graph: Dict[str, Dict[str, float]],
        net_w:           float,
        net_l:           float,
        floor_number:    int = 0,
        total_floors:    int = 1,
    ) -> List[Tuple[str, float, float, float, float]]:
        """
        Predict room placements for one floor using GNN + Diffusion.

        Parameters
        ----------
        rooms            : EnrichedRoom list for this floor.
        adjacency_graph  : Per-instance adjacency weights (room_id → room_id → weight).
        net_w, net_l     : Net buildable dimensions in feet.
        floor_number     : 0 = ground, 1 = first, etc.
        total_floors     : Total floors in the plan (for floor_level feature).

        Returns
        -------
        List of (room_id, x_ft, y_ft, width_ft, length_ft).

        Raises
        ------
        RuntimeError : if weights not loaded or inference output is degenerate.
        """
        if not self.is_ready():
            raise RuntimeError("DiffusionLayoutEngine: weights not loaded")

        if not rooms:
            return []

        N = len(rooms)
        log.info("DiffusionEngine: placing %d rooms on floor %d (%.1f×%.1f ft)",
                 N, floor_number, net_w, net_l)

        # ── Build attached_pairs map ──────────────────────────────────────────
        attached_pairs: Dict[str, str] = {}
        for room in rooms:
            if room.attached_bathroom_id:
                attached_pairs[room.room_id] = room.attached_bathroom_id

        # ── Compute total floor area ──────────────────────────────────────────
        total_area = sum(r.target_area_sqft for r in rooms) or 1.0

        # ── Build node features ───────────────────────────────────────────────
        node_feat = _build_node_features(
            rooms        = rooms,
            net_w        = net_w,
            net_l        = net_l,
            total_area   = total_area,
            attached_pairs = attached_pairs,
            floor_level  = floor_number,
            max_floors   = total_floors,
        )   # (N, 24)

        # ── Build edge graph ──────────────────────────────────────────────────
        edge_index, edge_feat = _build_edge_index_and_features(
            rooms           = rooms,
            adjacency_graph = adjacency_graph,
            net_w           = net_w,
            net_l           = net_l,
        )   # (2, E), (E, 7)

        log.debug("Node features: %s  Edge graph: %s edges", node_feat.shape, edge_index.shape[1])

        # ── GNN Encoder ───────────────────────────────────────────────────────
        try:
            node_emb, global_emb = self._gnn_enc.forward_numpy(
                node_feat, edge_index, edge_feat
            )   # (N, 256), (1, 256)
        except Exception as exc:
            raise RuntimeError(f"GNN encoder failed: {exc}") from exc

        # ── Diffusion Decoder (DDIM sampling) ─────────────────────────────────
        try:
            boxes_norm, _floor_pred = self._diff_dec.sample_numpy(
                node_emb, global_emb
            )   # boxes: (N, 4) — [x1,y1,x2,y2] in [0,1]
        except Exception as exc:
            raise RuntimeError(f"Diffusion decoder failed: {exc}") from exc

        # ── Validate raw output ───────────────────────────────────────────────
        if boxes_norm is None or np.any(np.isnan(boxes_norm)):
            raise RuntimeError("Diffusion decoder returned NaN boxes")

        # ── Post-process: scale to feet, snap, enforce sizes ──────────────────
        placements = _resolve_boxes(boxes_norm, rooms, net_w, net_l)

        # ── Sanity check ──────────────────────────────────────────────────────
        if not _validate_boxes(placements, net_w, net_l):
            raise RuntimeError("Diffusion decoder output failed validation")

        log.info(
            "DiffusionEngine: placed %d rooms in [%.1f×%.1f ft]",
            len(placements), net_w, net_l,
        )
        return placements


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON (lazy)
# ─────────────────────────────────────────────────────────────────────────────

_engine_singleton: Optional[DiffusionLayoutEngine] = None


def get_diffusion_engine() -> DiffusionLayoutEngine:
    """
    Return the module-level singleton DiffusionLayoutEngine.
    Instantiated once and reused across all calls.
    """
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = DiffusionLayoutEngine()
    return _engine_singleton
