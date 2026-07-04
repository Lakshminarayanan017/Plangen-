"""
autoregressive_engine.py
========================
Inference engine for the Option 4 Autoregressive Layout Transformer.

Replaces DiffusionLayoutEngine (diffusion_engine.py).

Pipeline per floor
------------------
1. Sort rooms by generation_order (enricher sets this)
2. Build GNN node features from the enriched rooms
3. Run GNNEncoderNumpy  → node_emb (N, 256), global_emb (1, 256)
4. Build LayoutTokenizer from plot dimensions
5. Run LayoutTransformerNumpy.generate()
   - room types forced from enricher (type_ids_forced)
   - positions/sizes predicted jointly by the transformer
6. Decode normalised (cx, cy, w, h) → feet
7. Post-process:
   - Clamp to buildable area
   - Resolve overlaps using soft gradient descent
   - Enforce NBC minimum widths (hard clamp)
8. Return List[PlacedRoom]

Usage
-----
  engine = AutoregressiveLayoutEngine(weights_dir="modules/step4_generate/weights")
  placed = engine.place_floor(enriched_plan, floor_number=0)
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from models import EnrichedPlan, EnrichedRoom, PlacedRoom

log = logging.getLogger("PlanGen.AREngine")

# ── Weight file names ─────────────────────────────────────────────────────────
_GNN_FILE = "gnn_encoder.npz"
_AR_FILE  = "ar_transformer.npz"

# ── Generation order priorities (lower = generated first) ────────────────────
# Anchor rooms get lower numbers → generated first → give more space
_GENERATION_PRIORITY: Dict[str, int] = {
    "living_room":    0,
    "drawing_room":   0,
    "master_bedroom": 1,
    "foyer":          2,
    "dining_room":    3,
    "kitchen":        3,
    "bedroom":        4,
    "study_room":     5,
    "study":          5,
    "pooja_room":     6,
    "balcony":        7,
    "bathroom":       8,
    "toilet":         8,
    "utility_room":   9,
    "store_room":     9,
    "staircase":      10,
    "car_parking":    11,
    "servant_room":   12,
    "hallway":        10,
    "passage":        10,
    "gym_room":       6,
    "home_theater":   5,
    "garden":         13,
    "terrace":        13,
    "verandah":       6,
}


def assign_generation_order(rooms: List[EnrichedRoom]) -> List[EnrichedRoom]:
    """
    Assign generation_order to each room based on its type priority.
    Rooms of the same priority are ordered by target_area_sqft (largest first).

    Called by the enricher after size assignment.
    """
    def _key(r: EnrichedRoom) -> Tuple[int, float]:
        prio = _GENERATION_PRIORITY.get(r.room_type, 7)
        return (prio, -r.target_area_sqft)   # largest within same priority first

    sorted_rooms = sorted(rooms, key=_key)
    for i, r in enumerate(sorted_rooms):
        r.generation_order = i
    return rooms


def assign_area_fractions(
    rooms: List[EnrichedRoom],
    net_buildable_area: float,
    floor_number: int,
) -> List[EnrichedRoom]:
    """
    Compute area_fraction for each room on a given floor.

    area_fraction = softmax(log_target_area) among rooms on that floor,
    clamped to NBC minimums.

    This gives a fractional budget allocation that:
    - Sums to 1.0 across all rooms on the floor
    - Is proportional to target_area_sqft from the statistical lookup
    - Never drops below NBC minimum (even very small rooms get ≥ 5% of their NBC min)
    """
    floor_rooms = [r for r in rooms if r.preferred_floor == floor_number]
    _EXTERNAL = {"car_parking", "garden", "terrace", "verandah", "balcony", "barsati"}
    internal   = [r for r in floor_rooms if r.room_type not in _EXTERNAL]
    external   = [r for r in floor_rooms if r.room_type in _EXTERNAL]

    if not internal:
        return rooms

    # Softmax over log(target_area)
    log_areas = np.array([math.log(max(r.target_area_sqft, 1.0)) for r in internal],
                         dtype=np.float64)
    log_areas -= log_areas.max()          # numerical stability
    weights   = np.exp(log_areas)
    weights  /= weights.sum()             # now sums to 1.0

    # Enforce NBC minimum fractions
    # Each room must get at least (nbc_min / net_area) fraction
    min_fracs = np.array([r.min_area_sqft / max(net_buildable_area, 1.0)
                          for r in internal], dtype=np.float64)
    min_fracs = np.minimum(min_fracs, 0.4)  # cap at 40% for any single room

    # Clip and re-normalise
    weights = np.maximum(weights, min_fracs)
    weights /= weights.sum()

    # Write back
    id_to_idx = {r.room_id: i for i, r in enumerate(internal)}
    for r in rooms:
        if r.room_id in id_to_idx:
            r.area_fraction = float(weights[id_to_idx[r.room_id]])
        elif r.room_type in _EXTERNAL:
            r.area_fraction = 0.0   # external rooms don't consume buildable area

    return rooms


# ── Room dimension clamping (max area + aspect ratio enforcement) ─────────────

def _clamp_rooms(
    boxes_ft:  np.ndarray,        # (N, 4): [x, y, w, h] in feet
    rooms:     List[EnrichedRoom],
    net_w:     float,
    net_l:     float,
) -> np.ndarray:
    """
    Production-grade post-placement dimension enforcement.

    Prevents two categories of garbage output from the AR model:

    1. **Oversized rooms**  — the model may assign 200+ sqft to a bathroom
       whose max_area_sqft is 50.  We scale both dimensions down proportionally
       so the room fits within its budget while preserving its shape.

    2. **Absurd aspect ratios** — e.g. 4.0 × 30.0 ft bathroom (ratio 1:7.5).
       NBC guidance and architectural practice limit habitable rooms to roughly
       1:2.5 and service rooms to 1:3.  We redistribute the excess dimension
       into the short side to approach a saner rectangle.

    Called AFTER NBC min enforcement and BEFORE overlap resolution, so overlaps
    are re-resolved on the corrected dimensions.
    """
    b = boxes_ft.copy()
    N = len(rooms)

    # Aspect-ratio limits: (min_ratio, max_ratio) where ratio = width/height
    _SERVICE_TYPES = {'bathroom', 'toilet', 'utility', 'store_room', 'storage',
                      'hallway', 'passage', 'staircase', 'balcony', 'laundry'}

    for i in range(N):
        r  = rooms[i]
        x, y, w, h = b[i]
        w = max(w, r.min_width_ft)
        h = max(h, r.min_length_ft)
        area = w * h

        # ── 1. Max-area enforcement ─────────────────────────────────────
        max_area = r.max_area_sqft
        if area > max_area * 1.05:  # 5% tolerance before clamping
            scale = math.sqrt(max_area / area)
            w *= scale
            h *= scale
            # Re-enforce NBC mins after scaling down
            w = max(w, r.min_width_ft)
            h = max(h, r.min_length_ft)
            log.debug("Clamp[%s]: area %.0f → %.0f sqft (max=%.0f)",
                      r.room_id, area, w * h, max_area)

        # ── 2. Aspect-ratio enforcement ─────────────────────────────────
        if r.room_type in _SERVICE_TYPES:
            min_ratio, max_ratio = 0.25, 4.0
        else:
            min_ratio, max_ratio = 0.35, 2.8

        ratio = w / max(h, 0.01)
        if ratio > max_ratio:
            # Too wide — redistribute width into height
            target_area = w * h
            h = math.sqrt(target_area / max_ratio)
            w = target_area / max(h, 0.01)
            w = max(w, r.min_width_ft)
            h = max(h, r.min_length_ft)
        elif ratio < min_ratio:
            # Too tall — redistribute height into width
            target_area = w * h
            w = math.sqrt(target_area * min_ratio)
            h = target_area / max(w, 0.01)
            w = max(w, r.min_width_ft)
            h = max(h, r.min_length_ft)

        # ── 3. Final hard max-area cap ──────────────────────────────────
        # Aspect-ratio redistribution can re-inflate the area, so we
        # apply a final proportional scale-down to guarantee compliance.
        final_area = w * h
        if final_area > max_area * 1.05:
            scale = math.sqrt(max_area / final_area)
            w *= scale
            h *= scale
            w = max(w, r.min_width_ft)
            h = max(h, r.min_length_ft)

        # ── 4. Re-clamp position to stay within buildable area ──────────
        x = np.clip(x, 0.0, max(0.0, net_w - w))
        y = np.clip(y, 0.0, max(0.0, net_l - h))

        b[i] = [x, y, w, h]

    return b.astype(np.float32)


# ── Overlap resolution via soft gradient descent ──────────────────────────────

def _resolve_overlaps(
    boxes:      np.ndarray,   # (N, 4): [x, y, w, h] in feet, bottom-left origin
    net_w:      float,
    net_l:      float,
    n_iters:    int   = 400,
    tolerance:  float = 0.02,   # ft — gaps smaller than this are acceptable
) -> np.ndarray:
    """
    Production-grade Projected-Gradient-Descent overlap resolver.

    Algorithm (from Autodesk Space Plan Generator, 2017, and RPLAN paper):

    Separation phase (n_iters rounds):
    ────────────────────────────────
    For each overlapping pair (i, j):
      1. Compute SAT penetration depth along X and Y axes.
      2. Choose the minimum-penetration axis for displacement (Separating-Axis
         Theorem — smallest displacement that resolves the overlap).
      3. Split the displacement inversely proportional to each room's area:
         larger rooms move less, smaller rooms move more.  This preserves
         the general structure of the anchor rooms (living room, master bedroom)
         while smaller rooms (bathrooms) absorb the adjustment.
      4. Project all rooms back into [0, net_w-w] × [0, net_l-h] after each pair.

    Early exit when total overlap energy < tolerance² × N (all rooms clean).

    Wall-snap phase (single post-pass):
    ────────────────────────────────────
    If two rooms are nearly touching (gap < SNAP_DIST ft ≈ 3.6 in), snap
    them to exact wall-to-wall contact.  This gives the clean flush-wall
    alignment seen in professional CAD drawings.

    Complexity: O(N² × n_iters) — acceptable for N ≤ 25 rooms.
    """
    b = boxes.copy().astype(np.float64)
    N = len(b)
    if N <= 1:
        return b.astype(np.float32)

    # Inverse-area weights (larger rooms are heavier → move less)
    areas   = np.maximum(b[:, 2] * b[:, 3], 1.0)
    inv_a   = 1.0 / areas

    for _iter in range(n_iters):
        total_energy = 0.0

        for i in range(N):
            for j in range(i + 1, N):
                xi, yi, wi, hi = b[i]
                xj, yj, wj, hj = b[j]

                ox = min(xi + wi, xj + wj) - max(xi, xj)
                oy = min(yi + hi, yj + hj) - max(yi, yj)

                if ox <= tolerance or oy <= tolerance:
                    continue   # no real overlap (touching is fine)

                total_energy += ox * oy

                # Weighted displacement: heavier (larger) rooms move less
                norm   = inv_a[i] + inv_a[j]
                frac_i = inv_a[i] / norm    # fraction absorbed by room i
                frac_j = inv_a[j] / norm    # fraction absorbed by room j

                if ox < oy:
                    push  = ox + tolerance
                    ci    = xi + wi * 0.5
                    cj    = xj + wj * 0.5
                    sign  = -1.0 if ci <= cj else 1.0
                    b[i, 0] += sign * push * frac_i
                    b[j, 0] -= sign * push * frac_j
                else:
                    push  = oy + tolerance
                    ci    = yi + hi * 0.5
                    cj    = yj + hj * 0.5
                    sign  = -1.0 if ci <= cj else 1.0
                    b[i, 1] += sign * push * frac_i
                    b[j, 1] -= sign * push * frac_j

        # Project every room to the feasible rectangle (size-preserving clamp)
        for i in range(N):
            b[i, 0] = np.clip(b[i, 0], 0.0, max(0.0, net_w - b[i, 2]))
            b[i, 1] = np.clip(b[i, 1], 0.0, max(0.0, net_l - b[i, 3]))

        if total_energy < tolerance * tolerance * N:
            break   # all overlaps resolved

    # ── Wall-snap post-pass ───────────────────────────────────────────────────
    # Close small gaps between nearly-touching rooms to achieve flush wall
    # alignment.  Only snap if the two rooms share a significant shared span
    # (≥ 30% of the shorter room's dimension) — avoids corner-touching false snaps.
    SNAP_DIST = 0.3   # ft (≈ 3.6 inches)

    for i in range(N):
        for j in range(i + 1, N):
            xi, yi, wi, hi = b[i]
            xj, yj, wj, hj = b[j]

            # Shared Y-span (for horizontal snap)
            y_shared = min(yi + hi, yj + hj) - max(yi, yj)
            if y_shared > min(hi, hj) * 0.3:
                # Check if i is to the left of j
                gap = xj - (xi + wi)
                if 0.0 < gap < SNAP_DIST:
                    half = gap * 0.5
                    b[i, 0] += half
                    b[j, 0] -= half
                # Check if j is to the left of i
                gap = xi - (xj + wj)
                if 0.0 < gap < SNAP_DIST:
                    half = gap * 0.5
                    b[j, 0] += half
                    b[i, 0] -= half

            # Shared X-span (for vertical snap)
            x_shared = min(xi + wi, xj + wj) - max(xi, xj)
            if x_shared > min(wi, wj) * 0.3:
                # Check if i is below j
                gap = yj - (yi + hi)
                if 0.0 < gap < SNAP_DIST:
                    half = gap * 0.5
                    b[i, 1] += half
                    b[j, 1] -= half
                # Check if j is below i
                gap = yi - (yj + hj)
                if 0.0 < gap < SNAP_DIST:
                    half = gap * 0.5
                    b[j, 1] += half
                    b[i, 1] -= half

    # Final boundary clamp after wall-snap
    for i in range(N):
        b[i, 0] = np.clip(b[i, 0], 0.0, max(0.0, net_w - b[i, 2]))
        b[i, 1] = np.clip(b[i, 1], 0.0, max(0.0, net_l - b[i, 3]))

    return b.astype(np.float32)


# ── Score helpers ──────────────────────────────────────────────────────────────

def _zone_score(x: float, y: float, w: float, h: float,
                net_w: float, net_l: float,
                preferred_zone: str, entrance_dir: str) -> float:
    """0.0–1.0: how well does this room's position satisfy its zone preference?"""
    cx = (x + w * 0.5) / max(net_w, 1.0)
    cy = (y + h * 0.5) / max(net_l, 1.0)

    # Map entrance direction to a (front_cx, front_cy) anchor
    _ENT = {
        "N": (0.5, 1.0), "S": (0.5, 0.0), "E": (1.0, 0.5), "W": (0.0, 0.5),
        "NE": (1.0, 1.0), "NW": (0.0, 1.0), "SE": (1.0, 0.0), "SW": (0.0, 0.0),
    }
    front_cx, front_cy = _ENT.get(entrance_dir.upper(), (0.5, 1.0))
    dist_to_front = math.hypot(cx - front_cx, cy - front_cy)
    dist_to_back  = math.hypot(cx - (1 - front_cx), cy - (1 - front_cy))

    if preferred_zone == "front":
        return max(0.0, 1.0 - dist_to_front)
    elif preferred_zone == "back":
        return max(0.0, 1.0 - dist_to_back)
    else:   # "middle"
        return max(0.0, 1.0 - abs(dist_to_front - dist_to_back))


def _adjacency_score(
    i: int,
    boxes: np.ndarray,
    room_ids: List[str],
    adj_graph: Dict[str, Dict[str, float]],
) -> float:
    """
    0.0-1.0: how well are high-weight adjacent rooms actually placed near each other?

    Optimised to iterate only over rooms in the adjacency graph (O(degree))
    instead of scanning all rooms (O(N)).
    """
    rid = room_ids[i]
    neighbors = adj_graph.get(rid, {})
    if not neighbors:
        return 1.0

    # Build reverse index: room_id -> array index (O(N) once, amortised)
    id_to_idx = {rid_j: j for j, rid_j in enumerate(room_ids)}

    total_w, satisfied_w = 0.0, 0.0
    xi, yi, wi, hi = boxes[i]
    cxi, cyi = xi + wi * 0.5, yi + hi * 0.5

    for neighbor_id, w_adj in neighbors.items():
        if abs(w_adj) < 0.3:
            continue
        j = id_to_idx.get(neighbor_id)
        if j is None or j == i:
            continue
        xj, yj, wj, hj = boxes[j]
        cxj, cyj = xj + wj * 0.5, yj + hj * 0.5
        dist = math.hypot(cxi - cxj, cyi - cyj)
        # "Adjacent" = within 1.5x the sum of half-dimensions
        touch_dist = (wi + wj) * 0.5 + (hi + hj) * 0.5
        proximity = max(0.0, 1.0 - dist / max(touch_dist * 1.5, 1.0))
        if w_adj > 0:
            satisfied_w += proximity * w_adj
        total_w += abs(w_adj)

    return satisfied_w / max(total_w, 1e-8)


# ── Main engine ────────────────────────────────────────────────────────────────

class AutoregressiveLayoutEngine:
    """
    End-to-end layout engine using the GNN + Autoregressive Transformer.

    Replaces DiffusionLayoutEngine as the primary generative solver
    (CP-SAT remains available as a fallback / gold standard).

    Loading behaviour
    -----------------
    The engine attempts to load trained weights from weights_dir.
    If ar_transformer.npz does not yet exist (model not yet trained),
    it falls back to a heuristic placement that still uses the
    area_fraction and generation_order fields from the enricher —
    which alone produces far better results than the old diffusion approach.
    """

    def __init__(self, weights_dir: Optional[str] = None) -> None:
        if weights_dir is None:
            weights_dir = os.path.join(
                os.path.dirname(__file__), "weights"
            )
        self._weights_dir = weights_dir
        self._gnn: Optional[object]  = None
        self._ar:  Optional[object]  = None
        self._load_models()

    def _load_models(self) -> None:
        """Lazily load GNN and AR transformer weights."""
        gnn_path = os.path.join(self._weights_dir, _GNN_FILE)
        ar_path  = os.path.join(self._weights_dir, _AR_FILE)

        # ── GNN encoder ───────────────────────────────────────────────────
        if os.path.exists(gnn_path):
            try:
                from modules.step4_generate.gnn_encoder import GNNEncoderNumpy
                self._gnn = GNNEncoderNumpy(gnn_path)
                log.info("AREngine: GNN encoder loaded from %s", gnn_path)
            except Exception as e:
                log.warning("AREngine: GNN load failed (%s) — using heuristic", e)

        # ── AR transformer ────────────────────────────────────────────────
        if os.path.exists(ar_path):
            try:
                from modules.step4_generate.autoregressive_transformer import (
                    LayoutTransformerNumpy,
                )
                self._ar = LayoutTransformerNumpy(ar_path)
                log.info("AREngine: AR transformer loaded from %s", ar_path)
            except Exception as e:
                log.warning("AREngine: AR transformer load failed (%s) — using heuristic", e)

    # ── Public API ─────────────────────────────────────────────────────────

    def place_floor(
        self,
        plan: EnrichedPlan,
        floor_number: int,
        temperature: float = 0.7,
        seed: int = 42,
    ) -> List[PlacedRoom]:
        """
        Generate placements for all rooms on a single floor.

        Returns a list of PlacedRoom objects with exact coordinates in feet.
        """
        t0 = time.perf_counter()

        rooms = plan.get_rooms_in_generation_order(floor_number)
        if not rooms:
            log.warning("AREngine: no rooms on floor %d", floor_number)
            return []

        net_w = plan.net_buildable_width_ft
        net_l = plan.net_buildable_length_ft

        log.info("AREngine: placing %d rooms on floor %d  (%.1f x %.1f ft)",
                 len(rooms), floor_number, net_w, net_l)

        if self._gnn is not None and self._ar is not None:
            placed = self._place_with_transformer(
                rooms, plan, net_w, net_l, floor_number, temperature, seed
            )
        else:
            log.info("AREngine: model weights not available — using area-budget heuristic")
            placed = self._place_heuristic(rooms, plan, net_w, net_l, floor_number)

        elapsed = (time.perf_counter() - t0) * 1000
        log.info("AREngine: floor %d placed in %.1f ms  (%d rooms)",
                 floor_number, elapsed, len(placed))
        return placed

    # ── Transformer-based placement ────────────────────────────────────────

    def _place_with_transformer(
        self,
        rooms:        List[EnrichedRoom],
        plan:         EnrichedPlan,
        net_w:        float,
        net_l:        float,
        floor_number: int,
        temperature:  float,
        seed:         int,
    ) -> List[PlacedRoom]:
        """Full GNN + AR Transformer placement pipeline."""
        from modules.step4_generate.autoregressive_transformer import (
            LayoutTokenizer,
            ROOM_VOCAB,
            _ROOM_NORMALISE,
        )
        from modules.step4_generate.training.data_prep import (
            build_sample_tensors,
            FloorPlanSample,
            RoomRecord,
            AdjacencyWeightTable,
            ROOM_VOCAB as TRAINING_VOCAB,
        )

        # ── Build node features for the GNN ──────────────────────────────
        # We use approximate normalised boxes from area_fraction for feature
        # computation (GNN only needs the relational features, not exact positions)
        sample_rooms: List[RoomRecord] = []
        total_area = sum(r.target_area_sqft for r in rooms)

        # Pack rooms into approximate non-overlapping boxes for GNN features
        # Simple strip packing (left-to-right, top-to-bottom)
        cursor_x, cursor_y, row_h = 0.0, 0.0, 0.0
        for r in rooms:
            frac = r.area_fraction if r.area_fraction > 0 else r.target_area_sqft / max(total_area, 1.0)
            area_ft = frac * net_w * net_l
            # Compute approximate width from target aspect ratio
            aspect = r.target_width_ft / max(r.target_length_ft, 0.1)
            w_ft   = min(math.sqrt(area_ft * aspect), net_w * 0.8)
            h_ft   = min(area_ft / max(w_ft, 0.1), net_l * 0.8)
            w_ft   = max(w_ft, r.min_width_ft)
            h_ft   = max(h_ft, r.min_length_ft)

            if cursor_x + w_ft > net_w:
                cursor_x = 0.0
                cursor_y += row_h
                row_h = 0.0

            norm_x1 = cursor_x / net_w
            norm_y1 = min(cursor_y / net_l, 0.99)
            norm_x2 = min((cursor_x + w_ft) / net_w, 1.0)
            norm_y2 = min((cursor_y + h_ft) / net_l, 1.0)

            rt = _ROOM_NORMALISE.get(r.room_type, r.room_type)
            # rt is a canonical string (e.g. "living_room") — RoomRecord stores strings
            sample_rooms.append(RoomRecord(
                room_type=rt,
                x1=norm_x1, y1=norm_y1,
                x2=max(norm_x2, norm_x1 + 0.01),
                y2=max(norm_y2, norm_y1 + 0.01),
                floor_level=float(floor_number > 0),
            ))
            cursor_x += w_ft
            row_h = max(row_h, h_ft)

        sample = FloorPlanSample(
            plan_id=f"inference_{floor_number}",
            source="inference",
            rooms=sample_rooms,
        )
        sample = build_sample_tensors(sample, AdjacencyWeightTable())

        # ── Run GNN encoder ───────────────────────────────────────────────
        node_feats   = sample.node_features                # (N, 24)
        edge_index   = sample.edge_index                   # (2, E)
        edge_feats   = sample.edge_features                # (E, 7)

        # GNNEncoderNumpy.forward() returns a (node_emb, global_emb) tuple
        node_emb, global_emb = self._gnn.forward(node_feats, edge_index, edge_feats)

        # ── Build tokenizer ───────────────────────────────────────────────
        tokenizer = LayoutTokenizer(
            net_w_ft=net_w,
            net_l_ft=net_l,
            entrance_dir=plan.entrance_direction,
            vastu_on=plan.vastu_enabled,
        )

        # ── Force room types from enricher ────────────────────────────────
        type_ids_forced = [
            LayoutTokenizer.room_type_to_id(r.room_type)
            for r in rooms
        ]

        # ── Run AR transformer ────────────────────────────────────────────
        boxes_norm, pred_types = self._ar.generate(
            gnn_node_emb    = node_emb,
            gnn_global_emb  = global_emb,
            tokenizer       = tokenizer,
            n_rooms         = len(rooms),
            type_ids_forced = type_ids_forced,
            temperature     = temperature,
            seed            = seed,
        )                                       # boxes_norm: (N, 4)  [cx, cy, w, h]

        # ── Decode to feet ────────────────────────────────────────────────
        boxes_ft = tokenizer.decode_boxes(boxes_norm)  # (N, 4) [x, y, w, h] in ft

        # ── NBC minimum enforcement ───────────────────────────────────────
        for i, r in enumerate(rooms):
            if boxes_ft[i, 2] < r.min_width_ft:
                boxes_ft[i, 2] = r.min_width_ft
            if boxes_ft[i, 3] < r.min_length_ft:
                boxes_ft[i, 3] = r.min_length_ft
            # Re-clamp after resizing
            boxes_ft[i, 0] = np.clip(boxes_ft[i, 0], 0.0, net_w - boxes_ft[i, 2])
            boxes_ft[i, 1] = np.clip(boxes_ft[i, 1], 0.0, net_l - boxes_ft[i, 3])

        # ── Max-area + aspect-ratio enforcement ───────────────────────────
        boxes_ft = _clamp_rooms(boxes_ft, rooms, net_w, net_l)

        # ── Resolve overlaps ──────────────────────────────────────────────
        boxes_ft = _resolve_overlaps(boxes_ft, net_w, net_l, n_iters=400)

        # ── Final clamp (overlap resolution can re-expand dimensions) ─────
        boxes_ft = _clamp_rooms(boxes_ft, rooms, net_w, net_l)

        return self._to_placed_rooms(
            rooms, boxes_ft, plan.adjacency_graph,
            net_w, net_l, plan.entrance_direction
        )

    # ── Heuristic placement (no trained model) ────────────────────────────

    def _place_heuristic(
        self,
        rooms:        List[EnrichedRoom],
        plan:         EnrichedPlan,
        net_w:        float,
        net_l:        float,
        floor_number: int,
    ) -> List[PlacedRoom]:
        """
        Area-budget strip packer.

        Used when the AR transformer weights don't exist yet.
        Far better than the old diffusion approach because:
          - Respects area_fraction (rooms sized jointly, not from a static table)
          - Respects generation_order (anchor rooms placed first)
          - Uses zone preferences to pick the strip orientation
        """
        boxes_ft = np.zeros((len(rooms), 4), dtype=np.float32)
        total_area = net_w * net_l

        # Divide floor into strips based on zone preferences
        # Rooms in "front" zone: top strip (y close to net_l)
        # Rooms in "back" zone:  bottom strip (y close to 0)
        # Rooms in "middle":     middle strip
        front_rooms  = [r for r in rooms if r.preferred_zone == "front"]
        back_rooms   = [r for r in rooms if r.preferred_zone == "back"]
        middle_rooms = [r for r in rooms if r.preferred_zone == "middle"]

        def _pack_strip(strip_rooms, x0, y0, strip_w, strip_h):
            """Simple left-to-right packing within a horizontal strip."""
            cx = x0
            row_y = y0
            row_h = 0.0
            row_rooms = []
            all_placed = []

            for r in strip_rooms:
                frac = r.area_fraction if r.area_fraction > 0 else 0.1
                area = frac * total_area
                aspect = r.target_width_ft / max(r.target_length_ft, 0.1)
                aspect = max(0.5, min(aspect, 3.0))
                w = min(math.sqrt(area * aspect), strip_w * 0.9)
                h = min(area / max(w, 0.1), strip_h * 0.9)
                w = max(w, r.min_width_ft)
                h = max(h, r.min_length_ft)

                if cx + w > x0 + strip_w + 0.1:
                    row_y += row_h
                    cx = x0
                    row_h = 0.0

                all_placed.append((r, cx, row_y, w, h))
                cx += w
                row_h = max(row_h, h)

            return all_placed

        strip_h = net_l / 3.0

        placed_data = []
        placed_data += _pack_strip(back_rooms,   0, 0,           net_w, strip_h)
        placed_data += _pack_strip(middle_rooms, 0, strip_h,     net_w, strip_h)
        placed_data += _pack_strip(front_rooms,  0, strip_h * 2, net_w, strip_h)

        # Map back to boxes_ft using room index (rooms list is generation-order sorted)
        room_id_map = {r.room_id: i for i, r in enumerate(rooms)}
        for r, x, y, w, h in placed_data:
            idx = room_id_map[r.room_id]
            boxes_ft[idx] = [
                np.clip(x, 0.0, net_w - w),
                np.clip(y, 0.0, net_l - h),
                w, h,
            ]

        # Max-area + aspect-ratio enforcement
        boxes_ft = _clamp_rooms(boxes_ft, rooms, net_w, net_l)

        # Resolve overlaps
        boxes_ft = _resolve_overlaps(boxes_ft, net_w, net_l, n_iters=500)

        # Final clamp (overlap resolution can re-expand dimensions)
        boxes_ft = _clamp_rooms(boxes_ft, rooms, net_w, net_l)

        return self._to_placed_rooms(
            rooms, boxes_ft, plan.adjacency_graph,
            net_w, net_l, plan.entrance_direction
        )

    # ── Conversion helper ──────────────────────────────────────────────────

    def _to_placed_rooms(
        self,
        rooms:         List[EnrichedRoom],
        boxes_ft:      np.ndarray,           # (N, 4): [x, y, w, h] in feet
        adj_graph:     Dict[str, Dict[str, float]],
        net_w:         float,
        net_l:         float,
        entrance_dir:  str,
    ) -> List[PlacedRoom]:
        """Convert raw box array to list of PlacedRoom with quality scores."""
        room_ids = [r.room_id for r in rooms]
        placed:  List[PlacedRoom] = []

        for i, r in enumerate(rooms):
            x, y, w, h = float(boxes_ft[i, 0]), float(boxes_ft[i, 1]), \
                         float(boxes_ft[i, 2]), float(boxes_ft[i, 3])
            w = max(w, r.min_width_ft)
            h = max(h, r.min_length_ft)

            # ── Min-buildable dimension warning ──────────────────────────
            # If the room ended up smaller than 50% of its NBC minimum area
            # after overlap resolution, flag it — the layout is suspect.
            actual_area = w * h
            if r.min_area_sqft > 0 and actual_area < r.min_area_sqft * 0.50:
                log.warning(
                    "Room %s (%s) is severely undersized: %.1f sqft vs "
                    "NBC min %.1f sqft (%.0f%%)",
                    r.room_id, r.display_name, actual_area,
                    r.min_area_sqft, actual_area / r.min_area_sqft * 100,
                )

            z_score = _zone_score(x, y, w, h, net_w, net_l,
                                  r.preferred_zone, entrance_dir)
            a_score = _adjacency_score(i, boxes_ft, room_ids, adj_graph)

            placed.append(PlacedRoom(
                room_id     = r.room_id,
                room_type   = r.room_type,
                display_name= r.display_name,
                floor       = r.preferred_floor,
                implicit_room = r.implicit_room,
                x_ft        = round(x, 3),
                y_ft        = round(y, 3),
                width_ft    = round(w, 3),
                length_ft   = round(h, 3),
                area_sqft   = round(w * h, 2),
                preferred_direction = r.preferred_direction,
                preferred_zone      = r.preferred_zone,
                zone_score      = round(z_score, 3),
                adjacency_score = round(a_score, 3),
            ))

        return placed
