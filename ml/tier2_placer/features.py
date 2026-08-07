"""
features.py — PreparedSample / EngineRequest -> model arrays (numpy).

Pure numpy so the SAME builders feed both torch training and the NumPy
inference path (no drift between train and production feature construction).

Per sample the model consumes:
  node ids   : type_ids (N,), zone_ids (N,), floor_ids (N,)  -> embedded
  edges      : edge_index (2, 2E) undirected (both directions)
  boundary   : (2, 64, 64) float  [interior footprint, entrance-edge band]
  global     : (G,) float  [log plot w, log plot h, aspect, n_rooms/scale]
Training also provides targets:
  target_cell (N,) long in [0,1024)  = row*32 + col
  target_size (N,) long in [0,40)    = size_class - 1

The room order is canonical generation order (already applied by the data
prep; re-applied for inference requests via engine rtypes).
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from modules.step4_generate.engine.contracts import SEED_GRID, EngineRequest
from ml.tier2_placer.config import (
    BOUNDARY_GRID, GLOBAL_DIM, room_type_id, zone_id,
)
from ml.training.vocab import generation_sort_key

_SIDE_TO_CHANNEL_SLICE = {
    # (rows, cols) of the entrance-edge band on the 64x64 grid
    "N": (slice(0, 6), slice(None)),
    "S": (slice(BOUNDARY_GRID - 6, BOUNDARY_GRID), slice(None)),
    "W": (slice(None), slice(0, 6)),
    "E": (slice(None), slice(BOUNDARY_GRID - 6, BOUNDARY_GRID)),
}


def build_boundary(mask64: np.ndarray, entrance_side: str) -> np.ndarray:
    """(2, 64, 64) float32: [interior footprint, entrance-edge band]."""
    interior = mask64.astype(np.float32)
    edge = np.zeros((BOUNDARY_GRID, BOUNDARY_GRID), dtype=np.float32)
    sl = _SIDE_TO_CHANNEL_SLICE.get(entrance_side)
    if sl is not None:
        edge[sl] = 1.0
    edge *= interior            # entrance band only where the footprint is
    return np.stack([interior, edge], axis=0)


def build_global(plot_w_ft: float, plot_h_ft: float, n_rooms: int
                 ) -> np.ndarray:
    """(GLOBAL_DIM,) float32 plot-context features. The boundary mask is
    stretched to a square frame, so plot ASPECT lives here, not in the mask."""
    w = max(1.0, float(plot_w_ft))
    h = max(1.0, float(plot_h_ft))
    return np.array([
        math.log(w) - 3.0,             # ~centered for typical 10-60ft
        math.log(h) - 3.0,
        math.log(w / h),               # aspect (signed)
        n_rooms / 12.0,                # rooms, scaled
    ], dtype=np.float32)


def _edges_both_ways(edges: List, n: int) -> np.ndarray:
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    src, dst = [], []
    for a, b in edges:
        if 0 <= a < n and 0 <= b < n:
            src += [a, b]; dst += [b, a]
    return np.array([src, dst], dtype=np.int64)


def sample_to_arrays(sample: Dict, mask64: np.ndarray) -> Dict:
    """Prepared samples.jsonl record + its mask -> model arrays (with targets)."""
    rooms = sample["rooms"]
    n = len(rooms)
    type_ids = np.array([room_type_id(r["rtype"]) for r in rooms], np.int64)
    zone_ids = np.array([zone_id(r["zone"]) for r in rooms], np.int64)
    floor_ids = np.zeros(n, np.int64)
    target_cell = np.array([r["row"] * SEED_GRID + r["col"] for r in rooms],
                           np.int64)
    target_size = np.array([r["size_class"] - 1 for r in rooms], np.int64)
    return {
        "type_ids": type_ids, "zone_ids": zone_ids, "floor_ids": floor_ids,
        "edge_index": _edges_both_ways(sample.get("edges", []), n),
        "boundary": build_boundary(mask64, sample["entrance_side"]),
        "global": build_global(sample["plot_w_ft"], sample["plot_h_ft"], n),
        "target_cell": target_cell, "target_size": target_size,
        "n_rooms": n,
    }


def request_to_arrays(request: EngineRequest,
                      mask64: np.ndarray = None) -> Dict:
    """EngineRequest -> model input arrays (no targets), rooms in canonical
    generation order. Edges come from the request's opening wishes; the
    boundary defaults to a full rectangle (the plot IS the footprint) unless
    a mask is supplied (irregular plots, later)."""
    specs = list(request.rooms)
    order = sorted(range(len(specs)),
                   key=lambda i: generation_sort_key(specs[i].rtype,
                                                     specs[i].target_sqft))
    specs = [specs[i] for i in order]
    name_to_idx = {s.name: i for i, s in enumerate(specs)}
    n = len(specs)

    type_ids = np.array([room_type_id(s.rtype) for s in specs], np.int64)
    zone_ids = np.array([zone_id(s.zone) for s in specs], np.int64)
    floor_ids = np.array([getattr(s, "floor", 0) for s in specs], np.int64)

    edges = []
    for w in request.wishes:
        a, b = name_to_idx.get(w.room_a), name_to_idx.get(w.room_b)
        if a is not None and b is not None:
            edges.append((a, b))

    if mask64 is None:
        mask64 = np.ones((BOUNDARY_GRID, BOUNDARY_GRID), dtype=np.float32)

    return {
        "specs": specs,                # generation-ordered RoomSpecs
        "type_ids": type_ids, "zone_ids": zone_ids, "floor_ids": floor_ids,
        "edge_index": _edges_both_ways(edges, n),
        "boundary": build_boundary(mask64, request.entrance_side),
        "global": build_global(request.plot_w_ft, request.plot_h_ft, n),
        "n_rooms": n,
    }
