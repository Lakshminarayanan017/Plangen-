"""
features.py — sample / EngineRequest -> model arrays (pure NumPy).

One builder feeds torch training and the NumPy production path, so features
cannot drift between train and serve. Everything the model consumes:

  identity   type_ids, zone_ids, floor_ids            (N,)
  vastu      vastu_dir_ids, vastu_strength_ids        (N,)      [R4]
  graph      edge_index                               (2, 2E)
  static     boundary (2,64,64), global (9,)          [R4 adds regime]
  dynamic    state (N, 5, 32, 32)                     [R1]
  targets    target_cell, target_size                 (N,)
             target_aspect, target_orientation,
             target_band                              (N,)      [R3]

Targets that a corpus genuinely cannot supply are written as IGNORE_INDEX and
masked out of the loss. CubiCasa has no carved geometry to derive an exact
aspect from, so those three heads learn during distillation and RL, where the
engine supplies the real thing. That is deliberate: a Finnish room's aspect is
not a target worth imitating anyway.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from modules.step4_generate.engine.contracts import SEED_GRID, EngineRequest
from ml.placer_v3.config import (
    BOUNDARY_GRID, GLOBAL_DIM, IGNORE_INDEX, REGIME_TO_ID, regime_for,
    room_type_id, vastu_dir_id, vastu_strength_id, zone_id,
)
from ml.placer_v3.state import build_state_stack
from ml.training.vocab import generation_sort_key

_SIDE_TO_BAND = {
    "N": (slice(0, 6), slice(None)),
    "S": (slice(BOUNDARY_GRID - 6, BOUNDARY_GRID), slice(None)),
    "W": (slice(None), slice(0, 6)),
    "E": (slice(None), slice(BOUNDARY_GRID - 6, BOUNDARY_GRID)),
}


def build_boundary(mask64: np.ndarray, entrance_side: str) -> np.ndarray:
    """(2, 64, 64) float32: [interior footprint, entrance-edge band]."""
    interior = mask64.astype(np.float32)
    edge = np.zeros((BOUNDARY_GRID, BOUNDARY_GRID), dtype=np.float32)
    band = _SIDE_TO_BAND.get(entrance_side)
    if band is not None:
        edge[band] = 1.0
    edge *= interior
    return np.stack([interior, edge], axis=0)


def build_global(plot_w_ft: float, plot_h_ft: float, n_rooms: int,
                 program_sqft: float = 0.0) -> np.ndarray:
    """(9,) plot context. The boundary mask is stretched to a square frame,
    so plot ASPECT lives here rather than in the raster — and so do the two
    facts v2 never told the model: how big the plot is in absolute terms, and
    how hard the program is pushing against it."""
    w = max(1.0, float(plot_w_ft))
    h = max(1.0, float(plot_h_ft))
    area = w * h
    regime = np.zeros(3, dtype=np.float32)
    regime[REGIME_TO_ID[regime_for(area)]] = 1.0
    return np.array([
        math.log(w) - 3.0,
        math.log(h) - 3.0,
        math.log(w / h),
        n_rooms / 12.0,
        regime[0], regime[1], regime[2],
        math.log(area) - 6.5,
        min(2.0, float(program_sqft) / area) if program_sqft else 0.0,
    ], dtype=np.float32)


def _edges_both_ways(edges, n: int) -> np.ndarray:
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    src, dst = [], []
    for a, b in edges:
        if 0 <= a < n and 0 <= b < n and a != b:
            src += [a, b]
            dst += [b, a]
    return np.array([src, dst], dtype=np.int64)


def _ignore(n: int) -> np.ndarray:
    return np.full(n, IGNORE_INDEX, dtype=np.int64)


def sample_to_arrays(sample: Dict, mask64: np.ndarray) -> Dict:
    """A prepared samples.jsonl record + its mask -> arrays with targets.

    Used for stage (a), imitation pretraining on CubiCasa. Aspect,
    orientation and band come back as IGNORE_INDEX unless a re-run of the
    prep added them (`prep_cubicasa --with-geometry`).
    """
    rooms = sample["rooms"]
    n = len(rooms)
    zone_ids = np.array([zone_id(r["zone"]) for r in rooms], np.int64)
    cells = [r["row"] * SEED_GRID + r["col"] for r in rooms]
    sizes = [r["size_class"] for r in rooms]
    boundary = build_boundary(mask64, sample["entrance_side"])

    def col(key, fn):
        if not all(key in r for r in rooms):
            return _ignore(n)
        return np.array([fn(r[key]) for r in rooms], np.int64)

    return {
        "type_ids": np.array([room_type_id(r["rtype"]) for r in rooms],
                             np.int64),
        "zone_ids": zone_ids,
        "floor_ids": np.zeros(n, np.int64),
        # CubiCasa carries no Vastu annotation — index 0 is "no opinion",
        # which is the truth here, not a missing value.
        "vastu_dir_ids": np.zeros(n, np.int64),
        "vastu_strength_ids": np.zeros(n, np.int64),
        "edge_index": _edges_both_ways(sample.get("edges", []), n),
        "boundary": boundary,
        "global": build_global(sample["plot_w_ft"], sample["plot_h_ft"], n,
                               sum(r.get("area_sqft", 0.0) for r in rooms)),
        "state": build_state_stack(boundary[0], sample["entrance_side"],
                                   cells, sizes, zone_ids),
        "target_cell": np.array(cells, np.int64),
        "target_size": np.array([s - 1 for s in sizes], np.int64),
        "target_aspect": col("aspect_class", int),
        "target_orientation": col("orientation_class", int),
        "target_band": col("band_index", int),
        "n_rooms": n,
    }


def selfplay_to_arrays(record: Dict) -> Dict:
    """A self-play corpus record -> arrays with the FULL target set.

    Stage (b). Every target is real here because the record came from an
    actual carved plan the reviewer scored, so the aspect / orientation /
    band heads get supervised on Indian-scored geometry.
    """
    rooms = record["rooms"]
    n = len(rooms)
    zone_ids = np.array([zone_id(r["zone"]) for r in rooms], np.int64)
    cells = [r["row"] * SEED_GRID + r["col"] for r in rooms]
    sizes = [r["size_class"] for r in rooms]
    mask64 = np.ones((BOUNDARY_GRID, BOUNDARY_GRID), dtype=np.float32)
    if record.get("mask") is not None:
        mask64 = np.asarray(record["mask"], dtype=np.float32)
    boundary = build_boundary(mask64, record["entrance_side"])

    return {
        "type_ids": np.array([room_type_id(r["rtype"]) for r in rooms],
                             np.int64),
        "zone_ids": zone_ids,
        "floor_ids": np.array([r.get("floor", 0) for r in rooms], np.int64),
        "vastu_dir_ids": np.array(
            [vastu_dir_id(r.get("vastu_dir")) for r in rooms], np.int64),
        "vastu_strength_ids": np.array(
            [vastu_strength_id(r.get("vastu_strength")) for r in rooms],
            np.int64),
        "edge_index": _edges_both_ways(record.get("edges", []), n),
        "boundary": boundary,
        "global": build_global(record["plot_w_ft"], record["plot_h_ft"], n,
                               record.get("program_sqft", 0.0)),
        "state": build_state_stack(boundary[0], record["entrance_side"],
                                   cells, sizes, zone_ids),
        "target_cell": np.array(cells, np.int64),
        "target_size": np.array([s - 1 for s in sizes], np.int64),
        "target_aspect": np.array([r["aspect_class"] for r in rooms],
                                  np.int64),
        "target_orientation": np.array(
            [r["orientation_class"] for r in rooms], np.int64),
        "target_band": np.array([r["band_index"] for r in rooms], np.int64),
        "reward": float(record.get("reward", 0.0)),
        "n_rooms": n,
    }


def request_to_arrays(request: EngineRequest,
                      mask64: Optional[np.ndarray] = None,
                      vastu: Optional[Dict[str, Dict[str, str]]] = None
                      ) -> Dict:
    """EngineRequest -> input arrays (no targets), in generation order.

    `vastu` maps room NAME -> {"direction": "SW", "strength": "soft"}. It is
    optional so this keeps working before phase 01 lands; absent, every room
    reads as "no Vastu opinion", which is exactly what a non-Vastu request
    means. The state stack starts empty and the sampler advances it.
    """
    specs = list(request.rooms)
    order = sorted(range(len(specs)),
                   key=lambda i: generation_sort_key(specs[i].rtype,
                                                     specs[i].target_sqft))
    specs = [specs[i] for i in order]
    name_to_idx = {s.name: i for i, s in enumerate(specs)}
    n = len(specs)
    vastu = vastu or {}

    edges = []
    for wish in request.wishes:
        a, b = name_to_idx.get(wish.room_a), name_to_idx.get(wish.room_b)
        if a is not None and b is not None:
            edges.append((a, b))

    if mask64 is None:
        mask64 = np.ones((BOUNDARY_GRID, BOUNDARY_GRID), dtype=np.float32)
    boundary = build_boundary(mask64, request.entrance_side)

    return {
        "specs": specs,
        "type_ids": np.array([room_type_id(s.rtype) for s in specs],
                             np.int64),
        "zone_ids": np.array([zone_id(s.zone) for s in specs], np.int64),
        "floor_ids": np.array([getattr(s, "floor", 0) for s in specs],
                              np.int64),
        "vastu_dir_ids": np.array(
            [vastu_dir_id((vastu.get(s.name) or {}).get("direction"))
             for s in specs], np.int64),
        "vastu_strength_ids": np.array(
            [vastu_strength_id((vastu.get(s.name) or {}).get("strength"))
             for s in specs], np.int64),
        "edge_index": _edges_both_ways(edges, n),
        "boundary": boundary,
        "global": build_global(request.plot_w_ft, request.plot_h_ft, n,
                               sum(s.target_sqft for s in specs)),
        "entrance_side": request.entrance_side,
        "n_rooms": n,
    }
