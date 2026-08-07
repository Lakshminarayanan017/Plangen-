"""
features.py — what the critic is allowed to look at.

Two blocks, both computed from the finished plan:

  1. the FULL reviewer breakdown — every named rule's evidence, not the
     scalar it collapses into. The soft score is one hand-chosen linear
     combination of these; the critic's job is to learn a better one.
  2. geometry statistics the rules do not currently weigh — the shape of
     the area-error distribution, aspect spread, daylight coverage,
     adjacency satisfaction, opening mix.

The order of FEATURE_NAMES is the contract between training and inference
and must never be reordered — only appended to (and a model trained before
the append refuses to load against the longer vector, by construction:
`extract` returns len(FEATURE_NAMES) values and GBTModel records the names
it was trained on).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from modules.step4_generate.core import units
from modules.step4_generate.core.grid_plan import GridPlan
from modules.step4_generate.engine.contracts import EngineConfig, EngineRequest, Verdict
from modules.step4_generate.engine.rules.base import CIRCULATION, HABITABLE, PRIVATE, SOCIAL, WET
from modules.step4_generate.engine.validator import BasicValidator

# Breakdown keys, in a frozen order. Missing keys read as 0.0 — a plan with
# no staircase genuinely has no stair evidence, and that is information.
BREAKDOWN_KEYS: List[str] = [
    "area_drift", "worst_aspect", "narrow_habitable", "nbc_violations",
    "privacy_doors", "wet_social_doors", "wide_openings",
    "circ_excess_hops", "daylightless", "windowless", "cross_vent_rooms",
    "freespace_violations", "circulation_fraction", "narrow_passages",
    "min_passage_ft", "hub_mean_dist", "wall_efficiency", "wall_jogs",
    "toothpick_walls", "min_shared_run_ft", "hinge_stub_violations",
    "wrong_swing_doors", "stair_entrance_hops", "stair_single_landing",
    "stair_frontage_ft",
]

GEOMETRY_KEYS: List[str] = [
    "n_rooms", "log_plot_sqft", "plot_aspect", "program_density",
    "aspect_mean", "aspect_max", "aspect_std",
    "area_err_mean", "area_err_max", "area_err_std",
    "smallest_room_sqft", "largest_room_frac",
    "doors_per_room", "wide_per_room", "windows_per_room",
    "daylight_coverage", "adjacency_satisfaction", "wish_satisfaction",
    "private_off_circulation", "wet_attached_frac",
    "mean_entrance_hops", "max_entrance_hops",
]

FEATURE_NAMES: List[str] = BREAKDOWN_KEYS + GEOMETRY_KEYS
N_FEATURES = len(FEATURE_NAMES)


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _aspect(plan: GridPlan, rid: int) -> float:
    x0, y0, x1, y1 = plan.face_bbox(rid)
    w, h = x1 - x0, y1 - y0
    return max(w, h) / min(w, h) if min(w, h) > 0 else 0.0


def geometry_features(plan: GridPlan, request: EngineRequest,
                      room_ids: Dict[str, int]) -> Dict[str, float]:
    """The block the reviewer does not already summarize."""
    specs = {s.name: s for s in request.rooms}
    ids = [room_ids[s.name] for s in request.rooms if s.name in room_ids]
    if not ids:
        return {k: 0.0 for k in GEOMETRY_KEYS}

    plot_sqft = float(request.plot_w_ft * request.plot_h_ft)
    aspects = np.array([_aspect(plan, rid) for rid in ids])
    areas = np.array([plan.area_sqft(rid) for rid in ids])
    targets = np.array([max(1.0, specs[n].target_sqft)
                        for n in room_ids if n in specs])
    # targets are absolute; compare against the plot-scaled program so a
    # small plot with a big wish list is not scored as pure failure
    scale = _safe_div(float(areas.sum()), float(targets.sum())) or 1.0
    errs = np.abs(areas - targets * scale) / np.maximum(targets * scale, 1.0)

    # opening mix
    doors = sum(1 for op in plan.openings
                if op.kind == "door" and not op.is_exterior)
    wides = sum(1 for op in plan.openings
                if op.kind == "wide" and not op.is_exterior)
    windows = sum(1 for op in plan.openings if op.kind == "window")

    # daylight: habitable rooms with any window (exterior or onto a shaft)
    windowed = {op.room_a for op in plan.openings if op.kind == "window"}
    windowed |= {op.room_b for op in plan.openings if op.kind == "window"}
    habitable = [room_ids[s.name] for s in request.rooms
                 if s.rtype in HABITABLE and s.name in room_ids]

    # adjacency: fraction of room pairs sharing a wall that also share an
    # opening — a plan full of shared walls with no openings is a maze
    adjacency = plan.adjacency()
    opened = {tuple(sorted((op.room_a, op.room_b))) for op in plan.openings
              if not op.is_exterior}
    interior_pairs = [p for p in adjacency
                      if p[0] in room_ids.values() and p[1] in room_ids.values()]

    wishes_ok = 0
    for wish in request.wishes:
        a, b = room_ids.get(wish.room_a), room_ids.get(wish.room_b)
        if a is not None and b is not None and tuple(sorted((a, b))) in opened:
            wishes_ok += 1

    # privacy topology: private rooms entered from circulation/social space
    graph: Dict[int, List[int]] = {}
    for op in plan.openings:
        if op.is_exterior:
            continue
        graph.setdefault(op.room_a, []).append(op.room_b)
        graph.setdefault(op.room_b, []).append(op.room_a)
    rtype_of = {rid: plan.rooms[rid].rtype for rid in ids}
    privates = [r for r in ids if rtype_of[r] in PRIVATE]
    off_circ = sum(
        1 for r in privates
        if any(rtype_of.get(n) in (CIRCULATION | SOCIAL)
               for n in graph.get(r, ())))
    wets = [r for r in ids if rtype_of[r] in WET]
    attached = sum(1 for r in wets
                   if any(rtype_of.get(n) in PRIVATE for n in graph.get(r, ())))

    # depth from the entrance
    entry = next((op.room_a for op in plan.openings
                  if op.is_exterior and op.kind == "door"), None)
    hops: Dict[int, int] = {}
    if entry is not None:
        hops[entry] = 0
        frontier = [entry]
        while frontier:
            cur = frontier.pop(0)
            for nxt in graph.get(cur, ()):
                if nxt not in hops:
                    hops[nxt] = hops[cur] + 1
                    frontier.append(nxt)
    depths = [hops.get(r, 4) for r in ids]

    n = float(len(ids))
    return {
        "n_rooms": n,
        "log_plot_sqft": math.log(max(plot_sqft, 1.0)),
        "plot_aspect": _safe_div(max(request.plot_w_ft, request.plot_h_ft),
                                 min(request.plot_w_ft, request.plot_h_ft)),
        "program_density": _safe_div(float(targets.sum()), plot_sqft),
        "aspect_mean": float(aspects.mean()),
        "aspect_max": float(aspects.max()),
        "aspect_std": float(aspects.std()),
        "area_err_mean": float(errs.mean()),
        "area_err_max": float(errs.max()),
        "area_err_std": float(errs.std()),
        "smallest_room_sqft": float(areas.min()),
        "largest_room_frac": _safe_div(float(areas.max()), float(areas.sum())),
        "doors_per_room": _safe_div(doors, n),
        "wide_per_room": _safe_div(wides, n),
        "windows_per_room": _safe_div(windows, n),
        "daylight_coverage": _safe_div(
            sum(1 for r in habitable if r in windowed), len(habitable)),
        "adjacency_satisfaction": _safe_div(
            sum(1 for p in interior_pairs if p in opened),
            len(interior_pairs)),
        "wish_satisfaction": _safe_div(wishes_ok, len(request.wishes)),
        "private_off_circulation": _safe_div(off_circ, len(privates)),
        "wet_attached_frac": _safe_div(attached, len(wets)),
        "mean_entrance_hops": _safe_div(float(sum(depths)), n),
        "max_entrance_hops": float(max(depths)) if depths else 0.0,
    }


def extract(plan: GridPlan, request: EngineRequest,
            room_ids: Dict[str, int], verdict: Optional[Verdict] = None,
            config: Optional[EngineConfig] = None) -> np.ndarray:
    """The feature vector, in FEATURE_NAMES order. `verdict` is recomputed
    when not supplied — the caller normally already has one."""
    if verdict is None:
        verdict = BasicValidator(config or EngineConfig()).check(
            plan, request, room_ids)
    breakdown = verdict.breakdown or {}
    geo = geometry_features(plan, request, room_ids)

    values: List[float] = []
    for key in BREAKDOWN_KEYS:
        raw = breakdown.get(key, 0.0)
        values.append(float(raw) if isinstance(raw, (int, float)) else 0.0)
    for key in GEOMETRY_KEYS:
        values.append(float(geo.get(key, 0.0)))
    return np.asarray(values, dtype=np.float64)


def as_dict(vector: Sequence[float]) -> Dict[str, float]:
    """Named view of a feature vector — for debugging and reports."""
    return {name: float(v) for name, v in zip(FEATURE_NAMES, vector)}


assert len(set(FEATURE_NAMES)) == N_FEATURES, "duplicate feature name"
assert units.CELLS_PER_FOOT == 8, "feature scales assume the 1.5in lattice"
