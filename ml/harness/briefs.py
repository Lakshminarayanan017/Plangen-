"""
briefs.py — the golden brief set.

Deterministic, programmatic briefs spanning the plot sizes, aspects,
entrance sides, and program sizes PlanGen must handle. Every engine change
is judged against this set (run_harness.py), never against a lucky demo.

Programs follow Indian residential conventions: drawing/living at entry,
dining hub + doorless kitchen, passage serving the bedroom cluster,
attached-style baths, parking on street-facing plots.
"""

from __future__ import annotations

from typing import List

from modules.step4_generate.engine.contracts import EngineRequest, RoomSpec

# (width_ft, height_ft, entrance_side, bhk)
_PLOTS = [
    (20, 30, "S", 1),
    (20, 45, "S", 3),
    (20, 45, "N", 2),
    (25, 40, "S", 2),
    (25, 50, "E", 3),
    (30, 40, "S", 2),
    (30, 40, "W", 2),
    (30, 50, "S", 3),
    (30, 60, "N", 3),
    (35, 50, "E", 3),
    (40, 30, "S", 2),
    (40, 60, "S", 4),
    (45, 45, "W", 3),
    (50, 40, "N", 3),
    (50, 70, "S", 4),
    (60, 40, "E", 3),
    (60, 90, "S", 4),
    (35, 35, "S", 2),
]

# Program fractions of net room area (roughly Indian-residential ratios).
_FRACTIONS = {
    "living": 0.20, "dining": 0.11, "kitchen": 0.09, "passage": 0.05,
    "master": 0.16, "bedroom": 0.13, "bath": 0.045, "parking": 0.12,
}


def _program(bhk: int, plot_sqft: float) -> List[RoomSpec]:
    net = plot_sqft * 0.80          # walls + inefficiency allowance
    f = _FRACTIONS
    rooms: List[RoomSpec] = [
        RoomSpec("Living Room", "living_room", net * f["living"],
                 zone="public"),
        RoomSpec("Kitchen", "kitchen", net * f["kitchen"], zone="service"),
    ]
    if plot_sqft >= 800:
        rooms.append(RoomSpec("Dining Room", "dining_room",
                              net * f["dining"], zone="service"))
    if plot_sqft >= 900:
        rooms.append(RoomSpec("Parking", "parking", net * f["parking"],
                              zone="public"))
    if bhk >= 2:
        rooms.append(RoomSpec("Passage", "hallway", net * f["passage"],
                              zone="private"))

    rooms.append(RoomSpec("Master Bedroom", "master_bedroom",
                          net * f["master"], zone="private"))
    for i in range(2, bhk + 1):
        rooms.append(RoomSpec(f"Bedroom {i}", "bedroom",
                              net * f["bedroom"], zone="private"))

    n_baths = max(1, (bhk + 1) // 2 + (1 if bhk >= 3 else 0))
    for i in range(1, n_baths + 1):
        rooms.append(RoomSpec(f"Bath {i}", "bathroom", net * f["bath"],
                              zone="private"))
    return rooms


# G+1 briefs (M2): the ground floor of a two-storey house. The staircase is
# NOT listed — the engine injects it (engine/program.py), which is exactly
# what these briefs exist to exercise. Chosen to span the range where a stair
# hurts: a 20' frontage where it competes for every foot, mid plots, and a
# large plot where it should be effortless.
_STAIR_PLOTS = [
    (20, 45, "S", 2),
    (25, 50, "E", 3),
    (30, 40, "S", 2),
    (30, 60, "N", 3),
    (40, 60, "S", 3),
    (50, 40, "N", 3),
]


# Irregular plots (M8): (name, boundary in feet, entrance side, bhk,
# setback ft). Real shapes, not decorative ones — an L from a corner sale,
# a road chamfer, a wedge from a curved street, a trapezoid.
_POLYGON_PLOTS = [
    ("L_40x50", [(0, 0), (40, 0), (40, 30), (22, 30), (22, 50), (0, 50)],
     "S", 2, 3.0),
    ("chamfer_35x45", [(0, 0), (35, 0), (35, 38), (27, 45), (0, 45)],
     "S", 3, 3.0),
    ("wedge_30x55", [(0, 0), (30, 0), (24, 55), (0, 55)], "N", 2, 2.0),
    ("trapezoid_45x40", [(0, 0), (45, 0), (38, 40), (7, 40)], "S", 3, 3.0),
]


def golden_briefs(k: int = 6, seed: int = 20260716,
                  include_stairs: bool = True,
                  include_polygons: bool = True) -> List[EngineRequest]:
    briefs = []
    for w, h, side, bhk in _PLOTS:
        briefs.append(EngineRequest(
            plot_w_ft=w, plot_h_ft=h, entrance_side=side,
            rooms=_program(bhk, w * h),
            k=k, seed=seed,
            name=f"{w}x{h}_{side}_{bhk}bhk",
        ))
    if include_stairs:
        briefs.extend(stair_briefs(k=k, seed=seed))
    if include_polygons:
        briefs.extend(polygon_briefs(k=k, seed=seed))
    return briefs


def polygon_briefs(k: int = 6, seed: int = 20260716) -> List[EngineRequest]:
    """Irregular plots. The program is sized against the BUILDABLE core,
    not the plot: a 1600 sqft L-shape with a 3' setback offers ~800 sqft to
    build in, and asking for a full 1600 sqft program would be testing the
    over-program warning rather than irregular geometry."""
    from modules.step4_generate.engine.site import build_site

    briefs = []
    for name, ring, side, bhk, setback in _POLYGON_PLOTS:
        site = build_site(ring, setback_ft=setback)
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        briefs.append(EngineRequest(
            plot_w_ft=int(max(xs) - min(xs)), plot_h_ft=int(max(ys) - min(ys)),
            entrance_side=side,
            rooms=_program(bhk, site.core_area_sqft),
            plot_polygon=ring, setback_ft=setback,
            k=k, seed=seed, name=name,
        ))
    return briefs


def stair_briefs(k: int = 6, seed: int = 20260716) -> List[EngineRequest]:
    """G+1 ground floors. `n_floors=2` is the whole trigger: the program
    engine injects the staircase and the STR-S rules judge it."""
    briefs = []
    for w, h, side, bhk in _STAIR_PLOTS:
        # a G+1 ground floor carries fewer bedrooms — they live upstairs
        rooms = [s for s in _program(bhk, w * h)
                 if s.rtype != "bedroom" or s.name == "Bedroom 2"]
        briefs.append(EngineRequest(
            plot_w_ft=w, plot_h_ft=h, entrance_side=side, rooms=rooms,
            n_floors=2, k=k, seed=seed,
            name=f"{w}x{h}_{side}_G+1",
        ))
    return briefs
