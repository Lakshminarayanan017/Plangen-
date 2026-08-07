"""
demo_carve.py — end-to-end proof of the wall-graph substrate.

Carves two plots (a 30'x40' and a narrow 20'x45' like the reference example),
verifies all structural invariants, cuts door/wide openings on shared walls,
prints target-vs-achieved areas, and renders SVGs to output/.

Run from the project root:  python -m modules.step4_generate.demos.demo_carve
"""

from __future__ import annotations

import os

from modules.step4_generate.carve.strip_carver import RoomBrief, carve
from modules.step4_generate.core import units
from modules.step4_generate.core.grid_plan import CarveError, GridPlan
from modules.step4_generate.render.svg_render import save_svg

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "output")


def build(plot_ft, briefs, wish_openings, tag, title):
    plan = GridPlan.from_feet(*plot_ft)
    carve(plan, 1, briefs)
    plan.verify()

    placed, skipped = [], []
    for a_name, b_name, kind in wish_openings:
        try:
            a, b = plan.find_room(a_name).id, plan.find_room(b_name).id
            plan.add_opening(a, b, kind)
            placed.append(f"{a_name} <-> {b_name} ({kind})")
        except (CarveError, KeyError) as e:
            skipped.append(f"{a_name} <-> {b_name}: {e}")
    plan.verify()

    print("=" * 64)
    print(title)
    print("=" * 64)
    print(plan.summary())
    print("\n  target vs achieved:")
    for brief in briefs:
        room = plan.find_room(brief.name)
        got = plan.area_sqft(room.id)
        drift = (got - brief.target_sqft) / brief.target_sqft * 100
        print(f"    {brief.name:<16} target {brief.target_sqft:>5.0f}  "
              f"got {got:>6.1f} sqft  ({drift:+.0f}%)")
    print(f"\n  openings placed : {len(placed)}")
    for p in placed:
        print(f"    + {p}")
    if skipped:
        print(f"  openings skipped (no shared wall yet — settle/parti will "
              f"fix adjacency):")
        for s_ in skipped:
            print(f"    - {s_}")

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.abspath(os.path.join(OUT_DIR, f"{tag}.svg"))
    save_svg(plan, path, title)
    print(f"\n  SVG -> {path}\n")


def main():
    build(
        (30, 40),
        [
            RoomBrief("Living Room", "living_room", 190),
            RoomBrief("Dining", "dining_room", 130),
            RoomBrief("Kitchen", "kitchen", 110),
            RoomBrief("Passage", "passage", 70),
            RoomBrief("Master Bedroom", "master_bedroom", 165),
            RoomBrief("Bath 1", "bathroom", 45),
            RoomBrief("Bedroom 2", "bedroom", 140),
            RoomBrief("Bath 2", "bathroom", 40),
        ],
        [
            ("Living Room", "Dining", "wide"),
            ("Dining", "Kitchen", "wide"),
            ("Living Room", "Passage", "wide"),
            ("Passage", "Master Bedroom", "door"),
            ("Passage", "Bedroom 2", "door"),
            ("Master Bedroom", "Bath 1", "door"),
            ("Bedroom 2", "Bath 2", "door"),
        ],
        "demo_30x40",
        "30' x 40' - 2 bed demo (v0 carver, pre-settle)",
    )

    build(
        (20, 45),
        [
            RoomBrief("Parking", "parking", 118),
            RoomBrief("Drawing Room", "drawing_room", 80),
            RoomBrief("Kitchen", "kitchen", 62),
            RoomBrief("Dining Room", "dining_room", 82),
            RoomBrief("Bed Room 3", "bedroom", 90),
            RoomBrief("C. Bath 2", "bathroom", 20),
            RoomBrief("Bed Room 1", "bedroom", 82),
            RoomBrief("Bed Room 2", "bedroom", 90),
            RoomBrief("C. Bath 1", "bathroom", 26),
        ],
        [
            ("Drawing Room", "Dining Room", "wide"),
            ("Dining Room", "Kitchen", "wide"),
            ("Parking", "Drawing Room", "door"),
            ("Dining Room", "Bed Room 3", "door"),
            ("Bed Room 1", "C. Bath 1", "door"),
            ("Bed Room 3", "C. Bath 2", "door"),
        ],
        "demo_20x45",
        "20' x 45' - 3 bed row house (v0 carver, pre-settle)",
    )


if __name__ == "__main__":
    main()
