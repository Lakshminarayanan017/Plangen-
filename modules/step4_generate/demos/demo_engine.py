"""
demo_engine.py — E2+E3 proof: full production pipeline.

propose → hub-carve (seed-guided bands) → squeeze & settle → openness
gradient (kitchen doorless, main door, reachability) → validate → rank.

Run from the project root:  python -m modules.step4_generate.demos.demo_engine
"""

from __future__ import annotations

import os

from modules.step4_generate.engine.contracts import EngineRequest, RoomSpec
from modules.step4_generate.engine.orchestrator import Orchestrator
from modules.step4_generate.render.svg_render import save_svg

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "output")


def run(request: EngineRequest, tag: str, title: str) -> None:
    result = Orchestrator().generate(request)

    print("=" * 70)
    print(title)
    print("=" * 70)
    for key, val in result.tier_notes.items():
        print(f"  {key:<16} {val}")
    print()
    print(f"  {'#':<3}{'score':<9}{'fidelity':<10}{'drift':<9}"
          f"{'wides':<7}{'aspect':<8}notes")
    for i, cand in enumerate(result.ranked):
        b = cand.verdict.breakdown
        interesting = [n for n in cand.notes
                       if n.startswith(("main door", "settle_error",
                                        "connectivity"))]
        print(f"  {i:<3}{cand.verdict.soft_score:<9}{cand.fidelity:<10.2f}"
              f"{b['area_drift']:<9}{b['wide_openings']:<7}"
              f"{b['worst_aspect']:<8}{'; '.join(interesting)}")
    for cand in result.discarded:
        print(f"  X  discarded: {cand.notes[-1] if cand.notes else '?'}")

    os.makedirs(OUT_DIR, exist_ok=True)
    if result.best:
        best = result.best
        print()
        print(best.plan.summary())
        doors = sum(1 for op in best.plan.openings if op.kind == "door")
        wides = sum(1 for op in best.plan.openings if op.kind == "wide")
        print(f"\n  openings: {doors} doors, {wides} wide (doorless) "
              f"+ main entrance")
        path = os.path.abspath(os.path.join(OUT_DIR, f"{tag}.svg"))
        save_svg(best.plan, path, title)
        print(f"  SVG -> {path}\n")


def main():
    run(
        EngineRequest(
            plot_w_ft=20, plot_h_ft=45, entrance_side="S",
            rooms=[
                RoomSpec("Parking", "parking", 118, zone="public"),
                RoomSpec("Drawing Room", "drawing_room", 80, zone="public"),
                RoomSpec("Dining Room", "dining_room", 82, zone="service"),
                RoomSpec("Kitchen", "kitchen", 62, zone="service"),
                RoomSpec("Passage", "hallway", 42, zone="private"),
                RoomSpec("Bed Room 3", "bedroom", 90, zone="private"),
                RoomSpec("C. Bath 2", "bathroom", 20, zone="private"),
                RoomSpec("Bed Room 1", "bedroom", 82, zone="private"),
                RoomSpec("C. Bath 1", "bathroom", 26, zone="private"),
                RoomSpec("Bed Room 2", "bedroom", 90, zone="private"),
            ],
            k=8, seed=7,
        ),
        "engine_20x45",
        "20' x 45' 3BHK row house — E2+E3 (hub carve + settle + gradient)",
    )

    run(
        EngineRequest(
            plot_w_ft=30, plot_h_ft=40, entrance_side="S",
            rooms=[
                RoomSpec("Living Room", "living_room", 190, zone="public"),
                RoomSpec("Dining", "dining_room", 130, zone="service"),
                RoomSpec("Kitchen", "kitchen", 110, zone="service"),
                RoomSpec("Master Bedroom", "master_bedroom", 165,
                         zone="private"),
                RoomSpec("Bath 1", "bathroom", 45, zone="private"),
                RoomSpec("Bedroom 2", "bedroom", 140, zone="private"),
                RoomSpec("Bath 2", "bathroom", 40, zone="private"),
            ],
            k=8, seed=11,
        ),
        "engine_30x40",
        "30' x 40' 2BHK — E2+E3 (hub carve + settle + gradient)",
    )


if __name__ == "__main__":
    main()
