#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/infer_diffusion.py
========================
Post-training inference runner for PlanGen's GNN + Autoregressive Layout Transformer.

Generates SVG floor plans using the trained AR model weights
(weights/gnn_encoder.npz + weights/ar_transformer.npz) without any LLM calls.

Usage
-----
    source planenv/bin/activate
    python3 tools/infer_diffusion.py                        # all 4 presets (AR)
    python3 tools/infer_diffusion.py --preset 3bhk_villa    # one preset
    python3 tools/infer_diffusion.py --out output/my_run    # custom output dir
    python3 tools/infer_diffusion.py --cpsat                # compare with CP-SAT
    python3 tools/infer_diffusion.py --both                 # AR + CP-SAT side-by-side

Output
------
    output/diffusion_inference/
        3bhk_villa/
            diffusion/ground_floor.svg      ← actually AR now
            cpsat/ground_floor.svg          ← only with --cpsat / --both
        4bhk_duplex/diffusion/...
        2bhk_apartment/diffusion/...
        5bhk_bungalow/diffusion/...
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional

# ── Project root ──────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from models import (
    BuildingRequirements, EnrichedPlan, EnrichedRoom,
    FloorPlan, Setbacks,
)
from modules.step4_generate.generator import LayoutGenerator
from modules.step4_generate.renderer import FloorPlanRenderer


# ═══════════════════════════════════════════════════════════════════════════════
# PRESET FLOOR PLAN CONFIGS
# ═══════════════════════════════════════════════════════════════════════════════

def _er(rid, rtype, display, floor, w, l,
        adj_prefs: Optional[Dict] = None) -> EnrichedRoom:
    """Quick EnrichedRoom builder with sane defaults."""
    area = w * l
    return EnrichedRoom(
        room_id=rid, room_type=rtype, display_name=display,
        preferred_floor=floor, implicit_room=False,
        target_width_ft=w, target_length_ft=l, target_area_sqft=area,
        min_width_ft=max(6.0, w * 0.6),
        min_length_ft=max(6.0, l * 0.6),
        min_area_sqft=max(36.0, area * 0.5),
        max_area_sqft=area * 1.5,
        ceiling_height_ft=9.0,
        adjacency_preferences=adj_prefs or {},
        should_not_be_adjacent_to=[],
    )


def _build_adjacency_graph(rooms: List[EnrichedRoom]) -> Dict:
    """
    Derive a room-id-keyed adjacency graph from each room's
    adjacency_preferences (which use room *types*, not IDs).

    adjacency_preferences: {room_type_str: weight_float}
    → adjacency_graph:     {room_id: {other_room_id: weight}}
    """
    type_to_ids: Dict[str, List[str]] = {}
    for r in rooms:
        type_to_ids.setdefault(r.room_type, []).append(r.room_id)

    graph: Dict[str, Dict[str, float]] = {}
    for room in rooms:
        if not room.adjacency_preferences:
            continue
        graph[room.room_id] = {}
        for adj_type, weight in room.adjacency_preferences.items():
            w = float(min(max(weight, 0.0), 1.0))
            for partner_id in type_to_ids.get(adj_type, []):
                if partner_id != room.room_id:
                    graph[room.room_id][partner_id] = w

    return graph


def _make_plan(
    project_name: str,
    plot_w: float, plot_l: float,
    setbacks: Dict[str, float],
    entrance: str,
    rooms: List[EnrichedRoom],
    n_floors: int = 1,
) -> EnrichedPlan:
    """Build a fully-formed EnrichedPlan from raw specs."""
    sw = setbacks.get("left", 1.5) + setbacks.get("right", 1.5)
    sl = setbacks.get("front", 3.0) + setbacks.get("rear", 2.0)
    net_w = plot_w - sw
    net_l = plot_l - sl

    floors: List[FloorPlan] = []
    for fn in range(n_floors):
        floor_rooms = [r for r in rooms if r.preferred_floor == fn]
        floors.append(FloorPlan(
            floor_number=fn,
            floor_label="Ground Floor" if fn == 0 else f"Floor {fn}",
            room_ids=[r.room_id for r in floor_rooms],
            gross_area_sqft=sum(r.target_area_sqft for r in floor_rooms),
        ))

    req = BuildingRequirements(
        plot_dimensions={"length": plot_l, "width": plot_w, "unit": "ft"},
        rooms=[{"room_type": r.room_type, "quantity": 1} for r in rooms],
    )

    return EnrichedPlan(
        original_requirements=req,
        match_quality_score=0.80,
        plot_width_ft=plot_w, plot_length_ft=plot_l,
        plot_area_sqft=plot_w * plot_l,
        setbacks=Setbacks(
            front=setbacks.get("front", 3.0),
            rear=setbacks.get("rear", 2.0),
            left=setbacks.get("left", 1.5),
            right=setbacks.get("right", 1.5),
            unit="ft",
        ),
        net_buildable_width_ft=net_w,
        net_buildable_length_ft=net_l,
        net_buildable_area_sqft=net_w * net_l,
        entrance_direction=entrance,
        north_direction="N",
        total_floors=n_floors,
        floors=floors,
        rooms=rooms,
        implicit_rooms_added=[],
        adjacency_graph=_build_adjacency_graph(rooms),
        vastu_enabled=True,
        vastu_direction_assignments={},
        max_ground_coverage_sqft=plot_w * plot_l * 0.5,
        max_far_total_sqft=plot_w * plot_l * 1.5,
        total_target_area_sqft=sum(r.target_area_sqft for r in rooms),
        area_budget_ok=True,
        enrichment_source="ar_inference",
        enrichment_warnings=[],
        gemini_decisions=[],
    )


# ── Preset: 3BHK Villa (30×40 ft) ────────────────────────────────────────────
def preset_3bhk_villa() -> tuple[str, EnrichedPlan]:
    rooms = [
        _er("lr1",  "living_room",    "Living Room",    0, 14, 12,
            adj_prefs={"dining_room": 0.9, "foyer": 0.8}),
        _er("ki1",  "kitchen",        "Kitchen",        0, 10,  8,
            adj_prefs={"dining_room": 0.9}),
        _er("dr1",  "dining_room",    "Dining Room",    0, 12,  8,
            adj_prefs={"living_room": 0.9, "kitchen": 0.9}),
        _er("mb1",  "master_bedroom", "Master Bedroom", 0, 14, 12,
            adj_prefs={"bathroom": 0.9}),
        _er("bd1",  "bedroom",        "Bedroom 2",      0, 12, 10),
        _er("bd2",  "bedroom",        "Bedroom 3",      0, 12, 10),
        _er("bt1",  "bathroom",       "Bathroom 1",     0,  6,  8),
        _er("bt2",  "bathroom",       "Bathroom 2",     0,  6,  8),
        _er("pr1",  "pooja_room",     "Pooja Room",     0,  6,  6),
        _er("st1",  "staircase",      "Staircase",      0,  8,  6),
        _er("cp1",  "car_parking",    "Car Parking",    0, 12,  8),
    ]
    plan = _make_plan(
        "3BHK Villa", 30, 40,
        {"front": 3.0, "rear": 2.0, "left": 1.5, "right": 1.5},
        "N", rooms, n_floors=1,
    )
    return "3bhk_villa", plan


# ── Preset: 4BHK Duplex (40×50 ft, G+1) ─────────────────────────────────────
def preset_4bhk_duplex() -> tuple[str, EnrichedPlan]:
    rooms = [
        # Ground floor
        _er("fo1",  "foyer",          "Foyer",          0, 10, 10,
            adj_prefs={"living_room": 1.0}),
        _er("lr1",  "living_room",    "Living Room",    0, 16, 14,
            adj_prefs={"dining_room": 0.9, "foyer": 1.0}),
        _er("dr1",  "dining_room",    "Dining Room",    0, 16, 12,
            adj_prefs={"kitchen": 0.9, "living_room": 0.9}),
        _er("ki1",  "kitchen",        "Kitchen",        0, 14, 10,
            adj_prefs={"dining_room": 0.9}),
        _er("mb1",  "master_bedroom", "Master Bedroom", 0, 14, 12,
            adj_prefs={"master_bath": 1.0}),
        _er("mbt",  "bathroom",       "Master Bath",    0, 14,  6,
            adj_prefs={"master_bedroom": 1.0}),
        _er("bt1",  "bathroom",       "Bathroom",       0,  8, 10),
        _er("ut1",  "utility",        "Utility",        0,  8,  8),
        _er("pr1",  "pooja_room",     "Pooja Room",     0,  6,  8),
        _er("st1",  "staircase",      "Staircase",      0,  8, 10),
        _er("cp1",  "car_parking",    "Car Parking",    0, 10, 10),
        # First floor
        _er("bd1",  "bedroom",        "Bedroom 2",      1, 14, 14),
        _er("bd2",  "bedroom",        "Bedroom 3",      1, 14, 14),
        _er("bd3",  "bedroom",        "Kids Bedroom",   1, 12, 12),
        _er("bt2",  "bathroom",       "Bathroom 1",     1, 12,  8),
        _er("bt3",  "bathroom",       "Bathroom 2",     1, 12,  8),
        _er("sr1",  "study_room",     "Study Room",     1, 14, 10),
        _er("bl1",  "balcony",        "Balcony",        1, 14,  6),
        _er("tr1",  "terrace",        "Terrace",        1, 12, 18),
        _er("st2",  "staircase",      "Staircase",      1,  8, 10),
    ]
    plan = _make_plan(
        "4BHK Duplex Villa", 40, 50,
        {"front": 5.0, "rear": 3.0, "left": 2.0, "right": 2.0},
        "N", rooms, n_floors=2,
    )
    return "4bhk_duplex", plan


# ── Preset: 2BHK Apartment (24×32 ft) ────────────────────────────────────────
def preset_2bhk_apartment() -> tuple[str, EnrichedPlan]:
    rooms = [
        _er("lr1",  "living_room",    "Living Room",    0, 12, 10,
            adj_prefs={"dining_room": 0.9}),
        _er("dr1",  "dining_room",    "Dining Room",    0, 10,  8),
        _er("ki1",  "kitchen",        "Kitchen",        0,  8,  8,
            adj_prefs={"dining_room": 0.9}),
        _er("mb1",  "master_bedroom", "Master Bedroom", 0, 12, 10),
        _er("bd1",  "bedroom",        "Bedroom 2",      0, 10, 10),
        _er("bt1",  "bathroom",       "Bathroom 1",     0,  5,  7),
        _er("bt2",  "bathroom",       "Bathroom 2",     0,  5,  7),
        _er("bl1",  "balcony",        "Balcony",        0, 10,  5),
    ]
    plan = _make_plan(
        "2BHK Apartment", 24, 32,
        {"front": 2.0, "rear": 1.5, "left": 1.0, "right": 1.0},
        "E", rooms, n_floors=1,
    )
    return "2bhk_apartment", plan


# ── Preset: 5BHK Bungalow (50×60 ft) ─────────────────────────────────────────
def preset_5bhk_bungalow() -> tuple[str, EnrichedPlan]:
    rooms = [
        _er("fo1",  "foyer",          "Foyer",          0, 12, 10),
        _er("lr1",  "living_room",    "Living Room",    0, 20, 16,
            adj_prefs={"dining_room": 0.9, "foyer": 1.0}),
        _er("dr1",  "dining_room",    "Dining Room",    0, 18, 14,
            adj_prefs={"kitchen": 0.9, "living_room": 0.9}),
        _er("ki1",  "kitchen",        "Kitchen",        0, 16, 12,
            adj_prefs={"dining_room": 0.9, "utility": 0.8}),
        _er("ut1",  "utility",        "Utility",        0, 10,  8),
        _er("mb1",  "master_bedroom", "Master Bedroom", 0, 16, 14,
            adj_prefs={"bathroom": 1.0}),
        _er("bd1",  "bedroom",        "Bedroom 2",      0, 14, 12),
        _er("bd2",  "bedroom",        "Bedroom 3",      0, 14, 12),
        _er("bd3",  "bedroom",        "Bedroom 4",      0, 14, 12),
        _er("bd4",  "bedroom",        "Bedroom 5",      0, 14, 12),
        _er("bt1",  "bathroom",       "Bathroom 1",     0,  8, 10),
        _er("bt2",  "bathroom",       "Bathroom 2",     0,  8, 10),
        _er("bt3",  "bathroom",       "Bathroom 3",     0,  8, 10),
        _er("pr1",  "pooja_room",     "Pooja Room",     0,  8,  8),
        _er("st1",  "staircase",      "Staircase",      0, 10, 10),
        _er("cp1",  "car_parking",    "Car Parking",    0, 20, 10),
        _er("gd1",  "garden",         "Garden",         0, 20, 15),
    ]
    plan = _make_plan(
        "5BHK Bungalow", 50, 60,
        {"front": 6.0, "rear": 4.0, "left": 3.0, "right": 3.0},
        "N", rooms, n_floors=1,
    )
    return "5bhk_bungalow", plan


PRESETS = {
    "3bhk_villa":     preset_3bhk_villa,
    "4bhk_duplex":    preset_4bhk_duplex,
    "2bhk_apartment": preset_2bhk_apartment,
    "5bhk_bungalow":  preset_5bhk_bungalow,
}


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_inference(
    preset_name: str,
    plan: EnrichedPlan,
    out_root: str,
    use_ar: bool = True,
    use_cpsat: bool = False,
) -> None:
    """Run layout generation + rendering for one preset."""
    print(f"\n{'─' * 60}")
    print(f"  {preset_name.upper().replace('_', ' ')}  ({plan.plot_width_ft:.0f}×{plan.plot_length_ft:.0f} ft)")
    print(f"  Rooms: {len(plan.rooms)} | Floors: {plan.total_floors}")
    print(f"{'─' * 60}")

    solvers = []
    if use_ar:
        solvers.append(("diffusion", True, False))   # label, use_ar, prefer_cpsat
    if use_cpsat:
        solvers.append(("cpsat",    False, True))

    for solver_label, _use_ar, prefer_cpsat in solvers:
        t0 = time.perf_counter()
        out_dir = os.path.join(out_root, preset_name, solver_label)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n  [{solver_label.upper()}] Generating layout …")
        # ── use_autoregressive=True → tries AR engine first, then CP-SAT fallback
        gen = LayoutGenerator(
            prefer_cpsat       = prefer_cpsat,
            use_autoregressive = _use_ar,
        )
        layout = gen.generate(plan)

        elapsed = time.perf_counter() - t0
        print(f"  Solver used : {layout.solver_used}")
        print(f"  Quality     : {layout.layout_quality_score:.3f}")
        print(f"  Rooms placed: {layout.total_rooms_placed}")
        print(f"  Time        : {elapsed:.2f}s")

        # ── Render ───────────────────────────────────────────────────────────
        tag = "[AR]" if solver_label == "diffusion" else "[CP-SAT]"
        project_title = (
            preset_name.replace("_", " ").title() + " " + tag
        )

        renderer = FloorPlanRenderer(
            layout,
            output_dir=out_dir,
            project_name=project_title,
        )
        svg_paths = renderer.render_all()

        for p in svg_paths:
            size_kb = os.path.getsize(p) / 1024
            print(f"  ✓  {os.path.relpath(p, _ROOT)}  ({size_kb:.1f} KB)")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PlanGen — Post-training AR inference runner"
    )
    parser.add_argument(
        "--preset", choices=list(PRESETS.keys()),
        default=None,
        help="Run a single preset (default: all presets)",
    )
    parser.add_argument(
        "--out", default=os.path.join(_ROOT, "output", "diffusion_inference"),
        help="Output root directory for SVG files",
    )
    parser.add_argument(
        "--cpsat", action="store_true",
        help="Also generate CP-SAT layouts for comparison",
    )
    parser.add_argument(
        "--both", action="store_true",
        help="Generate both AR AND CP-SAT (implies --cpsat)",
    )
    args = parser.parse_args()

    use_ar    = True
    use_cpsat = args.cpsat or args.both

    # ── Check AR weights ──────────────────────────────────────────────────────
    from modules.step4_generate.autoregressive_engine import AutoregressiveLayoutEngine
    eng = AutoregressiveLayoutEngine()
    if not eng.is_ready():
        _weights = os.path.join(_ROOT, "modules", "step4_generate", "weights")
        print("ERROR: AR Transformer weights not found.")
        print(f"  Expected: {_weights}/ar_transformer.npz")
        print()
        print("  Option A — retrain from scratch (recommended after the 5 bug fixes):")
        print("    python3 modules/step4_generate/training/model_trainer.py \\")
        print("            --cache_dir modules/step4_generate/weights/cache \\")
        print("            --out_dir   modules/step4_generate/weights \\")
        print("            --epochs 50")
        print()
        print("  Option B — re-export an existing checkpoint (if you have one):")
        print("    python3 tools/reexport_weights.py --verify")
        sys.exit(1)

    print("=" * 60)
    print("  PlanGen — AR Transformer Inference Runner")
    print("=" * 60)
    print(f"  Weights  : modules/step4_generate/weights/")
    print(f"  Output   : {os.path.relpath(args.out, _ROOT)}/")
    print(f"  Solvers  : AR" + (" + CP-SAT" if use_cpsat else ""))
    print(f"  Presets  : {args.preset or 'all (' + str(len(PRESETS)) + ')'}")

    # ── Run presets ───────────────────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    t_total = time.perf_counter()

    preset_items = (
        [(args.preset, PRESETS[args.preset])]
        if args.preset
        else list(PRESETS.items())
    )

    for preset_name, preset_fn in preset_items:
        _, plan = preset_fn()
        run_inference(
            preset_name, plan,
            out_root=args.out,
            use_ar=use_ar,
            use_cpsat=use_cpsat,
        )

    print(f"\n{'═' * 60}")
    print(f"  ✅  Done in {time.perf_counter() - t_total:.1f}s")
    print(f"  SVGs saved to: {os.path.relpath(args.out, _ROOT)}/")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
