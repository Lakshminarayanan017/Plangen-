# -*- coding: utf-8 -*-
"""
generator.py
============
Step 4 - Layout Generator orchestrator.

Converts an EnrichedPlan (Step 3 output) into a LayoutPlan with exact
2-D room coordinates (x, y, width, length in feet) on each floor.

Solver cascade per floor:
  1. GNN + Diffusion engine (primary, highest quality) -- if weights exist
  2. CP-SAT solver (timeout 20s)  --  fallback
  3. GreedyPlacer                 --  final fallback

Output: LayoutPlan (Pydantic model defined in models.py)
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

from models import (
    EnrichedPlan,
    EnrichedRoom,
    LayoutFloor,
    LayoutPlan,
    PlacedRoom,
)
from modules.step4_generate.grid import EXTERNAL_ROOM_TYPES, PlotGrid
from modules.step4_generate.greedy_placer import WALL_TOL_FT, GreedyPlacer
from modules.step4_generate.solver import CP_SAT_TIMEOUT_S, CPSATSolver, RoomPlacement
from modules.step4_generate.autoregressive_engine import AutoregressiveLayoutEngine

log = logging.getLogger("PlanGen.Generator")

# Use greedy directly when <= this many rooms on a floor (faster, equally good)
GREEDY_THRESHOLD = 4

# Floor labels
_FLOOR_LABELS = {
    0: "Ground Floor",
    1: "First Floor",
    2: "Second Floor",
    3: "Third Floor",
}


class LayoutGenerator:
    """
    Orchestrates Step 4: EnrichedPlan -> LayoutPlan.

    Thread-safe: stateless between calls.
    """

    def __init__(
        self,
        prefer_cpsat:        bool  = True,
        cpsat_timeout_s:     float = CP_SAT_TIMEOUT_S,
        use_autoregressive:  bool  = True,
    ) -> None:
        self._prefer_cpsat  = prefer_cpsat
        self._timeout       = cpsat_timeout_s
        self._cp_solver     = CPSATSolver()
        self._greedy_placer = GreedyPlacer()

        # GNN + Autoregressive engine — loaded lazily; safe if weights missing
        self._ar_engine: Optional[AutoregressiveLayoutEngine] = None
        if use_autoregressive:
            try:
                self._ar_engine = AutoregressiveLayoutEngine()
                log.info("AutoregressiveLayoutEngine active — primary solver")
            except Exception as exc:
                log.warning("Could not init AutoregressiveLayoutEngine: %s", exc)
                self._ar_engine = None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        plan: EnrichedPlan,
        run_id: str = "",
        source_json: str = "",
    ) -> LayoutPlan:
        """
        Generate a LayoutPlan from an EnrichedPlan.

        Args:
            plan        : Fully-specified EnrichedPlan from Step 3.
            run_id      : Pipeline run ID (for traceability).
            source_json : Path to the step3 JSON file (for reference).

        Returns:
            LayoutPlan with all rooms placed on their floors.
        """
        t0 = time.perf_counter()
        log.info(
            "LayoutGenerator starting -- %d rooms, %d floors, %s x %s ft net",
            len(plan.rooms), plan.total_floors,
            plan.net_buildable_width_ft, plan.net_buildable_length_ft,
        )

        net_w = plan.net_buildable_width_ft
        net_l = plan.net_buildable_length_ft
        warnings: List[str] = []

        # Build the coordinate grid for zone + compass scoring
        grid = PlotGrid(
            net_width_ft  = net_w,
            net_length_ft = net_l,
            entrance_dir  = plan.entrance_direction,
            north_dir     = plan.north_direction,
        )

        # ---- Place floor by floor ----------------------------------------
        layout_floors: List[LayoutFloor] = []
        staircase_anchor: Optional[Tuple[float, float, float, float]] = None
        total_solver_ms = 0.0
        solver_types_used: List[str] = []

        for floor_num in range(plan.total_floors):
            all_on_floor   = plan.get_rooms_on_floor(floor_num)
            internal_rooms = [r for r in all_on_floor
                              if r.room_type not in EXTERNAL_ROOM_TYPES]
            external_rooms = [r for r in all_on_floor
                              if r.room_type in EXTERNAL_ROOM_TYPES]

            log.info(
                "Floor %d: %d internal + %d external rooms",
                floor_num, len(internal_rooms), len(external_rooms),
            )

            if not internal_rooms and not external_rooms:
                layout_floors.append(LayoutFloor(
                    floor_number  = floor_num,
                    floor_label   = _FLOOR_LABELS.get(floor_num, f"Floor {floor_num}"),
                    net_width_ft  = net_w,
                    net_length_ft = net_l,
                ))
                continue

            # -- Solve internal rooms --------------------------------------
            floor_t0 = time.perf_counter()
            placements, solver_used, solver_status = self._solve_floor(
                rooms            = internal_rooms,
                net_w            = net_w,
                net_l            = net_l,
                grid             = grid,
                adj_graph        = plan.adjacency_graph,
                staircase_anchor = staircase_anchor,
                floor_num        = floor_num,
                warnings         = warnings,
                total_floors     = plan.total_floors,
                plan             = plan,
            )
            floor_ms = (time.perf_counter() - floor_t0) * 1000
            total_solver_ms += floor_ms
            solver_types_used.append(solver_used)

            # -- Track staircase anchor for upper floors -------------------
            if staircase_anchor is None:
                staircase_anchor = self._find_staircase_anchor(
                    internal_rooms, placements
                )

            # -- Build PlacedRoom list with scores -------------------------
            placed_rooms = self._build_placed_rooms(
                internal_rooms, placements, floor_num,
                grid, plan.adjacency_graph, warnings,
            )

            # -- Place external rooms in setback zone ----------------------
            ext_placed = self._place_external_rooms(
                external_rooms, floor_num, plan, grid
            )
            placed_rooms.extend(ext_placed)

            # -- Build LayoutFloor ----------------------------------------
            floor_area = sum(
                r.area_sqft for r in placed_rooms
                if r.room_type not in EXTERNAL_ROOM_TYPES
            )
            coverage   = (
                round(floor_area / (net_w * net_l) * 100, 1)
                if net_w * net_l > 0 else 0.0
            )
            adj_score  = self._floor_adjacency_score(placed_rooms, plan.adjacency_graph)
            zone_score = self._floor_zone_score(placed_rooms)

            layout_floors.append(LayoutFloor(
                floor_number           = floor_num,
                floor_label            = _FLOOR_LABELS.get(floor_num, f"Floor {floor_num}"),
                net_width_ft           = net_w,
                net_length_ft          = net_l,
                rooms                  = placed_rooms,
                floor_area_placed_sqft = round(floor_area, 2),
                floor_coverage_pct     = coverage,
                floor_adjacency_score  = adj_score,
                floor_zone_score       = zone_score,
            ))

            log.info(
                "Floor %d done: %d rooms, %.1f sqft (%.1f%%), "
                "adj=%.3f zone=%.3f [%s, %.0fms]",
                floor_num, len(placed_rooms), floor_area, coverage,
                adj_score, zone_score, solver_used, floor_ms,
            )

        # ---- Aggregate metrics -------------------------------------------
        total_rooms = sum(len(f.rooms) for f in layout_floors)
        total_area  = sum(
            r.area_sqft for f in layout_floors
            for r in f.rooms if r.room_type not in EXTERNAL_ROOM_TYPES
        )
        all_placed  = [r for f in layout_floors for r in f.rooms]
        overall_adj  = self._floor_adjacency_score(all_placed, plan.adjacency_graph)
        overall_zone = self._floor_zone_score(all_placed)
        quality      = round(0.5 * overall_adj + 0.5 * overall_zone, 3)

        # Dominant solver  (autoregressive > cp_sat > greedy)
        sc = Counter(solver_types_used)
        if sc.get("autoregressive", 0) > 0:
            dominant_solver = "autoregressive"
            solver_status_str = "ar_sampled"
        elif sc.get("cp_sat", 0) >= sc.get("greedy", 0):
            dominant_solver = "cp_sat"
            solver_status_str = "optimal"
        else:
            dominant_solver = "greedy"
            solver_status_str = "greedy"

        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.info(
            "LayoutGenerator done: %d rooms, %.1f sqft, quality=%.3f, "
            "solver=%s, %.0fms total",
            total_rooms, total_area, quality, dominant_solver, elapsed_ms,
        )

        return LayoutPlan(
            run_id                  = run_id,
            source_step3_json       = source_json,
            plot_width_ft           = plan.plot_width_ft,
            plot_length_ft          = plan.plot_length_ft,
            net_buildable_width_ft  = net_w,
            net_buildable_length_ft = net_l,
            setback_front_ft        = plan.setbacks.front or 0.0,
            setback_rear_ft         = plan.setbacks.rear  or 0.0,
            setback_left_ft         = plan.setbacks.left  or 0.0,
            setback_right_ft        = plan.setbacks.right or 0.0,
            entrance_direction      = plan.entrance_direction,
            north_direction         = plan.north_direction,
            vastu_enabled           = plan.vastu_enabled,
            total_floors            = plan.total_floors,
            floors                  = layout_floors,
            total_rooms_placed      = total_rooms,
            total_area_placed_sqft  = round(total_area, 2),
            overall_adjacency_score = overall_adj,
            overall_zone_score      = overall_zone,
            layout_quality_score    = quality,
            solver_used             = dominant_solver,
            solve_time_ms           = round(elapsed_ms, 1),
            solver_status           = solver_status_str,
            layout_warnings         = warnings,
        )

    # ------------------------------------------------------------------ #
    #  Per-floor solver cascade                                            #
    # ------------------------------------------------------------------ #

    def _solve_floor(
        self,
        rooms:            List[EnrichedRoom],
        net_w:            float,
        net_l:            float,
        grid:             PlotGrid,
        adj_graph:        Dict,
        staircase_anchor: Optional[Tuple],
        floor_num:        int,
        warnings:         List[str],
        total_floors:     int = 1,
        plan:             Optional[EnrichedPlan] = None,
    ) -> Tuple[List[RoomPlacement], str, str]:
        """
        Solver cascade: AR Transformer → CP-SAT → Greedy.

        1. AutoregressiveLayoutEngine — generates room types AND positions
           jointly, conditioned on all previously placed rooms.  Uses
           area_fraction + generation_order set by the enricher.  Returns
           PlacedRoom objects directly (not RoomPlacement), so they are
           converted for compatibility.

        2. CP-SAT — constraint programming, globally optimal within timeout.
           Used when AR engine is not loaded or as a quality comparison.

        3. GreedyPlacer — fast fallback, always succeeds.
        """

        # 1. AR Transformer (primary — enabled when weights loaded)
        _AR_QUALITY_GATE = 0.25   # minimum acceptable quality to keep AR output
        if self._ar_engine is not None and plan is not None and len(rooms) >= 2:
            try:
                placed_rooms = self._ar_engine.place_floor(
                    plan         = plan,
                    floor_number = floor_num,
                )
                if placed_rooms:
                    # ── Quality gate: reject obviously bad AR layouts ──────
                    # Compute mean adjacency + zone score from placed rooms.
                    # With trained weights this should be ≥ 0.40; with
                    # placeholder weights it may be < 0.20 (garbage).
                    n_placed = len(placed_rooms)
                    avg_adj  = sum(r.adjacency_score for r in placed_rooms) / max(n_placed, 1)
                    avg_zone = sum(r.zone_score for r in placed_rooms) / max(n_placed, 1)
                    ar_quality = 0.5 * avg_adj + 0.5 * avg_zone

                    log.info(
                        "Floor %d: AR quality check — adj=%.3f zone=%.3f "
                        "combined=%.3f (gate=%.2f)",
                        floor_num, avg_adj, avg_zone, ar_quality, _AR_QUALITY_GATE,
                    )

                    if ar_quality < _AR_QUALITY_GATE:
                        log.warning(
                            "Floor %d: AR output quality %.3f < %.2f gate "
                            "— rejecting, will try CP-SAT",
                            floor_num, ar_quality, _AR_QUALITY_GATE,
                        )
                        warnings.append(
                            f"Floor {floor_num}: AR quality {ar_quality:.3f} below "
                            f"threshold — using CP-SAT instead."
                        )
                    else:
                        # Convert PlacedRoom → RoomPlacement for compatibility
                        placements = [
                            RoomPlacement(
                                room_id   = r.room_id,
                                x_ft      = r.x_ft,
                                y_ft      = r.y_ft,
                                width_ft  = r.width_ft,
                                length_ft = r.length_ft,
                            )
                            for r in placed_rooms
                        ]
                        log.info("Floor %d: AR Transformer placed %d rooms "
                                 "(quality=%.3f)",
                                 floor_num, len(placements), ar_quality)
                        return placements, "autoregressive", "ar_sampled"
            except Exception as exc:
                log.warning(
                    "Floor %d: AR engine failed (%s) — falling back to CP-SAT",
                    floor_num, exc,
                )
                warnings.append(
                    f"Floor {floor_num}: AR engine failed ({exc}) — using CP-SAT."
                )

        # 2. Small floors: greedy is faster and equally good
        if not self._prefer_cpsat or len(rooms) <= GREEDY_THRESHOLD:
            placements = self._greedy_placer.place(
                rooms, net_w, net_l, grid, adj_graph, staircase_anchor
            )
            return placements, "greedy", "greedy"

        # 3. CP-SAT
        placements, status = self._cp_solver.solve(
            rooms, net_w, net_l, grid, adj_graph,
            timeout_s        = self._timeout,
            staircase_anchor = staircase_anchor,
        )
        if placements:
            return placements, "cp_sat", status

        # 4. Greedy fallback
        reason = "infeasible" if status == "infeasible" else "timeout"
        warnings.append(
            f"Floor {floor_num}: CP-SAT {reason} — using greedy fallback."
        )
        log.warning("Floor %d: falling back to greedy (%s)", floor_num, status)
        placements = self._greedy_placer.place(
            rooms, net_w, net_l, grid, adj_graph, staircase_anchor
        )
        return placements, "greedy", "greedy"

    # ------------------------------------------------------------------ #
    #  Staircase alignment                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _find_staircase_anchor(
        rooms:      List[EnrichedRoom],
        placements: List[RoomPlacement],
    ) -> Optional[Tuple[float, float, float, float]]:
        stair_ids = {r.room_id for r in rooms if r.room_type == "staircase"}
        for p in placements:
            if p.room_id in stair_ids:
                return (p.x_ft, p.y_ft, p.width_ft, p.length_ft)
        return None

    # ------------------------------------------------------------------ #
    #  Build PlacedRoom objects                                            #
    # ------------------------------------------------------------------ #

    def _build_placed_rooms(
        self,
        enriched_rooms: List[EnrichedRoom],
        placements:     List[RoomPlacement],
        floor_num:      int,
        grid:           PlotGrid,
        adj_graph:      Dict,
        warnings:       List[str],
    ) -> List[PlacedRoom]:
        enrich_map: Dict[str, EnrichedRoom] = {r.room_id: r for r in enriched_rooms}
        placed_map: Dict[str, PlacedRoom]   = {}

        for p in placements:
            er = enrich_map.get(p.room_id)
            if not er:
                continue

            area       = round(p.width_ft * p.length_ft, 2)
            zone_score = grid.score_placement(
                p.x_ft, p.y_ft, p.width_ft, p.length_ft,
                er.preferred_zone, er.preferred_direction,
            )

            placed_map[p.room_id] = PlacedRoom(
                room_id             = p.room_id,
                room_type           = er.room_type,
                display_name        = er.display_name,
                floor               = floor_num,
                implicit_room       = er.implicit_room,
                x_ft                = round(p.x_ft,      2),
                y_ft                = round(p.y_ft,      2),
                width_ft            = round(p.width_ft,  2),
                length_ft           = round(p.length_ft, 2),
                area_sqft           = area,
                preferred_direction = er.preferred_direction,
                preferred_zone      = er.preferred_zone,
                zone_score          = zone_score,
                adjacency_score     = 0.0,
            )

        placed_rooms = list(placed_map.values())
        for room in placed_rooms:
            room.adjacency_score = self._room_adjacency_score(
                room, placed_map, adj_graph
            )

        # Warn about unplaced rooms
        placed_ids = {p.room_id for p in placements}
        for er in enriched_rooms:
            if er.room_id not in placed_ids:
                warnings.append(
                    f"Floor {floor_num}: '{er.display_name}' could not be placed."
                )

        return placed_rooms

    def _place_external_rooms(
        self,
        external_rooms: List[EnrichedRoom],
        floor_num:      int,
        plan:           EnrichedPlan,
        grid:           PlotGrid,
    ) -> List[PlacedRoom]:
        placed: List[PlacedRoom] = []
        if floor_num != 0:
            return placed

        front_sb = plan.setbacks.front or 5.0
        net_w    = plan.net_buildable_width_ft
        x_cursor = 0.0

        for er in external_rooms:
            w = min(er.target_width_ft,  net_w)
            l = min(er.target_length_ft, front_sb - 0.5)
            l = max(l, er.min_length_ft)

            placed.append(PlacedRoom(
                room_id             = er.room_id,
                room_type           = er.room_type,
                display_name        = er.display_name,
                floor               = floor_num,
                implicit_room       = er.implicit_room,
                x_ft                = round(x_cursor, 2),
                y_ft                = round(-front_sb, 2),
                width_ft            = round(w, 2),
                length_ft           = round(l, 2),
                area_sqft           = round(w * l, 2),
                preferred_direction = er.preferred_direction,
                preferred_zone      = "front",
                zone_score          = 1.0,
                adjacency_score     = 0.0,
            ))
            x_cursor = min(x_cursor + w, net_w - w)

        return placed

    # ------------------------------------------------------------------ #
    #  Scoring helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _room_adjacency_score(
        room:      PlacedRoom,
        placed:    Dict[str, PlacedRoom],
        adj_graph: Dict,
    ) -> float:
        prefs = adj_graph.get(room.room_id, {})
        if not prefs:
            return 1.0
        total_weight = sum(prefs.values())
        if total_weight <= 0:
            return 1.0
        satisfied = 0.0
        for partner_id, weight in prefs.items():
            if partner_id not in placed:
                continue
            if _rooms_share_edge(room, placed[partner_id], WALL_TOL_FT):
                satisfied += weight
        return round(satisfied / total_weight, 3)

    @staticmethod
    def _floor_adjacency_score(
        rooms: List[PlacedRoom], adj_graph: Dict
    ) -> float:
        if not rooms:
            return 0.0
        return round(sum(r.adjacency_score for r in rooms) / len(rooms), 3)

    @staticmethod
    def _floor_zone_score(rooms: List[PlacedRoom]) -> float:
        if not rooms:
            return 0.0
        return round(sum(r.zone_score for r in rooms) / len(rooms), 3)


# --------------------------------------------------------------------------- #
#  Module-level geometry helper                                                #
# --------------------------------------------------------------------------- #

def _rooms_share_edge(
    a:   PlacedRoom,
    b:   PlacedRoom,
    tol: float = WALL_TOL_FT,
) -> bool:
    """True if rooms a and b share a wall edge within tolerance."""
    h_adj = (
        abs((a.x_ft + a.width_ft) - b.x_ft) < tol or
        abs((b.x_ft + b.width_ft) - a.x_ft) < tol
    )
    y_overlap = (a.y_ft < b.y_ft + b.length_ft) and (b.y_ft < a.y_ft + a.length_ft)

    v_adj = (
        abs((a.y_ft + a.length_ft) - b.y_ft) < tol or
        abs((b.y_ft + b.length_ft) - a.y_ft) < tol
    )
    x_overlap = (a.x_ft < b.x_ft + b.width_ft) and (b.x_ft < a.x_ft + a.width_ft)

    return (h_adj and y_overlap) or (v_adj and x_overlap)
