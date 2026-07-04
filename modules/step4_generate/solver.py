"""
solver.py
=========
CP-SAT based 2-D rectangle packing solver for Step 4 — Layout Generator.

Uses Google OR-Tools CP-SAT (already installed via ortools package).

Algorithm:
  For each room on a floor, create integer position + dimension variables.
  Apply:
    - Hard: AddNoOverlap2D (no rooms overlap)
    - Hard: boundary constraints (rooms stay within net buildable area)
    - Hard: minimum dimension constraints
    - Soft: zone preference (bool vars + weighted objective)
    - Soft: area target deviation (minimize deviation from target area)

Resolution: 0.5 ft (multiply ft by 2 → integer "half-feet").
Timeout: 20 seconds → falls back to GreedyPlacer if not feasible.

Output: list of (room_id, x_ft, y_ft, width_ft, length_ft) tuples.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

from models import EnrichedRoom

log = logging.getLogger("PlanGen.Solver")

# CP-SAT resolution: 1 unit = 0.5 ft (multiply all ft by 2)
_RES = 2   # units per foot

# Weight of zone compliance in objective vs area-target deviation
_ZONE_WEIGHT    = 100   # bonus per room fully in preferred zone
_AREA_DEV_SCALE = 1     # penalty per unit^2 away from target area

# Solver timeout (seconds) before declaring infeasible / timed out
CP_SAT_TIMEOUT_S = 20.0


class RoomPlacement:
    """Result for one room from the CP-SAT solver."""
    __slots__ = ("room_id", "x_ft", "y_ft", "width_ft", "length_ft")

    def __init__(
        self,
        room_id:   str,
        x_ft:      float,
        y_ft:      float,
        width_ft:  float,
        length_ft: float,
    ) -> None:
        self.room_id   = room_id
        self.x_ft      = x_ft
        self.y_ft      = y_ft
        self.width_ft  = width_ft
        self.length_ft = length_ft


class CPSATSolver:
    """
    Solves the 2-D room packing problem for one floor using CP-SAT.

    Usage:
        solver = CPSATSolver()
        placements, status = solver.solve(rooms, net_w, net_l, grid, adj_graph)
    """

    def solve(
        self,
        rooms:      List[EnrichedRoom],
        net_w:      float,
        net_l:      float,
        grid,                               # PlotGrid instance
        adj_graph:  Dict[str, Dict[str, float]],
        timeout_s:  float = CP_SAT_TIMEOUT_S,
        staircase_anchor: Optional[Tuple[float, float, float, float]] = None,
    ) -> Tuple[List[RoomPlacement], str]:
        """
        Place all rooms in `rooms` within (net_w × net_l) ft.

        Args:
            rooms:            EnrichedRoom list for ONE floor.
            net_w / net_l:    Net buildable dimensions in ft.
            grid:             PlotGrid for zone scoring.
            adj_graph:        room_id → {room_id: weight} adjacency.
            timeout_s:        CP-SAT max solve time.
            staircase_anchor: (x, y, w, l) in ft — if not None, lock the
                              staircase to this position (floor alignment).

        Returns:
            (list of RoomPlacement, status_string)
            status: "optimal" | "feasible" | "timeout" | "infeasible" | "error"
        """
        try:
            from ortools.sat.python import cp_model
        except ImportError:
            log.warning("OR-Tools not available — CP-SAT solver disabled.")
            return [], "error"

        if not rooms:
            return [], "optimal"

        t0     = time.perf_counter()
        model  = cp_model.CpModel()
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = timeout_s
        solver.parameters.num_search_workers  = 1   # deterministic

        # ── Grid units: 1 unit = 0.5 ft ──────────────────────────────────
        W = int(round(net_w  * _RES))
        L = int(round(net_l  * _RES))

        # Pre-compute staircase anchor in grid units if provided
        anc = None
        if staircase_anchor:
            ax, ay, aw, al = staircase_anchor
            anc = (int(round(ax*_RES)), int(round(ay*_RES)),
                   int(round(aw*_RES)), int(round(al*_RES)))

        # ── Variables ────────────────────────────────────────────────────
        x_vars = {}
        y_vars = {}
        w_vars = {}
        l_vars = {}
        x_ivars = []
        y_ivars = []
        room_ids = []

        for room in rooms:
            rid = room.room_id

            min_w = max(1, int(round(room.min_width_ft  * _RES)))
            min_l = max(1, int(round(room.min_length_ft * _RES)))
            tgt_w = max(min_w, int(round(room.target_width_ft  * _RES)))
            tgt_l = max(min_l, int(round(room.target_length_ft * _RES)))
            max_w = min(W, max(tgt_w + 2, int(round(room.max_area_sqft**0.5 * _RES * 1.2))))
            max_l = min(L, max(tgt_l + 2, int(round(room.max_area_sqft**0.5 * _RES * 1.2))))

            # Staircase alignment: lock to anchor position
            if room.room_type == "staircase" and anc:
                ax_, ay_, aw_, al_ = anc
                x_v = model.NewConstant(ax_)
                y_v = model.NewConstant(ay_)
                w_v = model.NewConstant(aw_)
                l_v = model.NewConstant(al_)
            else:
                x_v = model.NewIntVar(0, max(0, W - min_w), f"x_{rid}")
                y_v = model.NewIntVar(0, max(0, L - min_l), f"y_{rid}")
                w_v = model.NewIntVar(min_w, max_w, f"w_{rid}")
                l_v = model.NewIntVar(min_l, max_l, f"l_{rid}")

            x_vars[rid] = x_v
            y_vars[rid] = y_v
            w_vars[rid] = w_v
            l_vars[rid] = l_v

            # Boundary: x + w <= W, y + l <= L
            model.Add(x_v + w_v <= W)
            model.Add(y_v + l_v <= L)

            # Interval variables for NoOverlap2D
            xi = model.NewIntervalVar(x_v, w_v, model.NewIntVar(0, W, f"xe_{rid}"), f"xi_{rid}")
            yi = model.NewIntervalVar(y_v, l_v, model.NewIntVar(0, L, f"ye_{rid}"), f"yi_{rid}")

            x_ivars.append(xi)
            y_ivars.append(yi)
            room_ids.append(rid)

        # ── Hard: no overlap ─────────────────────────────────────────────
        model.AddNoOverlap2D(x_ivars, y_ivars)

        # ── Soft: zone preference ─────────────────────────────────────────
        # For each room, add a boolean "in_zone" that gets bonus in objective
        objective_terms = []

        for room in rooms:
            rid    = room.room_id
            region = grid.preferred_region(room.preferred_zone, room.preferred_direction)

            # Zone bounds in grid units
            zx_min = int(round(region.x_min * _RES))
            zx_max = int(round(region.x_max * _RES))
            zy_min = int(round(region.y_min * _RES))
            zy_max = int(round(region.y_max * _RES))

            in_zone = model.NewBoolVar(f"zone_{rid}")

            # in_zone = True iff room CENTER is inside zone bounds
            cx_var = model.NewIntVar(0, W * 2, f"cx_{rid}")
            cy_var = model.NewIntVar(0, L * 2, f"cy_{rid}")
            model.Add(cx_var * 2 == x_vars[rid] * 2 + w_vars[rid])
            model.Add(cy_var * 2 == y_vars[rid] * 2 + l_vars[rid])

            # Linearise: in_zone => cx in [zx_min, zx_max] AND cy in [zy_min, zy_max]
            # We use implications with big-M relaxation
            if zx_max > zx_min and zy_max > zy_min:
                b_cx_lo = model.NewBoolVar(f"cx_lo_{rid}")
                b_cx_hi = model.NewBoolVar(f"cx_hi_{rid}")
                b_cy_lo = model.NewBoolVar(f"cy_lo_{rid}")
                b_cy_hi = model.NewBoolVar(f"cy_hi_{rid}")

                model.Add(cx_var >= zx_min * 2).OnlyEnforceIf(b_cx_lo)
                model.Add(cx_var < zx_min * 2).OnlyEnforceIf(b_cx_lo.Not())
                model.Add(cx_var <= zx_max * 2).OnlyEnforceIf(b_cx_hi)
                model.Add(cx_var > zx_max * 2).OnlyEnforceIf(b_cx_hi.Not())
                model.Add(cy_var >= zy_min * 2).OnlyEnforceIf(b_cy_lo)
                model.Add(cy_var < zy_min * 2).OnlyEnforceIf(b_cy_lo.Not())
                model.Add(cy_var <= zy_max * 2).OnlyEnforceIf(b_cy_hi)
                model.Add(cy_var > zy_max * 2).OnlyEnforceIf(b_cy_hi.Not())

                # in_zone iff all 4 bounds satisfied
                model.AddBoolAnd([b_cx_lo, b_cx_hi, b_cy_lo, b_cy_hi]).OnlyEnforceIf(in_zone)
                model.AddBoolOr([b_cx_lo.Not(), b_cx_hi.Not(),
                                 b_cy_lo.Not(), b_cy_hi.Not()]).OnlyEnforceIf(in_zone.Not())

            # Weight by room area (larger rooms get more zone bonus)
            area_w = max(1, int(round(room.target_area_sqft / 10.0)))
            objective_terms.append(_ZONE_WEIGHT * area_w * in_zone)

        # ── Lock width+length to target values (pre-computed by enricher) ─
        # The enricher already calculated NBC-clamped optimal dimensions.
        # We fix w and l to their target values (rather than optimising them
        # with non-linear AddMultiplicationEquality which is very slow).
        for room in rooms:
            rid   = room.room_id
            tgt_w = max(int(round(room.min_width_ft  * _RES)),
                        int(round(room.target_width_ft  * _RES)))
            tgt_l = max(int(round(room.min_length_ft * _RES)),
                        int(round(room.target_length_ft * _RES)))
            # Clamp to plot bounds
            tgt_w = min(tgt_w, W)
            tgt_l = min(tgt_l, L)
            model.Add(w_vars[rid] == tgt_w)
            model.Add(l_vars[rid] == tgt_l)

        # Maximize zone compliance score
        model.Maximize(sum(objective_terms))

        # ── Solve ─────────────────────────────────────────────────────────
        status_code = solver.Solve(model)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        _STATUS_MAP = {
            0: "unknown",
            1: "model_invalid",
            2: "feasible",
            3: "infeasible",
            4: "optimal",
        }
        status_str = _STATUS_MAP.get(status_code, "unknown")

        if status_code in (2, 4):  # feasible or optimal
            placements: List[RoomPlacement] = []
            for room in rooms:
                rid = room.room_id
                x_val = solver.Value(x_vars[rid]) / _RES
                y_val = solver.Value(y_vars[rid]) / _RES
                w_val = solver.Value(w_vars[rid]) / _RES
                l_val = solver.Value(l_vars[rid]) / _RES
                placements.append(RoomPlacement(rid, x_val, y_val, w_val, l_val))

            log.info(
                "CP-SAT: %s in %.0fms — %d rooms placed",
                status_str, elapsed_ms, len(placements),
            )
            return placements, status_str

        elif status_code == 3:
            log.warning("CP-SAT: infeasible in %.0fms", elapsed_ms)
            return [], "infeasible"
        else:
            log.warning("CP-SAT: timeout/unknown after %.0fms", elapsed_ms)
            return [], "timeout"
