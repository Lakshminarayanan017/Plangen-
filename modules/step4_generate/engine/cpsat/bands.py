"""
bands.py — exact band assignment: which rooms share a depth band, and how
deep each band is.

This is the repair for the single most common way a request produces NO
plan: the greedy packer walks the depth-sorted rooms, closes a band when it
looks deep enough, and then discovers the minimum sizes do not fit the span
it just committed to. It cannot back up. A solver can — it chooses all the
band boundaries and all the band depths at once, so "no feasible packing"
becomes a proven statement instead of a greedy accident.

The proposal's depth ORDER is preserved as a hard constraint (rooms keep
their proposed front-to-back sequence), because that ordering is the
proposer's contribution and the number `fidelity` measures. The solver only
decides where the cuts fall and how deep each band is.

Returns None on any failure — the caller keeps the greedy packing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from modules.step4_generate.engine import cpsat


@dataclass(frozen=True)
class BandRoom:
    """What the assignment solver needs to know about one room."""
    area_cells: int
    min_side: int
    min_long: int = 0          # run requirement (staircase); 0 = none


@dataclass(frozen=True)
class BandSolution:
    bands: List[List[int]]     # room indices per band, entrance-side first
    depths: List[int]          # band depth in cells, same order
    waste_cells: int           # objective: the WORST band's slack (see below)


def exact_bands(rooms: Sequence[BandRoom], *,
                across_span: int, depth_span: int, wall: int,
                max_per_band: int, min_band_depth: int = 0,
                max_bands: int = 6,
                timeout_ms: int = 2000) -> Optional[BandSolution]:
    """Assign depth-ordered `rooms` to bands and size every band.

    `rooms` MUST already be sorted entrance-first — the solver keeps that
    order. Returns the packing that wastes the least band area, or None.
    """
    n = len(rooms)
    if n == 0 or not cpsat.available():
        return None
    if across_span <= 0 or depth_span <= 0:
        return None

    best: Optional[BandSolution] = None
    upper = min(max_bands, n)
    for n_bands in range(1, upper + 1):
        # every band must be non-empty and no band may exceed max_per_band
        if n_bands * max_per_band < n:
            continue
        sol = _solve_fixed(rooms, n_bands, across_span=across_span,
                           depth_span=depth_span, wall=wall,
                           max_per_band=max_per_band,
                           min_band_depth=min_band_depth,
                           timeout_ms=timeout_ms)
        # ties go to FEWER bands: the loop runs ascending and only a strict
        # improvement replaces the incumbent, so a packing that adds a band
        # without reducing the worst band's slack never wins
        if sol is not None and (best is None
                                or sol.waste_cells < best.waste_cells):
            best = sol
    return best


def _solve_fixed(rooms: Sequence[BandRoom], n_bands: int, *,
                 across_span: int, depth_span: int, wall: int,
                 max_per_band: int, min_band_depth: int,
                 timeout_ms: int) -> Optional[BandSolution]:
    cp = cpsat.cp_model()
    n = len(rooms)
    avail_depth = depth_span - (n_bands - 1) * wall
    if avail_depth <= 0:
        return None

    model = cp.CpModel()

    # band index per room, monotone and gap-free => bands are contiguous
    # slices of the depth-ordered list, and none is empty
    band = [model.NewIntVar(0, n_bands - 1, f"b{i}") for i in range(n)]
    model.Add(band[0] == 0)
    model.Add(band[n - 1] == n_bands - 1)
    for i in range(n - 1):
        model.Add(band[i] <= band[i + 1])
        model.Add(band[i + 1] - band[i] <= 1)

    # membership booleans, channeled to the band index
    inb = [[model.NewBoolVar(f"x{i}_{k}") for k in range(n_bands)]
           for i in range(n)]
    for i in range(n):
        model.AddExactlyOne(inb[i])
        for k in range(n_bands):
            model.Add(band[i] == k).OnlyEnforceIf(inb[i][k])
            model.Add(band[i] != k).OnlyEnforceIf(inb[i][k].Not())

    depth = [model.NewIntVar(1, avail_depth, f"d{k}") for k in range(n_bands)]
    model.Add(sum(depth) == avail_depth)

    waste_terms = []
    for k in range(n_bands):
        members = [inb[i][k] for i in range(n)]
        count = sum(members)
        model.Add(count >= 1)
        model.Add(count <= max_per_band)

        # every member must fit across the band: its own minimum width plus
        # the partition wall beside it
        model.Add(sum((rooms[i].min_side + wall) * inb[i][k]
                      for i in range(n)) <= across_span + wall)

        # the band must be deep enough for every member's minimum side, and
        # for any member that needs a RUN (a staircase's flight)
        for i in range(n):
            need = max(rooms[i].min_side, rooms[i].min_long, min_band_depth)
            if need > 0:
                model.Add(depth[k] >= need).OnlyEnforceIf(inb[i][k])

        # the band must hold its members' areas ...
        area = sum(rooms[i].area_cells * inb[i][k] for i in range(n))
        model.Add(depth[k] * across_span >= area)
        # ... and everything it holds beyond them is slack
        waste = model.NewIntVar(0, across_span * avail_depth, f"w{k}")
        model.Add(waste == depth[k] * across_span - area)
        waste_terms.append(waste)

    # Minimize the WORST band's slack, not the total.
    #
    # Total slack is a constant: sum_k (depth_k * across) - sum(areas) =
    # across * avail_depth - total_area, which does not depend on the
    # assignment at all. Minimizing it makes every feasible packing tie, so
    # the solver returned an arbitrary one — measured as a 16-point mean
    # drop against the greedy packer on the golden harness. The minimax
    # objective spreads the slack evenly instead, which is exactly what the
    # greedy packer's min_band_depth heuristic was approximating.
    worst = model.NewIntVar(0, across_span * avail_depth, "worst")
    model.AddMaxEquality(worst, waste_terms)
    model.Minimize(worst)

    solver = cp.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_ms / 1000.0
    solver.parameters.num_search_workers = 1      # determinism over speed
    status = solver.Solve(model)
    if status not in (cp.OPTIMAL, cp.FEASIBLE):
        return None

    bands: List[List[int]] = [[] for _ in range(n_bands)]
    for i in range(n):
        bands[int(solver.Value(band[i]))].append(i)
    depths = [int(solver.Value(d)) for d in depth]
    return BandSolution(bands=bands, depths=depths,
                        waste_cells=int(solver.ObjectiveValue()))


def solution_signature(sol: BandSolution) -> Tuple:
    """Hashable identity of a packing — used by tests to assert
    determinism across runs."""
    return (tuple(tuple(b) for b in sol.bands), tuple(sol.depths))
