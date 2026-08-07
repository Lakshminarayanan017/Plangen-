"""
distribute.py — exact integer allocation of a span among rooms.

The greedy `carve.hub_carver._distribute` splits a span proportionally, then
claws back overshoot from whoever has the most slack. It is fast and usually
fine, and it has two failure modes this replaces:

  • it declares infeasible whenever its proportional first guess cannot be
    clawed back, even when a feasible allocation exists
  • its module snapping dumps ALL rounding residue on the last part, which
    is how a 3'0" bath ends up 3'1.5" while a bedroom absorbs nothing

The exact version minimizes total RELATIVE area error (a 6" error on a bath
matters more than 6" on a living room) subject to the minimums, the exact
span, and the 3" design module — an optimum, not a repair.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from modules.step4_generate.core import units
from modules.step4_generate.engine import cpsat

# CP-SAT objectives are integer, so the per-part weight 1/ideal has to be
# scaled up. 1e6 (not 1e3) because integer-dividing at 1e3 quantized the
# weights coarsely enough that the "optimal" split could lose to the greedy
# one on the true relative-error metric by ~0.3% — an optimizer that is
# only approximately optimal is not worth the dependency.
_SCALE = 1_000_000
_MILLI = 1000          # ideals are kept in thousandths of a cell


def exact_distribute(total: int, weights: Sequence[float],
                     minimums: Sequence[int], *,
                     snap: bool = True,
                     timeout_ms: int = 2000) -> Optional[List[int]]:
    """Split `total` cells into len(weights) integer parts.

    Returns None (never raises) when ortools is absent, the minimums alone
    exceed `total`, or the solver does not finish — the caller keeps its
    greedy result. Parts are >= their minimum, sum to exactly `total`, and
    every part except the residue-absorbing largest one sits on the 3"
    design module when `snap` is set.
    """
    n = len(weights)
    if n == 0 or n != len(minimums):
        return None
    if sum(minimums) > total:
        return None
    if not cpsat.available():
        return None
    if n == 1:
        return [total]

    cp = cpsat.cp_model()
    wsum = float(sum(weights)) or 1.0
    # Ideals are held in MILLI-CELLS. Rounding them to whole cells shifts
    # the target by up to half a cell, which is enough to move the optimum
    # of a relative-error objective — the exact split then loses to the
    # greedy one on the true metric, which defeats the point.
    ideals = [max(1, int(round(_MILLI * total * w / wsum))) for w in weights]

    model = cp.CpModel()
    parts = [model.NewIntVar(int(minimums[i]), total, f"p{i}")
             for i in range(n)]
    model.Add(sum(parts) == total)

    # module snapping: every part but the largest (which absorbs the
    # residue) lands on the 3" module, so no room carries the rounding.
    if snap and units.MODULE_CELLS > 1:
        absorber = max(range(n), key=ideals.__getitem__)
        for i in range(n):
            if i == absorber:
                continue
            if minimums[i] % units.MODULE_CELLS:
                continue        # a min off the module would be unsatisfiable
            model.AddModuloEquality(0, parts[i], units.MODULE_CELLS)

    # objective: sum of RELATIVE deviations from the proportional ideal
    terms = []
    for i in range(n):
        dev = model.NewIntVar(0, _MILLI * total, f"d{i}")
        model.AddAbsEquality(dev, _MILLI * parts[i] - ideals[i])
        terms.append(dev * max(1, round(_SCALE / ideals[i])))
    model.Minimize(sum(terms))

    solver = cp.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_ms / 1000.0
    solver.parameters.num_search_workers = 1     # determinism over speed
    status = solver.Solve(model)
    if status not in (cp.OPTIMAL, cp.FEASIBLE):
        return None
    return [int(solver.Value(p)) for p in parts]
