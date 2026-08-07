"""
engine.cpsat — the constraint solver, used only where it wins.

CP-SAT is a SIDE TOOL here, never the layout engine. It decides nothing
about topology (that is the proposer's job) and it is never on the critical
path: every entry point returns None on unavailability, infeasibility or
timeout, and every caller keeps its greedy result when that happens. The
pure-Python paths remain the reference behavior — this only replaces a
greedy allocation with a provably optimal one where the greedy allocation is
the thing standing between a request and a plan.

Two surgical jobs (implementation_plan_v2.md §6):
  1. band/column ASSIGNMENT repair — greedy packing declared the program
     infeasible; solve rooms→(band, depth) exactly before giving up
  2. exact RE-DIMENSIONING — integer widths within one band under NBC
     minimums and area targets, instead of proportional-then-patch

ortools is an OPTIONAL dependency. `available()` is the single check; import
of this package never fails.
"""

from __future__ import annotations

from typing import Optional

_IMPORT_ERROR: Optional[str] = None

try:                                    # optional dependency, by design
    from ortools.sat.python import cp_model as _cp_model
except Exception as exc:                # pragma: no cover - env dependent
    _cp_model = None
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def available() -> bool:
    """True when a CP-SAT solver can actually be constructed."""
    return _cp_model is not None


def unavailable_reason() -> Optional[str]:
    """Why the solver is missing, for honest logging (None when present)."""
    return _IMPORT_ERROR


def cp_model():
    """The cp_model module. Callers must check `available()` first."""
    if _cp_model is None:               # pragma: no cover - env dependent
        raise RuntimeError(f"ortools unavailable ({_IMPORT_ERROR})")
    return _cp_model


from modules.step4_generate.engine.cpsat.bands import exact_bands              # noqa: E402,F401
from modules.step4_generate.engine.cpsat.distribute import exact_distribute    # noqa: E402,F401

__all__ = ["available", "unavailable_reason", "cp_model",
           "exact_bands", "exact_distribute"]
