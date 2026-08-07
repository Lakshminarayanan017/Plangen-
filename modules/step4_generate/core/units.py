"""
units.py — the precision substrate.

Every coordinate in PlanGen Remastered is an INTEGER number of lattice cells.
One cell = 1.5 inches. This makes all real construction dimensions exact:

    4.5"  internal partition  = 3 cells
    9"    external wall       = 6 cells
    3"    design module       = 2 cells
    1'    (12")               = 8 cells
    10'   room span           = 80 cells

No floats exist in the geometry core; floats appear only at the display /
area-reporting boundary.
"""

from __future__ import annotations

# ── Lattice constants ─────────────────────────────────────────────────────────
CELL_INCHES = 1.5          # size of one lattice cell, in inches
CELLS_PER_FOOT = 8         # 12" / 1.5"
MODULE_CELLS = 2           # 3" design module — user-facing dims snap to this

INT_WALL_CELLS = 3         # 4.5" internal partition
EXT_WALL_CELLS = 6         # 9"   external / load-bearing wall

SQFT_PER_CELL2 = (CELL_INCHES / 12.0) ** 2   # 0.015625 sqft per cell²


# ── Constructors ──────────────────────────────────────────────────────────────
def cells(feet: int, inches: float = 0.0) -> int:
    """Exact conversion of a feet+inches dimension to lattice cells.

    Raises if the dimension does not sit on the 1.5" lattice — a dimension
    that can't be built on the lattice is a spec error, not a rounding job.
    """
    total_inches = feet * 12 + inches
    raw = total_inches / CELL_INCHES
    result = round(raw)
    if abs(raw - result) > 1e-9:
        raise ValueError(
            f"{feet}'{inches}\" ({total_inches}\") is not on the 1.5\" lattice"
        )
    return result


def cells_from_inches(inches: float) -> int:
    return cells(0, inches)


def snap_to_module(n_cells: int) -> int:
    """Snap a cell count to the nearest 3" design module (ties round up —
    integer arithmetic, immune to float banker's rounding)."""
    return (n_cells + MODULE_CELLS // 2) // MODULE_CELLS * MODULE_CELLS


# ── Display ───────────────────────────────────────────────────────────────────
def to_inches(n_cells: int) -> float:
    return n_cells * CELL_INCHES


def fmt_ft_in(n_cells: int) -> str:
    """122 cells → 183.0" → 15'3"; half inches shown as .5 (e.g. 7'6.5")."""
    total_inches = to_inches(n_cells)
    feet = int(total_inches // 12)
    rem = total_inches - feet * 12
    if rem == 0:
        return f"{feet}'0\""
    if rem == int(rem):
        return f"{feet}'{int(rem)}\""
    return f"{feet}'{rem}\""


def area_sqft(cell_area: int) -> float:
    """Area of `cell_area` cells² in square feet (exact multiple of 1/64)."""
    return cell_area * SQFT_PER_CELL2


def fmt_area(cell_area: int) -> str:
    return f"{area_sqft(cell_area):.1f} sqft"
