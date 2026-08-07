"""
hub_carver.py — Tier 3 core: seed-guided band carving (milestone E2+).

Turns a LayoutProposal (seed cells + size classes) into an exact partition
that HONORS the proposal: rooms are packed into depth bands measured from
the entrance, ordered within each band by their proposed across-position.
Because the proposal drives band membership and ordering, proposal fidelity
is structural, not accidental — the number the engine telemetry watches.

Quality mechanics beyond plain banding:
  • area budgets are clamped to NBC minimums (carve/standards.py) so tight
    programs compress the big rooms, never below-code the small ones
  • band depth adapts to plot proportions (wide plots get deeper bands, so
    large rooms grow deep instead of turning into wide slivers)
  • COLUMN STACKING: two small rooms (bath + passage, bath + store …)
    share one column, split front/back — exactly how real plans absorb
    service rooms without slivering a band

The carved plan carries NO openings; the settle optimizer runs first, then
connection typing (carve/connections.py) creates the openness gradient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from modules.step4_generate.carve.standards import clamp_to_minimums, nbc_min_area_cells, \
    type_min_long, type_min_side
from modules.step4_generate.core import units
from modules.step4_generate.core.grid_plan import INT_WALL_CELLS, CarveError, GridPlan
from modules.step4_generate.engine.contracts import (
    SEED_GRID, SQFT_PER_SIZE_CLASS, EngineConfig, EngineRequest,
    LayoutProposal, RoomSpec,
)

# Fraction of interior area assumed consumed by internal walls when scaling
# room targets to the plot. Settle trues this up afterwards.
WALL_ALLOWANCE = 0.90

# A room whose stand-alone column would be narrower than this fraction of
# the band depth is a stacking candidate (it would come out tall+narrow).
_STACK_WIDTH_RATIO = 0.55


@dataclass
class _RoomPlan:
    name: str
    rtype: str
    depth: float          # 0 = at entrance wall, 1 = rear wall
    across: float         # 0 = left/top edge, 1 = right/bottom edge
    area_cells: int
    min_side: int
    # Minimum LONG side (0 = none). A staircase needs its flight to fit:
    # the band it lands in is forced at least this deep, and it is barred
    # from column stacking (which would halve exactly that dimension).
    min_long: int = 0


def _seed_to_depth_across(cell: Tuple[int, int], side: str
                          ) -> Tuple[float, float]:
    r, c = cell
    n = SEED_GRID - 1
    if side == "S":
        return (n - r) / n, c / n
    if side == "N":
        return r / n, c / n
    if side == "E":
        return (n - c) / n, r / n
    if side == "W":
        return c / n, r / n
    raise ValueError(f"bad entrance side {side!r}")


def _distribute(total: int, weights: List[float], minimums: List[int],
                snap: bool = True) -> List[int]:
    """Split `total` cells into parts ∝ weights, honoring per-part minimums.
    Raises CarveError when the minimums alone don't fit."""
    if sum(minimums) > total:
        raise CarveError(
            f"minimum sizes ({sum(minimums)} cells) exceed available "
            f"span ({total} cells)"
        )
    wsum = sum(weights)
    parts = [max(m, int(total * w / wsum)) for w, m in zip(weights, minimums)]

    def slack(i):
        return parts[i] - minimums[i]

    excess = sum(parts) - total
    while excess > 0:
        i = max(range(len(parts)), key=slack)
        take = min(excess, slack(i))
        if take == 0:
            raise CarveError("cannot satisfy minimum sizes")
        parts[i] -= take
        excess -= take
    if excess < 0:
        parts[parts.index(max(parts))] += -excess
    if snap:
        for i in range(len(parts) - 1):
            snapped = units.snap_to_module(parts[i])
            diff = parts[i] - snapped
            # the last part absorbs snap residue — never below ITS minimum
            if snapped >= minimums[i] and parts[-1] + diff >= minimums[-1]:
                parts[-1] += diff
                parts[i] = snapped
    return parts


def _cut_slab(plan: GridPlan, face: int, side: str, depth: int,
              min_side: int) -> Tuple[int, int]:
    """Cut a slab of `depth` cells off the entrance side of `face`.
    Returns (slab_id, remainder_id)."""
    x0, y0, x1, y1 = plan.face_bbox(face)
    if side == "S":
        new = plan.split(face, "h", y1 - depth - INT_WALL_CELLS,
                         min_side=min_side)
        return new, face
    if side == "N":
        new = plan.split(face, "h", y0 + depth, min_side=min_side)
        return face, new
    if side == "E":
        new = plan.split(face, "v", x1 - depth - INT_WALL_CELLS,
                         min_side=min_side)
        return new, face
    if side == "W":
        new = plan.split(face, "v", x0 + depth, min_side=min_side)
        return face, new
    raise ValueError(side)


def carve_from_proposal(plan: GridPlan, request: EngineRequest,
                        proposal: LayoutProposal,
                        config: Optional[EngineConfig] = None
                        ) -> Dict[str, int]:
    """Carve the plot per the proposal. Returns room name → face id.

    `config` selects the band-packing strategy (see EngineConfig.cpsat_mode);
    omitted, the pure-python greedy packing is used exactly as before."""
    side = request.entrance_side
    specs = {s.name: s for s in request.rooms}

    # ── Room plans: plot-scaled areas with an NBC floor ──────────────────
    budgets = scaled_budgets(proposal.placements, specs,
                             plan.face_area_cells(1) * WALL_ALLOWANCE)
    rooms = room_plans(proposal.placements, specs, budgets, side)

    ids: Dict[str, int] = {}
    carve_region(plan, 1, rooms, side, ids, config)
    plan.verify()
    return ids


def room_plans(placements, specs: Dict[str, RoomSpec],
               budgets: Dict[str, int], side: str) -> List[_RoomPlan]:
    """Placements + area budgets -> the carver's internal room records."""
    out: List[_RoomPlan] = []
    for p in placements:
        depth, across = _seed_to_depth_across(p.seed_cell, side)
        out.append(_RoomPlan(
            name=p.room,
            rtype=specs[p.room].rtype,
            depth=depth, across=across,
            area_cells=budgets[p.room],
            min_side=type_min_side(specs[p.room].rtype),
            min_long=type_min_long(specs[p.room].rtype),
        ))
    return out


def scaled_budgets(placements, specs: Dict[str, RoomSpec],
                   usable_cells: float) -> Dict[str, int]:
    """Size classes -> cell budgets scaled to the space available, with the
    NBC area floor applied (tight programs compress the big rooms, never
    below-code the small ones)."""
    raw = {p.room: p.size_class * SQFT_PER_SIZE_CLASS / units.SQFT_PER_CELL2
           for p in placements}
    scale = usable_cells / max(sum(raw.values()), 1e-9)
    scaled = {name: cells * scale for name, cells in raw.items()}
    floors = {name: float(nbc_min_area_cells(specs[name].rtype) or 0)
              for name in scaled}
    return clamp_to_minimums(scaled, floors)


def carve_region(plan: GridPlan, face: int, rooms: List[_RoomPlan],
                 side: str, ids: Dict[str, int],
                 config: Optional[EngineConfig] = None) -> None:
    """Band-and-column carve of `rooms` into ONE rectangular face.

    Factored out of carve_from_proposal so the same packing runs inside any
    rectangle — the whole plot, or a residual region left over after a
    footprint has been reserved (carve/reserved.py). A band must be deep
    enough not to read as a sliver, no band may be a lone tiny room, and on
    wide regions the minimum depth grows so large rooms come out deep
    rather than wide (the aspect fix).
    """
    if not rooms:
        return
    x0, y0, x1, y1 = plan.face_bbox(face)
    across_span = (x1 - x0) if side in ("S", "N") else (y1 - y0)
    depth_span = (y1 - y0) if side in ("S", "N") else (x1 - x0)
    max_per_band = max(2, across_span // units.cells(9))
    min_band_depth = max(units.cells(7),
                         min(units.cells(12), across_span // 5))
    rooms = sorted(rooms, key=lambda r: r.depth)       # entrance first

    bands, depths = _pack_bands(
        rooms, across_span=across_span, depth_span=depth_span,
        max_per_band=max_per_band, min_band_depth=min_band_depth,
        config=config)

    # ── Carve slabs from the entrance inward ─────────────────────────────
    for i, band in enumerate(bands):
        if i < len(bands) - 1:
            slab, face = _cut_slab(plan, face, side, depths[i],
                                   min_side=min(r.min_side for r in band))
        else:
            slab = face
        _carve_band(plan, slab, band, side, ids, config)


# ── Band packing: greedy first, solver only where greedy gives up ────────────

def _greedy_bands(rooms: List[_RoomPlan], across_span: int,
                  max_per_band: int, min_band_depth: int
                  ) -> List[List[_RoomPlan]]:
    """The original packer: walk the depth-ordered rooms, close a band once
    it is deep enough or full. Fast, order-preserving, and unable to back
    up once it has committed to a band — which is what _pack_bands' solver
    branch exists to rescue."""
    bands: List[List[_RoomPlan]] = []
    cur: List[_RoomPlan] = []
    for room in rooms:
        cur.append(room)
        band_depth = sum(r.area_cells for r in cur) / across_span
        if len(cur) >= max_per_band or (
                len(cur) >= 2 and band_depth >= min_band_depth):
            bands.append(cur)
            cur = []
    if cur:
        leftover_depth = sum(r.area_cells for r in cur) / across_span
        if bands and (leftover_depth < min_band_depth or len(cur) == 1) \
                and len(bands[-1]) + len(cur) <= max_per_band + 1:
            bands[-1].extend(cur)
        else:
            bands.append(cur)
    return bands


def _band_depths(bands: List[List[_RoomPlan]], depth_span: int
                 ) -> List[int]:
    """Band depths ∝ band area, honoring member minimums.

    A band holding a run-constrained room (a staircase) must be at least as
    deep as that run: its column can be widened later, but nothing can make
    the band deeper, so the requirement has to be met here."""
    avail_depth = depth_span - (len(bands) - 1) * INT_WALL_CELLS
    return _distribute(
        avail_depth,
        [sum(r.area_cells for r in b) for b in bands],
        [max(max(r.min_side for r in b), max(r.min_long for r in b))
         for b in bands],
    )


def _solved_bands(rooms: List[_RoomPlan], across_span: int, depth_span: int,
                  max_per_band: int, min_band_depth: int,
                  timeout_ms: int) -> Optional[Tuple[List[List[_RoomPlan]],
                                                     List[int]]]:
    """CP-SAT band assignment (engine.cpsat.bands). None if the solver is
    absent, the program is provably unpackable, or the solve times out."""
    from modules.step4_generate.engine.cpsat import available, exact_bands
    from modules.step4_generate.engine.cpsat.bands import BandRoom
    if not available():
        return None
    solution = exact_bands(
        [BandRoom(r.area_cells, r.min_side, r.min_long) for r in rooms],
        across_span=across_span, depth_span=depth_span, wall=INT_WALL_CELLS,
        max_per_band=max_per_band, min_band_depth=min_band_depth,
        timeout_ms=timeout_ms)
    if solution is None:
        return None
    bands = [[rooms[i] for i in group] for group in solution.bands]
    return bands, solution.depths


def _pack_bands(rooms: List[_RoomPlan], *, across_span: int, depth_span: int,
                max_per_band: int, min_band_depth: int,
                config: Optional[EngineConfig]
                ) -> Tuple[List[List[_RoomPlan]], List[int]]:
    """Assign depth-ordered rooms to bands and size every band.

    Strategy per EngineConfig.cpsat_mode. "repair" — the default — runs the
    greedy packer and only calls the solver when the greedy depths cannot be
    distributed at all, so the solver can turn a NO-PLAN into a plan but can
    never change a layout the greedy path already handled."""
    mode = (config.cpsat_mode if config else "off")
    timeout = config.cpsat_timeout_ms if config else 2000

    if mode == "always":
        solved = _solved_bands(rooms, across_span, depth_span, max_per_band,
                               min_band_depth, timeout)
        if solved is not None:
            return solved

    bands = _greedy_bands(rooms, across_span, max_per_band, min_band_depth)
    try:
        return bands, _band_depths(bands, depth_span)
    except CarveError:
        if mode == "off":
            raise
        solved = _solved_bands(rooms, across_span, depth_span, max_per_band,
                               min_band_depth, timeout)
        if solved is None:
            raise
        return solved


# ── Within-band carving with column stacking ─────────────────────────────────

def _columns(band: List[_RoomPlan], depth: int) -> List[List[_RoomPlan]]:
    """Group across-sorted band members into columns. Two neighbors whose
    stand-alone widths would be tall slivers share one stacked column."""
    band = sorted(band, key=lambda r: r.across)
    cols: List[List[_RoomPlan]] = []
    i = 0
    while i < len(band):
        room = band[i]
        if i + 1 < len(band) and not room.min_long and not band[i + 1].min_long:
            nxt = band[i + 1]
            w_room = room.area_cells / depth
            w_nxt = nxt.area_cells / depth
            col_w = (room.area_cells + nxt.area_cells) / depth
            pieces_ok = all(
                (r.area_cells / max(col_w, 1)) >= r.min_side
                for r in (room, nxt))
            if (w_room < _STACK_WIDTH_RATIO * depth
                    and w_nxt < _STACK_WIDTH_RATIO * depth
                    and col_w >= max(room.min_side, nxt.min_side)
                    and pieces_ok):
                cols.append([room, nxt])
                i += 2
                continue
        cols.append([room])
        i += 1
    return cols


def _split_span(total: int, weights: List[float], minimums: List[int],
                config: Optional[EngineConfig], *, snap: bool = True
                ) -> List[int]:
    """`_distribute` with the same repair policy as band packing: when the
    greedy proportional split cannot satisfy the minimums, the solver gets
    one bounded attempt at an exact allocation before the carve is
    abandoned (EngineConfig.cpsat_mode)."""
    mode = (config.cpsat_mode if config else "off")
    timeout = config.cpsat_timeout_ms if config else 2000
    if mode == "always":
        exact = _exact_split(total, weights, minimums, snap, timeout)
        if exact is not None:
            return exact
    try:
        return _distribute(total, weights, minimums, snap=snap)
    except CarveError:
        if mode == "off":
            raise
        exact = _exact_split(total, weights, minimums, snap, timeout)
        if exact is None:
            raise
        return exact


def _exact_split(total: int, weights: List[float], minimums: List[int],
                 snap: bool, timeout_ms: int) -> Optional[List[int]]:
    from modules.step4_generate.engine.cpsat import exact_distribute
    return exact_distribute(total, weights, minimums, snap=snap,
                            timeout_ms=timeout_ms)


def _carve_band(plan: GridPlan, slab: int, band: List[_RoomPlan],
                side: str, ids: Dict[str, int],
                config: Optional[EngineConfig] = None) -> None:
    if len(band) == 1:
        plan.rename(slab, band[0].name, band[0].rtype)
        ids[band[0].name] = slab
        return

    x0, y0, x1, y1 = plan.face_bbox(slab)
    across_axis = "v" if side in ("S", "N") else "h"
    span = (x1 - x0) if across_axis == "v" else (y1 - y0)
    depth = (y1 - y0) if across_axis == "v" else (x1 - x0)

    cols = _columns(band, depth)
    widths = _split_span(
        span - (len(cols) - 1) * INT_WALL_CELLS,
        [sum(r.area_cells for r in col) for col in cols],
        [max(r.min_side for r in col) for col in cols],
        config,
    )

    face = slab
    for i, col in enumerate(cols):
        if i < len(cols) - 1:
            fx0, fy0, fx1, fy1 = plan.face_bbox(face)
            lo = fx0 if across_axis == "v" else fy0
            rest_min = min(max(r.min_side for r in c)
                           for c in cols[i + 1:])
            col_min = max(r.min_side for r in col)
            new = plan.split(face, across_axis, lo + widths[i],
                             min_side=min(col_min, rest_min))
            col_face, face = face, new
        else:
            col_face = face
        _carve_column(plan, col_face, col, side, ids, config)


def _carve_column(plan: GridPlan, face: int, col: List[_RoomPlan],
                  side: str, ids: Dict[str, int],
                  config: Optional[EngineConfig] = None) -> None:
    if len(col) == 1:
        plan.rename(face, col[0].name, col[0].rtype)
        ids[col[0].name] = face
        return

    # stacked pair: split along the band's depth axis; the deeper-seeded
    # room takes the far half
    near, far = sorted(col, key=lambda r: r.depth)
    x0, y0, x1, y1 = plan.face_bbox(face)
    depth_axis = "h" if side in ("S", "N") else "v"
    span = (y1 - y0) if depth_axis == "h" else (x1 - x0)
    parts = _split_span(
        span - INT_WALL_CELLS,
        [far.area_cells, near.area_cells],
        [far.min_side, near.min_side],
        config,
    )
    lo = y0 if depth_axis == "h" else x0
    # far side of the plot = lower coordinate for S/E entrances
    first, second = (far, near) if side in ("S", "E") else (near, far)
    first_span = parts[0] if first is far else parts[1]
    new = plan.split(face, depth_axis, lo + first_span,
                     min_side=min(first.min_side, second.min_side))
    plan.rename(face, first.name, first.rtype)
    ids[first.name] = face
    plan.rename(new, second.name, second.rtype)
    ids[second.name] = new
