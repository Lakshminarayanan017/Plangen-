"""
continuity.py — "the same house, with the change I asked for".

THE PROBLEM. An edit re-runs the pipeline (see `step3_enrich/plan_edit.py`
for why it re-derives rather than mutates), and re-running produces a fresh
carve. Holding the engine's seed fixed makes the new plan *tend* to resemble
the old one, because the proposer starts from the same place — but tending is
not guaranteeing. A user who asks for a bigger kitchen and gets a different
house has been given a new plan, not an edit, and no amount of "the seed was
the same" makes that acceptable.

THE FIX. The engine already generates k candidates and keeps every one that
survives the reviewer. So instead of hoping the top-scored candidate resembles
the previous plan, RANK them by resemblance blended with quality:

    rank = (1 - w) * quality  +  w * 100 * similarity

which is the same shape the codebase already uses in two places — the learned
critic in `orchestrator.blended_score`, and vertical agreement in
`multifloor._pick_floor`. Continuity becomes a property the selection
guarantees rather than one the seed happens to deliver.

WHY DISPLACEMENT IS AREA-WEIGHTED. Moving a living room across the plot is a
different plan; moving a 40 sqft bathroom two feet is the same plan. An
unweighted mean would score those alike, so each room's contribution is
weighted by its share of the floor.

WHY THE INTERSECTION. An edit that adds a study or removes a store SHOULD
change the room set, so rooms present on only one side are not counted as
displacement. Coverage is reported separately, and a caller that cares can
read it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Share of the ranking given to resemblance. 0 = pick purely on quality (the
# behaviour before this module), 1 = pick the most similar plan no matter how
# bad.
#
# MEASURED, on a 30x45 brief re-run under four edits:
#
#     edit                                   w=0    w=0.35
#     living room much bigger                94%      94%
#     add a study room                       66%      66%
#     move master bedroom to the north-east  42%     100%
#     kitchen 200 sqft + dining smaller      40%      92%
#
# So the seed alone is enough for a simple resize and this module changes
# nothing there — it earns its place on edits that restructure the program,
# where seed-only continuity collapsed to ~40%.
#
# A weight sweep over {0.15, 0.25, 0.35, 0.5, 0.7} on those two structural
# edits reached 96% at EVERY value: the effect is a step, not a gradient,
# because the candidate that preserves the layout is generally also a good
# one. 0.35 is chosen as a mid-range default that stays robust if quality and
# familiarity ever do pull apart. (The same sweep could not measure a quality
# COST: `layout_quality_score` was saturated at its cap throughout, which is
# why the explicit floor below exists rather than trusting the blend.)
DEFAULT_WEIGHT = 0.35

# Familiarity is a tie-breaker among acceptable plans, never a reason to ship
# a bad one. A candidate more than this far below the best available is
# refused outright, whatever it preserves.
QUALITY_FLOOR_TOLERANCE = 12.0


@dataclass(frozen=True)
class RoomFootprint:
    """One room, normalized to the plot so two different lattices compare."""
    cx: float
    cy: float
    w: float
    h: float
    area_share: float


@dataclass(frozen=True)
class LayoutSignature:
    """Where the rooms sat, in plot-relative coordinates.

    Normalized on purpose: an edit may change the footprint (phase 02 sizes
    the building to the program, so adding a room can enlarge it), and a
    signature in absolute cells would then report a total redesign when
    nothing actually moved.
    """
    rooms: Dict[str, RoomFootprint] = field(default_factory=dict)

    @property
    def names(self) -> Iterable[str]:
        return self.rooms.keys()

    def __bool__(self) -> bool:
        return bool(self.rooms)


def _norm_name(name: str) -> str:
    """Room identity across a re-run.

    The engine names duplicates "Bedroom 2 (2)" when a program repeats a
    display name, and an edit can renumber them. Matching on the trimmed
    lowercase name is what survives that.
    """
    return " ".join(str(name).strip().lower().split())


def signature_of_plan(plan, room_ids: Dict[str, int]) -> LayoutSignature:
    """From a carved GridPlan (engine side)."""
    total = 0.0
    raw: Dict[str, Tuple[float, float, float, float, float]] = {}
    for name, rid in room_ids.items():
        try:
            x0, y0, x1, y1 = plan.face_bbox(rid)
        except (KeyError, ValueError):
            continue
        area = float((x1 - x0) * (y1 - y0))
        total += area
        raw[_norm_name(name)] = ((x0 + x1) / 2.0 / plan.w,
                                 (y0 + y1) / 2.0 / plan.h,
                                 (x1 - x0) / plan.w,
                                 (y1 - y0) / plan.h, area)
    if total <= 0:
        return LayoutSignature()
    return LayoutSignature(rooms={
        name: RoomFootprint(cx, cy, w, h, area / total)
        for name, (cx, cy, w, h, area) in raw.items()})


def signature_of_floor(floor) -> LayoutSignature:
    """From a LayoutFloor (API side), so a signature survives in a session
    after the GridPlan itself is gone.

    THE Y AXIS IS FLIPPED HERE, and it has to be. `LayoutPlan`'s origin is the
    SW corner (`engine_bridge._floor_from_plan` writes
    `y_ft = (plan.h - y1) / cells_per_foot`) while the grid's is NW. `compare`
    puts two signatures in ONE coordinate space and measures the distance
    between them, so signatures from the two builders MUST agree — reading
    each in "its own frame" mirrors every room about the horizontal centre
    line and reports the rooms furthest from the middle as the ones that moved
    most.

    That was a live bug: it made the API edit path prefer the candidate LEAST
    like the plan on screen, which is the exact inverse of the property this
    module exists to provide. `test_the_two_builders_agree` pins it.
    """
    width = float(getattr(floor, "net_width_ft", 0) or 0)
    height = float(getattr(floor, "net_length_ft", 0) or 0)
    rooms = list(getattr(floor, "rooms", []) or [])
    if width <= 0 or height <= 0 or not rooms:
        return LayoutSignature()
    total = sum(float(r.area_sqft) for r in rooms) or 1.0
    out: Dict[str, RoomFootprint] = {}
    for r in rooms:
        top_ft = height - (float(r.y_ft) + float(r.length_ft))
        out[_norm_name(r.display_name)] = RoomFootprint(
            cx=(float(r.x_ft) + float(r.width_ft) / 2.0) / width,
            cy=(top_ft + float(r.length_ft) / 2.0) / height,
            w=float(r.width_ft) / width,
            h=float(r.length_ft) / height,
            area_share=float(r.area_sqft) / total)
    return LayoutSignature(rooms=out)


def signature_to_dict(sig: LayoutSignature) -> Dict[str, List[float]]:
    """Serialisable form, so the signature the engine RANKED with is the one
    the session stores. Two builders producing the same frame is a property
    that has to be maintained; carrying the engine's own answer forward means
    the edit path never depends on maintaining it.
    """
    return {name: [f.cx, f.cy, f.w, f.h, f.area_share]
            for name, f in sig.rooms.items()}


def signature_from_dict(blob) -> LayoutSignature:
    if not blob:
        return LayoutSignature()
    rooms: Dict[str, RoomFootprint] = {}
    for name, vals in dict(blob).items():
        try:
            cx, cy, w, h, share = (float(v) for v in vals)
        except (TypeError, ValueError):
            continue
        rooms[_norm_name(name)] = RoomFootprint(cx, cy, w, h, share)
    return LayoutSignature(rooms=rooms)


@dataclass
class ContinuityScore:
    similarity: float          # 0..1, area-weighted positional agreement
    coverage: float            # share of the previous rooms still present
    moved: List[str] = field(default_factory=list)
    gained: List[str] = field(default_factory=list)
    lost: List[str] = field(default_factory=list)

    def describe(self) -> str:
        bits = [f"{self.similarity:.0%} of the layout held"]
        if self.moved:
            bits.append(f"moved: {', '.join(sorted(self.moved)[:3])}")
        if self.gained:
            bits.append(f"new: {', '.join(sorted(self.gained)[:3])}")
        if self.lost:
            bits.append(f"gone: {', '.join(sorted(self.lost)[:3])}")
        return " · ".join(bits)


# A room whose centroid moves less than this share of the plot diagonal has
# not meaningfully moved. ~8% of a 30x40 plot is about 4 feet.
_MOVED_THRESHOLD = 0.08


def compare(previous: LayoutSignature,
            current: LayoutSignature) -> ContinuityScore:
    """How much of `previous` survives in `current`.

    Similarity is 1 - the area-weighted mean centroid displacement over the
    rooms both plans have, with displacement measured against the plot
    diagonal so it is scale-free.
    """
    if not previous or not current:
        return ContinuityScore(similarity=0.0, coverage=0.0)

    shared = set(previous.names) & set(current.names)
    lost = sorted(set(previous.names) - shared)
    gained = sorted(set(current.names) - shared)
    coverage = len(shared) / max(len(previous.rooms), 1)
    if not shared:
        return ContinuityScore(similarity=0.0, coverage=0.0,
                               gained=gained, lost=lost)

    # weights come from the PREVIOUS plan: the question is how much of what
    # the user was looking at is still where they left it
    weight_total = sum(previous.rooms[n].area_share for n in shared) or 1.0
    displacement = 0.0
    moved: List[str] = []
    diagonal = 2.0 ** 0.5
    for name in shared:
        a, b = previous.rooms[name], current.rooms[name]
        d = ((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2) ** 0.5 / diagonal
        displacement += d * a.area_share
        if d > _MOVED_THRESHOLD:
            moved.append(name)
    similarity = max(0.0, 1.0 - displacement / weight_total * 2.0)
    return ContinuityScore(similarity=round(similarity, 4),
                           coverage=round(coverage, 4),
                           moved=sorted(moved), gained=gained, lost=lost)


def rank_candidates(candidates: Sequence, previous: LayoutSignature, *,
                    weight: float = DEFAULT_WEIGHT,
                    quality_floor: Optional[float] = None,
                    floor_tolerance: float = QUALITY_FLOOR_TOLERANCE
                    ) -> List[Tuple[object, ContinuityScore, float]]:
    """Re-rank engine candidates so an edit keeps the plan recognisable.

    Returns [(candidate, continuity, blended_rank), ...] best first. With no
    reference, or weight 0, this is the engine's own order untouched — so a
    first run is unaffected by the existence of this module.

    `quality_floor` refuses candidates below it outright: familiarity is a
    tie-breaker among acceptable plans, never a reason to ship a bad one.
    """
    # An edit must not be able to trade the plan's quality away for
    # familiarity. Relative to the best candidate available, so the bar
    # adapts to a brief the engine finds hard rather than being absolute.
    if quality_floor is None and candidates and previous and weight > 0:
        best_quality = max(
            (c.verdict.soft_score if c.verdict else 0.0) for c in candidates)
        quality_floor = best_quality - floor_tolerance

    scored: List[Tuple[object, ContinuityScore, float]] = []
    for cand in candidates:
        quality = float(cand.verdict.soft_score) if cand.verdict else 0.0
        if not previous or weight <= 0.0:
            scored.append((cand, ContinuityScore(0.0, 0.0), quality))
            continue
        if quality_floor is not None and quality < quality_floor:
            continue
        current = signature_of_plan(cand.plan, cand.room_ids)
        score = compare(previous, current)
        w = min(1.0, max(0.0, weight))
        blended = (1.0 - w) * quality + w * 100.0 * score.similarity
        scored.append((cand, score, blended))

    if not scored and candidates:
        # every candidate was below the floor — keep the best one rather
        # than return nothing, and let the caller report the drop
        best = max(candidates,
                   key=lambda c: c.verdict.soft_score if c.verdict else 0.0)
        current = signature_of_plan(best.plan, best.room_ids)
        scored = [(best, compare(previous, current),
                   float(best.verdict.soft_score) if best.verdict else 0.0)]

    scored.sort(key=lambda t: -t[2])
    return scored
