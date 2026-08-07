"""
multifloor.py — plan a building, not a floor.

Floors are generated bottom-up, each one CONDITIONED on the one below,
because the vertical constraints are not preferences: the stair must land in
the same cells, the drainage stacks want to line up, and the partitions want
something under them. Generating floors independently and hoping they agree
produces two plans that cannot be built as one house.

The conditioning is applied where it belongs — at the PROPOSER. A
`FloorConditionedProposer` wraps any Tier-2 proposer (prior or trained) and:

  • pins the staircase seed to the stair below (hard: VRT-001 depends on it)
  • pulls wet rooms toward the wet cells below (soft, `vertical_bias`)

Everything downstream is unchanged, so the carver, settler, reviewer and the
stair fitter treat an upper floor exactly like a ground floor. After each
floor is chosen, engine/vertical.py reviews it against the floor below and
a floor whose vertical verdict has hard violations is rejected in favour of
the next-best candidate — the same "generate several, keep what survives"
discipline the single-floor engine already uses.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from modules.step4_generate.core.grid_plan import GridPlan
from modules.step4_generate.engine.contracts import (
    SEED_GRID, Candidate, EngineConfig, EngineRequest, LayoutProposal,
    Placement, RoomSpec,
)
from modules.step4_generate.engine.orchestrator import Orchestrator
from modules.step4_generate.engine.program import inject_implicit_rooms
from modules.step4_generate.engine.realizer import HubRealizer
from modules.step4_generate.engine.rules.base import KITCHEN, WET
from modules.step4_generate.engine.settle import SqueezeSettler
from modules.step4_generate.engine.vertical import (
    VerticalVerdict, check_stacking, stair_footprint, wet_footprints,
)

log = logging.getLogger("PlanGen.Engine")

_WET_LIKE = WET | KITCHEN


# ── conditioning ─────────────────────────────────────────────────────────────

@dataclass
class FloorContext:
    """What the floor below dictates to the floor above.

    The stair is a RESERVATION (an exact cell rectangle, honored by the
    carver) while the wet rooms are a BIAS (seed hints the packer may
    override). That split is deliberate: VRT-001 is a hard rule and a hint
    cannot satisfy a hard rule — measured at 0-1% footprint overlap when
    the stair was merely seeded — whereas wet stacking is a soft
    preference that must not cost the floor its layout."""
    plan: GridPlan
    room_ids: Dict[str, int]
    stair_name: Optional[str] = None
    stair_rect: Optional[Tuple[int, int, int, int]] = None
    wet_seeds: List[Tuple[int, int]] = field(default_factory=list)

    @classmethod
    def from_plan(cls, plan: GridPlan,
                  room_ids: Dict[str, int]) -> "FloorContext":
        rect = stair_footprint(plan, room_ids)
        name = next((n for n, rid in room_ids.items()
                     if plan.rooms[rid].rtype == "staircase"), None)
        return cls(plan=plan, room_ids=room_ids, stair_name=name,
                   stair_rect=rect,
                   wet_seeds=[_seed_of(plan, box)
                              for box in wet_footprints(plan, room_ids)])

    def reservation(self, request: EngineRequest
                    ) -> Optional[Tuple[str, Tuple[int, int, int, int]]]:
        """(room name, rect) for the realizer — the upper floor's own name
        for its staircase, pinned to the footprint below."""
        if self.stair_rect is None:
            return None
        upstairs = next((s.name for s in request.rooms
                         if s.rtype == "staircase"), None)
        return (upstairs, self.stair_rect) if upstairs else None


def _seed_of(plan: GridPlan,
             box: Optional[Tuple[int, int, int, int]]
             ) -> Optional[Tuple[int, int]]:
    """Face bbox -> the seed cell whose centroid it is."""
    if box is None:
        return None
    x0, y0, x1, y1 = box
    n = SEED_GRID - 1
    return (max(0, min(n, round((y0 + y1) / 2 / plan.h * n))),
            max(0, min(n, round((x0 + x1) / 2 / plan.w * n))))


class FloorConditionedProposer:
    """Wraps a proposer; rewrites the placements the floor below constrains.

    Deliberately a WRAPPER rather than a new proposer: the trained placer
    and the statistical prior both keep working, and the vertical rules
    apply to whichever one is in use."""

    def __init__(self, inner, context: FloorContext,
                 config: Optional[EngineConfig] = None):
        self.inner = inner
        self.context = context
        self.config = config or EngineConfig()

    def propose(self, request: EngineRequest, variant: int = 0
                ) -> LayoutProposal:
        proposal = self.inner.propose(request, variant=variant)
        bias = max(0.0, min(1.0, self.config.vertical_bias))
        specs = {s.name: s for s in request.rooms}
        stair_seed = _seed_of(self.context.plan, self.context.stair_rect)

        wet_pool = list(self.context.wet_seeds)
        placements: List[Placement] = []
        for p in proposal.placements:
            spec = specs.get(p.room)
            rtype = spec.rtype if spec else ""
            if rtype == "staircase" and stair_seed:
                # the realizer RESERVES the exact rect; the seed is moved
                # too so the surrounding rooms are proposed around it
                placements.append(Placement(p.room, stair_seed,
                                            p.size_class))
                continue
            if rtype in _WET_LIKE and wet_pool and bias > 0:
                target = min(wet_pool,
                             key=lambda s: (s[0] - p.seed_cell[0]) ** 2
                             + (s[1] - p.seed_cell[1]) ** 2)
                wet_pool.remove(target)
                row = round(p.seed_cell[0] + bias * (target[0] - p.seed_cell[0]))
                col = round(p.seed_cell[1] + bias * (target[1] - p.seed_cell[1]))
                n = SEED_GRID - 1
                placements.append(Placement(
                    p.room, (max(0, min(n, row)), max(0, min(n, col))),
                    p.size_class))
                continue
            placements.append(p)

        return LayoutProposal(placements=placements,
                              source=f"{proposal.source}+vertical",
                              confidence=proposal.confidence)


# ── the building ─────────────────────────────────────────────────────────────

@dataclass
class Floor:
    index: int
    request: EngineRequest
    candidate: Optional[Candidate]
    vertical: Optional[VerticalVerdict] = None
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.candidate is not None


@dataclass
class BuildingResult:
    floors: List[Floor]
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.floors) and all(f.ok for f in self.floors)

    @property
    def score(self) -> float:
        """Mean per-floor soft score, less the vertical penalties. A house
        is only as good as the way its floors agree."""
        if not self.ok:
            return 0.0
        per_floor = [f.candidate.verdict.soft_score for f in self.floors]
        vertical = [f.vertical.soft_score for f in self.floors
                    if f.vertical is not None]
        base = sum(per_floor) / len(per_floor)
        if not vertical:
            return round(base, 2)
        # vertical agreement scales the floors' own quality: 100 = perfect
        return round(base * (sum(vertical) / len(vertical)) / 100.0, 2)

    def summary(self) -> Dict:
        return {
            "floors": len(self.floors),
            "ok": self.ok,
            "score": self.score,
            "per_floor": [
                {"floor": f.index,
                 "score": f.candidate.verdict.soft_score if f.ok else 0.0,
                 "vertical": f.vertical.soft_score if f.vertical else None,
                 "hard": (f.vertical.hard if f.vertical else [])}
                for f in self.floors],
            "warnings": list(self.warnings),
        }


def floor_request(base: EngineRequest, floor_index: int,
                  rooms: Sequence[RoomSpec]) -> EngineRequest:
    """One floor's own request: same plot, same entrance, its own program.

    `floor_index` rides along so the contracts carry vertical context (they
    were designed to from day one) and so seeds differ per floor."""
    return dataclasses.replace(
        base, rooms=[dataclasses.replace(s, floor=floor_index)
                     for s in rooms],
        floor_index=floor_index, wishes=[],
        seed=base.seed + 1_000 * floor_index,
        name=f"{base.name}#F{floor_index}" if base.name else "")


def generate_building(request: EngineRequest,
                      floor_programs: Sequence[Sequence[RoomSpec]], *,
                      config: Optional[EngineConfig] = None,
                      orchestrator: Optional[Orchestrator] = None,
                      proposer=None) -> BuildingResult:
    """Plan every floor bottom-up.

    `floor_programs[i]` is floor i's room list (the ground floor first). The
    staircase is injected per floor by the program engine — every floor of a
    multi-floor building needs one, including the top, which is where the
    flight from below arrives.
    """
    config = config or EngineConfig()
    orch = orchestrator or Orchestrator(config=config, proposer=proposer)
    base_proposer = orch.proposer

    base_realizer = orch.realizer
    floors: List[Floor] = []
    warnings: List[str] = []
    context: Optional[FloorContext] = None

    for index, rooms in enumerate(floor_programs):
        req = floor_request(request, index, rooms)
        if context is None:
            orch.proposer, orch.realizer = base_proposer, base_realizer
        else:
            orch.proposer = FloorConditionedProposer(base_proposer, context,
                                                     config)
            # the staircase is reserved, not requested: the carver isolates
            # the exact rectangle from the floor below before placing a
            # single room (VRT-001 is hard, so it cannot rest on a hint)
            reservation = context.reservation(
                inject_implicit_rooms(
                    req, floor_height_ft=config.floor_height_ft)[0])
            orch.realizer = HubRealizer(config, reservation=reservation)
            # ... and the settler must not then optimize the reservation
            # away: without freezing it, settle chases the stair's area
            # target and the footprint overlap collapses to 23-59%
            orch.settler = SqueezeSettler(
                config, frozen_rooms=[reservation[0]] if reservation else [])
        result = orch.generate(req)

        chosen, vertical, notes = _pick_floor(result, context, config, index)
        if chosen is None:
            reasons = sorted({c.notes[-1][:90] for c in result.discarded
                              if c.notes})
            warnings.append(
                f"floor {index}: no candidate satisfied the vertical rules "
                f"({'; '.join(reasons[:2]) or 'no candidates'})")
            floors.append(Floor(index=index, request=req, candidate=None,
                                notes=notes))
            break

        floors.append(Floor(index=index, request=req, candidate=chosen,
                            vertical=vertical, notes=notes))
        context = FloorContext.from_plan(chosen.plan, chosen.room_ids)

    # leave the orchestrator exactly as we found it
    orch.proposer, orch.realizer = base_proposer, base_realizer
    return BuildingResult(floors=floors, warnings=warnings)


def _pick_floor(result, context: Optional[FloorContext],
                config: EngineConfig, index: int
                ) -> Tuple[Optional[Candidate], Optional[VerticalVerdict],
                           List[str]]:
    """Best candidate that ALSO satisfies the vertical rules.

    Vertical hard rules are a FILTER (a floor that does not meet the one
    below is not a floor); among the survivors the choice is the product of
    per-floor quality and vertical agreement, so a slightly worse floor that
    stacks its plumbing beats a slightly better one that does not. Taking
    the first clean candidate in per-floor rank order would ignore stacking
    entirely — it is a soft rule, and soft rules only matter if something
    actually compares them."""
    notes: List[str] = []
    if not result.ranked:
        return None, None, notes
    if context is None:
        return result.ranked[0], None, notes

    scored = []
    for cand in result.ranked:
        verdict = check_stacking(context.plan, context.room_ids,
                                 cand.plan, cand.room_ids, config,
                                 upper_floor=index)
        if verdict.ok:
            scored.append((cand.verdict.soft_score * verdict.soft_score / 100.0,
                           cand, verdict))
        else:
            log.debug("floor %d candidate rejected: %s", index, verdict.hard)

    if not scored:
        return None, None, notes

    rejected = len(result.ranked) - len(scored)
    if rejected:
        notes.append(f"floor {index}: skipped {rejected} candidate(s) that "
                     f"broke the vertical rules")
    scored.sort(key=lambda t: -t[0])
    _, cand, verdict = scored[0]
    notes.append(
        f"floor {index}: vertical score {verdict.soft_score} "
        f"(stair overlap {verdict.breakdown.get('stair_overlap', 0):.0%}, "
        f"wet stack {verdict.breakdown.get('wet_stack_fraction', 0):.0%}, "
        f"walls aligned {verdict.breakdown.get('wall_alignment', 0):.0%})")
    return cand, verdict, notes
