"""
realizer.py — Tier 3 adapters: LayoutProposal → exact GridPlan.

HubRealizer (default): seed-guided band carving with a bounded RETRY loop —
when a carve is infeasible (minimum sizes collide on tight plots), the
seeds are deterministically perturbed and carving retried instead of
discarding the candidate. Fidelity is always reported against the
ORIGINAL proposal, so retries can never inflate the number the ML gate
depends on (engine doc §8.3).

StripRealizer (legacy baseline): the v0 order-based carver, kept for
A/B comparison in the harness.

Both carve rooms ONLY — openings are added after settle by the Connector.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

from modules.step4_generate.carve.hub_carver import carve_from_proposal
from modules.step4_generate.carve.reserved import carve_with_reservation
from modules.step4_generate.carve.strip_carver import RoomBrief, carve
from modules.step4_generate.core.grid_plan import CarveError, GridPlan
from modules.step4_generate.engine.contracts import (
    SEED_GRID, SQFT_PER_SIZE_CLASS, EngineConfig, EngineRequest,
    LayoutProposal, Placement,
)


def proposal_fidelity(plan: GridPlan, proposal: LayoutProposal,
                      room_ids: Dict[str, int]) -> float:
    """1 − mean normalized distance between each room's proposed seed cell
    and its achieved centroid. Low values mean the realizer is overriding
    the proposer and model-vs-fallback A/Bs are not yet meaningful."""
    dists = []
    for p in proposal.placements:
        x0, y0, x1, y1 = plan.face_bbox(room_ids[p.room])
        got_r = (y0 + y1) / 2 / plan.h * (SEED_GRID - 1)
        got_c = (x0 + x1) / 2 / plan.w * (SEED_GRID - 1)
        d = math.hypot(got_r - p.seed_cell[0], got_c - p.seed_cell[1])
        dists.append(d / (math.sqrt(2) * (SEED_GRID - 1)))
    return max(0.0, 1.0 - sum(dists) / len(dists))


def _perturbed(proposal: LayoutProposal, attempt: int,
               seed: int) -> LayoutProposal:
    """Deterministic seed jitter for retry `attempt` (≥1). Jitter grows
    with the attempt number: small nudges first, then real shakes that
    change band membership."""
    rng = random.Random(seed * 7_919 + attempt)
    amp = attempt  # cells on the 32-grid
    placements: List[Placement] = []
    n = SEED_GRID - 1
    for p in proposal.placements:
        r, c = p.seed_cell
        r = max(0, min(n, r + rng.randint(-amp, amp)))
        c = max(0, min(n, c + rng.randint(-amp, amp)))
        placements.append(Placement(p.room, (r, c), p.size_class))
    return LayoutProposal(placements=placements,
                          source=proposal.source, confidence=proposal.confidence)


class HubRealizer:
    """Production Tier 3: seed-guided band carving + bounded retry +
    OTS shaft insertion.

    OTS (open-to-sky) loop: after a successful carve, rooms needing light
    (habitable/wet) with NO exterior wall are detected; an OTS shaft is
    added to the program seeded AT the landlocked room's centroid (so band
    packing lands it adjacent) and the plot is re-carved. Bounded at 2
    shafts — the row-house device from the reference plan, applied only
    when geometry demands it."""

    MAX_OTS = 2
    _NEEDS_LIGHT = {"living_room", "drawing_room", "dining_room", "kitchen",
                    "bedroom", "master_bedroom", "bathroom", "toilet"}

    def __init__(self, config: Optional[EngineConfig] = None,
                 reservation: Optional[Tuple[str, Tuple[int, int, int, int]]]
                 = None):
        """`reservation` = (room name, exact cell rect) pins one room's
        footprint before anything else is placed — how an upper floor keeps
        its staircase over the flight below (carve/reserved.py). None on a
        ground floor, which is every single-floor request."""
        self.config = config or EngineConfig()
        self.reservation = reservation

    def realize(self, request: EngineRequest, proposal: LayoutProposal
                ) -> Tuple[GridPlan, Dict[str, int], float, List[str]]:
        import dataclasses

        ots_specs: List = []
        ots_places: List[Placement] = []
        notes: List[str] = []

        for ots_round in range(self.MAX_OTS + 1):
            aug_request = dataclasses.replace(
                request, rooms=list(request.rooms) + ots_specs)
            aug_proposal = LayoutProposal(
                placements=list(proposal.placements) + ots_places,
                source=proposal.source, confidence=proposal.confidence)

            plan, all_ids, retry_note = self._carve_with_retry(
                aug_request, aug_proposal)

            landlocked = self._landlocked(plan, request, all_ids)
            if not landlocked or len(ots_specs) >= self.MAX_OTS:
                room_ids = {s.name: all_ids[s.name] for s in request.rooms}
                fidelity = proposal_fidelity(plan, proposal, room_ids)
                if retry_note:
                    notes.append(retry_note)
                if ots_specs:
                    notes.append(f"OTS shafts inserted: {len(ots_specs)}")
                if landlocked:
                    notes.append("landlocked despite OTS: "
                                 + ", ".join(landlocked))
                return plan, room_ids, fidelity, notes

            host = landlocked[0]
            n = SEED_GRID - 1
            x0, y0, x1, y1 = plan.face_bbox(all_ids[host])
            seed = (max(0, min(n, round((y0 + y1) / 2 / plan.h * n))),
                    max(0, min(n, round((x0 + x1) / 2 / plan.w * n))))
            from modules.step4_generate.engine.contracts import RoomSpec
            host_zone = next(s.zone for s in request.rooms if s.name == host)
            ots_specs.append(RoomSpec(f"OTS {len(ots_specs) + 1}", "ots",
                                      14.0, zone=host_zone))
            ots_places.append(Placement(ots_specs[-1].name, seed, 1))

        raise CarveError("OTS insertion loop exceeded bounds")   # unreachable

    def _carve_with_retry(self, request: EngineRequest,
                          proposal: LayoutProposal
                          ) -> Tuple[GridPlan, Dict[str, int], str]:
        last_err: Optional[CarveError] = None
        for attempt in range(self.config.realize_attempts):
            trial = proposal if attempt == 0 else \
                _perturbed(proposal, attempt, request.seed)
            plan = GridPlan.from_feet(request.plot_w_ft, request.plot_h_ft)
            try:
                if self.reservation is not None:
                    name, rect = self.reservation
                    room_ids = carve_with_reservation(
                        plan, request, trial, reserved_room=name, rect=rect,
                        config=self.config)
                else:
                    room_ids = carve_from_proposal(plan, request, trial,
                                                   self.config)
            except CarveError as e:
                last_err = e
                continue
            note = ("" if attempt == 0 else
                    f"realized on retry {attempt} (seeds perturbed)")
            return plan, room_ids, note
        raise CarveError(
            f"carve failed after {self.config.realize_attempts} attempts: "
            f"{last_err}")

    def _landlocked(self, plan: GridPlan, request: EngineRequest,
                    room_ids: Dict[str, int]) -> List[str]:
        """Requested rooms needing light that touch neither an exterior
        wall (run ≥ 2') nor an existing OTS shaft."""
        ots_ids = {rid for rid, room in plan.rooms.items()
                   if room.rtype == "ots"}
        ots_adjacent: set = set()
        if ots_ids:
            for sw in plan.shared_walls():
                if sw.room_a in ots_ids and sw.length >= 16:
                    ots_adjacent.add(sw.room_b)
                elif sw.room_b in ots_ids and sw.length >= 16:
                    ots_adjacent.add(sw.room_a)

        out = []
        for spec in request.rooms:
            if spec.rtype not in self._NEEDS_LIGHT:
                continue
            rid = room_ids[spec.name]
            if rid in ots_adjacent:
                continue
            has_ext = any(
                hi - lo >= 16
                for side in ("N", "E", "S", "W")
                for lo, hi in plan.exterior_runs(rid, side))
            if not has_ext:
                out.append(spec.name)
        return out


class StripRealizer:
    """Legacy v0 realizer — baseline for harness A/Bs."""

    def realize(self, request: EngineRequest, proposal: LayoutProposal
                ) -> Tuple[GridPlan, Dict[str, int], float, List[str]]:
        specs = {s.name: s for s in request.rooms}
        briefs = [
            RoomBrief(p.room, specs[p.room].rtype,
                      p.size_class * SQFT_PER_SIZE_CLASS)
            for p in proposal.placements
        ]
        plan = GridPlan.from_feet(request.plot_w_ft, request.plot_h_ft)
        ids = carve(plan, 1, briefs)
        room_ids = {plan.rooms[i].name: i for i in ids}
        fidelity = proposal_fidelity(plan, proposal, room_ids)
        return plan, room_ids, fidelity, []
