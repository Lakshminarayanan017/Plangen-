"""
orchestrator.py — the conductor of the layout engine.

Pipeline per candidate:
    propose → realize (carve) → settle (calibrate) → fit stairs → connect
    (openness gradient + reachability) → validate → [repair → re-validate]*
    → rank

Production behaviors:
  • request validation up front (bad briefs fail fast with named problems)
  • per-stage wall-clock timings on every candidate
  • bounded repair loop for hard violations (config.max_repair_rounds)
  • duplicate-candidate collapse (identical grids waste ranking slots)
  • per-stage failure attribution and structured tier notes
  • logging via the standard `logging` module (PlanGen.Engine)

Tiers are plug-ins behind the contracts; swapping the fallback proposer for
the trained Tier-2 placer changes ONE constructor argument.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Set

from dataclasses import replace as _dc_replace
from typing import List

from modules.step4_generate.engine import repair as repair_mod
from modules.step4_generate.engine.connector import RuleConnector
from modules.step4_generate.engine.contracts import (
    Candidate, EngineConfig, EngineRequest, EngineResult,
)
from modules.step4_generate.engine.fallbacks import PriorProposer
from modules.step4_generate.engine.realizer import HubRealizer
from modules.step4_generate.engine.program import inject_implicit_rooms
from modules.step4_generate.engine.rules.base import PARKING
from modules.step4_generate.engine.settle import SqueezeSettler
from modules.step4_generate.engine.site import core_request, site_for
from modules.step4_generate.engine.stairs import StairFitter
from modules.step4_generate.engine.validator import BasicValidator

log = logging.getLogger("PlanGen.Engine")


def blended_score(soft_score: float, critic_score: Optional[float],
                  weight: float) -> float:
    """The ONE definition of how the rules and the learned critic combine
    (implementation_plan_v2.md §4.3):

        rank = (1 - w) * soft_score + w * critic_score

    Both terms live on the same 0-100 scale, so `w` means what it says.
    w = 0 (or no critic) leaves ranking exactly as the rules left it. Lives
    here, in the engine, so the engine never has to import the critic."""
    if critic_score is None or weight <= 0.0:
        return soft_score
    w = min(1.0, weight)
    return (1.0 - w) * soft_score + w * critic_score


class Orchestrator:
    def __init__(self, proposer=None, realizer=None, settler=None,
                 connector=None, validator=None, critic=None,
                 stair_fitter=None, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.proposer = proposer or PriorProposer()
        self.realizer = realizer or HubRealizer(self.config)
        self.settler = settler or SqueezeSettler(self.config)
        self.stair_fitter = stair_fitter or StairFitter(self.config)
        self.connector = connector or RuleConnector(self.config)
        self.validator = validator or BasicValidator(self.config)
        self.critic = critic            # Tier 5 slot; None until E6

    # ── the loop ─────────────────────────────────────────────────────────
    def generate(self, request: EngineRequest) -> EngineResult:
        warnings = request.validate()
        for w in warnings:
            log.warning("request %s: %s", request.name or "<unnamed>", w)

        # implicit program (a multi-floor request implies a staircase) is
        # resolved ONCE, before any tier sees the request, so proposer,
        # carver, settler and reviewer all work on the same room list
        # irregular plot -> build inside its largest buildable rectangle;
        # the engine below this line only ever sees a rectangle
        site = site_for(request)
        if site is not None:
            request = core_request(request, site)
            warnings = list(warnings) + [site.describe()]
            log.info("site: %s", site.describe())

        request, implicit = inject_implicit_rooms(
            request, floor_height_ft=self.config.floor_height_ft)
        if implicit:
            request.validate()          # the injected program must still fit
            warnings = list(warnings) + implicit
            for note in implicit:
                log.info("request %s: %s", request.name or "<unnamed>", note)

        result = self._generate_once(request, warnings)

        if (not result.ranked and not self.config.parking_strict
                and any(spec.rtype in PARKING for spec in request.rooms)):
            # relaxed mode: this plot could not gate parking on ANY seed —
            # retry once with parking dropped rather than return nothing.
            # A geometry-level "undo the carve" isn't attempted: GridPlan
            # has no room-merge primitive, and hacking one into the repair
            # loop risks the structural invariants the engine treats as
            # sacred — a full regeneration on a smaller, valid program is
            # the safe way to honor "warn and deliver" over "fail outright".
            dropped = [s.name for s in request.rooms if s.rtype in PARKING]
            stripped = _dc_replace(
                request,
                rooms=[s for s in request.rooms if s.rtype not in PARKING])
            retry = self._generate_once(stripped, warnings)
            if retry.ranked:
                retry.warnings = list(retry.warnings) + [
                    f"parking dropped (relaxed mode): no candidate could "
                    f"give {', '.join(dropped)} a vehicle gate onto any "
                    f"exterior wall of this plot."]
                return retry
            # neither attempt produced a plan — return the original
            # (parking-included) result so its discard reasons stay visible

        return result

    def _generate_once(self, request: EngineRequest,
                       warnings: List[str]) -> EngineResult:
        kept, discarded = [], []
        seen_grids: Set[bytes] = set()
        dedup_hits = 0

        # adaptive budget: when candidates die or duplicate, keep generating
        # fresh variants (up to 2k) so the user still gets k real choices
        budget = request.k * 2 if self.config.top_up_candidates else request.k
        for variant in range(budget):
            if len(kept) >= request.k:
                break
            cand = self._build_candidate(request, variant)
            if cand.discarded:
                discarded.append(cand)
                continue
            if self.config.dedup_candidates:
                key = cand.plan.grid.tobytes()
                if key in seen_grids:
                    dedup_hits += 1
                    continue
                seen_grids.add(key)
            kept.append(cand)

        kept.sort(key=self._rank_key, reverse=True)

        mean_fid = (sum(c.fidelity for c in kept) / len(kept)) if kept else 0.0
        total_ms = sum(sum(c.timings_ms.values()) for c in kept + discarded)
        return EngineResult(
            ranked=kept,
            discarded=discarded,
            warnings=list(warnings),
            tier_notes={
                "proposer": self.proposer.__class__.__name__,
                "realizer": self.realizer.__class__.__name__,
                "settler": self.settler.__class__.__name__,
                "mean_fidelity": f"{mean_fid:.2f}",
                "kept/discarded/dup": f"{len(kept)}/{len(discarded)}"
                                      f"/{dedup_hits}",
                "total_ms": f"{total_ms:.0f}",
            },
        )

    # ── one candidate ────────────────────────────────────────────────────
    def _build_candidate(self, request: EngineRequest,
                         variant: int) -> Candidate:
        timings = {}
        stage = "propose"
        try:
            t0 = time.perf_counter()
            proposal = self.proposer.propose(request, variant=variant)
            timings["propose"] = (time.perf_counter() - t0) * 1000

            stage = "realize"
            t0 = time.perf_counter()
            plan, room_ids, fidelity, notes = \
                self.realizer.realize(request, proposal)
            timings["realize"] = (time.perf_counter() - t0) * 1000

            stage = "settle"
            t0 = time.perf_counter()
            settle_err = self.settler.settle(plan, request, room_ids)
            timings["settle"] = (time.perf_counter() - t0) * 1000

            # stairs are fitted AFTER settle (settle is what finally fixes a
            # face's dimensions) and BEFORE connect (which must treat the
            # stair as a real, enterable room)
            stage = "stairs"
            t0 = time.perf_counter()
            notes += self.stair_fitter.fit(plan, request, room_ids)
            timings["stairs"] = (time.perf_counter() - t0) * 1000

            stage = "connect"
            t0 = time.perf_counter()
            notes += self.connector.connect(plan, request, room_ids)
            timings["connect"] = (time.perf_counter() - t0) * 1000

            stage = "validate"
            t0 = time.perf_counter()
            verdict = self.validator.check(plan, request, room_ids)
            rounds = 0
            while verdict.hard and rounds < self.config.max_repair_rounds:
                fixes = repair_mod.attempt(plan, request, room_ids,
                                           verdict.hard)
                if not fixes:
                    break
                rounds += 1
                notes += fixes
                verdict = self.validator.check(plan, request, room_ids)
            timings["validate"] = (time.perf_counter() - t0) * 1000

            cand = Candidate(plan=plan, proposal=proposal, verdict=verdict,
                             request=request, room_ids=room_ids,
                             fidelity=fidelity, notes=notes,
                             repair_rounds=rounds, timings_ms=timings)
            cand.notes.append(f"settle_error={settle_err:.3f}")
            if verdict.hard:
                cand.discarded = True
                cand.notes.extend(f"[hard] {h}" for h in verdict.hard)
            return cand

        except Exception as e:
            log.debug("candidate %d failed at %s: %s", variant, stage, e)
            return Candidate(plan=None, proposal=None, discarded=True,
                             timings_ms=timings,
                             notes=[f"{stage} failed: {e}"])

    def _rank_key(self, cand: Candidate) -> float:
        """Rules first, critic second. A critic failure never costs a
        candidate its place — it is an opinion on ordering, not a gate."""
        score = cand.verdict.soft_score
        if self.critic is None or self.config.critic_weight <= 0:
            return score
        try:
            critic_score = self.critic.score_candidate(cand)
        except Exception as exc:
            log.warning("critic failed on candidate (%s); "
                        "ranking by rules alone", exc)
            return score
        cand.notes.append(f"critic={critic_score:.1f}")
        return blended_score(score, critic_score, self.config.critic_weight)
