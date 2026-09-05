"""
proposer.py — Placer v3 behind the engine's Tier-2 contract.

Drop-in for `engine.fallbacks.PriorProposer`: same
`propose(request, variant) -> LayoutProposal`. Wiring it into the engine is
one constructor argument — which was already true in v2, and is worth
restating: integration was never the blocker, a model that earns the slot is.

Below tau (or with no weights loaded) it defers to the statistical proposer
and says why in the log, so a weak model degrades to the current behaviour
instead of degrading the product.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

from modules.step4_generate.engine.contracts import (
    EngineRequest, LayoutProposal, Placement,
)
from modules.step4_generate.engine.fallbacks import PriorProposer
from ml.placer_v3 import decode as dec
from ml.placer_v3.features import request_to_arrays

log = logging.getLogger("PlanGen.Engine")


class PlacerV3Proposer:
    """Tier-2 proposer backed by a trained PlacerNetV3."""

    def __init__(self, net, *, fallback: Optional[PriorProposer] = None,
                 tau: float = dec.DEFAULT_TAU, temperature: float = 1.0,
                 device: str = "cpu", boundary_fn=None, vastu_fn=None):
        """`vastu_fn(request) -> {room_name: {"direction","strength"}}` feeds
        R4 conditioning. Left None until phase 01 populates Vastu on the
        RoomSpec, at which point this is the only line that changes."""
        self.net = net
        self.fallback = fallback or PriorProposer()
        self.tau = tau
        self.temperature = temperature
        self.device = device
        self.boundary_fn = boundary_fn
        self.vastu_fn = vastu_fn
        # Telemetry the merge gate reports. A deployed arm that merely ties
        # the baseline because it fell back on every brief is NOT a model
        # that learned anything, and this is what distinguishes the two.
        self.stats = {"model": 0, "fallback": 0, "conf_sum": 0.0}
        # Per-room shape hints from the last accepted proposal, keyed by room
        # name. The carver consumes these once phase 01/02 teach it to; until
        # then they are recorded and ignored, which costs nothing.
        self.last_hints: Dict[str, Dict[str, int]] = {}

    @property
    def fallback_rate(self) -> float:
        n = self.stats["model"] + self.stats["fallback"]
        return self.stats["fallback"] / n if n else 0.0

    @property
    def mean_confidence(self) -> float:
        n = self.stats["model"] + self.stats["fallback"]
        return self.stats["conf_sum"] / n if n else 0.0

    def propose(self, request: EngineRequest, variant: int = 0
                ) -> LayoutProposal:
        mask = self.boundary_fn(request) if self.boundary_fn else None
        vastu = self.vastu_fn(request) if self.vastu_fn else None
        arrays = request_to_arrays(request, mask, vastu)

        rng = np.random.default_rng(request.seed * 1_000_003 + variant)
        result = dec.sample(self.net, arrays, temperature=self.temperature,
                            rng=rng, greedy=(variant == 0),
                            device=self.device)

        self.stats["conf_sum"] += result.confidence
        if result.confidence < self.tau:
            self.stats["fallback"] += 1
            log.info("proposer=fallback reason=low_confidence "
                     "conf=%.3f<%.2f (top1=%.3f) variant=%d",
                     result.confidence, self.tau, result.top1_confidence,
                     variant)
            return self.fallback.propose(request, variant)

        self.stats["model"] += 1
        specs = arrays["specs"]
        self.last_hints = {
            specs[p.room_index].name: {
                "aspect_class": p.aspect_class,
                "orientation_class": p.orientation_class,
                "band_index": p.band_index,
            } for p in result.placements
        }
        return LayoutProposal(
            placements=[
                Placement(room=specs[p.room_index].name,
                          seed_cell=(p.row, p.col), size_class=p.size_class)
                for p in result.placements],
            source="placer-v3",
            confidence=round(result.confidence, 3))
