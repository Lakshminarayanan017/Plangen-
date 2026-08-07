"""
tier2_placer.py — the trained proposer behind the LayoutProposal contract.

Drop-in for engine.fallbacks.PriorProposer: same
`propose(request, variant) -> LayoutProposal`. Runs the model with
constrained decoding (masked_decode); if the model's confidence is below τ
(or no weights are loaded), it falls back to the statistical proposer and
logs the reason — so the engine always ships a valid plan, and a weak/absent
model never degrades output. Merged into the engine only after beating the
fallback on the golden harness (the merge gate).

Backend-agnostic: takes any object exposing the decode interface
(encode/decode_step/cell_count/size_count) — TorchPlacer for dev, the NumPy
model for production.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from modules.step4_generate.engine.contracts import LayoutProposal, Placement, EngineRequest
from modules.step4_generate.engine.fallbacks import PriorProposer
from ml.tier2_placer import masked_decode as md
from ml.tier2_placer.features import request_to_arrays

log = logging.getLogger("PlanGen.Engine")

# Fallback threshold on masked_decode's calibrated confidence (mean 3x3
# neighborhood mass). A perfectly fitted model scores ~0.779 (the sigma=1
# label-smoothing ceiling), so 0.35 == "at least 45% of a perfect fit's
# certainty". Do NOT compare this against DecodeResult.top1_confidence,
# whose ceiling is 0.159 — see masked_decode.neighborhood_confidence.
DEFAULT_TAU = 0.35


class Tier2Placer:
    def __init__(self, backend, *, fallback: Optional[PriorProposer] = None,
                 tau: float = DEFAULT_TAU, temperature: float = 1.0,
                 boundary_fn=None):
        """`backend` implements encode/decode_step/cell_count/size_count.
        `boundary_fn(request) -> 64x64 mask` supplies irregular-plot masks
        (default: full rectangle)."""
        self.backend = backend
        self.fallback = fallback or PriorProposer()
        self.tau = tau
        self.temperature = temperature
        self.boundary_fn = boundary_fn
        # telemetry: how often the model actually drove the layout. The merge
        # gate reports it, and in production a rising fallback rate is the
        # first sign the deployed weights no longer fit the traffic.
        self.stats = {"model": 0, "fallback": 0, "conf_sum": 0.0}

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
        arrays = request_to_arrays(request, mask)
        rng = np.random.default_rng(request.seed * 1_000_003 + variant)
        result = md.sample(self.backend, arrays,
                           temperature=self.temperature, rng=rng,
                           greedy=(variant == 0))

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
        placements = [
            Placement(room=specs[p.room_index].name,
                      seed_cell=(p.row, p.col), size_class=p.size_class)
            for p in result.placements]
        return LayoutProposal(placements=placements, source="tier2-model",
                              confidence=round(result.confidence, 3))
